import os
import cv2
import json
import logging
import re
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import pypdfium2

from skills.doc_scanner_skill import (
    detect_document_corners,
    four_point_transform,
    remove_shadows_and_enhance,
    compile_images_to_pdf
)

logger = logging.getLogger(__name__)

def purge_colored_ink(image: np.ndarray) -> np.ndarray:
    """
    Purges teacher grading marks (red ink, checkmarks, score numbers, circles)
    and colored student ink (blue ballpoint, green/highlighter) using HSV color space.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Red ink (teacher grading checkmarks, score circles, comments)
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 50, 40]), np.array([180, 255, 255]))
    mask_red = mask_red1 | mask_red2

    # 2. Blue / Purple ink (student blue ballpoint / fountain pen)
    mask_blue = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([145, 255, 255]))

    # 3. Green / Highlighter ink
    mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    combined_mask = mask_red | mask_blue | mask_green

    # Dilate mask slightly to cover antialiased stroke edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    if cv2.countNonZero(dilated_mask) == 0:
        return image

    # Inpaint colored stroke regions using paper background
    cleaned = cv2.inpaint(image, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned

def restore_horizontal_ruled_lines(roi: np.ndarray) -> np.ndarray:
    """
    For answer areas with printed ruled lines (e.g. Science/Chinese 2-3 lines below questions):
    Detects the horizontal line positions, clears the handwriting, and redraws crisp straight lines.
    """
    h, w = roi.shape[:2]
    if h < 10 or w < 30:
        return np.full_like(roi, 255)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Invert threshold: lines/text are white, background is black
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    # Wide horizontal kernel to isolate printed horizontal lines
    kernel_len = max(20, int(w * 0.15))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    line_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    # Detect horizontal line row coordinates
    line_rows = np.where(np.sum(line_mask, axis=1) > 0)[0]

    # Create clean white background
    cleaned_roi = np.full_like(roi, 255)

    if len(line_rows) > 0:
        # Group adjacent row indices into distinct lines
        lines = []
        current_line = [line_rows[0]]
        for r in line_rows[1:]:
            if r - current_line[-1] <= 3:
                current_line.append(r)
            else:
                lines.append(int(np.mean(current_line)))
                current_line = [r]
        if current_line:
            lines.append(int(np.mean(current_line)))

        # Redraw crisp straight ruled lines
        for y_pos in lines:
            cv2.line(cleaned_roi, (0, y_pos), (w - 1, y_pos), (50, 50, 50), 1, cv2.LINE_AA)

    return cleaned_roi

def semantic_clean_with_gemini(image: np.ndarray, client=None) -> np.ndarray:
    """
    Uses Gemini 2.5 Flash spatial vision to identify student answer areas,
    math working spaces, and rough drafts without touching printed questions or diagrams.
    """
    if client is None:
        return image

    h, w = image.shape[:2]
    
    # Resize temporarily to ~1024 max dimension for fast transmission to Gemini
    max_dim = 1024.0
    scale = min(max_dim / h, max_dim / w, 1.0)
    scaled_h, scaled_w = int(h * scale), int(w * scale)
    scaled_img = cv2.resize(image, (scaled_w, scaled_h))

    # Convert to PIL Image for Gemini SDK
    rgb_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    prompt = """
    You are an expert exam paper layout analyzer.
    Your task is to identify all regions containing STUDENT HANDWRITING, WRITTEN ANSWERS, 
    MATH WORKING STEPS, and ROUGH DRAFTS that should be erased so the exam paper can be re-tested by a student.
    
    Return a strictly valid JSON array of objects:
    [
      {
        "box_2d": [ymin, xmin, ymax, xmax],
        "has_ruled_lines": true/false
      }
    ]
    Where:
    - [ymin, xmin, ymax, xmax] are normalized coordinates from 0 to 1000.
    - "has_ruled_lines" is true if the answer area contains printed horizontal lines (______) or answer lines.
    - "has_ruled_lines" is false if it is a blank math working area, scratchwork box, or margin draft.

    CRITICAL RULES:
    1. DO NOT include printed question text, problem numbers, or formulas!
    2. DO NOT include diagrams, geometry figures, coordinate axes, or illustrations!
    3. ONLY return regions where a student filled in answers, calculations, or drafts.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_img, prompt],
            config={
                "response_mime_type": "application/json"
            }
        )

        raw_text = response.text.strip()
        # Clean any markdown code blocks
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        boxes_data = json.loads(raw_text.strip())
        if not isinstance(boxes_data, list):
            return image

        cleaned_image = image.copy()

        for item in boxes_data:
            box = item.get("box_2d")
            has_lines = item.get("has_ruled_lines", False)

            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            # Scale coordinates back to original image dimensions
            y1 = max(0, min(h - 1, int(ymin * h / 1000.0)))
            y2 = max(0, min(h, int(ymax * h / 1000.0)))
            x1 = max(0, min(w - 1, int(xmin * w / 1000.0)))
            x2 = max(0, min(w, int(xmax * w / 1000.0)))

            if y2 <= y1 or x2 <= x1:
                continue

            roi = cleaned_image[y1:y2, x1:x2]

            if has_lines:
                # Answer area with ruled lines: restore clean lines
                cleaned_roi = restore_horizontal_ruled_lines(roi)
                cleaned_image[y1:y2, x1:x2] = cleaned_roi
            else:
                # Math working space or blank answer box: wipe completely white
                cleaned_image[y1:y2, x1:x2] = [255, 255, 255]

        return cleaned_image

    except Exception as e:
        logger.warning(f"Semantic Gemini layout cleaning skipped or failed: {e}")
        return image

def clean_exam_page(image: np.ndarray, client=None, enhance: bool = True) -> np.ndarray:
    """
    Cleans a single exam page:
    1. Purges red teacher marks, blue ink, highlighter.
    2. Cleans answer zones and working boxes via semantic vision (Gemini).
    3. Enhances paper contrast with Magic Color.
    """
    # 1. Color Purge
    color_purged = purge_colored_ink(image)

    # 2. Semantic Layout Clean
    layout_cleaned = semantic_clean_with_gemini(color_purged, client=client)

    # 3. Magic Color Contrast Enhancement
    if enhance:
        final_enhanced = remove_shadows_and_enhance(layout_cleaned, mode="magic_color")
        return final_enhanced

    return layout_cleaned

class ExamCleanerTools:
    """
    High-level skill for cleaning exam papers from photos or multi-page PDFs.
    """
    def __init__(self, output_dir: str = "images/clean_exams", client=None):
        self.output_dir = output_dir
        self.client = client
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_pdf_file(self, pdf_path: str, output_pdf_name: str = "Cleaned_Exam.pdf") -> Tuple[List[str], str]:
        """
        Processes a multi-page PDF document page by page, removing handwriting and answer contents.
        Returns (list_of_cleaned_page_preview_paths, final_output_pdf_path).
        """
        doc = pypdfium2.PdfDocument(pdf_path)
        total_pages = len(doc)
        cleaned_paths = []

        logger.info(f"Processing PDF '{pdf_path}' with {total_pages} page(s)...")

        for idx, page in enumerate(doc):
            # Render page at 2.0 scale (approx 200-300 DPI)
            pil_img = page.render(scale=2.0).to_pil()
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cleaned_page = clean_exam_page(bgr_img, client=self.client)

            preview_path = os.path.join(self.output_dir, f"clean_page_{idx + 1}.jpg")
            cv2.imwrite(preview_path, cleaned_page)
            cleaned_paths.append(preview_path)

        output_pdf_path = os.path.join(self.output_dir, output_pdf_name)
        final_pdf = compile_images_to_pdf(cleaned_paths, output_pdf_path)
        return cleaned_paths, final_pdf

    def clean_photo_file(self, photo_path: str, output_pdf_name: str = "Cleaned_Exam.pdf") -> Tuple[str, str]:
        """
        Processes a single photo of an exam paper (deskewing + handwriting cleaning + PDF export).
        """
        image = cv2.imread(photo_path)
        if image is None:
            raise ValueError(f"Could not read image from {photo_path}")

        # Deskew & perspective warp if corners detected
        corners = detect_document_corners(image)
        if corners is not None:
            warped = four_point_transform(image, corners)
        else:
            warped = image

        cleaned_page = clean_exam_page(warped, client=self.client)

        base_name = os.path.splitext(os.path.basename(photo_path))[0]
        preview_path = os.path.join(self.output_dir, f"{base_name}_cleaned.jpg")
        cv2.imwrite(preview_path, cleaned_page)

        output_pdf_path = os.path.join(self.output_dir, output_pdf_name)
        final_pdf = compile_images_to_pdf([preview_path], output_pdf_path)
        return preview_path, final_pdf

    def clean_multiple_photos(self, photo_paths: List[str], output_pdf_name: str = "Cleaned_Exam.pdf") -> Tuple[List[str], str]:
        """
        Processes multiple exam paper photos and compiles them into a single cleaned PDF.
        """
        cleaned_paths = []
        for idx, p in enumerate(photo_paths):
            image = cv2.imread(p)
            if image is None:
                continue

            corners = detect_document_corners(image)
            warped = four_point_transform(image, corners) if corners is not None else image
            cleaned_page = clean_exam_page(warped, client=self.client)

            preview_path = os.path.join(self.output_dir, f"clean_photo_p{idx + 1}.jpg")
            cv2.imwrite(preview_path, cleaned_page)
            cleaned_paths.append(preview_path)

        output_pdf_path = os.path.join(self.output_dir, output_pdf_name)
        final_pdf = compile_images_to_pdf(cleaned_paths, output_pdf_path)
        return cleaned_paths, final_pdf
