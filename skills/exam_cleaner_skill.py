import os
import cv2
import json
import logging
import re
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
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
    Saves and preserves orange/brown printed diagram lines and problem badges (Hue 9-20).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Red ink (teacher grading checkmarks, score circles, comments)
    # Pure red ink: Hue 0-6 and 168-180, Saturation >= 65, Value >= 50
    # Note: Orange/brown diagram lines and question badges have Hue 9-20, so they are fully spared.
    mask_red1 = cv2.inRange(hsv, np.array([0, 65, 50]), np.array([6, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([168, 65, 50]), np.array([180, 255, 255]))
    mask_red = mask_red1 | mask_red2

    # 2. Blue / Purple ink (student blue ballpoint / fountain pen)
    mask_blue = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([145, 255, 255]))

    # 3. Green / Highlighter ink
    mask_green = cv2.inRange(hsv, np.array([38, 45, 40]), np.array([85, 255, 255]))

    combined_mask = mask_red | mask_blue | mask_green

    # Dilate mask slightly to cover antialiased stroke edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    if cv2.countNonZero(dilated_mask) == 0:
        return image

    # Inpaint colored stroke regions using paper background
    cleaned = cv2.inpaint(image, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned

def sample_local_paper_background(image: np.ndarray, y1: int, x1: int, y2: int, x2: int, margin: int = 12) -> Tuple[int, int, int]:
    """
    Samples the local paper background color from a ring surrounding the bounding box [y1:y2, x1:x2].
    Excludes dark text and ink strokes so we extract the true, natural paper tone.
    """
    h, w = image.shape[:2]
    samples = []

    # Top margin
    if y1 > 0:
        samples.append(image[max(0, y1 - margin):y1, x1:x2].reshape(-1, 3))
    # Bottom margin
    if y2 < h:
        samples.append(image[y2:min(h, y2 + margin), x1:x2].reshape(-1, 3))
    # Left margin
    if x1 > 0:
        samples.append(image[y1:y2, max(0, x1 - margin):x1].reshape(-1, 3))
    # Right margin
    if x2 < w:
        samples.append(image[y1:y2, x2:min(w, x2 + margin)].reshape(-1, 3))

    if not samples:
        return (255, 255, 255)

    all_px = np.concatenate(samples, axis=0)
    gray = cv2.cvtColor(all_px.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()

    # Filter for paper pixels (high brightness >= 160)
    bright_px = all_px[gray >= 160]
    if len(bright_px) < 10:
        bright_px = all_px[gray >= np.percentile(gray, 70)]

    if len(bright_px) == 0:
        return (255, 255, 255)

    med_bgr = np.median(bright_px, axis=0).astype(int)
    return (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))

def blend_answer_box(
    image: np.ndarray,
    y1: int,
    x1: int,
    y2: int,
    x2: int,
    fill_roi: Optional[np.ndarray] = None,
    feather_px: int = 10
) -> np.ndarray:
    """
    Seamlessly cleans or replaces an answer/calculation box by:
    1. Sampling the true local paper background color from surrounding pixels.
    2. Feathering the outer edges using a Gaussian blur mask so there are NO visible rectangular boundaries
       or stark white patches.
    """
    h_box = y2 - y1
    w_box = x2 - x1
    if h_box <= 0 or w_box <= 0:
        return image

    bg_col = sample_local_paper_background(image, y1, x1, y2, x2)

    if fill_roi is not None:
        replacement = fill_roi.copy()
    else:
        replacement = np.full((h_box, w_box, 3), bg_col, dtype=np.uint8)

    feather = min(feather_px, max(1, min(h_box // 6, w_box // 6)))
    if feather > 1 and h_box > 2 * feather and w_box > 2 * feather:
        mask = np.zeros((h_box, w_box), dtype=np.float32)
        mask[feather : h_box - feather, feather : w_box - feather] = 1.0
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather / 2.0)
        max_v = np.max(mask)
        if max_v > 0:
            mask = mask / max_v
        mask_3d = mask[..., None]
        roi = image[y1:y2, x1:x2].astype(np.float32)
        blended = (replacement.astype(np.float32) * mask_3d + roi * (1.0 - mask_3d)).astype(np.uint8)
    else:
        blended = replacement

    image[y1:y2, x1:x2] = blended
    return image

def restore_horizontal_ruled_lines(roi: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    For answer areas with printed ruled lines (e.g. Science/Chinese 2-3 lines below questions):
    Detects the horizontal line positions, clears the handwriting, and redraws crisp straight lines.
    """
    h, w = roi.shape[:2]
    if h < 10 or w < 30:
        return np.full_like(roi, bg_color, dtype=np.uint8)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    kernel_len = max(20, int(w * 0.15))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    line_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    line_rows = np.where(np.sum(line_mask, axis=1) > 0)[0]
    cleaned_roi = np.full_like(roi, bg_color, dtype=np.uint8)

    if len(line_rows) > 0:
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

        for y_pos in lines:
            cv2.line(cleaned_roi, (0, y_pos), (w - 1, y_pos), (60, 60, 60), 1, cv2.LINE_AA)

    return cleaned_roi

def clip_against_diagrams(
    answer_boxes: List[Tuple[int, int, int, int, bool]],
    diagram_boxes: List[Tuple[int, int, int, int]],
    pad_h: int = 15,
    pad_w: int = 15,
    img_h: int = 1000,
    img_w: int = 1000
) -> List[Tuple[int, int, int, int, bool]]:
    """
    Enforces absolute diagram protection:
    Takes diagram bounding boxes, expands them with a safety padding buffer, and tests
    candidate answer boxes against them. Any answer box that falls inside or largely overlaps
    a diagram is completely discarded. Any box that partially intersects is clipped strictly
    outside the diagram boundary.
    This guarantees diagrams and their original printed numbers/labels (e.g. 118°, 130°, 92°) are never erased!
    """
    if not diagram_boxes:
        return answer_boxes

    # Expand diagram boxes with safety padding
    padded_diagrams = []
    for (dy1, dx1, dy2, dx2) in diagram_boxes:
        py1 = max(0, dy1 - pad_h)
        px1 = max(0, dx1 - pad_w)
        py2 = min(img_h, dy2 + pad_h)
        px2 = min(img_w, dx2 + pad_w)
        padded_diagrams.append((py1, px1, py2, px2))

    safe_boxes = []
    for (ay1, ax1, ay2, ax2, has_lines) in answer_boxes:
        cur_y1, cur_x1, cur_y2, cur_x2 = ay1, ax1, ay2, ax2
        discard = False

        for (dy1, dx1, dy2, dx2) in padded_diagrams:
            # Check overlap
            inter_y1 = max(cur_y1, dy1)
            inter_y2 = min(cur_y2, dy2)
            inter_x1 = max(cur_x1, dx1)
            inter_x2 = min(cur_x2, dx2)

            if inter_y2 > inter_y1 and inter_x2 > inter_x1:
                box_area = (cur_y2 - cur_y1) * (cur_x2 - cur_x1)
                inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                center_y = (cur_y1 + cur_y2) / 2.0
                center_x = (cur_x1 + cur_x2) / 2.0

                # If overlap is more than 25% of answer box or center is inside diagram: discard completely
                if (box_area <= 0) or (inter_area / box_area > 0.25) or (dy1 <= center_y <= dy2 and dx1 <= center_x <= dx2):
                    discard = True
                    break

                # Otherwise clip answer box outside the diagram based on relative position
                if center_x < dx1:
                    cur_x2 = min(cur_x2, dx1)
                elif center_x > dx2:
                    cur_x1 = max(cur_x1, dx2)
                elif center_y < dy1:
                    cur_y2 = min(cur_y2, dy1)
                elif center_y > dy2:
                    cur_y1 = max(cur_y1, dy2)
                else:
                    discard = True
                    break

                if (cur_y2 - cur_y1 < 15) or (cur_x2 - cur_x1 < 15):
                    discard = True
                    break

        if not discard and (cur_y2 - cur_y1 >= 15) and (cur_x2 - cur_x1 >= 15):
            safe_boxes.append((cur_y1, cur_x1, cur_y2, cur_x2, has_lines))

    return safe_boxes

def semantic_clean_with_gemini(image: np.ndarray, client=None) -> np.ndarray:
    """
    Uses Gemini 2.5 Flash spatial vision to identify student answer areas,
    math working spaces, and rough drafts without touching printed questions or diagrams.
    Strictly preserves diagrams and all numbers/labels inside or adjacent to diagrams.
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
    Your task is to identify student handwritten answers and rough drafts to be erased, 
    WHILE STRICTLY PROTECTING all printed questions, problem numbers, diagrams, and diagram numbers/labels.

    Return a strictly valid JSON object with TWO arrays:
    {
      "diagram_boxes": [
        [ymin, xmin, ymax, xmax]
      ],
      "answer_boxes": [
        {
          "box_2d": [ymin, xmin, ymax, xmax],
          "has_ruled_lines": false
        }
      ]
    }

    Where:
    - [ymin, xmin, ymax, xmax] are normalized coordinates from 0 to 1000.
    - "diagram_boxes": Bounding boxes around ALL diagrams, graphs, geometric shapes (trapezium, rhombus, triangle, circle), coordinate grids, tables, and illustrations.
    - "answer_boxes": Bounding boxes around student handwritten calculations, answer blanks, and scratchwork to be erased.
    - "has_ruled_lines": true ONLY if the answer area contains printed horizontal lines (______) or answer lines.

    CRITICAL RULES:
    1. ABSOLUTE DIAGRAM & NUMBER PROTECTION:
       - Every diagram and geometric shape must be included in "diagram_boxes".
       - NEVER include any part of a diagram or graph in "answer_boxes"!
       - NEVER erase numbers, angles, or labels in or near diagrams (e.g. 118°, 130°, 92°, 48°, vertex letters like A, B, C, D, J, K, L, M, P).
       - Original printed numbers in graphs MUST be 100% preserved.
    2. ABSOLUTE QUESTION TEXT PROTECTION:
       - DO NOT include printed question text, problem numbers (e.g. "6.", "7."), or sub-questions ("(a)", "(b)") in "answer_boxes".
       - Question text must remain completely clear and untouched.
    3. WHAT TO CLEAN:
       - Student calculations and working steps in the blank spaces below or beside questions.
       - Student handwritten numbers on final answer lines (e.g. "Ans: ______").
       - Ruled answer lines containing student written sentences.
    4. Group multi-line calculations in the same working area into a single coherent bounding box.
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
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        parsed_data = json.loads(raw_text.strip())
        
        # Support both new dict format and legacy list format
        if isinstance(parsed_data, dict):
            raw_diagrams = parsed_data.get("diagram_boxes", [])
            raw_answers = parsed_data.get("answer_boxes", [])
        elif isinstance(parsed_data, list):
            raw_diagrams = []
            raw_answers = parsed_data
        else:
            return image

        # Parse diagram boxes in image pixel coordinates
        diagram_boxes = []
        for d_box in raw_diagrams:
            if isinstance(d_box, list) and len(d_box) == 4:
                d_y1 = max(0, min(h - 1, int(d_box[0] * h / 1000.0)))
                d_x1 = max(0, min(w - 1, int(d_box[1] * w / 1000.0)))
                d_y2 = max(0, min(h, int(d_box[2] * h / 1000.0)))
                d_x2 = max(0, min(w, int(d_box[3] * w / 1000.0)))
                if d_y2 > d_y1 and d_x2 > d_x1:
                    diagram_boxes.append((d_y1, d_x1, d_y2, d_x2))

        # Parse answer boxes
        candidate_boxes = []
        for item in raw_answers:
            box = item.get("box_2d") if isinstance(item, dict) else None
            has_lines = item.get("has_ruled_lines", False) if isinstance(item, dict) else False
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            y1 = max(0, min(h - 1, int(ymin * h / 1000.0)))
            y2 = max(0, min(h, int(ymax * h / 1000.0)))
            x1 = max(0, min(w - 1, int(xmin * w / 1000.0)))
            x2 = max(0, min(w, int(xmax * w / 1000.0)))

            if y2 > y1 and x2 > x1:
                candidate_boxes.append((y1, x1, y2, x2, has_lines))

        # Strictly enforce diagram safety barrier
        safe_boxes = clip_against_diagrams(
            candidate_boxes,
            diagram_boxes,
            pad_h=int(h * 0.015),
            pad_w=int(w * 0.015),
            img_h=h,
            img_w=w
        )

        # Merge vertically adjacent boxes with significant horizontal overlap to prevent gaps
        if safe_boxes:
            safe_boxes = sorted(safe_boxes, key=lambda b: b[0])
            merged_boxes = [list(safe_boxes[0])]
            for cur in safe_boxes[1:]:
                prev = merged_boxes[-1]
                overlap_x = max(0, min(prev[3], cur[3]) - max(prev[1], cur[1]))
                min_w = min(prev[3] - prev[1], cur[3] - cur[1])
                if (min_w > 0 and overlap_x / min_w > 0.4) and (cur[0] - prev[2] <= 40):
                    merged_boxes[-1] = [
                        min(prev[0], cur[0]),
                        min(prev[1], cur[1]),
                        max(prev[2], cur[2]),
                        max(prev[3], cur[3]),
                        prev[4] or cur[4]
                    ]
                else:
                    merged_boxes.append(list(cur))
            safe_boxes = [tuple(b) for b in merged_boxes]

        cleaned_image = image.copy()

        for (y1, x1, y2, x2, has_lines) in safe_boxes:
            roi = cleaned_image[y1:y2, x1:x2]
            if has_lines:
                bg_col = sample_local_paper_background(cleaned_image, y1, x1, y2, x2)
                cleaned_roi = restore_horizontal_ruled_lines(roi, bg_color=bg_col)
                cleaned_image = blend_answer_box(cleaned_image, y1, x1, y2, x2, fill_roi=cleaned_roi, feather_px=8)
            else:
                # Math working space or blank box: blend seamlessly into local paper background color
                cleaned_image = blend_answer_box(cleaned_image, y1, x1, y2, x2, feather_px=10)

        return cleaned_image

    except Exception as e:
        logger.warning(f"Semantic Gemini layout cleaning skipped or failed: {e}")
        return image

def clean_exam_page(image: np.ndarray, client=None, enhance: bool = True) -> np.ndarray:
    """
    Cleans a single exam page:
    1. Purges red teacher marks, blue ink, green pencil (preserves orange diagram lines).
    2. Normalizes lighting and paper background FIRST to ensure crisp contrast and even tone.
    3. Cleans answer zones and working boxes via semantic vision (Gemini) with background matching
       and strict diagram protection.
    """
    # 1. Color Purge (red, blue, green)
    color_purged = purge_colored_ink(image)

    # 2. Normalize background FIRST (ensures paper is evenly white before erasing)
    if enhance:
        normalized = remove_shadows_and_enhance(color_purged, mode="magic_color")
    else:
        normalized = color_purged

    # 3. Semantic Layout Clean on the evenly normalized paper
    final_cleaned = semantic_clean_with_gemini(normalized, client=client)

    return final_cleaned

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
