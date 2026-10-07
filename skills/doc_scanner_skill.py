import os
import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points clockwise: [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left point has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right point has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies perspective warp to give a flat top-down 90-degree rectangular view of the document.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Ensure minimum reasonable dimensions
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

def detect_document_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detects the 4 corners of a document in an image.
    Returns ordered (4, 2) coordinates if found, or None if no confident 4-corner polygon is detected.
    """
    orig_h, orig_w = image.shape[:2]
    target_h = 800.0
    ratio = orig_h / target_h
    new_w = int(orig_w / ratio)
    
    resized = cv2.resize(image, (new_w, int(target_h)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Edge detection with Canny + Morphological Closing
    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    total_area = float(new_w * int(target_h))
    
    for c in contours[:5]:
        area = cv2.contourArea(c)
        # Document should take at least 15% of the frame
        if area < 0.15 * total_area:
            break
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2) * ratio
            return pts
            
    # 2. Fallback: Adaptive thresholding if Canny missed faint paper borders
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours[:5]:
        area = cv2.contourArea(c)
        if area < 0.15 * total_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2) * ratio
            return pts

    return None

def remove_shadows_and_enhance(image: np.ndarray, mode: str = "magic_color") -> np.ndarray:
    """
    Removes uneven lighting/shadows and enhances contrast.
    
    Modes:
      - 'magic_color': CamScanner style: paper background becomes pure bright white while
                       preserving pen colors (blue ink, red grading marks) and dark text.
      - 'bw': High-contrast black and white binarization.
      - 'original_flat': Keep natural colors without bleaching.
    """
    if mode == "original_flat":
        return image
        
    if mode == "bw":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Background illumination estimation
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        # Division normalization
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # Adaptive binarization for clean print
        bw = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # Default: 'magic_color'
    # Process each BGR plane independently
    planes = cv2.split(image)
    result_planes = []
    
    for plane in planes:
        # 1. Dilate to suppress dark characters and estimate paper background
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        # 2. Median blur to get a smooth illumination surface
        bg = cv2.medianBlur(dilated, 21)
        
        # 3. Illumination flattening via division
        # plane / bg * 255 normalizes paper background to ~255 (white)
        plane_f = plane.astype(np.float32)
        bg_f = np.maximum(bg.astype(np.float32), 1.0)
        normalized = np.clip((plane_f / bg_f) * 255.0, 0, 255).astype(np.uint8)
        
        # 4. Contrast enhancement: stretch values so text is dark and paper is crisp white
        # Clip top 2% brightest to pure 255
        p_low = np.percentile(normalized, 1)
        p_high = np.percentile(normalized, 95)
        if p_high > p_low:
            stretched = np.clip((normalized.astype(np.float32) - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
        else:
            stretched = normalized
            
        result_planes.append(stretched)
        
    enhanced = cv2.merge(result_planes)
    
    # Mild sharpening to make handwritten strokes and text crisp
    kernel_sharpening = np.array([
        [0, -0.5, 0],
        [-0.5, 3.0, -0.5],
        [0, -0.5, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpening)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def process_document_image(
    input_path: str,
    output_path: Optional[str] = None,
    mode: str = "magic_color"
) -> Tuple[np.ndarray, str]:
    """
    Reads an image, detects document boundaries, crops/deskews to right angles,
    removes shadows/enhances colors, and saves the output.
    
    Returns (enhanced_image_bgr, final_saved_path).
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_path}")
        
    # Step 1: Detect corners & warp perspective
    corners = detect_document_corners(image)
    if corners is not None:
        warped = four_point_transform(image, corners)
    else:
        logger.info("No confident document corners found. Keeping full image frame.")
        warped = image
        
    # Step 2: Remove shadows and enhance colors
    enhanced = remove_shadows_and_enhance(warped, mode=mode)
    
    # Step 3: Save result if path provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_scanned.jpg"
        
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    
    return enhanced, output_path

def compile_images_to_pdf(image_paths: List[str], output_pdf_path: str) -> str:
    """
    Compiles a list of processed image file paths into a single multi-page PDF.
    """
    if not image_paths:
        raise ValueError("Cannot create PDF with an empty list of images.")
        
    pil_images = []
    for path in image_paths:
        if not os.path.exists(path):
            logger.warning(f"Image path not found: {path}")
            continue
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        pil_images.append(img)
        
    if not pil_images:
        raise ValueError("No valid images could be opened to create PDF.")
        
    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)
    
    first_image = pil_images[0]
    other_images = pil_images[1:] if len(pil_images) > 1 else []
    
    first_image.save(
        output_pdf_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=other_images
    )
    
    return output_pdf_path

class DocScannerTools:
    """
    High-level skill class for integration into Gemini agent or Telegram Bot.
    """
    def __init__(self, output_dir: str = "images/scans"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def scan_single_image(self, image_path: str, mode: str = "magic_color") -> Tuple[str, str]:
        """
        Scans a single image, returning (preview_image_path, pdf_path).
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        preview_path = os.path.join(self.output_dir, f"{base_name}_enhanced.jpg")
        pdf_path = os.path.join(self.output_dir, f"{base_name}.pdf")
        
        _, saved_preview = process_document_image(image_path, preview_path, mode=mode)
        final_pdf = compile_images_to_pdf([saved_preview], pdf_path)
        
        return saved_preview, final_pdf

    def scan_multiple_images(self, image_paths: List[str], output_filename: str = "document.pdf", mode: str = "magic_color") -> Tuple[List[str], str]:
        """
        Processes multiple images and bundles them into a single multi-page PDF.
        Returns (list_of_enhanced_image_paths, pdf_path).
        """
        enhanced_paths = []
        for idx, img_p in enumerate(image_paths):
            base_name = f"page_{idx + 1}"
            preview_path = os.path.join(self.output_dir, f"{base_name}_enhanced.jpg")
            _, saved_preview = process_document_image(img_p, preview_path, mode=mode)
            enhanced_paths.append(saved_preview)
            
        pdf_path = os.path.join(self.output_dir, output_filename)
        final_pdf = compile_images_to_pdf(enhanced_paths, pdf_path)
        return enhanced_paths, final_pdf
