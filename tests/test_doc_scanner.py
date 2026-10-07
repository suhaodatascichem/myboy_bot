import os
import sys
import unittest
import numpy as np
import cv2
from PIL import Image

# Ensure skills can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skills.doc_scanner_skill import (
    order_points,
    four_point_transform,
    detect_document_corners,
    remove_shadows_and_enhance,
    process_document_image,
    compile_images_to_pdf,
    DocScannerTools
)

class TestDocScanner(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_output"
        os.makedirs(self.test_dir, exist_ok=True)

    def create_synthetic_doc_image(self, filename="synthetic_doc.jpg"):
        """
        Creates a synthetic test image: a dark wooden desk background,
        a tilted white sheet of paper with black text and a blue line,
        and an uneven shadow cast diagonally across the paper.
        """
        img_h, img_w = 1200, 1000
        # Dark desk background
        img = np.full((img_h, img_w, 3), (40, 50, 70), dtype=np.uint8)
        
        # Paper polygon (angled sheet)
        pts = np.array([
            [150, 120],  # top-left
            [850, 180],  # top-right
            [800, 1050], # bottom-right
            [120, 950]   # bottom-left
        ], dtype=np.int32)
        
        # Draw white paper
        cv2.fillPoly(img, [pts], (235, 235, 235))
        
        # Add text onto paper
        cv2.putText(img, "Math Homework - Chapter 5", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
        cv2.putText(img, "Q1. Calculate the area of triangle:", (220, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
        # Blue pen note
        cv2.putText(img, "Good Job! 100%", (250, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 50, 30), 3) # Blue in BGR
        # Red teacher checkmark
        cv2.putText(img, "[Checked A+]", (250, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 220), 3) # Red in BGR

        # Simulate harsh diagonal shadow across the center
        shadow_mask = np.zeros((img_h, img_w), dtype=np.float32)
        for y in range(img_h):
            for x in range(img_w):
                # Gradient shadow from top-right to bottom-left
                shadow_intensity = max(0.0, min(0.6, (x + y * 0.5) / 1200.0 - 0.2))
                shadow_mask[y, x] = shadow_intensity
        
        # Apply shadow
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] * (1.0 - shadow_mask * 0.7), 0, 255).astype(np.uint8)

        file_path = os.path.join(self.test_dir, filename)
        cv2.imwrite(file_path, img)
        return file_path, pts

    def test_order_points(self):
        pts = np.array([[200, 300], [50, 50], [210, 40], [40, 310]], dtype=np.float32)
        ordered = order_points(pts)
        # top-left should be (50, 50)
        self.assertTrue(np.allclose(ordered[0], [50, 50]))
        # top-right should be (210, 40)
        self.assertTrue(np.allclose(ordered[1], [210, 40]))
        # bottom-right should be (200, 300)
        self.assertTrue(np.allclose(ordered[2], [200, 300]))
        # bottom-left should be (40, 310)
        self.assertTrue(np.allclose(ordered[3], [40, 310]))

    def test_detect_corners_and_warp(self):
        file_path, true_pts = self.create_synthetic_doc_image("test_warp.jpg")
        img = cv2.imread(file_path)
        
        detected_corners = detect_document_corners(img)
        self.assertIsNotNone(detected_corners, "Document corners should be detected")
        self.assertEqual(len(detected_corners), 4)
        
        warped = four_point_transform(img, detected_corners)
        self.assertGreater(warped.shape[0], 500)
        self.assertGreater(warped.shape[1], 400)

    def test_shadow_removal_magic_color(self):
        file_path, _ = self.create_synthetic_doc_image("test_shadow.jpg")
        img = cv2.imread(file_path)
        
        enhanced = remove_shadows_and_enhance(img, mode="magic_color")
        self.assertEqual(enhanced.shape, img.shape)
        # Paper regions should have increased brightness (whiter)
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(img)))

    def test_process_and_pdf_generation(self):
        file_p1, _ = self.create_synthetic_doc_image("page1.jpg")
        file_p2, _ = self.create_synthetic_doc_image("page2.jpg")
        
        tools = DocScannerTools(output_dir=os.path.join(self.test_dir, "scans"))
        
        # Test single scan
        preview, pdf1 = tools.scan_single_image(file_p1)
        self.assertTrue(os.path.exists(preview))
        self.assertTrue(os.path.exists(pdf1))
        self.assertGreater(os.path.getsize(pdf1), 1000)
        
        # Test multi-page scan
        previews, multi_pdf = tools.scan_multiple_images([file_p1, file_p2], output_filename="two_pages.pdf")
        self.assertEqual(len(previews), 2)
        self.assertTrue(os.path.exists(multi_pdf))
        self.assertGreater(os.path.getsize(multi_pdf), 2000)

if __name__ == "__main__":
    unittest.main()
