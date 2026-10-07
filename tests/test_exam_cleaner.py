import os
import sys
import unittest
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skills.exam_cleaner_skill import (
    purge_colored_ink,
    restore_horizontal_ruled_lines,
    clean_exam_page,
    ExamCleanerTools
)
from skills.doc_scanner_skill import compile_images_to_pdf

class TestExamCleaner(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_output/cleaner"
        os.makedirs(self.test_dir, exist_ok=True)

    def create_synthetic_exam_page(self, filename="synthetic_exam.jpg"):
        """
        Creates an exam page with:
        - Black printed question text.
        - Red teacher checkmark and score (100).
        - Blue handwritten answer.
        - Ruled lines with handwriting.
        """
        img_h, img_w = 1200, 900
        img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

        # 1. Printed Question 1 (Math)
        cv2.putText(img, "1. Calculate: 15 * 8 = ?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        # Blue handwritten answer
        cv2.putText(img, "120", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 30, 20), 2) # Blue in BGR
        # Red teacher checkmark
        cv2.putText(img, "[CHECK OK 100]", (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 220), 3) # Red in BGR

        # 2. Printed Question 2 (Science - Ruled lines)
        cv2.putText(img, "2. Explain why the sky is blue:", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        # Ruled line 1
        cv2.line(img, (50, 340), (850, 340), (60, 60, 60), 1)
        # Blue student handwriting on line 1
        cv2.putText(img, "Rayleigh scattering of sunlight", (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 40, 30), 2)
        # Ruled line 2
        cv2.line(img, (50, 420), (850, 420), (60, 60, 60), 1)
        # Blue student handwriting on line 2
        cv2.putText(img, "by air particles.", (60, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 40, 30), 2)

        file_path = os.path.join(self.test_dir, filename)
        cv2.imwrite(file_path, img)
        return file_path

    def test_purge_colored_ink(self):
        file_path = self.create_synthetic_exam_page("test_color_purge.jpg")
        img = cv2.imread(file_path)

        # Before cleaning: Check red pixels exist
        hsv_before = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask_before = cv2.inRange(hsv_before, np.array([0, 50, 40]), np.array([10, 255, 255]))
        self.assertGreater(cv2.countNonZero(red_mask_before), 100)

        cleaned = purge_colored_ink(img)

        # After cleaning: Red pixels should be purged
        hsv_after = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
        red_mask_after = cv2.inRange(hsv_after, np.array([0, 50, 40]), np.array([10, 255, 255]))
        self.assertEqual(cv2.countNonZero(red_mask_after), 0, "Red teacher marks should be 100% eliminated")

    def test_restore_horizontal_ruled_lines(self):
        # Create a small ROI with 2 ruled lines and handwriting crossing them
        roi = np.full((120, 400, 3), 255, dtype=np.uint8)
        cv2.line(roi, (10, 40), (390, 40), (50, 50, 50), 1)
        cv2.line(roi, (10, 90), (390, 90), (50, 50, 50), 1)
        # Simulated handwriting scribbles
        cv2.putText(roi, "my handwritten words", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

        restored = restore_horizontal_ruled_lines(roi)
        self.assertEqual(restored.shape, roi.shape)

        # Check that the horizontal lines are reconstructed
        gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        self.assertLess(int(np.min(gray[40, :])), 100, "Ruled line at y=40 should exist")
        self.assertLess(int(np.min(gray[90, :])), 100, "Ruled line at y=90 should exist")

    def test_clean_pdf_and_photo(self):
        # Create a 2-page PDF
        p1 = self.create_synthetic_exam_page("p1.jpg")
        p2 = self.create_synthetic_exam_page("p2.jpg")

        test_pdf = os.path.join(self.test_dir, "input_test_exam.pdf")
        compile_images_to_pdf([p1, p2], test_pdf)
        self.assertTrue(os.path.exists(test_pdf))

        tools = ExamCleanerTools(output_dir=self.test_dir)

        # Test PDF cleaning
        previews, out_pdf = tools.clean_pdf_file(test_pdf, "Cleaned_Output.pdf")
        self.assertEqual(len(previews), 2)
        self.assertTrue(os.path.exists(out_pdf))
        self.assertGreater(os.path.getsize(out_pdf), 1000)

        # Test photo cleaning
        preview_img, photo_pdf = tools.clean_photo_file(p1, "Cleaned_Photo_Exam.pdf")
        self.assertTrue(os.path.exists(preview_img))
        self.assertTrue(os.path.exists(photo_pdf))

if __name__ == "__main__":
    unittest.main()
