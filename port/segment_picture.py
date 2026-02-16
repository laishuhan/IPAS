import cv2
import numpy as np
import os

def order_points(pts):
    try:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    except Exception as e:
        print(f"Error in order_points: {e}")
        return None

def extract_ultrasound_regions(image_path, min_area=50000, max_aspect_ratio=2.0):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image at {image_path} could not be loaded.")
            return []

        # Convert to grayscale and apply binary threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        blurred = cv2.GaussianBlur(closed, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if aspect_ratio > max_aspect_ratio:
                continue
            valid_contours.append((rect, cnt))

        valid_contours.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

        results = []
        for rect, cnt in valid_contours:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            ordered_pts = order_points(box).astype(np.float32)
            if ordered_pts is None:
                continue
            (tl, tr, br, bl) = ordered_pts
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            max_width = max(int(width_top), int(width_bottom))
            height_left = np.linalg.norm(bl - tl)
            height_right = np.linalg.norm(br - tr)
            max_height = max(int(height_left), int(height_right))
            dst = np.array([
                [0, 0],
                [max_width-1, 0],
                [max_width-1, max_height-1],
                [0, max_height-1]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(ordered_pts, dst)
            warped = cv2.warpPerspective(image, M, (max_width, max_height))
            results.append(warped)
        return results
    except Exception as e:
        print(f"Error in extract_ultrasound_regions: {e}")
        return []

def segment(file_path, segment_dir):
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            print(f"Error: The file at {file_path} does not exist.")
            return -1

        # # Create output folder
        # output_folder = os.path.join(segment_dir, os.path.basename(file_path).split('.')[0])
        # os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

        # Call ultrasound region extraction
        regions = extract_ultrasound_regions(file_path)

        if not regions:
            print(f"No regions found for image: {file_path}")
            return -1

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Save each extracted region as a separate image
        for i, region in enumerate(regions):
            output_path = os.path.join(segment_dir, f"{base_name}_region_{i+1}.png")
            cv2.imwrite(output_path, region)
            print(f"Saved: {output_path}")

        print("Segmentation completed successfully.")
        return 0
    except Exception as e:
        print(f"Error in segment function: {e}")
        return -1





