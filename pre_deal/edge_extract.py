import cv2
import os

input_dir = "/home/wqliu/Workspace/2023/preprocess/train_resized/imgs"
output_dir = "/home/wqliu/Workspace/2023/preprocess/train_resized/edges"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, filename in enumerate(os.listdir(input_dir)):
    try:
    # Load image and convert to grayscale
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection
        edges_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
        edges_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
        edges = cv2.convertScaleAbs(cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0))

    # Save output image
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
        cv2.imwrite(output_path, edges)

        print(f"Processed image {i+1}/{len(os.listdir(input_dir))}: {filename}")

    except Exception as e:
        print(f"Failed to process image {i + 1}/{len(os.listdir(input_dir))}: {filename} - {str(e)}")