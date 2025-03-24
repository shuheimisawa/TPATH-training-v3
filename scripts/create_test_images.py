import os
import numpy as np
import cv2

def create_test_image(output_path, size=(512, 512)):
    # Create a random image
    image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    # Save the image
    cv2.imwrite(output_path, image)

def main():
    # Create train image
    train_image_path = os.path.join('data', 'train', 'image_001.jpg')
    create_test_image(train_image_path)
    
    # Create validation image
    val_image_path = os.path.join('data', 'val', 'image_002.jpg')
    create_test_image(val_image_path)
    
    # Create test image
    test_image_path = os.path.join('data', 'test', 'image_003.jpg')
    create_test_image(test_image_path)

if __name__ == '__main__':
    main() 