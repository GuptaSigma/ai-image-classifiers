"""
AI Image Classifier Module

This module provides functionality for classifying images using pre-trained deep learning models.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path


class ImageClassifier:
    """
    A class for classifying images using pre-trained MobileNetV2 model.
    
    The classifier uses ImageNet weights and can identify 1000 different categories.
    """
    
    def __init__(self):
        """Initialize the image classifier with a pre-trained MobileNetV2 model."""
        print("Loading pre-trained MobileNetV2 model...")
        self.model = MobileNetV2(weights='imagenet')
        print("Model loaded successfully!")
    
    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Load and preprocess an image for classification.
        
        Args:
            image_path (str): Path to the image file
            target_size (tuple): Target size for the image (default: 224x224)
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def classify(self, image_path, top_k=5):
        """
        Classify an image and return the top predictions.
        
        Args:
            image_path (str): Path to the image file
            top_k (int): Number of top predictions to return (default: 5)
            
        Returns:
            list: List of tuples containing (class_id, class_name, probability)
        """
        # Check if file exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess the image
        processed_image = self.load_and_preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Decode predictions
        decoded_predictions = decode_predictions(predictions, top=top_k)[0]
        
        return decoded_predictions
    
    def classify_and_print(self, image_path, top_k=5):
        """
        Classify an image and print the results in a readable format.
        
        Args:
            image_path (str): Path to the image file
            top_k (int): Number of top predictions to return (default: 5)
        """
        print(f"\nClassifying image: {image_path}")
        print("-" * 60)
        
        try:
            predictions = self.classify(image_path, top_k)
            
            print(f"Top {top_k} predictions:")
            for i, (class_id, class_name, probability) in enumerate(predictions, 1):
                print(f"{i}. {class_name:20s} - {probability*100:.2f}%")
                
        except Exception as e:
            print(f"Error classifying image: {str(e)}")


def main():
    """Main function to demonstrate the image classifier."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <image_path> [top_k]")
        print("Example: python classifier.py cat.jpg 5")
        return
    
    image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Create classifier instance
    classifier = ImageClassifier()
    
    # Classify and print results
    classifier.classify_and_print(image_path, top_k)


if __name__ == "__main__":
    main()
