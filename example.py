"""
Example script demonstrating how to use the AI Image Classifier.
"""

from classifier import ImageClassifier
from pathlib import Path


def example_usage():
    """Demonstrate basic usage of the ImageClassifier."""
    
    print("=" * 60)
    print("AI Image Classifier - Example Usage")
    print("=" * 60)
    
    # Initialize the classifier
    classifier = ImageClassifier()
    
    # Example 1: Classify a single image
    print("\n--- Example 1: Basic Classification ---")
    print("To classify an image, use:")
    print("  predictions = classifier.classify('path/to/image.jpg')")
    print("\nOr use the convenient print method:")
    print("  classifier.classify_and_print('path/to/image.jpg')")
    
    # Example 2: Getting different number of predictions
    print("\n--- Example 2: Custom Number of Predictions ---")
    print("To get top 10 predictions:")
    print("  predictions = classifier.classify('image.jpg', top_k=10)")
    
    # Example 3: Processing predictions programmatically
    print("\n--- Example 3: Processing Results ---")
    print("The classify() method returns a list of tuples:")
    print("  [(class_id, class_name, probability), ...]")
    print("\nExample code:")
    print("  predictions = classifier.classify('image.jpg', top_k=3)")
    print("  for class_id, class_name, prob in predictions:")
    print("      print(f'{class_name}: {prob*100:.2f}%')")
    
    print("\n" + "=" * 60)
    print("To test with your own images:")
    print("  python classifier.py your_image.jpg")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
