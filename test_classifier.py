"""
Unit tests for the AI Image Classifier.

Note: These tests verify the code structure and logic without requiring
actual model downloads or inference, which makes them suitable for CI/CD.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path


class TestImageClassifier(unittest.TestCase):
    """Test cases for the ImageClassifier class."""
    
    @patch('classifier.MobileNetV2')
    def test_classifier_initialization(self, mock_model):
        """Test that the classifier initializes correctly."""
        from classifier import ImageClassifier
        
        # Mock the model
        mock_model.return_value = Mock()
        
        # Create classifier
        classifier = ImageClassifier()
        
        # Verify model was loaded with correct parameters
        mock_model.assert_called_once_with(weights='imagenet')
        self.assertIsNotNone(classifier.model)
    
    @patch('classifier.image.load_img')
    @patch('classifier.image.img_to_array')
    @patch('classifier.preprocess_input')
    def test_load_and_preprocess_image(self, mock_preprocess, mock_img_to_array, mock_load_img):
        """Test image loading and preprocessing."""
        from classifier import ImageClassifier
        
        # Setup mocks
        mock_img = Mock()
        mock_load_img.return_value = mock_img
        mock_array = np.zeros((224, 224, 3))
        mock_img_to_array.return_value = mock_array
        mock_preprocess.return_value = mock_array
        
        # Create classifier with mocked model
        with patch('classifier.MobileNetV2'):
            classifier = ImageClassifier()
        
        # Test preprocessing
        result = classifier.load_and_preprocess_image('test.jpg')
        
        # Verify calls
        mock_load_img.assert_called_once_with('test.jpg', target_size=(224, 224))
        mock_img_to_array.assert_called_once_with(mock_img)
        # Result should have batch dimension added
        self.assertIsNotNone(result)
    
    @patch('classifier.Path')
    @patch('classifier.decode_predictions')
    def test_classify_file_not_found(self, mock_decode, mock_path):
        """Test that FileNotFoundError is raised for non-existent files."""
        from classifier import ImageClassifier
        
        # Mock path to return False for exists()
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj
        
        # Create classifier with mocked model
        with patch('classifier.MobileNetV2'):
            classifier = ImageClassifier()
        
        # Test that FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            classifier.classify('nonexistent.jpg')
    
    @patch('classifier.Path')
    @patch('classifier.decode_predictions')
    def test_classify_success(self, mock_decode, mock_path):
        """Test successful image classification."""
        from classifier import ImageClassifier
        
        # Mock path to return True for exists()
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj
        
        # Mock predictions
        mock_predictions = [
            ('n02124075', 'Egyptian_cat', 0.8),
            ('n02123045', 'tabby', 0.15),
            ('n02123159', 'tiger_cat', 0.05)
        ]
        mock_decode.return_value = [mock_predictions]
        
        # Create classifier with mocked model
        with patch('classifier.MobileNetV2') as mock_model:
            mock_model_instance = Mock()
            mock_model_instance.predict.return_value = np.array([[0.8, 0.15, 0.05]])
            mock_model.return_value = mock_model_instance
            
            classifier = ImageClassifier()
            
            # Mock the preprocessing
            with patch.object(classifier, 'load_and_preprocess_image') as mock_preprocess:
                mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
                
                # Test classification
                results = classifier.classify('test.jpg', top_k=3)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][1], 'Egyptian_cat')
        self.assertEqual(results[0][2], 0.8)
    
    @patch('classifier.Path')
    @patch('builtins.print')
    def test_classify_and_print(self, mock_print, mock_path):
        """Test the classify_and_print method."""
        from classifier import ImageClassifier
        
        # Mock path to return True for exists()
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj
        
        # Create classifier with mocked model
        with patch('classifier.MobileNetV2'):
            classifier = ImageClassifier()
            
            # Mock classify method
            mock_predictions = [
                ('n02124075', 'Egyptian_cat', 0.8),
                ('n02123045', 'tabby', 0.15)
            ]
            
            with patch.object(classifier, 'classify') as mock_classify:
                mock_classify.return_value = mock_predictions
                
                # Test classify_and_print
                classifier.classify_and_print('test.jpg', top_k=2)
        
        # Verify print was called (results were printed)
        self.assertTrue(mock_print.called)


class TestModuleStructure(unittest.TestCase):
    """Test the overall module structure."""
    
    def test_module_imports(self):
        """Test that the classifier module can be imported."""
        try:
            import classifier
            self.assertTrue(hasattr(classifier, 'ImageClassifier'))
            self.assertTrue(hasattr(classifier, 'main'))
        except ImportError as e:
            self.fail(f"Failed to import classifier module: {e}")
    
    def test_class_has_required_methods(self):
        """Test that ImageClassifier has all required methods."""
        from classifier import ImageClassifier
        
        self.assertTrue(hasattr(ImageClassifier, '__init__'))
        self.assertTrue(hasattr(ImageClassifier, 'load_and_preprocess_image'))
        self.assertTrue(hasattr(ImageClassifier, 'classify'))
        self.assertTrue(hasattr(ImageClassifier, 'classify_and_print'))


if __name__ == '__main__':
    unittest.main()
