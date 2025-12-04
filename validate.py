#!/usr/bin/env python3
"""
Validation script to verify the AI Image Classifier implementation.
This script performs basic validation without requiring model downloads.
"""

import sys
import os

def check_file_exists(filepath, description):
    """Check if a file exists and print result."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_imports():
    """Check if all required modules can be imported."""
    print("\nChecking imports...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    return True

def check_module_structure():
    """Check if the classifier module has the correct structure."""
    print("\nChecking module structure...")
    try:
        import classifier
        print("✓ classifier module imported")
        
        # Check for main class
        if hasattr(classifier, 'ImageClassifier'):
            print("✓ ImageClassifier class exists")
            
            # Check for required methods
            methods = ['__init__', 'load_and_preprocess_image', 'classify', 'classify_and_print']
            for method in methods:
                if hasattr(classifier.ImageClassifier, method):
                    print(f"  ✓ {method} method exists")
                else:
                    print(f"  ✗ {method} method missing")
                    return False
        else:
            print("✗ ImageClassifier class not found")
            return False
            
        # Check for main function
        if hasattr(classifier, 'main'):
            print("✓ main function exists")
        else:
            print("✗ main function missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error checking module structure: {e}")
        return False

def run_tests():
    """Run the unit tests."""
    print("\nRunning unit tests...")
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', 'test_classifier', '-v'],
        capture_output=True,
        text=True
    )
    
    # Check if tests passed
    if result.returncode == 0:
        # Count test results
        lines = result.stderr.split('\n')
        for line in lines:
            if line.startswith('Ran'):
                print(f"✓ {line}")
                return True
    else:
        print("✗ Some tests failed")
        print(result.stderr)
        return False
    
    return False

def main():
    """Run all validation checks."""
    print("=" * 60)
    print("AI Image Classifier Validation")
    print("=" * 60)
    
    # Check files
    print("\nChecking project files...")
    all_files_exist = True
    all_files_exist &= check_file_exists('README.md', 'README file')
    all_files_exist &= check_file_exists('requirements.txt', 'Requirements file')
    all_files_exist &= check_file_exists('classifier.py', 'Classifier module')
    all_files_exist &= check_file_exists('example.py', 'Example script')
    all_files_exist &= check_file_exists('test_classifier.py', 'Test file')
    all_files_exist &= check_file_exists('.gitignore', 'Git ignore file')
    
    if not all_files_exist:
        print("\n✗ Some required files are missing!")
        return False
    
    # Check imports
    if not check_imports():
        print("\n✗ Import checks failed!")
        return False
    
    # Check module structure
    if not check_module_structure():
        print("\n✗ Module structure checks failed!")
        return False
    
    # Run tests
    if not run_tests():
        print("\n✗ Tests failed!")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All validation checks passed!")
    print("=" * 60)
    print("\nThe AI Image Classifier is ready to use!")
    print("\nNote: To use the classifier with real images, you need to:")
    print("1. Ensure you have internet access for model download")
    print("2. Run: python classifier.py your_image.jpg")
    print("\n" + "=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
