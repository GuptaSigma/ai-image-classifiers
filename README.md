# AI Image Classifiers

A Python-based image classification system using pre-trained deep learning models. This project provides an easy-to-use interface for classifying images into 1000 different categories using the MobileNetV2 model trained on ImageNet.

## Features

- **Pre-trained Model**: Uses MobileNetV2 with ImageNet weights for accurate classification
- **Easy to Use**: Simple API and CLI interface
- **Fast**: Optimized for quick inference
- **1000+ Categories**: Can identify over 1000 different object categories
- **Top-K Predictions**: Get multiple predictions with confidence scores

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GuptaSigma/ai-image-classifiers.git
cd ai-image-classifiers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Classify an image from the command line:

```bash
python classifier.py path/to/your/image.jpg
```

Get top 10 predictions:

```bash
python classifier.py path/to/your/image.jpg 10
```

### Python API

Use the classifier in your Python code:

```python
from classifier import ImageClassifier

# Initialize the classifier
classifier = ImageClassifier()

# Classify an image
predictions = classifier.classify('image.jpg', top_k=5)

# Print results
for class_id, class_name, probability in predictions:
    print(f"{class_name}: {probability*100:.2f}%")

# Or use the convenience method
classifier.classify_and_print('image.jpg')
```

### Example Script

Run the example script to see usage demonstrations:

```bash
python example.py
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy 1.24+
- Pillow 10.0+

See `requirements.txt` for full dependencies.

## How It Works

The classifier uses the MobileNetV2 architecture, a lightweight convolutional neural network designed for mobile and edge devices. The model is pre-trained on the ImageNet dataset, which contains over 14 million images across 1000 categories.

When you classify an image:
1. The image is loaded and resized to 224x224 pixels
2. Preprocessing is applied to normalize pixel values
3. The image is passed through the neural network
4. The model outputs probabilities for each of the 1000 categories
5. The top predictions are decoded and returned

## Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- GIF (first frame)

## Categories

The model can identify 1000 different categories from ImageNet, including:
- Animals (dogs, cats, birds, etc.)
- Vehicles (cars, trucks, airplanes, etc.)
- Objects (furniture, electronics, tools, etc.)
- Food items
- Plants and flowers
- And many more!

## Project Structure

```
ai-image-classifiers/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── classifier.py       # Main classifier module
├── example.py         # Example usage script
└── .gitignore         # Git ignore file
```

## License

This project is open source and available for educational and commercial use.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

- Pre-trained model: MobileNetV2 from TensorFlow/Keras
- Dataset: ImageNet
- Framework: TensorFlow
