# Indian Sign Language (ISL) Recognition and Translation

![ISL Recognition](https://img.shields.io/badge/ISL-Recognition-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

An end-to-end system for recognizing and translating Indian Sign Language (ISL) gestures into text and speech. This project utilizes computer vision and deep learning to bridge the communication gap between the hearing-impaired community and the general public.

## ✨ Features

- **Real-time ISL Recognition**: Detect and classify hand gestures using your webcam
- **Text-to-ISL Translation**: Convert written text into corresponding ISL gestures
- **Speech-to-ISL**: Convert spoken language into ISL gestures
- **Interactive Web Interface**: User-friendly interface for easy interaction
- **Model Training**: Tools to train and improve the recognition model

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time recognition)
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KrupaMohan/Indian-Sign-Language-ISL--recognition-and-translation.git
   cd Indian-Sign-Language-ISL--recognition-and-translation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model and data**
   - The pre-trained model (`model.h5`) and data files (`X_train.npy`, `y_train.npy`) are included in the repository
   - For custom training, place your dataset in the `data` directory with subdirectories for each letter (A-Z)

## 🖥️ Usage

### Web Application
Run the Flask web application:
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

### Real-time Detection
For real-time ISL recognition using your webcam:
```bash
python realtime_detection.py
```

### Text to ISL
Convert text to ISL gestures:
```bash
python text_to_isl.py --text "HELLO"
```

### Speech to ISL
Convert speech to ISL gestures (requires microphone):
```bash
python speech_to_isl.py
```

## 🏗️ Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Dataset directory (not included in git)
├── static/               # Static files (CSS, JS, images)
│   └── isl_output/      # Generated ISL gesture images
├── templates/            # HTML templates for the web app
├── model.h5              # Pre-trained model
├── model_analysis.py     # Model evaluation and visualization
├── predict.py            # Prediction utilities
├── realtime_detection.py # Real-time ISL detection
├── speech_to_isl.py      # Speech to ISL conversion
├── text_splitter.py      # Text processing utilities
├── text_to_isl.py        # Text to ISL conversion
├── train_model.py        # Model training script
└── variables.py          # Global variables and configurations
```

## 🧠 Model Architecture

The recognition model uses a Convolutional Neural Network (CNN) with the following architecture:

1. Input Layer: 200x200x3 (RGB images)
2. Convolutional Layers with MaxPooling
3. Dense Layers with Dropout for regularization
4. Output Layer with Softmax activation for classification

## 📊 Performance

The model achieves the following performance metrics:

- Training Accuracy: ~98%
- Validation Accuracy: ~95%
- Test Accuracy: ~94%

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Special thanks to all the open-source contributors and researchers in the field of computer vision and sign language recognition.
- This project was inspired by the need for better communication tools for the hearing-impaired community in India.
