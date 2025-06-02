# Number Plate Recognition Web App

This project is a simple web application that allows users to upload an image of a vehicle, and it detects and displays the number plate text using Optical Character Recognition (OCR).

## 🔧 Features

- Upload vehicle images and detect license plate text
- Beautiful neon-themed user interface (now in blue)
- Feedback messages for both successful and failed detections
- Image preview for uploaded files

## 📦 Tech Stack

- Python
- Flask (for the web backend)
- OpenCV (for image preprocessing)
- pytesseract (for OCR)
- HTML5 + CSS3 (for frontend)
- Jinja2 (Flask templating engine)

## 📸 Example Use

1. Upload a clear image with a visible number plate.
2. The app displays the detected plate number.
3. If detection fails, a message will prompt you to upload a better image.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- `pip` installed

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/number-plate-recognition.git
cd number-plate-recognition
