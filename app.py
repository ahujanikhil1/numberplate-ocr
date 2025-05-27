from flask import Flask, render_template, request, redirect, url_for
import cv2
import pytesseract
import numpy as np
import re
from PIL import Image
import os
from werkzeug.utils import secure_filename
import platform


app = Flask(__name__)

# Set Tesseract path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'D:/ocr/tesseract.exe'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Your existing helper functions (order_points, perspective_transform, preprocess_image, etc.) here:

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def perspective_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        return image
    rect = order_points(screenCnt.reshape(4, 2))
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = perspective_transform(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def extract_text(image_path):
    processed_img = preprocess_image(image_path)
    config = "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(processed_img, config=config).strip()
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text)
    return cleaned_text

def correct_plate_errors(plate):
    corrections = {
        "A": "4",
        "O": "0",
        "0": "C",
        "I": "1",
        "B": "8",
        "8": "B",
        "S": "5",
        "5": "S",
        "L": "4",
        "9": "3",
        "Z": "4",
        "Y": "4"
    }
    corrected_plate = plate[:2]
    for i in range(2, len(plate)):
        char = plate[i]
        if char == "0" and i + 1 < len(plate) and plate[i + 1].isdigit():
            corrected_plate += "C"
        else:
            corrected_plate += char
    corrected_plate = "".join([corrections.get(char, char) for char in corrected_plate])
    if corrected_plate[2] == "0" and len(corrected_plate) > 3 and corrected_plate[3].isdigit():
        corrected_plate = corrected_plate[:2] + "C" + corrected_plate[3:]
    if is_valid_plate(corrected_plate):
        return corrected_plate
    return plate

def is_valid_plate(plate):
    normal_plate_pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"
    bh_plate_pattern = r"^[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}$"
    dl_plate_pattern = r"^DL[0-9]{1,2}C[A-Z]{1,2}[0-9]{4}$"
    return (
        bool(re.match(normal_plate_pattern, plate)) or
        bool(re.match(bh_plate_pattern, plate)) or
        bool(re.match(dl_plate_pattern, plate))
    )


@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    corrected_text = ""
    valid = False
    filename = None
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            extracted_text = extract_text(filepath)
            corrected_text = correct_plate_errors(extracted_text)
            valid = is_valid_plate(corrected_text)

    return render_template("index.html",
                           filename=filename,
                           extracted_text=corrected_text if valid else "",
                           valid=valid)

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8081))
    app.run(host="0.0.0.0",port=port,debug=True)





