from flask import Flask, request, send_file, render_template_string
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/crop-face": {"origins": "*"}})

# Initialize Mediapipe face detector
mp_face_detection = mp.solutions.face_detection

# ---------- HTML template ----------
HTML_TEMPLATE = """ 
<!-- [unchanged HTML template - same as yours above] -->
"""  # Use your original full HTML_TEMPLATE here

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/health")
def health():
    return {"status": "healthy", "service": "face-cropper-api"}

@app.route("/crop-face", methods=["POST", "OPTIONS"])
def crop_face():
    if request.method == "OPTIONS":
        return "", 204

    if "image" not in request.files or request.files["image"].filename == "":
        return "No image provided", 400

    try:
        img_array = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return "Invalid image", 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            results = detector.process(img_rgb)

        if not results.detections:
            return "No face found", 404

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = img.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Add 50% margin
        mx, my = int(bw * 0.5), int(bh * 0.5)
        x1, y1 = max(x - mx, 0), max(y - my, 0)
        x2, y2 = min(x + bw + mx, w), min(y + bh + my, h)

        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (600, 600))

        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")

    except Exception as ex:
        return f"Error: {ex}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
