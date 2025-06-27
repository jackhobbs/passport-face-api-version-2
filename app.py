from flask import Flask, request, send_file, render_template_string
from flask_cors import CORS                       # NEW
import numpy as np
import cv2
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/crop-face": {"origins": "*"}})   # NEW – allow any origin
# • In production replace "*" with ["http://127.0.0.1:5501"] or your real domain.

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- HTML template ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Cropper API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { text-align: center; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; margin: 20px 0; border-radius: 10px; }
        .upload-area:hover { border-color: #007bff; }
        input[type="file"] { margin: 20px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Cropper API</h1>
        <p>Upload an image to automatically crop and resize faces to 600 × 600 px</p>
        
        <div class="upload-area">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageInput" name="image" accept="image/*" required>
                <br>
                <button type="submit">Crop Face</button>
            </form>
        </div>
        
        <div id="result" class="result"></div>
        
        <h2>API Usage</h2>
        <p><strong>Endpoint:</strong> <code>POST /crop-face</code></p>
        <p><strong>Parameters:</strong> <code>image</code> (file upload)</p>
        <p><strong>Returns:</strong> Cropped face image (600×600 JPEG)</p>
        
        <h3>Example using curl:</h3>
        <pre>curl -X POST -F "image=@your_photo.jpg" http://localhost:5000/crop-face --output cropped_face.jpg</pre>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = function (e) {
            e.preventDefault();
            const formData = new FormData();
            const fileInput = document.getElementById('imageInput');
            const resultDiv = document.getElementById('result');

            if (!fileInput.files[0]) {
                resultDiv.innerHTML = '<p class="error">Please select an image</p>';
                return;
            }
            formData.append('image', fileInput.files[0]);
            resultDiv.innerHTML = '<p>Processing…</p>';

            /* KEY CHANGE: fetch absolute URL on port 5000 */
            fetch('http://localhost:5000/crop-face', {
                method: 'POST',
                body: formData
            })
            .then(resp => {
                if (resp.ok) return resp.blob();
                throw new Error('Error ' + resp.status);
            })
            .then(blob => {
                const url = URL.createObjectURL(blob);
                resultDiv.innerHTML = `
                    <p class="success">Face cropped successfully!</p>
                    <img src="${url}" alt="Cropped face" style="max-width: 300px; border:1px solid #ccc; border-radius:5px;">
                    <br><br>
                    <a href="${url}" download="cropped_face.jpg"><button>Download Image</button></a>
                `;
            })
            .catch(err => {
                resultDiv.innerHTML = '<p class="error">' + err.message + '</p>';
            });
        };
    </script>
</body>
</html>
"""

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
        # Let Flask-CORS handle preflights automatically.
        return "", 204

    if "image" not in request.files or request.files["image"].filename == "":
        return "No image provided", 400

    try:
        img_array = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return "Invalid image", 400

        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        if len(faces) == 0:
            return "No face found", 404

        # Choose largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add 50 % margin
        mx, my = int(w * .5), int(h * .5)
        x1, y1 = max(x - mx, 0), max(y - my, 0)
        x2, y2 = min(x + w + mx, img.shape[1]), min(y + h + my, img.shape[0])

        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (600, 600))
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")

    except Exception as ex:
        return f"Error: {ex}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
