from flask import Flask, render_template, request
import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi model InsightFace
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)

def swap_faces(src_img_path, dst_img_path, output_path="static/output.jpg"):
    src_img = cv2.imread(src_img_path)
    dst_img = cv2.imread(dst_img_path)

    src_faces = face_app.get(src_img)
    dst_faces = face_app.get(dst_img)

    if not src_faces or not dst_faces:
        raise ValueError("Face not detected in one or both images.")

    result = dst_img.copy()
    result = swapper.get(result, dst_faces[0], src_faces[0], paste_back=True)

    cv2.imwrite(output_path, result)
    return output_path
def swap_faces_in_one_image(img_path, output_path="static/output_swap.jpg"):
    img = cv2.imread(img_path)
    faces = face_app.get(img)

    if len(faces) < 2:
        raise ValueError("Minimal dua wajah diperlukan dalam satu gambar.")

    # Ambil dua wajah pertama
    face1 = faces[0]
    face2 = faces[1]

    # Salin gambar untuk manipulasi
    swapped_img = img.copy()

    # Tukar wajah face1 ke face2 dan sebaliknya
    swapped_img = swapper.get(swapped_img, face1, face2, paste_back=True)
    swapped_img = swapper.get(swapped_img, face2, face1, paste_back=True)

    cv2.imwrite(output_path, swapped_img)
    return output_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        source_file = request.files["source"]
        target_file = request.files["target"]

        src_path = os.path.join(UPLOAD_FOLDER, "source.jpg")
        tgt_path = os.path.join(UPLOAD_FOLDER, "target.jpg")

        # Simpan file upload ke uploads/
        source_file.save(src_path)
        target_file.save(tgt_path)

        # Simpan salinan untuk ditampilkan di web
        from PIL import Image
        Image.open(src_path).save("static/source_preview.jpg")
        Image.open(tgt_path).save("static/target_preview.jpg")

        try:
            output_path = swap_faces(src_path, tgt_path)
            return render_template("index.html", result_img=output_path)
        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html", result_img=None)

@app.route("/single", methods=["GET", "POST"])
def single_face_swap():
    if request.method == "POST":
        image_file = request.files["image"]
        img_path = os.path.join(UPLOAD_FOLDER, "single.jpg")
        image_file.save(img_path)

        # Simpan preview
        Image.open(img_path).save("static/single_preview.jpg")

        try:
            output_path = swap_faces_in_one_image(img_path)
            return render_template("single.html", result_img=output_path)
        except Exception as e:
            return f"Error saat swap di satu gambar: {e}"

    return render_template("single.html", result_img=None)

if __name__ == "__main__":
    app.run(debug=True)

