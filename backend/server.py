from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
import sqlite3
import os
from datetime import datetime
import traceback

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ================= APP =================
app = Flask(__name__)

DB_FILE = "database.db"
MODEL_FILE = "model.h5"
IMG_SIZE = 224


# ================= DATABASE =================
def init_db():

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        result TEXT,
        confidence REAL,
        image_path TEXT,
        report TEXT
    )
    """)

    con.commit()
    con.close()


init_db()


# ================= MODEL =================
def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(224,224,3)),

        tf.keras.layers.Conv2D(16,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64,activation="relu"),

        tf.keras.layers.Dense(3,activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.save(MODEL_FILE)


if not os.path.exists(MODEL_FILE):
    create_model()


model = tf.keras.models.load_model(MODEL_FILE)

# Build model
model.predict(np.zeros((1,224,224,3)))


CLASSES = ["Normal","Benign","Malignant"]


# ================= SAVE RECORD =================
def save_record(name,result,conf,img,report):

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
    INSERT INTO patients
    (name,date,result,confidence,image_path,report)
    VALUES(?,?,?,?,?,?)
    """,(name,
         datetime.now().strftime("%d-%m-%Y %H:%M"),
         result,
         conf,
         img,
         report))

    con.commit()
    con.close()


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        name = request.form.get("name","Unknown")

        if "file" not in request.files:
            return jsonify({"error":"No File"}),400


        file = request.files["file"]

        os.makedirs("uploads", exist_ok=True)
        os.makedirs("heatmaps", exist_ok=True)


        path = "uploads/" + file.filename

        file.save(path)


        # ===== OpenCV Load =====
        img = cv2.imread(path)

        if img is None:
            return jsonify({"error":"Invalid Image"}),400


        img_resize = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img_norm = img_resize / 255.0

        arr = np.expand_dims(img_norm,0)


        # ===== Predict =====
        pred = model.predict(arr)[0]

        idx = int(np.argmax(pred))

        result = CLASSES[idx]

        conf = float(pred[idx])

        report = "Lung Status : " + result


        # ===== Heatmap (OpenCV) =====
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray,(15,15),0)

        heat = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img,0.6,heat,0.4,0)


        heat_path = "heatmaps/heat_" + file.filename

        cv2.imwrite(heat_path, overlay)


        # ===== Save DB =====
        save_record(
            name,
            result,
            conf,
            path,
            report
        )


        return jsonify({

            "prediction": result,

            "confidence": round(conf,2),

            "report": report,

            "treatment": "Consult doctor",

            "lifestyle": "Healthy food",

            "heatmap": heat_path
        })


    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":"Processing Failed",

            "details":str(e)

        }),500


# ================= HISTORY =================
@app.route("/history")
def history():

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("SELECT * FROM patients")

    rows = cur.fetchall()

    con.close()


    data = []

    for r in rows:

        data.append({

            "id":r[0],

            "name":r[1],

            "date":r[2],

            "result":r[3],

            "confidence":r[4],

            "image":r[5],

            "report":r[6]
        })


    return jsonify(data)


# ================= SEND HEATMAP =================
@app.route("/heatmap/<path:file>")
def send_heatmap(file):

    return send_file(file, mimetype="image/png")


# ================= PDF =================
@app.route("/generate_pdf/<int:pid>")
def generate_pdf(pid):

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("SELECT * FROM patients WHERE id=?", (pid,))

    row = cur.fetchone()

    con.close()


    if row is None:
        return jsonify({"error":"No record"})


    file_name = f"report_{pid}.pdf"


    c = canvas.Canvas(file_name, pagesize=A4)

    w,h = A4


    c.setFont("Helvetica-Bold",22)
    c.drawString(150,h-50,"AI Lung Health Report")


    c.setFont("Helvetica",14)

    y = h-120


    c.drawString(50,y,f"Name : {row[1]}"); y-=30
    c.drawString(50,y,f"Date : {row[2]}"); y-=30
    c.drawString(50,y,f"Result : {row[3]}"); y-=30
    c.drawString(50,y,f"Confidence : {round(row[4]*100,2)}%"); y-=40

    c.drawString(50,y,"Treatment : Consult Doctor"); y-=30
    c.drawString(50,y,"Lifestyle : Healthy Diet"); y-=40

    c.drawString(50,y,"Doctor Advice : Regular Checkup")

    c.save()


    return jsonify({"file":file_name})


# ================= RUN =================
if __name__=="__main__":

    app.run(debug=True)
