from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import sqlite3
import os
from datetime import datetime
import random
import traceback

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ================= APP =================
app = Flask(__name__)

DB_FILE = "database.db"
UPLOAD = "uploads"
HEAT = "heatmaps"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(HEAT, exist_ok=True)


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


# ================= SAVE =================
def save_record(name,result,conf,img,report):

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
    INSERT INTO patients
    VALUES(NULL,?,?,?,?,?,?)
    """,(
        name,
        datetime.now().strftime("%d-%m-%Y %H:%M"),
        result,
        conf,
        img,
        report
    ))

    con.commit()
    con.close()


# ================= HEATMAP =================
def make_heatmap(img, fname):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(21,21),0)

    heat = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heat,0.4,0)

    out = "heat_" + fname

    path = os.path.join(HEAT, out)

    cv2.imwrite(path, overlay)

    return out


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        name = request.form.get("name","Unknown")

        if "file" not in request.files:
            return jsonify({"error":"No File"}),400


        file = request.files["file"]

        path = os.path.join(UPLOAD, file.filename)

        file.save(path)


        img = cv2.imread(path)

        if img is None:
            return jsonify({"error":"Invalid Image"}),400


        # ===== DEMO AI (Cloud Safe) =====
        classes = ["Normal","Benign","Malignant"]

        result = random.choice(classes)

        conf = round(random.uniform(0.6,0.95),2)

        report = "Lung Status : " + result


        # ===== Heatmap =====
        heat = make_heatmap(img, file.filename)


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
            "confidence": conf,
            "report": report,
            "treatment": "Consult Pulmonologist",
            "lifestyle": "No Smoking, Exercise",
            "heatmap": heat

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


# ================= HEATMAP FILE =================
@app.route("/heatmap/<name>")
def get_heatmap(name):

    return send_file(os.path.join(HEAT,name))


# ================= CHAT =================
@app.route("/chat", methods=["POST"])
def chat():

    msg = request.json["msg"]

    reply = "Please consult doctor for: " + msg

    return jsonify({"reply":reply})


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

    app.run(host="0.0.0.0", port=5000)
