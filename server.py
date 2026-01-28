from flask import Flask, request, jsonify, send_file
import cv2
import os
import sqlite3
import random
import traceback
import requests

from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ================= APP =================
app = Flask(__name__)


# ================= CONFIG =================
DB = "database.db"
UPLOAD = "uploads"
HEAT = "heatmaps"

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")


os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(HEAT, exist_ok=True)


# ================= DATABASE =================
def init_db():

    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        result TEXT,
        confidence REAL,
        image TEXT,
        report TEXT
    )
    """)

    con.commit()
    con.close()


init_db()


# ================= SAVE =================
def save(name, res, conf, img, rep):

    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("""
    INSERT INTO patients VALUES(NULL,?,?,?,?,?,?)
    """, (
        name,
        datetime.now().strftime("%d-%m-%Y %H:%M"),
        res,
        conf,
        img,
        rep
    ))

    con.commit()
    con.close()


# ================= HEATMAP =================
def make_heatmap(img, fname):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    heat = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    final = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

    out = "heat_" + fname

    path = os.path.join(HEAT, out)

    cv2.imwrite(path, final)

    return out


# ================= DEMO AI =================
def predict_ai():

    classes = ["Normal", "Benign", "Malignant"]

    result = random.choices(
        classes,
        weights=[0.5, 0.3, 0.2]
    )[0]

    if result == "Normal":
        conf = random.uniform(0.7, 0.95)

    elif result == "Benign":
        conf = random.uniform(0.6, 0.85)

    else:
        conf = random.uniform(0.75, 0.98)

    return result, round(conf, 2)


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        name = request.form.get("name", "Unknown")

        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400


        file = request.files["file"]

        fname = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename

        path = os.path.join(UPLOAD, fname)

        file.save(path)


        img = cv2.imread(path)

        if img is None:
            return jsonify({"error": "Invalid Image"}), 400


        # AI
        result, conf = predict_ai()

        report = "Lung Status : " + result


        # Heatmap
        heat = make_heatmap(img, fname)


        # Save DB
        save(name, result, conf, path, report)


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
            "error": "Failed",
            "details": str(e)
        }), 500


# ================= HISTORY =================
@app.route("/history")
def history():

    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("SELECT * FROM patients")

    rows = cur.fetchall()

    con.close()


    data = []

    for r in rows:

        data.append({

            "id": r[0],
            "name": r[1],
            "date": r[2],
            "result": r[3],
            "confidence": r[4],
            "image": r[5],
            "report": r[6]
        })


    return jsonify(data)


# ================= HEATMAP FILE =================
@app.route("/heatmap/<name>")
def heat(name):

    return send_file(os.path.join(HEAT, name))


# ================= OPENROUTER CHAT =================
@app.route("/chat", methods=["POST"])
def chat():

    try:

        msg = request.json["msg"]


        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }


        data = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical assistant. Give safe health advice."
                },
                {
                    "role": "user",
                    "content": msg
                }
            ],
            "max_tokens": 250
        }


        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )


        res = r.json()


        if "choices" not in res:

            return jsonify({
                "reply": "AI service error"
            })


        reply = res["choices"][0]["message"]["content"]


        return jsonify({"reply": reply})


    except Exception as e:

        return jsonify({
            "reply": "AI unavailable. Consult doctor."
        })


# ================= PDF =================
@app.route("/generate_pdf/<int:pid>")
def generate_pdf(pid):

    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("SELECT * FROM patients WHERE id=?", (pid,))
    r = cur.fetchone()

    con.close()


    if r is None:
        return jsonify({"error": "No record"})


    file = f"report_{pid}.pdf"

    c = canvas.Canvas(file, pagesize=A4)

    w, h = A4


    c.setFont("Helvetica-Bold", 22)
    c.drawString(150, h-50, "AI Lung Health Report")

    c.setFont("Helvetica", 14)

    y = h-120


    c.drawString(50, y, f"Name : {r[1]}"); y -= 30
    c.drawString(50, y, f"Date : {r[2]}"); y -= 30
    c.drawString(50, y, f"Result : {r[3]}"); y -= 30
    c.drawString(50, y, f"Confidence : {r[4]*100:.2f}%"); y -= 40

    c.drawString(50, y, "Treatment : Consult Doctor"); y -= 30
    c.drawString(50, y, "Lifestyle : Healthy Diet"); y -= 40

    c.drawString(50, y, "Doctor Advice : Regular Checkup")

    c.save()


    return send_file(
        file,
        as_attachment=True,
        download_name=file
    )


# ================= RUN (RENDER READY) =================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
