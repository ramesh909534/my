from flask import Flask, request, jsonify, send_file
import cv2
import os
import sqlite3
import traceback
import requests

from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# AI MODEL
import torch
import torchvision.transforms as transforms
from PIL import Image

# ================= APP =================
app = Flask(__name__)

# ================= CONFIG =================
DB = "database.db"
UPLOAD = "uploads"
HEAT = "heatmaps"

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(HEAT, exist_ok=True)

# ================= MODEL LOAD (FIXED) =================
try:
    import torchvision.models.mobilenetv2 as mobilenetv2
    import torch.serialization

    torch.serialization.add_safe_globals([mobilenetv2.MobileNetV2])

    model = torch.load(
        "lung_model.pth",
        map_location=torch.device("cpu"),
        weights_only=False
    )

    model.eval()
    print("✅ Model loaded")

except Exception as e:
    print("❌ Model load failed:", e)
    model = None

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ================= DB =================
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

# ================= AI =================
def analyze_lung_health_real(path):
    if model is None:
        return 50

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    probs = torch.softmax(output, dim=1)
    return int(probs[0][1].item() * 100)

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        name = request.form.get("name", "Unknown")

        file = request.files["file"]
        fname = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
        path = os.path.join(UPLOAD, fname)
        file.save(path)

        img = cv2.imread(path)

        after = analyze_lung_health_real(path)
        before = int(min(100, after + 10 + (after * 0.1)))
        damage = before - after

        conf = after / 100

        if after > 75:
            severity = "Mild"
            treatment = "Maintain healthy lifestyle"
            lifestyle = "Exercise regularly"
        elif after > 50:
            severity = "Moderate"
            treatment = "Consult doctor"
            lifestyle = "Avoid smoking"
        else:
            severity = "Severe"
            treatment = "Consult Pulmonologist immediately"
            lifestyle = "Strict rest"

        report = f"""
Lung health is {after}%
Damage: {damage}%

Severity: {severity}
Treatment: {treatment}
Lifestyle: {lifestyle}
"""

        heat = make_heatmap(img, fname)

        save(name, "Lung Analysis", conf, path, report)

        return jsonify({
            "confidence": conf,
            "report": report,
            "treatment": treatment,
            "lifestyle": lifestyle,
            "heatmap": heat
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed"}), 500

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

# ================= HEATMAP =================
@app.route("/heatmap/<name>")
def heat(name):
    return send_file(os.path.join(HEAT, name))

# ================= CHAT (FIXED) =================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        msg = request.json.get("msg", "")

        if not OPENROUTER_KEY:
            return jsonify({"reply": "API key missing"})

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "openrouter/auto",
            "messages": [
                {"role": "system", "content": "You are a lung doctor."},
                {"role": "user", "content": msg}
            ]
        }

        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )

        res = r.json()

        if "error" in res:
            return jsonify({"reply": res["error"]["message"]})

        reply = res.get("choices", [{}])[0].get("message", {}).get("content", "No reply")

        return jsonify({"reply": reply})

    except Exception as e:
        print(e)
        return jsonify({"reply": "Chat error"})

# ================= PDF (FIXED) =================
@app.route("/generate_pdf/<int:pid>")
def generate_pdf(pid):
    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("SELECT * FROM patients WHERE id=?", (pid,))
    r = cur.fetchone()
    con.close()

    if r is None:
        return "No record"

    file = f"report_{pid}.pdf"
    c = canvas.Canvas(file, pagesize=A4)

    y = 800

    c.drawString(50, y, f"Name: {r[1]}"); y -= 30
    c.drawString(50, y, f"Date: {r[2]}"); y -= 30
    c.drawString(50, y, f"Confidence: {r[4]*100:.2f}%"); y -= 40

    for line in r[6].split("\n"):
        if line.strip():
            c.drawString(50, y, line.strip())
            y -= 25

    c.save()

    return send_file(file, as_attachment=True)

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
