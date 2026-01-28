from flask import Flask, request, jsonify, send_file
import cv2, os, sqlite3, random, traceback, requests
from datetime import datetime


# ================= CONFIG =================

OPENAI_KEY = os.getenv("OPENAI_API_KEY")   # From Render ENV
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ================= APP =================

app = Flask(__name__)

DB = "database.db"
UPLOAD = "uploads"
HEAT = "heatmaps"

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


# ================= AI DOCTOR (OPENROUTER) =================

def ai_medical_report(name, result, conf, history):

    if not OPENAI_KEY:
        return "AI API Key not configured."

    prompt = f"""
You are a lung specialist doctor.

Patient Name: {name}

History:
{history}

Current Result: {result}
Confidence: {conf}

Explain clearly:

1. Before COVID lung condition
2. After COVID impact
3. Current lung health
4. Possible problems
5. Treatment advice
6. Lifestyle guidance

Add medical disclaimer.
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://my-r6pu.onrender.com",
        "X-Title": "AI Lung Health Assistant"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a medical doctor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 700
    }

    try:

        r = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if r.status_code != 200:
            print("OpenRouter Error:", r.text)
            return "AI Doctor temporarily unavailable."

        return r.json()["choices"][0]["message"]["content"]

    except:
        return "AI Doctor service error."


# ================= PREDICT =================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        name = request.form.get("name", "Unknown")

        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]

        path = os.path.join(UPLOAD, file.filename)

        file.save(path)

        img = cv2.imread(path)

        if img is None:
            return jsonify({"error": "Invalid Image"}), 400


        # ---------- Prediction ----------

        result, conf = predict_ai()

        report = "Lung Status : " + result


        # ---------- Heatmap ----------

        heat = make_heatmap(img, file.filename)


        # ---------- History ----------

        con = sqlite3.connect(DB)
        cur = con.cursor()

        cur.execute(
            "SELECT date,result FROM patients WHERE name=?",
            (name,)
        )

        rows = cur.fetchall()

        con.close()

        history_text = ""

        for r in rows:
            history_text += f"{r[0]} : {r[1]}\n"


        # ---------- AI Doctor ----------

        ai_report = ai_medical_report(
            name,
            result,
            conf,
            history_text
        )


        # ---------- Save ----------

        save(name, result, conf, path, report)


        return jsonify({

            "prediction": result,

            "confidence": conf,

            "report": report,

            "treatment": "Consult Pulmonologist",

            "lifestyle": "No Smoking, Exercise",

            "heatmap": heat,

            "ai_doctor": ai_report
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


# ================= HEATMAP =================

@app.route("/heatmap/<name>")
def heat(name):

    return send_file(os.path.join(HEAT, name))


# ================= CHAT =================

@app.route("/chat", methods=["POST"])
def chat():

    msg = request.json["msg"]

    reply = "Please consult doctor regarding: " + msg

    return jsonify({"reply": reply})


# ================= PDF =================

@app.route("/generate_pdf/<int:pid>")
def generate_pdf(pid):

    return jsonify({"msg": "Use browser download"})


# ================= RUN =================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
