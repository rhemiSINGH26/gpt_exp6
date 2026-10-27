"""
app.py
------
Flask Web Application for:
24CS2506 - Building Applications using GPT-4
EXPERIMENT 5 - AI-Powered Medical Symptom Assistant using GPT-4

Provides web dashboard, REST API endpoints, clinical report persistence,
and sample demo loaders for college viva demonstrations.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_file, session
from dotenv import load_dotenv
import medical_engine

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medical_symptom_assistant_secret_2026")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# In-memory storage for active reports in current runtime session
ACTIVE_REPORTS = {}

SAMPLE_SCENARIOS = {
    "respiratory": {
        "title": "Lab Manual Example: Respiratory Symptoms",
        "description": "Fever, cough, and sore throat for 2 days (Pages 1-3 of Lab Manual)",
        "symptoms_text": "I have had a sore throat, mild fever and cough since yesterday. I also feel a bit tired.",
        "age": 22,
        "sex": "Female",
        "duration": "2 days",
        "severity": "Mild",
        "medical_history": "None",
        "medications": "None",
        "allergies": "None"
    },
    "headache": {
        "title": "Neurological / Headache Scenario",
        "description": "Throbbing headache with screen sensitivity and fatigue",
        "symptoms_text": "I have a throbbing headache on one side of my head, sensitivity to light, and feel nauseous after working long hours.",
        "age": 28,
        "sex": "Male",
        "duration": "1 day",
        "severity": "Moderate",
        "medical_history": "Occasional tension headaches",
        "medications": "Paracetamol occasionally",
        "allergies": "None"
    },
    "emergency": {
        "title": "Emergency Escalation Scenario (Red Flag)",
        "description": "Crushing chest pain radiating to arm with shortness of breath",
        "symptoms_text": "I am having sudden crushing chest pain radiating to my left arm and jaw, along with acute shortness of breath and cold sweats.",
        "age": 54,
        "sex": "Male",
        "duration": "45 minutes",
        "severity": "Severe",
        "medical_history": "Hypertension, High cholesterol",
        "medications": "Amlodipine 5mg",
        "allergies": "Penicillin"
    }
}


@app.route("/")
def index():
    """Renders the main clinical assistant web dashboard."""
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns preset clinical scenarios for quick testing during viva evaluations."""
    return jsonify({"success": True, "samples": SAMPLE_SCENARIOS})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns system status, configured model, and API connection information."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    has_key = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here")

    return jsonify({
        "status": "online",
        "model": model_name,
        "has_api_key": has_key,
        "fallback_ready": True,
        "experiment": "24CS2506 - Experiment 5: AI-Powered Medical Symptom Assistant using GPT-4"
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Executes the clinical symptom analysis pipeline:
    Validates input -> Builds clinical prompt -> Calls GPT-4 / Clinical Fallback -> Returns structured analysis.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Request body must be JSON."}), 400

        result, err = medical_engine.analyze_symptoms(data)
        if err:
            return jsonify({"success": False, "error": err}), 400

        # Auto-save report to disk as required by Step 10 of workflow
        report_id, txt_path, json_path = medical_engine.save_report_to_disk(result, REPORTS_DIR)
        result["report_id"] = report_id
        ACTIVE_REPORTS[report_id] = result

        return jsonify({
            "success": True,
            "data": result,
            "report_id": report_id
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal Server Error: {str(e)}"}), 500


@app.route("/api/save-report", methods=["POST"])
def save_report():
    """Saves the current report to disk and returns file details."""
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"success": False, "error": "Report data is required."}), 400

        report_data = data["data"]
        report_id, txt_path, json_path = medical_engine.save_report_to_disk(report_data, REPORTS_DIR)
        ACTIVE_REPORTS[report_id] = report_data

        return jsonify({
            "success": True,
            "report_id": report_id,
            "message": f"Report successfully saved to disk as {os.path.basename(txt_path)}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/download-report/<report_id>", methods=["GET"])
def download_report(report_id):
    """
    Downloads saved clinical report in .txt or .json format.
    Query param `format`: 'txt' (default) or 'json'.
    """
    fmt = request.args.get("format", "txt").lower()
    filename = f"{report_id}.{fmt if fmt in ['txt', 'json'] else 'txt'}"
    filepath = os.path.join(REPORTS_DIR, filename)

    if not os.path.exists(filepath):
        # Check active reports in memory if file is not on disk yet
        if report_id in ACTIVE_REPORTS:
            medical_engine.save_report_to_disk(ACTIVE_REPORTS[report_id], REPORTS_DIR)

    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": f"Report '{report_id}' not found."}), 404

    mimetype = "text/plain" if fmt == "txt" else "application/json"
    return send_file(
        filepath,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"Medical_Symptom_Analysis_{report_id}.{fmt}"
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("=" * 70)
    print(" 24CS2506 - Building Applications using GPT-4")
    print(" EXPERIMENT 5: AI-Powered Medical Symptom Assistant using GPT-4")
    print("=" * 70)
    print(f" Starting Medical Symptom Assistant server at: http://127.0.0.1:{port}")
    print(f" Ready for interactive browser demonstrations and college viva.")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=True)
