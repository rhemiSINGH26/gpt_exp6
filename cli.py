"""
cli.py
------
Terminal / Command-Line Interface for:
24CS2506 - Building Applications using GPT-4
EXPERIMENT 5 - AI-Powered Medical Symptom Assistant using GPT-4

Allows direct command-line execution and interactive viva testing.
"""

import os
import sys
import json
import medical_engine


def print_banner():
    print("=" * 75)
    print(" 24CS2506 - Building Applications using GPT-4")
    print(" EXPERIMENT 5: AI-Powered Medical Symptom Assistant using GPT-4")
    print(" CLI Diagnostic Mode & Lab Viva Demonstrator")
    print("=" * 75)


def run_sample(choice):
    samples = {
        "1": {
            "name": "Respiratory Symptoms (Lab Manual Sample)",
            "symptoms_text": "I have had a sore throat, mild fever and cough since yesterday.",
            "age": 22,
            "sex": "Female",
            "duration": "2 days",
            "severity": "Mild",
            "medical_history": "None",
            "medications": "None",
            "allergies": "None"
        },
        "2": {
            "name": "Neurological / Tension & Migraine Headache",
            "symptoms_text": "I have a throbbing headache on one side of my head, sensitivity to bright light, and feel nauseous.",
            "age": 28,
            "sex": "Male",
            "duration": "1 day",
            "severity": "Moderate",
            "medical_history": "None",
            "medications": "None",
            "allergies": "None"
        },
        "3": {
            "name": "Emergency Red Flag (Acute Chest Pain)",
            "symptoms_text": "I have crushing chest pain radiating to my left arm, shortness of breath, and feeling dizzy.",
            "age": 55,
            "sex": "Male",
            "duration": "30 minutes",
            "severity": "Severe",
            "medical_history": "Hypertension",
            "medications": "Blood pressure medicine",
            "allergies": "Penicillin"
        }
    }

    if choice not in samples:
        print("Invalid choice.")
        return

    sample = samples[choice]
    print(f"\n[+] Running Pre-configured Scenario: {sample['name']}")
    print(f"    Symptom input: \"{sample['symptoms_text']}\"")
    print(f"    Age: {sample['age']} | Severity: {sample['severity']} | Duration: {sample['duration']}")
    execute_and_display(sample)


def run_custom():
    print("\n--- Enter Patient Profile & Symptoms ---")
    symptoms = input("Enter Symptoms in Natural Language: ").strip()
    if not symptoms:
        print("[-] Symptoms cannot be empty.")
        return

    age_str = input("Patient Age: ").strip()
    try:
        age = int(age_str)
    except ValueError:
        print("[-] Invalid age. Defaulting to 25.")
        age = 25

    sex = input("Patient Sex (Male/Female/Other): ").strip() or "Not specified"
    duration = input("Duration (e.g. 2 days, 1 week): ").strip() or "2 days"
    severity = input("Severity (Mild/Moderate/Severe): ").strip().capitalize() or "Mild"
    history = input("Pre-existing Medical Conditions (or None): ").strip() or "None"
    meds = input("Current Medications (or None): ").strip() or "None"
    allergies = input("Known Allergies (or None): ").strip() or "None"

    patient = {
        "symptoms_text": symptoms,
        "age": age,
        "sex": sex,
        "duration": duration,
        "severity": severity,
        "medical_history": history,
        "medications": meds,
        "allergies": allergies
    }

    execute_and_display(patient)


def execute_and_display(patient_data):
    print("\n[+] Processing workflow: Extracting symptoms -> Analyzing context -> GPT-4 Inference...")
    result, err = medical_engine.analyze_symptoms(patient_data)
    if err:
        print(f"[-] Error: {err}")
        return

    # Print the structured text report
    print("\n" + result["formatted_text_report"])

    # Step 10: Save Report
    report_id, txt_path, json_path = medical_engine.save_report_to_disk(result)
    print(f"[✓] Report automatically saved to disk:")
    print(f"    Text: {txt_path}")
    print(f"    JSON: {json_path}")
    print("=" * 75)


def main():
    print_banner()
    while True:
        print("\nSelect an Option:")
        print("  1. Run Lab Manual Sample 1 (Respiratory: Sore Throat, Fever, Cough)")
        print("  2. Run Sample 2 (Headache & Light Sensitivity)")
        print("  3. Run Sample 3 (Emergency Red Flag Escalation)")
        print("  4. Enter Custom Symptoms & Patient Context")
        print("  5. Exit")

        choice = input("\nEnter selection (1-5): ").strip()
        if choice in ["1", "2", "3"]:
            run_sample(choice)
        elif choice == "4":
            run_custom()
        elif choice == "5":
            print("\nExiting. Thank you!")
            sys.exit(0)
        else:
            print("[-] Invalid selection. Please enter 1 to 5.")


if __name__ == "__main__":
    main()
