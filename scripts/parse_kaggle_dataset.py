import os
import re
import json
import random
from typing import List, Dict, Any

random.seed(42)

STAGE_MAP = {
    0: "none",
    1: "impersonation",
    2: "allegation",
    3: "isolation",
    4: "coercion",
    5: "payment"
}


def clean_text(text: str) -> str:
    """Cleans placeholders and extra whitespace from text."""
    # Remove leading numbering like "1. ", "12. "
    text = re.sub(r"^\d+\.\s*", "", text).strip()
    
    # Replace common template placeholders with natural Indian/global phrasing
    text = text.replace("[Greetings]", "Hello")
    text = text.replace("[Company]", "Cyber Division")
    text = text.replace("[Product]", "account services")
    text = text.replace("[Name]", "Sharma")
    text = text.replace("[Number]", "9876543210")
    text = text.replace("[Date]", "today")
    text = text.replace("[Time]", "immediate")
    text = text.replace("[City]", "Mumbai")
    text = text.replace("[Title]", "Officer")
    
    # Clean multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_triggers(text: str, is_scam: int) -> Dict[str, int]:
    """Detects multi-label psychological triggers using domain keyword patterns."""
    if is_scam == 0:
        return {"authority": 0, "urgency": 0, "isolation": 0, "payment_pressure": 0}

    lower_text = text.lower()

    # Authority keywords
    auth_kw = ["police", "cbi", "officer", "bank", "manager", "trai", "department", "government", "ed", "court", "inspector", "customer care", "grant", "support", "security", "bureau", "agency"]
    authority = 1 if any(kw in lower_text for kw in auth_kw) else 0

    # Urgency keywords
    urg_kw = ["urgent", "immediate", "immediately", "24 hours", "23 hours", "expire", "block", "suspend", "cancel", "deactivate", "today", "right away", "emergency", "fast", "soon", "15 minutes"]
    urgency = 1 if any(kw in lower_text for kw in urg_kw) else 0

    # Isolation keywords
    iso_kw = ["secret", "don't tell", "dont tell", "keep", "room", "lock", "private", "confidential", "only one", "video call", "loudspeaker", "alone"]
    isolation = 1 if any(kw in lower_text for kw in iso_kw) else 0

    # Payment pressure keywords
    pay_kw = ["fee", "pay", "payment", "transfer", "money", "credit card", "bank account", "otp", "kyc", "wire", "claim", "deposit", "award", "prize", "fund", "balance", "upi", "rtgs", "charge"]
    payment_pressure = 1 if any(kw in lower_text for kw in pay_kw) else 0

    # Default scam triggers if none matched
    if authority == 0 and urgency == 0 and isolation == 0 and payment_pressure == 0:
        authority = 1
        urgency = 1

    return {
        "authority": authority,
        "urgency": urgency,
        "isolation": isolation,
        "payment_pressure": payment_pressure
    }


def detect_stage(text: str, is_scam: int) -> int:
    """Categorizes scam stage from dialogue text."""
    if is_scam == 0:
        return 0

    lower_text = text.lower()

    if any(k in lower_text for k in ["transfer", "otp", "kyc", "fee", "pay", "credit card", "bank account", "wire", "deposit", "escrow", "upi"]):
        return 5  # payment
    elif any(k in lower_text for k in ["arrest", "jail", "block", "deactivate", "expire", "legal action", "warrant", "court order", "freeze"]):
        return 4  # coercion
    elif any(k in lower_text for k in ["secret", "don't tell", "lock room", "video call", "private", "confidential"]):
        return 3  # isolation
    elif any(k in lower_text for k in ["illegal", "virus", "breach", "suspicious", "crime", "drugs", "contraband", "warrant", "tax bill", "discrepancies"]):
        return 2  # allegation
    elif any(k in lower_text for k in ["calling from", "customer care", "police", "cbi", "trai", "officer", "department", "government", "grant"]):
        return 1  # impersonation
    else:
        return 1


def parse_file_scam_txt(filepath: str, is_scam: int, prefix: str) -> List[Dict[str, Any]]:
    samples = []
    if not os.path.exists(filepath):
        return samples

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = [clean_text(line) for line in content.split("\n") if line.strip()]
    
    for i, line in enumerate(lines):
        if len(line) < 10:
            continue
        triggers = detect_triggers(line, is_scam)
        stage = detect_stage(line, is_scam)
        sample = {
            "id": f"{prefix}_{i+1:04d}",
            "text": line,
            "is_scam": is_scam,
            "triggers": triggers,
            "scam_stage": stage,
            "scam_stage_name": STAGE_MAP[stage]
        }
        samples.append(sample)
        
    return samples


def parse_file_fraud_call(filepath: str) -> List[Dict[str, Any]]:
    samples = []
    if not os.path.exists(filepath):
        return samples

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, raw_line in enumerate(lines):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split("\t")
        if len(parts) >= 2:
            label_str = parts[0].strip().lower()
            text = clean_text(parts[1].strip())
            if len(text) < 5:
                continue

            is_scam = 1 if label_str in ["fraud", "scam", "spam"] else 0
            triggers = detect_triggers(text, is_scam)
            stage = detect_stage(text, is_scam)

            sample = {
                "id": f"fraudcall_{i+1:04d}",
                "text": text,
                "is_scam": is_scam,
                "triggers": triggers,
                "scam_stage": stage,
                "scam_stage_name": STAGE_MAP[stage]
            }
            samples.append(sample)

    return samples


def main():
    data_dir = "data"
    output_file = os.path.join(data_dir, "scam_dataset.json")

    file_scam = os.path.join(data_dir, "English_Scam.txt")
    file_nonscam = os.path.join(data_dir, "English_NonScam.txt")
    file_fraudcall = os.path.join(data_dir, "fraud_call.file")

    samples_scam = parse_file_scam_txt(file_scam, is_scam=1, prefix="eng_scam")
    samples_nonscam = parse_file_scam_txt(file_nonscam, is_scam=0, prefix="eng_legit")
    samples_fraudcall = parse_file_fraud_call(file_fraudcall)

    all_samples = samples_scam + samples_nonscam + samples_fraudcall
    
    # Shuffle for training uniformity
    random.shuffle(all_samples)

    scam_count = sum(1 for s in all_samples if s["is_scam"] == 1)
    legit_count = len(all_samples) - scam_count

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print("==================================================")
    print("      KAGGLE DATASET PARSED & MERGED SUCCESS      ")
    print("==================================================")
    print(f"Output File Saved: '{output_file}'")
    print(f"Total Combined Samples: {len(all_samples)}")
    print(f" - Scam / Fraud Samples: {scam_count}")
    print(f" - Legitimate Samples:   {legit_count}")
    print("==================================================")


if __name__ == "__main__":
    main()
