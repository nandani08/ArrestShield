import os
import json
import random
from typing import List, Dict, Any

# Ensure deterministic output
random.seed(42)

# Schema Definitions
STAGE_MAP = {
    0: "none",
    1: "impersonation",
    2: "allegation",
    3: "isolation",
    4: "coercion",
    5: "payment"
}

TRIGGER_KEYS = ["authority", "urgency", "isolation", "payment_pressure"]


def generate_scam_samples(count: int = 250) -> List[Dict[str, Any]]:
    """Generates synthetic annotated digital-arrest and social engineering scam samples in Hinglish."""
    scam_templates = [
        # Stage 1: Impersonation (Authority + Urgency)
        {
            "stage": 1,
            "stage_name": "impersonation",
            "triggers": {"authority": 1, "urgency": 1, "isolation": 0, "payment_pressure": 0},
            "phrases": [
                "Namaste, main Mumbai Police Headquarters Cyber Cell se Senior Inspector Sharma bol raha hu. Aapka Aadhar number ek major criminal investigation se link hua hai.",
                "Attention! Main TRAI Department Delhi se speak kar raha hu. Aapka mobile number agle 2 ghante mein permanently block aur disconnect kar diya jayega.",
                "Good afternoon, this is Officer Vikram Singh from CBI Cyber Crime Division New Delhi. Aapke naam par ek suspicious bank account active mila hai.",
                "Sir, main Delhi Customs Department se Call Officer Roy bol raha hu. Mumbai International Airport par aapka parcel intercept kiya gaya hai.",
                "Warning! Main Crime Branch Officer Rajesh Varma bol raha hu. Supreme Court se aapke khilaaf arrest warrant issue ho gaya hai.",
                "Hello, main RBI Security Bureau se Officer Vivek speak kar raha hu. Aapka bank account fraudulent transactions ke liye flagged hua hai.",
                "Attention citizen! Main Enforcement Directorate (ED) Head Office New Delhi se Senior Dy. Director speak kar raha hu."
            ]
        },
        # Stage 2: Allegation (Authority + Urgency + Isolation)
        {
            "stage": 2,
            "stage_name": "allegation",
            "triggers": {"authority": 1, "urgency": 1, "isolation": 1, "payment_pressure": 0},
            "phrases": [
                "Aapke FedEx courier package mein 5 contraband passports, 140 grams MDMA drugs aur illegal SIM cards paaye gaye hain. Yeh serious national security crime hai.",
                "Aapke HDFC account se 45 Lakhs ki money laundering narcotics syndicate ke saath track hui hai. Aap primary suspect ho is hawala case mein.",
                "Aapke ID se 12 illegal SIM cards register karke cyber fraud aur blackmailing violence execute hui hai. Abhi immediate clearance chahiye.",
                "Aapke name par registered Canara Bank account human trafficking aur illegal fund transfers ke liye use hua hai. Fast reply karo varna team bhej rahe hain.",
                "Aapke parcel mein unauthorized foreign currency aur confidential documents mile hain. Enforcement Directorate aapko interrogate karegi.",
                "Aap par illegal money trail aur hawala funding का charge laga hai. Aap Supreme Court hearing attend karenge."
            ]
        },
        # Stage 3: Isolation (Authority + Urgency + Isolation)
        {
            "stage": 3,
            "stage_name": "isolation",
            "triggers": {"authority": 1, "urgency": 1, "isolation": 1, "payment_pressure": 0},
            "phrases": [
                "Yeh official Digital Arrest procedure hai. Aap room ka darwaza lock kar lo, kisi family member ya lawyer ko contact nahi karoge jab tak interrogation chal rahi hai.",
                "Camera aur microphone continuously ON rakho. Ghar ke baahar nikalna ya kisi se baat karna court Order violation mana jayega.",
                "Aap kisi bank officer ya police ko secret disclose nahi karoge. High-profile national secrecy matter hai, confidentiality maintain rakho.",
                "Virtual Digital Arrest start ho chuka hai. WhatsApp video call cut mat karna, varna local PCR van 10 minute mein aapke address par arrest karne aayegi.",
                "Darwaza band kar lo aur phone loudspeaker par rakho. Agar kisi aur ki aawaaz aayi toh obstruction of justice charge lagega.",
                "Aap akele ek shant kamre mein baitho aur kisi se bhi koi contact mat banao."
            ]
        },
        # Stage 4: Coercion (Authority + Urgency + Isolation + Payment Pressure)
        {
            "stage": 4,
            "stage_name": "coercion",
            "triggers": {"authority": 1, "urgency": 1, "isolation": 1, "payment_pressure": 1},
            "phrases": [
                "Agar aap immediate non-bailable arrest warrant cancel karwana chahte ho toh abhi RBI Financial Clearance Cell ko cooperate karo.",
                "Aapke saare bank accounts aur fixed deposits next 30 minutes mein freeze kar diye jayenge agar aapne official verification audit nahi karwaya.",
                "Supreme Court emergency bench judgment se pehle aapko safety clearance certificate issue karwana padega, varna Tihar Jail bheja jayega.",
                "Aap par IPC Section 420 aur Prevention of Money Laundering Act (PMLA) ka trial chalega agar abhi formal clearance security process follow nahi kiya.",
                "Fast decision lo! Cyber Crime Cell immediate physical custody ka order execute karne wala hai.",
                "Agle 15 minute mein jail ya clearance deposit, choice aapki hai!"
            ]
        },
        # Stage 5: Payment (Authority + Urgency + Payment Pressure)
        {
            "stage": 5,
            "stage_name": "payment",
            "triggers": {"authority": 1, "urgency": 1, "isolation": 0, "payment_pressure": 1},
            "phrases": [
                "Abhi ke abhi aap apne HDFC Bank balance ka 80% RBI Secret Clearance Escrow UPI ID `rbi.sec.clearance@okicici` par transfer karo. Audit ke baad refund ho jayega.",
                "Immediate safety deposit ke liye ₹2,50,000 RTGS/IMPS through is verified Government Audit Account number 918237465012 IFSC SBIN0004321 par bhejo.",
                "Aapke verification ke liye ₹50,000 Google Pay / PhonePe par UPI `gov.cyber.verification@paytm` par send karein. Verification letter 15 min mein mil jayega.",
                "Arrest Order freeze karne ke liye ₹1,20,000 ka clearance demand draft deposit karein is Court Registry Account ID par.",
                "Security bond deposit karein immediate online fund transfer se. Failure will result in immediate police raid at your current location.",
                "Aap turant ₹75,000 is official Escrow account par transfer karke verification receipt get karein."
            ]
        }
    ]

    fillers = [
        "Jaldi suniye,", "Listen to me carefully,", "Aap samajh rahe ho?", "Yeh mazak nahi hai,",
        "Samjhe aap?", "Immediately follow the instructions,", "Do not cut the call,",
        "Main warning de raha hu,"
    ]

    samples = []
    for i in range(count):
        tmpl = scam_templates[i % len(scam_templates)]
        filler = random.choice(fillers)
        base_text = random.choice(tmpl["phrases"])
        text = f"{filler} {base_text}" if random.random() > 0.3 else base_text
        
        sample = {
            "id": f"scam_{i+1:03d}",
            "text": text,
            "is_scam": 1,
            "triggers": tmpl["triggers"],
            "scam_stage": tmpl["stage"],
            "scam_stage_name": tmpl["stage_name"]
        }
        samples.append(sample)
        
    return samples


def generate_legit_samples(count: int = 250) -> List[Dict[str, Any]]:
    """Generates synthetic legitimate Hinglish conversation samples."""
    legit_phrases = [
        "Hello sir, main HDFC Customer Care se bol raha hu. Main aapko batana chahta hu ki aapki credit card statement generation complete ho gayi hai.",
        "Aapka Amazon delivery agent gate par khada hai. Package receive karne ke liye OTP share kar sakte hain kya?",
        "Haan bhai, main office pahunch gaya hu. Shaam ko meeting ke baad project discussion finish karte hain.",
        "Mumma main abhi college lecture mein hu, shaam ko 6 baje ghar aaunga. Kuch grocery lana hai kya?",
        "Good morning doctor, mera appointment 4:00 PM par scheduled tha. Kya main 15 minute late aa sakta hu?",
        "Sir aapka Zomato order deliver ho gaya hai. Kindly app par delivery rate kar dijiye. Thank you!",
        "Hello ji, Airtel broadband bill pay karna hai. Kya new offer active hai 399 plan par?",
        "Aapka SBI Debit Card dispatch ho gaya hai speed post se. Tracking code SMS par send kar diya hai.",
        "Bhai kal ki train ticket confirm ho gayi hai. 8:30 PM par station pahunch jana time se.",
        "Sir mera Wi-Fi connection down hai subah se. Net check karke resolve kar do please.",
        "Ji main Swiggy driver bol raha hu, aapka exact flat number bata dijiye.",
        "Aapka electric bill Rs 1450 pay ho gaya hai successfully. Confirmation code: TXN98721.",
        "Arre dost, shaam ko cricket khelne chalna hai kya playground par?",
        "Hello, Flipkart customer support se call hai. Kya aapka issue status resolve hua?",
        "Aapki LIC policy ka premium due date 25th August hai. Online payment link available hai portal par.",
        "Sir aapka Ola cab booking confirmed hai. Main 2 minute mein pick-up point par pahunch raha hu.",
        "Bhai laptop repair ho gaya hai, dukaan se aakar collect kar lo.",
        "Hello Ji, main Reliance Digital se bol raha hu. Aapka TV installation aaj afternoon mein schedule hai."
    ]

    samples = []
    for i in range(count):
        text = random.choice(legit_phrases)
        if i > len(legit_phrases):
            var_prefix = random.choice(["Hi, ", "Hello, ", "Sunie, ", "Ji, ", ""])
            text = f"{var_prefix}{text}"
            
        sample = {
            "id": f"legit_{i+1:03d}",
            "text": text,
            "is_scam": 0,
            "triggers": {"authority": 0, "urgency": 0, "isolation": 0, "payment_pressure": 0},
            "scam_stage": 0,
            "scam_stage_name": "none"
        }
        samples.append(sample)

    return samples


def main():
    output_dir = "data"
    output_file = os.path.join(output_dir, "scam_dataset.json")

    os.makedirs(output_dir, exist_ok=True)

    scam_samples = generate_scam_samples(count=250)
    legit_samples = generate_legit_samples(count=250)

    dataset = scam_samples + legit_samples
    random.shuffle(dataset)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset successfully synthesized and saved to '{output_file}'")
    print(f"Total samples: {len(dataset)}")
    print(f"Scam samples: {len(scam_samples)}")
    print(f"Legitimate samples: {len(legit_samples)}")


if __name__ == "__main__":
    main()
