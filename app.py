from flask import Flask, request, jsonify
import random
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from pyngrok import ngrok, conf

# === Inisialisasi Flask App ===
app = Flask(__name__)

# === Load Resources ===
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pickle", "rb") as f:
    lbl_encoder = pickle.load(f)
with open("responses.pkl", "rb") as f:
    responses = pickle.load(f)

model = load_model("chatbot_model.h5")
max_len = 20

# === Global States ===
fail_count = 0
MAX_FAILS = 3
context = None
current_topic = None
user_name = ""
last_tag = None

# === Tips & Decline (ISI dengan lengkap di file asli kamu) ===
tips_responses = {
    "stress_due_to_academic": ["Contoh tips akademik..."]
}

decline_responses = {
    "stress_due_to_academic": ["Contoh decline akademik..."]
}

universal_intents = {"get_support_professional"}

allowed_general_intents = {
    "stress_general", "anxiety_general", "self_worth_general",
    "heartbreak_general", "loneliness_general", "grief_general", "depression_general"
}

# === Fungsi Bantuan ===
def extract_name(text):
    text = text.lower()
    patterns = [
        r"namaku\s+([a-zA-Z]+)", r"nama\s+saya\s+([a-zA-Z]+)", r"nama\s+aku\s+([a-zA-Z]+)",
        r"saya\s+([a-zA-Z]+)", r"aku\s+([a-zA-Z]+)", r"panggil\s+aku\s+([a-zA-Z]+)", r"([a-zA-Z]+)$"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).capitalize()
    return "Teman"

def get_specific_tip(tag):
    if tag in tips_responses:
        return random.choice(tips_responses[tag]).replace("{user_name}", user_name or "kamu")
    return None

def get_specific_decline(tag):
    if tag in decline_responses:
        return random.choice(decline_responses[tag]).replace("{user_name}", user_name or "kamu")
    return None

def reset_chat():
    global context, user_name, fail_count, current_topic, last_tag
    context = None
    user_name = ""
    fail_count = 0
    current_topic = None
    last_tag = None
    return "🔄 Obrolan telah direset.😊"

# === Fungsi Utama ===
def chatbot_response(user_input):
    global context, user_name, fail_count, current_topic, last_tag

    user_input = user_input.strip().lower()

    if user_input in ["reset", "ulang", "mulai lagi", "ulang yuk"]:
        return reset_chat()

    if user_input in ["makasih", "terima kasih", "thanks", "thank you"]:
        return f"Sama-sama ya, {user_name or 'teman'} 🌷"

    if context is None:
        context = "awaiting_name"
        return "Hai! Boleh aku tahu siapa namamu?"

    if context == "awaiting_name":
        user_name = extract_name(user_input)
        context = "awaiting_feeling"
        return f"Hai {user_name}! Gimana perasaanmu hari ini?"

    if context == "awaiting_tip_permission":
        if user_input in ["iya", "ya", "mau", "boleh", "lanjut", "oke"]:
            context = "conversation_end"
            tip_response = get_specific_tip(last_tag)
            if tip_response:
                return tip_response
            return "🙏 Maaf, aku belum punya tips khusus buat itu. Tapi kamu nggak sendiri ya 🤍"

        elif user_input in ["tidak", "ga", "gak", "nggak", "skip", "gausah", "enggak"]:
            context = "conversation_end"
            decline_response = get_specific_decline(last_tag)
            if decline_response:
                return decline_response
            return "Aku ngerti kok, kadang kita cuma butuh didengarkan tanpa solusi dulu 🤍"

        return "Kamu ingin aku bantu kasih tips? (Ya/Tidak)"

    filler_words = {'nih', 'deh', 'dong', 'tuh', 'loh', 'ya', 'kok', 'sih', 'ah', 'eh', 'kan'}
    user_input = ' '.join([word for word in user_input.split() if word not in filler_words])

    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, truncating="post", maxlen=max_len)
    pred = model.predict(padded, verbose=0)[0]
    confidence = float(np.max(pred))
    tag = lbl_encoder.inverse_transform([np.argmax(pred)])[0]

    print(f"[DEBUG] Tag: {tag} | Confidence: {confidence:.2f} | Context: {context}")

    if tag in universal_intents:
        return responses.get(tag, ["Kamu bisa cari bantuan profesional ya, {user_name}"])[0].replace("{user_name}", user_name or "kamu")

    if context == "awaiting_feeling" and tag in allowed_general_intents:
        current_topic = tag.split("_")[0]
        context = "awaiting_reason"
        return random.choice(responses.get(tag, ["Ceritain ya, aku siap dengerin."])).replace("{user_name}", user_name or "kamu")

    if context == "awaiting_reason" and current_topic and tag.startswith(current_topic):
        context = "awaiting_tip_permission"
        last_tag = tag
        base_response = responses.get(tag, ["Aku denger kok."])[0].replace("{user_name}", user_name or "kamu")
        return f"{base_response}\n\nMau aku bantu kasih tips untuk itu, {user_name}?"

    if confidence < 0.2:
        fail_count += 1
        if fail_count >= MAX_FAILS:
            return reset_chat()
        return random.choice([
            "Hmm... aku belum yakin maksudmu 😥 Bisa dijelaskan dengan cara lain?",
            "Aku agak bingung nangkap maksud kamu. Bisa diperjelas?",
            "Maaf, aku belum paham betul. Bisa diulangi dengan kalimat yang berbeda?"
        ])

    response_pool = responses.get(tag, ["Maaf, aku belum punya jawaban untuk itu."])
    response = random.choice(response_pool[0] if isinstance(response_pool[0], list) else response_pool)
    fail_count = 0
    return response.replace("{user_name}", user_name or "kamu")

# === Endpoint Flask ===
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Pesan tidak boleh kosong"}), 400
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})

# === Jalankan Server ===
if __name__ == "__main__":
    # Autentikasi ngrok
    conf.get_default().auth_token = "2xzvR1EEpOMyLmoqVDTB9aE9C61_2QBc7HqmQts6HCpthGzGF"

    # Buka koneksi ke port Flask
    port = 5000
    public_url = ngrok.connect(port)
    print(f"💬 Chatbot online di: {public_url}")

    # Jalankan Flask server
    app.run(port=port)
