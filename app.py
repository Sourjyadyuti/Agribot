import streamlit as st
import os
import json
import re
import string
import random
import numpy as np
from PIL import Image
import google.generativeai as genai
import tensorflow as tf
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# =========================
# CONFIG
# =========================
# API key comes from Streamlit secrets (Streamlit Cloud: Settings -> Secrets;
# locally: .streamlit/secrets.toml, which should be in .gitignore and never
# committed). Falls back to an env var for other hosts (e.g. HF Spaces).
API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.environ.get("OPENWEATHER_API_KEY", ""))

# Paths are relative to the repo root so this works after a `git clone`.
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH      = os.path.join(BASE_DIR, "data", "agribot_unified_dataset.json")
MODEL_PATH        = os.path.join(BASE_DIR, "models", "plant_disease_model.keras")
CLASS_NAMES_PATH  = os.path.join(BASE_DIR, "models", "class_names.json")
MODEL_NAME    = "gemini-flash-latest"
MAX_HISTORY   = 5
IMG_SIZE      = (224, 224)

st.set_page_config(page_title="AgriBot", page_icon="🌾", layout="centered")

# Small CSS tweaks so things breathe a bit more on narrow (phone) screens
# without hurting the desktop layout.
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #d1f2d1; }
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 2rem; }
    h1 { font-size: 1.6rem !important; }
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriBot — Agriculture Chatbot")
st.caption("For Students and Farmers of Assam")

# =========================
# LOADERS (cached so they only run once per session)
# =========================
@st.cache_resource
def load_chat_resources():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    qa     = data.get("qa", [])
    mcq    = data.get("mcq", [])
    topics = data.get("topics", [])
    entries = []
    for item in qa:
        text = item["question"] + " " + " ".join(item.get("keywords", []))
        entries.append({"type": "qa", "data": item, "text": text})
    for item in mcq:
        entries.append({"type": "mcq", "data": item, "text": item["question"]})
    for item in topics:
        entries.append({"type": "topic", "data": item, "text": item["topic"] + " " + item["summary"]})
    texts      = [e["text"] for e in entries]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix     = vectorizer.fit_transform(texts)
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    return qa, mcq, topics, vectorizer, matrix, entries, model


@st.cache_resource
def load_disease_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


# =========================
# CHATBOT HELPERS (unchanged from your original)
# =========================
HINGLISH = {
    "kya hai": "what is", "kya hote hain": "what are",
    "kaise hota": "how does", "ke baare mein": "about",
    "kaise": "how", "kyun": "why", "kab": "when",
    "batao": "explain", "abt": "about", "diff": "difference",
    "nd": "and", "b/w": "between",
}
TYPOS = {
    "irrigaton": "irrigation", "fertilzer": "fertilizer",
    "khariff": "kharif", "soill": "soil", "seads": "seeds",
}
Q_WORDS = {"what", "why", "how", "when", "where", "which",
           "explain", "define", "list", "give", "tell", "show"}


def preprocess(raw):
    text = raw.strip().lower()
    for k, v in sorted(HINGLISH.items(), key=lambda x: -len(x[0])):
        if " " in k and k in text:
            text = text.replace(k, v)
    words = text.split()
    fixed = []
    for w in words:
        c = w.strip(string.punctuation)
        if c in HINGLISH and " " not in HINGLISH[c]:
            fixed.append(HINGLISH[c])
        elif c in TYPOS:
            fixed.append(TYPOS[c])
        else:
            fixed.append(w)
    text = " ".join(fixed)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not any(w in text.split() for w in Q_WORDS) and len(text.split()) <= 3:
        text = "what is " + text
    return text


def search(query, vectorizer, matrix, entries, top_k=3):
    clean  = preprocess(query)
    vec    = vectorizer.transform([clean])
    scores = cosine_similarity(vec, matrix)[0]
    best   = {}
    for i, (entry, score) in enumerate(zip(entries, scores)):
        key = (entry["type"], entry["data"].get("id", i))
        if key not in best or score > best[key]["score"]:
            best[key] = {"score": float(score), "type": entry["type"], "data": entry["data"]}
    results = sorted(best.values(), key=lambda x: -x["score"])
    return [r for r in results if r["score"] > 0.05][:top_k], clean


def build_context(results):
    if not results:
        return "No specific context found."
    parts = []
    for r in results:
        d = r["data"]
        if r["type"] == "qa":
            parts.append(f"Q: {d['question']}\nA: {d['answer']}\nSource: {d['source']}")
        elif r["type"] == "topic":
            parts.append(f"Topic: {d['topic']}\n{d['content'][:300]}")
    return "\n\n---\n\n".join(parts)


def ask_gemini(model, question, context, history):
    history_text = ""
    for turn in history[-MAX_HISTORY:]:
        history_text += f"Student: {turn['user']}\nAgriBot: {turn['bot']}\n\n"
    prompt = f"""You are AgriBot, a helpful agriculture chatbot for students and farmers in Assam, India.
Use the dataset context below to answer. Be clear, educational and friendly.

Understand Hinglish and broken grammar.

--- DATASET CONTEXT ---
{context}
-----------------------
{history_text}Student: {question}
AgriBot:"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


# =========================
# DISEASE DETECTION HELPERS
# =========================
def format_label(raw_label):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    if "___" in raw_label:
        crop, disease = raw_label.split("___", 1)
    else:
        crop, disease = raw_label, ""
    crop = crop.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    return crop, disease


def predict_disease(model, class_names, pil_image, top_k=3):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)          # 0-255 range on purpose:
    arr = np.expand_dims(arr, axis=0)               # the model has its own
    preds = model.predict(arr, verbose=0)[0]        # Rescaling/Normalization
    top_idx = preds.argsort()[-top_k:][::-1]        # layers built in.
    return [(class_names[i], float(preds[i])) for i in top_idx]


def ask_gemini_for_treatment(model, crop, disease, history):
    if "healthy" in disease.lower():
        prompt = f"""You are AgriBot, an agriculture assistant for farmers in Assam, India.
A farmer scanned a {crop} leaf and it looks healthy (no disease detected).
Give 2-3 short, practical tips to keep the {crop} plant healthy. Be clear and friendly."""
    else:
        prompt = f"""You are AgriBot, an agriculture assistant for farmers in Assam, India.
A farmer scanned a {crop} plant leaf and our model detected: "{disease}".
Explain in simple, friendly language:
1. What this disease is and how to recognise it
2. What causes it / how it spreads
3. Practical treatment steps (organic and chemical options if relevant)
4. How to prevent it next season
Keep it concise and actionable for a farmer, understandable even with basic education."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


# =========================
# WEATHER HELPERS
# =========================
def get_current_weather(city):
    """Fetch current weather for a city using OpenWeatherMap. Returns a dict or None."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "city": data.get("name", city),
            "temp": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"].title(),
            "wind_speed": data["wind"]["speed"],
            "rain_1h": data.get("rain", {}).get("1h", 0),
        }
    except Exception:
        return None


def get_forecast(city, days=5):
    """Fetch a simple multi-day forecast (one reading per day, midday) using OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        daily = {}
        for entry in data.get("list", []):
            date = entry["dt_txt"].split(" ")[0]
            hour = entry["dt_txt"].split(" ")[1]
            if hour == "12:00:00" and date not in daily:
                daily[date] = {
                    "date": date,
                    "temp": entry["main"]["temp"],
                    "condition": entry["weather"][0]["description"].title(),
                    "humidity": entry["main"]["humidity"],
                    "rain_prob": entry.get("pop", 0) * 100,
                }
        return list(daily.values())[:days]
    except Exception:
        return []


def ask_gemini_for_farming_advice(model, weather):
    prompt = f"""You are AgriBot, an agriculture assistant for farmers in Assam, India.
Current weather conditions:
- Temperature: {weather['temp']}°C (feels like {weather['feels_like']}°C)
- Condition: {weather['condition']}
- Humidity: {weather['humidity']}%
- Wind speed: {weather['wind_speed']} m/s
- Rain (last hour): {weather['rain_1h']} mm

Based on these conditions, give a farmer 2-3 short, practical tips for today: e.g. whether to
irrigate, any disease/pest risk from this humidity/temperature combination, and any precautions
needed. Keep it concise, friendly, and actionable."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "quiz_q" not in st.session_state:
    st.session_state.quiz_q = None
if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False
if "score" not in st.session_state:
    st.session_state.score = {"correct": 0, "total": 0}

try:
    qa, mcq, topics, vectorizer, matrix, entries, gemini_model = load_chat_resources()
    chat_loaded = True
except Exception as e:
    st.error(f"Failed to load chatbot resources: {e}")
    chat_loaded = False

try:
    disease_model, class_names = load_disease_model()
    disease_loaded = True
except Exception as e:
    st.error(f"Failed to load disease detection model: {e}")
    disease_loaded = False

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🌿 Navigation")
    mode = st.radio("Select mode", ["💬 Chat", "🔍 Disease Detection", "🌦️ Weather", "📝 Quiz", "ℹ️ About"])
    
    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()
    st.markdown("### 💡 Sample questions")
    samples = ["What is loamy soil?", "kharif crop kya hai",
               "drip irrigation benefits", "NPK fertilizer", "rice blast disease"]
    for s in samples:
        if st.button(s, key=f"s_{s}"):
            st.session_state["prefill"] = s

# =========================
# MODE: CHAT
# =========================
if "Chat" in mode:
    st.markdown("### 💬 Chat with AgriBot")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prefill    = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask your agriculture question...") or prefill
    if user_input and chat_loaded:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            results, clean = search(user_input, vectorizer, matrix, entries)
            context        = build_context(results)
            answer         = ask_gemini(gemini_model, user_input, context, st.session_state.history)
        sources = list(set(r["data"].get("source", "") for r in results if r["data"].get("source")))
        full    = answer + (f"\n\n📚 *Source: {', '.join(sources)}*" if sources else "")
        with st.chat_message("assistant"):
            st.markdown(full)
        st.session_state.messages.append({"role": "user",      "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": full})
        st.session_state.history.append({"user": user_input,   "bot": answer})

# =========================
# MODE: DISEASE DETECTION
# =========================
elif "Disease Detection" in mode:
    st.markdown("### 🔍 Plant Disease Detection")
    st.caption("Upload a photo of a leaf and AgriBot will identify the disease and suggest treatment.")

    if not disease_loaded:
        st.error("Disease detection model isn't loaded. Check MODEL_PATH and CLASS_NAMES_PATH.")
    else:
        uploaded_file = st.file_uploader(
            "Upload a leaf image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col_img, col_result = st.columns([1, 1.3])
            with col_img:
                st.image(image, caption="Uploaded leaf", use_container_width=True)

            with st.spinner("Analysing leaf..."):
                top_preds = predict_disease(disease_model, class_names, image, top_k=3)

            best_label, best_conf = top_preds[0]
            crop, disease = format_label(best_label)

            with col_result:
                if "healthy" in disease.lower():
                    st.success(f"✅ {crop}: looks healthy ({best_conf*100:.1f}% confidence)")
                else:
                    st.error(f"⚠️ {crop}: {disease} ({best_conf*100:.1f}% confidence)")

                with st.expander("See other possibilities"):
                    for label, conf in top_preds[1:]:
                        c, d = format_label(label)
                        st.write(f"- {c}: {d or 'healthy'} — {conf*100:.1f}%")

            if chat_loaded:
                with st.spinner("Getting advice from AgriBot..."):
                    advice = ask_gemini_for_treatment(
                        gemini_model, crop, disease, st.session_state.history
                    )
                st.markdown("#### 🩺 AgriBot's advice")
                st.markdown(advice)
            else:
                st.info("Chat model isn't loaded, so detailed treatment advice isn't available. "
                         "The detected label above is still valid.")

# =========================
# MODE: WEATHER
# =========================
elif "Weather" in mode:
    st.markdown("### 🌦️ Weather for Farming")
    st.caption("Check current conditions and get AgriBot's farming advice based on the weather.")

    if not WEATHER_API_KEY:
        st.error("Weather API key isn't set. Add OPENWEATHER_API_KEY in Streamlit secrets.")
    else:
        city = st.text_input("Enter your location (city/town)", value="Silchar")

        if st.button("Get Weather", type="primary") or "weather_data" in st.session_state:
            if st.session_state.get("weather_city") != city:
                st.session_state.pop("weather_data", None)

            if "weather_data" not in st.session_state:
                with st.spinner(f"Fetching weather for {city}..."):
                    weather = get_current_weather(city)
                st.session_state.weather_data = weather
                st.session_state.weather_city = city

            weather = st.session_state.get("weather_data")

            if weather is None:
                st.error(f"Couldn't find weather for '{city}'. Check the spelling and try again.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🌡️ Temperature", f"{weather['temp']}°C")
                c2.metric("🤒 Feels Like",  f"{weather['feels_like']}°C")
                c3.metric("💧 Humidity",    f"{weather['humidity']}%")
                c4.metric("💨 Wind",        f"{weather['wind_speed']} m/s")
                st.info(f"**Condition:** {weather['condition']}")

                if chat_loaded:
                    with st.spinner("Getting farming advice..."):
                        advice = ask_gemini_for_farming_advice(gemini_model, weather)
                    st.markdown("#### 🌾 AgriBot's advice for today")
                    st.markdown(advice)

                st.markdown("---")
                st.markdown("#### 📅 5-Day Forecast")
                with st.spinner("Loading forecast..."):
                    forecast = get_forecast(city)
                if forecast:
                    cols = st.columns(len(forecast))
                    for col, day in zip(cols, forecast):
                        with col:
                            st.markdown(f"**{day['date'][5:]}**")
                            st.markdown(f"{day['temp']}°C")
                            st.caption(day['condition'])
                            st.caption(f"🌧️ {day['rain_prob']:.0f}%")
                else:
                    st.caption("Forecast unavailable right now.")

# =========================
# MODE: QUIZ
# =========================
elif "Quiz" in mode:
    st.markdown("### 📝 Quiz Mode")
    sc = st.session_state.score
    c1, c2, c3 = st.columns(3)
    c1.metric("Correct", sc["correct"])
    c2.metric("Total",   sc["total"])
    c3.metric("Score",
        str(int(sc["correct"] / sc["total"] * 100)) + "%"
        if sc["total"] > 0 else "0%")

    st.markdown("---")

    if chat_loaded:
        if st.session_state.quiz_q is None:
            if st.button("Next Question", type="primary"):
                st.session_state.quiz_q       = random.choice(mcq)
                st.session_state.quiz_answered = False
                st.session_state.quiz_result   = None
                st.rerun()

        if st.session_state.quiz_q:
            q    = st.session_state.quiz_q
            opts = [k + ". " + v for k, v in q["options"].items()]

            st.markdown("**Q: " + q["question"] + "**")

            if not st.session_state.quiz_answered:
                sel = st.radio(
                    "Choose your answer:",
                    opts,
                    key="q_" + str(q["id"]),
                    index=None
                )

                if st.button("Submit Answer", type="primary"):
                    if sel is None:
                        st.warning("Please select an answer first!")
                    else:
                        chosen = sel[0]
                        st.session_state.score["total"] += 1
                        st.session_state.quiz_answered   = True
                        if chosen == q["correct_answer"]:
                            st.session_state.score["correct"] += 1
                            st.session_state.quiz_result = "correct"
                        else:
                            st.session_state.quiz_result = chosen
                        st.rerun()

            if st.session_state.quiz_answered:
                result = st.session_state.get("quiz_result")

                if result == "correct":
                    st.success("Correct! Well done!")
                else:
                    correct_option = q["options"][q["correct_answer"]]
                    correct_text   = q["correct_answer"] + ". " + correct_option
                    st.error("Wrong! You selected: " + str(result))
                    st.success("Correct answer: " + correct_text)

                st.info("Explanation: " + q["explanation"])
                st.caption("Source: " + q["source"])
                st.markdown("---")

                if st.button("Next Question", type="primary"):
                    st.session_state.quiz_q       = random.choice(mcq)
                    st.session_state.quiz_answered = False
                    st.session_state.quiz_result   = None
                    st.rerun()

# =========================
# MODE: ABOUT
# =========================
elif "About" in mode:
    st.markdown("### ℹ️ About AgriBot")
    st.markdown("""
    **AgriBot** is an AI agriculture chatbot for students and farmers in Assam.
    - 💬 Chat mode — Q&A in English and Hinglish
    - 🔍 Disease Detection — upload a leaf photo for instant diagnosis + treatment advice
    - 📝 Quiz mode — MCQ exam preparation
    - 🌾 Knowledge base: NCERT · ICAR · TNAU
    - 🤖 Powered by Google Gemini + a custom EfficientNet plant disease model
    """)
    if chat_loaded:
        c1, c2, c3 = st.columns(3)
        c1.metric("Q&A Pairs", len(qa))
        c2.metric("MCQ",       len(mcq))
        c3.metric("Topics",    len(topics))
