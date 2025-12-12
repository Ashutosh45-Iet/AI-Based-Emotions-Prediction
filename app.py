import streamlit as st
import pickle

st.set_page_config(page_title="AI-Based Emotion Prediction", page_icon="🎭", layout="centered")

st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: 800;
            background: linear-gradient(90deg, #6366F1, #EC4899);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 4px;
        }
        .subtitle {
            text-align: center;
            font-size: 15px;
            color: #7C3AED;
            font-weight: 500;
            margin-top: -6px;
            opacity: 0.85;
            margin-bottom: 20px;
        }
        .stTextArea > div > div > textarea {
            min-height: 150px;
            font-size: 15px;
            border-radius: 12px !important;
            border: 1.5px solid #A78BFA !important;
            box-shadow: 0 0 8px rgba(124,58,237,0.15);
        }
        .prediction-box {
            padding: 20px;
            border-radius: 16px;
            background: linear-gradient(135deg, #ffffff 0%, #f4f1ff 100%);
            border: 1.5px solid #C4B5FD;
            box-shadow: 0 8px 24px rgba(109,40,217,0.12);
            display: flex;
            align-items: center;
            gap: 16px;
            margin-top: 20px;
            animation: fadeIn 0.4s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0px); }
        }
        .accent-bar {
            width: 10px;
            height: 70px;
            border-radius: 10px;
            flex: 0 0 10px;
        }
        .emoji {
            font-size: 26px;
        }
        .emotion-text {
            font-size: 22px;
            font-weight: 700;
        }
        .emotion-sub {
            font-size: 13px;
            color: #6B7280;
            margin-top: 4px;
        }
        .stButton>button {
            padding: 10px 18px;
            border-radius: 12px;
            background: linear-gradient(90deg, #8B5CF6, #EC4899);
            color: white;
            border: none;
            font-size: 16px;
            font-weight: 600;
            transition: 0.25s;
            box-shadow: 0 4px 12px rgba(147,51,234,0.3);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(147,51,234,0.45);
            opacity: 0.95;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎭 AI-Based Emotion Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotional tone instantly with advanced NLP models</div>', unsafe_allow_html=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    tfidf = load_pickle("tfidf_vectorizer.pkl")
    model = load_pickle("tfidf_svc_model.pkl")
    st.success("Model Loaded Successfully ✔", icon="✅")
except Exception:
    st.error("❌ Pickle Files Not Found!")
    st.stop()

st.write("### ✍️ Enter text to analyze emotion:")
user_text = st.text_area("", placeholder="Type something like 'I feel amazing today!'...")

label_map = {
    4: "sadness",
    0: "anger",
    3: "love",
    5: "surprise",
    1: "fear",
    2: "joy"
}

emotion_emojis = {
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "surprise": "😮",
    "fear": "😨",
    "joy": "😊"
}

emotion_colors = {
    "sadness": "#6B7280",
    "anger": "#DC2626",
    "love": "#BE185D",
    "surprise": "#D97706",
    "fear": "#374151",
    "joy": "#10B981"
}

if st.button("🔍 Predict Emotion"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        try:
            vector = tfidf.transform([user_text])
            prediction_raw = model.predict(vector)[0]
            predicted_emotion = label_map.get(int(prediction_raw), "Unknown")
            emoji_icon = emotion_emojis.get(predicted_emotion, "🤖")
            color = emotion_colors.get(predicted_emotion, "#6366F1")

            result_html = f"""
                <div class='prediction-box'>
                    <div class='accent-bar' style="background:{color}"></div>
                    <div style="display:flex;flex-direction:column;">
                        <div style="display:flex;align-items:center;gap:12px;">
                            <div class='emoji'>{emoji_icon}</div>
                            <div>
                                <div class='emotion-text' style="color:{color};">
                                    {predicted_emotion.capitalize()}
                                </div>
                                <div class='emotion-sub'>
                                    Model Label: {int(prediction_raw)}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top:8px;font-size:14px;color:#374151;">
                            Input: <span style="color:#111827;font-weight:600">
                                {user_text if len(user_text) <= 120 else user_text[:117] + '...'}
                            </span>
                        </div>
                    </div>
                </div>
            """

            st.markdown(result_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
