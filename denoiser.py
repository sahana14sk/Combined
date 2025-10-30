import os
import io
import threading
import tempfile
import requests
import numpy as np
import noisereduce as nr
from flask import Flask, request, send_file, jsonify
from pydub import AudioSegment
import streamlit as st

# ================================
# 🎧 1️⃣ Flask Backend
# ================================
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "🎧 Audio Denoising Backend is running!"})

@app.route('/denoise', methods=['POST'])
def denoise_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['file']
    sound = AudioSegment.from_file(file)
    samples = np.array(sound.get_array_of_samples())

    # Simple noise reduction
    reduced = nr.reduce_noise(y=samples, sr=sound.frame_rate)

    clean_audio = AudioSegment(
        reduced.tobytes(),
        frame_rate=sound.frame_rate,
        sample_width=sound.sample_width,
        channels=sound.channels
    )

    output = io.BytesIO()
    clean_audio.export(output, format="wav")
    output.seek(0)
    return send_file(output, mimetype="audio/wav")

# ================================
# 2️⃣ Run Flask in Background Thread
# ================================
def run_flask():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# ================================
# 🎨 3️⃣ Streamlit Frontend
# ================================
st.set_page_config(page_title="🎧 Audio Denoiser", page_icon="🎵")
st.title("🎶 Audio Denoiser")
st.write("Upload your noisy audio file and get a clean version!")

FLASK_API_URL = "http://127.0.0.1:5000/denoise"

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("Processing your audio... Please wait ⏳")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio:
            files = {"file": (os.path.basename(tmp_path), audio, "audio/wav")}
            response = requests.post(FLASK_API_URL, files=files)

        if response.status_code == 200:
            output_path = "denoised_output.wav"
            with open(output_path, "wb") as f:
                f.write(response.content)

            st.success("✅ Denoising complete!")
            st.audio(output_path, format="audio/wav")
            st.download_button("⬇️ Download Denoised Audio", data=open(output_path, "rb"), file_name="denoised.wav")
        else:
            st.error(f"❌ Error from server: {response.text}")

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")

    finally:
        os.remove(tmp_path)
