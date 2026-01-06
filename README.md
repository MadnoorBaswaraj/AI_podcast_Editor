
🎙️ AI-Powered Automated Podcast Editor

An end-to-end AI system that automates multi-camera podcast editing using speech recognition, speaker diarization, and transcript-driven video editing.
Designed to reduce manual editing time while producing broadcast-quality podcast videos.

🚀 Project Highlights
🎥 Dual-Camera Podcast Support
🧠 Speaker Diarization & Word-Level Transcription (WhisperX)
✂️ Transcript-Guided Video Editing
⚡ 90% Reduction in Editing Time
🌐 Flask-Based Web Application
🔧 Modular, Scalable Architecture

🧠 How It Works (Pipeline Overview)

Stage 1 – Speech Understanding & Transcription
a.Users upload raw video files from two cameras
b.Audio is extracted automatically

-WhisperX performs:
a. Automatic Speech Recognition (ASR)
b. Speaker Diarization
c.Word-level timestamp alignment
A time-aligned transcript file is generated

Stage 2 – Transcript-Driven Video Editing

a.Users upload the generated transcript
b.Transcript is analyzed for:
c.Speaker changes
d.Silence regions
e.Conversational flow
f.FFmpeg automatically:
g.Switches camera angles
h.Trims silence
i.Synchronizes video with speech

A final edited podcast video is produced

🏗️ System Architecture

User Upload (Dual Cameras)
        ↓
Audio Extraction
        ↓
WhisperX (ASR + Diarization + Alignment)
        ↓
Transcript Generation
        ↓
Transcript Analysis
        ↓
FFmpeg Video Editing
        ↓
Final Podcast Output

🛠️ Tech Stack
_____________________________________________________
| Category            | Tools                        |
|------------------- | ----------------------------- |
| Backend             | Flask                        |
| Speech Recognition  | WhisperX                     |
| Speaker Diarization | pyannote-audio               |
| Audio Processing    | Demucs                       |
| Video Editing       | FFmpeg                       |
| ML Models           | Whisper (large-v2), wav2vec2 |
| UI                  | HTML, CSS                    |
| Platform            | Windows / Linux              |
|Hardware             | CPU / GPU (CUDA supported)   |
|____________________________________________________|

📁 Project Structure

ai-podcast-editor/
│
├── app.py
├── stage1_transcription.py
├── stage2_video_editing.py
│
├── utils/
│   ├── audio_utils.py
│   ├── video_utils.py
│   └── transcript_parser.py
│
├── templates/
│   ├── index.html
│   └── upload.html
│
├── static/
│   └── styles.css
│
├── uploads/
│   ├── raw_videos/
│   ├── transcripts/
│   └── output/
│
└── README.md


⚙️ Installation & Setup
1️⃣ Clone the Repository
-git clone https://github.com/your-username/ai-podcast-editor.git
cd ai-podcast-editor

2️⃣ Install Python Dependencies
-pip install -r requirements.txt

3️⃣ Install FFmpeg
-sudo apt install ffmpeg

🔐 Speaker Diarization Setup
To enable diarization:
a.Create a Hugging Face access token
b.Accept:
pyannote/segmentation-3.0
pyannote/speaker-diarization-3.1

Set token:
export HF_TOKEN=your_huggingface_token

▶️ Running the Application

python app.py
Open in browser:

arduino
Copy code
http://localhost:5000

📊 Performance & Results

⏱️ 1-hour podcast edited in under 30 minutes
📉 90% reduction in manual editing
🎯 Accurate word-level timestamps
👥 Effective multi-speaker handling

⚠️ Limitations

a.Overlapping speech can reduce diarization accuracy
b.Audio quality affects speaker separation
c.GPU recommended for large models

🔮 Future Enhancements

🎬 Emotion-based camera switching
🧠 LLM-based content summarization
📌 Automatic highlights & chapters
☁️ Cloud deployment (AWS / GCP)
🎙️ Real-time podcast editing

👨‍💻 Author

Madnoor Baswaraj
B.Tech – Artificial Intelligence & Data Science
GitHub: https://github.com/MadnoorBaswaraj
LinkedIn: https://linkedin.com/in/madnoor-baswaraj-85b28028a
Email: madnoorbaswaraj@gmail.com

📜 License
This project is licensed under the MIT License.

⭐ If you find this project useful, please star the repository!
