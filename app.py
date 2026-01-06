import os
from pdb import run
# Set FFmpeg path for whisperx
os.environ["FFMPEG_PATH"] = r"C:\ffmpeg\bin\ffmpeg.exe"  # Adjust to your FFmpeg location
import os

print("PATH:", os.environ.get("PATH"))
import traceback
print(traceback.format_exc())
import logging
import re
import tempfile
import subprocess
print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout)
from datetime import timedelta
from flask import Flask, request, send_from_directory, redirect
import torch
import whisperx
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("whisperx_flask_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# STORAGE ON E: (change here if needed)
# =========================
BASE_DIR = os.environ.get("WHISPERX_BASE", r"E:\stud\STUD")

UPLOAD_STAGE1 = os.path.join(BASE_DIR, "uploads", "stage1")   # best-audio single video
UPLOAD_STAGE2 = os.path.join(BASE_DIR, "uploads", "stage2")   # audio for transcription
UPLOAD_STAGE3 = os.path.join(BASE_DIR, "uploads", "stage3")   # two videos for clip building

OUTPUT_AUDIO = os.path.join(BASE_DIR, "output", "audio")              # extracted wavs
OUTPUT_TRANSCRIPTS = os.path.join(BASE_DIR, "output", "transcripts")  # .srt/.txt
FINAL_OUTPUTS = os.path.join(BASE_DIR, "output","final_video")                      # final merged video(s)
TMP_DIR = os.path.join(BASE_DIR, "tmp")                                # temp space on E:

for d in [UPLOAD_STAGE1, UPLOAD_STAGE2, UPLOAD_STAGE3,
          OUTPUT_AUDIO, OUTPUT_TRANSCRIPTS, FINAL_OUTPUTS, TMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Force all libs to use E:\stud\STUD\whisperx_flask_app\tmp
tempfile.tempdir = TMP_DIR
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR
os.environ["FFMPEG_TMPDIR"] = TMP_DIR   # helps ffmpeg temp on E:

# =========================
# STYLES
# =========================
BASE_STYLE = """
<style>
:root { color-scheme: dark light; }
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  margin: 0; padding: 2rem; background: #0f172a; color: #e2e8f0;
}
a { color: #60a5fa; text-decoration: none; }
a:hover { text-decoration: underline; }
.container { max-width: 980px; margin: 0 auto; }
.card {
  background: #111827; border: 1px solid #1f2937; border-radius: 16px;
  padding: 1.25rem; margin: 1rem 0; box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
h1,h2,h3 { margin: .25rem 0 1rem 0; }
hr { border: none; border-top: 1px solid #1f2937; margin: 1.25rem 0; }
button, .btn {
  display: inline-block; background: #2563eb; color: white; border: none;
  padding: .65rem 1rem; border-radius: 12px; cursor: pointer; font-weight: 600;
  box-shadow: 0 6px 18px rgba(37,99,235,.35);
}
button:hover, .btn:hover { background: #1d4ed8; }
input[type="file"], select {
  width: 100%; padding: .6rem .75rem; border-radius: 12px; border: 1px solid #334155; background: #0b1220; color:#e2e8f0;
}
label { display:block; margin:.5rem 0 .25rem; font-weight:600; color:#cbd5e1; }
code, .mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  background:#0b1220; border:1px solid #1f2937; padding:.15rem .35rem; border-radius:8px; color:#cbd5e1;
}
.small { font-size: .9rem; color:#94a3b8; }
.grid { display:grid; gap:1rem; }
.grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.notice { background:#0b1220; border-left:4px solid #60a5fa; padding:.75rem 1rem; border-radius:8px; }
footer { margin-top: 2rem; color:#94a3b8; font-size:.9rem; }
</style>
"""

def wrap_page(title: str, body_html: str) -> str:
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{title}</title>
      {BASE_STYLE}
    </head>
    <body>
      <div class="container">
        <h1>{title}</h1>
        {body_html}
        <footer><hr><a href="/">Home</a></footer>
      </div>
    </body>
    </html>
    """

# =========================
# FLASK
# =========================
app = Flask(__name__)


# GPU/CPU Detection and Optimal Settings
import os
import torch
import whisperx


try:
    if not torch.cuda.is_available():
        logger.error("CUDA GPU is required but not found. Please enable your NVIDIA GPU.")
        raise RuntimeError("CUDA GPU is required but not found. Please enable your NVIDIA GPU.")
except Exception as e:
    logger.exception("Error during CUDA check")
    raise

import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL_SIZE = "medium"   # good fit for RTX 3050 6GB

torch.set_grad_enabled(False)
logger.info(f"Loading WhisperX ({MODEL_SIZE}) on {DEVICE} [{COMPUTE_TYPE}] ...")

# ------------------------
# 1. Load WhisperX ASR
# ------------------------
try:
    asr_model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("WhisperX model loaded successfully.")
except Exception as e:
    logger.exception("Error loading WhisperX model")
    logger.warning("Falling back to CPU...")
    asr_model = whisperx.load_model(MODEL_SIZE, device="cpu", compute_type="float32")

# ------------------------
# 2. Load diarization pipeline (NO HF TOKEN)
# ------------------------
logger.info("Loading diarization pipeline ...")

try:
    from whisperx.diarize import DiarizationPipeline
    
    # No HF token → just pass device
    diarize_pipeline = DiarizationPipeline(device=DEVICE)
    logger.info("Diarization pipeline loaded successfully.")
except Exception as e:
    logger.exception("Error loading diarization pipeline")
    diarize_pipeline = None

ALLOWED_VIDEO = {"mp4", "mov", "mkv", "avi", "m4v"}
ALLOWED_AUDIO = {"wav", "mp3", "m4a", "aac", "flac"}

# =========================
# UTILITIES
# =========================
def _allowed(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

def extract_audio_to_wav(video_path: str, wav_path: str):
    try:
        logger.info(f"Extracting audio from {video_path} to {wav_path}")
        ffmpeg_candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            "ffmpeg",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            os.environ.get("FFMPEG_PATH", "ffmpeg")
        ]
        ffmpeg_path = None
        for candidate in ffmpeg_candidates:
            try:
                subprocess.run([candidate, "-version"], capture_output=True, check=True)
                ffmpeg_path = candidate
                break
            except Exception as e:
                logger.warning(f"FFmpeg {candidate} not found or failed: {e}")
        if not ffmpeg_path:
            logger.error("FFmpeg executable not found.")
            raise FileNotFoundError("FFmpeg executable not found.")
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", "-af", "volume=1.0", wav_path
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Audio extracted successfully to {wav_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise
    except Exception as e:
        logger.exception(f"extract_audio_to_wav failed for {video_path}")
        raise

    
def srt_time_to_seconds(t: str) -> float:
    """
    Convert SRT 'HH:MM:SS,mmm' to seconds (float).
    Enhanced parsing with error handling.
    """
    try:
        # Handle both comma and dot decimal separators
        t = t.replace(',', '.')
        parts = t.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid time format: {t}")
        
        h, m, s_ms = parts
        h, m = int(h), int(m)
        s = float(s_ms)
        
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError) as e:
        raise ValueError(f"Cannot parse time '{t}': {e}")

def parse_diarized_srt(srt_path: str):
    """
    Parse an SRT where each caption line contains '[SPEAKER_XX]: text'
    Return: list of dicts with {start, end, speaker, text}
    Enhanced parsing with better error handling.
    """
    segments = []
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise RuntimeError(f"Cannot read SRT file {srt_path}: {e}")

    # Split by double newlines to separate subtitle blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    
    for block_idx, b in enumerate(blocks):
        lines = b.strip().splitlines()
        if len(lines) < 3:  # Need at least: number, timing, text
            continue
            
        try:
            # Skip the subtitle number (line 0)
            timing = lines[1]
            text = " ".join(lines[2:]).strip()

            # Parse timing line
            timing_match = re.match(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", timing)
            if not timing_match:
                print(f"Warning: Invalid timing format in block {block_idx}: {timing}")
                continue
                
            start_s = srt_time_to_seconds(timing_match.group(1))
            end_s = srt_time_to_seconds(timing_match.group(2))

            # Extract speaker and text
            spk_match = re.match(r"\[([^\]]+)\]\s*:\s*(.*)", text)
            if spk_match:
                speaker = spk_match.group(1).strip()
                text_clean = spk_match.group(2).strip()
            else:
                speaker = "Unknown"
                text_clean = text.strip()

            if start_s < end_s and text_clean:  # Valid segment
                segments.append({
                    "start": start_s, 
                    "end": end_s, 
                    "speaker": speaker, 
                    "text": text_clean
                })
        except Exception as e:
            print(f"Warning: Error parsing block {block_idx}: {e}")
            continue
            
    print(f"Parsed {len(segments)} valid segments from SRT")
    return segments

def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp format"""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_transcripts(segments, base_name: str):
    """Save transcripts in both TXT and SRT formats"""
    txt_path = os.path.join(OUTPUT_TRANSCRIPTS, f"{base_name}.txt")
    srt_path = os.path.join(OUTPUT_TRANSCRIPTS, f"{base_name}.srt")

    try:
        with open(txt_path, "w", encoding="utf-8") as f_txt, open(srt_path, "w", encoding="utf-8") as f_srt:
            for i, seg in enumerate(segments, start=1):
                st, et, spk, text = seg["start"], seg["end"], seg.get("speaker", "Unknown"), seg["text"].strip()
                
                # TXT format
                f_txt.write(f"[{spk}] {st:.2f} --> {et:.2f} {text}\n")
                
                # SRT format
                f_srt.write(f"{i}\n{format_srt_time(st)} --> {format_srt_time(et)}\n[{spk}]: {text}\n\n")
        logger.info(f"Transcripts saved: {txt_path}, {srt_path}")
        
        print(f"Transcripts saved: {txt_path}, {srt_path}")
        return srt_path, txt_path
    except Exception as e:
         logger.exception(f"Failed to save transcripts: {e}")
         raise
def merge_contiguous_same_speaker(segments, gap_tolerance: float = 1.0):
    """
    Merge consecutive segments if they belong to the same speaker and
    the gap between them is within tolerance. Handles edge cases more accurately.
    """
    if not segments:
        return []

    # Sort by start time for safety
    segments = sorted(segments, key=lambda x: x["start"])
    merged = []
    prev = None

    for seg in segments:
        spk = seg.get("speaker", "").strip().upper()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()

        if prev is None:
            prev = {"start": start, "end": end, "speaker": spk, "text": text}
            continue

        # Only merge if same speaker and gap is within tolerance and no overlap
        gap = start - prev["end"]
        if spk == prev["speaker"] and 0 <= gap <= gap_tolerance:
            prev["end"] = max(prev["end"], end)
            prev["text"] = (prev["text"].rstrip() + " " + text.lstrip()).strip()
        else:
            merged.append(prev)
            prev = {"start": start, "end": end, "speaker": spk, "text": text}

    if prev is not None:
        merged.append(prev)

    print(f"Merged {len(segments)} segments into {len(merged)} segments")
    return merged
# -------------------------
# FFmpeg helpers - Enhanced for better performance
# -------------------------
def get_ffmpeg_path():
    """Get the correct FFmpeg path with fallbacks"""
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.environ.get("FFMPEG_PATH", "ffmpeg")
    ]
    
    for candidate in candidates:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    raise FileNotFoundError("FFmpeg not found in any expected location")

def ffmpeg_cut_precise(src_video: str, start: float, end: float, out_path: str):
    """
    Precise cutting with minimal re-encoding for smooth playback.
    Uses keyframe-aware cutting with minimal quality loss.
    """
    if end <= start:
        raise ValueError("End must be greater than start.")
    
    ffmpeg_path = get_ffmpeg_path()
    duration = end - start
    
    # Use precise cutting with minimal re-encoding
    cmd = [
        ffmpeg_path, "-y",
        "-ss", f"{start:.3f}",
        "-i", src_video,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-crf", "18",  # High quality
        "-preset", "veryfast",  # Fast encoding
        "-avoid_negative_ts", "make_zero",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",  # Web optimization
        out_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Cut segment: {start:.2f}s to {end:.2f}s -> {out_path}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg cut failed: {error_message}")

def ffmpeg_concat_with_transitions(video_files, out_mp4: str):
    """
    Concatenate videos with smooth crossfade transitions.
    Enhanced for better performance and smoother output.
    """
    if not video_files:
        raise ValueError("No video files to concatenate")
    
    if len(video_files) == 1:
        # Single file, just copy
        import shutil
        shutil.copy2(video_files[0], out_mp4)
        return
    
    ffmpeg_path = get_ffmpeg_path()
    
    # Build filter complex for smooth transitions
    inputs = []
    filter_parts = []
    
    transition_duration = 0.3  # Reduced for less delay
    
    for i, vfile in enumerate(video_files):
        inputs.extend(["-i", vfile])
    
    # Create crossfade chain
    prev_label = "[0:v][0:a]"
    for i in range(1, len(video_files)):
        if i == 1:
            # First transition
            v_out = f"v{i}"
            a_out = f"a{i}"
            filter_parts.append(f"[0:v][{i}:v]xfade=transition=fade:duration={transition_duration}:offset=0[{v_out}]")
            filter_parts.append(f"[0:a][{i}:a]acrossfade=duration={transition_duration}[{a_out}]")
            prev_v_label = f"[{v_out}]"
            prev_a_label = f"[{a_out}]"
        else:
            # Chain subsequent transitions
            v_out = f"v{i}"
            a_out = f"a{i}"
            filter_parts.append(f"{prev_v_label}[{i}:v]xfade=transition=fade:duration={transition_duration}:offset=0[{v_out}]")
            filter_parts.append(f"{prev_a_label}[{i}:a]acrossfade=duration={transition_duration}[{a_out}]")
            prev_v_label = f"[{v_out}]"
            prev_a_label = f"[{a_out}]"
    
    if len(video_files) == 1:
        # No transitions needed
        cmd = [ffmpeg_path, "-y"] + inputs + ["-c", "copy", out_mp4]
    else:
        filter_complex = ";".join(filter_parts)
        final_v = prev_v_label.strip("[]")
        final_a = prev_a_label.strip("[]")
        
        cmd = [
            ffmpeg_path, "-y"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[{final_v}]",
            "-map", f"[{final_a}]",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            out_mp4
        ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Concatenated {len(video_files)} files to {out_mp4}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg concat failed: {error_message}")

def build_interleaved_output(v1_path: str, v2_path: str, segments, out_path: str):
    """
    Build final_interleaved.mp4 following speaker timeline with smooth transitions.
    Enhanced for better performance and reduced switching delays.
    """
    if not segments:
        raise RuntimeError("No valid speaker segments to build interleaved output.")

    # Filter and sort segments
    valid_segments = []
    for seg in segments:
        spk = seg["speaker"].upper().strip()
        if "SPEAKER_00" in spk or "SPEAKER_01" in spk:
            # Ensure minimum segment duration to avoid micro-cuts
            if seg["end"] - seg["start"] >= 0.5:
                valid_segments.append(seg)
    
    if not valid_segments:
        raise RuntimeError("No valid segments with sufficient duration found.")
    
    valid_segments.sort(key=lambda x: x["start"])
    
    print(f"Building interleaved output with {len(valid_segments)} segments")
    
    # Create individual clips first
    temp_clips = []
    try:
        for idx, seg in enumerate(valid_segments):
            spk = seg["speaker"].upper().strip()
            if "SPEAKER_00" in spk:
                src_video = v1_path
            else:  # SPEAKER_01
                src_video = v2_path
            
            clip_path = os.path.join(TMP_DIR, f"clip_{idx:03d}.mp4")
            ffmpeg_cut_precise(src_video, seg["start"], seg["end"], clip_path)
            temp_clips.append(clip_path)
        
        # Concatenate all clips with transitions
        ffmpeg_concat_with_transitions(temp_clips, out_path)
        
    finally:
        # Cleanup temporary clips
        for clip in temp_clips:
            try:
                if os.path.exists(clip):
                    os.remove(clip)
            except:
                pass



# =========================
# STAGE 1 — Upload one video (best audio) → extract WAV
# =========================
@app.route("/stage1", methods=["GET", "POST"])
def stage1():
    try:
        if request.method == "POST":
            try:
                # Require both videos
                if "video1" not in request.files or "video2" not in request.files:
                    logger.warning("Both videos not uploaded in Stage 1 POST.")
                    return wrap_page("Stage 1 — Error", "<div class='card notice'>Upload both videos.</div>")
                v1 = request.files["video1"]
                v2 = request.files["video2"]
                if v1.filename == "" or not _allowed(v1.filename, ALLOWED_VIDEO):
                    logger.warning("Invalid Video 1 format uploaded in Stage 1 POST.")
                    return wrap_page("Stage 1 — Error", "<div class='card notice'>Invalid Video 1 format. Allowed: mp4, mov, mkv, avi, m4v</div>")
                if v2.filename == "" or not _allowed(v2.filename, ALLOWED_VIDEO):
                    logger.warning("Invalid Video 2 format uploaded in Stage 1 POST.")
                    return wrap_page("Stage 1 — Error", "<div class='card notice'>Invalid Video 2 format. Allowed: mp4, mov, mkv, avi, m4v</div>")

                # Save both videos
                v1_path = os.path.join(UPLOAD_STAGE1, v1.filename)
                v2_path = os.path.join(UPLOAD_STAGE1, v2.filename)
                try:
                    v1.save(v1_path)
                    v2.save(v2_path)
                    logger.info(f"Videos saved to {v1_path} and {v2_path}")
                except Exception as e:
                    logger.exception("Failed to save uploaded videos.")
                    return wrap_page("Stage 1 — Error", f"<div class='card notice'>Failed to save videos: {str(e)}</div>")

                # Get which video has best audio
                best_audio = request.form.get("best_audio")
                if best_audio not in ["video1", "video2"]:
                    logger.warning("Best audio selection missing in Stage 1 POST.")
                    return wrap_page("Stage 1 — Error", "<div class='card notice'>Please select which video has the best audio.</div>")
                best_video_path = v1_path if best_audio == "video1" else v2_path
                base = os.path.splitext(os.path.basename(best_video_path))[0]
                wav_path = os.path.join(OUTPUT_AUDIO, f"{base}.wav")

                # Get merge tolerance from user input
                merge_tolerance = request.form.get("merge_tolerance", "3.0")
                try:
                    merge_tolerance = float(merge_tolerance)
                except Exception:
                    merge_tolerance = 3.0

                try:
                    extract_audio_to_wav(best_video_path, wav_path)
                    logger.info(f"Audio extracted to {wav_path}")
                except Exception as e:
                    logger.exception(f"Audio extraction failed for {best_video_path}")
                    return wrap_page("Stage 1 — Error", f"<div class='card notice'>Audio extraction failed: {str(e)}</div>")

                # Pass both video filenames and tolerance to stage2/3 for automation
                return run_stage2_auto(wav_path, v1.filename, v2.filename, merge_tolerance)

            except Exception as e:
                logger.exception("Exception in Stage 1 POST logic")
                error_body = f"""
                <div class="card notice">
                  <h3>Stage 1 failed ❌</h3>
                  <p>Error: {str(e)}</p>
                  <p><a href="/stage1">Try again</a></p>
                </div>
                """
                return wrap_page("Stage 1 — Error", error_body)

        # GET request: show upload form
        form = """
        <div class="card">
          <h2>Stage 1 — Upload TWO videos</h2>
          <form method="post" enctype="multipart/form-data" class="grid">
            <div class="grid grid-2">
              <div>
                <label>Video 1</label>
                <input type="file" name="video1" accept="video/*" required>
              </div>
              <div>
                <label>Video 2</label>
                <input type="file" name="video2" accept="video/*" required>
              </div>
            </div>
            <div>
              <label>Which video has the best audio?</label>
              <input type="radio" name="best_audio" value="video1" required> Video 1 &nbsp;
              <input type="radio" name="best_audio" value="video2" required> Video 2
            </div>
            <div>
              <label>Merge tolerance (seconds)</label>
              <input type="number" name="merge_tolerance" value="1.0" step="0.1" min="0" max="5" required>
              <p class="small">Segments within this time gap will be merged if they're from the same speaker.</p>
            </div>
            <div><button type="submit">Upload &amp; Extract Audio</button></div>
          </form>
          <p class="small">Both videos will be used for the final output. The selected video will be used for audio extraction and transcription.</p>
        </div>
        """
        return wrap_page("Stage 1 — Upload Videos", form)

    except Exception as e:
        logger.exception("Error in /stage1")
        return wrap_page("Stage 1 — Error", f"<div class='card notice'>Error: {str(e)}</div>")



def run_stage2_auto(audio_path, video1_filename, video2_filename, merge_tolerance=1.0):
    try:
        logger.info(f"Processing audio file: {audio_path}")

        # 1) Load and transcribe audio
        audio = whisperx.load_audio(audio_path)
        asr_result = asr_model.transcribe(audio, language="en")
        if not asr_result.get("segments"):
            logger.warning("No speech detected in audio.")
            return wrap_page("Stage 2 — Error", "<div class='card notice'>No speech detected in audio. Try a different file.</div>")

        # 2) Align for better timestamps
        try:
            align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
            aligned = whisperx.align(asr_result["segments"], align_model, metadata, audio, DEVICE, return_char_alignments=False)
            segments_to_diarize = aligned["segments"]
        except Exception as e:
            logger.warning(f"Alignment failed, using raw transcription: {e}")
            segments_to_diarize = asr_result["segments"]

        if diarize_pipeline is not None:
            try:
                diar_segments = diarize_pipeline(audio_path, min_speakers=1, max_speakers=2)
                final_result = whisperx.assign_word_speakers(diar_segments, {"segments": segments_to_diarize})
                final_segments = final_result["segments"]
            except Exception as e:
                logger.warning(f"Diarization failed, using single speaker: {e}")
                final_segments = []
                for seg in segments_to_diarize:
                    seg_copy = seg.copy()
                    seg_copy["speaker"] = "SPEAKER_00"
                    final_segments.append(seg_copy)
        else:
            final_segments = []
            for seg in segments_to_diarize:
                seg_copy = seg.copy()
                seg_copy["speaker"] = "SPEAKER_00"
                final_segments.append(seg_copy)

        converted_segments = []
        for seg in final_segments:
            converted_segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "text": seg.get("text", "").strip()
            })
        merged_segments = merge_contiguous_same_speaker(converted_segments, gap_tolerance=merge_tolerance)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        srt_path, txt_path = save_transcripts(merged_segments, base)

        logger.info(f"Transcription and diarization complete for {audio_path}, SRT: {srt_path}")
        # Pass video filenames and SRT to stage3
        return redirect(f"/stage3?auto_srt={os.path.basename(srt_path)}&video1={video1_filename}&video2={video2_filename}")
    except Exception as e:
        logger.exception("Error in run_stage2_auto")
        return wrap_page("Stage 2 — Error", f"<div class='card notice'>Error: {str(e)}</div>")



@app.route("/stage2", methods=["GET", "POST"])
def stage2():
    if request.method == "POST":
        try:
            if "audio" not in request.files:
                return wrap_page("Stage 2 — Error", "<div class='card notice'>No audio uploaded.</div>")
            a = request.files["audio"]
            if a.filename == "" or not _allowed(a.filename, ALLOWED_AUDIO):
                return wrap_page("Stage 2 — Error", "<div class='card notice'>Invalid audio format. Allowed: wav, mp3, m4a, aac, flac</div>")

            audio_path = os.path.join(UPLOAD_STAGE2, a.filename)
            a.save(audio_path)
            
            print(f"Processing audio file: {a.filename}")

            # 1) Load and transcribe audio
            print("Loading audio...")
            audio = whisperx.load_audio(audio_path)
            
            print("Transcribing with WhisperX...")
            asr_result = asr_model.transcribe(audio, language="en")
            
            if not asr_result.get("segments"):
                return wrap_page("Stage 2 — Error", "<div class='card notice'>No speech detected in audio. Try a different file.</div>")

            # 2) Align for better timestamps
            print("Aligning transcription...")
            try:
                align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
                aligned = whisperx.align(asr_result["segments"], align_model, metadata, audio, DEVICE, return_char_alignments=False)
                segments_to_diarize = aligned["segments"]
            except Exception as e:
                print(f"Alignment failed, using raw transcription: {e}")
                segments_to_diarize = asr_result["segments"]

            # 3) Diarization (speaker separation)
            print("Performing speaker diarization...")
            if diarize_pipeline is not None:
                try:
                    diar_segments = diarize_pipeline(audio_path, min_speakers=1, max_speakers=2)
                    
                    # 4) Assign speakers to transcription segments
                    print("Assigning speakers to words...")
                    final_result = whisperx.assign_word_speakers(diar_segments, {"segments": segments_to_diarize})
                    final_segments = final_result["segments"]
                except Exception as e:
                    print(f"Diarization failed, using single speaker: {e}")
                    # Fallback: assign all to SPEAKER_00
                    final_segments = []
                    for seg in segments_to_diarize:
                        seg_copy = seg.copy()
                        seg_copy["speaker"] = "SPEAKER_00"
                        final_segments.append(seg_copy)
            else:
                print("Diarization pipeline not available, using single speaker")
                final_segments = []
                for seg in segments_to_diarize:
                    seg_copy = seg.copy()
                    seg_copy["speaker"] = "SPEAKER_00"
                    final_segments.append(seg_copy)

            # Convert to our format and merge consecutive same-speaker segments
            converted_segments = []
            for seg in final_segments:
                converted_segments.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "speaker": seg.get("speaker", "SPEAKER_00"),
                    "text": seg.get("text", "").strip()
                })
            
            # Merge consecutive same-speaker segments
            # Use default tolerance for manual stage2
            merged_segments = merge_contiguous_same_speaker(converted_segments, gap_tolerance=1.0)
            
            # Save transcripts
            base = os.path.splitext(os.path.basename(a.filename))[0]
            srt_path, txt_path = save_transcripts(merged_segments, base)

            # Clear GPU memory
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            # Count speakers for user info
            speakers = set(seg["speaker"] for seg in merged_segments)
            speaker_info = f"Detected {len(speakers)} speaker(s): {', '.join(sorted(speakers))}"

            body = f"""
            <div class="card">
              <h3>Stage 2 complete ✅</h3>
              <p>Audio processed: <code>{a.filename}</code></p>
              <p>{speaker_info}</p>
              <p>Segments: {len(merged_segments)} total</p>
              <p>SRT saved: <a href='/transcripts/{os.path.basename(srt_path)}'>{os.path.basename(srt_path)}</a></p>
              <p>TXT saved: <a href='/transcripts/{os.path.basename(txt_path)}'>{os.path.basename(txt_path)}</a></p>
              <p class="small">Next: <a class="btn" href='/stage3'>Stage 3 — Build final video(s) from two sources</a></p>
            </div>
            """
            return wrap_page("Stage 2 — Done", body)
            
        except Exception as e:
            # Clear GPU memory on error
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            
            error_body = f"""
            <div class="card notice">
              <h3>Stage 2 failed ❌</h3>
              <p>Error: {str(e)}</p>
              <p><a href="/stage2">Try again</a></p>
            </div>
            """
            return wrap_page("Stage 2 — Error", error_body)

    # Check for available audio files from Stage 1
    available_audio = []
    if os.path.exists(OUTPUT_AUDIO):
        available_audio = [f for f in os.listdir(OUTPUT_AUDIO) if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
    
    audio_options = ""
    if available_audio:
        audio_options = f"""
        <div class="notice">
          <p><strong>Available audio from Stage 1:</strong></p>
          <ul>
            {''.join([f'<li><a href="/audio/{f}">{f}</a></li>' for f in available_audio])}
          </ul>
        </div>
        """

    form = f"""
    <div class="card">
      <h2>Stage 2 — Upload AUDIO for transcription + diarization</h2>
      <form method="post" enctype="multipart/form-data" class="grid">
        <div>
          <label>Audio file</label>
          <input type="file" name="audio" accept="audio/*" required>
          <p class="small">Supported: {', '.join(ALLOWED_AUDIO)}</p>
        </div>
        <div><button type="submit">Upload &amp; Transcribe</button></div>
      </form>
      <p class="small">Device: <code>{DEVICE}</code> | Model: <code>{MODEL_SIZE}</code> | Compute: <code>{COMPUTE_TYPE}</code></p>
      {audio_options}
    </div>
    """
    return wrap_page("Stage 2 — Transcribe & Diarize", form)

# =========================
# MERGE SRT ROUTE (enhanced)
# =========================
@app.route("/merge_srt", methods=["GET", "POST"])
def merge_srt():
    if request.method == "POST":
        try:
            if "srt_file" not in request.files:
                return wrap_page("Merge SRT — Error", "<div class='card notice'>No SRT file uploaded.</div>")
            srt_file = request.files["srt_file"]
            if srt_file.filename == "" or not srt_file.filename.lower().endswith(".srt"):
                return wrap_page("Merge SRT — Error", "<div class='card notice'>Invalid SRT file. Must be .srt format.</div>")
            
            # Save uploaded SRT temporarily
            temp_srt_path = os.path.join(TMP_DIR, srt_file.filename)
            srt_file.save(temp_srt_path)
            base = os.path.splitext(os.path.basename(srt_file.filename))[0] + "_merged"

            # Parse and merge
            try:
               segments = parse_diarized_srt(temp_srt_path)
            except Exception as e:
               logger.exception(f"Failed to parse SRT: {temp_srt_path}")
               raise
            if not segments:
                return wrap_page("Merge SRT — Error", "<div class='card notice'>No valid segments found in SRT file.</div>")
            
            gap_tolerance = float(request.form.get("gap_tolerance", "1.0"))
            merged_segments = merge_contiguous_same_speaker(segments, gap_tolerance=gap_tolerance)
            srt_path, txt_path = save_transcripts(merged_segments, base)

            # Cleanup temp file
            try:
                os.remove(temp_srt_path)
            except:
                pass

            reduction_pct = ((len(segments) - len(merged_segments)) / len(segments) * 100) if segments else 0

            body = f"""
            <div class="card">
              <h3>Merging complete ✅</h3>
              <p>Original segments: {len(segments)}</p>
              <p>Merged segments: {len(merged_segments)} ({reduction_pct:.1f}% reduction)</p>
              <p>Gap tolerance: {gap_tolerance}s</p>
              <p>Merged SRT: <a href='/transcripts/{os.path.basename(srt_path)}'>{os.path.basename(srt_path)}</a></p>
              <p>Merged TXT: <a href='/transcripts/{os.path.basename(txt_path)}'>{os.path.basename(txt_path)}</a></p>
            </div>
            """
            return wrap_page("Merge SRT — Done", body)
        except Exception as e:
            error_body = f"""
            <div class="card notice">
              <h3>Merge failed ❌</h3>
              <p>Error: {str(e)}</p>
              <p><a href="/merge_srt">Try again</a></p>
            </div>
            """
            return wrap_page("Merge SRT — Error", error_body)

    form = """
    <div class="card">
      <h2>Merge consecutive same-speaker SRT segments</h2>
      <form method="post" enctype="multipart/form-data">
        <label>Upload diarized SRT file</label>
        <input type="file" name="srt_file" accept=".srt" required>
        
        <label>Gap tolerance (seconds)</label>
        <input type="number" name="gap_tolerance" value="1.0" step="0.1" min="0" max="5">
        <p class="small">Segments within this time gap will be merged if they're from the same speaker.</p>
        
        <button type="submit">Merge &amp; Download</button>
      </form>
      <p class="small">This will merge all consecutive same-speaker segments and let you download the result.</p>
    </div>
    """
    return wrap_page("Merge SRT", form)

# =========================
# STAGE 3 — Upload two videos + choose transcript → build final video(s)
# =========================

# ...existing code...
@app.route("/stage3", methods=["GET", "POST"])
def stage3():
    # Automatic mode: use query params for video1, video2, auto_srt
    auto_srt = request.args.get("auto_srt")
    video1 = request.args.get("video1")
    video2 = request.args.get("video2")
    if auto_srt and video1 and video2:
        try:
            v1_path = os.path.join(UPLOAD_STAGE1, video1)
            v2_path = os.path.join(UPLOAD_STAGE1, video2)
            srt_path = os.path.join(OUTPUT_TRANSCRIPTS, auto_srt)
            if not (os.path.isfile(v1_path) and os.path.isfile(v2_path) and os.path.isfile(srt_path)):
                logger.error("Required files not found for final video creation.")
                return wrap_page("Stage 3 — Error", "<div class='card notice'>Required files not found for final video creation.</div>")

            raw_segments = parse_diarized_srt(srt_path)
            segments = [s for s in raw_segments if s["speaker"].upper().strip() in ("SPEAKER_00","SPEAKER_01")]
            if not segments:
                logger.error("No valid SPEAKER_00 or SPEAKER_01 segments found in transcript.")
                return wrap_page("Stage 3 — Error", "<div class='card notice'>No valid SPEAKER_00 or SPEAKER_01 segments found in transcript.</div>")
            segments = merge_contiguous_same_speaker(segments, gap_tolerance=1.0)

            interleaved_out = os.path.join(FINAL_OUTPUTS, "final_interleaved.mp4")
            try:
                build_interleaved_output(v1_path, v2_path, segments, interleaved_out)
                logger.info("Interleaved video created successfully")
            except Exception as e:
                logger.exception("Interleaved build error")
                return wrap_page("Stage 3 — Error", f"<div class='card notice'>Interleaved build failed: {str(e)}</div>")

            size_mb = os.path.getsize(interleaved_out) / (1024*1024) if os.path.isfile(interleaved_out) else 0
            body = f"""
            <div class="card">
              <h3>Stage 3 complete ✅</h3>
              <p>Processed {len(segments)} speaker segments</p>
              <ul>
                <li>Interleaved (timeline): <a href='/outputs/{os.path.basename(interleaved_out)}'>{os.path.basename(interleaved_out)}</a> ({size_mb:.1f}MB)</li>
              </ul>
              <p class="small">Speaker mapping: <code>SPEAKER_00 → Video 1 ({video1})</code>, <code>SPEAKER_01 → Video 2 ({video2})</code></p>
            </div>
            """
            return wrap_page("Stage 3 — Done", body)
        except Exception as e:
            logger.exception("Stage 3 auto mode failed")
            return wrap_page("Stage 3 — Error", f"<div class='card notice'>Error: {str(e)}</div>")

    # Manual fallback: show upload form (only interleaved option)
    srt_list = []
    if os.path.exists(OUTPUT_TRANSCRIPTS):
        srt_list = [f for f in os.listdir(OUTPUT_TRANSCRIPTS) if f.lower().endswith(".srt")]
    srt_options = "".join([f"<option value='{s}'>{s}</option>" for s in sorted(srt_list)])
    srt_info = ""
    if srt_list:
        srt_info = f"""
        <div class="notice">
          <p><strong>Available SRT files from Stage 2:</strong></p>
          <p class="small">{len(srt_list)} file(s) found in transcripts folder</p>
        </div>
        """

    form = f"""
    <div class="card">
      <h2>Stage 3 — Upload two videos &amp; choose transcript</h2>
      <div class="notice small">
        <strong>Important:</strong> Use the same recording session for both videos. 
        Videos are cut using absolute timestamps from the SRT, so timeline synchronization is crucial.
      </div>
      <form method="post" enctype="multipart/form-data" class="grid">
        <div class="grid grid-2">
          <div>
            <label>Video 1 (Speaker: <code>SPEAKER_00</code>)</label>
            <input type="file" name="video1" accept="video/*" required>
          </div>
          <div>
            <label>Video 2 (Speaker: <code>SPEAKER_01</code>)</label>
            <input type="file" name="video2" accept="video/*" required>
          </div>
        </div>
        <div>
          <label>Output type</label>
          <select name="output_type">
            <option value="interleaved">Interleaved only (timeline order)</option>
          </select>
        </div>
        <div>
          <label>Select existing SRT from server</label>
          <select name="srt_name">
            <option value="">-- Choose existing SRT --</option>
            {srt_options}
          </select>
        </div>
        <div>
          <label>OR upload a new .srt file</label>
          <input type="file" name="srt_file" accept=".srt">
          <p class="small">Upload takes priority over selection above</p>
        </div>
        <div><button type="submit">Build Final Video</button></div>
      </form>
      <p class="small">
        <strong>Output type:</strong><br>
        • <strong>Interleaved:</strong> Switches between speakers following original timeline<br>
        • Output includes smooth crossfade transitions
      </p>
    </div>
    {srt_info}
    """
    return wrap_page("Stage 3 — Build Final", form)
# =========================
# ENHANCED ROUTES FOR BETTER UX
# =========================
@app.route("/status")
def status():
    """System status and diagnostics"""
    # Check GPU status
    gpu_info = "Not available"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        used_memory = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_info = f"{gpu_name} ({used_memory:.1f}/{total_memory:.1f} GB used)"
    
    # Check FFmpeg
    ffmpeg_status = "Not found"
    try:
        ffmpeg_path = get_ffmpeg_path()
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            ffmpeg_status = f"Available: {version_line}"
    except Exception as e:
        ffmpeg_status = f"Error: {str(e)}"
    
    # Count files in each directory
    stage1_files = len([f for f in os.listdir(UPLOAD_STAGE1) if os.path.isfile(os.path.join(UPLOAD_STAGE1, f))]) if os.path.exists(UPLOAD_STAGE1) else 0
    stage2_files = len([f for f in os.listdir(UPLOAD_STAGE2) if os.path.isfile(os.path.join(UPLOAD_STAGE2, f))]) if os.path.exists(UPLOAD_STAGE2) else 0
    stage3_files = len([f for f in os.listdir(UPLOAD_STAGE3) if os.path.isfile(os.path.join(UPLOAD_STAGE3, f))]) if os.path.exists(UPLOAD_STAGE3) else 0
    audio_files = len([f for f in os.listdir(OUTPUT_AUDIO) if os.path.isfile(os.path.join(OUTPUT_AUDIO, f))]) if os.path.exists(OUTPUT_AUDIO) else 0
    transcript_files = len([f for f in os.listdir(OUTPUT_TRANSCRIPTS) if os.path.isfile(os.path.join(OUTPUT_TRANSCRIPTS, f))]) if os.path.exists(OUTPUT_TRANSCRIPTS) else 0
    output_files = len([f for f in os.listdir(FINAL_OUTPUTS) if os.path.isfile(os.path.join(FINAL_OUTPUTS, f))]) if os.path.exists(FINAL_OUTPUTS) else 0

    body = f"""
    <div class="card">
      <h2>System Status</h2>
      
      <h3>Hardware</h3>
      <p><strong>Device:</strong> <code>{DEVICE}</code></p>
      <p><strong>Compute Type:</strong> <code>{COMPUTE_TYPE}</code></p>
      <p><strong>Model Size:</strong> <code>{MODEL_SIZE}</code></p>
      <p><strong>GPU:</strong> <code>{gpu_info}</code></p>
      <p><strong>FFmpeg:</strong> <code>{ffmpeg_status}</code></p>
      
      <h3>File Counts</h3>
      <p>Stage 1 uploads: <code>{stage1_files}</code></p>
      <p>Stage 2 uploads: <code>{stage2_files}</code></p>
      <p>Stage 3 uploads: <code>{stage3_files}</code></p>
      <p>Extracted audio: <code>{audio_files}</code></p>
      <p>Transcripts: <code>{transcript_files}</code></p>
      <p>Final outputs: <code>{output_files}</code></p>
      
      <h3>Storage</h3>
      <p><strong>Base directory:</strong> <code>{BASE_DIR}</code></p>
      <p><strong>Temp directory:</strong> <code>{TMP_DIR}</code></p>
    </div>
    """
    return wrap_page("System Status", body)

@app.route("/cleanup", methods=["GET", "POST"])
def cleanup():
    """Clean up temporary and old files"""
    if request.method == "POST":
        try:
            cleaned_count = 0
            
            # Clean temp directory
            if os.path.exists(TMP_DIR):
                for f in os.listdir(TMP_DIR):
                    file_path = os.path.join(TMP_DIR, f)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            cleaned_count += 1
                    except:
                        pass
            
            # Clean GPU memory
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            
            body = f"""
            <div class="card">
              <h3>Cleanup complete ✅</h3>
              <p>Cleaned {cleaned_count} temporary files</p>
              <p>GPU memory cleared</p>
            </div>
            """
            return wrap_page("Cleanup — Done", body)
        except Exception as e:
            error_body = f"""
            <div class="card notice">
              <h3>Cleanup failed ❌</h3>
              <p>Error: {str(e)}</p>
            </div>
            """
            return wrap_page("Cleanup — Error", error_body)

    form = """
    <div class="card">
      <h2>System Cleanup</h2>
      <p>This will clean temporary files and clear GPU memory.</p>
      <form method="post">
        <button type="submit">Clean Up System</button>
      </form>
      <p class="small">Note: This will not delete your uploads, transcripts, or final outputs.</p>
    </div>
    """
    return wrap_page("System Cleanup", form)

# =========================
# SIMPLE HOME + DOWNLOADS
# =========================
@app.route("/")
def home():
    # Get some basic stats
    audio_count = len([f for f in os.listdir(OUTPUT_AUDIO) if f.lower().endswith(('.wav', '.mp3'))]) if os.path.exists(OUTPUT_AUDIO) else 0
    transcript_count = len([f for f in os.listdir(OUTPUT_TRANSCRIPTS) if f.lower().endswith('.srt')]) if os.path.exists(OUTPUT_TRANSCRIPTS) else 0
    output_count = len([f for f in os.listdir(FINAL_OUTPUTS) if f.lower().endswith('.mp4')]) if os.path.exists(FINAL_OUTPUTS) else 0

    body = f"""
    <div class="card">
      <h2>WhisperX Video Processing Pipeline</h2>
      <p class="small">3-stage pipeline for multi-speaker video processing with AI transcription</p>
      
      <div class="grid grid-2">
        <div>
          <h3>Processing Stages</h3>
          <ol>
            <li><a href="/stage1"><strong>Extract Audio</strong></a><br>
                <span class="small">Upload video → extract WAV for transcription</span></li>
            <li><a href="/stage2"><strong>Transcribe & Diarize</strong></a><br>
                <span class="small">Upload audio → AI transcription + speaker separation</span></li>
            <li><a href="/stage3"><strong>Build Final Videos</strong></a><br>
                <span class="small">Upload two videos + transcript → create final output</span></li>
          </ol>
        </div>
        
        <div>
          <h3>Quick Stats</h3>
          <p>Audio files: <code>{audio_count}</code></p>
          <p>Transcripts: <code>{transcript_count}</code></p>
          <p>Final videos: <code>{output_count}</code></p>
          <p>Device: <code>{DEVICE}</code></p>
          <p>Model: <code>{MODEL_SIZE}</code></p>
        </div>
      </div>
      
      <hr>
      
      <div class="grid grid-2">
        <div>
          <h3>Utilities</h3>
          <p><a href="/merge_srt">Merge SRT segments</a></p>
          <p><a href="/status">System status</a></p>
          <p><a href="/cleanup">Clean temporary files</a></p>
        </div>
        <div>
          <h3>Storage</h3>
          <p><strong>Base:</strong> <code>{os.path.basename(BASE_DIR)}</code></p>
          <p class="small">All processing happens in: <code>{BASE_DIR}</code></p>
        </div>
      </div>
    </div>
    """
    return wrap_page("WhisperX 3-Stage Pipeline", body)

@app.route("/audio/<path:filename>")
def dl_audio(filename):
    """Download audio files"""
    try:
        return send_from_directory(OUTPUT_AUDIO, filename, as_attachment=True)
    except FileNotFoundError:
        return wrap_page("File Not Found", "<div class='card notice'>Audio file not found.</div>")

@app.route("/transcripts/<path:filename>")
def dl_transcript(filename):
    """Download transcript files"""
    try:
        return send_from_directory(OUTPUT_TRANSCRIPTS, filename, as_attachment=True)
    except FileNotFoundError:
        return wrap_page("File Not Found", "<div class='card notice'>Transcript file not found.</div>")

@app.route("/outputs/<path:filename>")
def dl_output(filename):
    """Download final output files"""
    try:
        return send_from_directory(FINAL_OUTPUTS, filename, as_attachment=True)
    except FileNotFoundError:
        return wrap_page("File Not Found", "<div class='card notice'>Output file not found.</div>")

# =========================
# ERROR HANDLERS
# =========================
@app.errorhandler(404)
def not_found(error):
    body = """
    <div class="card notice">
      <h3>Page Not Found</h3>
      <p>The requested page could not be found.</p>
      <p><a href="/">Return to home</a></p>
    </div>
    """
    return wrap_page("404 - Not Found", body), 404

@app.errorhandler(500)
def internal_error(error):
    # Clear GPU memory on error
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    body = f"""
    <div class="card notice">
      <h3>Internal Server Error</h3>
      <p>An error occurred while processing your request.</p>
      <p class="small">Error: {str(error)}</p>
      <p><a href="/">Return to home</a></p>
    </div>
    """
    return wrap_page("500 - Internal Error", body), 500

@app.errorhandler(413)
def file_too_large(error):
    body = """
    <div class="card notice">
      <h3>File Too Large</h3>
      <p>The uploaded file is too large. Please try a smaller file.</p>
      <p><a href="/">Return to home</a></p>
    </div>
    """
    return wrap_page("413 - File Too Large", body), 413

# =========================
# ENHANCED FFMPEG FUNCTIONS FOR SMOOTH VIDEO PROCESSING
# =========================
def validate_video_files(v1_path: str, v2_path: str):
    """Validate that video files exist and are readable"""
    for path in [v1_path, v2_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"Video file is empty: {path}")

def get_video_info(video_path: str):
    """Get video duration and basic info using ffprobe"""
    ffmpeg_path = get_ffmpeg_path()
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    
    cmd = [
        ffprobe_path, "-v", "quiet", "-print_format", "json", 
        "-show_format", "-show_streams", video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        return {"duration": duration, "info": info}
    except Exception as e:
        print(f"Warning: Could not get video info for {video_path}: {e}")
        return {"duration": 0, "info": None}

def create_smooth_transitions_filter(segments, v1_path: str, v2_path: str):
    """
    Create advanced filter complex for smooth video transitions.
    Reduces switching delays and improves visual continuity.
    """
    if not segments:
        raise ValueError("No segments provided for filter creation")
    
    # Validate video files first
    validate_video_files(v1_path, v2_path)
    
    # Get video info for validation
    v1_info = get_video_info(v1_path)
    v2_info = get_video_info(v2_path)
    
    # Get frame rate (default to 30 if not found)
    def get_fps(info):
        try:
            for stream in info["info"]["streams"]:
                if stream["codec_type"] == "video":
                    r = stream.get("r_frame_rate", "30/1")
                    num, denom = map(float, r.split('/'))
                    return num / denom if denom else 30.0
        except Exception:
            pass
        return 30.0

    fps_v1 = get_fps(v1_info)
    fps_v2 = get_fps(v2_info)
    
    filter_parts = []
    transition_duration = 0.1  # Shorter transitions for less delay
    
    # Pre-process segments for optimal cutting
    processed_segments = []
    for i, seg in enumerate(segments):
        start_time = max(0, seg["start"] - 0.1)  # Small buffer for smooth cuts
        end_time = seg["end"] + 0.1  # Small buffer
        
        # Validate timing against video duration
        spk = seg["speaker"].upper().strip()
        max_duration = v1_info["duration"] if "SPEAKER_00" in spk else v2_info["duration"]
        fps = fps_v1 if "SPEAKER_00" in spk else fps_v2
        
        if end_time > max_duration:
            end_time = max_duration
        if start_time >= end_time:
            continue  # Skip invalid segments
            
        processed_segments.append({
            **seg,
            "start": start_time,
            "end": end_time,
            "video_index": 0 if "SPEAKER_00" in spk else 1,
            "fps": fps
        })
    
    if not processed_segments:
        raise ValueError("No valid segments after processing")
    
    # Build filter chain for smooth concatenation
    segment_labels = []
    for idx, seg in enumerate(processed_segments):
        start, end = seg["start"], seg["end"]
        video_idx = seg["video_index"]
        fps = seg["fps"]
        duration = end - start
        
        v_label = f"v{idx}"
        a_label = f"a{idx}"
        
        # Fade durations in frames for video, seconds for audio
        fade_duration_frames = int(min(0.5, duration / 2) * fps)
        fade_out_start_frame = int((duration - min(0.5, duration / 2)) * fps)
        fade_duration_audio = min(0.5, duration / 2)
        fade_out_start_audio = duration - fade_duration_audio
        
        # Video filter
        filter_parts.append(
            f"[{video_idx}:v]trim={start}:{end},setpts=PTS-STARTPTS,"
            f"fade=in:0:{fade_duration_frames}:alpha=1,"
            f"fade=out:st={fade_out_start_frame}:d={fade_duration_frames}:alpha=1[{v_label}]"
        )
        # Audio filter
        filter_parts.append(
            f"[{video_idx}:a]atrim={start}:{end},asetpts=PTS-STARTPTS,"
            f"afade=in:st=0:d={fade_duration_audio},"
            f"afade=out:st={fade_out_start_audio}:d={fade_duration_audio}[{a_label}]"
        )
        
        segment_labels.append((v_label, a_label))
    
    # Concatenate all segments
    if len(segment_labels) > 1:
        v_inputs = "".join([f"[{v}]" for v, a in segment_labels])
        a_inputs = "".join([f"[{a}]" for v, a in segment_labels])
        filter_parts.append(f"{v_inputs}concat=n={len(segment_labels)}:v=1:a=0[outv]")
        filter_parts.append(f"{a_inputs}concat=n={len(segment_labels)}:v=0:a=1[outa]")
    else:
        # Single segment
        v_label, a_label = segment_labels[0]
        filter_parts.append(f"[{v_label}]copy[outv]")
        filter_parts.append(f"[{a_label}]copy[outa]")
    
    return ";".join(filter_parts), "[outv]", "[outa]"

def build_interleaved_output_enhanced(v1_path: str, v2_path: str, segments, out_path: str):
    """
    Enhanced interleaved output with smooth transitions and reduced delays.
    """
    if not segments:
        raise RuntimeError("No valid speaker segments to build interleaved output.")

    print(f"Building enhanced interleaved output with {len(segments)} segments")
    
    try:
        # Create filter complex for smooth transitions
        filter_complex, map_v, map_a = create_smooth_transitions_filter(segments, v1_path, v2_path)
        
        ffmpeg_path = get_ffmpeg_path()
        cmd = [
            ffmpeg_path, "-y",
            "-i", v1_path,
            "-i", v2_path,
            "-filter_complex", filter_complex,
            "-map", map_v,
            "-map", map_a,
            "-c:v", "libx264",
            "-crf", "18",  # High quality
            "-preset", "veryfast",  # Fast encoding for RTX 3050
            "-tune", "film",  # Better for video content
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",  # Web optimization
            "-avoid_negative_ts", "make_zero",
            out_path
        ]
        
        # Use hardware encoding if available (for RTX 3050)
        if DEVICE == "cuda":
            try:
                # Test if NVENC is available
                test_cmd = [ffmpeg_path, "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1", "-c:v", "h264_nvenc", "-f", "null", "-"]
                subprocess.run(test_cmd, capture_output=True, check=True, timeout=10)
                # Replace software encoding with hardware
                cmd[cmd.index("-c:v")+1] = "h264_nvenc"
                cmd[cmd.index("-preset")+1] = "fast"
                if "-tune" in cmd:
                    tune_idx = cmd.index("-tune")
                    del cmd[tune_idx:tune_idx+2]  # Remove tune option for NVENC
                print("Using NVIDIA hardware encoding")
            except:
                print("NVENC not available, using software encoding")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print(f"Interleaved output created: {out_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to build interleaved output: {str(e)}")


# =========================
# ROUTE UPDATES WITH ENHANCED FUNCTIONS
# =========================

# Update the build functions in stage3 route
def build_interleaved_output(v1_path: str, v2_path: str, segments, out_path: str):
    """Wrapper to maintain compatibility - calls enhanced version"""
    return build_interleaved_output_enhanced(v1_path, v2_path, segments, out_path)



# =========================
# CONFIGURATION AND FLASK SETTINGS
# =========================

# Flask configuration for better file handling
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB max file size
app.config['UPLOAD_EXTENSIONS'] = ALLOWED_VIDEO.union(ALLOWED_AUDIO)

# =========================
# MAIN
# =========================

def print_startup_info():
    """Print startup information"""
    print("=" * 50)
    print("WhisperX Flask Application Starting")
    print("=" * 50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Compute Type: {COMPUTE_TYPE}")
    print(f"Model Size: {MODEL_SIZE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print(f"Diarization: {'Available' if diarize_pipeline else 'Not Available'}")
    
    # Check FFmpeg
    try:
        ffmpeg_path = get_ffmpeg_path()
        print(f"FFmpeg: {ffmpeg_path}")
    except:
        print("FFmpeg: Not Found - Please install FFmpeg")
    
    print("=" * 50)
    print("Server starting at http://localhost:5000")
    print("=" * 50)

if __name__ == "__main__":
    try:
        print_startup_info()
        
        # Avoid Flask reloader duplicating GPU memory load
        app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Clear GPU memory on shutdown
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
                print("GPU memory cleared")
            except:
                pass
    except Exception as e:
        print(f"Startup error: {e}")
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
                print("GPU memory cleared")
            except:
                pass