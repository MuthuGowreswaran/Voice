import os
import time
import numpy as np
import sounddevice as sd
import whisper
import tempfile
from scipy.io.wavfile import write
from llama_chat import ask_groq_llama

# Add ffmpeg to PATH
os.environ["PATH"] += os.pathsep + os.path.abspath("bin")

# Load whisper model (use 'base' or 'tiny' for speed)
model = whisper.load_model("small")

# Audio Config
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.3
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.03
SILENCE_TIMEOUT = 4.0
MIN_SPEECH_DURATION = 1.0

def record_audio():
    print("🎙️ Listening... (Speak now)")
    buffer = []
    rms_values = []
    silent_chunks = 0
    recording = False
    silence_limit = int(SILENCE_TIMEOUT / CHUNK_DURATION)
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal silent_chunks, recording
        rms = np.sqrt(np.mean(indata**2))
        rms_values.append(rms)
        buffer.append(indata.copy())

        if rms > SILENCE_THRESHOLD:
            silent_chunks = 0
            if not recording:
                print("🔊 Detected speech...")
            recording = True
        elif recording:
            silent_chunks += 1

    with sd.InputStream(callback=callback, channels=CHANNELS,
                        samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, dtype='float32'):
        while True:
            sd.sleep(int(CHUNK_DURATION * 1000))
            if recording and silent_chunks > silence_limit:
                break
            if time.time() - start_time > 60:  # max 20s to avoid being stuck
                break

    if not recording:
        print(f"🔇 No speech detected. Max RMS: {max(rms_values, default=0):.4f}")
        return None

    audio = np.concatenate(buffer)
    scaled = (audio * 32767).astype(np.int16)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp_wav.name, SAMPLE_RATE, scaled)

    duration = len(audio) / SAMPLE_RATE
    print(f"🔈 Recorded {duration:.1f}s | Max RMS: {max(rms_values):.4f} | Avg RMS: {np.mean(rms_values):.4f}")
    return temp_wav.name, scaled

def transcribe_audio(path, audio_data=None):
    try:
        print("🔍 Transcribing audio...")
        if audio_data is not None:
            try:
                # Flatten the 2D array for Whisper (shape: [samples])
                mono_audio = audio_data.flatten().astype(np.float32) / 32768.0
                result = model.transcribe(mono_audio, fp16=False, language='en')
                return result["text"].strip()
            except Exception as e:
                print(f"⚠️ In-memory transcription failed: {e}")

        # Fallback to file-based transcription
        result = model.transcribe(
            path,
            fp16=False,
            language='en',
            initial_prompt="Testing concepts, software terms, educational content"
        )
        return result["text"].strip()
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return ""
    finally:
        try:
            os.remove(path)
        except:
            pass

def select_input_device():
    devices = sd.query_devices()
    print("Available devices:")
    for idx, dev in enumerate(devices):
        print(f"{idx}: {dev['name']} (Input Channels: {dev['max_input_channels']})")

    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"✅ Selecting input device: {idx} - {dev['name']}")
            return idx
    print("⚠️ No suitable input device found. Using default.")
    return None

if __name__ == "__main__":
    print("🎤 Voice Chatbot Ready! (Say 'exit' to quit)\n")
    input_device = select_input_device()
    if input_device is not None:
        sd.default.device = input_device

    while True:
        try:
            audio_result = record_audio()
            if not audio_result:
                print("🔇 Try again...\n")
                continue

            audio_path, raw_audio = audio_result
            user_input = transcribe_audio(audio_path, raw_audio)

            if not user_input:
                print("🤷 No transcription. Try again...\n")
                continue

            print(f"🧑 You said: {user_input}")

            if user_input.lower() in {"exit", "quit", "stop"}:
                print("👋 Bye!")
                break

            reply = ask_groq_llama(user_input)
            print(f"🤖 Bot: {reply}\n")

        except KeyboardInterrupt:
            print("\n👋 Session ended by user")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
