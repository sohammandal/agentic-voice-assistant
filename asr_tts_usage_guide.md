# ASR & TTS Modules - Simple Setup



## Usage

### ASR (Speech → Text)

```python
from src.asr.processor import ASRProcessor

# Initialize the ASR
# Specify whisper model size; small should be sufficient
asr = ASRProcessor(model_name="small")

# Use encode_audio to get text from audio
text = asr.encode_audio("user_question.wav")
print(text)
```

### TTS (Text → Speech)

```python
from src.tts.processor import TTSProcessor

# Initialize the TTS processor
tts = TTSProcessor()

# Text set manually here, but will come from the agent
text = "If you are hearing this spoken aloud, then I am working properly."

# Use decode_audio to convert text to audio
# Uses random voice via random speaker embeddings
audio_path = tts.decode_audio(text)
print(audio_path)  # Returns audio as response.wav
```
