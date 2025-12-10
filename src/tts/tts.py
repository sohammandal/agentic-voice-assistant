"""
This module contains all of the TTS (Text to Speech) code
for converting text to speech using SpeechT5 from Hugging Face.
"""

# Imports
import os

import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor


class TTSProcessor:
    """Text-to-Speech using SpeechT5."""

    def __init__(self, gender="male"):
        """Initializes the TTS models."""

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set the initial voice gender
        self.current_gender = gender
        print(f"Loading TTS models...")

        # Load the text processing processor (SpeechT5Processor)
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

        # Load the model to handle text and speech embeddings
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)

        # Load the vocoder
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(
            self.device
        )

        # Load speaker embeddings
        print(f"Loading speaker embeddings ({gender} voice)...")
        self._load_speaker_embeddings()
        self._set_speaker_embeddings(gender)

        print("TTS ready")

    def _load_speaker_embeddings(self):
        """Load speaker embeddings from local file or fallback to seeded random."""

        embeddings_path = "src/tts/speaker_embeddings.pt"

        if os.path.exists(embeddings_path):
            try:
                embeddings = torch.load(embeddings_path, map_location="cpu")
                self.male_embedding = embeddings["male"].unsqueeze(0)
                self.female_embedding = embeddings["female"].unsqueeze(0)
                print(f"Loaded CMU ARCTIC embeddings")
                return
            except Exception as e:
                print(f"Could not load embeddings: {e}")

        # Fallback: Use consistent seeded random embeddings
        print("Using random speaker embeddings")

        # Male random seeded embedding
        torch.manual_seed(7)
        self.male_embedding = torch.randn(1, 512)

        # Female random seeded embedding
        torch.manual_seed(42)
        self.female_embedding = torch.randn(1, 512)

    def _set_speaker_embeddings(self, gender):
        """Set speaker embeddings based on gender selection."""

        # Set the speaker embeddings based on gender
        if gender.lower() == "female":
            self.speaker_embeddings = self.female_embedding.to(self.device)
            self.current_gender = "female"
        else:
            self.speaker_embeddings = self.male_embedding.to(self.device)
            self.current_gender = "male"

    def change_voice(self, gender):
        """Change the voice gender."""

        print(f"Switching to {gender} voice...")
        self._set_speaker_embeddings(gender)

    def decode_audio(self, answer: str) -> torch.Tensor:
        """Decode text to audio waveform."""

        # Assign the audio decoder inputs
        inputs = self.processor(text=answer, return_tensors="pt").to(self.device)

        # Convert to waveform
        with torch.no_grad():
            waveform = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=self.speaker_embeddings,
                vocoder=self.vocoder,
            )

        return waveform


if __name__ == "__main__":
    tts = TTSProcessor()
    print("TTS module ready")
