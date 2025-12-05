"""
This module contains all of the TTS (Text to Speech) code 
for converting text to speech using SpeechT5 from Hugging Face.
"""

# Imports
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf


# Create custom class for easy usage
class TTSProcessor:
    """Text-to-Speech using SpeechT5."""
    
    def __init__(self):
        """Initialize TTS models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TTS models...")
        
        # Load the processor to handle text processing
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        
        # Load the model to handle text and speech embeddings
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        
        # Load the vocoder to handle speech to waveform conversion
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Create default speaker embeddings (standard voice)
        self.speaker_embeddings = torch.randn(1, 512).to(self.device)
        
        print("TTS ready!")
    
    def decode_audio(self, answer: str) -> torch.Tensor:
        """Decode text to audio waveform."""

        # Assign the audio decoder inputs
        inputs = self.processor(
            text= answer,
            return_tensors= "pt"
        ).to(self.device)
        
        # Convert to waveform
        with torch.no_grad():
            waveform = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings= self.speaker_embeddings,
                vocoder= self.vocoder
            )
        
        return waveform


if __name__ == "__main__":
    tts = TTSProcessor()
    print("TTS module ready!")
