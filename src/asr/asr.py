"""
This module contains all of the ASR (Automatic Speech Recognition)
code for converting speech to text using the Whisper model from OpenAI.
"""

# Imports
import whisper


# Create custom class for easy usage
class ASRProcessor:
    """Automatic Speech Recognition using Whisper."""

    def __init__(self, model_name: str = "small"):
        """Initialize ASR with Whisper model."""
        print(f"Loading Whisper model: {model_name}...")
        self.asr_model = whisper.load_model(model_name)
        print("ASR ready")

    def encode_audio(self, audio_path: str) -> dict:
        """
        Encode an audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, m4a, etc.)

        Returns:
            Dictionary with 'text' and other transcription details

        Example:
            >>> asr = ASRProcessor()
            >>> result = asr.encode_audio("question.wav")
            >>> print(result['text'])
        """
        question = self.asr_model.transcribe(audio_path)
        return question


if __name__ == "__main__":
    asr = ASRProcessor()
    print("ASR ready")
