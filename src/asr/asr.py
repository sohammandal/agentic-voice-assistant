"""
This module contains all of the ASR (Automatic Speech Recognition) 
code for converting speech to text using the Whisper model from OpenAI.
"""

# Imports
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np


# Create custom class for easy usage
class ASRProcessor:
    """Automatic Speech Recognition using Whisper."""
    
    def __init__(self, model_name: str = "small"):
        """Initialize ASR with Whisper model."""
        print(f"Loading Whisper model: {model_name}...")
        self.asr_model = whisper.load_model(model_name)
        print("ASR ready!")
    
    def record_audio(self, duration: int = 5, sample_rate: int = 16000, output_file: str = "recorded_question.wav") -> str:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds (default: 5)
            sample_rate: Sample rate in Hz (default: 16000)
            output_file: Path to save the recording (default: "recorded_question.wav")
        
        Returns:
            Path to the saved audio file
        
        Example:
            >>> asr = ASRProcessor()
            >>> audio_file = asr.record_audio(duration=5)
            >>> result = asr.encode_audio(audio_file)
            >>> print(result['text'])
        """
        print(f"Recording for {duration} seconds...")
        print("Speak now!")
        
        # Record audio
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        print("Recording finished!")
        
        # Save to file
        sf.write(output_file, recording, sample_rate)
        print(f"Saved to: {output_file}")
        
        return output_file
    
    def encode_audio(self, audio_path: str) -> dict:
        """
        Encode an audio to text.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with 'text' and other transcription details
        """
        question = self.asr_model.transcribe(audio_path)
        return question
    
    def record_and_transcribe(self, duration: int = 5) -> dict:
        """
        Record audio and transcribe in one step.
        
        Args:
            duration: Recording duration in seconds
        
        Returns:
            Dictionary with transcription results
        
        Example:
            >>> asr = ASRProcessor()
            >>> result = asr.record_and_transcribe(duration=5)
            >>> print(f"You said: {result['text']}")
        """
        audio_file = self.record_audio(duration=duration)
        result = self.encode_audio(audio_file)
        return result


if __name__ == "__main__":
    asr = ASRProcessor()
    print("ASR module ready!")
