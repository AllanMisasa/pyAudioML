import whisper
import torch
import numpy as np

'''
Internally, the transcribe() method reads the entire file and processes 
the audio with a sliding 30-second window, 
performing autoregressive sequence-to-sequence predictions on each window.
'''
'''
Available models are:
- "tiny" 39M parameters vram ~ 1GB tiny.en or tiny which is multilingual
- "base" 74M parameters vram ~ 1GB base.en or base which is multilingual
- "small" 244M parameters vram ~ 2GB small.en or small which is multilingual
- "medium" 769M parameters vram ~ 5GB medium.en or medium which is multilingual
- "large" 1.55B parameters vram ~ 10GB large is only multilingual
'''


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def base_transcription(model: str, audio_file_path: str):
    model = whisper.load_model(model)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    result = model.transcribe(audio_file_path)
    with open("transcription.txt", "w") as text_file:
        text_file.write(result["text"])
    return result["text"]

def alt_transcription(model: str, audio_file_path: str):
    model = whisper.load_model(model)
    audio = whisper.Audio.from_file(audio_file_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    print(result.text)

base_transcription("tiny.en", "Transcription/QuakeCon2013.mp3")