# ghost_results_whisper_with_noise_reduction.py

import os
import numpy as np
from scipy.io import wavfile
import whisper
from jiwer import wer
import noisereduce as nr

def load_audio(audio_path):
    # Ensure the audio file is a WAV file with the correct format
    # WAV format, mono channel, 16kHz sample rate
    rate, data = wavfile.read(audio_path)
    if rate != 16000:
        raise ValueError("Audio sample rate must be 16kHz")
    if len(data.shape) > 1 and data.shape[1] != 1:
        # Convert to mono by averaging channels
        data = np.mean(data, axis=1).astype(np.int16)
    # Save the processed audio (if any changes were made)
    processed_audio_path = "processed_audio.wav"
    wavfile.write(processed_audio_path, rate, data)
    return processed_audio_path

def reduce_noise(audio_path):
    # Read the audio file
    rate, data = wavfile.read(audio_path)
    # If stereo, convert to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Normalize data to range [-1, 1]
    data = data / np.max(np.abs(data))
    # Select a portion of the audio where there is only noise (adjust as needed)
    noise_sample_duration = int(rate * 0.5)  # First 0.5 seconds
    noise_sample = data[:noise_sample_duration]
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise=noise_sample)
    # Scale back to original range
    reduced_noise = reduced_noise * 32767
    # Convert to int16 type
    reduced_noise = reduced_noise.astype(np.int16)
    # Save the filtered audio
    filtered_audio_path = "filtered_audio.wav"
    wavfile.write(filtered_audio_path, rate, reduced_noise)
    return filtered_audio_path

def transcribe_audio_whisper(audio_path, model_size='base', apply_confidence_filter=False, confidence_threshold=0.60):
    # Load the Whisper model
    model = whisper.load_model(model_size)
    # Transcribe the audio
    result = model.transcribe(audio_path, fp16=False)
    # Process the transcription
    transcript = ''
    for segment in result['segments']:
        text = segment['text']
        # Simple confidence estimation using average probability
        avg_logprob = segment.get('avg_logprob', 0)
        confidence = np.exp(avg_logprob)
        if apply_confidence_filter and confidence < confidence_threshold:
            transcript += '[inaudible]'
        else:
            transcript += text
    return transcript.strip()

def calculate_metrics(reference_text, hypothesis_text):
    # Calculate Word Error Rate (WER)
    error = wer(reference_text, hypothesis_text)
    return error

def count_ghost_results(transcript):
    # Count the number of '[inaudible]' tokens as ghost results
    return transcript.count('[inaudible]')

def main():
    # Paths to files
    AUDIO_PATH = 'audio_2.wav'  # Path to your WAV audio file
    REFERENCE_TRANSCRIPT_PATH = 'audio_2.txt'  # Path to the reference transcript file

    # Step 1: Baseline Transcription without Enhancements
    print("Performing baseline transcription without enhancements...")
    baseline_transcript = transcribe_audio_whisper(
        AUDIO_PATH, model_size='base', apply_confidence_filter=False
    )
    print("\nBaseline Transcription:")
    print(baseline_transcript)

    # Step 2: Enhanced Transcription with Noise Reduction and Confidence Filtering
    # 2a: Apply Noise Reduction to the audio file
    print("\nApplying noise reduction to the audio file...")
    enhanced_audio_path = reduce_noise(AUDIO_PATH)
    print("Noise reduction completed.")

    # 2b: Transcribe the enhanced audio with confidence filtering
    print("\nPerforming enhanced transcription with confidence filtering...")
    enhanced_transcript = transcribe_audio_whisper(
        enhanced_audio_path, model_size='base', apply_confidence_filter=True, confidence_threshold=0.75
    )
    print("\nEnhanced Transcription:")
    print(enhanced_transcript)

    # Step 3: Read Reference Transcript from File
    with open(REFERENCE_TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        reference_transcript = f.read().strip()

    # Step 4: Calculate Metrics and Compare
    print("\nCalculating Metrics...")

    # Word Error Rate (WER)
    baseline_wer = calculate_metrics(reference_transcript.lower(), baseline_transcript.lower())
    enhanced_wer = calculate_metrics(reference_transcript.lower(), enhanced_transcript.lower())

    # Ghost Results Count
    baseline_ghost_count = count_ghost_results(baseline_transcript)
    enhanced_ghost_count = count_ghost_results(enhanced_transcript)

    # Display Metrics
    print(f"\nBaseline WER: {baseline_wer*100:.2f}%")
    print(f"Enhanced WER: {enhanced_wer*100:.2f}%")
    print(f"\nBaseline Ghost Results Count: {baseline_ghost_count}")
    print(f"Enhanced Ghost Results Count: {enhanced_ghost_count}")

    # Improvement Calculation
    if baseline_wer != 0:
        wer_improvement = ((baseline_wer - enhanced_wer) / baseline_wer) * 100
    else:
        wer_improvement = 0

    if baseline_ghost_count != 0:
        ghost_reduction = ((baseline_ghost_count - enhanced_ghost_count) / baseline_ghost_count) * 100
    else:
        ghost_reduction = 0

    print(f"\nWER Improvement: {wer_improvement:.2f}%")
    print(f"Ghost Results Reduction: {ghost_reduction:.2f}%")

if __name__ == '__main__':
    main()
