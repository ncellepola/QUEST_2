# Ghost Results Mitigation in Speech-to-Text Transcriptions

This project aims to reduce ghost results—inaccurate or fabricated transcriptions—in speech-to-text (STT) systems by implementing noise reduction and confidence scoring techniques using OpenAI's Whisper model.

## Table of Contents
- [Introduction](#introduction)
- [Features]()
- [Prerequisites]()
- [Installation]()
- [Usage]()
- [Preparing Audio Files]()
- [Creating Reference Transcripts]()
- [Running the Script]()
- [Adjustable Parameters]()
- [Results Interpretation]()
- [Troubleshooting]()
- [Contributing]()
- [License]()

## Introduction
Ghost results in speech-to-text systems are transcriptions that are inaccurate or do not correspond to any spoken words, often caused by background noise or model misinterpretations. This project implements techniques to mitigate ghost results by:

- Performing noise reduction using spectral gating with the noisereduce library.
- Applying confidence scoring to filter out low-confidence transcriptions.
- Comparing baseline and enhanced transcriptions against a reference transcript.
- Calculating metrics such as Word Error Rate (WER) and the number of ghost results.

## Features

- Noise Reduction: Reduces background noise in audio files using spectral gating.
- Confidence Filtering: Filters out low-confidence transcribed words to reduce ghost results.
- Metrics Calculation: Computes WER and ghost results count for both baseline and enhanced transcriptions.
- Model Flexibility: Utilizes OpenAI's Whisper model, with options to select different model sizes.

## Prerequisites
- Python: Version 3.7 or higher.
- FFmpeg: Required by Whisper to process audio files.

## Installation
1. Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/ghost-results-mitigation.git
cd ghost-results-mitigation
2. Create a Virtual Environment (Optional but Recommended)

```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
3. Install Required Packages


```
pip install -r requirements.txt

```

Note: If you have a CUDA-capable GPU and want to use it for faster processing, install the GPU version of PyTorch by following the instructions on PyTorch's website.

4. Install FFmpeg

- macOS (Using Homebrew):

```
brew install ffmpeg
```

- Windows:

Download FFmpeg from the official website.
Add the bin directory to your system's PATH.
- Linux (Ubuntu/Debian):

```
sudo apt update
sudo apt install ffmpeg
```

# Usage
## Preparing Audio Files
- Format: WAV
- Sample Rate: 16,000 Hz (16kHz)
- Channels: Mono
- Convert Audio Files Using FFmpeg:


```ffmpeg -i input_audio_original.wav -ac 1 -ar 16000 input_audio.wav```

Place your audio file (input_audio.wav) in the same directory as the script.

## Creating Reference Transcripts
- Create a text file named reference_transcript.txt containing the exact transcription of your audio file.
- Ensure the transcript matches the audio content accurately for proper WER calculation.

## Running the Script
Execute the script by running:


```python ghost_results_whisper_with_noise_reduction.py```
Script Output:

- Baseline Transcription: STT output without enhancements.
- Enhanced Transcription: STT output after noise reduction and confidence filtering.
- Metrics:
1. Word Error Rate (WER) for both baseline and enhanced transcriptions.
2. Ghost Results Count for both cases.
3. Percentage improvements.

# Adjustable Parameters
- Noise Sample Duration in reduce_noise Function:


```noise_sample_duration = int(rate * 0.5)  # Adjust duration (in seconds)```

Adjust based on where the noise-only segment is in your audio file.

- Confidence Threshold in main Function:

```
confidence_threshold=0.75  # Adjust between 0 and 1 
```

Fine-tune to balance between filtering out ghost results and retaining valid words.
- Whisper Model Size:


```model_size='base'  # Options: 'tiny', 'base', 'small', 'medium', 'large'```

Larger models may improve accuracy but require more resources.


# Results Interpretation
- Word Error Rate (WER):

Lower WER indicates better transcription accuracy.
Calculated by comparing transcriptions against the reference transcript.
- Ghost Results Count:

Number of [inaudible] tokens indicating low-confidence transcriptions.
Reduction signifies fewer ghost results.
- Percentage Improvements:

WER Improvement: Positive values indicate error reduction.
Ghost Results Reduction: Positive values indicate fewer ghost results.
# Troubleshooting
- No Improvement or Worse Results:

    - Adjust Noise Reduction Parameters:
        - Ensure the noise sample contains only noise.
        - Modify noise_sample_duration to capture an appropriate noise profile.

- Experiment with Confidence Threshold:
        
    - Try different values to find the optimal threshold.
- Errors During Execution:

    - Module Not Found: Ensure all dependencies are installed in the active environment.
    - FFmpeg Not Found: Verify FFmpeg is installed and added to your system's PATH.
    - Audio Format Issues: Confirm your audio file meets the required specifications.
- Runtime Warnings:

May occur during noise reduction but usually don't affect execution.
If severe, consider adjusting the noise reduction settings or skipping noise reduction.


