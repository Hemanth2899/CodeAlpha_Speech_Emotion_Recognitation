# Speech Emotion Recognition — VS Code Project

A beginner-friendly Python project that classifies emotions in speech using MFCC (Mel-Frequency Cepstral Coefficients) features and a CNN model. It is configured for the RAVDESS **audio-only speech** dataset.

Predicted classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

> Important: the dataset is not bundled inside this ZIP because it is a large third-party dataset. The included `download_dataset.py` downloads it from the official RAVDESS Zenodo record when you run it. Internet access is required for the first dataset download.

## 1. Requirements

- Windows 10/11
- Python 3.11 (recommended for this requirements file)
- Visual Studio Code
- VS Code Python extension
- Internet connection for installing libraries and downloading the dataset

## 2. Open the project

1. Extract the ZIP file.
2. In VS Code, select **File → Open Folder**.
3. Select the extracted `Speech_Emotion_Recognition_VSCode` folder.
4. Select the Python interpreter from `.venv` if VS Code does not select it automatically.

## 3. Set up on Windows PowerShell

Open a new VS Code terminal and run these commands from the project folder:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `Set-ExecutionPolicy` command applies only to the current PowerShell terminal session. Alternatively, choose **Command Prompt** as the VS Code terminal and activate with:

```cmd
.venv\Scripts\activate.bat
```

If `py -3.11` is not recognized, install Python 3.11 and enable **Add Python to PATH** during installation.

## 4. Download the dataset

With the virtual environment active:

```powershell
python download_dataset.py
```

This downloads and extracts the official RAVDESS audio-speech ZIP into `data/ravdess/`. The first download may take several minutes.

Official dataset page: https://zenodo.org/records/1188976

## 5. Train the model

```powershell
python main.py --train
```

Training extracts MFCC features, trains a CNN, prints evaluation metrics, and saves:

- `emotion_model.keras` — trained model
- `labels.json` — model's class order

Training time depends on your computer. The first run processes all the dataset audio files.

## 6. Predict an emotion

Put a WAV recording in the project folder, for example `my_voice.wav`, then run:

```powershell
python main.py --predict "my_voice.wav"
```

You can also provide a full path:

```powershell
python main.py --predict "C:\Users\YourName\Music\my_voice.wav"
```

For best results, use a clear speech recording with one speaker and little background noise. The feature extractor uses the first 3 seconds of the audio.

## 7. Expected output

Exact results vary; an example of the output format is:

```text
Audio file: my_voice.wav
Predicted emotion: HAPPY
Model score: 78.42%

Scores:
  happy       78.42%
  calm        10.20%
  neutral      5.80%
  ...
```

These are example numbers, not guaranteed results. A model score is not a reliable measurement of a person's true emotional state. The model can be wrong, especially on accents, languages, microphones, background noise, or speech unlike the acted training recordings.

## 8. Troubleshooting

### PowerShell says scripts are disabled

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `No module named ...`

Activate `.venv` and run:

```powershell
python -m pip install -r requirements.txt
```

### Dataset folder not found

Run:

```powershell
python download_dataset.py
```

### Model not found when predicting

Train first:

```powershell
python main.py --train
```

### TensorFlow installation fails

Check your version:

```powershell
python --version
```

Use 64-bit Python 3.11, then recreate the virtual environment and reinstall requirements if needed.

### Dataset download fails

Open the official RAVDESS page linked above, download `Audio_Speech_Actors_01-24.zip`, and place the ZIP inside the project's `data` folder. Then run `python download_dataset.py` again.

## Project structure

```text
Speech_Emotion_Recognition_VSCode/
├── main.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── .vscode/
│   └── settings.json
└── data/
    └── README.md
```

The dataset, virtual environment, and trained model are created locally after setup/training and are intentionally excluded from the ZIP.

## Method overview

1. Load each WAV recording.
2. Resample audio to 22,050 Hz and use a 3-second segment.
3. Extract 40 MFCC coefficients.
4. Normalize and pad features to a consistent size.
5. Train a 2D CNN classifier.
6. Evaluate on a held-out test set.
7. Use the saved model to predict an audio recording.

Note: this is an educational baseline. The random train/test split can put recordings from the same actors in both sets, which may overestimate real-world performance. For a research-grade evaluation, split by actor.
