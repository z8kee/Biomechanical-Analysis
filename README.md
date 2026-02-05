# Athletic Analysis - PaceyAI

## Project Overview
PaceyAI is a biomechanical analysis toolkit for sprinting:
- Extracts pose landmarks from videos using MediaPipe and windows them into fixed-size samples (`.npy`).
- Runs a temporal convolutional neural network `PhaseClassifier` to label short windows of a run into phases (start, acceleration, max_velocity, deceleration, transition).
- Provides a simple Streamlit app to upload a video, runs feature extraction, visualises a phase timeline, and an optional chatbot that gives more precise feedback and improvements on form.
- Genuinely made because I don't have a consistent sprint coach and I have competitions coming up (02/02/26).

## Key files
- [scripts/app.py](scripts/app.py): Streamlit web app (upload video → extract features → run model → show results + optional precise feedback).
- [scripts/featextract.py](scripts/featextract.py): Extracts pose landmarks from videos and windows them into `.npy` samples.
- [scripts/phaseclassifier.py](scripts/phaseclassifier.py): PyTorch `PhaseClassifier` model and dataset class to turn samples into tensors.
- [scripts/biomechanics.py](scripts/biomechanics.py): Set of rules to display flags on form imperfections which can be parsed into a GPT api for more detailed feedback.
- `models/phase_classifier.pth`: Trained PyTorch model weights.
- `models/pose_landmarker_full.task`: MediaPipe pose landmarker model asset used by `featextract.py`.
- `data/`: Storage for videos, generated `.npy` windows and `metadata.csv`.

## Data layout
- `data/video/`: place source `.mp4 / .mov / .mpeg4` or other supported videos here for feature extraction.
- `data/npy/`: feature windows saved as `sample{ID}.npy` by `featextract.py`.
- `data/metadata.csv`: CSV mapping `sample_id` → `video_file` and frame range (used when building datasets).

## Requirements
This project uses Python and the following packages (approx):
- Python 3.8+
- streamlit
- torch (use the appropriate installation for CPU or CUDA)  
- numpy
- pandas
- opencv-python
- mediapipe
- requests

If you have a GPU, install the `torch` wheels matching your CUDA version (see https://pytorch.org).

## How to run
1. Make sure `models/phase_classifier.pth` and `models/pose_landmarker_full.task` exist. The Streamlit app and feature extractor expect these paths.

2. Run the Streamlit app:

```bash
streamlit run scripts/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and use the UI to upload a video and click **Analyse Sprint**. Additionally,  you can click **More Feedback** for... more feedback.

## Details
- The Streamlit app runs `extract_features_for_app()` from `scripts/featextract.py` to produce a torch tensor of windows for the uploaded video.
- Videos from `data/video/` are read and processed with Mediapipe, windows the frames, and save `sample{ID}.npy` files into `data/npy/`. It also writes rows to `data/metadata.csv` describing each sample.
- It loads the PyTorch model from `models/phase_classifier.pth`, runs the model, and converts predicted class indices to labels using the mapping in `scripts/app.py`.
- Once the phases are classified, the `biomechanics.py` script checks if the landmarks create angles or satisfy conditions upon conditions.
- The app shows a phase timeline (time ranges per window), a bar chart of phase distribution, and a list of flags that were detected during the run.
- An additional button labelled 'More Feedback' parses the flags into a chatbot API from `bot.py` and delivers more feedback.

## Notes & troubleshooting
- MediaPipe model: `models/pose_landmarker_full.task` must be accessible. If `featextract` fails to create the landmarker, it returns `(None, 0, 0)` - the app checks for this and shows an error.
- FPS detection: If OpenCV reports `fps == 0`, the code defaults to `30` fps.
- GPU: If you want to use CUDA for inference, ensure `torch` is installed with CUDA support and that `torch.cuda.is_available()` returns `True`.
- If the Streamlit app reports `Model file not found`, make sure the path `models/phase_classifier.pth` exists.

## Extending / training
- `scripts/phaseclassifier.py` contains the model class `PhaseClassifier` and a simple `PoseDataset` for loading `.npy` windows with labels from a metadata CSV. If you want to add more data, start from the biggest sample id and run `featextract.py`, label windows with appropriate phases as shown in [phaseclassifier.ipynb cell 2](scripts/phaseclassifier.ipynb).

https://github.com/z8kee/Biomechanical-Analysis/blob/main/athlete.mp4
