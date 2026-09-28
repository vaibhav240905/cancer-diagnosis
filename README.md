# Cancer Diagnosis

## Requirements

* Python 3.10
* PyTorch 2.2
* Dependencies as specified in `requirements.txt` (OpenCV, Albumentations, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

No specific environment variables are required for localhost execution.
Make sure the required model weights (`unet_best.pth`, `efficientnet_b0_roi_best.pth`, `efficientnet_b0_full_best.pth`) are placed in the `checkpoints/` directory.

## Run Locally

```bash
source .venv/bin/activate
streamlit run app.py
```

## Open in Browser

Open the following URL in your browser:
http://localhost:8501

## Troubleshooting

* If you get `FileNotFoundError` for the model weights, ensure they are placed inside the `checkpoints/` folder.
* If port 8501 is already in use, Streamlit will automatically try 8502. Check the terminal output for the correct URL.
