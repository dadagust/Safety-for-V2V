# Safety-for-V2V

1) Install:
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

2) Quick smoke test (no big datasets):
   python scripts/make_toy_dataset.py --out-dir toy_data --n 20
   python app.py det-train --train-csv toy_data/train.csv --val-csv toy_data/val.csv --out-dir runs/toy
   python app.py det-eval --csv toy_data/val.csv --ckpt runs/toy/best.pt --out-dir runs/toy_eval

3) TTS + watermark:
   python app.py tts --text "Hello, this is watermarked." --out out_wm.wav --watermark
   python app.py wm-detect --input out_wm.wav

4) install cuda pytorch

python app.py det-train --train-csv data\asvspoof2019_la_train.csv --val-csv data\asvspoof2019_la_dev.csv --out-dir runs\asvspoof2019_la_cnn

python app.py det-eval --csv data\asvspoof2019_la_eval.csv --ckpt runs\asvspoof2019_la_cnn\best.pt --out-dir runs\asvspoof2019_la_eval

