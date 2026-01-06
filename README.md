# Student Overall Score Predictor (Sklearn + Streamlit)

## Files
- `train.py` : trains a model and saves `model.joblib` + `model_meta.json`
- `app.py`   : Streamlit UI for prediction
- `01_train_and_export.ipynb` : notebook version of training
- `requirements.txt`

## Train
```bash
pip install -r requirements.txt
python train.py --data Student_Performance.csv --mode full --outdir .
# or:
python train.py --data Student_Performance.csv --mode early --outdir .
```

## Run locally
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push these files to a GitHub repo
2. On Streamlit Cloud: create a new app
3. Select `app.py` as the entry point
4. Ensure the repo includes `requirements.txt`
5. Either:
   - Commit `model.joblib` + `model_meta.json` to the repo **after training**, or
   - Train inside the repo (notebook) and commit artifacts.

Tip: If you retrain, replace the two model files.
