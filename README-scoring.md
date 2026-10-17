# Grammar scoring for spoken audio (SHL Hiring Assessment 2026)

Predicts a grammar score between 0 and 5 for 45 to 60 second speech recordings.

## Result
| | |
|---|---|
| Training RMSE | 0.3294 |
| Cross-validated RMSE (all clips / score > 0 clips) | 0.5140 / 0.5160 |
| Cross-validated Pearson (all clips / score > 0 clips) | 0.911 / 0.862 |
| Kaggle public leaderboard score | 0.3965 |

For comparison, always predicting the mean gives an RMSE of 1.238 on all clips and 1.014 on the clips with a score above 0.

## Approach in short
1. Transcribe each clip with Whisper (small).
2. Text features: counts for fluency and structure, GPT-2 loss, MiniLM sentence embedding.
3. Audio features: simple signal measurements plus embeddings from pretrained speech encoders (Whisper small, medium and large-v3-turbo, and WavLM-large), several layers each, as the average and the spread over time.
4. Ridge regression and SVR on top (text and audio embeddings reduced with PCA), averaged. The final prediction averages three models that use different audio feature sets.
5. Everything is validated with repeated, stratified 5-fold cross-validation, with scaling and PCA fitted inside each fold.

The full explanation, tables and plots are in `grammar_scoring_final_0.3965.ipynb`.

## Files
- `grammar_scoring_final_0.3965.ipynb`: the notebook that produced the submission, with its outputs and the written report.
- `submission.csv`: the predictions that scored 0.3965.

## Running it
The notebook was run on Kaggle (GPU, Internet on) with the competition data attached. It expects the competition folder layout with `train/`, `test/`, `train.csv` and `test.csv`. The audio and the CSV files are not included here because they belong to the competition. The slow steps (transcription and encoder embeddings) save their results to `/kaggle/working`, so rerunning is much faster.

Libraries: pandas, numpy, scipy, scikit-learn, matplotlib, librosa, torch, transformers and sentence-transformers (as installed on Kaggle).
