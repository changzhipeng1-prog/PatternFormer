# preprocess/ — offline data preparation (run once)

Turns raw `data_gen/` output into the training datasets in `../data/*.pt`
(GT solution lookup + train/val/test split). Run once, then never touched again.
This **writes** the dataset files; it is NOT used during training.

Run: `python <script>.py` (defaults read/write `../data`).
