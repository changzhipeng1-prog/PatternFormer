# code/data/ — online data loading (used every training step)

A Python package (`import data.dataset`) that `train.py` calls inside the training
loop to **read** the prepared datasets in `../data/*.pt` and stream them to the model.
It does NOT create any files — think of it as "data loading", not "data preparation".

- `dataset.py`        — PDEContextDataset + samplers + collate (batching/padding)
- `sequence_builder.py` — builds the Qwen input token sequence from (params, solutions)

The offline dataset-building step lives in `../../preprocess/`.
