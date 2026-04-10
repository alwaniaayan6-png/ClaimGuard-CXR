# Cross-Dataset Transfer Results

Trained on CheXpert Plus, evaluated on OpenI (Indiana U) zero-shot.

| Metric | CheXpert Plus | OpenI | Δ |
|---|---:|---:|---:|
| Accuracy | 0.9831 | 0.8509 | -0.1322 |
| Macro F1 | 0.9808 | 0.8307 | -0.1501 |
| Contradicted F1 | 0.9743 | 0.7723 | -0.2021 |
| ECE (calibrated) | 0.0066 | 0.0279 | +0.0212 |
| Temperature | 1.1555 | 1.5000 | — |

## Conformal FDR Control (inverted cfBH)

| α | CheXpert n_green | CheXpert FDR | CheXpert Power | OpenI n_green | OpenI FDR | OpenI Power |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 9,935 | 0.0130 | 0.9806 | 501 | 0.0040 | 0.4158 |
| 0.10 | 10,204 | 0.0248 | 0.9951 | 712 | 0.0154 | 0.5842 |
| 0.15 | 10,348 | 0.0363 | 0.9972 | 888 | 0.0304 | 0.7175 |
| 0.20 | 10,577 | 0.0559 | 0.9986 | 991 | 0.0565 | 0.7792 |