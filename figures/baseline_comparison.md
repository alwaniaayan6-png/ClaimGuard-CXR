# Baseline Comparison (binary hallucination detection)

| Method | Accuracy | Macro F1 | Contra F1 | Contra Prec | Contra Rec | AUROC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based (oracle ev.) | 0.6634 | 0.5478 | 0.3191 | 0.4899 | 0.2366 | 0.5569 | — |
| Untrained RoBERTa-large | 0.6667 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | — | 0.1348 |
| Zero-shot LLM judge (n=1500) | 0.6653 | 0.5048 | 0.2229 | 0.4932 | 0.1440 | 0.5739 | — |
| CheXagent-8b (VLM, image+text) | 0.6721 | 0.4019 | 0.0000 | 0.0000 | 0.0000 | 0.4996 | — |
| Trained binary v2 (oracle ev.) | 0.9831 | 0.9808 | 0.9743 | 0.9851 | 0.9638 | 0.9952 | 0.0066 |
| Trained binary v2 (retrieved ev.) | 0.9809 | 0.9784 | 0.9711 | 0.9791 | 0.9632 | — | 0.0060 |