# Data

This project loads training and preference datasets directly from Hugging Face inside `code/CS_5782_InstructGPT_Final_Project.ipynb`. The full datasets are not committed to this repository because they can be downloaded reproducibly and may be large.

## Datasets Used

- **Stanford Alpaca**: instruction-response examples used for supervised fine-tuning (SFT) of DistilGPT2.
- **Anthropic Helpful-Harmless RLHF**: chosen/rejected preference pairs used to train the reward model; the project used a 5,000-pair subset.
- **UltraFeedback Binarized**: preference pairs used for the Direct Preference Optimization (DPO) experiment.

## How to Reproduce

Install the dataset tooling:

```bash
pip install datasets
```

Then run the data-loading cells in the project notebook. The notebook uses Hugging Face `datasets.load_dataset(...)` calls and performs preprocessing inline before SFT, reward-model training, PPO/RLHF, and DPO.