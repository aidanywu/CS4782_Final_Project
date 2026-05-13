# Small-Scale InstructGPT: Reproducing RLHF with DistilGPT2

This repository contains a CS 5782 final project re-implementing the core ideas from **Training Language Models to Follow Instructions with Human Feedback** by Ouyang et al. The project reproduces the InstructGPT alignment pipeline at a smaller scale using DistilGPT2, public preference datasets, Hugging Face Transformers, TRL, and PyTorch.

## Introduction

InstructGPT showed that language models can better follow user intent when trained with supervised demonstrations and human preference feedback. Our project tests the same idea with an 82M-parameter DistilGPT2 model: supervised fine-tuning (SFT), reward modeling, PPO-based RLHF, and a Direct Preference Optimization (DPO) extension.

## Chosen Result

We focused on the paper's headline result from Figure 1: a smaller aligned InstructGPT model can be preferred over a much larger unaligned GPT-3 model. Since we could not reproduce the original human-labeler study, we tested the same claim at small scale by comparing SFT, PPO-aligned, and DPO-aligned versions of DistilGPT2.

## GitHub Contents

- `code/CS_5782_InstructGPT_RLHF_with_PPO_and_DPO.ipynb`: full reimplementation notebook.
- `data/`: dataset staging directory; datasets are loaded from Hugging Face in the notebook.
- `results/`: generated figures of results.
- `poster/`: final poster PDF for the in-class presentation.
- `report/`: final project report PDF.
- `final_project_instructions.pdf`: course instructions for the deliverables.

## Re-implementation Details

The notebook fine-tunes DistilGPT2 on Stanford Alpaca instruction-response examples, trains a DistilGPT2 reward model on 5,000 Anthropic Helpful-Harmless preference pairs, runs PPO with the SFT model as policy/reference, and runs DPO on a copy of the fine-tuned model using UltraFeedback Binarized. Evaluation uses reward-model scores, pairwise win rates, response lengths, and qualitative inspection of responses to a few select prompts.

## Reproduction Steps

Use Google Colab with a GPU for the closest reproduction path.

```bash
pip install transformers trl datasets accelerate evaluate peft torch matplotlib python-pptx
```

Then open `code/CS_5782_InstructGPT_RLHF_with_PPO_and_DPO.ipynb`, mount Google Drive if using the saved-path cells, and run the notebook sections in order: setup, SFT, reward model, PPO, DPO, evaluation, and plot generation. The notebook saves intermediate model checkpoints to paths such as `/content/drive/MyDrive/distilgpt2-sft`, so update those paths if running locally.

## Results and Insights

Preference optimization improved reward-model scores over SFT. The average reward scores were **SFT: 0.4592**, **PPO/RLHF: 0.6896**, and **DPO: 0.5965**. PPO beat SFT on 80% of prompt-level reward comparisons, while DPO beat SFT on 70%. Qualitatively, aligned outputs became more instruction-shaped, but DistilGPT2 still hallucinated, repeated text, and often improved style more than factual reliability.

## Conclusion

The project reproduced the mechanics and direction of InstructGPT-style alignment, but not the original paper's scale or human-evaluation quality. The main lesson is that RLHF is a multi-stage system whose success depends heavily on model capacity, preference data quality, reward calibration, and evaluation design.

## References

- Long Ouyang et al. 2022. *Training language models to follow instructions with human feedback*. arXiv:2203.02155.
- Rafael Rafailov et al. 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290.
- Stanford Alpaca, Anthropic Helpful-Harmless RLHF, UltraFeedback Binarized, Hugging Face Transformers, TRL, and PyTorch.

## Acknowledgements

This work was completed by Gokulanath Mahesh Kumar and Aidan Wu for Cornell CS 5782: Graduate Deep Learning. We thank the course staff and project reviewers for feedback on the poster and reproduction scope.
