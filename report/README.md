# Report workspace

Drafts, notes and anything else for the PDF report go here. The final file has to follow
the `<groupno>_A03_EN3150` format: put in our group number, not the group name.

## Where to find things for each section

| Report section | Pull from |
|---|---|
| 1. Dataset | `outputs/results/dataset_summary.json`, `outputs/figures/dataset_samples.png` |
| 2. Architecture (20 marks) | `docs/architecture.md`, output of `scripts/02_model_summary.py` |
| 3. Optimizer (15 marks) | `outputs/figures/optim_*.png`, `outputs/results/optimizer_study_model_b.json` |
| 4. Custom models (25 marks) | `outputs/figures/model_*_loss.png`, `model_*_confusion.png`, `custom_models_curves.png`, first table in `comparison.md` |
| 5. SOTA (20 marks) | `outputs/figures/mobilenet_v2_*`, `shufflenet_v2_*`, second table in `comparison.md` |
| 6. Trade-offs (20 marks) | `outputs/figures/accuracy_vs_size.png`, both tables |

## Things worth discussing (marks go to interpretation, not just numbers)

- Which classes get confused, and why it makes sense (e.g. Highway vs River are both long, thin lines; AnnualCrop vs PermanentCrop look alike from above)
- How much accuracy Model B gave up for its ~6x smaller size and ~8x fewer MACs
- Whether Model B's training time per epoch actually dropped as much as its MACs did (on a GPU it usually doesn't: depthwise convs are memory-bound, not compute-bound)
- Momentum: does 0.99 overshoot or oscillate? How many epochs does each run need to reach 85% validation accuracy?
- Pre-trained vs from scratch: is the accuracy gain worth 20-35x the memory on a microcontroller?

## Checklist before submitting

- [ ] Group number, names and index numbers on the cover
- [ ] GitHub link included
- [ ] Every figure has a caption and is mentioned in the text
- [ ] Dataset sheet updated
- [ ] Code zipped separately with the same `<groupno>_A03_EN3150` name
