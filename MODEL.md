# Two-tower taste model

Learns what each member likes. Members and titles are each turned into a 64-number vector, and a member's best matches are the titles whose vectors score highest with theirs.

- **Title tower:** the catalog description (TF-IDF) plus genre, type, maturity rating, release year and runtime go through a small neural network. It uses no title ID, so a brand-new title still gets a vector.
- **Member tower:** the average vector of the titles the member watched, plus their device, country and kids-profile mix, goes through a small neural network. While training, the viewing being predicted is left out of that average, so the model can't copy the answer.
- **Training:** a softmax over the whole catalog (about 300 titles, so it is exact), trained only on `data/clean/train.csv`.

## Score: Hit Rate@10 on the hidden final week (10 to 16 Nov, 358 events)

Hit Rate@10 is the share of viewings where the title the member actually watched is in their top 10.

| | Hit Rate@10 |
|---|---|
| **Two-tower model** (mean of 5 seeds, ±0.010) | **0.545** |
| Most-popular-titles baseline | 0.223 |
| Random | 0.028 |

The model beats popularity by +0.321 (95% interval +0.246 to +0.395, bootstrapped over members), so the gain is not luck.

**Second score, new titles only.** 104 of the 358 events are re-watches, usually the next episode of a series, and the main score lets the model recommend these. Counting only the 254 events where the member had not watched the title before, and hiding what they had already seen: model **0.667**, popularity **0.224**, random 0.039.

## How the training rounds were chosen
The final week was never used for tuning. Tuning trained on events before 3 Nov and scored 3 to 9 Nov, trying 16 settings with 3 seeds each. The winner was learning rate 0.001, weight decay 1e-4, temperature 0.2, no member-ID embedding, **9 rounds (epochs)**. On that tuning week it scored 0.581 against popularity's 0.198. The model was then refit on all training data with those settings and scored once on the final week. Adding a member-ID embedding did not help, since members have only about 7 viewings each.

## Files
- `train_two_tower.py` runs the whole thing (tuning, final fit, scoring) in about 6 minutes.
- `model/two_tower.pt` holds the weights (seed 0). The reported score is the mean of 5 seeds; seed 0 scored 0.542.
- `model/title_embeddings_64d.csv` and `model/member_embeddings_64d.csv` hold the 64-number vectors.
- `model/metrics.json` and `model/tuning_results.json` hold all the numbers above.

## Weights & Biases
Copy `.env.example` to `.env` and put your key after `WANDB_API=` (`WANDB_API_KEY` also works). `.env` is git-ignored. With a key, runs go online to the `netflix-two-tower` project. Without one they are saved offline, and `wandb sync wandb/offline-run-*` uploads them later.

## Caveats
- Final-week score is from 358 events, so expect roughly ±0.03 of noise; the seed spread (±0.010) is smaller than that.
- Ties in a ranking are counted in the model's and the baseline's favour, which if anything helps popularity.
