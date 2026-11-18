# Prompt used to train the taste model

## 1. The request as given to Claude Code (verbatim)

> Now we'll build a two-tower taste model that learns what each member likes to watch. One side learns about members, and the other learns about titles from their descriptions. Each one gets turned into a list of 64 numbers, so we can match people with shows. We'll check it with Hit Rate, which counts how often the right show lands in someone's top ten. It has to beat just picking the most popular shows. You'll send me one GitHub link with the model and that score. In the bigger picture, these numbers power the top picks every member will see.
>
> Deliverable: Trained taste model. Format: GitHub commit link. Resource: Netflix title catalog, November 2026.
> Tasks: Choose training rounds and Hit Rate@10 scoring; Save the Weights & Biases API key; Write the prompt to train the model.
> Tools: Claude Code, PyTorch, Weights & Biases, Python, GitHub.

Context from the earlier step: the viewing history was already cleaned and split at 2026-11-10 (`data/clean/train.csv`, `data/clean/holdout.csv`).

## 2. The training prompt, written out in full

> Build a two-tower taste model in PyTorch in `train_two_tower.py`, using only `data/clean/train.csv` to learn and the catalog for title text.
>
> **Towers (64-number output each, L2-normalised, scored by dot product)**
> - Title tower: TF-IDF of the catalog description plus genre, title type, maturity rating, release year and runtime, through a small MLP. Use no title-ID embedding, so brand-new titles still get a vector.
> - Member tower: the average title vector of what the member watched (shared title tower), plus device, country and kids-profile mix, through a small MLP. While training, leave the target viewing out of that average so the model cannot copy the answer.
>
> **Loss:** softmax over the whole catalog (about 300 titles, so it is exact) with a temperature, masking titles not yet released on the event date.
>
> **Score: Hit Rate@10** = share of events whose true title is in the member's top 10. The candidates are all catalog titles released by the event date. Re-watches are valid picks, because many are the next episode of a series. Also report a second score on new-to-the-member titles only, with already-watched titles hidden.
>
> **Baselines:** most-popular-titles (counted from train only) and random. The model must beat popularity. Count ties in the model's and baseline's favour, which is conservative for us.
>
> **Choose the training rounds without touching the hidden week.** Tune on train only: fit on events before 2026-11-03 and score on 2026-11-03 to 2026-11-09. Try a grid over learning rate, weight decay, temperature and member-ID embedding on or off, with 3 seeds each. Choose the setting and epoch count from a 3-epoch smoothed validation curve. Then refit on all of train and score holdout (2026-11-10 to 2026-11-16) once, over 5 seeds. Report the mean, the spread, and a member-level bootstrap interval for the lift over popularity.
>
> **Weights & Biases:** log every tuning run and the final run to project `netflix-two-tower`. Read the key from a git-ignored `.env` (`WANDB_API`, or the standard `WANDB_API_KEY`); never commit it. Fall back to offline mode if there is no key.
>
> **Save** to `model/`: the weights, the 64-number member and title vectors, `metrics.json` and `tuning_results.json`. Write `MODEL.md` with the score table, how the rounds were chosen, and caveats. Commit and send one GitHub commit link.

## 3. Result of running it
Holdout Hit Rate@10: **0.545** (mean of 5 seeds) vs **0.223** for most-popular and 0.028 for random. Details are in `MODEL.md` and `model/metrics.json`.
