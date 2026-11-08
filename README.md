# Open-Set Face Recognition: a Face-Gated Building Entrance

A face-recognition engine for a building door: it must **open for enrolled members**, naming the right person,
and **stay closed for everyone else**, including people it has never seen. Built with Siamese networks, triplet
loss and a pair classifier in PyTorch, evaluated on a hardened version of *Labeled Faces in the Wild* (LFW),
and finished with a cost-based deployment analysis.

> Coursework for **CMPS 364 – Advanced Machine Learning** (American University of Beirut), Assignment 4, Part II.
> The data pipeline, trainer, scoreboard and check functions in `helpers/` were provided by the course staff;
> they are included so the notebook can be run, and remain the course staff's work (see [Acknowledgements](#acknowledgements)).

---

## Results at a glance

Final model (`E_final`) vs the course baselines, on the **test** set (people never seen in training;
thresholds chosen on validation only):

| Model | V-F1 | AUC | EER ↓ | TAR @ FAR 1% | O-F1 | DIR @ FAR 1% | **S** |
|---|---|---|---|---|---|---|---|
| Softmax classifier (A) | 0.626 | 0.874 | 0.204 | 0.219 | 0.241 | 0.065 | 0.433 |
| Pair classifier (C) | 0.677 | 0.905 | 0.165 | 0.095 | 0.134 | 0.015 | 0.405 |
| Triplet, semi-hard (best baseline, B2) | 0.720 | 0.929 | 0.146 | 0.355 | 0.329 | 0.109 | 0.524 |
| **Final model (E_final)** | **0.794** | **0.956** | **0.112** | **0.527** | **0.452** | **0.268** | **0.623** |

*S = ½ (V-F1 + O-F1): the average of the verification F1 and the open-set (door) F1.*

- **S +0.10** over the best baseline, and **O-F1 (the door) +37%**.
- With the door set so only 1% of strangers get in, it admits **2.5× more members** correctly (DIR 0.11 → 0.27).
- The improvement is real, not test-set luck: a person-level bootstrap gives ΔAUC vs B2 = 0.027, 95% CI [0.019, 0.035].
- Trained with **only 17% of the allowed compute budget** (0.51 M of 3.07 M training images), because longer training overfit.

---

## The problem

- **Data.** LFW: 13,233 photos of 5,749 people. Train, validation and test contain **different people**
  (1,176 / 168 / 336 identities), so every evaluation is on unseen faces. People with a single photo
  (4,069) act as extra negatives in training and as **strangers** at the door.
- **Made harder on purpose.**
  - About 40% of validation/test photos are degraded (blur, low resolution, noise, sunglasses, face masks, lighting, grayscale).
  - Half of all verification pairs are adversarially "hard": same-person photos that look different, and look-alike strangers.
  - Test pairs are only 20% same-person, versus 50% on validation (a prior shift).
- **Two tasks.**
  - *Verification:* are these two photos the same person?
  - *Open-set identification (the door):* compare a probe photo against a gallery of one clean ID photo per member,
    then admit the nearest member if the distance is below a threshold τ, or reject the probe as a stranger.
- **Constraints.** Only torchvision ImageNet-1k weights: no face-pretrained models and no external face data.
  Nothing is fitted on test data, and the final run has a fixed training-image budget.

## Method

An embedding network maps each face to a unit vector. The same network embeds every photo (a Siamese network),
and faces are compared by squared Euclidean distance.

```
photo (144 px, resized) → CNN backbone (ImageNet-1k) → Linear → BatchNorm → L2-normalise → embedding f(x)
```

Implemented from scratch in this project:

- **Losses**
  - Vectorised **triplet loss** with four mining strategies (random, semi-hard, batch-hard, batch-all),
    with distractors handled as negatives only.
  - A DeepFace-style **pair classifier head** with balanced binary cross-entropy.
- **Metrics**, all O(n log n) and in NumPy:
  - F1 sweep with sample weights,
  - ROC / AUC / EER / TAR@FAR,
  - open-set decision, O-F1, best threshold, DIR@FAR,
  - Platt calibration (Newton's method) and prior shift.
- **Analysis tools**
  - Gallery-size scaling of false accepts,
  - person-level (clustered) bootstrap with design effect,
  - AS-norm score normalisation,
  - cost-based operating points.

---

## Controlled experiments (Parts 3–6)

Every baseline uses the same ResNet-50 and changes **one** thing at a time:

| Experiment | What changed | Finding |
|---|---|---|
| A | softmax classification over the training people | the classifier's own confidence is a weak stranger detector; the embedding below it is a decent baseline |
| B1–B3 | mining: random / semi-hard / hard | **semi-hard wins** (S 0.524 vs 0.468 for hard); random mining leaves only ~6% of triplets active |
| B4 | no L2 normalisation | unbounded distances make training less stable |
| B5 | margin 0.1 / 0.3 / 0.9 | differences are within bootstrap noise (person-level CI of ΔAUC includes 0) |
| B6 | clean augmentation only | short-run comparison; see the challenge, where removing degradation augmentation clearly hurts |
| D1–D2 | 10% flipped labels | hard mining is hit much harder than semi-hard (S 0.359 vs 0.461): batch-hard selects the mislabelled photo as its "hardest positive" |
| C | pair classifier | strong pair AUC, but its head is not a distance and scales poorly as a door |

## Challenge: a model search in five rounds (Part 7)

Goal: maximise S under a fixed budget, choosing **everything on validation**. Each round changes one thing
against the current best and keeps a change only if it beats the noise level (seed-to-seed spread ≈ ±0.005 in S).
Scores below are **validation S** at 4,000 steps unless noted.

**Round 1: loss** (ResNet-50). Reference: semi-hard triplet (B2) 0.540.

| CosFace + triplet | ArcFace + triplet | CosFace only |
|---|---|---|
| 0.467 | 0.468 | 0.458 |

Angular-margin losses *lost*. With about 4.8 photos per identity, the classification term learns slowly and
pulls the embedding toward memorising training people. **Keep semi-hard triplet.**

**Round 2: one change at a time** (base = semi-hard triplet, ResNet-50, 0.538)

| Change | ConvNeXt-Tiny | ResNet-101 | 512-d embedding | 8k steps | + light CosFace | clean augmentation |
|---|---|---|---|---|---|---|
| val S | **0.627 (+0.09)** | 0.580 | 0.563 | 0.556 | 0.512 | 0.452 |

The backbone is by far the biggest lever. ConvNeXt-Tiny is the newest allowed architecture, and at about the same
cost as ResNet-50 it beats the twice-as-deep ResNet-101. Removing the degradation augmentation is clearly harmful.

**Round 3: combine winners, tune margin and length** (ConvNeXt-Tiny, 512-d)

| margin 0.2 | 0.3 | **0.4** | 0.4 @ 6k | 0.4 @ 8k | 0.4 @ 12k |
|---|---|---|---|---|---|
| 0.624 | 0.627 | **0.633** | 0.632 | 0.637 | 0.612 |

The margin hardly matters. Training to 12k steps **overfits**: `frac_active` collapses and the door gets worse.

**Round 4: generalisation tricks** (each alone vs 0.633, then the winners combined)

| RegNetY-3.2GF | random erasing | **K=3, P=32** | 64 distractors | weight decay 0.05 | **160 px** | **K=3 + 160 px** |
|---|---|---|---|---|---|---|
| 0.560 | 0.606 | **0.648** | 0.625 | 0.632 | ≈0.64 | **0.651–0.653** |

**K = 3 photos × 32 people per batch** helps. It reduces duplicated positives: 64% of training people have fewer
than 4 photos, so K = 4 often repeats the same image. A **higher input resolution** (160 px) also helps.

**Round 5: resolution, length and evaluation tricks**

- 192 px and 224 px: ≈0.645, no better than 160 px (upsampling the stored 144 px images adds no detail).
- 8k steps: no gain.
- Multi-scale test-time augmentation: borderline (+0.005), not adopted.
- Within-class whitening fitted on training embeddings: hurt.
- A 2-model ensemble gained ≈+0.007 to +0.024 but was **not used**, to keep strictly to "one final run".

**Final recipe:** ConvNeXt-Tiny (ImageNet-1k) · 512-d · semi-hard triplet, margin 0.4 · P = 32, K = 3 + 32
distractors · 160 px · degradation augmentation · 4,000 steps · single model.

---

## Robustness and statistics

- **Degradations (Q8).**
  - Test AUC falls from 0.972 (both photos clean) to 0.949 (one degraded) to 0.922 (both degraded).
  - Degradation mostly lowers the **same-person** similarity (mean cosine 0.53 → 0.41), while different-person
    similarity barely moves (0.03 → 0.05).
- **Score normalisation.** AS-norm with a training cohort raised validation V-F1 (0.851 → 0.882) but not test V-F1
  (0.794 → 0.780), so a gain tuned on validation did not transfer.
- **Are the differences real? (Q9)** The 12,000 test pairs come from only 336 people.
  - A person-level bootstrap gives a design effect of **≈3.7**, so the pairs are worth only about **3,300** independent ones.
  - Under that bootstrap, E_final > B2 > B3 hold; the B5 margins cannot be ranked.

## Deployment analysis: should the door ship? (Part 8)

A co-working space with 48 members, 2,500 entries a day and 30% strangers. Admitting a wrong person costs $250;
sending a member to the front desk costs $5. Thresholds are chosen on validation and evaluated on test:

| Operating point | Test cost / day (95% CI) | Strangers admitted | Members sent to the desk |
|---|---|---|---|
| τ minimising cost | $7,559 [6,513, 8,587] | 1.2% | 61% |
| τ maximising F1 | $21,321 [14,311, 31,016] | 5.5% | 43% |
| τ for "≤ 1 stranger / 30 days" | $6,945 [6,078, 7,799] | 0.6% | 66% |

- The data-protection rule (false-accept rate ≤ 4.4 × 10⁻⁵) **cannot be verified**: it would need about
  67,500 stranger probes, and only 2,748 exist.
- **Doubling the membership** multiplies the expected cost by about 1.34.
- **Conclusion:** without face-specific pretraining, this model is not reliable enough to be the only key.
  It needs a second factor (a badge or PIN) or a desk fallback.

---

## Repository contents

| File | |
|---|---|
| `A4_PartII_Face_Recognition.ipynb` | the full notebook with all outputs: implementations, experiments, the challenge rounds and the analysis |
| `README.md` | this file |
| `helpers/` | course-provided modules (`gate_data`, `gate_train`, `gate_eval`, …): data pipeline, trainer, scoreboard, checks, figures |
| `LICENSE` | MIT licence (covers my own code only, not `helpers/`) |

Not included: the LFW data (downloaded automatically by scikit-learn on first run) and the model checkpoints.

**To run:** open the notebook on a CUDA GPU (e.g. Google Colab), keep `helpers/` next to it (or in Google Drive),
set `AUBNET` / `FULL_NAME`, and run all cells. `SMOKE = True` gives a fast CPU debug run.

## Possible extensions

- **A face-specific backbone:** an ArcFace-style model trained on a large face dataset (licence permitting).
- **A demo UI:** a Gradio or Streamlit app to enrol a face from the webcam and test the door, with a threshold slider showing the cost trade-off.
- **Liveness / anti-spoofing:** detecting photo attacks.
- **Bias and privacy:** demographic evaluation on a balanced benchmark, and data-retention policies for biometric data.

## Acknowledgements

- Assignment design and the `helpers/` modules (data pipeline, trainer, scoreboard, check functions, figures):
  **CMPS 364 course staff, AUB**. They are included for reproducibility and are **not** covered by this repository's licence.
- Data: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) (Huang et al., 2007).
- Key ideas:
  - FaceNet (Schroff et al., 2015),
  - *In Defense of the Triplet Loss* (Hermans et al., 2017),
  - ArcFace (Deng et al., 2019),
  - ConvNeXt (Liu et al., 2022),
  - AS-norm score normalisation (Auckenthaler et al., 2000).

## Licence

MIT, © 2026 Ward Hashish, for my own code in this repository. The `helpers/` modules and the assignment text
belong to the CMPS 364 course staff and are excluded from this licence.
