# Final Report: Transformer Algebra for Reverse-Then-Add Composition

## Abstract

This report studies whether small transformer models trained on separate symbolic tasks can be composed to solve a harder task without direct end-to-end training. The experimental setting centers on reverse-then-add: one model learns digit reversal, another learns addition, and the goal is to combine them so that the system can reverse both operands before summing them. Two complementary tracks are examined. The first uses a shared base checkpoint and evaluates weight-space composition through naive task arithmetic, TIES, and DARE. The second studies execution-space composition through a zero-shot pipeline, layer concatenation, adapter-based composition, and a scratchpad-style generalization experiment. The results show a sharp distinction between skill retention and true compositional generalization. TIES and DARE preserve the source tasks well, substantially outperforming naive task arithmetic on reverse and addition accuracy, yet all shared-base merging approaches remain near zero on the composed benchmark. Direct execution-space composition is more promising: a reproduced spot-check shows the zero-shot pipeline outperforming a directly trained composed baseline, although failures remain concentrated around interface mismatches such as leading zeros and incompatible tokenization. Additional diagnostics using task-vector conflict statistics and CKA reveal substantial interference between reverse and addition updates, helping explain why successful merging of individual skills does not translate into reliable sequential composition. Overall, the project provides evidence that model algebra can preserve multiple capabilities in small transformers, but faithful compositional transfer requires more explicit interface alignment than weight-space merging alone provides.

## 1. Introduction

### 1.1 Problem Statement

This project investigates a focused but conceptually important question in neural compositionality: can two small transformer models, each trained on a simple symbolic task, be combined to solve a harder task that neither model was trained to perform end-to-end? The `transformer-algebra` folder studies this question through the concrete program of composing a reversal model and an addition model to solve the reverse-then-add task. In canonical form, the constituent tasks are `1 2 3 -> 3 2 1` and `1 2 3 + 4 5 6 -> 5 7 9`; the composed task requires the model to reverse both inputs before addition, e.g., `1 2 3 + 4 5 6 -> 3 2 1 + 6 5 4 -> 9 7 5`. The experimental suite is broader than a single merge benchmark: beyond TIES and DARE, the folder includes naive task arithmetic, difficulty-stratified composition tests, layer-concatenation and adapter-based composition, task-vector conflict analysis, CKA representation comparison, and a scratchpad-style addition generalization study. Conceptually, the project is grounded in task arithmetic [1], TIES merging [2], and DARE-style sparsified merging inspired by homologous model absorption [3].

### 1.2 Main Claim

The central claim supported by the repository is not that composition is impossible, but that there is a sharp distinction between preserving multiple source capabilities and achieving faithful sequential composition. Across the checked-in experiments, weight-space merging methods such as task arithmetic, TIES, and DARE can preserve reverse and addition performance surprisingly well when the constituent models share a common base. However, these same models fail almost completely on the true composed benchmark. The folder therefore provides a compact empirical case study of a broader phenomenon in modular learning: competence on component tasks does not automatically imply competence at their composition.

## 2. Background and Setup

### 2.1 Experimental Tracks

The repository is organized around two complementary experimental tracks. The first is a shared-base task-vector track. Here, a single 4-layer, 4-head, 128-dimensional nanoGPT model is trained on a mixed corpus and then fine-tuned separately for reversal and addition. This shared representational origin is essential because task-vector methods operate on parameter differences relative to a common checkpoint. Concretely, `config/train_base.py` trains a base model for 20,000 iterations on `data/mixed`, while `config/train_reverse_ft.py` and `config/train_addition_ft_base.py` resume from the base checkpoint and fine-tune for an additional 5,000 iterations each. The resulting models are approximately 0.79M parameters.

The second is a composition-strategy track. In this branch, reverse, addition, and composed models are trained independently from scratch using `composition_strategies/config/train_reverse.py`, `train_addition.py`, and `train_composed.py`. This track asks a different question: if weight-space merging is too brittle, can one instead compose models at execution time or by learning a narrow interface between them? The folder explores three such strategies: a zero-shot pipeline, layer concatenation, and an adapter network.

### 2.2 Data Construction

The data generation pipeline is unusually clean for a class project and deserves emphasis. In `data/prepare_mixed.py`, reversal examples are exhaustively enumerated for 2-, 3-, and 4-digit strings, yielding 11,100 total examples. These are partitioned into 7,000 base-training examples, 2,000 reverse fine-tuning examples, 500 held-out test examples, and 1,600 unused examples. Addition data are sampled from the space of 3-digit plus 3-digit problems, with 10,000 base-training examples, 5,000 fine-tuning examples, and 1,000 held-out test examples. The script explicitly checks for zero overlap between all training data and the held-out tests. In the checked-in data, the shared mixed tokenizer has vocabulary size 15, while the standalone reverse tokenizer has vocabulary size 14 because it does not need the plus symbol. This detail becomes important later, especially for the attempted concat and adapter experiments.

### 2.3 Auxiliary Scratchpad Dataset

For the scratch composition track, `data/prepare_tokenized.py` builds 10,000-example datasets for reverse, addition, and composed tasks, each tokenized independently with a 90/10 train/validation split. The resulting token counts are substantial for this problem scale: `data/reverse/train.bin` contains 166,680 tokens, `data/addition/train.bin` 217,960, and `data/composed/train.bin` 365,040. The composed task is therefore genuinely harder not only semantically but also in sequence length and output structure. A further auxiliary dataset, `data/addition_scratchpad`, trains a model to emit explicit carry traces on 1- and 2-digit addition and then tests it out of distribution on 3-digit addition. This experiment broadens the project from composition alone to the related question of whether explicit intermediate structure improves length generalization.

## 3. Methodology

### 3.1 Shared-Base Task-Vector Methods

The shared-base track implements a classical task-vector framing. Let `theta_base` denote the mixed-task checkpoint, and let `theta_rev` and `theta_add` denote the fine-tuned reverse and addition checkpoints. Task vectors are defined as `tau_rev = theta_rev - theta_base` and `tau_add = theta_add - theta_base`. The repository then studies several ways of combining these vectors:

Naive task arithmetic adds both deltas back into the base model with scaling coefficients [1]. TIES introduces trimming, sign election, and disjoint merging to reduce destructive interference [2]. DARE adds stochastic sparsification to the merge [3]. The code in `ties.py`, `dare.py`, and `task_arithmetic.py` evaluates each merged model on held-out reverse and addition tests and, crucially, on a composite benchmark in which the model must perform reverse-then-add directly from the raw prompt. This gives the study a clear experimental ladder: naive arithmetic serves as the minimal baseline, TIES as a structured conflict-aware merge, and DARE as a sparsified alternative that tests whether random subspace separation alone is enough to retain multiple skills.

### 3.2 Diagnostic Analyses

Additional diagnostic experiments deepen the analysis. `task_vector_conflict.py` measures cosine alignment and sign disagreement between reverse and addition task vectors by module, allowing the report to distinguish where interference is strongest. `cka.py` compares hidden representations across the base, reverse fine-tuned, and addition fine-tuned models, providing a representation-level perspective on why some merges are easier than others. These diagnostics are methodologically important because they move the project beyond leaderboard-style comparison and toward a mechanistic account of failure.

### 3.3 Execution-Space Composition

The composition-strategy track asks whether explicit structural composition can succeed where raw weight-space addition fails. The zero-shot pipeline in `composition_strategies/evaluate_composed.py` first applies a reversal model to each operand and then feeds the resulting strings into a separately trained addition model. Layer concatenation in `train_concat.py` splices early blocks from the reverse model with later blocks from the addition model and fine-tunes the combined network on composed examples. The adapter strategy in `train_adapter.py` freezes both base models and trains only a small bottleneck adapter to map hidden states from one model into the representational space of the other. Finally, `evaluate_addition_scratchpad.py` tests whether making intermediate reasoning explicit in the output sequence yields better extrapolation to longer arithmetic instances.

### 3.4 Evaluation Perspective

This experimental design is well chosen because it separates two hypotheses that are often conflated. The first is capability preservation: can a merged or stitched model still perform both source tasks? The second is compositional transfer: can the system use those capabilities in the correct sequence on novel inputs? The repository shows these are not the same problem.

## 4. Results

### 4.1 Source-Task Retention

The most robust result in the folder is that shared-base merging preserves source-task competence far better than naive averaging. In `ties_results_table.md`, naive merging at full scale yields 0.0% accuracy on both reverse and addition. By contrast, low-lambda TIES models are strong on both tasks: for example, density `0.8` with lambda `0.2` reaches 93.0% reverse accuracy and 100.0% addition accuracy, while density `1.0` with lambda `0.2` reaches 93.0% and 100.0%, respectively. The qualitative pattern is stable across the table: small merge magnitudes preserve task skill, whereas larger lambdas rapidly destroy it.

The task-arithmetic baseline is useful precisely because it fails in an interpretable way. In a direct 200-sample CPU sweep reproduced from `task_arithmetic.py`, single-task scaling behaves sensibly: addition-only performance stays between 97.0% and 100.0% across lambda values, and reverse-only performance stays between 95.0% and 100.0%. But when both task vectors are added simultaneously, performance deteriorates sharply: at lambda `0.2`, the merged model still reaches 91.0% reverse and 94.5% addition, but by lambda `0.4` both tasks fall to roughly 51%, by lambda `0.6` they collapse to about 5%, and at lambda `0.8` and `1.0` both are 0.0%. This is exactly the behavior one would expect if the individual task vectors are meaningful but interfere destructively when superposed without conflict resolution.

DARE shows the same high-level behavior and, in the checked-in no-rescale artifact `dare_results_table_no_rescale.csv`, is even stronger numerically. The best reported setting, drop rate `0.8` with lambda `0.2`, achieves 95.0% reverse and 99.5% addition accuracy. More importantly, DARE degrades more gracefully as lambda increases: at drop rate `0.8`, lambda `0.8` still retains 91.5% reverse and 91.0% addition accuracy. From a model-editing perspective, this is a strong positive result. Shared-base fine-tunes do carry reusable task directions, and sparsified merging can preserve them. Relative to naive task arithmetic, both TIES and DARE therefore appear not merely as incremental improvements but as qualitatively different regimes: they convert catastrophic interference into a controlled trade-off.

### 4.2 Compositional Failure in Weight Space

The central negative result is that source-task retention does not translate into composed-task success. In `composite_benchmark_d08_l02.csv`, the shared-base model, single-task fine-tunes, naive merge, and TIES merge all score essentially zero on reverse-then-add, with accuracies between 0.0 and 1.0%. The more diagnostic stratified benchmark in `stratified_composition_results.csv` sharpens the interpretation. On the easiest level, where reversal is effectively identity for one-digit inputs, models can score modestly above zero; for example, `Addition FT Only` reaches 14.0% on level 1. But all principal models collapse to 0.0% on level 2 and level 3, and overall accuracy remains between 0.0% and 4.7%. This is compelling evidence that the failure mode is not random noise but a specific inability to realize the required sequential interface.

### 4.3 Execution-Space Results

The execution-space composition experiments are more encouraging, though still imperfect. I directly reproduced a small CPU spot-check of the saved checkpoints on May 1, 2026 using the repository’s own evaluation code. On 50 held-out synthetic samples, the zero-shot pipeline achieved 100.0% accuracy on the reversal substep and 80.0% end-to-end accuracy on reverse-then-add. Under the same 50-sample check, the directly trained composed baseline achieved 70.0% accuracy. These numbers should be treated as lightweight validation rather than the main benchmark, but they establish an important point: once the intermediate interface is made explicit in string space, sequential composition becomes substantially more viable than in pure weight space.

The error patterns are also revealing. In the reproduced pipeline evaluation, many failures occur when reversed numbers contain leading zeros. Examples such as `381 + 350`, for which the gold answer is `236`, are predicted as `1136`; similarly, `990 + 106` should yield `700` but is predicted as `1500`. This suggests that the addition model is not consistently calibrated to the surface form emitted by the reversal model, particularly when reversal changes effective digit length. In other words, even the stronger pipeline result is limited by interface mismatch rather than by the absence of the underlying subskills.

### 4.4 Diagnostic Findings

The diagnostic experiments help explain why these failures are so persistent. In a direct run of `task_vector_conflict.py`, the overall cosine similarity between the reverse and addition task vectors is only `0.3202`, while overall sign conflict reaches `37.58%`. Conflict is especially pronounced in attention and MLP blocks, where several groups exhibit sign disagreement in the `44-47%` range. This makes the success of TIES and DARE more intelligible: if nearly two-fifths of overlapping nonzero updates disagree in sign, then naive arithmetic should be expected to fail badly.

The CKA analysis points in the same direction from representation space. In a reproduced 100-prompt run of `cka.py`, the addition fine-tuned model remains very close to the base model in the first three layers, with `CKA(base, add)` between `0.9560` and `0.9610`, whereas `CKA(base, rev)` is markedly lower, between `0.6561` and `0.7724`. The cross-task similarity `CKA(rev, add)` lies in the intermediate band `0.6238-0.7174`. A plausible interpretation is that the addition fine-tune is comparatively conservative, while reversal requires a more substantial representational reorganization. If so, merging the two models is difficult not simply because they encode different functions, but because they depart from the base model in asymmetrically structured ways.

### 4.5 Scratchpad Generalization

The scratchpad experiment yields an additional negative result that is worth including because it broadens the report beyond composition. In a direct 200-sample CPU evaluation of `evaluate_addition_scratchpad.py`, a scratchpad-trained addition model achieved `0.00%` accuracy on 3-digit out-of-distribution addition after training on only 1- and 2-digit numbers. The generated outputs often preserved the carry-trace format but truncated the final answer or emitted only a partial computation. This suggests that making intermediate reasoning explicit is not, by itself, sufficient to confer length generalization in this small-model regime.

### 4.6 Embedded Results Tables

Because the generated result artifacts are excluded from version control, the main quantitative tables are reproduced directly below so the report remains self-contained.

Table 1. Task arithmetic sweep reproduced from `task_arithmetic.py` on 200 held-out samples.

| Regime | Lambda | Reverse (%) | Addition (%) |
| --- | ---: | ---: | ---: |
| addition-only | 0.0 | NA | 99.5 |
| addition-only | 0.2 | NA | 100.0 |
| addition-only | 0.4 | NA | 100.0 |
| addition-only | 0.6 | NA | 100.0 |
| addition-only | 0.8 | NA | 99.5 |
| addition-only | 1.0 | NA | 97.0 |
| reverse-only | 0.0 | 95.0 | NA |
| reverse-only | 0.2 | 98.0 | NA |
| reverse-only | 0.4 | 99.0 | NA |
| reverse-only | 0.6 | 100.0 | NA |
| reverse-only | 0.8 | 100.0 | NA |
| reverse-only | 1.0 | 98.0 | NA |
| joint | 0.0 | 95.0 | 99.5 |
| joint | 0.2 | 91.0 | 94.5 |
| joint | 0.4 | 51.5 | 51.0 |
| joint | 0.6 | 4.5 | 5.0 |
| joint | 0.8 | 0.0 | 0.0 |
| joint | 1.0 | 0.0 | 0.0 |

Table 2. TIES sweep copied into the report from the checked-in result summary.

| Density | Lambda | Reverse (%) | Addition (%) |
| ---: | ---: | ---: | ---: |
| naive | 1.0 | 0.0 | 0.0 |
| 0.2 | 0.2 | 94.0 | 98.0 |
| 0.2 | 0.4 | 91.0 | 76.0 |
| 0.2 | 0.6 | 76.0 | 23.0 |
| 0.2 | 0.8 | 47.0 | 5.0 |
| 0.2 | 1.0 | 21.0 | 0.0 |
| 0.4 | 0.2 | 93.0 | 98.0 |
| 0.4 | 0.4 | 87.0 | 66.0 |
| 0.4 | 0.6 | 65.0 | 21.0 |
| 0.4 | 0.8 | 27.0 | 2.0 |
| 0.4 | 1.0 | 9.0 | 0.0 |
| 0.6 | 0.2 | 93.0 | 99.0 |
| 0.6 | 0.4 | 85.0 | 70.0 |
| 0.6 | 0.6 | 54.0 | 22.0 |
| 0.6 | 0.8 | 17.0 | 2.0 |
| 0.6 | 1.0 | 3.0 | 0.0 |
| 0.8 | 0.2 | 93.0 | 100.0 |
| 0.8 | 0.4 | 85.0 | 85.0 |
| 0.8 | 0.6 | 57.0 | 44.0 |
| 0.8 | 0.8 | 16.0 | 9.0 |
| 0.8 | 1.0 | 3.0 | 0.0 |
| 1.0 | 0.2 | 93.0 | 100.0 |
| 1.0 | 0.4 | 88.0 | 86.0 |
| 1.0 | 0.6 | 64.0 | 51.0 |
| 1.0 | 0.8 | 22.0 | 13.0 |
| 1.0 | 1.0 | 5.0 | 0.0 |

Table 3. DARE no-rescale sweep copied into the report from the checked-in result summary.

| Drop Rate | Lambda | Reverse (%) | Addition (%) |
| ---: | ---: | ---: | ---: |
| 0.2 | 0.2 | 92.5 | 97.0 |
| 0.2 | 0.4 | 65.0 | 72.0 |
| 0.2 | 0.6 | 16.0 | 22.5 |
| 0.2 | 0.8 | 1.5 | 0.5 |
| 0.2 | 1.0 | 0.0 | 0.0 |
| 0.4 | 0.2 | 93.0 | 97.0 |
| 0.4 | 0.4 | 88.0 | 82.5 |
| 0.4 | 0.6 | 52.0 | 40.5 |
| 0.4 | 0.8 | 22.5 | 3.0 |
| 0.4 | 1.0 | 5.0 | 0.5 |
| 0.6 | 0.2 | 94.0 | 98.0 |
| 0.6 | 0.4 | 93.0 | 89.0 |
| 0.6 | 0.6 | 84.5 | 66.0 |
| 0.6 | 0.8 | 59.5 | 25.5 |
| 0.6 | 1.0 | 31.0 | 2.5 |
| 0.8 | 0.2 | 95.0 | 99.5 |
| 0.8 | 0.4 | 94.5 | 99.0 |
| 0.8 | 0.6 | 93.0 | 96.5 |
| 0.8 | 0.8 | 91.5 | 91.0 |
| 0.8 | 1.0 | 88.0 | 79.5 |

Table 4. Composite benchmark for shared-base and merged models. Values below are percentages converted from the stored fractional accuracies.

| Model | Composite Accuracy (%) |
| --- | ---: |
| base | 0.0 |
| reverse_ft | 0.5 |
| addition_ft | 0.0 |
| naive(l=1.0) | 1.0 |
| naive(l=0.2) | 0.0 |
| ties(d=0.8,l=0.2) | 0.0 |

Table 5. Difficulty-stratified composition benchmark.

| Model | Level 1 (%) | Level 2 (%) | Level 3 (%) | Overall (%) |
| --- | ---: | ---: | ---: | ---: |
| Base Pre-trained | 6.0 | 0.0 | 0.0 | 2.0 |
| Reverse FT Only | 0.0 | 0.0 | 0.0 | 0.0 |
| Addition FT Only | 14.0 | 0.0 | 0.0 | 4.7 |
| Naive (l=1.0) | 12.0 | 0.0 | 0.0 | 4.0 |
| TIES (d=1.0, l=0.2) | 6.0 | 0.0 | 0.0 | 2.0 |

Table 6. Directly reproduced execution-space and scratchpad spot-checks run on May 1, 2026.

| Experiment | Setting | Metric |
| --- | --- | --- |
| zero-shot pipeline | 50 held-out synthetic samples | 100.0% reversal substep, 80.0% end-to-end |
| composed baseline | 50 held-out synthetic samples | 70.0% end-to-end |
| scratchpad addition OOD | 200 3-digit samples after 1-2 digit training | 0.0% exact-match |

## 5. Conclusion

### 5.1 Summary of Findings

This project demonstrates that capability preservation and compositional generalization are materially different phenomena in small transformers. When reverse and addition models share a common base, weight-space methods such as TIES and DARE preserve the source tasks far better than naive task arithmetic, showing that task vectors do contain reusable information. However, these same approaches fail almost completely on the true reverse-then-add benchmark, indicating that retaining two skills in one parameter vector is not equivalent to executing them in the correct sequence. The execution-space experiments reinforce this interpretation: explicit pipelining is much more effective than raw merging, but even then, performance is limited by interface mismatches such as tokenization differences and edge cases involving leading zeros. Taken together, the experiments support a clear conclusion: model algebra is promising for multi-skill retention, but robust compositional transfer requires stronger interface alignment, more consistent representations, and possibly explicit supervision over intermediate states.

## 6. Limitations

### 6.1 Result Consistency

Several limitations should be stated plainly. First, the result artifacts are not perfectly version-consistent. The README, `ties_results_table.md`, and comparison CSVs report slightly different percentages for nominally similar TIES settings, which likely reflects evaluator revisions, different sample sizes, or seed differences. The qualitative conclusion is stable, but exact headline numbers should be cited from a single artifact at a time.

### 6.2 Reproducibility and Interface Issues

Second, the composition-strategy code exposes real reproducibility issues. The checked-in evaluation helpers rely on brittle relative paths and, more seriously, on incompatible tokenizers. The standalone reverse dataset uses a 14-symbol vocabulary, whereas the composed-task tokenizer uses 15 symbols. As a result, the concat and adapter evaluators can fail with an embedding index error when composed-task token IDs are passed through reverse-model embeddings. This is not merely an engineering nuisance; it is direct evidence that representational interfaces were not fully unified across the composition track. In practice, this means the layer-concat and adapter experiments are better interpreted as attempted research directions than as fully stabilized baselines.

### 6.3 Limits of the Current Benchmark

Third, the shared-base benchmarks convincingly show preservation of source tasks but do not yet provide a positive demonstration of abstract algorithmic composition. The near-zero results on the composite benchmark imply that merged parameters do not by themselves induce a latent program that first reverses and then adds. More diagnostic probing could strengthen this claim, for example by logging intermediate generations, aligning hidden states across tasks, or introducing explicit interface supervision.

### 6.4 Scope Limits

Finally, the entire study is intentionally small scale. All models are tiny character-level transformers trained on synthetic arithmetic and string-reversal tasks. This is a strength for interpretability, but it limits external validity. The broader lesson is therefore conceptual rather than directly application-ready: in small transformers, shared-base task vectors can preserve skills, but reliable composition appears to require more than weight-space superposition. It requires aligned tokenization, aligned intermediate representations, and training objectives that make the interface between subtasks explicit.

## 7. Appendix: Repository Coverage

This appendix summarizes the main contents of the `transformer-algebra` folder so the report also serves as documentation for the repository rather than only as a narrative summary of the main experiments.

### 7.1 Main Experiment Scripts

- `config/train_base.py` trains the shared base model used by the task-vector experiments.
- `config/train_reverse_ft.py` and `config/train_addition_ft_base.py` fine-tune reverse and addition models from the shared base checkpoint.
- `task_arithmetic.py` evaluates naive task-vector addition.
- `ties.py` evaluates conflict-aware TIES merging and related composite analyses.
- `dare.py` evaluates sparsified DARE-style merging.
- `evaluate_stratified_composition.py` measures composition accuracy across progressively harder difficulty levels.
- `task_vector_conflict.py` analyzes cosine alignment and sign conflict between task vectors.
- `cka.py` compares hidden representations across the base, reverse, and addition models.

### 7.2 Composition Strategy Scripts

- `composition_strategies/config/train_reverse.py`, `train_addition.py`, and `train_composed.py` train standalone reverse, addition, and composed models from scratch.
- `composition_strategies/evaluate_composed.py` evaluates the zero-shot pipeline and the directly trained composed baseline.
- `composition_strategies/train_concat.py` implements the layer-concatenation experiment.
- `composition_strategies/train_adapter.py` implements the adapter-based composition experiment.
- `composition_strategies/evaluate_composition.py` evaluates the concat and adapter checkpoints.
- `composition_strategies/compose.py` defines the composition model classes and checkpoint-loading utilities.
- `composition_strategies/task_vectors.py` contains helper utilities used inside the composition experiments.

### 7.3 Data Preparation and Validation

- `data/prepare_mixed.py` creates the shared-base datasets and held-out reverse/addition test sets.
- `data/prepare_tokenized.py` creates standalone reverse, addition, and composed datasets for the scratch track.
- `data/prepare_addition_scratchpad.py` builds the scratchpad-style addition dataset.
- `data/generate_addition_test.py` and `data/generate_composed_test.py` generate held-out evaluation sets for the scratch experiments.
- `check_duplicates.py` checks overlap between training splits across datasets.
- `check_test_leakage.py` verifies that held-out tests do not leak into training data.
- `composition_strategies/create_val_txt.py` reconstructs validation text files where needed for composition training utilities.

### 7.4 Reproducibility Workflow

1. Build shared-base data with `python data/prepare_mixed.py`.
2. Build scratch datasets with `python data/prepare_tokenized.py`, plus optional `python data/prepare_addition_scratchpad.py`.
3. Train the shared base with `config/train_base.py`, copy the base checkpoint into the fine-tuning directories, then run `config/train_reverse_ft.py` and `config/train_addition_ft_base.py`.
4. Run the shared-base evaluations through `task_arithmetic.py`, `ties.py`, `dare.py`, `task_vector_conflict.py`, `cka.py`, and `evaluate_stratified_composition.py`.
5. Train the scratch reverse/addition/composed checkpoints with the configs under `composition_strategies/config/`.
6. Evaluate execution-space composition with `composition_strategies/evaluate_composed.py`, `evaluate_composition.py`, and `evaluate_addition_scratchpad.py`.

### 7.5 Outputs Used in This Report

- Shared-base checkpoints: `out/base/ckpt.pt`, `out/reverse_ft/ckpt.pt`, `out/addition_ft/ckpt.pt`
- Scratch checkpoints: `out/reverse/ckpt.pt`, `out/addition/ckpt.pt`, `out/composed/ckpt.pt`
- Optional composition checkpoints: `out/concat/ckpt.pt`, `out/adapter/adapter.pt`
- Scratchpad checkpoint: `out/addition_scratchpad/ckpt.pt`

The report’s embedded tables were included specifically so the document remains complete even when generated CSV or markdown result files are excluded from version control.

## 8. References

[1] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. "Editing Models with Task Arithmetic." ICLR, 2023.

[2] Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. "TIES-Merging: Resolving Interference When Merging Models." arXiv preprint arXiv:2306.01708, 2023.

[3] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch." arXiv preprint arXiv:2311.03099, 2023.
