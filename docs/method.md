# HYSET Method Overview

## 1. Problem

Tool retrieval for LLM agents is a set problem. Real queries usually need multiple APIs used together, so returning a ranked list of individually relevant tools is structurally insufficient.

HYSET treats the retrieved tool set itself as the prediction target.

## 2. Formulation

Let `E` be a candidate tool set and `x` a query. HYSET scores:

```text
F(x, E) = F_set(E) + F_align(x, E)
```

The trainable parameters are:

- tool embeddings `Z`
- cardinality-specific interaction matrices `{M_m}`
- projection matrix `P`

The query encoder stays frozen.

## 3. Set-level scoring

For a set `E = {t_j1, ..., t_jm}`:

```text
F_set(E) = Σ_{1≤a<b≤m} z_ja^T M_m z_jb
```

The same tool pair can contribute differently at different set sizes because `M_m` depends on `m`.

## 4. Query-set alignment

The frozen encoder maps the query to `r(x)`. HYSET computes:

```text
ℓ(x, t_j) = r(x)^T P z_j
α_k(x, E) = exp(ℓ(x, t_jk)) / Σ_q exp(ℓ(x, t_jq))
F_align(x, E) = Σ_k α_k(x, E) · ℓ(x, t_jk)
```

This keeps the alignment term set-conditioned even though it is built from per-tool scores.

## 5. Training

HYSET uses:

- retrieval loss over one positive set and `K_neg - 1` negatives
- reward-weighted self-training loss from frozen-agent execution feedback
- Frobenius regularization on `{M_m}`

The full objective is:

```text
L = L_ret + η L_self + λ Σ_m ||M_m||_F^2
```

The implementation enforces unit-norm tool embeddings after every optimizer step.

## 6. Negative sampling

For each query, the candidate pool uses the paper’s fixed 50/30/20 mixture:

- 50% size-matched uniform negatives
- 30% in-batch negatives
- 20% hard negatives from nearest-neighbor tool replacement

The default pool size is `K_neg=64`.

## 7. Inference

Inference follows the paper’s two-stage procedure.

Stage 1:

- score all singleton tools with `r(x)^T P z_j`
- keep the top `K1`
- add `K_pool - K1` tools by complementarity with `S0` under `M_M`

Stage 2:

- enumerate all subsets of the shortlist with cardinality `1..M`
- score each subset with the full HYSET scorer
- return the best set

The implementation also emits a greedy marginal ranking of length 5 for Recall, NDCG, and COMP.