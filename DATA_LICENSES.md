# Dataset and model attribution

The repository's MIT license applies to project code, not the datasets or pretrained weights.

## S1: UCI SMS Spam Collection

- Authors: Tiago A. Almeida and José María Gómez Hidalgo.
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- Dataset: [UCI repository, DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
- Reference: T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, “Contributions to the
  study of SMS spam filtering: New collection and results,” DocEng, 2011,
  pp. 259–262, [DOI 10.1145/2034691.2034742](https://doi.org/10.1145/2034691.2034742).

## S2: Super SMS Dataset

M. Salman, M. Ikram, and M. A. Kaafar, “Investigating evasive techniques in SMS spam
filtering: A comparative analysis of machine learning models,” IEEE Access, vol. 12,
pp. 24306–24324, 2024, [DOI 10.1109/ACCESS.2024.3364671](https://doi.org/10.1109/ACCESS.2024.3364671).

The [pinned author repository](https://github.com/smspamresearch/spstudy/tree/a0889f36fb59e23a7f853a6932485d93d18b8afc)
publicly releases the research dataset and requests citation, but has no explicit
dataset license file. This project limits use to non-commercial academic work and
does not redistribute raw or processed SMS. That restriction is a project policy,
not a claim that an explicit license grant exists. Download scripts retrieve the
data directly from the authors. Local data remain excluded from Git.

## DistilBERT

`distilbert-base-uncased` is published under Apache-2.0. Cite V. Sanh, L. Debut,
J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: Smaller, faster,
cheaper and lighter,” [arXiv:1910.01108](https://arxiv.org/abs/1910.01108), 2019.

See each installed library's distribution for its license. Fine-tuned transformer
weights must remain outside Git, and any future model publication must include
this provenance and the intended academic use.
