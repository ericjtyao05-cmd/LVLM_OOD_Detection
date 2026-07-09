# Retraction note

Runs `smoke`, `full`, and `full_v2` were removed from this branch tip
(2026-07-08); they remain in git history for provenance.

Their fake-containing test metrics (`fake_probe`, contaminated `ood_test`,
`auroc_fake` etc.) measured real-vs-AI separation, which cannot be attributed
to physical impossibility: every fake pool -- including WHOOPS!, whose images
are designer-made *with Midjourney/DALL-E/Stable Diffusion* -- is AI-generated,
while the ID/OOD pools are real photos.

`contamination_v1` is the cleaned extraction of full_v2's valid subset:
contaminated id_train, clean real id_test vs clean real ood_test
(the *_ood00 conditions). That comparison is real-vs-real and unconfounded.
