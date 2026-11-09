"""EchoCLIP as a frozen feature extractor over the intervendor cohort.

The trunk is never fine-tuned: with 118 patients there is no honest way to
train 150M parameters, so EchoCLIP only ever runs in ``torch.no_grad`` and the
learning happens in a small head on top of its embeddings
(:mod:`motion_estimator.echoclip.heads`).

That split is what makes the work tractable. Embeddings are a deterministic
function of (clip, encoder settings), so one GPU pass writes a cache
(:mod:`embed_cache`) and every training run afterwards reads 10 MB of float32
instead of 3.5 GB of ``data.mat``.

* :mod:`config`      -- model id and the settings the cache is keyed on
* :mod:`encoder`     -- frame selection, preprocessing, image/text encoding
* :mod:`embed_cache` -- build and load the on-disk embedding cache
"""
