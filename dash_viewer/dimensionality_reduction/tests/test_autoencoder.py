"""Autoencoder training against awkward cohort sizes.

The failures these cover all depend on how many studies came back from the
query, so they appear for one cohort and not the next: a cohort smaller than
one batch used to divide by zero, and a cohort leaving a single trailing sample
used to raise from BatchNorm1d partway through training.
"""
import threading

import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='autoencoder needs torch')

from dash_viewer.dimensionality_reduction import AutoencoderDm as ae_module
from dash_viewer.dimensionality_reduction.AutoencoderDm import calculate_autoencoder

BATCH_SIZE = 32          # what calculate_autoencoder passes through
ENCODING_DIM = 4
N_FEATURES = 60


@pytest.fixture
def curves(rng):
    def build(n_studies):
        return (rng.normal(size=(n_studies, N_FEATURES)) * 2 - 9).astype('float32')
    return build


@pytest.fixture(autouse=True)
def no_plots(monkeypatch, tmp_path):
    """Image logging pulls in matplotlib, which aborts the process next to torch."""
    monkeypatch.delenv('CARDIO_AE_PLOTS', raising=False)


def run_training(curves_array, log_dir, epochs=2, kind='mlp', **kwargs):
    """Drive one training run.

    Pinned to the dense model because several tests below instrument
    AutoencoderDm directly to observe the training loop. calculate_autoencoder
    now defaults to the convolutional model, so without this the spies would
    watch a class that is never constructed and silently observe nothing.
    """
    return calculate_autoencoder(encoding_dim=ENCODING_DIM, array_curves=curves_array,
                                 epochs=epochs, log_dir=str(log_dir), kind=kind, **kwargs)


@pytest.mark.parametrize('n_studies', [
    20,                  # smaller than one batch: len(data) // batch_size == 0
    BATCH_SIZE + 1,      # trailing batch of exactly one sample
    BATCH_SIZE * 2,      # exact multiple, the case that always worked
    2,                   # the smallest cohort BatchNorm1d can handle
])
def test_trains_for_any_cohort_size(curves, tmp_path, n_studies):
    encoded, _ = run_training(curves(n_studies), tmp_path / f'log_{n_studies}')

    assert encoded.shape == (n_studies, ENCODING_DIM)
    assert np.isfinite(encoded).all()


def test_single_study_reports_why_it_cannot_train(curves, tmp_path):
    with pytest.raises(ValueError, match='at least 2 studies'):
        run_training(curves(1), tmp_path / 'log_single')


def test_no_study_is_dropped_from_training(curves, tmp_path, monkeypatch):
    """A trailing sample is folded into the previous batch, never skipped."""
    seen = []
    original_forward = ae_module.AutoencoderDm.forward

    def counting_forward(self, x):
        if self.training:
            seen.append(len(x))
        return original_forward(self, x)

    monkeypatch.setattr(ae_module.AutoencoderDm, 'forward', counting_forward)
    n_studies = BATCH_SIZE + 1
    # val_fraction=0 so this measures batching alone, not the held-out split
    run_training(curves(n_studies), tmp_path / 'log_fold', epochs=1, val_fraction=0)

    assert sum(seen) == n_studies, f'expected every study in an epoch, batches were {seen}'
    assert min(seen) >= 2, 'BatchNorm1d cannot train on a batch of one'


def test_training_mode_survives_image_logging(curves, tmp_path, monkeypatch):
    """log_reconstruction switches the model to eval; training must switch it back."""
    monkeypatch.setenv('CARDIO_AE_PLOTS', '1')
    modes = []
    logged_at = []
    original_forward = ae_module.AutoencoderDm.forward

    def recording_forward(self, x):
        modes.append(self.training)
        return original_forward(self, x)

    def fake_log(writer, model, data, epoch):
        model.eval()            # what the real one does
        logged_at.append(len(modes))

    monkeypatch.setattr(ae_module.AutoencoderDm, 'forward', recording_forward)
    monkeypatch.setattr(ae_module, 'log_reconstruction', fake_log)
    run_training(curves(BATCH_SIZE * 2), tmp_path / 'log_mode', epochs=51)

    assert logged_at, 'image logging should fire at epoch 50 when opted in'
    after_logging = modes[logged_at[0]:]
    assert any(after_logging), (
        'every forward pass after image logging ran in eval mode - dropout and '
        'BatchNorm stayed frozen for the rest of training')


def test_runs_on_a_worker_thread(curves, tmp_path):
    outcome = {}

    def run():
        try:
            outcome['encoded'], _ = run_training(curves(BATCH_SIZE + 1), tmp_path / 'log_thread')
        except BaseException as exc:  # noqa: BLE001
            outcome['error'] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=300)

    assert 'error' not in outcome, f"training failed off the main thread: {outcome.get('error')}"
    assert outcome['encoded'].shape == (BATCH_SIZE + 1, ENCODING_DIM)


def test_batches_are_reshuffled_every_epoch(curves, tmp_path, monkeypatch):
    """Fixed slices train on the same partition every epoch and skew BatchNorm."""
    batches = []
    original_forward = ae_module.AutoencoderDm.forward

    def recording_forward(self, x):
        if self.training:
            batches.append(x.detach().cpu().numpy().copy())
        return original_forward(self, x)

    monkeypatch.setattr(ae_module.AutoencoderDm, 'forward', recording_forward)
    run_training(curves(BATCH_SIZE * 2), tmp_path / 'log_shuffle', epochs=2, val_fraction=0)

    first_epoch, second_epoch = batches[0], batches[len(batches) // 2]
    assert not np.array_equal(first_epoch, second_epoch), (
        'the same batch composition was reused across epochs')


def test_best_weights_are_kept_not_the_last_ones(curves, tmp_path, monkeypatch):
    """The loss oscillates, so the final epoch is not necessarily the best one."""
    restored = []
    original_load = ae_module.AutoencoderDm.load_state_dict

    def recording_load(self, state, *args, **kwargs):
        restored.append(True)
        return original_load(self, state, *args, **kwargs)

    monkeypatch.setattr(ae_module.AutoencoderDm, 'load_state_dict', recording_load)
    run_training(curves(BATCH_SIZE * 2), tmp_path / 'log_best', epochs=5)

    assert restored, 'training finished without restoring the best weights'


def test_training_stops_early_when_it_stops_improving(curves, tmp_path, capsys):
    run_training(curves(BATCH_SIZE * 2), tmp_path / 'log_early', epochs=500, patience=3)

    assert 'Early stop at epoch' in capsys.readouterr().out


def test_validation_split_is_held_out_of_training(curves, tmp_path):
    data = torch.FloatTensor(curves(40))
    train_data, val_data = ae_module.split_train_val(data, val_fraction=0.25, seed=42)

    assert len(val_data) == 10
    assert len(train_data) == 30


def test_tiny_cohort_keeps_all_data_for_training(curves, tmp_path):
    """A split that would leave fewer than 2 training rows is refused."""
    data = torch.FloatTensor(curves(2))
    train_data, val_data = ae_module.split_train_val(data, val_fraction=0.5, seed=42)

    assert val_data is None
    assert len(train_data) == 2


def test_explained_variance_is_reported_for_comparison_with_pca(curves, tmp_path, capsys):
    """MSE alone cannot be compared against PCA's cumulative explained variance."""
    run_training(curves(BATCH_SIZE * 2), tmp_path / 'log_r2', epochs=2)

    out = capsys.readouterr().out
    assert 'R2' in out and 'components' in out


def test_explained_variance_of_a_perfect_reconstruction_is_one(curves):
    class Identity(torch.nn.Module):
        training = False

        def eval(self):
            return self

        def __call__(self, x):
            return x, x

    data = torch.FloatTensor(curves(20))

    assert ae_module.explained_variance(Identity(), data) == pytest.approx(1.0)
