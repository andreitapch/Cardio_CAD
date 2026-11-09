# Environment notes

Machine-specific gotchas that cost time to rediscover. `environment.yml` and
`requirements.txt` remain the authoritative dependency lists.

## `WinError 127` loading `torch/lib/shm.dll`

```
OSError: [WinError 127] The specified procedure could not be found.
Error loading "...\torch\lib\shm.dll" or one of its dependencies.
```

Cause: MKL picks its OpenMP runtime at load time, and whichever of conda's MKL
and torch claims `libiomp5md.dll` first wins the process. If MKL wins, torch's
`shm.dll` resolves against the wrong runtime.

**The trigger is a BLAS call, not an import** -- bisected, because guessing at
this wastes an afternoon. `import numpy`, `import scipy`, `import sklearn`,
`import h5py` and `import matplotlib.pyplot` all leave torch loadable
afterwards. `np.linalg.svd`, or a matplotlib figure being *rendered*, does not.
So "import torch first" is not the rule; "before any array work" is.

The fix that makes ordering irrelevant, because MKL then threads through TBB and
never loads an OpenMP runtime at all:

```
MKL_THREADING_LAYER=TBB
```

It has to be set before the process starts to be unconditional. Verified with a
BLAS call *and* a figure render ahead of the torch import: fine with the
variable preset, `WinError 127` without it.

Where it is set:

- `conftest.py` and `dash_viewer/dash_main.py` set it, then import torch, as
  their first two statements -- covering the test suite and the dashboard.
- **Jupyter kernels load neither**, so it is in this environment's kernelspec:
  `<conda env>/share/jupyter/kernels/python3/kernel.json`, as
  `"env": {"MKL_THREADING_LAYER": "TBB"}` (a `kernel.json.bak` sits beside it).
  A kernel started after that change is safe in any cell order. Recreating the
  env drops this -- re-add it.
- `motion_estimator/notebooks/loftr_point_tracking.ipynb` still opens with the
  two lines as a fallback for other environments, and turns the DLL traceback
  into a message naming the fix.

The binding is per-process: once a kernel has raised this, no import order
repairs it. Restart the kernel.

The related symptom from the same conflict is `OMP: Error #15` aborting the
process outright, which `MKL_THREADING_LAYER=TBB` also prevents.

## pip and conda fail with `CERTIFICATE_VERIFY_FAILED`

Symptom — every index, not just one:

```
SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate'))
```

Cause: Norton intercepts HTTPS and re-signs it with its own root CA, which is
not in the certifi bundle Python verifies against. Two clues confirm it rather
than a corporate proxy:

```
NODE_EXTRA_CA_CERTS = C:\ProgramData\Norton\Antivirus\wscert.pem
SSLKEYLOGFILE       = \\.\nllMonFltProxy\...
```

Node was already configured for it; Python was not. Compounding this, activating
a conda env sets `SSL_CERT_FILE` to *that env's* `cacert.pem`, so a `python.exe`
from a different prefix inherits a bundle it does not own.

Fix — a bundle holding both the public roots and Norton's:

```powershell
$certifi = "C:\Users\andre\Anaconda3\Lib\site-packages\certifi\cacert.pem"
$norton  = "C:\ProgramData\Norton\Antivirus\wscert.pem"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.certs" | Out-Null
Set-Content "$env:USERPROFILE\.certs\norton-combined-ca.pem" `
    ((Get-Content $certifi -Raw) + "`n" + (Get-Content $norton -Raw)) -NoNewline
```

Then point the tooling at it for the session:

```powershell
$env:SSL_CERT_FILE     = "$env:USERPROFILE\.certs\norton-combined-ca.pem"
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
Remove-Item Env:SSL_CERT_DIR -ErrorAction SilentlyContinue
```

To make it permanent instead: `pip config set global.cert
%USERPROFILE%\.certs\norton-combined-ca.pem`, and set `SSL_CERT_FILE` /
`REQUESTS_CA_BUNDLE` as user environment variables. Prefer this over
`--trusted-host`, which skips verification entirely rather than fixing it.

The same bundle is what lets `huggingface_hub` download the EchoCLIP weights;
without it the model load fails with the identical error.

`conda` hits this too, and reports it as `CondaSSLError` against
`repo.anaconda.com`. It needs telling separately:

```powershell
conda config --set ssl_verify "$env:USERPROFILE\.certs\norton-combined-ca.pem"
```

Regenerate the bundle after a Norton update replaces `wscert.pem`.

## Which interpreter

**`cardioCad` is the project environment** —
`C:\Users\andre\Anaconda3\envs\cardioCad\python.exe`, Python 3.11.16, numpy
2.4.6, pandas 3.0.5, scipy 1.17.1, dash 4.4.1, pingouin 0.6.1. It matches
`name: cardioCad` in `environment.yml`.

`conda env list` **does not show it** — it is missing from
`~/.conda/environments.txt` — which makes it easy to conclude it does not
exist. It does; address it by full path, or `conda activate
C:\Users\andre\Anaconda3\envs\cardioCad`. To get it listed again:

```powershell
Add-Content "$env:USERPROFILE\.conda\environments.txt" "C:\Users\andre\Anaconda3\envs\cardioCad"
```

Two other envs are decoys:

- **Anaconda base** (`C:\Users\andre\Anaconda3`) has a large legacy stack
  (numpy 1.26, scipy 1.12, pandas 2.1) but **no `dash` and no `pingouin`**, so
  it cannot run `dash_viewer`. It also has a stale C runtime — see below.
- **`cardio-cad`** (hyphenated, the one `conda` activates by default in this
  shell) is unrelated tooling with only numpy. Not the project env.

## Modern binary wheels fail to import in Anaconda base

Symptom, on `import torch` or any recent compiled wheel:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine
failed. Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

`c10.dll` fails even with a minimal `PATH`, so it is not DLL shadowing from
`Library\bin`. The Windows Application event log names the real culprit:

```
Faulting module name: msvcp140.dll, version: 14.29.30153.0
Exception code: 0xc0000005
```

`C:\Users\andre\Anaconda3\msvcp140.dll` is **14.29** (VS 2019 era) and sits
beside `python.exe`, so it wins the DLL search over System32's 14.51. torch
2.14 needs roughly 14.40+. `cardioCad` ships 14.51 and is unaffected — another
reason to work there.

If base ever does need modern wheels: `conda install -n base vc14_runtime`
(14.51 is already cached under `pkgs\`). MSVC runtimes are backward compatible,
so existing packages keep working — but it does replace the C runtime of the
Anaconda install.

## GPU

RTX 2060, 6 GB, driver 576.83, compute capability **sm_75** (Turing), with
torch 2.14.0+cu126.

Use `precision="fp16"` when loading EchoCLIP, **not** the `precision="bf16"` in
the upstream `zero_shot_example.py`. Measured on the EchoCLIP image encoder,
20 frames per pass:

| precision | result |
|---|---|
| fp16 | 58 ms |
| bf16 | `RuntimeError: GET was unable to find an engine to execute this computation` |
| fp32 | 197 ms |

`torch.cuda.is_bf16_supported()` returns `True` here, which is misleading — it
reports emulation as support, and the convolutions still have no bf16 kernel.
Do not use it to gate the dtype on this card.

## EchoCLIP, as actually loaded

Confirmed against `hf-hub:mkaichristensen/echo-clip` with open_clip 3.3.0:

- **not gated** (`gated: False`), MIT licensed, no `HF_TOKEN` needed — the
  upstream comment telling you to `huggingface-cli login` does not apply. An
  unauthenticated-request rate-limit warning is printed and is harmless.
- embedding dim **512**; text context length 77 (standard CLIP BPE, via
  `open_clip.get_tokenizer(...)` → `SimpleTokenizer`).
- `preprocess_val` is `Resize(224) → CenterCrop(224) → RGB → ToTensor →
  Normalize`. Our frames are square (500×500), so the centre crop discards
  nothing; it would crop for non-square input.
- weights land in the usual `~/.cache/huggingface`; first load took ~37 s
  including the download.
