# Test task for Claude Code on the M3 Max: CPU fixes A1–A9 and the restart (A8) on Apple/Metal

A step-by-step check for a Claude Code session on the Apple M3 Max. Follow the steps in order,
check each step's expectation before going on, and end with the report of step 9.

## Context

These changes were developed and checked on a GB10 (CUDA). They have **not** been built or
run on Apple Silicon:

| what | where | status |
|---|---|---|
| A1, A2: the writer doesn't resample a reused background; banded resampling | js-contrib `pair_h5.py`, `bulk_sources.py` | merged (js-contrib PR #56) |
| A3–A7, A9: string-source check, anisotropy diagnostic switch, store packing, GPU edge test, optional frame store | MUSIC4GPU PR #19, X-SCAPE PRs #166, #167 | merged; **the Metal edge test (`MetalPipelines::reduce_max_edge`) never compiled** |
| A8: MUSIC_2 starts from MUSIC_1's state (`MusicCheckpoint`), on by default in `run_prod_jet.py` | branches `music2_restart` in X-SCAPE, MUSIC4GPU, js-contrib | not merged; **the Metal save/restore (`MetalPipelines::save_snapshots` etc.) never compiled** |

Design and GB10 results:
- `docs/Plans/PLAN_cpu_fixes_A.md` and `docs/Plans/PLAN_music2_restart.md` (in js-contrib);
- the user docs: `prod_AuAu_0_10_jet/README.md`, *MUSIC_2 from MUSIC_1's state (default)*.

**Expected on any machine:** every output file bit-identical to the old code, and faster.
GB10, one job alone:
- two-stage Au+Au, 4 events, `--reuse 2`: 115 → 93 s, jet events −25 to −28 %;
- hydro-only Au+Au: −16 %;
- O+O: −15 %.

## Rules for this session

- **Don't commit, push, or open PRs.** Don't change source files. If something fails, find out
  why and report it with the evidence: the log lines, the file and line. Suggest a fix, but
  don't apply it unless the user asks.
- **Don't delete** test outputs or anything outside `~/xs_test_m3` without asking.
- **Don't rebuild while a job runs:** it replaces the libraries the job has loaded.
- **Long jobs:** run them in the background and wait for them. A job is 1–4 min, a rebuild
  several minutes.
- **Don't run jobs in parallel:** the timings must be of one job alone.
- **Note the starting branches** in step 1 and restore them in step 8.
- **Shell state doesn't carry over between commands** in Claude Code. Every command block
  below therefore starts with `source ~/xs_test_m3/env.sh`, which step 1 writes, and the
  helpers are scripts in `~/xs_test_m3/`.

## Step 1. Find the setup, write `~/xs_test_m3/env.sh`

The usual layout (`docs/BENCHMARK_M3MAX.md`):
- conda env `fno_env_mlx`, Homebrew libomp;
- the X-SCAPE checkout with its Metal build in `build_gpu`;
- `external_packages/music4gpu` and `external_packages/js-contrib` as their own git checkouts.

Find the actual paths. Then create `~/xs_test_m3/env.sh` with the Write tool, with the paths
and env name adjusted:

```bash
export XS=$HOME/JetScape/X-SCAPE                  # the X-SCAPE checkout (adjust)
export M4=$XS/external_packages/music4gpu
export JSC=$XS/external_packages/js-contrib
export BUILD=$XS/build_gpu                        # the Metal build directory (adjust)
export T=$HOME/xs_test_m3
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=1
eval "$(conda shell.bash hook)" && conda activate fno_env_mlx
```

Then check:

```bash
source ~/xs_test_m3/env.sh
which python
grep -E "^USE_METAL:|^USE_MUSIC:|^CMAKE_BUILD_TYPE:" $BUILD/CMakeCache.txt
for r in $XS $M4 $JSC; do echo "$r: $(git -C $r rev-parse --abbrev-ref HEAD) $(git -C $r rev-parse --short HEAD)"; git -C $r status --short | grep -v '^??'; done
python -c "import numpy; print(numpy.__version__); numpy.show_config()" | grep -iE "^[0-9]|blas|accelerate|openblas" | head
```

**Expect:**
- `python` from the env, and `USE_METAL:BOOL=ON`.
- No local changes in the three repos. If there are some, stop and ask the user (stash or
  not).
- Write down the three branches and commits for step 8, and numpy's BLAS (OpenBLAS or
  Accelerate) for the report.

## Step 2. Helper scripts

Create these three files in `~/xs_test_m3/` with the Write tool, then `chmod +x
~/xs_test_m3/*.sh`.

**`run.sh`** — one job, alone:

```bash
#!/bin/bash
# run.sh TAG SCRIPT [driver options] -> ~/xs_test_m3/TAG/  (log.txt, /usr/bin/time -l at the end)
source ~/xs_test_m3/env.sh
tag=$1; script=$2; shift 2
d=$T/$tag; rm -rf "$d"; mkdir -p "$d"; cd "$d" || exit 1
/usr/bin/time -l python "$JSC/contribs/PyJetscape/example/$script" \
  --build "$BUILD" --main-xml "$XS/config/jetscape_main.xml" --seed-registry none \
  --outdir "$d" "$@" > "$d/log.txt" 2>&1
echo "$tag exit $?"
```

**`show.sh`** — wall time, peak memory, per-event times, restart lines, GPU line, errors:

```bash
#!/bin/bash
# show.sh TAG...
for tag in "$@"; do L=$HOME/xs_test_m3/$tag/log.txt
  echo "== $tag: $(grep -m1 ' real' "$L" | awk '{print $1}') s real, $(grep 'maximum resident' "$L" | awk '{printf "%.2f GB", $1/1e9}') peak"
  grep -E "done in" "$L" | sed -E 's/.*(event [0-9/]+ done in [0-9.]+ s).*/   \1/'
  grep -E "restart  |Restart:|restart verify" "$L" | sed -E $'s/\x1b\\[[0-9;]*m//g' | sort | uniq -c
  grep -m2 -iE "metal|MUSIC-GPU\].*(device|GPU grid)" "$L"
  grep -iE "falling back to CPU|Traceback|Error" "$L" | head -3
done
```

**`compare.py`** — every dataset of the `.h5` files of dir A against the same files in dir B:

```python
"""compare.py A B"""
import sys, glob, os
import numpy as np, h5py
try:
    import hdf5plugin  # noqa: F401  (Blosc)
except ImportError:
    pass
a_dir, b_dir = sys.argv[1:3]
bad = 0
for fa in sorted(glob.glob(os.path.join(a_dir, "*.h5"))):
    fb = os.path.join(b_dir, os.path.basename(fa))
    if not os.path.exists(fb):
        print("missing in B:", os.path.basename(fa)); bad += 1; continue
    with h5py.File(fa, "r") as A, h5py.File(fb, "r") as B:
        na, nb = [], []
        A.visititems(lambda n, o: na.append(n) if isinstance(o, h5py.Dataset) else None)
        B.visititems(lambda n, o: nb.append(n) if isinstance(o, h5py.Dataset) else None)
        if set(na) != set(nb):
            print("  dataset sets differ:", sorted(set(na) ^ set(nb))); bad += 1
        same = 0
        for n in na:
            if n not in B or any(s in n for s in ("wall_s", "uuid", "path")):
                continue
            x, y = A[n][()], B[n][()]
            if n.endswith("source/flux"):          # 1-ulp run-to-run noise, known
                ok = x.shape == y.shape and np.allclose(x, y, rtol=1e-12, atol=0)
            elif getattr(x, "dtype", None) is not None and x.dtype.kind in "fc":
                ok = x.shape == y.shape and np.array_equal(x.view(np.uint8), y.view(np.uint8))
            else:
                ok = np.array_equal(x, y)
            if ok:
                same += 1
            else:
                print("  DIFF", os.path.basename(fa), n); bad += 1
        print(f"{os.path.basename(fa)}: {same} datasets identical")
print("RESULT:", "identical" if bad == 0 else f"{bad} DIFFERENCES")
```

## Step 3. Baseline: the code before all of these changes

```bash
source ~/xs_test_m3/env.sh
git -C $XS fetch origin && git -C $XS checkout 733b2838      # X-SCAPE contrib before PR #166
git -C $M4 fetch origin && git -C $M4 checkout bc8543a       # MUSIC4GPU XSCAPE before PR #19
git -C $JSC fetch origin && git -C $JSC checkout 309bb90     # js-contrib main before PR #56
cmake --build $BUILD -j 14 > $T/build_base.log 2>&1; echo "build exit $?"; tail -3 $T/build_base.log
ls -la $JSC/contribs/PyJetscape/python/jetscape/pyjetscape_core*.so
```

**Expect:** build exit 0, and `pyjetscape_core*.so` with a fresh timestamp.

Then the five baseline jobs, one after the other, in the background:

```bash
source ~/xs_test_m3/env.sh
$T/run.sh base_norm prod_AuAu_0_10/run_prod.py --events 3 --seed 1
$T/run.sh base_oo   prod_OO_isr/run_prod_oo_isr.py --events 4 --seed 101
$T/run.sh base_jet  prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize both
$T/run.sh base_jetq prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize both --particlize-only
$T/run.sh base_pjet prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize jet
$T/show.sh base_norm base_oo base_jet base_jetq base_pjet
```

**Expect:**
- each `exit 0`;
- `show.sh` lists the events, a Metal/GPU line, and no `falling back to CPU` or Traceback.

If a baseline job fails, stop and report: the problem is then not these changes.

## Step 4. The new code: build

```bash
source ~/xs_test_m3/env.sh
git -C $XS  checkout -B music2_restart origin/music2_restart
git -C $M4  checkout -B music2_restart origin/music2_restart
git -C $JSC checkout -B music2_restart origin/music2_restart
cmake --build $BUILD -j 14 > $T/build_new.log 2>&1; echo "build exit $?"
grep -nE "error|warning" $T/build_new.log | grep -iE "MetalPipelines|evolve|advance|music_checkpoint|HydroinfoMUSIC|hydro_source|MusicWrapper|bind_music" | head -30
```

**Expect:**
- build exit 0;
- no errors, and no new warnings in those files. This is the first time the Metal code
  compiles: `MetalPipelines::reduce_max_edge`, `state_floats`, `alloc_state`, `free_state`,
  `save_snapshots`, `restore_snapshots`, `read_state` in `src/gpu/MetalPipelines.mm`.

On a compile error, report the message with its file and line; don't fix it.

## Step 5. Unit tests

```bash
source ~/xs_test_m3/env.sh
cd $JSC/contribs/PyJetscape && python -m pytest -q tests/test_resample.py tests/test_pair_h5.py tests/test_particlize_h5.py 2>&1 | tail -5
```

**Expect:** all passed. `test_resample.py` is the important one: the banded resampling (A2)
must be bit for bit the old einsum version with this Mac's numpy and BLAS, which was checked
only on the GB10's OpenBLAS. If it fails, report how many values differ and by how much (the
assertion message shows it).

## Step 6. The new code: the same jobs, plus the restart's own checks

One after the other, in the background:

```bash
source ~/xs_test_m3/env.sh
$T/run.sh new_norm    prod_AuAu_0_10/run_prod.py --events 3 --seed 1
$T/run.sh new_oo      prod_OO_isr/run_prod_oo_isr.py --events 4 --seed 101
$T/run.sh new_jet     prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize both
$T/run.sh new_jet_off prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize both --no-music2-restart
MUSIC_RESTART_VERIFY=1 $T/run.sh new_jet_verify prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize both
$T/run.sh new_jetq    prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize both --particlize-only
$T/run.sh new_pjet    prod_AuAu_0_10_jet/run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize jet
$T/show.sh new_norm new_oo new_jet new_jet_off new_jet_verify new_jetq new_pjet
```

**Expect:**

| run | log |
|---|---|
| `new_jet`, `new_jetq` | `restart  MUSIC_2 starts from MUSIC_1's state at tau <= 1 fm/c …`; per MUSIC_1 run `Checkpoint at step N, tau = … fm/c: GPU state 96 MB, …`; 4 × `Restart: started from the checkpoint at step N` (N = 28 on the GB10; it may differ only if the XML's `tau_delay`, `Delta_Tau` or the strings' τ differ) |
| `new_jet_verify` | 4 × `restart verify at step N: identical to the checkpoint`; **any `DIFFERS` is a failure** (report the bracketed items) |
| `new_jet_off` | `restart  OFF: --no-music2-restart` |
| `new_pjet` | `restart  OFF: MUSIC_2 builds the freeze-out surface and MUSIC_1 does not …` |
| all | a Metal/GPU line, no `falling back to CPU`, no Traceback |

If MUSIC_2 logs `run from tau0: <reason>` in `new_jet`, report the reason: that is a
restart that didn't happen, not an error.

## Step 7. Compare the outputs

```bash
source ~/xs_test_m3/env.sh
for p in "base_norm new_norm" "base_oo new_oo" "base_jet new_jet" "base_jet new_jet_off" \
         "base_jet new_jet_verify" "base_jetq new_jetq" "base_pjet new_pjet"; do
  a=${p% *}; b=${p#* }; echo "######## $a vs $b"; python $T/compare.py $T/$a $T/$b
done 2>&1 | grep -E "####|DIFF|missing|differ|RESULT"
```

**Expect:** 7 × `RESULT: identical`.

The root attributes are not compared, and they differ by design: `prod_user_xml`, the wall
times, `file_uuid`, `prod_music2_restart`. On a difference, report the `DIFF` lines. Then
check the restart alone: compare `new_jet_off` with `new_jet`. If those two agree, the
difference comes from A1–A7 and not from the restart.

## Step 8. Restore

```bash
source ~/xs_test_m3/env.sh
git -C $XS checkout <branch from step 1>
git -C $M4 checkout <branch from step 1>
git -C $JSC checkout <branch from step 1>
cmake --build $BUILD -j 14 > $T/build_restore.log 2>&1; echo "build exit $?"
```

Keep `~/xs_test_m3`; ask the user before deleting it.

## Step 9. Report

```
Machine: M3 Max, macOS <version>, numpy <version> with <BLAS>
Build (step 4): ok / errors (file:line, message); new warnings: none / list
pytest (step 5): N passed / failures (test_resample: values differing, max difference)
Restart (step 6): new_jet: started at step N in k/4 events; verify: identical k/4 (or DIFFERS: …);
                  new_jet_off / new_pjet: OFF lines as expected yes/no
Comparison (step 7): 7 × identical / list of DIFF lines
Timings (s, wall of the job; per-event "done in" times):
  hydro-only   base … → new …   (… %)
  O+O          base … → new …   (… %)
  two-stage    base … → new_jet_off … → new_jet …   jet events: …
  particlize-only  base … → new …
Peak memory (GB): base_jet …, new_jet …, new_jetq …
Anything unexpected: …
```
