<!-- Plan, written 2026-10-11. Status: phase 1 done on branch cpu_fixes_a (not merged). -->

# Plan: the bit-identical CPU fixes (A1–A9 of REPORT_CPU_GPU.md)

## Context

REPORT_CPU_GPU.md (2026-10-11, 2 × RTX 3090) found every Au+Au and O+O production host-dominated
per event: 81–87 % of an event is host work. Its §5 lists changes in three groups:
- **A:** less CPU, nothing added on the GPU, bit-identical output.
- **B:** CPU work moved to the GPU.
- **C:** less GPU work.

This plan covers all of A. The savings below are core-s per event of one job alone on the 3090
(H = hydro only, C = two-stage campaign, P = two-stage plain):
- A1–A7: H ~12 of 40 (−30 %), C ~14 of 76 (−19 %), P ~13 of 133 (−10 %).
- With A8 as well: C ~40 (−53 %), P ~39 (−29 %).

Events/h rise only where the host sets the rate (the 3090 box, slots with few cores per GPU). On
the GB10 and the M3 Max they free cores, and only A8's GPU part raises events/h.

## Corrections to the report, from the code (MUSIC4GPU `bc8543a`, X-SCAPE `contrib`, js-contrib `main`)

- **A6:** the store already reserves its memory: `HydroinfoMUSIC::set_grid_infomatioin` reserves
  `nt/N + 2` frames. The cost is more likely page faults, in two places:
  - `evolve.cpp` creates a new 19 MB `std::vector<fluidCell_ideal> packed_frame` for every frame
    (a fresh mmap each time);
  - each frame is then copied into reserved store pages that have never been touched.

  A5 and A6 are therefore one change: pack straight into the end of the store, through one pinned
  staging buffer that is reused.
- **A1:** `pair_h5.py` decides which events share a background "from the data, never from
  `reuse`". Skipping the resample still needs a cheap fingerprint of the data:
  - the raw native frames of the background leg, without resampling;
  - on a match, the cached `bg_hashes`, background edge maxima and `bg_first`.

  The `full` layout writes every background frame again for every event, so it keeps the current
  path. The campaign (`shared`) and `--particlize-only` (no store) get the full saving.
- **A2:** "bit-identical" was checked only on the 3090 machine's BLAS. It is re-checked here
  against the current `einsum` code on every machine (OpenBLAS on Arm, Accelerate on the M3 Max).
  The prototype `resample_test/t_resample_fast.py` is not on the GB10.
- **A3:** besides the electric-charge list, the baryon list should also be ignored when
  `turn_on_rhob == 0`.
- **A4:** nothing reads `momentum_anisotropy_*.dat` / `eccentricities_*.dat`. `run_prod.py` only
  deletes them.
- **A7:** a GPU max(e) reduction already exists. A7 adds the same reduction over the edge bands
  only, with the threshold comparison done on the host in double, so the result stays exact. It
  needs both a CUDA and a Metal kernel.
- **A9:** switching off MUSIC_2's store must keep the τ bookkeeping (`itaumax`, `hydroTauMax`).
  Otherwise `ntau_jet_music`, which comes from `get_bulk_info`, changes.

## Phases

Each repo gets one PR against its final base (js-contrib `main`, X-SCAPE `contrib`, MUSIC4GPU
`XSCAPE`), with separate commits and no stacked PRs.

### Phase 1: js-contrib only (A1 + A2)

C ~−11 core-s per event. Should also stop the writer slowing down during long jobs.
- **A2:** `bulk_sources.resample()` does the three passes as matrix products over contiguous
  memory. A test compares it bit for bit with the `einsum` version, which is kept in the test
  file as the reference.
  - On the GB10 (one BLAS thread) the `einsum` code takes 11 ms per frame, not the 30–60 ms of
    the 3090 box. A plain contiguous rewrite was no faster here: the time goes into dense
    products whose weight matrices are ~96 % zeros, and into new buffers allocated for every
    frame.
  - As built: features leading, one matmul per pass into buffers reused for the whole call,
    and per block of 4 output rows only the source rows it touches. That takes 3.6 ms per
    frame (3.8 in the prod image), bit-identical (`tests/test_resample.py`).
  - Still to do: run `tests/test_resample.py` on the 3090 box (x86 OpenBLAS) and on the M3 Max
    (Accelerate).
- **A1:** `PairH5Writer._write_event` fingerprints the background from MUSIC_1's native frames
  without resampling them. On a match it reuses the cached hashes, edge maxima and `bg_first`.
  This applies only to the `shared` layout or `store_evolution=False`; `full` still resamples
  every event.
- **Validation:** `tests/test_pair_h5.py` and the production regression check
  (`run_prod_jet.py --reuse 2 --write-particlize both`), every dataset compared bit for bit
  (`source/flux` within a tolerance).
- **Done 2026-10-11 (GB10, branch `cpu_fixes_a`):**
  - Ran `run_prod_jet.py --events 4 --seed 1 --reuse 2 --write-particlize both` with js-contrib
    `309bb90` and with the branch. All 38 datasets of the pair file and all 33 of the
    particlize file are identical; only `prod_wall_s_total` and `file_uuid` differ.
  - Jet events with a reused background: 16.5 → 14.8 s and 20.4 → 18.0 s. Events with a new
    background: −1.0 and −1.4 s. User CPU of the job: 370 → 356 s.
  - The GB10's resampling was already fast, so the gain here is small. The 3090 box, where
    the report measured 6 s per jet event for the background alone, should gain more; not
    measured yet.

### Phase 2: small MUSIC4GPU fixes, plus the switch in X-SCAPE (A3, A4, A5/A6, A7)

- **A3:** `has_active_sources_current_tau()` follows `DATA.turn_on_QS` and `DATA.turn_on_rhob`.
- **A4:** a `DATA` parameter gates `output_momentum_anisotropy_vs_etas`. It stays on by default in
  MUSIC; MusicWrapper reads it from XML, and the production XMLs turn it off.
- **A5/A6:** in `CUDAPipelines::pack_evolution_ideal`, on GPUs without shared host memory only:
  - a device buffer plus one pinned buffer, filled with an async copy;
  - then copied into the store's tail through a new `HydroinfoMUSIC` append-in-place API.

  The GB10 (shared host memory) and Metal paths are unchanged. If faults on new store pages
  remain, add `madvise(MADV_HUGEPAGE)` on the reserved store.
- **A7:** an edge-band max(e) reduction (CUDA and Metal) in `transverse_edge_is_hot`, with no
  full-grid copy.
- **Validation:**
  - hydro-only `run_prod.py`, the two-stage regression check and a short O+O run, all bit for bit;
  - timing from `MUSIC_PROFILE` and py-spy on the GB10 (core-s per event);
  - events/h on the 3090 with the amd64 image built by CI.
- **Done 2026-10-11 (GB10, branches `cpu_fixes_a`):**
  - Commits: MUSIC4GPU `c8725c9` (A3), `111b14b` (A4), `c211513` (A5/A6), `4d83c10` (A7);
    X-SCAPE `9762530b`; js-contrib `b05b96d` (production XMLs).
  - As built, A5/A6 always packs into the store's tail and `madvise`s it to huge pages (not
    only "if faults remain"). On the GB10 the pinned path is off (shared host memory).
  - Output bit-identical to the old build: hydro-only Au+Au (3 events, seed 1), O+O (4 events,
    seed 101: hydro, particlize and in-job jet hadron files), and two-stage Au+Au (4 events,
    `--reuse 2`, pair and particlize files).
  - Wall time per job, old → new: hydro-only 38.7 → 32.6 s (CPU 99 → 84 s); O+O 22.0 → 18.7 s;
    two-stage 108 → 99 s (CPU 356 → 337 s).
  - Hydro-only MUSIC timers over 3 events:
    - string fills 100 → 12 (the 4 real ones per event);
    - freeze-out check 0.93 → 0.016 s;
    - no GPU → host grid copies (0.94 s before);
    - 19.8 → 16.4 ms per step.

### Phase 3: A9 (MUSIC4GPU + X-SCAPE + js-contrib)

- **MUSIC4GPU:** a per-instance `store_evolution` setter that skips the pack and the append but
  keeps the τ bookkeeping.
- **X-SCAPE:** exposed through MusicWrapper and pybind, like `set_dump_hydro_only`.
- **js-contrib:** `PairH5Writer.attach` turns it off on MUSIC_2 when `read_jet=False`.
- **Gain:** −2.8 s per particlize-only event, and no ~2 GB jet-leg store per job.

### Phase 4: A8, MUSIC_2 starting from MUSIC_1's state

A short design note comes first. Main decisions:
- **Checkpoints:** MUSIC_1 keeps 1–3 copies of its state on the GPU (~100 MB each: `snap_prev`,
  `snap_curr`, the source buffer). The first is right after the strings are deposited; the later
  ones are at configurable τ values.
- **Host state to carry over:**
  - the step index, `emax_last_check_` and the freeze-out bookkeeping;
  - the Cornelius previous arena, and the surface elements found so far (the dilute edge has a
    surface from the first steps);
  - the leading frames of the store.
- **Using a checkpoint:** MUSIC_2 starts from the latest checkpoint before its earliest droplet.
  The droplets are known by the time MUSIC_2 runs. If none qualifies, MUSIC_2 runs in full and
  logs it. A grid and parameter check comes first.
- **Fallback:** keep only the four string-fill buffers, so the CPU part is saved even when a
  droplet comes early. If a droplet falls in a string step, the sum must stay in double.
- **Code:** MUSIC4GPU (CUDA and Metal), X-SCAPE (`MusicWrapper` and a binding), and a
  `run_prod_jet.py` flag that stays off until validated.
- **Verification:** a `MUSIC_RESTART_VERIFY` mode that also runs the full MUSIC_2 and compares the
  states at the checkpoint bit for bit, plus pair and particlize comparisons over a few hundred
  events.
- **Gain:** C, P −26 core-s, and −6 to −13 % GPU on the jet leg.
