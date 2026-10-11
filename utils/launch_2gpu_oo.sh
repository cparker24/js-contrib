#!/usr/bin/env bash
# launch_2gpu_oo.sh: prod_OO_isr (O+O 200 GeV with ISR and Matter, see its README.md) on both GPUs
# of this machine (2 x RTX 3090, 24 cores), with the best measured setup: one container per GPU,
# each running run_jobs.sh -j 8 --mps with OMP_NUM_THREADS=3 and MUSIC_CUDA_SYNC=block
# (~3,500 events/h, ~37 GB host memory; README_launch.md, "O+O").
#
#   ./launch_2gpu_oo.sh [launcher options] NJOBS EVENTS_PER_JOB [FIRST_SEED] [OUTBASE] [run_prod_oo_isr.py args...]
#   ./launch_2gpu_oo.sh 16 15 1 /work/oo_test --bulk-hadrons 10 --no-particlize --no-hydro-file
#                    # a test: seeds 1..16 (GPU0 1..8, GPU1 9..16), hadrons only, ~5 min
#   ./launch_2gpu_oo.sh 64 100 0 /data/OO_had --campaign had1 --bulk-hadrons 100 --no-particlize \
#       --no-hydro-file
#                    # hadrons only (bulk with 100 oversamples + jet hadrons): 6,400 events, ~2 h;
#                    # ~0.15 MB per event with 10 oversamples, roughly 1.3 MB with 100 (not measured)
#   ./launch_2gpu_oo.sh --upload OO_c1 --delete all --min-free 100G 64 100 0 /data/OO_c1 --campaign c1
#                    # default outputs (<stem>.h5 hydro, _particlize.h5, _hadrons_jet_frag.h5; the
#                    # bulk later with hadronize.py), uploaded to osdf:///fno4hic/OO_c1/gpuN while
#                    # running, hydro and particlize files deleted locally once verified.
#                    # ~29 MB per event: ~105 GB/h, so upload (or a large disk) is needed for hours.
#
# Measured (2026-10-11, image cu124-20261011-84df104, 15 events/job, hadrons only):
# 8, 12, 16 and 20 jobs/GPU all give ~3,500 events/h (makespan); more jobs only make each event
# slower and cost more CPU.  The host is the limit (24 cores; the GPUs ~65 % busy), not memory
# (~2.5 GB per job).  The default outputs run as fast (~3,650 events/h at 12/GPU).
# NJOBS a multiple of 16 keeps both GPUs busy to the end.
#
# Everything else is launch_2gpu.sh with PROD=oo: the same launcher options (--upload, --delete,
# --keep-free, --verify-every, --stagger, --min-free; not --lbt-cache: no LBT here), OUTBASE,
# seeds, resume and Ctrl-C behaviour; see its header (./launch_2gpu.sh --help) and README_launch.md.
# Environment overrides: JOBS_PER_GPU (8), OMP_THREADS (3), MUSIC_CUDA_SYNC (block), IMAGE (needs
# js-contrib 8765281 or later, e.g. jhputschke/xscape-prod:cu124 from 2026-10-11), WORKDIR.
# A job that throws in an event skips that event (Matter's "minus log of sudakov negative",
# ~1 in 400) and counts it as done (events_skipped in the .json).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD=oo LAUNCH_HELP="$0" exec "$HERE/launch_2gpu.sh" "$@"
