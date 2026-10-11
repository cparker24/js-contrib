# prod_OO_isr — O+O with initial-state radiation and jet energy loss, straight to HDF5

This folder runs O+O 200 GeV events with X-SCAPE's initial-state radiation (ISR) workflow:
the hard scattering's initial-state partons are showered backward in time by iMATTER, the
3D MC-Glauber strings are made from what the hard scattering left, MUSIC (music4gpu) evolves
that medium, and Matter showers the jet through it. There is **one** MUSIC run per event (no
two-stage hydro, no jet deposition). The hadrons can be made in the same job, or the bulk
later from stored freeze-out surfaces (see [Hadronization](#hadronization-in-the-job-or-afterwards)).

```
IS: 3dMCGlauber (O16, positions only)
 -> Hard: PythiaIsrGun at one binary collision
          -> IsrManager / iMATTER       ISR, backward from 0 to -maxT
          -> MCGlauberGenString         strings, the hard partons' momentum subtracted
 -> NullPreDynamics -> MUSIC            the medium
 -> Eloss: Matter in MUSIC's medium     the final-state shower, from -maxT
 -> JetHadronization: iColored          jet hadrons (final partons + beam remnants)
 -> SoftParticlization: iSS             bulk hadrons, only with --bulk-hadrons
```

The chain is what X-SCAPE's `examples/custom_examples/PythiaIsrMUSIC.cc` builds by hand.
Here X-SCAPE builds it from the XML: a `<Hard><ISR>` block (next to `<PythiaGun>`) switches
the automatic task list to it.

> **Status (2026-10-10, branches `oo_isr_test` in X-SCAPE and js-contrib).** Runs on
> `build_gpu` (music4gpu, CUDA, GB10). It needs X-SCAPE's `<Hard><ISR>` task list,
> `<JetHadronization><name>iColored` and the iSS fix for events without a medium (all on
> branch `oo_isr_test`).
>
> **Tuned O+O parameters.** The 3dMCGlauber and MUSIC settings in
> `OO_MCGlauber_MUSIC_isr.xml` are the tuned O+O set of X-SCAPE's
> `config/BulkFastTest/OO_one_event.xml` (O16 configurations from `nucleusConfigs`,
> `tau_form_mean` 0.5, MUSIC's viscosities and freeze-out), tuned at √s_NN = 200 GeV. Three
> settings differ on purpose: EOS 9 instead of 91 (as the Au+Au production: with 91 iSS
> leaves the resonances undecayed, see [Physics notes](#physics-notes-and-limits)),
> `generateOnlyPositions` 1 (the strings are made after the ISR) and b ≤ 3 fm.

| file | purpose |
|---|---|
| `OO_MCGlauber_MUSIC_isr.xml` | user XML: tuned 3dMCGlauber + MUSIC for O+O, `<Hard>` with `<PythiaGun>` + `<ISR>`, Matter, iColored |
| `grid_oo.yaml` | output grid of the hydro file: 64 × 64 × 32 at 0.3125 fm × 0.3125 fm × 0.3125 (x, y ±9.84 fm) |
| `run_prod_oo_isr.py` | one job: one seed, N events |
| `run_jobs.sh` | many jobs, `-j P` at a time, resumable (wraps `../prod_AuAu_0_10/run_jobs.sh`) |
| `oo_isr_overview.ipynb` | one job's output: the medium's energy density at three τ (with the freeze-out contour and the dijet), and η, φ, pT and Δφ-to-the-leading-parton distributions of the bulk and jet hadrons |

## Run

```bash
conda activate js_fno        # sets PYTHIA8DATA: iMATTER reads Pythia's PDF tables next to it
cd external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr

python run_prod_oo_isr.py --events 1 --seed 1 --dry-run     # check the XML/grid, print the plan
python run_prod_oo_isr.py --events 10 --seed 1              # pTHat 20-40 GeV, b < 3 fm
python run_prod_oo_isr.py --events 10 --seed 1 --pthat-min 10 --pthat-max 20
python run_prod_oo_isr.py --events 10 --seed 1 --b-max 5    # more peripheral
python run_prod_oo_isr.py --events 10 --seed 1 --sqrts 5360 # LHC energy (tuned at 200 GeV)
python run_prod_oo_isr.py --events 30 --seed 1 --pthat-bins 10-20,20-40,40-60
                                                            # 3 pTHat windows, every event complete

./run_jobs.sh 20 25 0                                       # 20 jobs x 25 events, unique seeds -> ./out
./run_jobs.sh -j 2 20 25 1 out_had --bulk-hadrons 100 --no-particlize
                                                            # seeds 1..20, all hadrons made in the jobs
```

**Look at a job:** [`oo_isr_overview.ipynb`](oo_isr_overview.ipynb) reads `<stem>.h5`,
`<stem>_hadrons_bulk_bg.h5` and `<stem>_hadrons_jet_frag.h5` (stem from `OO_STEM`, default
`out/OO_isr_seed0001`); it is stored with the output of a 20-event job made with
`--events 20 --seed 1 --bulk-hadrons 100 --no-particlize`.

```bash
OO_STEM=out/OO_isr_seed0001 jupyter nbconvert --to notebook --execute --inplace oo_isr_overview.ipynb
```

**No hydro reuse, also with pT̂ windows.** With ISR the strings, and so the medium, are made
from what each event's hard scattering left (its partons' momentum is subtracted from the
colliding nucleons), so a medium belongs to one hard event. Every event runs the whole chain
(3dMCGlauber, ISR, strings, MUSIC, Matter, hadronization); X-SCAPE refuses
`<setReuseHydro>` with ISR (`JetScape::Init`), and this driver has no `--reuse`.
`--pthat-bins 10-20,20-40,40-60` runs one Pythia per window (PythiaIsrGun's `<pTHatBins>`,
as PythiaGun's) and event *i* uses window *i* mod K; unlike the Au+Au `--pthat-bins`, no
background is shared between the windows. Per event `pthat_bin` is recorded (`diag/`,
`events/`); per window the file attributes `pthat_bins`, `pthat_bin_sigma_gen`,
`pthat_bin_sigma_err`, `pthat_bin_n_accepted`, `pthat_bin_seeds` (and the `.json`) give the
cross sections for weighting. Use a multiple of K events for the same number per window.

The seeds, campaign names, the seed registry and the working directories are those of the
Au+Au productions: see [`../prod_AuAu_0_10_jet/README.md`](../prod_AuAu_0_10_jet/README.md)
(*Campaigns with `run_jobs.sh`*). Every option of `run_prod_oo_isr.py` can follow the
numbers of `run_jobs.sh`. The same options and seed give the same files, bit for bit.

## Containers

The production image (`utils/Dockerfile.prod`, [`docs/README_2stage.md`](../../../../docs/README_2stage.md))
starts in `prod_AuAu_0_10_jet`. For O+O, point its working directory at this folder, or call
the scripts by their full path: they find their XML, grid and helpers next to themselves.

```
/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
```

> **Needs a new image.** The published `xscape-prod` images (X-SCAPE `d31946c`, js-contrib
> `main`) have neither this folder nor X-SCAPE's `<Hard><ISR>` task list. Build one from the
> `oo_isr_test` branches (pushed to GitHub: the Dockerfile fetches the refs), or from
> `contrib` / `main` once they are merged:
>
> ```bash
> docker build -t xscape-prod:oo-test -f utils/Dockerfile.prod \
>   --build-arg XSCAPE_REF=oo_isr_test --build-arg JS_CONTRIB_REF=oo_isr_test .
> ```

**Docker:** `-w` sets the working directory.

```bash
OO=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=4 \
  -v "$HOME/prod:/work" -w "$OO" $IMG \
  ./run_jobs.sh -j 2 20 25 0 /work/OO_a --bulk-hadrons 100 --no-particlize
```

**Apptainer:** `--pwd` sets it (Apptainer ignores the image's working directory anyway).

```bash
OO=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr
apptainer exec --nv --env OMP_NUM_THREADS=4 \
  --pwd "$OO" --bind "$SCRATCH/prod:/work" xscape_prod.sif \
  ./run_jobs.sh -j 2 20 25 0 /work/OO_a --bulk-hadrons 100 --no-particlize
```

- **Without changing the directory:** `"$OO"/run_jobs.sh ...` or `python "$OO"/run_prod_oo_isr.py ...`
  work from anywhere, e.g. from the default `prod_AuAu_0_10_jet`.
- **Output under the bound `/work`** (`/work/OO_a`): the default `./out` is inside the
  image, read-only under Apptainer. The seed registry goes next to OUTDIR, so `/work` too.
- **The bulk afterwards** (setup C of [Hadronization](#hadronization-in-the-job-or-afterwards)): `hadronize.py` is in the Au+Au folder, e.g. with
  `-w "$OO"`: `../prod_AuAu_0_10_jet/run_hadronize.py /work/OO_a -j 8 --tags bulk_bg --oversample 100`
  (no GPU needed).
- **Nothing to set for iMATTER:** the image sets `PYTHIA8DATA`, and iMATTER finds Pythia's
  PDF tables next to it.
- **The notebook** `oo_isr_overview.ipynb` needs Jupyter, which the image does not have: run
  it on the host or in the analysis venv ([`utils/analysis_env`](../../../../utils/analysis_env/README.md))
  on the files in `/work`.

## Hadronization: in the job or afterwards

An O+O event's hadrons come in two files, both in the `jetscape.hadrons_h5` format of the
Au+Au productions:

| hadrons | file | made by | when |
|---|---|---|---|
| **jet** | `<stem>_hadrons_jet_frag.h5` | iColoredHadronization: the final partons and the beam remnants, through the ISR colour flow | **always in the job** (unless `--no-jet-hadrons`) |
| **bulk** | `<stem>_hadrons_bulk_bg.h5` | iSS on MUSIC's freeze-out surface, N oversamples per event | **in the job** with `--bulk-hadrons N`, **or afterwards** with `hadronize.py --tags bulk_bg` on the particlize file |

The jet hadrons cannot be made afterwards: iColoredHadronization needs the event's beam
remnants (momenta from 3dMCGlauber) and the ISR colour connections, which exist only during
the job. The bulk can be made either way. Choose one of three setups:

| setup | command | files | afterwards |
|---|---|---|---|
| **A. everything in the job**, no surfaces kept | `run_prod_oo_isr.py ... --bulk-hadrons 100 --no-particlize` | `.h5`, `_hadrons_bulk_bg.h5`, `_hadrons_jet_frag.h5` | nothing to do |
| **B. in the job, surfaces kept** | `run_prod_oo_isr.py ... --bulk-hadrons 100` | A + `_particlize.h5` | more oversamples or other iSS settings any time |
| **C. bulk afterwards** (the default) | `run_prod_oo_isr.py ...` | `.h5`, `_particlize.h5`, `_hadrons_jet_frag.h5` | `hadronize.py --tags bulk_bg` |

Add `--no-hydro-file` to any of them to leave out the hydro evolution (`<stem>.h5`): MUSIC
still runs and Matter still reads its medium, only nothing of it is stored. A + `--no-hydro-file`
is a hadrons-only job: `_hadrons_bulk_bg.h5`, `_hadrons_jet_frag.h5`, `.json`, `.xml`.

```bash
# A: all hadrons in the job, nothing stored to hadronize later
python run_prod_oo_isr.py --events 10 --seed 1 --bulk-hadrons 100 --no-particlize
# A without the hydro evolution: hadrons only
python run_prod_oo_isr.py --events 10 --seed 1 --bulk-hadrons 100 --no-particlize --no-hydro-file
# A with smaller hadron files (hadronize.py's selection options, on both hadron files)
python run_prod_oo_isr.py --events 10 --seed 1 --bulk-hadrons 100 --no-particlize \
       --eta-max 2 --charged --keep-bits-p 12

# C: the bulk afterwards, on CPUs, from the stored surfaces
python run_prod_oo_isr.py --events 10 --seed 1
python ../prod_AuAu_0_10_jet/hadronize.py out/OO_isr_seed0001_particlize.h5 \
       --tags bulk_bg --oversample 100
python ../prod_AuAu_0_10_jet/run_hadronize.py out -j 4 --tags bulk_bg --oversample 100
                                                    # every particlize file in out/
```

Options for the hadrons made in the job:

| option | effect |
|---|---|
| `--bulk-hadrons N` | run iSS in the job, N oversamples per event → `<stem>_hadrons_bulk_bg.h5` (default 0: no bulk hadrons in the job) |
| `--hadronize-xml FILE` | iSS settings for `--bulk-hadrons` (default `../prod_AuAu_0_10_jet/hadronize.xml`, the settings `hadronize.py` runs, so in-job and offline bulk are sampled alike) |
| `--no-particlize` | no `<stem>_particlize.h5`: no surfaces, partons or initiators stored |
| `--no-hydro-file` | no `<stem>.h5`: the hydro evolution is not stored (`hydro_row` is then -1 for every event) |
| `--no-jet-hadrons` | no iColoredHadronization, no `<stem>_hadrons_jet_frag.h5` |
| `--eta-max X`, `--charged`, `--no-x`, `--keep-bits-p B`, `--keep-bits-x B` | store only \|η\| < X / charged hadrons / no positions / rounded p, x — as `hadronize.py`, on both in-job hadron files; iSS and Pythia still sample everything |

**An event at hadron level** is unit *i* of the jet file plus unit *i* of the bulk file
(units are events in both): sample *k* of the bulk together with the jet hadrons of the same
event. The *N* bulk oversamples share one fluid: average over them, they are not independent
collisions.

Good to know:
- **In the job and afterwards agree.** Both run iSS with `hadronize.xml`'s settings; only
  the seeds differ. In a 3-event test with 50 oversamples each, dN/dη at |η| < 0.5 per event
  was 32.0 / 37.9 / 32.1 in the job and 32.0 / 36.8 / 31.2 from `hadronize.py`; ⟨pT⟩, the
  charged fraction and the π⁺, K⁺, p yields agree as well. The in-job file has the attribute
  `inline`, and each unit its iSS `seed`, `n_cells` and `hydro_row`.
- **B and C combined.** `hadronize.py` keeps an existing `_hadrons_bulk_bg.h5` (it stops
  with "exists"): write the offline set elsewhere (`--out-dir`) or replace it (`--force`).
- **Always `--tags bulk_bg` with `hadronize.py`.** It refuses `jet_frag` on these
  particlize files (attribute `jet_frag_inline`): ColorlessHadronization would drop the hard
  partons (status 1000, see below) and overwrite the in-job jet hadrons.
- **`--bulk-hadrons` and `--no-hydro-file` change the jet showers' random numbers.**
  X-SCAPE seeds each task from the order it was made, and the per-parton Matter copies are
  made during the event, after the job's other tasks: iSS (`--bulk-hadrons`) and the hydro
  writer (absent with `--no-hydro-file`) change their number, so they get other seeds. The
  showers are then a different, equally valid sample (the hydro is identical). Keep one
  choice per campaign if jets of different jobs are to be compared event by event.
- **Events without a medium** (no string, `hydro_row` -1) get an empty bulk unit; the run
  goes on.

## Output

Each job writes, next to each other (stem `OO_isr_seedNNNN`, or `OO_isr_<campaign>_NNNN`):

| file | content | written by |
|---|---|---|
| `<stem>.h5` | MUSIC's evolution (e, vx, vy, vz) on `grid_oo.yaml`, read from the framework copy Matter also reads; `diag/` per event: `event`, `pthat`, `sigma_gen`, `sigma_err`, `event_weight`, `glauber_npart`, `glauber_ncoll`, `glauber_nstrings`, `glauber_b`, `tau0_music`, `hit_grid_boundary` (`--no-hydro-file`: not written) | `jetscape.fast_h5_bulk.H5BulkWriter` (`grid_mode="framework"`) |
| `<stem>_particlize.h5` | MUSIC's freeze-out surface (`surface/bg`, one unit per event), the final partons (`partons/`) and the shower initiators; `events/` as `diag/` plus `hydro_row`, `wall_s` (`--no-particlize`: not written) | `jetscape.particlize_h5.ParticlizeH5Writer` (`legs=("bg",)`) |
| `<stem>_hadrons_bulk_bg.h5` | the bulk hadrons: in the job with `--bulk-hadrons N`, else later from `hadronize.py` | `jetscape.hadrons_h5.HadronH5Writer` |
| `<stem>_hadrons_jet_frag.h5` | the jet hadrons of iColoredHadronization, one unit per event, with the initiators | `jetscape.hadrons_h5.HadronH5Writer`, in the job |
| `<stem>.xml`, `.json` | the job XML; a summary (events, cross section, outputs, wall time) | |

- **`hydro_row`** links the files: the row of the event in `<stem>.h5`, or -1 when it has
  none: for an event without a medium, and for every event with `--no-hydro-file`.
  **`has_medium`** (1/0, in `diag/`, `events/` and the hadron files' `units/`) says which:
  without a medium (no string) there is no evolution, Matter showers in vacuum, and the
  surface and bulk unit are empty.
- **The jet hadrons** are those of the whole hard-scattered nucleon pair: dijet, Pythia's MPI
  partons, the ISR emissions and the remnants.
- **Final-parton status codes.** PythiaIsrGun marks the outgoing hard partons 1000 and
  Matter's daughters keep that status, so the showered jet partons are mostly status 1000;
  the showers of the ISR emissions are status 0. Status 1000 does not mean "not showered".

Measured on the GB10 (2026-10-10, one job, default settings): ~5 s per event, 2.8 GB peak
memory; per event ~17 MB hydro file, ~10 MB particlize file. `--bulk-hadrons 50` adds ~0.3 s
per event (~5,600 hadrons per event over the 50 oversamples).

## Physics notes and limits

- **EOS 9, not BulkFastTest's 91.** With EOS 91 iSS samples the SMASH hadron list and
  leaves the resonance decays to SMASH, which neither the job nor `hadronize.py` runs: ρ, ω,
  K*, Δ, η ... would stay undecayed (`hadronize.py` warns). In a 3-event test the midrapidity
  dN/dη was 15.5 per sample with EOS 91 against 34.8, all decayed, with EOS 9. EOS 9 (UrQMD
  list) is also what the Au+Au productions run.
- **One hard scattering per event**, at a binary collision of 3dMCGlauber drawn uniformly.
  The events are therefore hard-triggered with no weight for the number of collisions:
  `glauber_ncoll` is recorded per event for that. `b` selects the centrality
  (`cenMin`/`cenMax` act only in the standalone 3dMCGlauber); `b < 3 fm` is about 0–20% of
  the O+O cross section.
- **The hard nucleon pair makes strings only if it is not yet connected**
  (`Glauber::decide_produce_string_num`), and its valence quarks and remnant lose the hard
  partons' momentum. Peripheral events can end up with no string at all (`hydro_row` -1);
  in p+p this is most events (9 of 12 in a test).
- **Jet energy loss is Matter alone.** LBT gives its partons no colour, which iColored
  hadronization cannot take: Matter + LBT stops with "Unpaired colors sent to Pythia"
  (tested). `run_prod_oo_isr.py` refuses `<Eloss><Lbt>` unless `--no-jet-hadrons`. Colorless
  hadronization, which would take LBT, does not work with ISR yet: see the next section.
- **ISR range.** iMATTER runs from 0 to -`<Eloss><maxT>` (50 fm/c, as in X-SCAPE's ISR
  examples) and the final-state shower starts there; `<Hard><ISR><maxT>`/`<deltaT>` override.
- **Not included:** X-SCAPE branch `Changing_iMatter_pTDefinition` (upstream, August 2025,
  not merged) changes iMATTER's sibling-pT and recoil kinematics; this branch runs iMATTER as
  it is on `contrib`.

## Not working yet: Matter + LBT with ColorlessHadronization

The Au+Au production's combination, Matter + LBT with ColorlessHadronization (here without
energy deposition, i.e. no liquefier), **runs with ISR but gives wrong jet hadrons**. It is not
an option of `run_prod_oo_isr.py`. Tested 2026-10-10 with `runJetscape` on this chain
(`<Hard><ISR>`, `<Eloss>` Matter + LBT, `<JetHadronization><name>colorless`), 5 events each:

| run | jet-parton energy with status 1000 (dropped) | energy Colorless takes | events with no jet hadrons |
|---|---|---|---|
| Matter + LBT | 45–92 GeV/event | 18–77 GeV/event | 0 of 5 |
| Matter only | 66–136 GeV/event | 0–44 GeV/event | 3 of 5 |

Two problems:

1. **Colorless drops the hard-scattered partons.** ColorlessHadronization takes only pstat 0
   (shower), 1 (recoil, `take_recoil`), 22 (photon) and -1 (hole, subtracted). With ISR the
   hard partons from PythiaIsrGun and all their Matter daughters have status 1000, so they are
   left out silently. With LBT, the partons LBT took over come back with status 0 and are
   kept; what left the medium, or never reached LBT, stays 1000 and is lost. With Matter alone
   almost nothing is hadronized.
2. **The fake beam remnant gets the wrong energy.** Colorless closes open strings with a fake
   remnant of momentum ~√s/6 along the beam, from `<JetHadronization><eCMforHadronization>`,
   whose main-XML default is 5020 GeV: such events carry 840–1740 GeV of hadron energy. The
   Au+Au `hadronize.xml` sets it to 200.

What a fix would take:
- X-SCAPE `src/hadronization/ColorlessHadronization.cc`: take status 1000 like status 0 (one
  line; only PythiaIsrGun makes status 1000, so other workflows are unchanged).
- The O+O XMLs: `<eCMforHadronization>` 200 with colorless.
- `run_prod_oo_isr.py`: a `--jet-hadronization colorless` choice that allows `<Eloss><Lbt>`.
- With Colorless the jet hadrons need no live event (no remnants): `hadronize.py --tags
  jet_frag` could then make them later from the particlize file's partons, and the
  `jet_frag_inline` guard would apply to iColored only.

Even fixed, Colorless with ISR ignores the ISR colour connections and the 3dMCGlauber beam
remnants: it hadronizes the final partons as independent strings, as the Au+Au Matter + LBT
production does, with LBT's holes and recoils through `take_recoil`.
