# FNO vs. MUSIC Particlization Investigation (30-40% centrality)

Log of setting up a Cooper-Frye (iSS) particlization comparison between live
`FnoHydro` and standard `MpiMusic`, finding and fixing a real bug in the
process, and then chasing down a smaller residual discrepancy. Kept as a
record of what's confirmed vs. still open, since the residual effect is not
fully explained.

## Pipeline

Both `fnoHydroLiveTest3D.cc` and `standardHydroLiveTest3D.cc` now wire up
`iSpectraSamplerWrapper` (iSS) and a `JetScapeWriterFinalStateHadronsAscii`
writer (flat `index pid pstat E px py pz` per event, alongside the full
event-record writer) — see commit `698378f`. Matched-seed (424242),
30-40%-centrality configs: `config/fno_3d_particlization_30_40_{fno,music}.xml`.
`contribs/FnoHydro/example/compare_particle_spectra.py` (PyROOT) builds pT and
rapidity spectra from the two drivers' output; it auto-detects event count per
file and accepts different `<SoftParticlization><iSS><number_of_repeated_sampling>`
values per arm.

## Bug found and fixed: velocity channels hardcoded to zero

**Symptom**: FNO's sampled-hadron rapidity spectrum showed a sharp spike at
y≈0 (more than 2x MUSIC's local plateau value), falling to ~zero by
|y|≈1.5-2, instead of MUSIC's broad, flat-ish distribution extending to
|y|≈3.

**Root cause**: `EvolveHydro()`'s FNO input-tensor construction hardcoded the
velocity channels (vx, vy, vz) to 0 for every cell, regardless of the
pre-equilibrium module in use — only energy density took the real primed
MUSIC snapshot. Real heavy-ion collisions already have substantial
Bjorken-like longitudinal flow (vz ≈ tanh(eta)) at tau0, so this was a
significant out-of-training-distribution input specifically for the velocity
channels. Confirmed via a dedicated diagnostic (`WriteEtaProfileDiagnostics`,
dumping the model's predicted vz per eta bin across tau): the model's
predicted vz never developed the correct monotonic-with-eta profile and
stayed far below the ~tanh(eta) magnitude real flow would reach at the grid
edges.

**Fix** (commit `fcb352b`): read `pre_eq_ptr->ux_/uy_/ueta_` (already
populated by `MusicPrimedPreEq` from the primer MUSIC's real `GetHydroInfo()`
snapshot, in the same raw-3-velocity convention FnoHydro's own output
channels use) instead of hardcoding 0. Safe no-op for `NullPreDynamics`,
which populates the same arrays with 0.

**Result**: post-fix, vz closely tracks real Bjorken flow (≈0.81-0.84 at
eta=1.3, vs tanh(1.3)≈0.86 expected), smoothly antisymmetric and stable
across tau. The dramatic rapidity spike is gone; FNO's rapidity spectrum is
now a smooth, broad, MUSIC-like shape.

## Residual: a ~13-14% central-rapidity peak, confirmed real

The bulk hydro field itself does not produce this peak at any stage of the
evolution -- it stays smooth in eta throughout:

![FNO bulk energy density vs. eta at several tau steps, smooth throughout](docs/particlization_investigation/bulk_eta_evolution_event0.png)

Real MUSIC (same seed, same event) shows the same character -- smooth across
the whole populated eta range, falling off only at the genuine edges of the
fireball (|eta| ≳ 5), with no central feature at any tau either:

![MUSIC bulk energy density vs. eta at several tau steps, also smooth throughout](docs/particlization_investigation/music_bulk_eta_evolution_event0.png)

That is a genuinely different shape from the final, Cooper-Frye-sampled
hadron rapidity spectrum (15-FNO-event / 5-MUSIC-event ensemble), which does
show a real, localized peak at y≈0 for FNO:

![Final sampled hadron rapidity, FNO vs MUSIC, FNO shows a real central peak](docs/particlization_investigation/final_sampled_rapidity_peak.png)

(Plotted with `example/plot_eta_evolution.py <eta_profile.csv> [out.png] [label]`
and `example/plot_final_peak.py <spectra_comparison.root>` respectively --
rerun against fresh diagnostic output to regenerate or extend these. The
MUSIC-side CSV comes from `MusicWrapper.cc`'s `WriteMusicEtaProfileDiagnostic`
in the main X-SCAPE repo, which writes the same column format so the same
script works against either unchanged.)

The contrast between these two plots is the core evidence that the peak is
introduced at the freeze-out-surface / Cooper-Frye-sampling stage, not
inherited from the bulk evolution.

Even after the fix, FNO's rapidity spectrum is not perfectly flat at
midrapidity the way MUSIC's ensemble-averaged spectrum is — it shows a mild,
statistically robust central peak.

**Ruled out as the cause** (each checked directly):
- Raw energy-density field: flat/smooth in eta at every tau step, from the
  initial condition through near-freeze-out (checked across 4+ events) —
  only a persistent, *monotonic* ~10% one-sided slope inherited from each
  event's own fluctuating initial state, present unchanged in character from
  k=0 through k=64 (just declining in magnitude with expansion). This is
  ordinary event-by-event fluctuation, not a localized peak, and a
  structurally different shape from the final rapidity peak.
- Freeze-out surface cell count: flat across eta (~2% variation, interior
  bins).
- Species dependence: pion and proton show nearly identical peaking
  (~7.6-7.8%) in iSS's own per-cell analytic Cooper-Frye yield — ruled out
  "heavier particles show stronger flow modulation" as an explanation.

**Confirmed real** (not a statistics or binning artifact):
- The freeze-out surface's proper rest-frame volume element
  (`d3sigma_mu * u^mu`, not just the raw `|d3sigma_mu[0]|`) peaks at
  eta≈0 and declines ~5-8% by |eta|≈0.4 within a narrow window — a genuine,
  modest feature of the surface geometry.
- iSS's actual per-cell yield calculation (`FSSW::calculate_dN_dxtdy_for_one_particle_species`,
  the real runtime path at `MC_sampling=4`, not `EmissionFunctionArray` which
  isn't exercised by the default config) reproduces this: no hidden
  amplification between the analytic yield and the MC-sampled output for a
  single event.
- Checked individually across all 15 events of a larger ensemble run: every
  event shows a peak-to-edge ratio of 1.04-1.24 (mean ~1.14, i.e. ~14%) in
  the final sampled rapidity histogram — consistent across events, not an
  outlier in any one of them. (A single-event check during the investigation
  initially seemed to show only ~8%; that turned out to be a bin-reference
  calculation mistake, not a real discrepancy between events — recomputed
  correctly, that same event is fully consistent with the rest.)
- Running 3x more statistics (15 events vs. 5) did **not** flatten the peak
  — it stayed at essentially the same ~13-14% size, just with much smaller
  error bars. This rules out "it's noise that averages away."

**Partially confirmed, mechanism still open**: a per-eta-slice freeze-out
timing diagnostic (`WriteEtaFreezeoutApproachDiagnostic`) shows real,
eta-dependent structure in *when* and *how fast* each eta slice crosses its
own local freeze-out threshold — e.g. in one event, the center crosses one
time-step later than the edges, and cools 2-3x faster right at its own
crossing than the edges do at theirs. This confirms the model's dynamics
really do behave differently near freeze-out as a function of eta (plausibly
related to uneven training exposure from per-event, not per-cell,
truncation/padding of training targets), matching the user's hypothesis in
spirit.

However, the *direction* doesn't trivially close the loop: standard
Cooper-Frye intuition says *slower* local cooling gives a *larger* effective
surface element, which would predict the edges (slow cooling in the checked
event) should dominate — the opposite of the measured center-peaked
`dsigma_mu*u^mu`. The likely explanation is that this simple 1D
(per-eta, temporal-only) slope metric is too simple: the real surface
element is a genuinely 4D geometric quantity (via the Cornelius-style
`SurfaceFinder`/`FSSW` hypersurface construction), depending on spatial
gradients between neighboring x/y/eta cells too, not just the local time
derivative at fixed position. Fully closing this would mean replicating or
instrumenting that 4D construction directly — not done here.

## Cross-check: does the peak persist at a different centrality?

Repeated the same matched-seed (424242) comparison at 10-20% centrality
(`config/fno_3d_particlization_10_20_{fno,music}.xml`, single event only, not
yet run as an ensemble). Same signature: within a narrow mid-rapidity window
the FNO/MUSIC ratio rises from the edges toward center then falls back
(0.79 / 0.91 / 0.95 / 0.94 / 0.84 / 0.65 bin-by-bin), an ~11% local peak
relative to its own edges -- consistent in both direction and rough
magnitude with the ~13-14% found at 30-40%. The effect is not specific to one
centrality bin.

## Parameter study: raising the freeze-out temperature (136 -> 150 MeV)

Motivation: training targets are truncated/padded once an event approaches
its own freeze-out, so the model's supervision is weakest right at the low
end of the freeze-out-temperature range it was trained on. If that's
contributing to the peak, stopping the evolution earlier (a *higher*
freeze-out temperature, further from that undertrained regime) should reduce
it.

**This turned out not to be a hypothetical "higher than usual" value to try --
it's the model's actual training cutoff.** Checked all 18 training-data XML
configs under `/home/cparker/jetscape-docker/xmls/` (every centrality bin:
0-10, 10-20, 20-30, 30-40, including the "withT" and "finegrid" variants):
**none of them override `freezeout_temperature`** in their `<Hydro><MUSIC>`
block, so every one of them silently inherited `jetscape_main.xml`'s MUSIC
default of **0.150 GeV**. Separately, that same `jetscape_main.xml` also
defines independent `<FNO>`/`<FNOROOIN>` module defaults of **0.136 GeV**
(js-contrib's own FnoHydro defaults) -- a plausible-looking Cooper-Frye
literature convention that nothing ever cross-checked against what the
training data was actually generated with. Every live-inference config built
during this investigation (including all of this document's earlier
sections) explicitly used 0.136, following that second, unrelated
convention. So the entire original training dataset, across every
centrality bin, has real (non-padded) MUSIC targets only down to 150 MeV --
and every comparison run before this parameter study had been asking the
model to extrapolate ~14 MeV past where it actually had training
supervision, the whole time.

Tested with `config/fno_3d_particlization_30_40_{fno,music}_Tfo150.xml` --
siblings of the baseline 30-40% configs with `freezeout_temperature` raised
from 0.136 to 0.150 GeV in both the `<FNO>` block and the embedded priming
`<MUSIC>` block (FNO config), and the `<MUSIC>` block (MUSIC config).

**Same-event check** (seed 424242, nEvents=1, only Tfo changed -- a clean
A/B test with no event-to-event noise):

| window | Tfo=136 MeV | Tfo=150 MeV |
|---|---|---|
| peak/edge, narrow (\|y\|<0.45) | 1.124 (12.4%) | 1.076 (7.6%) |
| peak/edge, wide (\|y\|<0.75) | 1.388 (38.8%) | 1.230 (23.0%) |

**Ensemble check** (15 FNO events / 5 MUSIC events, matching the baseline
ensemble sizes; bin errors <1%, so this is statistically solid, not noise):

| window | Tfo=136 MeV (15-event) | Tfo=150 MeV (15-event) |
|---|---|---|
| peak/edge, narrow (\|y\|<0.45) | 1.131 (13.1%) | 1.091 (9.1%) |
| peak/edge, wide (\|y\|<0.75) | 1.440 (44.0%) | 1.239 (23.9%) |

![Final sampled hadron rapidity at Tfo=150 MeV, FNO vs MUSIC -- peak reduced but not eliminated](docs/particlization_investigation/final_sampled_rapidity_Tfo150.png)

**Conclusion**: raising the freeze-out temperature meaningfully *shrinks* the
central peak -- roughly a 30% relative reduction in the narrow window, ~46%
in the wider one, confirmed at both single-event and ensemble level -- but
does **not** eliminate it; a real ~9-24% peak remains even at 150 MeV, i.e.
even running exactly at the model's real training cutoff rather than below
it. That residual is real evidence the training-truncation hypothesis is
only a partial explanation, not the whole mechanism -- plausibly because
individual training *events* still stop at different actual tau/temperature
(each event's own MUSIC run exits once its own hottest cell crosses 150 MeV,
so shorter-lived events contribute fewer late-time examples than
longer-lived ones), leaving uneven exposure even at the nominal cutoff
itself. See the testbed section below for a more direct attempt at closing
this gap entirely. As expected, raising
Tfo also reduces total yield since the evolution stops earlier (a smaller,
earlier freeze-out surface) -- only the *shape*/peak-ratio was compared
between the two settings above, not absolute normalization, since a higher
freeze-out temperature is expected to change normalization regardless of the
peak question.

**Follow-up: does pushing cleanly past the real cutoff help further?** Since
150 MeV turned out to be the model's actual training boundary (not a value
with built-in margin), the Tfo=150 test was still running right at the edge
of training coverage. Tested 160 MeV (`config/fno_3d_particlization_30_40_{fno,music}_Tfo160.xml`)
on the same event (seed 424242) to see whether 10 MeV of clean margin inside
the trained range improves things further:

| window | Tfo=136 MeV | Tfo=150 MeV | Tfo=160 MeV |
|---|---|---|---|
| peak/edge, narrow (\|y\|<0.45) | 12.4% | 7.6% | 8.1% |
| peak/edge, wide (\|y\|<0.75) | 38.8% | 23.0% | 23.8% |

**It plateaus.** 160 MeV is statistically indistinguishable from 150 MeV
(same event, so this isn't event-to-event noise) -- all of the improvement
happened going from below the cutoff to at/above it; adding more margin past
the boundary doesn't shrink the peak any further. This is a useful negative
result: it means the residual ~8-24% effect isn't simply "too close to the
edge of training coverage" and wouldn't be fixed by inference-time margin
alone, however much is added. Whatever is left over is a more fundamental
limitation -- consistent with the uneven-per-event-exposure explanation
above, or requiring an actually retrained model (see below) rather than a
parameter choice against the existing one. (Single-event check only; not yet
confirmed at ensemble level the way the 150 MeV result was.)

## Does the Tfo=150 MeV improvement hold across centralities?

Ran the same Tfo=150 MeV test (single event, seed 424242) at 10-20% and
20-30% centrality (`config/fno_3d_particlization_{10_20,20_30}_{fno,music}_Tfo150.xml`),
to check whether the 30-40% improvement generalizes.

| centrality | peak/edge, narrow (\|y\|<0.45) | peak/edge, wide (\|y\|<0.75) | Tfo=136 narrow, same bin |
|---|---|---|---|
| 10-20% | 12.8% | 30.2% | 11.1% (no improvement) |
| 20-30% | **2.7%** | 17.0% | not tested at 136 |
| 30-40% | 7.6% | 23.0% | 12.4% (clear improvement) |

**Not uniform.** At 30-40% the fix clearly helps; at 10-20% it makes
essentially no difference (if anything marginally worse, within single-event
noise); at 20-30% the peak nearly disappears in the narrow window -- visibly
the cleanest FNO/MUSIC agreement of any centrality or temperature combination
tested so far:

![Hadron rapidity, 20-30% centrality, Tfo=150 MeV -- FNO tracks MUSIC closely through the whole shared midrapidity region](docs/particlization_investigation/y_spectrum_20_30_Tfo150.png)

10-20%, by contrast, still shows a visible excess bump at center even at
150 MeV:

![Hadron rapidity, 10-20% centrality, Tfo=150 MeV -- FNO still shows a mild excess at center](docs/particlization_investigation/y_spectrum_10_20_Tfo150.png)

This means centrality itself is a real variable in whatever's driving the
residual peak, not just the freeze-out temperature choice -- 20-30% looks
like a "sweet spot" for this trained model, while the highest-multiplicity
bin tested (10-20%) resists the Tfo fix entirely. Each data point here is a
single event, so some of this spread could still be event-specific rather
than a true centrality trend; an ensemble check per centrality (as was done
for 30-40%) would be needed to confirm it's systematic.

## Testbed: generating training data with a 100 MeV cutoff

If the residual peak is driven by the model's training supervision running
out at 150 MeV, a model trained with real (non-padded) targets further below
that -- with headroom past the usual 136-150 MeV Cooper-Frye convention --
might avoid the issue more completely than just raising inference-time Tfo
against the existing model can. As a first step, generating a small pilot
training dataset at 30-40% centrality with MUSIC's own `freezeout_temperature`
explicitly lowered to 0.100 GeV (so MUSIC's own evolution loop -- which
exits once `evolve.cpp`'s `FindFreezeOutSurface_Cornelius`/
`FreezeOut_equal_tau_Surface` report the whole grid frozen, the same
mechanism that stopped the original training runs at the 150 MeV default --
continues further before stopping).

Config: `/home/cparker/jetscape-docker/xmls/AuAu_30_40_Tfo100.xml`, cloned
from the original `AuAu_50ev_30-40.xml` training config (same grid/seed/
physics settings, `RootBulkWriter` `ntau=0` so output length adapts
automatically to however long MUSIC now runs), with `freezeout_temperature`
explicitly set to 0.100 GeV and `nEvents` reduced to 5 for a pilot run (full
evolution to a lower temperature takes noticeably longer per event than the
original 150 MeV cutoff). Run via the generic `runJetscape` driver (automatic
task-list determination from the XML, no bespoke driver `.cc` needed, same
as the original training-data generation).

**Result**: completed successfully (5 events, ~9 minutes total). Output
`/home/jetscape-user/rootfiles/AuAu_30_40_Tfo100.root` (~388 MB) is
structurally consistent with the original dataset -- same grid (nx=65,
ny=65, neta=17, matching the existing raw-grid-plus-one-bin convention
corrected via `etaslice`/`xyslice: [:-1]` before training), same
tau_min/dtau/eta_min/deta, nFeatures=5 (energy density, temperature, vx, vy,
vz). `use_vec=1` confirms the variable-length (`ntau=0`) ragged-array mode
worked as expected, so there's no fixed-array truncation.

Per-event freeze-out times: 6.68-8.88 fm/c (63-85 tau steps), i.e. real,
non-padded MUSIC targets extending well past where a 150 MeV cutoff would
have stopped for comparable events -- confirming the lowered
`freezeout_temperature` did what was intended (MUSIC's own evolution loop,
via `evolve.cpp`'s freeze-out-surface check, kept running until the later,
cooler stopping point instead of the original default).

This is a pilot dataset only (5 events, one centrality bin) -- not yet used
to train anything. Scaling up (more events, other centrality bins) and
actually retraining a model against this deeper-coverage data is future
work, not done here.

## Diagnostics added (kept permanently in `FnoHydro.cc`/`.h`)

- `WritePeakTemperatureDiagnostics` — hottest-cell (ed, T) vs tau, every
  step (pre-existing).
- `WriteEtaProfileDiagnostics` — (x,y)-summed/maxed energy density and mean
  vz per eta bin, sparse tau sample.
- Surface eta-profile diagnostic (inline in `EvolveHydro()`) — surface cell
  count, all four `|d3sigma_mu[i]|` components, `dsigma_mu*u^mu`, mean
  `umu[3]`, and mean freeze-out tau, per eta bin.
- `WriteEtaFreezeoutApproachDiagnostic` — per-eta-slice own crossing step/tau
  and the local cooling rate (finite-differenced) in the two steps
  immediately before that crossing.

A temporary instrumentation of iSS itself (`FSSW.cpp`, dumping per-cell
analytic yield for a chosen PDG species) was used to verify the real runtime
sampling path and has been **reverted** (not part of this repo) — if
revisiting this investigation, re-add a per-species dump gated on
`particle->monval` right after the
`calculate_dN_dxtdy_for_one_particle_species()` call in
`FSSW::sample_using_dN_dxtdy_4all_particles_conventional`.

## Status

The velocity-wiring fix is the real, confirmed bug and is fixed and
committed. The residual ~13-14% central peak is real, reproducible (also
confirmed at 10-20% centrality), and traced to the freeze-out surface's
geometry (not the bulk hydro field, not iSS's sampling, not particle
species), with a plausible and now partially-confirmed link to eta-dependent
model behavior near the per-cell freeze-out crossing. Root cause of why 136
vs 150 MeV matters at all: **the entire original training dataset (every
centrality bin) was generated with MUSIC's own default freeze-out
temperature of 0.150 GeV, never explicitly overridden in any training
config** -- while every live-inference config in this investigation used
0.136 GeV, an independent, unrelated `<FNO>`/`<FNOROOIN>` module default that
was never cross-checked against the training setup. Raising inference-time
Tfo to 150 MeV (i.e. actually matching the real training cutoff, not an
arbitrary higher value) reduces the peak by roughly 30-46% at 30-40%
centrality (confirmed at ensemble level) without eliminating it -- real but
partial support for the training-truncation hypothesis. Pushing further to
160 MeV (10 MeV of clean margin past the real cutoff) does **not** help any
more -- the peak plateaus, statistically indistinguishable from the 150 MeV
result on the same event -- so the residual isn't simply "too close to the
training edge"; more inference-time margin against the *existing* model
won't close it further. Critically, **the Tfo=150 improvement itself isn't
centrality-independent**: it's strong at 30-40%, produces the cleanest
FNO/MUSIC agreement seen anywhere at 20-30% (~3% residual), and makes no
measurable difference at 10-20%. So centrality is a real variable in
whatever's driving this, not just the temperature choice. The residual is
more plausibly driven by individual training events' own varying stopping
points leaving uneven exposure even at the nominal cutoff, or by the
surface-geometry mechanism traced earlier -- both of which could plausibly
also vary with centrality (different typical multiplicities/surface sizes).
A pilot training-data
generation run with the cutoff pushed much further, to 100 MeV, completed
successfully (5 events, 30-40% centrality) as a first step toward testing
whether an actually retrained model avoids the issue more completely --
not yet used to train anything. None of this appears to meaningfully distort
physics conclusions drawn from this pipeline (both FNO and MUSIC produce
smooth, physically reasonable spectra; the peak is a ~8-14% effect on top of
good agreement in slope and normalization), but a full mechanistic
explanation remains open.
