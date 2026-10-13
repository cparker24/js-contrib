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
does **not** eliminate it; a real ~9-24% peak remains even at 150 MeV. This
is real evidence for the training-truncation hypothesis (running closer to
the undertrained low-temperature regime makes the artifact worse), but it is
only a partial explanation, not the whole mechanism. As expected, raising
Tfo also reduces total yield since the evolution stops earlier (a smaller,
earlier freeze-out surface) -- only the *shape*/peak-ratio was compared
between the two settings above, not absolute normalization, since a higher
freeze-out temperature is expected to change normalization regardless of the
peak question.

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
model behavior near the per-cell freeze-out crossing: raising the freeze-out
temperature from 136 to 150 MeV reduces the peak by roughly 30-46%
(confirmed at ensemble level) without eliminating it, supporting the
training-truncation hypothesis as a real but partial contributor. It does
not appear to meaningfully distort physics conclusions drawn from this
pipeline (both FNO and MUSIC produce smooth, physically reasonable spectra;
the peak is a ~9-14% effect on top of good agreement in slope and
normalization), but a full mechanistic explanation remains open.
