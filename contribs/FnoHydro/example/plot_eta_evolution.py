#!/usr/bin/env python3
"""Plot energy-density-vs-eta at several tau steps from an
WriteEtaProfileDiagnostics-format CSV (FnoHydro's own, or MusicWrapper's
WriteMusicEtaProfileDiagnostic -- same columns, so this script works
unchanged against either), to show the raw hydro field stays smooth in eta
throughout the evolution (no central peak at any stage) -- contrasted
against the final sampled-hadron rapidity spectrum, which does show a
central peak. Used for PARTICLIZATION_INVESTIGATION.md.

Usage:
  python3 plot_eta_evolution.py <eta_profile.csv> [out.png] [label]
"""
import csv
import sys
from collections import defaultdict

import ROOT

ROOT.gROOT.SetBatch(True)


def load_eta_profile(path):
    by_k = defaultdict(dict)
    taus = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row["k"])
            eta = float(row["eta_value"])
            by_k[k][eta] = float(row["sum_energy_density_GeV_fm3"])
            taus[k] = float(row["tau_fm"])
    return by_k, taus


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "eta_profile_event0.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "bulk_eta_evolution.png"
    label = sys.argv[3] if len(sys.argv) > 3 else "FNO"

    by_k, taus = load_eta_profile(csv_path)
    ks = sorted(by_k.keys())
    totals = {k: sum(by_k[k].values()) for k in ks}
    # Drop any leading steps that are still essentially empty (e.g. MUSIC's
    # string-formation-time ramp-up, near-zero before tau_form_mean) -- a
    # step whose total is <5% of the NEXT dumped step's isn't representative
    # of the field yet.
    start = 0
    while (start + 1 < len(ks)
           and totals[ks[start]] < 0.05 * totals[ks[start + 1]]):
        start += 1
    ks = ks[start:]
    # Pick a handful of steps spanning the evolution, skipping any
    # post-freeze-out-cutoff steps (where sum_ed collapses by >90% between
    # consecutive sparse samples -- those are extrapolation noise, not
    # physical).
    chosen_ks = []
    prev_total = None
    for k in ks:
        total = totals[k]
        if prev_total is not None and total < 0.05 * prev_total:
            break
        chosen_ks.append(k)
        prev_total = total
    # Thin down to ~5 representative steps evenly spaced across the valid
    # range (always keeps the first and last).
    max_steps = 5
    if len(chosen_ks) > max_steps:
        idx = [round(i * (len(chosen_ks) - 1) / (max_steps - 1)) for i in range(max_steps)]
        chosen_ks = [chosen_ks[i] for i in sorted(set(idx))]

    colors = [ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kOrange + 1,
              ROOT.kRed + 1, ROOT.kMagenta + 1, ROOT.kBlack]

    c = ROOT.TCanvas("c_eta_evol", "bulk eta evolution", 900, 650)
    c.SetLogy()
    leg = ROOT.TLegend(0.68, 0.55, 0.88, 0.88)
    graphs = []
    frame_max = 0.0
    frame_min = 1e18
    for i, k in enumerate(chosen_ks):
        etas = sorted(by_k[k].keys())
        vals = [by_k[k][e] for e in etas]
        frame_max = max(frame_max, max(vals))
        frame_min = min(frame_min, min(v for v in vals if v > 0))
        g = ROOT.TGraph(len(etas))
        for j, (e, v) in enumerate(zip(etas, vals)):
            g.SetPoint(j, e, v)
        g.SetLineColor(colors[i % len(colors)])
        g.SetLineWidth(2)
        g.SetMarkerColor(colors[i % len(colors)])
        g.SetMarkerStyle(20)
        g.SetMarkerSize(0.8)
        graphs.append(g)
        leg.AddEntry(g, f"#tau = {taus[k]:.2f} fm/c", "lp")

    mg = ROOT.TMultiGraph()
    for g in graphs:
        mg.Add(g, "LP")
    mg.SetTitle(f"{label} bulk energy density vs. eta, several #tau "
                "(smooth throughout -- no central peak);"
                "eta;(x,y)-summed energy density [GeV/fm^{3}]")
    mg.Draw("A")
    mg.GetYaxis().SetRangeUser(frame_min * 0.5, frame_max * 2.0)
    leg.Draw()
    c.SaveAs(out_png)
    print(f"Wrote {out_png} using steps {chosen_ks} (tau = "
          f"{[round(taus[k],2) for k in chosen_ks]} fm/c)")


if __name__ == "__main__":
    main()
