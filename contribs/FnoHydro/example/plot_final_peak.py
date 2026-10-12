#!/usr/bin/env python3
"""Plot the final FNO-vs-MUSIC rapidity spectra from a compare_particle_spectra.py
output .root file, for side-by-side comparison against
plot_eta_evolution.py's bulk-field plot -- see PARTICLIZATION_INVESTIGATION.md.

Usage:
  python3 plot_final_peak.py <spectra_comparison.root> [out.png] [title]
"""
import sys

import ROOT

ROOT.gROOT.SetBatch(True)


def main():
    root_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else "final_sampled_rapidity_peak.png"
    title = sys.argv[3] if len(sys.argv) > 3 else "Final Cooper-Frye sampled hadron rapidity"

    f = ROOT.TFile(root_path)
    h_fno = f.Get("hY_FNO")
    h_music = f.Get("hY_MUSIC")

    c = ROOT.TCanvas("c_final_peak", "final sampled rapidity", 900, 650)
    h_music.SetTitle(f"{title};rapidity y;dN/dy")
    h_music.SetMinimum(0)
    h_music.SetMaximum(1.15 * max(h_fno.GetMaximum(), h_music.GetMaximum()))
    h_music.SetLineColor(ROOT.kOrange + 7)
    h_music.SetMarkerColor(ROOT.kOrange + 7)
    h_music.SetMarkerStyle(20)
    h_fno.SetLineColor(ROOT.kAzure + 1)
    h_fno.SetMarkerColor(ROOT.kAzure + 1)
    h_fno.SetMarkerStyle(20)
    h_music.Draw("E1")
    h_fno.Draw("E1 SAME")
    leg = ROOT.TLegend(0.6, 0.72, 0.88, 0.88)
    leg.AddEntry(h_music, "MUSIC + iSS", "lep")
    leg.AddEntry(h_fno, "FNO + iSS", "lep")
    leg.Draw()
    c.SaveAs(out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
