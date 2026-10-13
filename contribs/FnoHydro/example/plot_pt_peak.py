#!/usr/bin/env python3
"""Plot the final FNO-vs-MUSIC pT spectra from a compare_particle_spectra.py
output .root file -- companion to plot_final_peak.py (rapidity). See
PARTICLIZATION_INVESTIGATION.md.

Usage:
  python3 plot_pt_peak.py <spectra_comparison.root> [out.png] [title]
"""
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def main():
    root_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else "pt_spectrum_peak.png"
    title = sys.argv[3] if len(sys.argv) > 3 else "Hadron p_{T} spectrum"

    f = ROOT.TFile(root_path)
    h_fno = f.Get("hPt_FNO")
    h_music = f.Get("hPt_MUSIC")
    h_fno.SetStats(0)
    h_music.SetStats(0)

    c = ROOT.TCanvas("c_pt_peak", "final pT spectrum", 900, 650)
    c.SetLogy()
    h_music.SetTitle(f"{title};p_{{T}} [GeV];dN/dp_{{T}}")
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
