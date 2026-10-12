#!/usr/bin/env python3
"""Compare FNO-hydro-vs-MUSIC Cooper-Frye (iSS) output: pT and rapidity
spectra, built with PyROOT so the histograms/canvas stay easily re-stylable
(reopen the .root file, no need to re-run the parser) rather than baked into
a one-shot matplotlib image.

Input: two JetScapeWriterFinalStateHadronsAscii files (flat, one header line
"# Event <n> weight <w> EPangle <a> N_hadrons <ipart>" per event followed by
"<index> <pid> <pstat> <E> <px> <py> <pz>" particle lines -- see
JetScapeWriterFinalStateStream.cc / JetScapeReaderFinalStateHadrons.cc).

Usage:
  python3 compare_particle_spectra.py <fno.dat> <music.dat> [n_oversamples] [out.root]
"""
import sys
import math

import ROOT

ROOT.gROOT.SetBatch(True)


def parse_final_state_hadrons(path):
    """Yields (pid, pstat, e, px, py, pz) for every particle in every event."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid, pstat = int(parts[1]), int(parts[2])
            e, px, py, pz = map(float, parts[3:7])
            yield pid, pstat, e, px, py, pz


def rapidity(e, pz):
    if e <= abs(pz):
        return 0.0
    return 0.5 * math.log((e + pz) / (e - pz))


def fill_spectra(path, n_oversamples, label, color, y_cut=(-1.0, 1.0)):
    y_min, y_max = y_cut
    h_pt = ROOT.TH1D(f"hPt_{label}",
                      "dN/dp_{T} per sampled event, |y|<%.1f;p_{T} [GeV];dN/dp_{T}" % y_max,
                      20, 0.0, 3.0)
    h_y = ROOT.TH1D(f"hY_{label}", "dN/dy per sampled event;rapidity y;dN/dy",
                     20, -3.0, 3.0)
    h_pt.Sumw2()
    h_y.Sumw2()

    n_particles = 0
    n_midrapidity = 0
    species = {}
    species_mid = {}
    for pid, pstat, e, px, py, pz in parse_final_state_hadrons(path):
        pt = math.sqrt(px * px + py * py)
        y = rapidity(e, pz)
        w = 1.0 / n_oversamples
        h_y.Fill(y, w)
        n_particles += 1
        species[pid] = species.get(pid, 0) + 1
        if y_min <= y < y_max:
            h_pt.Fill(pt, w)
            n_midrapidity += 1
            species_mid[pid] = species_mid.get(pid, 0) + 1

    # Normalize bin content to a density (divide by bin width) so differently
    # binned histograms stay comparable.
    h_pt.Scale(1.0 / h_pt.GetBinWidth(1))
    h_y.Scale(1.0 / h_y.GetBinWidth(1))

    h_pt.SetLineColor(color)
    h_pt.SetMarkerColor(color)
    h_pt.SetMarkerStyle(20)
    h_y.SetLineColor(color)
    h_y.SetMarkerColor(color)
    h_y.SetMarkerStyle(20)

    dndy_mid = n_midrapidity / (n_oversamples * (y_max - y_min))
    print(f"{label}: {n_particles} total sampled hadrons "
          f"({n_particles / n_oversamples:.1f} per oversample, n_oversamples={n_oversamples})")
    print(f"{label} mid-rapidity (|y|<{y_max:.1f}): {n_midrapidity} hadrons "
          f"({n_midrapidity / n_oversamples:.1f} per oversample, dN/dy = {dndy_mid:.1f})")
    top_species = sorted(species_mid.items(), key=lambda kv: -kv[1])[:10]
    print(f"{label} top mid-rapidity PIDs: {top_species}")

    return h_pt, h_y


def main():
    fno_path = sys.argv[1]
    music_path = sys.argv[2]
    n_oversamples = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    out_root = sys.argv[4] if len(sys.argv) > 4 else "spectra_comparison.root"
    y_min = float(sys.argv[5]) if len(sys.argv) > 5 else -1.0
    y_max = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

    h_pt_fno, h_y_fno = fill_spectra(fno_path, n_oversamples, "FNO", ROOT.kAzure + 1,
                                      y_cut=(y_min, y_max))
    h_pt_music, h_y_music = fill_spectra(music_path, n_oversamples, "MUSIC", ROOT.kOrange + 7,
                                          y_cut=(y_min, y_max))

    c_pt = ROOT.TCanvas("c_pt", "pT spectrum", 800, 600)
    c_pt.SetLogy()
    h_pt_music.SetTitle("Mid-rapidity (%.1f<y<%.1f) hadron p_{T} spectrum, 30-40%% Au+Au;"
                         "p_{T} [GeV];dN/dp_{T}" % (y_min, y_max))
    h_pt_music.Draw("E1")
    h_pt_fno.Draw("E1 SAME")
    leg_pt = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    leg_pt.AddEntry(h_pt_music, "MUSIC + iSS", "lep")
    leg_pt.AddEntry(h_pt_fno, "FNO + iSS", "lep")
    leg_pt.Draw()
    c_pt.SaveAs("pt_spectrum_comparison.png")

    c_y = ROOT.TCanvas("c_y", "rapidity spectrum", 800, 600)
    h_y_music.SetTitle("Hadron rapidity spectrum, 30-40% Au+Au;rapidity y;dN/dy")
    h_y_music.SetMinimum(0)
    h_y_music.SetMaximum(1.1 * max(h_y_fno.GetMaximum(), h_y_music.GetMaximum()))
    h_y_music.Draw("E1")
    h_y_fno.Draw("E1 SAME")
    leg_y = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    leg_y.AddEntry(h_y_music, "MUSIC + iSS", "lep")
    leg_y.AddEntry(h_y_fno, "FNO + iSS", "lep")
    leg_y.Draw()
    line_lo = ROOT.TLine(y_min, 0, y_min, h_y_music.GetMaximum())
    line_hi = ROOT.TLine(y_max, 0, y_max, h_y_music.GetMaximum())
    for line in (line_lo, line_hi):
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw()
    c_y.SaveAs("y_spectrum_comparison.png")

    out = ROOT.TFile(out_root, "RECREATE")
    h_pt_fno.Write()
    h_pt_music.Write()
    h_y_fno.Write()
    h_y_music.Write()
    c_pt.Write()
    c_y.Write()
    out.Close()
    print(f"\nWrote {out_root}, pt_spectrum_comparison.png, y_spectrum_comparison.png")


if __name__ == "__main__":
    main()
