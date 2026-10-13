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
  python3 compare_particle_spectra.py <fno.dat> <music.dat> [n_oversamples_fno] [n_oversamples_music] [out.root] [y_min] [y_max] [centrality_label]

n_oversamples_fno/n_oversamples_music are each file's own
<SoftParticlization><iSS><number_of_repeated_sampling> -- pass different
values when comparing runs that used different oversampling (e.g. a
doubled-statistics check on one arm only). centrality_label (e.g. "10-20")
is used only in plot titles, default "30-40".
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


def count_events(path):
    """Number of per-event header lines ('# Event <n> weight ...'), so a
    multi-event file normalizes correctly without the caller tracking it."""
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#") and "Event" in line.split():
                n += 1
    return n


def fill_spectra(path, n_oversamples, label, color, y_cut=(-1.0, 1.0)):
    y_min, y_max = y_cut
    n_events = count_events(path)
    n_draws = n_oversamples * n_events
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
        w = 1.0 / n_draws
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

    dndy_mid = n_midrapidity / (n_draws * (y_max - y_min))
    print(f"{label}: {n_events} event(s) x {n_oversamples}x oversampling = {n_draws} total draws, "
          f"{n_particles} total sampled hadrons ({n_particles / n_draws:.1f} per draw)")
    print(f"{label} mid-rapidity (|y|<{y_max:.1f}): {n_midrapidity} hadrons "
          f"({n_midrapidity / n_draws:.1f} per draw, dN/dy = {dndy_mid:.1f})")
    top_species = sorted(species_mid.items(), key=lambda kv: -kv[1])[:10]
    print(f"{label} top mid-rapidity PIDs: {top_species}")

    return h_pt, h_y


def main():
    fno_path = sys.argv[1]
    music_path = sys.argv[2]
    n_oversamples_fno = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n_oversamples_music = int(sys.argv[4]) if len(sys.argv) > 4 else n_oversamples_fno
    out_root = sys.argv[5] if len(sys.argv) > 5 else "spectra_comparison.root"
    y_min = float(sys.argv[6]) if len(sys.argv) > 6 else -1.0
    y_max = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    centrality_label = sys.argv[8] if len(sys.argv) > 8 else "30-40"

    h_pt_fno, h_y_fno = fill_spectra(fno_path, n_oversamples_fno, "FNO", ROOT.kAzure + 1,
                                      y_cut=(y_min, y_max))
    h_pt_music, h_y_music = fill_spectra(music_path, n_oversamples_music, "MUSIC", ROOT.kOrange + 7,
                                          y_cut=(y_min, y_max))

    c_pt = ROOT.TCanvas("c_pt", "pT spectrum", 800, 600)
    c_pt.SetLogy()
    h_pt_music.SetTitle("Mid-rapidity (%.1f<y<%.1f) hadron p_{T} spectrum, %s%% Au+Au;"
                         "p_{T} [GeV];dN/dp_{T}" % (y_min, y_max, centrality_label))
    h_pt_music.Draw("E1")
    h_pt_fno.Draw("E1 SAME")
    leg_pt = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    leg_pt.AddEntry(h_pt_music, "MUSIC + iSS", "lep")
    leg_pt.AddEntry(h_pt_fno, "FNO + iSS", "lep")
    leg_pt.Draw()
    c_pt.SaveAs("pt_spectrum_comparison.png")

    c_y = ROOT.TCanvas("c_y", "rapidity spectrum", 800, 600)
    h_y_music.SetTitle(f"Hadron rapidity spectrum, {centrality_label}% Au+Au;rapidity y;dN/dy")
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
