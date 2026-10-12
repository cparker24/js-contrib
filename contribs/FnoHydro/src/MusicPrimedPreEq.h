/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Modular, task-based framework for simulating all aspects of heavy-ion collisions
 *
 * For the list of contributors see AUTHORS.
 *
 * Report issues at https://github.com/JETSCAPE/JETSCAPE/issues
 *
 * or via email to bugs.jetscape@gmail.com
 *
 * Distributed under the GNU General Public License 3.0 (GPLv3 or later).
 * See COPYING for details.
 ******************************************************************************/

#ifndef MUSICPRIMEDPREEQ_H
#define MUSICPRIMEDPREEQ_H

#include <memory>
#include <cmath>

#include "PreequilibriumDynamics.h"
#include "NullPreDynamics.h"
#include "MusicWrapper.h"
#include "JetScapeLogger.h"

using namespace Jetscape;

// FluidDynamics::ini and ::pre_eq_ptr are protected (normally wired by
// JetScapeSignalManager for a jetscape->Add()'d module); this re-exposes
// them so MusicPrimedPreEq can wire an embedded, not-Add()'d MpiMusic
// instance by hand.
class PrimerMpiMusic : public MpiMusic {
public:
  using MpiMusic::ini;
  using MpiMusic::pre_eq_ptr;
};

// Bridges MCGlauber's real, validated MUSIC string-source physics
// (InitialProfile=131) into the single-instant energy-density snapshot
// interface FnoHydro (and any other PreequilibriumDynamics consumer)
// expects.
//
// Internally drives its own embedded MpiMusic instance through the SAME
// <Hydro><MUSIC> XML settings the standard-hydro comparison arm uses,
// evolves it only up to <Preequilibrium><taus> (the FNO's trained grid
// tau0), then samples that primed medium pointwise via MpiMusic's own
// GetHydroInfo() onto ini's grid -- the same query interface any hydro
// consumer (MATTER, a second hydro stage, ...) would use.
//
// This deliberately does NOT reimplement MUSIC's string-source physics
// (hydro_source_strings.cpp's hyperbolic-trajectory Gaussian-smeared
// deposition, ~13+ fields per string): that stays inside MUSIC's own
// tested code. This module only decides *when* to stop and hand off, not
// *how* strings become a field.
//
// Not registered via RegisterJetScapeModule -- like NullPreDynamics is
// used directly in some drivers, this is meant to be constructed explicitly
// (make_shared<MusicPrimedPreEq>()) by the executable, not looked up by
// name from XML.
class MusicPrimedPreEq : public PreequilibriumDynamics {
public:
  MusicPrimedPreEq() {
    preequilibrium_status_ = NOT_STARTED;
    SetId("MusicPrimedPreEq");
  }

  void EvolvePreequilibrium() {
    // preequilibrium_tau_max_ comes from <Preequilibrium><taus>, read by
    // the PreequilibriumDynamics base class's Init() before this runs.
    double tau_prime_end = preequilibrium_tau_max_;
    JSINFO << "MusicPrimedPreEq: priming a real MCGlauber+MUSIC medium up to "
           << "tau = " << tau_prime_end << " fm/c ...";

    // Embedded MpiMusic needs a non-null pre_eq_ptr (dereferenced
    // unconditionally in a few places) even though InitialProfile 13/131
    // ignore its contents and read strings from `ini` directly instead.
    auto primer_stub = std::make_shared<NullPreDynamics>();
    primer_stub->ini = ini;
    primer_stub->preequilibrium_tau_0_ = 0.0;
    primer_stub->preequilibrium_tau_max_ = tau_prime_end;

    auto primer = std::make_shared<PrimerMpiMusic>();
    primer->ini = ini;
    primer->pre_eq_ptr = primer_stub;
    primer->InitializeHydro(Parameter());
    // MUSIC::run_hydro_upto() rounds its target down to a whole number of
    // internal sub-steps, so priming to exactly tau_prime_end can leave the
    // stored history's TauMax() one step short of tau_prime_end itself --
    // CheckInRange() then rejects the query we're about to make below and
    // silently hands back a zero-valued cell. Pad the priming target so the
    // query point sits safely inside (not right on the edge of) the stored
    // range.
    const double priming_margin = 0.1;
    primer->EvolveHydroUpto(tau_prime_end + priming_margin);
    // EvolveHydroUpto() alone leaves bulk_info (the framework-side cache
    // GetHydroInfo() reads from) empty -- that only gets populated at the
    // tail of the full EvolveHydro() pipeline. Populate it explicitly from
    // MUSIC's current (partially-evolved) internal state, matching what
    // EvolveHydro() itself would do.
    // (GetHydroInfo_MUSIC() was tried first as a shortcut -- it crashes:
    // HydroinfoMUSIC::getHydroValues expects Minkowski (t,x,y,z) too, but
    // via a different, incompatible internal grid.)
    primer->PassHydroEvolutionHistoryToFramework();

    double dx = ini->GetXStep();
    double dy = ini->GetYStep();
    double deta = ini->GetZStep();
    double x_min = -ini->GetXMax();
    double y_min = -ini->GetYMax();
    double eta_min = -ini->GetZMax();
    int nx = ini->GetXSize();
    int ny = ini->GetYSize();
    int neta = ini->GetZSize();

    preequilibrium_status_ = INIT;
    // Fill order must be (x, y, eta) with eta fastest-varying, matching
    // GetPreqCellIndex's indexing formula -- which in turn matches the
    // framework-wide convention (EvolutionHistory::CellIndex: id_x*ny*neta +
    // id_y*neta + id_eta). This loop previously nested eta outermost and y
    // innermost, filling e_ in a completely different order than
    // FnoHydro::GetPreqCellIndex expects to read it back in: every lookup
    // landed on the wrong (x,y,eta) cell. Same pool of values (so aggregate
    // stats like the peak magnitude looked reasonable), scrambled locations
    // (so the spatial pattern the FNO model actually saw bore no resemblance
    // to the real initial condition) -- this is almost certainly why live
    // FNO inference diverges from real MUSIC while offline evaluation on
    // correctly-ordered training data does not.
    for (int i = 0; i < nx; ++i) {
      double x = x_min + i * dx;
      for (int j = 0; j < ny; ++j) {
        double y = y_min + j * dy;
        for (int k = 0; k < neta; ++k) {
          double eta = eta_min + k * deta;
          // GetHydroInfo (-> EvolutionHistory::get_tz) takes genuine
          // Minkowski (t,z), converting internally via
          // tau=sqrt(t^2-z^2), eta=atanh(z/t) -- NOT (tau,eta) directly.
          // Convert our Milne query point the other way first.
          double t_lab = tau_prime_end * std::cosh(eta);
          double z_lab = tau_prime_end * std::sinh(eta);
          std::unique_ptr<FluidCellInfo> cell;
          primer->GetHydroInfo(t_lab, x, y, z_lab, cell);

          e_.push_back(cell->energy_density);
          P_.push_back(cell->pressure);
          // ux_/uy_/ueta_ store raw 3-velocity components (vx, vy, vz),
          // matching FluidCellInfo's own convention -- FnoHydro::EvolveHydro()
          // reads these directly into the FNO model's vx/vy/vz input
          // channels with no further transform, so this must stay consistent
          // with that convention, not a genuine 4-velocity. utau_ is left as
          // an unused approximation (true gamma not computed) since nothing
          // downstream reads it.
          utau_.push_back(1.0);
          ux_.push_back(cell->vx);
          uy_.push_back(cell->vy);
          ueta_.push_back(cell->vz);
          pi00_.push_back(0.); pi01_.push_back(0.); pi02_.push_back(0.); pi03_.push_back(0.);
          pi11_.push_back(0.); pi12_.push_back(0.); pi13_.push_back(0.);
          pi22_.push_back(0.); pi23_.push_back(0.);
          pi33_.push_back(0.);
          bulk_Pi_.push_back(0.);
        }
      }
    }
    preequilibrium_status_ = DONE;
    JSINFO << "MusicPrimedPreEq: primed " << e_.size() << " fluid cells "
           << "(nx=" << nx << " ny=" << ny << " neta=" << neta << ").";
  }
};

#endif // MUSICPRIMEDPREEQ_H
