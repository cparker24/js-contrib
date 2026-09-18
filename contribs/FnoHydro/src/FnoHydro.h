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

#ifndef FNOHYDRO_H
#define FNOHYDRO_H

#include <memory>

#include "FluidDynamics.h"
//#include "hydro_source_base.h"
#include "LiquefierBase.h"
//#include "data_struct.h" /in iSS ..
#include "JetScapeConstants.h"
#include "MakeUniqueHelper.h"
#include "eos.h"

#include <torch/script.h>
#include <torch/torch.h>

using namespace Jetscape;

class FnoHydro : public FluidDynamics {
private:

  Jetscape::real freezeout_temperature; //!< [GeV]
  //int doCooperFrye;                     //!< flag to run Cooper-Frye freeze-out
                                        //!< for soft particles
  //bool has_source_terms;
  //std::shared_ptr<HydroSourceJETSCAPE> hydro_source_terms_ptr;

  Jetscape::real x_min_preq, dx_preq;
  Jetscape::real y_min_preq, dy_preq;
  Jetscape::real eta_min_preq, deta_preq;
  int nx_preq, ny_preq, neta_preq;

  Jetscape::real x_min_fno, dx_fno;
  Jetscape::real y_min_fno, dy_fno;
  Jetscape::real eta_min_fno, deta_fno;
  Jetscape::real dtau_fno;
  int nx_fno, ny_fno;
  int ntau_fno;
  int neta_fno;

  int n_features;

  // Whether the model's energy-density channel is epsilon*tau (needs dividing
  // by tau on read-back) or raw epsilon (no division). The n_features==3 (2D+1)
  // / n_features==4 (3D+1) branches historically assumed epsilon*tau always;
  // that only holds for models trained with an etau75-style normalization.
  // Defaults to true to preserve behavior for those existing models; set
  // <Hydro><FNO><edensity_times_tau>0</edensity_times_tau></FNO></Hydro> for
  // a model trained on raw (un-normalized) epsilon.
  bool edensity_times_tau_fno;

  // neta_fno > 1 selects the full 3D+1 (x,y,eta,tau) pipeline; neta_fno == 1
  // keeps the original boost-invariant 2D+1 (x,y,tau) pipeline byte-for-byte,
  // so existing 2D+1-trained traced models keep working unchanged.
  inline bool Is3DPlus1() const { return neta_fno > 1; }

  torch::jit::script::Module module;
  torch::Tensor output;
  torch::Device device;

  std::unique_ptr<EOS> fnoEOS;
  float GetTemperatureFromEos(float ed);

  // Allows the registration of the module so that it is available to be
  // used by the Jetscape framework.
  static RegisterJetScapeModule<FnoHydro> reg;

  inline int GetPreqIdX(Jetscape::real x) const {
    return (static_cast<int>((x - x_min_preq) / dx_preq));
  }

  inline int GetPreqIdY(Jetscape::real y) const {
    return (static_cast<int>((y - y_min_preq) / dy_preq));
  }

  // Collapses to 0 for a 2D+1 (boost-invariant) pre-equilibrium grid.
  inline int GetPreqIdEta(Jetscape::real eta) const {
    if (neta_preq <= 1 || deta_preq == 0.0) return 0;
    return (static_cast<int>((eta - eta_min_preq) / deta_preq));
  }

  // Flattening matches the (x,y,eta) layout used by FreestreamMilne /
  // JETSCAPE's EvolutionHistory::CellIndex (eta fastest-varying); reduces to
  // the original id_x*ny_preq + id_y scheme when neta_preq == 1.
  int GetPreqCellIndex(int id_x, int id_y, int id_eta) const;
  void GetCellIndicesFromGlobalPreqIndex(int global_index, int& id_x, int& id_y, int& id_eta) const;

public:

  FnoHydro();
  ~FnoHydro();

  void InitializeHydro(Parameter parameter_list);

  void EvolveHydro();

  // Override Clear() so that bulk_info.data survives JetScapeTask::ClearTasks()
  // at the end of each event.  EvolveHydro() already calls
  // clear_up_evolution_data() at its own start, so multi-event runs are safe.
  // Only the freeze-out surface (re-computed each event) is cleared here.
  void Clear() { clearSurfaceCellVector(); }
  void GetHydroInfo(Jetscape::real t, Jetscape::real x, Jetscape::real y,
                    Jetscape::real z, std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr);

  void GetHydroInfo_JETSCAPE(Jetscape::real t, Jetscape::real x, Jetscape::real y,
                    Jetscape::real z, std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr);

  void SetPreEqGridInfo();
  void SetHydroGridInfo();
  void PassPreEqEvolutionHistoryToFramework();
  void PassHydroEvolutionHistoryToFramework();
  //void PassHydroSurfaceToFramework();

  //void add_a_liquefier(std::shared_ptr<LiquefierBase> new_liqueifier) {
    //liquefier_ptr = new_liqueifier;
    //hydro_source_terms_ptr->add_a_liquefier(liquefier_ptr.lock());
    //}

  //void GetHyperSurface(Jetscape::real T_cut,
  //                    SurfaceCellInfo *surface_list_ptr){};
  //void collect_freeze_out_surface();
};

#endif // FNOHYDRO_H
