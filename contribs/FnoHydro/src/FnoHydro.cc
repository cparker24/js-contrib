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

#include <stdio.h>
#include <sys/stat.h>
#include <MakeUniqueHelper.h>

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>

#include "JetScapeLogger.h"
//#include "surfaceCell.h"
#include "FnoHydro.h"
#include "util.h"

#include <Riostream.h>
#include "TRandom.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TF1.h"
#include "TMath.h"
#include "TFile.h"
#include "TString.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

using namespace Jetscape;

// Register the module with the base class
RegisterJetScapeModule<FnoHydro> FnoHydro::reg("FnoHydro");

//****************************************************************************************

// Get from actual grid ...
// Think about resizing etc maybe use same interpoolation as for fluid cells !???
// put in class and read in from preq module ...
/* *
const int nx =150;
const int ny =150;

// https://github.com/JETSCAPE/JETSCAPE/pull/254/files
// Rotation etc fix ... make sure not to repeat here !!!
// x vs y axis ...
int GetPreqCellIndex(int id_x, int id_y)
{
    id_x = std::min(nx - 1, std::max(0, id_x));
    id_y = std::min(ny - 1, std::max(0, id_y));
    return (id_x * ny + id_y);
}

void GetCellIndicesFromGlobalPreqIndex(int global_index, int& id_x, int& id_y)
{
    id_x = global_index / ny;
    id_y = global_index % ny;
}
*/

//****************************************************************************************

FnoHydro::FnoHydro() : device({}) {
  hydro_status = NOT_START;
  freezeout_temperature = 0.0;
  energy_density_normalization = "none";
  max_rollout_windows = 1;
  //doCooperFrye = 0;

  x_min_preq = dx_preq = y_min_preq = dy_preq = 0.0;
  eta_min_preq = deta_preq = 0.0;
  nx_preq = ny_preq = 0;
  neta_preq = 1;

  x_min_fno = dx_fno = y_min_fno = dy_fno = 0.0;
  nx_fno = ny_fno = 60;
  ntau_fno = 50;
  n_features = 4;

  neta_fno = 1;
  eta_min_fno = deta_fno = 0.0;

  //device = torch::Device({});
  //has_source_terms = false;
  SetId("FnoHydro");
  //hydro_source_terms_ptr =
  //    std::shared_ptr<HydroSourceJETSCAPE>(new HydroSourceJETSCAPE());
}

FnoHydro::~FnoHydro() {}

void FnoHydro::InitializeHydro(Parameter parameter_list) {
  JSINFO << "Initialize FnoHydro ...";
  VERBOSE(8);

  //*************************************************************************************************
  //REMARK: Somehow no real improvment when using more threads for tensor and model operations !????
  //*************************************************************************************************

  // For testing now, overwrites the env settings ...
  torch::set_num_threads(1);
  JSINFO << "Number of threads (libtorch OMP): " << torch::get_num_threads();
  JSINFO << "Default device: " << device; // << std::endl;

  //REMARK: Same issue as with tracing, not everything in FNO model gets moved to the proper other device ...
  /*
    if (torch::mps::is_available()) {
        JSINFO<< "MPS device available :-). Use it ...";// << std::endl;
        device=torch::Device(torch::kMPS);
    }
    else if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        JSINFO << "CUDA device available :-). Use it ...";// << std::endl;
    }
    else {
        JSINFO << "MPS or CUDA device(s) not available. Use CPU ..."; // << std::endl;
        device = torch::Device(torch::kCPU);
    }
  */
  try {

    string input_model_file = GetXMLElementText({"Hydro", "FNO", "model_file"});
    JSINFO<<"Loading the traced Pytorch model : "<<input_model_file.c_str();//<<endl;
    module = torch::jit::load(input_model_file.c_str()); //, device);
    //module.to(device);

    /// ------------------------------------------------------------------
  }
  catch (const c10::Error& e) {
    JSWARN << "error loading the model :-( ...\n";
    exit(-1);
  }

  JSINFO << "--> traced Pytorch model loaded";
  /// ------------------------------------------------------------------

  // ******************************************************************************************************************************************
  //REMARK: Super-resolution does not seem to work, related to similar error in the embedding layer, might be hardcoded when serialzed !!!????
  // ---> try to follow up !!!! Check if it works in Python ....
  // ******************************************************************************************************************************************

  x_min_fno = GetXMLElementDouble({"Hydro", "FNO", "x_min"});
  y_min_fno = GetXMLElementDouble({"Hydro", "FNO", "y_min"});
  //z_min_fno = GetBinMin({"Hydro", "FNO", "z_min"});

  nx_fno = GetXMLElementInt({"Hydro", "FNO", "nx"});
  ny_fno = GetXMLElementInt({"Hydro", "FNO", "ny"});
  //nz_fno = GetXMLElementInt({"Hydro", "FNO", "nz"});

  dx_fno = -2*x_min_fno/(double) nx_fno;
  dy_fno = -2*y_min_fno/(double) ny_fno;
  //dz_fno = -2*z_min_fno/(double) nz_fno;

  ntau_fno = GetXMLElementInt({"Hydro", "FNO", "ntau"});
  neta_fno = GetXMLElementInt({"Hydro", "FNO", "neta"});

  deta_fno = GetXMLElementDouble({"Hydro", "FNO", "deta"});
  // Only meaningful for a 3D+1 (neta > 1) model; for a 2D+1/boost-invariant
  // model bulk_info.eta_min is unused (see SetHydroGridInfo).
  eta_min_fno = GetXMLElementDouble({"Hydro", "FNO", "eta_min"});
  dtau_fno = GetXMLElementDouble({"Hydro", "FNO", "dtau"});

  n_features = GetXMLElementInt({"Hydro", "FNO", "n_features"});

  JSINFO<<"# of FNO training features = "<<n_features;
  if (Is3DPlus1()) {
    JSINFO << "FnoHydro running in full 3D+1 mode: neta = " << neta_fno
           << ", eta_min = " << eta_min_fno << ", deta = " << deta_fno;
  }

  //cout<<x_min_fno<<" "<<y_min_fno<<" "<<nx_fno<<" "<<ny_fno<<" "<<endl;
  //cout<<dx_fno<<" "<<dy_fno<<" "<<endl;
  //cout<<ntau_fno<<" "<<dtau_fno<<" "<<endl;

  freezeout_temperature = GetXMLElementDouble({"Hydro", "FNO", "freezeout_temperature"});

  int EOS_id_MUSIC = GetXMLElementInt({"Hydro", "FNO", "EOS_id_MUSIC"});
  JSINFO<<"Use EOS (Music id) = "<<EOS_id_MUSIC;
  fnoEOS=make_unique<EOS>(EOS_id_MUSIC);

  // Optional override: absent means "keep the true default" (none -- raw
  // epsilon, no transform to invert).
  std::string norm_text = GetXMLElementText(
      {"Hydro", "FNO", "energy_density_normalization"}, false);
  if (!norm_text.empty()) {
    energy_density_normalization = norm_text;
  }
  if (energy_density_normalization != "none" &&
      energy_density_normalization != "etau75_3d") {
    JSWARN << "Unsupported energy_density_normalization: '"
           << energy_density_normalization << "' -- supported: none, etau75_3d "
           << "(loc_libs/norm_utils.py has other schemes, e.g. etau75_eta_3d, "
           << "cosh_etau75_3d, that need trained reference data this module "
           << "doesn't load; not implemented here).";
    exit(-1);
  }
  JSINFO << "FNO energy-density channel normalization: "
         << energy_density_normalization;

  max_rollout_windows = GetXMLElementInt({"Hydro", "FNO", "max_rollout_windows"});
  if (max_rollout_windows < 1) max_rollout_windows = 1;
  JSINFO << "FNO rollout: up to " << max_rollout_windows
         << " window(s) of ntau=" << ntau_fno << " each (stops early once the "
         << "hottest cell drops to/below freezeout_temperature)";

  /*
  if (freezeout_temperature > 0.05) {
    music_hydro_ptr->set_parameter("T_freeze", freezeout_temperature);
  } else {
    JSWARN << "The input freeze-out temperature is too low! T_frez = "
           << freezeout_temperature << " GeV!";
    exit(1);
  }

  music_hydro_ptr->add_hydro_source_terms(hydro_source_terms_ptr);
  */
}

void FnoHydro::EvolveHydro() {
  VERBOSE(8);
  JSINFO << "Initialize density profiles in FnoHydro ...";

  auto start = std::chrono::high_resolution_clock::now();

  if (pre_eq_ptr == nullptr) {
    JSWARN << "Missing the pre-equilibrium module ...";
    exit(1);
  }

  dx_preq = ini->GetXStep();
  dy_preq = ini->GetYStep();
  x_min_preq = - ini->GetXMax();
  y_min_preq = - ini->GetYMax();
  double tau0 = pre_eq_ptr->GetPreequilibriumEndTime();

  JSINFO << "hydro initial time set by PreEq module tau0 = " << tau0 << " fm/c";
  //JSINFO << "initial density profile dx = " << dx_preq << " fm";

  SetPreEqGridInfo();

  //cout<<dx_preq<<" "<<dy_preq<<" "<<endl;
  //cout<<x_min_preq<<" "<<y_min_preq<<" "<<endl;
  //cout<<nx_preq<<" "<<ny_preq<<" "<<neta_preq<<" "<<endl;

  bool is3D = Is3DPlus1();

  // last dimension, number of timesteps for predictions fix to only 1 for now !!!!
  torch::Tensor fno_input_tensor = is3D
      ? torch::zeros({n_features, nx_fno, ny_fno, neta_fno, 1})
      : torch::zeros({n_features, nx_fno, ny_fno, 1});

  string tensorShapeInput = "Input Tesnor shape from Preq : ";
  c10::IntArrayRef shape = fno_input_tensor.sizes();
  //cout<< "Tensor shape: ";
    for (int i = 0; i < shape.size(); ++i) {
      //std::cout << shape[i] << " ";
      tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
    }
  //std::cout<< tensorShapeInput.c_str() << std::endl;
  JSINFO << tensorShapeInput.c_str();

  //REBIN attempt ...
  //kinda takes time maybe some of the torch operations quicker ... think aboutit !!!
  // OMP fore the loops ...
  // try with getting torch tensor and dimension from flat input vector ... !!!!
  int k=0; // last dimension, number of timesteps for predictions fix to only 1 for now
  for (int i=0;i<nx_fno;i++)
    for (int j=0;j<ny_fno;j++) {

        double x_In = x_min_fno + i*dx_fno;
        double y_In = y_min_fno + j*dy_fno;

        if (!is3D) {
          int preq_glob_index = GetPreqCellIndex(GetPreqIdX(x_In),GetPreqIdY(y_In),0);
          double ed = pre_eq_ptr->e_[preq_glob_index];

          if (n_features == 4 ) {

              double T = GetTemperatureFromEos(ed);

              fno_input_tensor[0][i][j][k] = NormalizeEnergyDensity(ed, tau0);
              fno_input_tensor[1][i][j][k] = T;
              // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
              fno_input_tensor[2][i][j][k] = 0;
              fno_input_tensor[3][i][j][k] = 0;
          }
          else if (n_features == 3) {
              fno_input_tensor[0][i][j][k] = NormalizeEnergyDensity(ed, tau0);
              // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
              fno_input_tensor[1][i][j][k] = 0;
              fno_input_tensor[2][i][j][k] = 0;
          }
          else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}
        }
        else {
          // 3D+1: same (x,y) rebin as above, plus a loop over the eta bins of
          // the FNO grid, each mapped onto the pre-equilibrium eta grid.
          for (int l=0;l<neta_fno;l++) {

            double eta_In = eta_min_fno + l*deta_fno;

            int preq_glob_index = GetPreqCellIndex(GetPreqIdX(x_In),GetPreqIdY(y_In),GetPreqIdEta(eta_In));
            double ed = pre_eq_ptr->e_[preq_glob_index];

            // n_features == 5: (energy_density, temperature, vx, vy, vz);
            // n_features == 4: (energy_density, vx, vy, vz) -- implicit
            // temperature, derived from channel 0 via the EOS on read-back.
            // Channel 0 is normalized the same way in both cases, see
            // energy_density_normalization and PassHydroEvolutionHistoryToFramework.
            if (n_features == 5) {

                double T = GetTemperatureFromEos(ed);

                fno_input_tensor[0][i][j][l][k] = NormalizeEnergyDensity(ed, tau0);
                fno_input_tensor[1][i][j][l][k] = T;
                // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
                fno_input_tensor[2][i][j][l][k] = 0;
                fno_input_tensor[3][i][j][l][k] = 0;
                fno_input_tensor[4][i][j][l][k] = 0;
            }
            else if (n_features == 4) {
                fno_input_tensor[0][i][j][l][k] = NormalizeEnergyDensity(ed, tau0);
                // only for null preq module ... extend here at some point ... when a real dynamic evolution is used and how to get the first time-step ....
                fno_input_tensor[1][i][j][l][k] = 0;
                fno_input_tensor[2][i][j][l][k] = 0;
                fno_input_tensor[3][i][j][l][k] = 0;
            }
            else {JSWARN<<" Not enough FNO features edensity, vx,vy,vz ... to be used further in JETSCAPE for a 3D+1 model!"; exit(-1);}
          }
        }
    }

  // The time-evolution axis is the last tensor dimension in both the 2D+1
  // ([feat,x,y,tau]) and 3D+1 ([feat,x,y,eta,tau]) layouts.
  int tau_dim = is3D ? 4 : 3;
  fno_input_tensor = is3D
      ? fno_input_tensor.repeat({1, 1, 1, 1, ntau_fno})
      : fno_input_tensor.repeat({1, 1, 1, ntau_fno});

  clear_up_evolution_data();
  PassPreEqEvolutionHistoryToFramework();

  SetHydroGridInfo();

  hydro_status = INITIALIZED;

  // Full (pre-freeze-out-cutoff) length of `output`, needed again below when
  // WritePeakTemperatureDiagnostics writes the model's complete raw
  // prediction even though bulk_info.ntau gets truncated at freeze-out.
  int full_ntau = 0;

  if (hydro_status == INITIALIZED) {

    JSINFO << "Running FNO model hydro time evolution prediction ...";

    // The batched tau axis shifts by one (a batch dim is prepended) relative
    // to tau_dim computed above.
    int batched_tau_dim = tau_dim + 1;
    torch::Tensor first_ntau_steps = fno_input_tensor.slice(tau_dim, 0, 1).unsqueeze(0);

    // Autoregressive rollout: run the model up to max_rollout_windows times,
    // each time feeding the previous window's last predicted step back in as
    // a fresh initial condition, stopping early once the hottest cell has
    // cooled to/below freezeout_temperature. max_rollout_windows == 1
    // reproduces the original single-window behavior exactly.
    torch::Tensor window_ic = fno_input_tensor;
    std::vector<torch::Tensor> window_outputs;

    for (int w = 0; w < max_rollout_windows; ++w) {
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(window_ic.unsqueeze(0));
      torch::Tensor window_out = module.forward(inputs).toTensor();
      window_outputs.push_back(window_out);

      for (auto &t : inputs) t.toTensor().reset();
      inputs.clear();

      if (w + 1 >= max_rollout_windows) break;

      float max_ed = 0.0f;
      window_ic = BuildNextWindowIC(window_out, batched_tau_dim, is3D, max_ed);
      float max_T = GetTemperatureFromEos(max_ed);

      JSINFO << "Rollout window " << w << " done, hottest cell: ed = " << max_ed
             << " GeV/fm^3, T = " << max_T << " GeV";

      if (max_T <= freezeout_temperature) {
        JSINFO << "Hottest cell at/below freezeout_temperature "
               << "(" << freezeout_temperature << " GeV) -- stopping rollout.";
        break;
      }
    }

    output = first_ntau_steps;
    for (auto &w : window_outputs) {
      output = torch::cat({output, w}, batched_tau_dim);
    }

    shape = output.sizes();
    tensorShapeInput = "Output Tesnor shape from FNO : ";
    //c10::IntArrayRef shape = fno_input_tensor.sizes();
    //cout<< "Tensor shape: ";
    for (int i = 0; i < shape.size(); ++i) {
        //std::cout << shape[i] << " ";
        tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
      }
    //std::cout<< tensorShapeInput.c_str() << std::endl;
    JSINFO << tensorShapeInput.c_str() << " (" << window_outputs.size()
           << " rollout window(s))";

    full_ntau = 1 + ntau_fno * (int) window_outputs.size();
    int freezeout_cutoff = FindFreezeoutCutoff(full_ntau);
    if (freezeout_cutoff < full_ntau) {
      JSINFO << "Hottest cell reached freezeout_temperature (" << freezeout_temperature
             << " GeV) at tau = " << (bulk_info.Tau0() + bulk_info.dtau * (freezeout_cutoff - 1))
             << " fm/c (step " << (freezeout_cutoff - 1) << "/" << (full_ntau - 1)
             << ") -- truncating the evolution there, matching MUSIC's own freeze-out "
             << "termination instead of predicting the full window regardless.";
    }
    bulk_info.ntau = freezeout_cutoff;

    hydro_status = FINISHED;
  }

  auto end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  //Grifd info for bulk history ...
  //cout<<bulk_info.Tau0()<<" "<<bulk_info.TauMax()<<" "<<bulk_info.ntau<<" "<<bulk_info.dtau<<endl;
  //cout<<bulk_info.XMin()<<" "<<bulk_info.XMax()<<" "<<bulk_info.nx<<" "<<bulk_info.dx<<endl;
  //cout<<bulk_info.YMin()<<" "<<bulk_info.YMax()<<" "<<bulk_info.ny<<" "<<bulk_info.dy<<endl;

  static int fno_event_counter = 0;
  WritePeakTemperatureDiagnostics(
      "peak_temperature_event" + std::to_string(fno_event_counter) + ".csv", full_ntau);
  ++fno_event_counter;

  PassHydroEvolutionHistoryToFramework();

  JSINFO << "number of fluid cells received by the JETSCAPE: "
             << bulk_info.data.size();


  clearSurfaceCellVector();
  FindAConstantTemperatureSurface(freezeout_temperature, surfaceCellVector_);

  output.reset();


}

float FnoHydro::MaxTemperatureAtStep(const torch::Tensor &flattened, int tau_axis, int k) {
  torch::Tensor step_channel0 = flattened.select(tau_axis, k).select(0, 0);
  float max_ed = step_channel0.max().item<float>();
  // DenormalizeEnergyDensity is monotonic increasing in raw (for raw >= 0),
  // so applying it to the raw max is equivalent to (and cheaper than)
  // denormalizing every cell and re-maxing. Channel 0 is normalized at
  // every k, including k==0 (EvolveHydro's input construction applies
  // NormalizeEnergyDensity to the initial condition too), so this must
  // run unconditionally whenever a normalization scheme is active --
  // independent of whether temperature is implicit or an explicit channel.
  if (energy_density_normalization != "none") {
    int k_local = (k == 0) ? 0 : ((k - 1) % ntau_fno) + 1;
    max_ed = DenormalizeEnergyDensity(max_ed, bulk_info.Tau0() + bulk_info.dtau * k_local);
  }
  return GetTemperatureFromEos(max_ed);
}

int FnoHydro::FindFreezeoutCutoff(int full_ntau) {
  torch::Tensor flattened = torch::squeeze(output, 0);
  int tau_axis = flattened.dim() - 1;
  for (int k = 0; k < full_ntau; ++k) {
    float max_T = MaxTemperatureAtStep(flattened, tau_axis, k);
    if (max_T <= freezeout_temperature) {
      return k + 1;
    }
  }
  return full_ntau;
}

void FnoHydro::WritePeakTemperatureDiagnostics(const std::string &out_path, int full_ntau) {
  torch::Tensor flattened = torch::squeeze(output, 0);
  // flattened: [n_features, nx, ny, (neta,) ntau] -- tau is always the last dim.
  int tau_axis = flattened.dim() - 1;

  std::ofstream f(out_path);
  f << "k,tau_fm,max_energy_density_GeV_fm3,max_temperature_GeV\n";
  for (int k = 0; k < full_ntau; ++k) {
    torch::Tensor step_channel0 = flattened.select(tau_axis, k).select(0, 0);
    float max_ed = step_channel0.max().item<float>();
    if (energy_density_normalization != "none") {
      int k_local = (k == 0) ? 0 : ((k - 1) % ntau_fno) + 1;
      max_ed = DenormalizeEnergyDensity(max_ed, bulk_info.Tau0() + bulk_info.dtau * k_local);
    }
    float max_T = GetTemperatureFromEos(max_ed);
    double tau_phys = bulk_info.Tau0() + bulk_info.dtau * k;
    f << k << "," << tau_phys << "," << max_ed << "," << max_T << "\n";
  }
  f.close();
  JSINFO << "Wrote peak-temperature diagnostics to " << out_path;
}

float FnoHydro::NormalizeEnergyDensity(float raw_ed, double tau_phys) const {
  if (energy_density_normalization == "none") {
    return raw_ed;
  }
  if (energy_density_normalization == "etau75_3d") {
    // Forward of NormFn3d.etau75_3d: channel0 = epsilon^0.75 * tau.
    double base = raw_ed < 0.0 ? 0.0 : raw_ed;
    return static_cast<float>(std::pow(base, 0.75) * tau_phys);
  }
  JSWARN << "Unsupported energy_density_normalization: " << energy_density_normalization;
  exit(-1);
}

float FnoHydro::DenormalizeEnergyDensity(float raw, double tau_phys) const {
  if (energy_density_normalization == "none") {
    return raw;
  }
  if (energy_density_normalization == "etau75_3d") {
    // Inverse of NormFn3d.etau75_3d: channel0 = epsilon^0.75 * tau, so
    // epsilon = clip(channel0/tau, 0, inf)^(4/3). Matches
    // NormFn3d.inv_etau75_3d's 1e-6 floor on tau.
    double base = raw / (tau_phys + 1e-6);
    if (base < 0.0) base = 0.0;
    return static_cast<float>(std::pow(base, 4.0 / 3.0));
  }
  // InitializeHydro already validated the name; unreachable in practice.
  JSWARN << "Unsupported energy_density_normalization: " << energy_density_normalization;
  exit(-1);
}

torch::Tensor FnoHydro::BuildNextWindowIC(const torch::Tensor &window_output,
                                           int batched_tau_dim, bool is3D,
                                           float &max_energy_density_out) {
  // window_output: [1, n_features, nx_fno, ny_fno, (neta_fno,) ntau_fno].
  // Clone so in-place ops below don't corrupt window_output's storage --
  // it's also sitting (unmodified/raw) in EvolveHydro's window_outputs
  // vector, headed for final concatenation into `output`.
  torch::Tensor last = window_output.select(batched_tau_dim, ntau_fno - 1)
                            .squeeze(0)
                            .clone();
  // last: [n_features, nx_fno, ny_fno, (neta_fno,)]

  if (energy_density_normalization != "none") {
    // Same local-window tau PassHydroEvolutionHistoryToFramework uses: the
    // model has no notion of the true (growing) global tau across rollout
    // windows, so its normalization can only be interpreted relative to its
    // own trained tau0, restarting every window.
    //
    // Denormalize onto a separate copy only for the physical max reported
    // below -- `last` itself must stay in the model's own normalized
    // representation, since it's repeated into next_ic and fed straight
    // back into the model. Mutating last[0] in place here was a bug: it
    // handed the next window raw physical epsilon where the model expects
    // e.g. epsilon^0.75*tau, the same systematic input-scale mismatch fixed
    // in EvolveHydro's initial-condition construction, compounding window
    // over window -- very likely the real driver of the "rollout reheating"
    // behavior observed with max_rollout_windows > 1.
    double denorm_tau = bulk_info.Tau0() + dtau_fno * ntau_fno;
    torch::Tensor ch0_physical = last[0].clone();
    float *data = ch0_physical.data_ptr<float>();
    for (int64_t idx = 0; idx < ch0_physical.numel(); ++idx) {
      data[idx] = DenormalizeEnergyDensity(data[idx], denorm_tau);
    }
    max_energy_density_out = ch0_physical.max().item<float>();
  } else {
    max_energy_density_out = last[0].max().item<float>();
  }

  int tau_dim = batched_tau_dim - 1;
  torch::Tensor next_ic = last.unsqueeze(tau_dim);
  next_ic = is3D
      ? next_ic.repeat({1, 1, 1, 1, ntau_fno})
      : next_ic.repeat({1, 1, 1, ntau_fno});
  return next_ic;
}

void FnoHydro::SetPreEqGridInfo() {

  bulk_info.tau_min = pre_eq_ptr->GetPreequilibriumStartTime();
  //bulk_info.dtau = pre_eq_ptr->GetPreequilibriumEvodtau();
  dx_preq = ini->GetXStep();
  dy_preq = ini->GetYStep();
  deta_preq = ini->GetZStep(); // "Z" is the space-time rapidity (eta) direction

  x_min_preq = - ini->GetXMax();
  y_min_preq = - ini->GetYMax();
  eta_min_preq = - ini->GetZMax();

  nx_preq = ini->GetXSize();
  ny_preq = ini->GetYSize();
  neta_preq = ini->GetZSize();

  bulk_info.x_min = x_min_preq;
  bulk_info.y_min = y_min_preq;

  JSINFO << "Use preEq evo: tau_0 = " << bulk_info.tau_min
         << " fm/c, Xmin = Ymin = "<<bulk_info.x_min<< " fm, EtaMin = "
         << eta_min_preq << ", nEta = " << neta_preq
         << ", for FNO hydro (grid) time evolution prediction ...";
}


void FnoHydro::SetHydroGridInfo() {

  //REMARK: HArdcoded for new make read in from xml or some other info wrt FNO model (w/ and w/o super-resolution) !!!
  //Or get from output tensor dimensions ... !!!!

  bulk_info.neta = neta_fno;
  bulk_info.nx = nx_fno;
  bulk_info.ny =ny_fno;
  //bulk_info.x_min = -music_hydro_ptr->get_hydro_x_max();
  bulk_info.dx =dx_fno;
  //bulk_info.y_min = -music_hydro_ptr->get_hydro_x_max();
  bulk_info.dy = dy_fno;
  bulk_info.deta = deta_fno;

  bulk_info.dtau=dtau_fno;
  bulk_info.ntau=ntau_fno;

  if (Is3DPlus1()) {
    // Full 3D+1 model: real eta dependence, no boost-invariant vz synthesis
    // in EvolutionHistory::get_tz -- vz comes straight from the FNO output.
    bulk_info.eta_min = eta_min_fno;
    bulk_info.boost_invariant = false;
  } else {
    bulk_info.eta_min = 0;
    bulk_info.boost_invariant = true;
  }
  /*
  if (flag_preEq_output_evo_to_memory == 0) {
    bulk_info.tau_min = music_hydro_ptr->get_hydro_tau0();
    bulk_info.dtau = music_hydro_ptr->get_hydro_dtau();
    bulk_info.ntau = music_hydro_ptr->get_ntau();
  } else {
    bulk_info.ntau = music_hydro_ptr->get_ntau() + pre_eq_ptr->get_ntau();
  }
  */
}

void FnoHydro::PassPreEqEvolutionHistoryToFramework() {
  JSINFO << "Passing preEq evolution information to JETSCAPE ... ";
  auto number_of_cells = pre_eq_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of preEq fluid cells: " << number_of_cells;

  //SetPreEqGridInfo();

  for (int i = 0; i < number_of_cells; i++) {
    std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
    pre_eq_ptr->get_fluid_cell_with_index(i, fluid_cell_info_ptr);
    StoreHydroEvolutionHistory(fluid_cell_info_ptr);
  }
  pre_eq_ptr->clear_evolution_data();
}

void FnoHydro::PassHydroEvolutionHistoryToFramework() {
  JSINFO << "Passing hydro evolution information to JETSCAPE ... ";

  int number_of_cells = output.numel()/(double) n_features; //music_hydro_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of FNO prediction hydro fluid cells: " << number_of_cells;

  //Works too after permute .. and quicker than loops ... !!!!
  torch::Tensor flattened_tensor = torch::squeeze(output, 0);
  string tensorShapeInput = "Squeezed Tesnor shape from FNO : ";
  c10::IntArrayRef shape = flattened_tensor.sizes();
  //cout<< "Tensor shape: ";
  for (int i = 0; i < shape.size(); ++i) {
      //std::cout << shape[i] << " ";
      tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
    }
  //std::cout<< tensorShapeInput.c_str() << std::endl;
  JSINFO << tensorShapeInput.c_str() <<" with bulkInfo nTau = "<<bulk_info.ntau;

  // Common tail shared by every cell, independent of dimensionality.
  auto fillTailAndStore = [this](std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
    fluid_cell_info_ptr->entropy_density = 0.0; //fluidCell_ptr->sd;
    fluid_cell_info_ptr->pressure = 0.0; //fluidCell_ptr->pressure;
    fluid_cell_info_ptr->mu_B = 0.0;
    fluid_cell_info_ptr->mu_C = 0.0;
    fluid_cell_info_ptr->mu_S = 0.0;
    fluid_cell_info_ptr->qgp_fraction = 0.0;
    for (int a = 0; a < 4; a++) {
      for (int b = 0; b < 4; b++) {
        fluid_cell_info_ptr->pi[a][b] = 0.0;
      }
    }
    fluid_cell_info_ptr->bulk_Pi = 0.0;
    bulk_info.data.push_back(*fluid_cell_info_ptr);
  };

  if (!Is3DPlus1()) {
    //Tremendous speed up !!!
    auto accessor = flattened_tensor.accessor<float, 4>();

    for (int k=0;k<bulk_info.ntau;k++)
      for (int i=0;i<bulk_info.nx;i++)
          for (int j=0;j<bulk_info.ny;j++)
          {
              std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);

              if (n_features == 4 ) {
                  // k local tau-index within the current rollout window --
                  // each window restarts at 1, k==0 is the global initial
                  // condition (see BuildNextWindowIC).
                  int k_local = (k == 0) ? 0 : ((k - 1) % ntau_fno) + 1;
                  double tau_phys = bulk_info.Tau0() + bulk_info.dtau * k_local;
                  fluid_cell_info_ptr->energy_density =
                      DenormalizeEnergyDensity(accessor[0][i][j][k], tau_phys);
                  fluid_cell_info_ptr->temperature = accessor[1][i][j][k];
                  fluid_cell_info_ptr->vx = accessor[2][i][j][k];
                  fluid_cell_info_ptr->vy = accessor[3][i][j][k];
              }
              else if (n_features == 3) {

                  // k local tau-index within the current rollout window --
                  // each window restarts at 1, k==0 is the global initial
                  // condition (see BuildNextWindowIC). Channel 0 is
                  // normalized at every k, including k==0 (EvolveHydro's
                  // input construction normalizes the initial condition too).
                  int k_local = (k == 0) ? 0 : ((k - 1) % ntau_fno) + 1;
                  float eNorm = DenormalizeEnergyDensity(
                      accessor[0][i][j][k], bulk_info.Tau0()+bulk_info.dtau*k_local);
                  float mTemperature = GetTemperatureFromEos(eNorm);

                  fluid_cell_info_ptr->energy_density = eNorm;
                  fluid_cell_info_ptr->temperature = mTemperature;
                  fluid_cell_info_ptr->vx = accessor[1][i][j][k];
                  fluid_cell_info_ptr->vy = accessor[2][i][j][k];
              }
              else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}

              fluid_cell_info_ptr->vz = 0.0; // boost-invariant: get_tz() synthesizes vz from eta
              fillTailAndStore(fluid_cell_info_ptr);
            }
  }
  else {
    // Full 3D+1: rank-5 tensor [feature][x][y][eta][tau]. Loop order
    // (tau,x,y,eta) matches EvolutionHistory::CellIndex so bulk_info.data
    // fills up in the exact order the framework expects.
    auto accessor = flattened_tensor.accessor<float, 5>();

    for (int k=0;k<bulk_info.ntau;k++)
      for (int i=0;i<bulk_info.nx;i++)
        for (int j=0;j<bulk_info.ny;j++)
          for (int l=0;l<bulk_info.neta;l++)
          {
              std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);

              if (n_features == 5) {
                  // k local tau-index within the current rollout window --
                  // each window restarts at 1, k==0 is the global initial
                  // condition (see BuildNextWindowIC). Channel 0 is
                  // normalized at every k, including k==0 (EvolveHydro's
                  // input construction normalizes the initial condition too);
                  // channel 1 (temperature) is passed through unchanged by
                  // NormFn3d.py, so it needs no inverse transform here.
                  int k_local = (k == 0) ? 0 : ((k - 1) % ntau_fno) + 1;
                  double tau_phys = bulk_info.Tau0() + bulk_info.dtau * k_local;
                  fluid_cell_info_ptr->energy_density =
                      DenormalizeEnergyDensity(accessor[0][i][j][l][k], tau_phys);
                  fluid_cell_info_ptr->temperature = accessor[1][i][j][l][k];
                  fluid_cell_info_ptr->vx = accessor[2][i][j][l][k];
                  fluid_cell_info_ptr->vy = accessor[3][i][j][l][k];
                  fluid_cell_info_ptr->vz = accessor[4][i][j][l][k];
              }
              else if (n_features == 4) {

                  // k local tau-index within the current rollout window --
                  // each window restarts at 1, k==0 is the global initial
                  // condition (see BuildNextWindowIC). Channel 0 is
                  // normalized at every k, including k==0 (EvolveHydro's
                  // input construction normalizes the initial condition too).
                  int k_local = (k == 0) ? 0 : ((k - 1) % ntau_fno) + 1;
                  float eNorm = DenormalizeEnergyDensity(
                      accessor[0][i][j][l][k], bulk_info.Tau0()+bulk_info.dtau*k_local);
                  float mTemperature = GetTemperatureFromEos(eNorm);

                  fluid_cell_info_ptr->energy_density = eNorm;
                  fluid_cell_info_ptr->temperature = mTemperature;
                  fluid_cell_info_ptr->vx = accessor[1][i][j][l][k];
                  fluid_cell_info_ptr->vy = accessor[2][i][j][l][k];
                  fluid_cell_info_ptr->vz = accessor[3][i][j][l][k];
              }
              else {JSWARN<<" Not enough FNO features edensity, vx,vy,vz ... to be used further in JETSCAPE for a 3D+1 model!"; exit(-1);}

              fillTailAndStore(fluid_cell_info_ptr);
            }
  }

  flattened_tensor.reset();
}

/*
// *****************************************************************************************
// Remark: Make sure that the tensor operation flatten is doing the same as the global index
// of the bulk history --> check with filling via loops !!! and ouput histogram !!!
// ==> looks like this falattening is not the same as the global index !!!
// Check with moving time axis to front and then flatten !!!
// *****************************************************************************************

void FnoHydro::PassHydroEvolutionHistoryToFramework() {
  JSINFO << "Passing hydro evolution information to JETSCAPE ... ";

  auto start = std::chrono::high_resolution_clock::now();

  int number_of_cells = output.numel()/(double) n_features;  //music_hydro_ptr->get_number_of_fluid_cells();
  JSINFO << "total number of FNO prediction hydro fluid cells: " << number_of_cells;

  //Works too after permute .. and quicker than loops ... !!!!
  torch::Tensor flattened_tensor = torch::squeeze(output, 0);
  flattened_tensor = flattened_tensor.permute({0, 3, 1, 2});
  flattened_tensor = flattened_tensor.reshape({flattened_tensor.size(0), -1});

  string tensorShapeInput = "Flattened Tesnor shape from FNO : ";
  c10::IntArrayRef shape = flattened_tensor.sizes();
  //cout<< "Tensor shape: ";
  for (int i = 0; i < shape.size(); ++i) {
      //std::cout << shape[i] << " ";
      tensorShapeInput += std::to_string(shape[i]) ; tensorShapeInput += " ";
    }
  //std::cout<< tensorShapeInput.c_str() << std::endl;
  JSINFO << tensorShapeInput.c_str();

  auto end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();

  //Tremendous speed up !!!
  auto accessor = flattened_tensor.accessor<float, 2>();

  for (int i = 0; i < number_of_cells; i++) {
    std::unique_ptr<FluidCellInfo> fluid_cell_info_ptr(new FluidCellInfo);
    //music_hydro_ptr->get_fluid_cell_with_index(i, fluidCell_ptr);

    if (n_features == 4 ) {
        fluid_cell_info_ptr->energy_density = accessor[0][i]; // flattened_tensor[0][i].item<double>();
        fluid_cell_info_ptr->temperature = accessor[1][i]; //flattened_tensor[1][i].item<double>();
        fluid_cell_info_ptr->vx = accessor[2][i]; //flattened_tensor[2][i].item<double>();
        fluid_cell_info_ptr->vy = accessor[3][i]; //flattened_tensor[3][i].item<double>();
    }
    else if (n_features == 3) {
        fluid_cell_info_ptr->energy_density = accessor[0][i]; // flattened_tensor[0][i].item<double>();
        fluid_cell_info_ptr->temperature = GetTemperatureFromEos(accessor[0][i]); //flattened_tensor[1][i].item<double>();
        fluid_cell_info_ptr->vx = accessor[1][i]; //flattened_tensor[2][i].item<double>();
        fluid_cell_info_ptr->vy = accessor[2][i];
    }
    else {JSWARN<<" Not enough FNO features edensity, vx,vy ... to be used further in JETSCAPE !"; exit(-1);}

    fluid_cell_info_ptr->vz = 0.0 ;//fluidCell_ptr->vz;
    fluid_cell_info_ptr->entropy_density = 0.0; //fluidCell_ptr->sd;
    fluid_cell_info_ptr->pressure = 0.0; //fluidCell_ptr->pressure;
    fluid_cell_info_ptr->mu_B = 0.0;
    fluid_cell_info_ptr->mu_C = 0.0;
    fluid_cell_info_ptr->mu_S = 0.0;
    fluid_cell_info_ptr->qgp_fraction = 0.0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        fluid_cell_info_ptr->pi[i][j] = 0.0;
      }
    }
    fluid_cell_info_ptr->bulk_Pi = 0.0;
    //StoreHydroEvolutionHistory(fluid_cell_info_ptr);
    bulk_info.data.push_back(*fluid_cell_info_ptr);

  }

  flattened_tensor.reset();

  end = std::chrono::high_resolution_clock::now();
  // Calculate duration
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken loop to copy to framework: " << duration.count() << " milliseconds" << std::endl;
}
*/

float FnoHydro::GetTemperatureFromEos(float ed) {
    return fnoEOS->get_temperature((float) ed/Util::hbarc, 0)*Util::hbarc;
}

void FnoHydro::GetHydroInfo(
    Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
    GetHydroInfo_JETSCAPE(t, x, y, z, fluid_cell_info_ptr);
  //GetHydroInfo_MUSIC(t, x, y, z, fluid_cell_info_ptr);
}

void FnoHydro::GetHydroInfo_JETSCAPE(
    Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) {
  auto temp = bulk_info.get_tz(t, x, y, z);
  fluid_cell_info_ptr = std::unique_ptr<FluidCellInfo>(new FluidCellInfo(temp));
}

int FnoHydro::GetPreqCellIndex(int id_x, int id_y, int id_eta) const
{
    id_x = std::min(nx_preq - 1, std::max(0, id_x));
    id_y = std::min(ny_preq - 1, std::max(0, id_y));
    int neta_clamped = std::max(1, neta_preq);
    id_eta = std::min(neta_clamped - 1, std::max(0, id_eta));
    // (x,y,eta) with eta fastest-varying, matching FreestreamMilne's
    // output_to_vectors layout and EvolutionHistory::CellIndex; reduces to
    // the original id_x*ny_preq + id_y scheme when neta_preq == 1.
    return (id_x * ny_preq * neta_clamped + id_y * neta_clamped + id_eta);
}

void FnoHydro::GetCellIndicesFromGlobalPreqIndex(int global_index, int& id_x, int& id_y, int& id_eta) const
{
    int neta_clamped = std::max(1, neta_preq);
    id_x = global_index / (ny_preq * neta_clamped); //check, not important since mostly symmetric ... !
    id_y = (global_index / neta_clamped) % ny_preq;
    id_eta = global_index % neta_clamped;
}
