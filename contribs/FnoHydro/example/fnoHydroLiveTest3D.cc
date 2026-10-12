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

//********************************************************************************************************************
//REMARK:  cmake .. -DUSE_ROOT=ON -DUSE_MUSIC=ON -DUSE_ISS=ON (wihh muscis and libtotch QMP confflict!!!
//Download in external:  get_iSS.sh get_music.sh (Iss just for creating full hadrons for later use too ...)
//
// This is fnoHydroTest.cc with FnoHydro actually enabled (fnoHydroTest.cc
// itself defaults to MpiMusic, with FnoHydro left commented out), pointed at
// a live 3D+1 FNO model (see config/fno_3d_live_test_cparker.xml).
//********************************************************************************************************************

#include <iostream>
#include <time.h>
#include <string>

// JetScape Framework includes ...
#include "JetScape.h"
#include "JetEnergyLoss.h"
#include "JetEnergyLossManager.h"
#include "JetScapeWriterStream.h"
#include "JetScapeSignalManager.h"
#ifdef USE_HEPMC
#include "JetScapeWriterHepMC.h"
//#include "JetScapeWriterRootHepMC.h"
#endif


// User modules derived from jetscape framework clasess
#include "TrentoInitial.h"
#include "MCGlauberWrapper.h"
#include "AdSCFT.h"
#include "Matter.h"
#include "LBT.h"
#include "Martini.h"
#include "Brick.h"
#include "GubserHydro.h"
#include "MusicWrapper.h"
#include "PythiaGun.h"
#include "iSpectraSamplerWrapper.h"
#include "TrentoInitial.h"
#include "NullPreDynamics.h"
#include "PGun.h"
#include "HadronizationManager.h"
#include "Hadronization.h"
#include "ColoredHadronization.h"
#include "ColorlessHadronization.h"
//#include "HydroFromFile.h"
#include "FnoHydro.h"
#include "MusicPrimedPreEq.h"

#include <chrono>
#include <thread>

using namespace std;
using namespace Jetscape;

// Forward declarations
// / -------------------------------------

void Show();

// -------------------------------------

int main(int argc, char** argv)
{
    clock_t t; t = clock();
    time_t start, end; time(&start);

    Show();

    JetScapeLogger::Instance()->SetDebug(false);
    JetScapeLogger::Instance()->SetRemark(false);
    //SetVerboseLevel (9 a lot of additional debug output ...)
    //If you want to suppress it: use SetVerboseLevle(0) or max  SetVerboseLevle(9) or 10
    JetScapeLogger::Instance()->SetVerboseLevel(0);

    // -------------------------------------

    // argv[1] = user XML config (default: the v1/normalization=none smoke
    // test), argv[2] = output filename (default: test_out.dat). Lets the
    // same binary drive an A/B comparison across FNO model versions without
    // a separate .cc/target per model -- see config/fno_3d_live_test_v2_etau75.xml
    // for the sibling config this was built to run alongside.
    string xmlUserFile =
        "../external_packages/js-contrib/contribs/FnoHydro/config/fno_3d_live_test_cparker.xml";
    if (argc > 1)
      xmlUserFile = argv[1];

    auto jetscape = make_shared<JetScape>();
    jetscape->SetXMLMainFileName("../config/jetscape_main.xml");
    jetscape->SetXMLUserFileName(xmlUserFile);
    //jetscape->SetReuseHydro (false);
    //jetscape->SetNReuseHydro (0);

    // Initial conditions and hydro
    // MCGlauber (not TrentoInitial): matches the initial-state model that
    // actually generated FNO4d's training data (xmls/AuAu_50ev_0_10.xml).
    auto trento = make_shared<MCGlauberWrapper>();
    // MusicPrimedPreEq (not plain NullPreDynamics): primes a real
    // MCGlauber+MUSIC medium (InitialProfile=131) up to <Preequilibrium><taus>
    // and hands FnoHydro that genuine snapshot, instead of a raw (unusable)
    // string list. See MusicPrimedPreEq.h.
    auto null_predynamics = make_shared<MusicPrimedPreEq> ();
    auto pGun= make_shared<PGun> ();
    auto hydro = make_shared<FnoHydro> ();
    //auto hydro = make_shared<MpiMusic> ();

    jetscape->Add(trento);
    jetscape->Add(null_predynamics);
    jetscape->Add(pGun);
    jetscape->Add(hydro);

    // Surface sampler: Cooper-Frye particlization from the freeze-out
    // surface FnoHydro already builds via the inherited
    // FindAConstantTemperatureSurface() call at the end of EvolveHydro().
    // No FnoHydro-specific wiring needed -- JetScape's signal manager
    // connects iSS's GetHydroHyperSurface signal to whatever FluidDynamics
    // task was Add()'d as the hydro module (FnoHydro here), automatically,
    // based on task-list membership alone.
    auto iSS = make_shared<iSpectraSamplerWrapper> ();
    jetscape->Add(iSS);

    auto jlossmanager = make_shared<JetEnergyLossManager> ();
    auto jloss = make_shared<JetEnergyLoss> ();
    auto matter = make_shared<Matter> ();
    //auto lbt = make_shared<LBT> ();

    jloss->Add(matter);
    //jloss->Add(lbt);
    jlossmanager->Add(jloss);
    jetscape->Add(jlossmanager);

    // Hadronization
    auto hadroMgr = make_shared<HadronizationManager> ();
    auto hadro = make_shared<Hadronization> ();
    //auto hadroModule = make_shared<ColoredHadronization> ();
    //hadro->Add(hadroModule);
    auto colorless = make_shared<ColorlessHadronization> ();
    hadro->Add(colorless);
    hadroMgr->Add(hadro);
    jetscape->Add(hadroMgr);

    string fName="test_out.dat";
    if (argc > 2)
      fName = argv[2];

    auto writer= make_shared<JetScapeWriterAscii> (fName);
    jetscape->Add(writer);


    jetscape->Init();
    jetscape->Exec();
    jetscape->Finish();

    // -------------------------------------

    INFO_NICE<<"Finished!";
    cout<<endl;

    t = clock() - t;
    time(&end);
    printf ("CPU time: %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
    printf ("Real time: %f seconds.\n",difftime(end,start));
    cout<<endl;

}

// -------------------------------------

void Show()
{
  INFO_NICE<<"------------------------------------------";
  INFO_NICE<<"| FNO Hydro Live Test (3D+1)  JetScape ... |";
  INFO_NICE<<"------------------------------------------";
  INFO_NICE;
}

// -------------------------------------
