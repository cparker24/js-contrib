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

// Standard-hydro (MUSIC) sibling of fnoHydroLiveTest3D.cc: identical
// PGun/IS/Preequilibrium/Eloss/Hadronization pipeline, MpiMusic in place of
// FnoHydro, so the two produce directly comparable ASCII output for an A/B
// jet-energy-loss comparison. See config/fno_3d_live_test_v2_music.xml.

#include <iostream>
#include <time.h>
#include <string>

// JetScape Framework includes ...
#include "JetScape.h"
#include "JetEnergyLoss.h"
#include "JetEnergyLossManager.h"
#include "JetScapeWriterStream.h"
#include "JetScapeWriterFinalStateStream.h"
#include "JetScapeSignalManager.h"
#ifdef USE_HEPMC
#include "JetScapeWriterHepMC.h"
#endif

// User modules derived from jetscape framework clasess
#include "TrentoInitial.h"
#include "MCGlauberWrapper.h"
#include "Matter.h"
#include "MusicWrapper.h"
#include "NullPreDynamics.h"
#include "PGun.h"
#include "iSpectraSamplerWrapper.h"
#include "HadronizationManager.h"
#include "Hadronization.h"
#include "ColorlessHadronization.h"

#include <chrono>
#include <thread>

using namespace std;
using namespace Jetscape;

void Show();

// Peak-temperature diagnostic (music_peak_temperature_event<N>.csv) is
// written from inside MpiMusic::EvolveHydro() itself (MusicWrapper.cc) --
// bulk_info gets cleared by the framework's per-event cleanup before Exec()
// returns control here, so it can't be read from the driver after the fact.

int main(int argc, char** argv)
{
    clock_t t; t = clock();
    time_t start, end; time(&start);

    Show();

    JetScapeLogger::Instance()->SetDebug(false);
    JetScapeLogger::Instance()->SetRemark(false);
    JetScapeLogger::Instance()->SetVerboseLevel(0);

    // argv[1] = user XML config, argv[2] = output filename (default: test_out.dat).
    string xmlUserFile =
        "../external_packages/js-contrib/contribs/FnoHydro/config/fno_3d_live_test_v2_music.xml";
    if (argc > 1)
      xmlUserFile = argv[1];

    auto jetscape = make_shared<JetScape>();
    jetscape->SetXMLMainFileName("../config/jetscape_main.xml");
    jetscape->SetXMLUserFileName(xmlUserFile);

    // Initial conditions and hydro
    // MCGlauber (not TrentoInitial): matches the initial-state model that
    // actually generated FNO4d's training data (xmls/AuAu_50ev_0_10.xml).
    auto trento = make_shared<MCGlauberWrapper>();
    auto null_predynamics = make_shared<NullPreDynamics> ();
    auto pGun= make_shared<PGun> ();
    auto hydro = make_shared<MpiMusic> ();

    jetscape->Add(trento);
    jetscape->Add(null_predynamics);
    jetscape->Add(pGun);
    jetscape->Add(hydro);

    // Surface sampler: Cooper-Frye particlization from MUSIC's own
    // freeze-out surface (surface_in_memory=1 in the <MUSIC> XML block).
    // Mirrors fnoHydroLiveTest3D.cc's iSS wiring, so the two arms produce
    // directly comparable sampled-hadron output for the same initial
    // conditions (matched <Random><seed>).
    auto iSS = make_shared<iSpectraSamplerWrapper> ();
    jetscape->Add(iSS);

    auto jlossmanager = make_shared<JetEnergyLossManager> ();
    auto jloss = make_shared<JetEnergyLoss> ();
    auto matter = make_shared<Matter> ();

    jloss->Add(matter);
    jlossmanager->Add(jloss);
    jetscape->Add(jlossmanager);

    // Hadronization
    auto hadroMgr = make_shared<HadronizationManager> ();
    auto hadro = make_shared<Hadronization> ();
    auto colorless = make_shared<ColorlessHadronization> ();
    hadro->Add(colorless);
    hadroMgr->Add(hadro);
    jetscape->Add(hadroMgr);

    string fName="test_out.dat";
    if (argc > 2)
      fName = argv[2];

    auto writer= make_shared<JetScapeWriterAscii> (fName);
    jetscape->Add(writer);

    // Flat (index pid pstat E px py pz) per-event hadron list -- see
    // fnoHydroLiveTest3D.cc for why this is added alongside the full writer.
    string fNameFinalState = fName;
    if (fNameFinalState.size() >= 4 &&
        fNameFinalState.compare(fNameFinalState.size() - 4, 4, ".dat") == 0)
      fNameFinalState.resize(fNameFinalState.size() - 4);
    fNameFinalState += "_final_state_hadrons.dat";
    auto finalStateWriter = make_shared<JetScapeWriterFinalStateHadronsAscii>();
    finalStateWriter->SetOutputFileName(fNameFinalState);
    jetscape->Add(finalStateWriter);

    jetscape->Init();
    jetscape->Exec();
    jetscape->Finish();

    INFO_NICE<<"Finished!";
    cout<<endl;

    t = clock() - t;
    time(&end);
    printf ("CPU time: %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
    printf ("Real time: %f seconds.\n",difftime(end,start));
    cout<<endl;
}

void Show()
{
  INFO_NICE<<"---------------------------------------------";
  INFO_NICE<<"| Standard Hydro (MUSIC) Live Test  JetScape |";
  INFO_NICE<<"---------------------------------------------";
  INFO_NICE;
}
