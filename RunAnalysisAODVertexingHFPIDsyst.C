#if !defined (__CINT__) || defined (__CLING__)
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "TGrid.h"
#include "TChain.h"
#include "TSystem.h"

#include "AliAnalysisAlien.h"
#include "AliAnalysisManager.h"
#include "AliAODInputHandler.h"
#include "AliMCEventHandler.h"
#include "AliAODPidHF.h"
#include "AliPhysicsSelectionTask.h"
#include "AliMultSelectionTask.h"
#include "AliAnalysisTaskPIDResponse.h"
#include "AliAnalysisTaskSEHFSystPID.h"

#endif

using namespace std;
enum system {kpp, kPbPb};

//SETTINGS
//************************************
bool runLocal=false;                        // flag to run locally on AliAOD.root + AliAOD.VertexingHF.root
TString pathToLocalAODfiles=".";            // path to find AOD files when running locally
bool runGridTest=false;                     // flag to run a grid test: true (+runLocal=false).
//To run job on GRID: runGridTest=false, runLocal=false sets the run grid mode: "full", "terminate"

void RunAnalysisAODVertexingHFPIDsyst(TString configfilename="runAnalysis_config_LHC17l3b_fast.yml", TString runMode = "full");
bool loadConfigFile(TString configfilename, std::vector<int> &runs, bool &isRunOnMC, TString &aliPhysVersion, TString &gridDataDir, TString &gridDataPattern);

void RunAnalysisAODVertexingHFPIDsyst(TString configfilename, TString runMode)
{
    // set if you want to run the analysis locally (true), or on grid (false)
    bool local = runLocal;
    // if you run on grid, specify test mode (true) or full grid model (false)
    bool gridTest = runGridTest;

    //load config
    YAML::Node config = YAML::LoadFile(configfilename.Data());
    string aliPhysVersion = config["AliPhysicVersion"].as<string>();
    string gridDataDir = config["datadir"].as<string>();
    string gridDataPattern = config["datapattern"].as<string>();
    string sSystem = config["system"].as<string>();
    int System=-1;
    if(sSystem=="pp" || sSystem=="pPb")
        System = kpp;
    else if(sSystem=="PbPb")
        System = kPbPb;
    else {
        cerr << "Only pp, pPb and PbPb are supported, please check your config file. Exit" << endl;
    }
    vector<int> runlist = config["runs"].as<vector<int>>();
    const int nruns = runlist.size();
    bool isRunOnMC = strstr(gridDataDir.c_str(),"sim");

	TString gridWorkingDir=gridDataDir;
	for(int iY=10; iY<=20; iY++) {
		gridWorkingDir.ReplaceAll(Form("/alice/sim/20%02d/",iY),"");
		gridWorkingDir.ReplaceAll(Form("/alice/data/20%02d/",iY),"");
	}
	if(gridDataPattern!="/AOD") {
		gridWorkingDir=Form("%s_%s",gridWorkingDir.Data(),gridDataPattern.data());
		gridWorkingDir.ReplaceAll("AOD","");
	}
	gridWorkingDir.ReplaceAll("/","");
	TString gridOutputDir="output";

	// since we will compile a class, tell root where to look for headers
	gInterpreter->ProcessLine(".include $ROOTSYS/include");
	gInterpreter->ProcessLine(".include $ALICE_ROOT/include");
	gInterpreter->ProcessLine(".include $ALICE_PHYSICS/include");

	// create the analysis manager
	AliAnalysisManager *mgr = new AliAnalysisManager("My Manager","My Manager");
	mgr->SetDebugLevel(10);

	// Input
	AliInputEventHandler *aodH = new AliAODInputHandler("myInpHand","title");//RP
	mgr->SetInputEventHandler(aodH);
	AliMCEventHandler  *mcH = new AliMCEventHandler("myInpHandMC","title");
	mgr->SetMCtruthEventHandler(mcH);

	if(runMode!="terminate") AliPhysicsSelectionTask *physseltask = reinterpret_cast<AliPhysicsSelectionTask *>(gInterpreter->ProcessLine(Form(".x %s (%d)", gSystem->ExpandPathName("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C"),isRunOnMC)));

	AliAnalysisTaskPIDResponse *pidResp = reinterpret_cast<AliAnalysisTaskPIDResponse *>(gInterpreter->ProcessLine(Form(".x %s (%d)", gSystem->ExpandPathName("$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C"),isRunOnMC)));

	AliMultSelectionTask *multSel = reinterpret_cast<AliMultSelectionTask *>(gInterpreter->ProcessLine(Form(".x %s", gSystem->ExpandPathName("$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C"))));
	multSel->SetAlternateOADBFullManualBypassMC("$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/data/OADB-LHC18q-DefaultMC-HIJING.root"); //for LHC18-anchored productions

	// gInterpreter->LoadMacro(".L AliAnalysisTaskSEHFSystPID.cxx++g");
	AliAnalysisTaskSEHFSystPID *task = reinterpret_cast<AliAnalysisTaskSEHFSystPID*>(gInterpreter->ProcessLine(Form(".x %s (%d,%d,\"%s\",%s,\"%s\", %f, %f, %f, %f, %f, %s, %s, %d)",gSystem->ExpandPathName("$ALICE_PHYSICS/PWGHF/vertexingHF/macros/AddTaskHFSystPID.C"),System,isRunOnMC, "", "AliVEvent::kCentral | AliVEvent::kINT7", "_pp_kINT7",0.2,1.1,0.,0.,100.,"AliAnalysisTaskSEHFSystPID::kCentV0M","AliESDtrackCuts::kOff",50)));
	task->SetfFillTreeWithRawPIDOnly();
	task->SetfFillTreeWithTrackQualityInfo();
	task->EnableDetectors(true,true,true,true);
	task->EnableParticleSpecies(true,true,true,true,true,true,true);

	if(!mgr->InitAnalysis()) return;
	mgr->SetDebugLevel(2);
	mgr->PrintStatus();
	mgr->SetUseProgressBar(1, 25);

	if(local) {
		// if you want to run locally, we need to define some input
		TChain* chainAOD = new TChain("aodTree");
		TChain *chainAODfriend = new TChain("aodTree");

		// add a few files to the chain (change this so that your local files are added)
		chainAOD->Add(Form("%s/AliAOD.root",pathToLocalAODfiles.Data()));
		chainAODfriend->Add(Form("%s/AliAOD.VertexingHF.root",pathToLocalAODfiles.Data()));
		chainAOD->AddFriend(chainAODfriend);

		// start the analysis locally, reading the events from the tchain
		mgr->StartAnalysis("local", chainAOD);
	}
	else {
		// if we want to run on grid, we create and configure the plugin
		AliAnalysisAlien *alienHandler = new AliAnalysisAlien();

		// also specify the include (header) paths on grid
		alienHandler->AddIncludePath("-I. -I$ROOTSYS/include -I$ALICE_ROOT -I$ALICE_ROOT/include -I$ALICE_PHYSICS/include");

		// make sure your source files get copied to grid
		// alienHandler->SetAnalysisSource("AliAnalysisTaskSEHFSystPID.cxx");
		// alienHandler->SetAdditionalLibs("AliAnalysisTaskSEHFSystPID.h AliAnalysisTaskSEHFSystPID.cxx");

		// select the aliphysics version. all other packages
		// are LOADED AUTOMATICALLY!
		alienHandler->SetAliPhysicsVersion(aliPhysVersion.data());

		// set the Alien API version
		alienHandler->SetAPIVersion("V1.1x");

		// select the input data
		alienHandler->SetGridDataDir(gridDataDir.data());
		alienHandler->SetDataPattern(Form("%s/*AliAOD.root",gridDataPattern.data()));
		alienHandler->SetFriendChainName("AliAOD.VertexingHF.root");

		// MC has no prefix, data has prefix 000
		if(!isRunOnMC)alienHandler->SetRunPrefix("000");

		for(int iRun=0; iRun<nruns; iRun++)
			alienHandler->AddRunNumber(runlist[iRun]);
		alienHandler->SetNrunsPerMaster(nruns);

		// number of files per subjob
		alienHandler->SetSplitMaxInputFileNumber(10);
		alienHandler->SetExecutable(Form("PIDsingletrack_%s.sh",gridWorkingDir.Data()));

		// specify how many seconds your job may take
		alienHandler->SetTTL(30000);
		alienHandler->SetJDLName("myTask.jdl");

		alienHandler->SetOutputToRunNo(true);
		alienHandler->SetKeepLogs(true);

		// merging: run with true to merge on grid
		// after re-running the jobs in SetRunMode("terminate")
		// (see below) mode, set SetMergeViaJDL(false)
		// to collect final results
		alienHandler->SetMaxMergeStages(3); //2, 3
		alienHandler->SetMergeViaJDL(false);

		// define the output folders
		alienHandler->SetGridWorkingDir(Form("PIDsingletrack_%s",gridWorkingDir.Data()));
		alienHandler->SetGridOutputDir(gridOutputDir.Data());

		// connect the alien plugin to the manager
		mgr->SetGridHandler(alienHandler);

		if(gridTest) {
			// speficy on how many files you want to run
			alienHandler->SetNtestFiles(1);
			// and launch the analysis
			alienHandler->SetRunMode("test");
			mgr->StartAnalysis("grid");
		}
		else {
			// else launch the full grid analysis
			alienHandler->SetRunMode(runMode.Data()); //terminate
			mgr->StartAnalysis("grid");
		}
	}
}
