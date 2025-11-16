# Fear_Generalization
Translation in Python + PyMC of the R and JAGS project at https://osf.io/sxjak/.

All the code is commented and an explanation of every part of the code is also present in the report.

To run the models needed for the Analysis and the Simulation one can just run run_pymc 
(no need to run data preprocessing code to generate the datasets or Simulation/simulation_2 
to generate the artificial simulated dataset, all the datasets should already be present in PYMC_input_data).

The fitting results files are not present because they exceeded GitHub file limit, but all the plots regarding the 
analysis(PPC, convergence, etc..), the data visualization and the simulation related figures are all already 
present in the Plots directory.

The report pdf is in the Report directory.

The PyMC models are in the model_definitions module.

(The analysis code in the Analysis folder, the data processing in Data_Processing (original repository rds file needed), 
the code for the simulation is in the Simulation folder.)

