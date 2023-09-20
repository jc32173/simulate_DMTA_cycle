# Simulating a DMTA cycle

Code for simulating a DMTA cycle, with docking used as a proxy for the Test stage.  The full workflow involves iteratively training an ML model, making predictions on a large dataset, selecting compounds for experimental/computational validation (in this case using docking to represent a binding affinity assay) and then retraining the models with the additional experimental/computational data.

- `run_iterations_script.py`: Main script to simulate the DMTA cycle.  This can be run over multiple CPUs, with the model training, prediction calculation and docking steps all parallelised.

- `run_iteration_fn.py`: Functions required by the `run_iterations_script.py`.  In particular this includes functions to ensure dataset files are only read/written to by one process at once.

- `selection_fns.py`: File containing different selection methods for choosing next round of compounds for docking and retraining ML models.  Current methods include:
    - Highest predicted value
    - Highest prediction uncertainty
    - Diverse set of molecules

The repository also includes the directory structure used to store the results of the runs.
