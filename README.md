# VAE-MDPs

## Installation
We provide two `conda` environment files that can be used to re-create our `python` 
environment and reproduce our results:
- `environment.yml` (using TensorFlow CPU)
- `environment_gpu.yml` (using TensorFlow GPU)

These files can be found in the `conda_environment` directory and explicitly list all the dependencies required
for our tool. 
Note that these environments have been tested under `conda 4.10.1`.

Note: `reverb` currently only supports Linux based OSes. Our tool can be used without `reverb` if you don't use
prioritized replay buffers.

In the following, we detail how to create automatically the conda environment from the environment CPU file,
but you can easily create an environment for GPU by replacing `environment.yml` by
`environment_gpu.yml`.
1. Create the environment from `environment.yml`:
   ```shell
   cd conda_environment
   conda env create -f environment.yml
   ```
2. The environment ``vae_mdp`` (or `vae_mdp_gpu`) is now created.
To use [`reverb`](https://github.com/deepmind/reverb) replay buffers, we need
   to indicate the variable `LD_LIBRARY_PATH` to conda.
   We provide the installation script `set_environment_variables.sh`
   that makes the environment variable become activate when the environment is activated:
   ````shell 
   conda activate vae_mdp  # or vae_mdp_gpu
   ./set_environment_variables
   # reactivate the environment to apply the changes
   conda deactivate
   conda activate vae_mdp
   ````
   The vae_mdp environment should now work properly on your machine.
3. (Optional) Alternatively, you can indicate manually the environment variable `LD_LIBRARY_PATH` to conda as follows:
   ```shell
   conda activate vae_mdp  # or vae_mdp_gpu
   cd $CONDA_PREFIX
   mkdir -p ./etc/conda/activate.d
   mkdir -p ./etc/conda/deactivate.d
   touch ./etc/conda/activate.d/env_vars.sh
   touch ./etc/conda/deactivate.d/env_vars.sh
   ```
   
   Edit `./etc/conda/activate.d/env_vars.sh` as follows:
   ```shell
   #!/bin/sh
   
   ENV_NAME=vae_mdp  # or vae_mdp_gpu
   export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
   export LD_LIBRARY_PATH=${HOME}/anaconda3/envs/${ENV_NAME}/lib/:${LD_LIBRARY_PATH}
   ```

   Edit `./etc/conda/deactivate.d/env_vars.sh` as follows:
   ```shell
   #!/bin/sh

   export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
   unset OLD_LD_LIBRARY_PATH
   ```