# A Method for Evaluating Hyperparameter Sensitivity in Reinforcement Learning

Code for running experiments as described in NeurIPS submission. 

## Requirements

Requirements are located in requirements.txt. Run:

```
pip install -r requirements.text
```

## Perform single run

```
python src/ppo_continuous_action.py  --alg_type <alg_name> --env_name <environment_name> --actor_lr <actor_learning_rate> --critic_lr <critic_learning_rate> --gae_lambda <lambda> --ent_coef <entropy coefficient> --sweep_idx -1
```

The alg_type flag indicates which PPO variant to run. Below is a map from the name given in the paper to the what it is referred to in the code. Use the value to know what to pass in for flag. E.g. if want to run Per-minibatch zero-mean normalization, pass --alg_type advn_norm_mean

PPO : lambda_ac

Per-minibatch zero-mean normalization: advn_norm_mean

Percentile scaling: advn_norm_ema

Lower bounded percentile scaling: advn_norm_max_ema

Symlog value target: symlog_critic_targets

Zero-mean normalization: norm_obs

Symlog observation: symlog_obs

The options for env_name are:

ant, hopper, halfcheetah, walker2d, swimmer

Note: if the flag sweep_idx is not set to -1, then the the script will not use hyperparameter settings passed to argparse. Rather, it will run the hyperparameter setting that the value sweep_idx corresponds to in the sweep. 

## Sweep
This repository relies on sweep utility functions provided by [PyExpUtils](https://github.com/andnp/PyExpUtils).
To define a hyperparameter sweep, create a json file over values that are to be swept as in experiments/ppo_variants_brax.json. Create an ExperimentDescription object:

```
exp = loadExperiment('experiments/ppo_variants_brax.json')
```
To see the hyperparameter settings corresponding to different values of sweep_idx:


```
params = exp.getPermutation(0)["metaParameters"]
print(params) # -> {'actor_lr': 1e-05, 'alg_type': 'lambda_ac', 'critic_lr': 1e-05, 'ent_coef': 0.001, 'env_name': 'hopper', 'gae_lambda': 0.1}

params = exp.getPermutation(100)["metaParameters"]
print(params) # -> {'actor_lr': 1e-05, 'alg_type': 'norm_obs', 'critic_lr': 0.001, 'ent_coef': 0.001, 'env_name': 'hopper', 'gae_lambda': 0.1}


params = exp.getPermutation(15000)["metaParameters"]
print(params) # ->  {'actor_lr': 1e-05, 'alg_type': 'symlog_critic_targets', 'critic_lr': 0.01, 'ent_coef': 0.001, 'env_name': 'halfcheetah', 'gae_lambda': 0.7}

```

To run a sweep setting for number_of_seeds runs:

```
python src/ppo_continuous_action.py --sweep_idx <sweep_index> --num_seeds <number_of_seeds>
```

## Plotting

Due to Github file size constraints we are unable to upload data for the individual runs from our sweep. We provide a csv file of expected performance averaged over runs for the non-divergent hyperparameters settings from our sweep in   analysis/data_expected_over_seeds.csv

To generate a performance-sensitivity plane: 

```
python analysis/generate_figures.py --plot_type sensitivity
```
To generate a effective hyperparameter dimensionality visualization: 

```
python analysis/generate_figures.py --plot_type dimensionality
```

## Citation
Please cite our work if you find it useful:

```
@inproceedings{hyperparameter_sensitivity,
    title={A Method for Evaluating Hyperparameter Sensitivity
in Reinforcement Learning},
    author={Adkins, Jacob and Bowling, Michael and White, Adam},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2024}
}
```
## Acknowledgement
The code in this repo is built on top of these libraries:
* [PureJaxRL](https://github.com/luchris429/purejaxrl)
* [Brax](https://github.com/google/brax)
* [PyExpUtils](https://github.com/andnp/PyExpUtils)