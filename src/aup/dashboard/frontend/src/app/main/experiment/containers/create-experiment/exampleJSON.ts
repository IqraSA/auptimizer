/* eslint-disable @typescript-eslint/naming-convention */
export const example = {
  name: 'new_exp',
  proposer: 'sequence',
  n_samples: 10,
  random_seed: 1,
  script: 'rosenbrock_hpo.py',
  parameter_config: [
    {
      name: 'x',
      range: [-5, 5],
      type: 'float',
    },
    {
      name: 'y',
      range: [-5, 5],
      type: 'float',
    },
  ],
  resource: 'cpu',
  n_parallel: 2,
  target: 'min',
};
