# hNODE: Hyper Neural Ordinary Differential Equation for Time Series Analysis

[![Build Status](https://github.com/dr-Fade/hNODE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dr-Fade/hNODE.jl/actions/workflows/CI.yml?query=branch%3Amain)

This repository contains the Julia implementation of the **hNODE** (Hyper Neural Ordinary Differential Equation) model, as described in the paper "Univariate time series analysis with hyper neural ODE". The hNODE is designed to address limitations of traditional neural ODE models by separating input data from the latent space where the neural ODE operates.

The implementation is generalized to handle multivariate time series data.

## Usage

Example of creating and running a simple model:

```julia
using hNODEModel, DifferentialEquations, Lux, Random

input_n = 32
latent_dims = 3

encoder = Dense(input_n => latent_dims)
decoder = Dense(latent_dims => 1)
ode = Dense(latent_dims => latent_dims)
control = Dense(input_n => Lux.parameterlength(ode))
solver = Euler()
hnode = hNODEModel.hNODE(; encoder=encoder, decoder=decoder, control=control, ode=ode, solver=solver)
ps, st = Lux.setup(Random.default_rng(), hnode)

data = rand(input_n, 1)
hnode_input = hNODEModel.hNODEInput(data, data)

(ys, u0s), st = hnode(input, ps, st)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
