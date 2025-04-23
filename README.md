# JNR skyrmions.
This repository contains Julia code developed for the paper [Locations of JNR skyrmions](), written by myself and Josh Cork. 

Given an initialised skyrmions the method `make_JNR!` allows one to define its pion field by the approximation of Harland-Sutcliffe. In addition, `JNR_positions` and `plot_JNR_positions` allow one to find the locations of consituent JNR skyrmions defined by poles and weights. To use these functions, download `jnr_skyrmions.jl` and in Julia run `include("/.../jnr_skyrmions.jl")`.

`tutorial.jl` demonstrates how the code may be used.

## Requirements
I shall not give a complete list of system requirements, but as a rough guideline:
* the code was written in [Julia](https://www.sagemath.org/) version 1.9.0,
* a basic essential requirement is to have [Skyrmions3D](https://github.com/chrishalcrow/Skyrmions3D.jl) installed in Julia (this code was developed with the current version of that package as of April 2025), 
* to use additional features beyond the most basic `make_JNR!` command, one requires the Julia packages [Optim](https://julianlsolvers.github.io/Optim.jl/stable/), [IntervalArithmetic](https://juliaintervals.github.io/IntervalArithmetic.jl/stable/), [IntervalRootFinding](https://juliaintervals.github.io/IntervalRootFinding.jl/stable/), and [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl).
