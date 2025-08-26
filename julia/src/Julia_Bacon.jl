module Julia_Bacon

export run_hmc, read_data, energy_function, mean_ages, plot_age_depth

using CSV, Random, DataFrames, Distributions, LinearAlgebra, ForwardDiff, ProgressMeter, Base.Threads, Statistics, Plots, SharedArrays

include("bacon_HMC.jl")

end
