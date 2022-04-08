module Bracketology

using CSV, DataFrames, SparseMatFac, Dates, SparseArrays,
      ScikitLearnBase, CUDA, HDF5, Distributions, LinearAlgebra

include("model.jl")
include("assemble_model.jl")
include("run_model.jl")
include("fill_bracket.jl")


end # module
