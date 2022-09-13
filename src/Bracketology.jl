module Bracketology

using SparseMatFac, Dates, SparseArrays, BSON, Flux, Functors,
      ScikitLearnBase, CUDA, Distributions, LinearAlgebra

include("model.jl")
include("layers.jl")
include("regularizers.jl")
include("assemble_model.jl")
include("run_model.jl")
include("fill_bracket.jl")
include("scoring.jl")

end # module
