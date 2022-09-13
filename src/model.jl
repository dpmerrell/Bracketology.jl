
export CompetitionModel, save_model, load_model

mutable struct CompetitionModel
    matfac::SparseMatFacModel
    team_vec::Vector{String}
    date_vec::Vector{String}
end


function CompetitionModel(team_a_vec::AbstractVector{<:AbstractString}, 
                          team_b_vec::AbstractVector{<:AbstractString}, 
                          date_vec::AbstractVector{<:AbstractString};
                          K=3, noise_model="poisson", reg_weight=1.0)

    model = assemble_model(team_a_vec, team_b_vec, date_vec;
                           K=K, noise_model=noise_model, reg_weight=reg_weight)

    return model
end

################################################
# Model file I/O

"""
    save_model(model, filename)

Save `model` to a BSON file located at `filename`.
"""
function save_model(model, filename)
    BSON.@save filename model
end

"""
    load_model(filename)

load a model from the BSON located at `filename`.
"""
function load_model(filename)
    d = BSON.load(filename, @__MODULE__)
    return d[:model]
end


