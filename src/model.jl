
export CompetitionModel, save_model, load_model

mutable struct CompetitionModel
    matfac::SparseMatFacModel
    team_vec::Vector{String}
    date_vec::Vector{String}
    uncertainty_param::Number
end


function CompetitionModel(team_a_vec::AbstractVector{<:AbstractString}, 
                          team_b_vec::AbstractVector{<:AbstractString}, 
                          date_vec::AbstractVector{<:AbstractString};
                          team_a_covariates::Union{Nothing,AbstractMatrix}=nothing,
                          team_b_covariates::Union{Nothing,AbstractMatrix}=nothing,
                          K=0, noise_model="poisson", reg_weight=1.0, 
                          constant_term=0.0)

    model = assemble_model(team_a_vec, team_b_vec, date_vec;
                           team_a_covariates=team_a_covariates,
                           team_b_covariates=team_b_covariates,
                           K=K, noise_model=noise_model, reg_weight=reg_weight,
                           constant_term=constant_term)

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


