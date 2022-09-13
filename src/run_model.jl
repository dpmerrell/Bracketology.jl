
import ScikitLearnBase: fit!

export model_data, impute_game, win_prob


"""
    fit!(model::CompetitionModel, team_a_vec, team_b_vec,
                                  a_score_vec, b_score_vec,
                                  date_vec; 
                                  lr=0.01, max_iter=1000, kwargs...)

    Fit a CompetitionModel on data from historical games.
    The data are provided in the form of 5 equal-length vectors
    containing team names, scores, and dates.

    Params:
        team_a_vec, team_b_vec: vectors of team names
        a_score_vec, b_score_vec: the teams' scores
        date_vec: the dates of the games

    Optional params: see keyword args for fit!(::SparseMatFacModel, ...)
"""
function fit!(model::CompetitionModel,
              team_a_vec::AbstractVector{<:AbstractString}, 
              team_b_vec::AbstractVector{<:AbstractString}, 
              a_score_vec::AbstractVector{<:Real}, 
              b_score_vec::AbstractVector{<:Real},
              date_vec::AbstractVector{<:AbstractString};
              lr=0.01, max_iter=1000, abs_tol=1e-12, rel_tol=1e-9, kwargs...)

    # Translate the games into a sparse I, J, V representation of scores.
    team_dates = collect(zip(model.team_vec, model.date_vec))
    team_date_to_idx = Dict(p => idx for (idx, p) in enumerate(team_dates))

    I = Int[]
    J = Int[]
    V = Float32[]

    for (date, a, b, ascore, bscore) in zip(date_vec, team_a_vec, team_b_vec,
                                            a_score_vec, b_score_vec)
        a_idx = team_date_to_idx[(a,date)]
        b_idx = team_date_to_idx[(b,date)]

        # Row="offense model", col="defense model"
        push!(I, a_idx)
        push!(J, b_idx)
        push!(V, ascore) 
        
        push!(I, b_idx)
        push!(J, a_idx)
        push!(V, bscore) 
    end
   
    # The GPU really makes a difference, if one is available
    matfac_gpu = gpu(model.matfac)
    V_gpu = gpu(V)
     
    fit!(matfac_gpu, I, J, V_gpu; lr=lr, max_iter=max_iter,
                                  abs_tol=abs_tol, rel_tol=rel_tol, kwargs...)

    model.matfac = cpu(matfac_gpu)

    return model
end


# TODO: accept "game_date" as a parameter 
#       and account for the elapsed time between 
#       game day and the teams' most recent games.
function impute_game(model::CompetitionModel, team_A::AbstractString, team_B::AbstractString)

    # Find the model indices corresponding 
    # to these teams' most recent games
    A_idx = findlast(model.team_vec .== team_A)
    B_idx = findlast(model.team_vec .== team_B)

    # Impute each team's score
    scores = impute(model.matfac, [A_idx, B_idx], [B_idx, A_idx])
    A_mean_score, B_mean_score = scores

    return Float64(A_mean_score), Float64(B_mean_score)
end


# TODO: accept "game_date" as a parameter and
#       compute win probabilities in a way that accounts for
#       time between game day and the teams' most recent game. 
"""
    win_prob(model, team_A, team_B; n_samples=10000)

    Estimate the probability of A winning against B.
    Use `model` to estimate each team's mean score; 
    assume the scores are Poisson-distributed, and estimate
    winning probability by sampling `n_samples`.
"""
function win_prob(model::CompetitionModel, team_A::AbstractString, team_B::AbstractString;
                  n_samples=10000)

    noise_model = model.matfac.noise_model 

    A_mean, B_mean = impute_game(model, team_A, team_B)

    if noise_model=="poisson"
        A_samples = rand(Poisson(A_mean), n_samples)
        B_samples = rand(Poisson(B_mean), n_samples)
    else
        throw(ArgumentError(string("For now we only support a `poisson` noise model! Not ", noise_model)))
    end

    p_win = sum(A_samples .>= B_samples) / n_samples
    return p_win
end



