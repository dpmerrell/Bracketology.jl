
import ScikitLearnBase: fit!

export fit!, fit_update!, simulate_game, impute_game, win_prob


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
              lr=0.01, max_iter=1000, abs_tol=1e-12, rel_tol=1e-9, 
              noise_model="poisson", kwargs...)

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
                                  abs_tol=abs_tol, rel_tol=rel_tol, 
                                  kwargs...)
    
    # For Normal-distributed data: 
    # estimate a variance from the residuals
    if matfac_gpu.noise_model == "normal" 
        model.uncertainty_param = compute_avg_residuals(matfac_gpu, I, J, V_gpu)
        println(string("ESTIMATED VARIANCE: ", model.uncertainty_param))
    end

    model.matfac = cpu(matfac_gpu)

    return model
end


function compute_avg_residuals(matfac, I, J, V)
    invlink_fn = SparseMatFac.INVLINK_FUNCTION_MAP[matfac.noise_model]
    loss_fn = SparseMatFac.LOSS_FUNCTION_MAP[matfac.noise_model]
    
    X_view = view(matfac.X, :, I)
    Y_view = view(matfac.Y, :, J)
    row_b_view = view(matfac.row_transform, I)
    col_b_view = view(matfac.col_transform, J)
    
    residual = SparseMatFac.neg_log_likelihood(X_view, Y_view, 
                                               row_b_view, col_b_view, V,
                                               invlink_fn, loss_fn)
    residual *= (2/length(V))

    return residual
end


"""
    fit_update!(model, A_vec, B_vec, 
                       A_scores, B_scores, 
                       dates; kwargs...)

    Update a given model's fit on a dataset that has new 
    games appended to it.  

    May be more economical than totally re-fitting the model
    every time we get new data.
"""
function fit_update!(model::CompetitionModel, A_vec::AbstractVector{<:AbstractString},
                                              B_vec::AbstractVector{<:AbstractString},
                                              A_scores::AbstractVector{<:Number},
                                              B_scores::AbstractVector{<:Number},
                                              dates::AbstractVector{<:AbstractString}; kwargs...)

    # Update the model's (team_vec, date_vec) with new teams and dates
    old_teamdates = collect(zip(model.team_vec, model.date_vec))
    new_teamdates = unique_team_dates(A_vec, B_vec, dates)
    K = size(model.matfac.X,1)
    new_N = length(new_teamdates)

    # Define a map from the old indices to the new indices
    new_teamdate_to_idx = Dict(td => i for (i, td) in enumerate(new_teamdates))
    old_idx_to_new_idx = [new_teamdate_to_idx[td] for td in old_teamdates]

    # Allocate the model's new parameters
    new_X = zeros(K, new_N)
    new_Y = zeros(K, new_N)
    new_row_trans_b = zeros(new_N)
    new_col_trans_b = zeros(new_N)
    new_regmat = assemble_regmat(new_teamdates)
    new_inv_view_mat = SparseMatrixCSC{Float32,Int32}(sparse(I, new_N, new_N))
    
    # Update the model's parameters
    new_X[:,old_idx_to_new_idx] .= model.matfac.X
    new_Y[:,old_idx_to_new_idx] .= model.matfac.Y
    new_row_trans_b[old_idx_to_new_idx] .= model.matfac.row_transform.b
    new_col_trans_b[old_idx_to_new_idx] .= model.matfac.col_transform.b

    model.team_vec = [p[1] for p in new_teamdates]
    model.date_vec = [p[2] for p in new_teamdates]
    model.matfac.X = new_X
    model.matfac.Y = new_Y
    model.matfac.row_transform.b = new_row_trans_b
    model.matfac.row_transform.inv_view_mat = new_inv_view_mat 
    model.matfac.col_transform.b = new_col_trans_b
    model.matfac.col_transform.inv_view_mat = new_inv_view_mat

    # Update regularizers
    model.matfac.X_reg.mat = new_regmat
    model.matfac.Y_reg.mat = new_regmat
    model.matfac.row_transform_reg.mat = new_regmat
    model.matfac.col_transform_reg.mat = new_regmat

    # Fit the model
    fit!(model, A_vec, B_vec, A_scores, B_scores, dates; kwargs...)

    return model
end


# TODO: accept "game_date" as a parameter 
#       and account for the elapsed time between 
#       game day and the teams' most recent games.
function impute_game(model::CompetitionModel, 
                     team_A::AbstractString, team_B::AbstractString)

    # Find the model indices corresponding 
    # to these teams' most recent games
    A_idx = findlast(model.team_vec .== team_A)
    B_idx = findlast(model.team_vec .== team_B)

    # Impute each team's score
    scores = impute(model.matfac, [A_idx, B_idx], [B_idx, A_idx])
    A_mean_score, B_mean_score = scores

    return Float64(A_mean_score), Float64(B_mean_score)
end

sigmoid(x) = 1 / (1 + exp(-x))

function simulate_game(model::CompetitionModel,
                       team_A::AbstractString, team_B::AbstractString,
                       date::AbstractString; n_samples=10000, noise_model="poisson")
    
    K = size(model.matfac.X, 1)
    A_idx = findlast(model.team_vec .== team_A)
    B_idx = findlast(model.team_vec .== team_B)
    A_date = model.date_vec[A_idx]
    B_date = model.date_vec[B_idx]
    date = Date(date)

    # Estimate the uncertainty due to passage of time since
    # the last game
    A_dt = Dates.days(date - Date(A_date))
    A_std = sqrt(A_dt / 365.0 / model.matfac.lambda_X)
    B_dt = Dates.days(date - Date(B_date))
    B_std = sqrt(B_dt / 365.0 / model.matfac.lambda_X)
    
    # Sample possible values of the parameters
    A_X = A_std.*randn(K, n_samples) .+ model.matfac.X[:,A_idx]
    A_Y = A_std.*randn(K, n_samples) .+ model.matfac.Y[:,A_idx]
    A_row_b = A_std.*randn(n_samples) .+ model.matfac.row_transform.b[A_idx]
    A_col_b = A_std.*randn(n_samples) .+ model.matfac.col_transform.b[A_idx]
    B_X = B_std.*randn(K, n_samples) .+ model.matfac.X[:,B_idx]
    B_Y = B_std.*randn(K, n_samples) .+ model.matfac.Y[:,B_idx]
    B_row_b = B_std.*randn(n_samples) .+ model.matfac.row_transform.b[B_idx]
    B_col_b = B_std.*randn(n_samples) .+ model.matfac.col_transform.b[B_idx]
   
    # Compute game score means from the sampled parameters 
    A_means = vec(sum(A_X .* B_Y; dims=1)) .+ A_row_b .+ B_col_b
    B_means = vec(sum(B_X .* A_Y; dims=1)) .+ B_row_b .+ A_col_b

    # Finally, sample scores from the appropriate distribution
    if noise_model == "poisson"
        A_means = exp.(A_means)
        B_means = exp.(B_means)
        A_scores = map(m -> rand(Poisson(m)), A_means)
        B_scores = map(m -> rand(Poisson(m)), B_means)
    elseif noise_model == "normal"
        sigma = sqrt(model.uncertainty_param)
        A_scores = map(m -> rand(Normal(m, sigma)), A_means)
        B_scores = map(m -> rand(Normal(m, sigma)), B_means)
    elseif noise_model == "bernoulli"
        A_scores = sigmoid.(A_means)
        B_scores = sigmoid.(B_means)
    end

    return A_scores, B_scores
end


# TODO: accept "game_date" as a parameter and
#       compute win probabilities in a way that accounts for
#       time between game day and the teams' most recent game. 
"""
    win_prob(model, team_A, team_B, date; n_samples=10000)

    Estimate the probability of A winning against B on `date`.
    Also estimate the mean scores of A and B.
    Rely on the probabilistic assumptions encoded in `model`.
    Quantities are computed by simulating the game `n_samples` times.
"""
function win_prob(model::CompetitionModel, team_A::AbstractString, 
                                           team_B::AbstractString, 
                                           date::AbstractString;
                                           n_samples=10000)

    noise_model = model.matfac.noise_model 

    if noise_model in ("poisson", "normal","bernoulli")
        A_samples, B_samples = simulate_game(model, team_A, team_B, date; 
                                             n_samples=n_samples, noise_model=noise_model)
    else
        throw(ArgumentError(string("For now we only support `poisson`, `normal`, and `bernoulli` noise models! Not ", noise_model)))
    end

    A_mean = sum(A_samples)/n_samples
    B_mean = sum(B_samples)/n_samples
    p_win = sum(A_samples .> B_samples) / n_samples
    p_draw = sum(A_samples .== B_samples) / n_samples

    return p_win + 0.5*p_draw, A_mean, B_mean
end



