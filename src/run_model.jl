
export model_data, impute_game


function model_data(game_df; k=3, reg_weight=1.0,
                             lr=0.01, max_iter=1000,
                             abs_tol=1e-12, rel_tol=1e-9)

    team_a_vec = game_df[:,:TeamA]
    team_b_vec = game_df[:,:TeamB]
    date_vec = map(string, game_df[:,:Date]) 
    a_scores = game_df[:,:ScoreA]
    b_scores = game_df[:,:ScoreB]

    model = assemble_model(team_a_vec, team_b_vec, date_vec;
                           K=k, loss="poisson", reg_weight=reg_weight)

    team_dates = collect(zip(model.team_vec, model.date_vec))

    team_date_to_idx = Dict(p => idx for (idx, p) in enumerate(team_dates))

    I = Int[]
    J = Int[]
    V = Float32[]

    for (date, a, b, ascore, bscore) in zip(date_vec, team_a_vec, team_b_vec,
                                            a_scores, b_scores)
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
    
    fit!(model.matfac, I, J, V; lr=lr, max_iter=max_iter,
                                abs_tol=abs_tol, rel_tol=rel_tol)

    return model
end



function impute_game(model::TeamModel, team_A, team_B;
                     noise_model="poisson")

    # Find the model indices corresponding 
    # to these teams
    println(string(team_A, " vs. ", team_B))
    A_idx = findlast(model.team_vec .== team_A)
    B_idx = findlast(model.team_vec .== team_B)

    # Interpolate the team's model parameters
    # based on the date of the game
    X = Matrix(model.matfac.X)
    Y = Matrix(model.matfac.Y)
    X_b = Vector(model.matfac.X_b)
    Y_b = Vector(model.matfac.Y_b)

    A_X = X[:,A_idx]
    A_X_b = X_b[A_idx]

    A_Y = Y[:,A_idx]
    A_Y_b = Y_b[A_idx]

    B_X = X[:,B_idx]
    B_X_b = X_b[B_idx]

    B_Y = Y[:,B_idx]
    B_Y_b = Y_b[B_idx]

    A_mean_score = dot(A_X, B_Y) + A_X_b + B_Y_b
    B_mean_score = dot(B_X, A_Y) + B_X_b + A_Y_b

    if noise_model=="poisson"
        A_mean_score = exp(A_mean_score)
        B_mean_score = exp(B_mean_score)
    end

    return Float64(A_mean_score), Float64(B_mean_score)
end


function win_prob(model::TeamModel, team_A, team_B;
                  noise_model="poisson",
                  n_samples=10000)
    
    A_mean, B_mean = impute_game(model, team_A, team_B;
                                 noise_model=noise_model)

    if noise_model=="poisson"
        A_samples = rand(Poisson(A_mean), n_samples)
        B_samples = rand(Poisson(B_mean), n_samples)
    else
        throw(ArgumentError(string("For now we only support a `poisson` noise model! Not ", noise_model)))
    end

    p_win = sum(A_samples .>= B_samples) / n_samples
    return p_win
end



