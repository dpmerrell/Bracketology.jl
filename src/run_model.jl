
export model_data

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


function interpolate_params(d, dates, params)
    
end


function impute_game(model::TeamModel, team_A, team_B, date;
                     noise_model="poisson")

    # Find the model indices corresponding 
    # to these teams
    A_idx = findall(model.team_vec == team_A)
    A_dates = model.date_vec[A_idx]
    B_idx = findall(model.team_vec == team_B)
    B_dates = model.date_vec[B_idx]

    # Interpolate the team's model parameters
    # based on the date of the game
    A_X = model.matfac.X[:,A_idx]
    A_X_b = model.matfac.X_b[:,A_idx]
    Ax = interpolate_params(date, A_dates, A_X)
    Axb = interpolate_params(date, A_dates, A_X_b)

    A_Y = model.matfac.Y[:,A_idx]
    A_Y_b = model.matfac.Y_b[:,A_idx]
    Ay = interpolate_params(date, A_dates, A_Y)
    Ayb = interpolate_params(date, A_dates, A_Y_b)

    B_X = model.matfac.X[:,B_idx]
    B_X_b = model.matfac.X_b[:,B_idx]
    Bx = interpolate_params(date, B_dates, B_X)
    Bxb = interpolate_params(date, B_dates, B_X_b)

    B_Y = model.matfac.Y[:,B_idx]
    B_Y_b = model.matfac.Y_b[:,B_idx]
    By = interpolate_params(date, B_dates, B_Y)
    Byb = interpolate_params(date, B_dates, B_Y_b)

    A_mean_score = dot(Ax, By) + Axb + Byb
    B_mean_score = dot(Bx, Ay) + Bxb + Ayb

    if noise_model=="poisson"
        A_mean_score = exp(A_mean_score)
        B_mean_score = exp(B_mean_score)
    end

    return A_mean_score, B_mean_score
end



