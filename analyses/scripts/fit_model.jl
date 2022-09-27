"""
Fit a CompetitionModel to the data provided in games_df.
Save the fitted model to a BSON file.
"""


using Bracketology, CSV, DataFrames, ScikitLearnBase

include("util.jl")
include("scoring.jl")

function split_data(game_df; split_dates=["2021-06-01", "2022-06-01"])

    train_df = game_df[game_df[:,:Date] .< split_dates[1], :]
    val_df = game_df[(game_df[:,:Date] .>= split_dates[1]) .& (game_df[:,:Date] .< split_dates[2]), :]
    test_df = game_df[game_df[:,:Date] .>= split_dates[2], :]
    return train_df, val_df, test_df
end


function basic_fit(train_df; K=1, reg_weight=15.0, noise_model="normal", opt_kwargs...)
    
    # If the noise model is normal, then we first log-transform
    # the scores
    if noise_model == "normal"
        println("Log-transforming scores")
        train_df[!,:ScoreA] .= log.(train_df[!,:ScoreA] .+ 1) 
        train_df[!,:ScoreB] .= log.(train_df[!,:ScoreB] .+ 1)
    elseif noise_model == "bernoulli"
        println("Binarizing scores into win/lose") 
        train_df[!,:ScoreA] .= (train_df[!,:ScoreA] .> train_df[:,:ScoreB])
        train_df[!,:ScoreB] .= (train_df[!,:ScoreB] .> train_df[:,:ScoreA])
    end

    team_a_vec, team_b_vec, a_scores, b_scores, date_vec = unpack_df(train_df)

    model = CompetitionModel(team_a_vec, team_b_vec, date_vec;
                             K=K, reg_weight=reg_weight, noise_model=noise_model)
    
    fit!(model, team_a_vec, team_b_vec, a_scores, b_scores, date_vec;
                verbosity=1, opt_kwargs...)

    return model
end


function hyperparam_tune(train_df, val_df; kwargs...)

    ks = collect(keys(kwargs))
    vals = [kwargs[k] for k in ks]

    best_score = Inf
    best_model = nothing
    best_kwargs = nothing

    for combo in Iterators.product(vals...)
        fit_kwargs = Dict(zip(ks, combo))
        println(fit_kwargs) 
        model = basic_fit(train_df; fit_kwargs...)
        score = brier_score_538(model, val_df)
        println(string("538 Brier score: ", round(-score, digits=2), " (Higher is better)"))
        if score < best_score
            best_score = score
            best_model = model
            best_kwargs = fit_kwargs
        end
    end

    return best_model, best_kwargs
end


function main(args)

    # Model hyperparameters
    K=[0]
    reg_weight=[15.0, 20.0, 25.0]
    noise_model = "poisson"
   
    # Optimizer hyperparameters
    lr=[0.5]
    max_iter=[10000]
    abs_tol=[1e-15] 
    rel_tol=[1e-9]

    # Dates for training/validation split
    split_dates = ["2021-06-01", "2022-06-01"]

    games_csv = args[1]
    out_bson = args[2]
    existing_model_bson = nothing
    if length(args) == 3
        existing_model_bson = args[3]
    end

    # Load data; convert Date structs to strings
    game_df = DataFrame(CSV.File(games_csv))
    game_df[!,:Date] .= string.(game_df[!,:Date])

    # If an existing model isn't provided, then
    # fit one from scratch. Tune hyperparameters, too.
    if existing_model_bson == nothing
        println("Fitting model from scratch")
        train_df, val_df, test_df = split_data(game_df; split_dates=split_dates)

        best_model, best_params = hyperparam_tune(train_df, val_df; K=K, 
                                                  reg_weight=reg_weight,
                                                  noise_model=[noise_model],
                                                  lr=lr, max_iter=max_iter,
                                                  abs_tol=abs_tol, 
                                                  rel_tol=rel_tol)
        println("Best hyperparameters:")
        println(best_params)

        println("Fitting final model on combined train+validation data...")
        best_model = basic_fit(vcat(train_df, val_df); best_params...)

        println("Scoring on test set:")
        score = -brier_score_538(best_model, test_df)
        println(string("538 Brier Score: ", round(score, digits=2), " (Higher is better)"))
    else
    # Otherwise, load the existing model and update it
        println("Updating existing model from ", existing_model_bson)
        best_model = load_model(existing_model_bson)
        a_vec, b_vec, a_scores, b_scores, dates = unpack_df(game_df)
        fit_update!(best_model, a_vec, b_vec, a_scores, b_scores, dates;
                    lr=lr[1], max_iter=max_iter[1], abs_tol=abs_tol[1],
                    rel_tol=rel_tol[1]) 
    end

    save_model(best_model, out_bson)
end


main(ARGS)
