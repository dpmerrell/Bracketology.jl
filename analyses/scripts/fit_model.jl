"""
Fit a CompetitionModel to the data provided in games_df.
Save the fitted model to a BSON file.
"""


using Bracketology, CSV, DataFrames, ScikitLearnBase

include("util.jl")
include("scoring.jl")

function split_data(game_df; split_date="2020-06-01")

    train_df = game_df[game_df[:,:Date] .< split_date, :]
    val_df = game_df[game_df[:,:Date] .>= split_date, :]
    return train_df, val_df
end


function basic_fit(train_df; K=1, reg_weight=15.0, opt_kwargs...)

    team_a_vec, team_b_vec, a_scores, b_scores, date_vec = unpack_df(train_df)

    model = CompetitionModel(team_a_vec, team_b_vec, date_vec;
                             K=K, reg_weight=reg_weight)
    
    fit!(model, team_a_vec, team_b_vec, a_scores, b_scores, date_vec;
                verbosity=0, opt_kwargs...)

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
        score = brier_score(model, val_df)
        println(string("Brier score: ", score))
        if score < best_score
            best_score = score
            best_model = model
            best_kwargs = fit_kwargs
        end
    end

    return best_kwargs
end


function main(args)

    # Model hyperparameters
    K=[1]
    reg_weight=[10.0, 15.0, 20.0]

    # Optimizer hyperparameters
    lr=[0.25]
    max_iter=[10000]
    abs_tol=[1e-15] 
    rel_tol=[1e-9]

    games_csv = args[1]
    out_bson = args[2]

    game_df = DataFrame(CSV.File(games_csv))
    game_df[!,:Date] .= string.(game_df[!,:Date])

    train_df, val_df = split_data(game_df)

    best_params = hyperparam_tune(train_df, val_df; K=K, 
                                                    reg_weight=reg_weight,
                                                    lr=lr, max_iter=max_iter,
                                                    abs_tol=abs_tol, 
                                                    rel_tol=rel_tol)
    println("Best hyperparameters:")
    println(best_params)

    println("Fitting final model on combined train+validation data...")
    best_model = basic_fit(game_df; best_params...)

    save_model(best_model, out_bson)
end


main(ARGS)
