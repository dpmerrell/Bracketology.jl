"""
Fit a CompetitionModel to the data provided in games_df.
Save the fitted model to a BSON file.
"""


using Bracketology, CSV, DataFrames, ScikitLearnBase
import Base.Threads: @threads

include("util.jl")
include("scoring.jl")

function split_data(game_df; split_dates=["2021-06-01", "2022-06-01"])

    train_df = game_df[game_df[:,:Date] .< split_dates[1], :]
    val_df = game_df[(game_df[:,:Date] .>= split_dates[1]) .& (game_df[:,:Date] .< split_dates[2]), :]
    test_df = game_df[game_df[:,:Date] .>= split_dates[2], :]
    return train_df, val_df, test_df
end


function basic_fit(train_df; K=1, reg_weight=15.0, noise_model="normal", print_iter=10, opt_kwargs...)
    
    # If the noise model is normal, then we first log-transform
    # the scores
    if noise_model == "normal"
        println("Log-transforming scores")
        train_df[!,:ScoreA] .= log.(train_df[!,:ScoreA] .+ 1) 
        train_df[!,:ScoreB] .= log.(train_df[!,:ScoreB] .+ 1)
        constant_term = 0.5*sum(train_df[:,:ScoreA] .+ train_df[:,:ScoreB])/size(train_df,1)
    elseif noise_model == "bernoulli"
        println("Binarizing scores into win/lose") 
        train_df[!,:ScoreA] .= (train_df[!,:ScoreA] .> train_df[:,:ScoreB])
        train_df[!,:ScoreB] .= (train_df[!,:ScoreB] .> train_df[:,:ScoreA])
        constant_term = 0.5*sum(train_df[:,:ScoreA] .+ train_df[:,:ScoreB])/size(train_df,1)
    elseif noise_model == "poisson"
        constant_term = 0.5*sum(log.(train_df[:,:ScoreA] .+ 1) .+ log.(train_df[:,:ScoreB] .+ 1))/size(train_df,1)
        constant_term = exp(constant_term)
        constant_term -= 1
        constant_term = log(constant_term)
    end

    team_a_vec, team_b_vec, a_scores, b_scores, date_vec, a_covariates, b_covariates = unpack_df(train_df)

    model = CompetitionModel(team_a_vec, team_b_vec, date_vec;
                             K=K, reg_weight=reg_weight, noise_model=noise_model,
                             team_a_covariates=a_covariates, 
                             team_b_covariates=b_covariates, 
                             constant_term=constant_term)
    
    fit!(model, team_a_vec, team_b_vec, a_scores, b_scores, date_vec;
                verbosity=1, print_iter=print_iter, opt_kwargs...)

    return model
end


function sequential_score(train_df, val_df; chunksize=96, fit_kwargs...)

    train_df = deepcopy(train_df)
    val_df = deepcopy(val_df)

    score = 0
    idx = 0
    N_val = size(val_df, 1)
    model = nothing
    chunk_idx = 1

    # For each chunk of validation samples
    while idx < N_val
        # fit the model to the training data
        println(string("\tFitting on chunk ", chunk_idx))
        model = basic_fit(train_df; fit_kwargs...)
 
        # score on the chunk of validation samples;
        # add the score to the total
        thischunk_size = min(chunksize, size(val_df, 1))
        score += brier_score_538(model, val_df[1:thischunk_size, :])

        # append the chunk of validation samples to the training data;
        append!(train_df, val_df[1:thischunk_size, :])
        
        # remove the chunk from the validation set
        if thischunk_size < size(val_df,1)
            val_df = val_df[(thischunk_size+1):end, :]
        end

        idx += thischunk_size
        chunk_idx += 1
    end

    return model, score
end


function hyperparam_tune(train_df, val_df; kwargs...)

    ks = collect(keys(kwargs))
    vals = [kwargs[k] for k in ks]
    combos = [Dict(zip(ks, combo)) for combo in Iterators.product(vals...)]

    scores = zeros(length(combos))
    models = Vector{CompetitionModel}(undef, length(combos)) 

    @threads for i=1:length(combos)
        fit_kwargs = combos[i]
        println(fit_kwargs) 
        model, score = sequential_score(train_df, val_df; fit_kwargs...)
        println(string(combos, "\n538 Brier score: ", round(-score, digits=2), " (Higher is better)"))

        scores[i] = score
        models[i] = model
    end

    best_idx = argmin(scores)
    best_model = models[best_idx]
    best_kwargs = combos[best_idx]

    return best_model, best_kwargs
end


function main(args)

    # Model hyperparameters
    K=[0]
    #reg_weight=[10.0, 20.0, 30.0]
    reg_weight=[30.0, 40.0, 50.0]
    noise_model = "poisson"
   
    # Optimizer hyperparameters
    lr=[0.2]
    max_iter=[1000]
    abs_tol=[1e-15] 
    rel_tol=[1e-9]

    # Dates for training/validation split
    split_dates = ["2021-06-01", "2022-06-01"]

    games_csv = args[1]
    out_bson = args[2]

    # Load data; convert Date structs to strings
    game_df = DataFrame(CSV.File(games_csv))
    game_df[!,:Date] .= string.(game_df[!,:Date])

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

    println("Scoring on test data...")
    append!(train_df, val_df)
    _, score = sequential_score(train_df, test_df; best_params...)
    println(string("538 Brier Score: ", round(score, digits=2), " (Higher is better)"))

    println("Fitting final model on train+val+test data")
    append!(train_df, test_df)
    best_model = basic_fit(train_df; best_params...) 
    println(string("Saving final model to file: ", out_bson))
    save_model(best_model, out_bson)
end


main(ARGS)
