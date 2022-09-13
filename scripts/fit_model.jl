"""
Fit a CompetitionModel to the data provided in games_df.
Save the fitted model to a BSON file.
"""
using Bracketology, CSV, DataFrames


function main(args)

    # Model hyperparameters
    K=1
    reg_weight=15.0

    # Optimizer hyperparameters
    lr=0.02
    max_iter=10000
    abs_tol=1e-15 
    rel_tol=1e-9

    games_csv = args[1]
    out_bson = args[2]

    games_df = DataFrame(CSV.File(games_csv))

    team_a_vec = game_df[:,:TeamA]
    team_b_vec = game_df[:,:TeamB]
    date_vec = map(string, game_df[:,:Date]) 
    a_scores = game_df[:,:ScoreA]
    b_scores = game_df[:,:ScoreB]

    model = CompetitionModel(team_a_vec, team_b_vec, date_vec;
                             K=K, reg_weight=reg_weight)
    
    fit!(model, team_a_vec, team_b_vec, a_scores, b_scores, date_vec;
                lr=lr, max_iter=max_iter,
                bs_tol=abs_tol, rel_tol=rel_tol)

    save_model(team_model, out_bson)
end


main(ARGS)
