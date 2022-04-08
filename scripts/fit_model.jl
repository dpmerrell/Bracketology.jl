
using Bracketology, CSV, DataFrames


function main(args)

    k=1
    reg_weight=15.0
    lr=0.02
    max_iter=10000
    abs_tol=1e-15 
    rel_tol=1e-9


    games_csv = args[1]
    out_hdf = args[2]

    games_df = DataFrame(CSV.File(games_csv))

    team_model = model_data(games_df, k=k, reg_weight=reg_weight,
                                      lr=lr, max_iter=max_iter,
                                      abs_tol=abs_tol, rel_tol=rel_tol)

    save_model(out_hdf, team_model)
end


main(ARGS)
