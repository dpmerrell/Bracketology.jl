
using Bracketology, CSV, DataFrames

function fill_bracket_script(model_hdf::String, team_csv::String)

    println("Loading model")
    model = load_model(model_hdf)

    team_df = DataFrame(CSV.File(team_csv; header=0))
    team_ls = team_df[:,1] 
    team_ls = convert(Vector{String}, team_ls)

    println("Filling bracket")
    bracket_df = fill_bracket(model, team_ls)

    return bracket_df
end


function main(args)

    model_hdf = args[1]
    team_csv = args[2]
    out_csv = args[3]

    bracket_df = fill_bracket_script(model_hdf, team_csv)

    println(string("Writing bracket to ", out_csv))
    CSV.write(out_csv, bracket_df)
end

main(ARGS)

