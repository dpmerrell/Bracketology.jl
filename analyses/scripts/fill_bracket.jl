"""
This script receives a saved model (BSON file)
and a CSV file of team names 
(their ordering defines the structure of the tournament).

It then outputs a single most probable bracket, in both
JSON and CSV format.
"""

using Bracketology, SparseMatFac, CSV, DataFrames, JSON


function fill_bracket_script(model_bson::String, team_ls)

    println("Loading model")
    model = Bracketology.load_model(model_bson)


    println("Filling bracket")
    bracket = fill_bracket(model, team_ls)

    return bracket
end


function main(args)

    model_bson = args[1]
    team_csv = args[2]
    out_csv = args[3]
    out_json = args[4]

    team_df = DataFrame(CSV.File(team_csv))
    team_ls = team_df[:,:TEAMS] 
    team_ls = convert(Vector{String}, team_ls)

    bracket = fill_bracket_script(model_bson, team_ls)
    bracket_df = bracket_to_df(bracket, team_ls)

    println(string("Writing bracket to ", out_csv))
    CSV.write(out_csv, bracket_df)

    bracket_dict = bracket_to_dict(bracket)

    println(string("Writing bracket to ", out_json))
    open(out_json, "w") do f
        JSON.print(f, bracket_dict, 4) 
    end
end

main(ARGS)

