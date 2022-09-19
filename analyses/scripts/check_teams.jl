"""
Utilities for checking whether you have consistent team names.
Often a team goes by different aliases or abbreviations
in different datasets; it's important to check for inconsistencies.
"""


using Bracketology, CSV, DataFrames


function check_teams(model, team_csv)

    model_teams = model.team_vec
   
    team_df = DataFrame(CSV.File(team_csv;header=0))
    team_ls = team_df[:,1]

    println("MODEL TEAMS:")
    for team in sort(collect(model_teams))
        println(team)
    end

    for team in team_ls
        if !in(team, model_teams)
            println(string(team, " NOT FOUND IN MODEL TEAMS"))
        end
    end

end

function main(args)

    model_bson = args[1]
    team_csv = args[2]

    model = load_model(model_bson)

    check_teams(model, team_csv)
end


main(ARGS)
