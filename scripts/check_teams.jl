
using HDF5, CSV, DataFrames


function check_teams(model_hdf, team_csv)

    f = h5open(model_hdf, "r")
    model_teams = Set(f["team_vec"][:])
   
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

    model_hdf = args[1]
    team_csv = args[2]

    check_teams(model_hdf, team_csv)
end


main(ARGS)
