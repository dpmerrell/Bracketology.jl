
using CSV, DataFrames, JSON


function rec_make_bracket(team_ls::Vector{String}, depth::Int)

    N = length(team_ls)
    if N == 2
        bracket = Dict(team_ls[1] => nothing,
                       team_ls[2] => nothing
                      )
    else
        Nd2 = div(N,2)
        l_subbracket = rec_make_bracket(team_ls[1:Nd2], depth+1)
        r_subbracket = rec_make_bracket(team_ls[(Nd2+1):end], depth+1)

        bracket = Dict(string("__",depth,"_a__") => l_subbracket,
                       string("__",depth,"_b__") => r_subbracket
                      )
    end

    return bracket
end


function make_empty_bracket(team_ls::Vector{String})

    N_teams = length(team_ls)
    @assert isinteger(log2(length(N_teams)))

    subbracket = rec_make_bracket(team_ls, 1)

    bracket = Dict("__0__" => subbracket)

    return bracket
end


function main(args)
    
    teams_csv = args[1]
    out_json = args[2]

    team_df = DataFrame(CSV.File(teams_csv))
    team_ls = convert(Vector{String}, team_df[:,:TEAMS])

    bracket = make_empty_bracket(team_ls)

    open(out_json, "w") do f
        JSON.print(f, bracket, 4)
    end

end

main(ARGS)


