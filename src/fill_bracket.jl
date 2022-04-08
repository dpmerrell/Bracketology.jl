

export fill_bracket


function rec_fill_bracket(model, team_list)

    N_teams = length(team_list)

    if N_teams == 1
        return (team_list[1], 1.0)
    else

        rec_N = div(N_teams,2)
        subbracket_A = rec_fill_bracket(model, team_list[1:rec_N])
        subbracket_B = rec_fill_bracket(model, team_list[(rec_N+1):end])
        
        team_A = subbracket_A[1]
        team_B = subbracket_B[1]
        
        pA = win_prob(model, team_A, team_B)
        if pA > 0.5
            winner = team_A
            pw = pA
        else
            winner = team_B
            pw = 1 - pA
        end

        return (winner, pw, subbracket_A, subbracket_B)
    end
end


function rec_fill_arr!(arr, bracket)

    M,N = size(arr)

    arr[1,N] = string(bracket[1], " (", round(bracket[2];digits=2), ")")
    
    if N > 1
        rec_fill_arr!(view(arr,1:div(M,2),1:(N-1)), bracket[3])
        rec_fill_arr!(view(arr,(div(M,2)+1):M, 1:(N-1)), bracket[4])
    end
end


function fill_bracket(model, team_list)

    @assert isinteger(log2(length(team_list))) "Length of team list must be a power of 2"

    bracket = rec_fill_bracket(model, team_list)
    println(bracket)

    N_teams = length(team_list)
    bracket_arr = fill("", N_teams, Int(log2(N_teams) + 1))

    rec_fill_arr!(view(bracket_arr,:,:), bracket)

    df = DataFrame(bracket_arr, :auto)

    return df
end


