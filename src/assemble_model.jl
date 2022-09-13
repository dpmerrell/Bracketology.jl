
"""
    Given three vectors describing the set of games: 
        (1) team A, (2) team B, and (3) date;

    return a vector of (team, date) pairs that covers 
    every appearance of every team in every game.
    
    The returned vector will be ordered s.t.:
        * the entries for a team appear contiguously; and
        * the entries for a team are sorted by date.
"""
function unique_team_dates(team_a_vec, team_b_vec, date_vec)

    # A dictionary mapping each team to the set of dates they played
    team_dates = Dict{String, Set}()

    for (team_a,date) in zip(team_a_vec,date_vec)
        if !haskey(team_dates, team_a)
            team_dates[team_a] = Set([date])
        else
            push!(team_dates[team_a], date)
        end
    end
    
    for (team_b,date) in zip(team_b_vec,date_vec)
        if !haskey(team_dates, team_b)
            team_dates[team_b] = Set([date])
        else
            push!(team_dates[team_b], date)
        end
    end

    # For each team, sort the dates into chronological order.
    team_dates = Dict{String, Vector}(k => sort(collect(v)) for (k,v) in team_dates)
    # Then flatten the dictionary into a vector of (team, date) pairs.
    team_dates = Tuple{String,String}[(k,v) for (k, v_ls) in team_dates for v in v_ls]

    return team_dates 
end


function assemble_model(team_a_vec, team_b_vec, date_vec;
                        K=3, noise_model="poisson", reg_weight=1.0)

    team_dates = unique_team_dates(team_a_vec, team_b_vec, date_vec) 
    regmat = assemble_regmat(team_dates)
    M = size(regmat,1)
 
    X_reg = MatrixRegularizer(regmat)
    Y_reg = MatrixRegularizer(regmat)
    row_trans_reg = ShiftRegularizer(regmat)
    col_trans_reg = ShiftRegularizer(regmat)

    matfac = SparseMatFacModel(M, M, K;
                               row_transform=ShiftLayer(M),
                               col_transform=ShiftLayer(M),
                               X_reg=X_reg, Y_reg=Y_reg, 
                               row_transform_reg=row_trans_reg, 
                               col_transform_reg=col_trans_reg,
                               noise_model=noise_model,
                               lambda_X=reg_weight,
                               lambda_Y=reg_weight,
                               lambda_row=reg_weight,
                               lambda_col=reg_weight)

    return CompetitionModel(matfac, team_dates)

end


