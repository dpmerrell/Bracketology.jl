

function unique_team_dates(team_a_vec, team_b_vec, date_vec)

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

    team_dates = Dict{String, Vector}(k => sort(collect(v)) for (k,v) in team_dates)
    team_dates = Tuple{String,String}[(k,v) for (k, v_ls) in team_dates for v in v_ls]

    return team_dates 
end


function assemble_regmat(team_a_vec, team_b_vec, date_vec;
                         epsilon=0.0, coeff=1.0)

    team_dates = unique_team_dates(team_a_vec, team_b_vec,
                                   date_vec)

    M = length(team_dates)

    diag = ones(M).*epsilon
    I = Int32[]
    J = Int32[]
    V = Float32[]

    prev_team, prev_date = team_dates[1]
    for idx=2:M
        team, date = team_dates[idx]
        
        if team == prev_team
            delta_t = Dates.days(Date(date) - Date(prev_date))/365.0
            weight = 1/delta_t
            diag[idx-1] += weight 
            diag[idx] += weight

            push!(I, idx-1)
            push!(J, idx)
            push!(V, -weight)
            
            push!(I, idx)
            push!(J, idx-1)
            push!(V, -weight)
        end

        prev_team = team
        prev_date = date
    end

    for (i, d) in enumerate(diag)
        push!(I, i)
        push!(J, i)
        push!(V, d)
    end

    V .*= Float32(coeff)

    regmat = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse(I,J,V))
    return regmat, team_dates
end


function assemble_model(team_a_vec, team_b_vec, date_vec;
                        K=3, loss="poisson", reg_weight=1.0)

    regmat, team_dates = assemble_regmat(team_a_vec, team_b_vec, date_vec;
                                         coeff=reg_weight)
    
    X_reg = fill(regmat, K)
    Y_reg = fill(regmat, K)

    matfac = SparseMatFacModel(X_reg, Y_reg, regmat, regmat;
                              loss=loss)

    return TeamModel(matfac, team_dates)

end


