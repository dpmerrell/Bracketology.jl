

export espn_score


function rec_build_depth_sets!(depth_sets, true_bracket_d, depth)

    if true_bracket_d == nothing
        return
    else
        if length(depth_sets) < depth
            push!(depth_sets, Set{String}())
        end

        for k in keys(true_bracket_d)
            push!(depth_sets[depth], k)
            rec_build_depth_sets!(depth_sets, true_bracket_d[k], depth+1)
        end
    end
end


function espn_depth_score(depth)
    if depth <= 5
        return 320 / (2^depth)
    else
        return 0
    end
end


function espn_score(bracket_d, true_bracket_d)

    true_depth_sets = Set{String}[]
    rec_build_depth_sets!(true_depth_sets, true_bracket_d, 1) 

    pred_depth_sets = Set{String}[]
    rec_build_depth_sets!(pred_depth_sets, bracket_d, 1)

    score = 0
    for (k, (p, t)) in enumerate(zip(pred_depth_sets, true_depth_sets))
        score += length(intersect(p,t))*espn_depth_score(k-1)
    end

    return score, pred_depth_sets, true_depth_sets
end




