"""
Receives JSON files for (1) a predicted bracket and (2) a true bracket.
Computes the predicted bracket's ESPN score and prints a summary to stdout.
"""

using Bracketology, JSON


function score_bracket(pred_bracket, true_bracket)

    espn_score(pred_bracket, true_bracket)

end

function main(args)

    pred_json = args[1]
    true_json = args[2]

    pred_bracket = JSON.parsefile(pred_json)
    true_bracket = JSON.parsefile(true_json)

    score, pred_sets, true_sets = score_bracket(pred_bracket, true_bracket)

    for (i, (p,t)) in enumerate(zip(reverse(pred_sets), reverse(true_sets)))
        pd = setdiff(p,t)
        td = setdiff(t,p)
        p_str = join(pd, ", ")
        t_str = join(td, ", ")
        println(string("Round ", i, ": predicted ", p_str, " instead of ", t_str))
    end
    println(score)
end

main(ARGS)


