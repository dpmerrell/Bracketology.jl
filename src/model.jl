
export TeamModel, save_model, load_model

mutable struct TeamModel
    matfac::SparseMatFacModel
    team_vec::Vector{String}
    date_vec::Vector{String}
end


function TeamModel(matfac, team_dates::Vector{<:Tuple})
    return TeamModel(matfac, String[p[1] for p in team_dates],
                             String[p[2] for p in team_dates])
end


function save_model(hdf_filename, model)
    h5open(hdf_filename, "w") do f
        write(f, "team_vec", model.team_vec)
        write(f, "date_vec", model.date_vec)
        write(f, "matfac", model.matfac)
    end
end

function load_model(hdf_filename, model)
    h5open(hdf_filename, "r") do f
        matfac = model_from_hdf(f, "matfac")
        team_vec = f["team_vec"][:]
        date_vec = f["date_vec"][:]

        return TeamModel(matfac, team_vec, date_vec)
    end
end
