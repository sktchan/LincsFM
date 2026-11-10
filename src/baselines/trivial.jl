# trivial label baselines on the test set: chance, majority class, and majority train class per plate / cell / (plate, dose)

# most frequent value, ties -> smallest class id
function _majority(v::AbstractVector{<:Integer})
    cnt = countmap(v)
    return minimum(c for (c, k) in cnt if k == maximum(values(cnt)))
end

# per-key majority train class, unseen keys -> global train majority; returns accuracy and fraction of test keys seen in train
function key_majority(ytr, ktr, yte, kte)
    g = _majority(ytr)
    groups = Dict{eltype(ktr),Vector{Int}}()
    for (y, k) in zip(ytr, ktr); push!(get!(groups, k, Int[]), y); end
    best = Dict(k => _majority(v) for (k, v) in groups)
    pred = [get(best, k, g) for k in kte]
    return mean(pred .== yte), mean(haskey(best, k) for k in kte)
end

function trivial_baselines(level::AbstractString = "lvl2"; data::Union{Nothing,LincsData} = nothing,
                           results_dir::AbstractString = joinpath(PKG_DIR, "results"))
    d = isnothing(data) ? load_data() : data
    idx, y, names = labels(d, level)
    sp = finetune_split(d, idx)
    m = d.meta
    plate = m.det_plate[idx]; cell = m.cell_iname[idx]; pd = collect(zip(m.det_plate[idx], m.pert_dose[idx]))
    ytr, yte = y[sp.train], y[sp.test]
    C = length(names); n = length(yte)
    rows = NamedTuple[]
    add(b, acc, seen) = push!(rows, (level = level, baseline = b, n_classes = C, n_train = length(ytr), n_test = n, acc = acc, test_key_in_train = seen))
    add("chance", 1 / C, NaN)
    add("majority", mean(yte .== _majority(ytr)), NaN)
    for (b, k) in (("plate_majority", plate), ("cell_majority", cell), ("plate_dose_majority", pd))
        add(b, key_majority(ytr, k[sp.train], yte, k[sp.test])...)
    end
    path = joinpath(results_dir, "trivial.tsv")
    foreach(r -> append_row(path, r), rows)
    foreach(r -> @printf("%s %-20s acc %.4f (test keys in train %.3f)\n", r.level, r.baseline, r.acc, r.test_key_in_train), rows)
    return rows
end
