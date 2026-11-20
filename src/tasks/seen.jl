# seen-target / unseen-drug task: reference treated profile + target id + plate id -> target treated profile (no drug embeddings)

struct SeenTask
    reference::String
    genes::Vector{String}; gene_ids::Vector{String}
    drugs::Vector{String}
    R::Matrix{Float32}             # genes x drugs: reference treated profile (condition mean, raw Level 3)
    targets::Vector{String}; plates::Vector{String}
    Y::Matrix{Float32}             # genes x wells: target treated wells, ordered by condition
    well_cond::Vector{Int}; well_plate::Vector{Int}; well_rna::Vector{Int}; well_sample::Vector{String}
    cond_drug::Vector{Int}; cond_target::Vector{Int}; cond_ptr::Vector{Int}
    seeds::Vector{String}
    split::Matrix{Int8}            # drugs x seeds: 1 train, 2 validation, 3 test
    py::NamedTuple                 # Python reference scores from the export (tests)
end

default_seen_path(ref::AbstractString = "NPC") = joinpath(PKG_DIR, "data", "tasks", "seen_$(ref)_10uM.h5")

function load_seen_task(path::AbstractString = default_seen_path())
    f = jldopen(path, "r")
    g(k) = f[k]
    ptr = Int.(g("cond_ptr")); nc = length(ptr) - 1
    well_cond = Vector{Int}(undef, ptr[end] - 1)
    for c in 1:nc
        well_cond[ptr[c]:ptr[c+1]-1] .= c
    end
    py = (models = g("py_models"), median_r = g("py_median_r"), median_r_drug = g("py_median_r_drug"),
          mse = g("py_mse"), ceiling_drug = g("py_ceiling_drug"), alpha = g("py_alpha"))
    t = SeenTask(splitext(basename(path))[1], g("gene_symbol"), g("gene_id"), g("drugs"), g("ref_profile"), g("targets"),
                 g("plates"), g("expr"), well_cond, Int.(g("well_plate")), Int.(g("well_rna")), g("well_sample"),
                 Int.(g("cond_drug")), Int.(g("cond_target")), ptr, g("seeds"), g("split"), py)
    close(f)
    return t
end

n_conditions(t::SeenTask) = length(t.cond_drug)
cond_wells(t::SeenTask, c::Integer) = t.cond_ptr[c]:t.cond_ptr[c+1]-1
seed_index(t::SeenTask, seed) = seed isa Integer && seed <= length(t.seeds) ? seed : findfirst(==(string(seed)), t.seeds)

# training wells and validation / test conditions of one drug split (drug groups never cross)
function seen_split(t::SeenTask, seed)
    s = seed_index(t, seed); lab = t.split[:, s]
    cl = lab[t.cond_drug]
    return (train = findall(==(Int8(1)), cl[t.well_cond]), val = findall(==(Int8(2)), cl), test = findall(==(Int8(3)), cl))
end

# model inputs of wells: reference profile (genes x n), target and plate ids (1-based)
well_inputs(t::SeenTask, wells) = (x = t.R[:, t.cond_drug[t.well_cond[wells]]], target = t.cond_target[t.well_cond[wells]],
                                   plate = t.well_plate[wells])

# random training batch for neural models: inputs of B training wells and their treated profiles (genes x B)
function seen_batch(t::SeenTask, train::AbstractVector{<:Integer}, B::Integer, rng::AbstractRNG = Random.default_rng())
    w = train[rand(rng, 1:length(train), B)]
    return (well_inputs(t, w)..., y = t.Y[:, w])
end

# condition means of the observed wells (genes x n)
seen_truth(t::SeenTask, conds) = reduce(hcat, [vec(mean(view(t.Y, :, cond_wells(t, c)); dims=2)) for c in conds])

# condition predictions from any per-well predictor f(x, target, plate) -> genes x n, averaged over each condition's wells
function predict_conditions(f, t::SeenTask, conds; chunk::Integer = 4096)
    wells = reduce(vcat, [collect(cond_wells(t, c)) for c in conds])
    P = Matrix{Float32}(undef, size(t.Y, 1), length(wells))
    for s in 1:chunk:length(wells)
        w = wells[s:min(end, s + chunk - 1)]; inp = well_inputs(t, w)
        P[:, s:s+length(w)-1] = f(inp.x, inp.target, inp.plate)
    end
    out = similar(P, size(P, 1), length(conds)); k = 0
    for (j, c) in enumerate(conds)
        n = length(cond_wells(t, c)); out[:, j] = vec(mean(view(P, :, k+1:k+n); dims=2)); k += n
    end
    return out
end

# Pearson per gene (row) across samples (columns); NaN where either side is constant
function per_gene_r(Y::AbstractMatrix, P::AbstractMatrix)
    A = Float64.(Y) .- mean(Y; dims=2); B = Float64.(P) .- mean(P; dims=2)
    return vec(sum(A .* B; dims=2) ./ sqrt.(sum(abs2, A; dims=2) .* sum(abs2, B; dims=2)))
end

nanmedian(v) = (w = filter(!isnan, v); isempty(w) ? NaN : median(w))

# per target (>= min_test conditions): per-gene r on raw profiles and on the drug part (minus the ID-only baseline), MSE
function score_seen(t::SeenTask, conds, pred::AbstractMatrix; base::Union{Nothing,AbstractMatrix} = nothing, min_test::Integer = 20)
    Y = seen_truth(t, conds); tg = t.cond_target[conds]
    out = Dict{String,Any}[]
    for k in eachindex(t.targets)
        loc = findall(==(k), tg)
        length(loc) < min_test && continue
        r = per_gene_r(Y[:, loc], pred[:, loc])
        rd = isnothing(base) ? fill(NaN, length(r)) : per_gene_r(Y[:, loc] .- base[:, loc], pred[:, loc] .- base[:, loc])
        push!(out, Dict{String,Any}("target" => t.targets[k], "n" => length(loc), "r" => r, "r_drug" => rd,
                                    "median_r" => nanmedian(r), "median_r_drug" => nanmedian(rd),
                                    "mse" => mean(abs2, Float64.(Y[:, loc]) .- pred[:, loc])))
    end
    return out
end
score_summary(s, key = "median_r") = isempty(s) ? NaN : mean(x[key] for x in s)

# ---- linear baselines: [R, target one-hot, plate one-hot]; Ridge penalty on R only, IDs unpenalized ----

struct SeenLinear
    xbar::Vector{Float64}; ybar::Vector{Float64}
    V::Matrix{Float64}; values::Vector{Float64}; E::Matrix{Float64}
    Cp::Matrix{Float64}; B::Matrix{Float64}; HI::Matrix{Float64}
    nt::Int
end

# exact sufficient statistics of the training wells from count matrices (no dense feature matrix)
function SeenLinear(t::SeenTask, train)
    G, D = size(t.R); T = length(t.targets); P = length(t.plates)
    d = t.cond_drug[t.well_cond[train]]; tg = t.cond_target[t.well_cond[train]]; p = t.well_plate[train]
    n = length(train); R = Float64.(t.R)
    nd = zeros(D); Cdt = zeros(D, T); Cdp = zeros(D, P); Ctp = zeros(T, P); nt = zeros(T); np_ = zeros(P)
    Sd = zeros(G, D); St = zeros(G, T); Sp = zeros(G, P)
    for (i, w) in enumerate(train)
        nd[d[i]] += 1; Cdt[d[i], tg[i]] += 1; Cdp[d[i], p[i]] += 1; Ctp[tg[i], p[i]] += 1; nt[tg[i]] += 1; np_[p[i]] += 1
        y = view(t.Y, :, w); Sd[:, d[i]] .+= y; St[:, tg[i]] .+= y; Sp[:, p[i]] .+= y
    end
    XX = [R * (nd .* R') R * Cdt R * Cdp; (R * Cdt)' Diagonal(nt) Ctp; (R * Cdp)' Ctp' Diagonal(np_)]
    XY = [R * Sd'; St'; Sp']
    xbar = vcat(R * nd, nt, np_) ./ n; ybar = vec(sum(Sd; dims=2)) ./ n
    Gc = XX .- n .* (xbar * xbar'); H = XY .- n .* (xbar * ybar')
    A, Bm, C = Gc[1:G, 1:G], Gc[1:G, G+1:end], Gc[G+1:end, G+1:end]
    Cp = pinv(Symmetric((C + C') ./ 2))
    HI = H[G+1:end, :]
    As = A - Bm * Cp * Bm'
    ev = eigen(Symmetric((As + As') ./ 2))
    return SeenLinear(xbar, ybar, ev.vectors, max.(ev.values, 0.0), ev.vectors' * (H[1:G, :] - Bm * Cp * HI), Cp, Bm, HI, T)
end

# weights for alpha (nothing = IDs only, 0 = minimum-norm least squares)
function seen_weights(m::SeenLinear, alpha)
    G = size(m.V, 1)
    if alpha === nothing
        WR = zeros(G, size(m.E, 2))
    else
        inv = alpha == 0 ? (keep = m.values .> maximum(m.values) * length(m.values) * eps(Float64); [k ? 1 / v : 0.0 for (k, v) in zip(keep, m.values)]) :
                           1 ./ (m.values .+ alpha)
        WR = m.V * (inv .* m.E)
    end
    return WR, m.Cp * (m.HI - m.B' * WR)
end

# centred condition features: R of the drug; target one-hot; plate weights = share of the condition's wells on each plate
function seen_features(m::SeenLinear, t::SeenTask, conds)
    G = size(m.V, 1)
    XR = Float64.(t.R[:, t.cond_drug[conds]]) .- m.xbar[1:G]
    XI = zeros(length(m.xbar) - G, length(conds))
    for (j, c) in enumerate(conds)
        XI[t.cond_target[c], j] = 1
        ws = cond_wells(t, c)
        for w in ws
            XI[m.nt + t.well_plate[w], j] += 1 / length(ws)
        end
    end
    return XR, XI .- m.xbar[G+1:end]
end

function seen_predict(m::SeenLinear, t::SeenTask, conds, alpha; features = seen_features(m, t, conds))
    WR, WI = seen_weights(m, alpha)
    return WR' * features[1] .+ WI' * features[2] .+ m.ybar
end

const SEEN_ALPHAS = [10.0^k for k in -6:6]

# ids / ridge / linear for one seed; ridge alpha = best validation mean-over-targets median per-gene r (ties: smaller)
function seen_linear_baselines(t::SeenTask, seed; alphas = SEEN_ALPHAS, min_test::Integer = 20)
    sp = seen_split(t, seed)
    m = SeenLinear(t, sp.train)
    fv, ft = seen_features(m, t, sp.val), seen_features(m, t, sp.test)
    val = Dict(a => score_summary(score_seen(t, sp.val, seen_predict(m, t, sp.val, a; features=fv); min_test)) for a in alphas)
    alpha = first(sort(alphas; by = a -> (isnan(val[a]) ? Inf : -val[a], a)))
    pred = Dict(name => seen_predict(m, t, sp.test, a; features=ft) for (name, a) in (("ids", nothing), ("ridge", alpha), ("linear", 0)))
    return (alpha = alpha, val = val, test = sp.test, pred = pred, base = pred["ids"], model = m)
end

# drug-part replicate ceiling per target and gene: wells minus their plate's training-well mean, alternating RNA-plate halves
function seen_drug_ceiling(t::SeenTask, seed; min_test::Integer = 20)
    sp = seen_split(t, seed); G = size(t.Y, 1)
    S = zeros(G, length(t.plates)); cnt = zeros(Int, length(t.plates))
    for w in sp.train
        S[:, t.well_plate[w]] .+= view(t.Y, :, w); cnt[t.well_plate[w]] += 1
    end
    halves = Dict{Int,Vector{Tuple{Vector{Float64},Vector{Float64}}}}()
    for c in sp.test
        ws = [w for w in cond_wells(t, c) if cnt[t.well_plate[w]] > 0]
        rnas = sort(unique(t.well_rna[ws]))
        length(rnas) < 2 && continue
        side = Dict(r => (k - 1) % 2 for (k, r) in enumerate(rnas))
        res(w) = Float64.(t.Y[:, w]) .- S[:, t.well_plate[w]] ./ cnt[t.well_plate[w]]
        a = mean(res(w) for w in ws if side[t.well_rna[w]] == 0); b = mean(res(w) for w in ws if side[t.well_rna[w]] == 1)
        push!(get!(halves, t.cond_target[c], []), (a, b))
    end
    out = Dict{String,Vector{Float64}}()
    for (k, h) in halves
        length(h) < min_test && continue
        r = per_gene_r(reduce(hcat, first.(h)), reduce(hcat, last.(h)))
        out[t.targets[k]] = sqrt.(max.(2 .* r ./ (1 .+ r), 0.0))
    end
    return out
end
