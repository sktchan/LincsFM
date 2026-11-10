# drug-cluster paired bootstrap of test predictions: resample the test classes (compounds for lvl2) with replacement,
# B = 2000, MersenneTwister(2026), one resample shared by every arm; arm table (acc, 95% CI, balanced acc) and contrast table
# source = finetune run dir (predstrues.jld2, preds_final), <dir>#best / #final, or <file.jld2>#key (e.g. ldarefit_final.jld2#preds_pool)
# contrast = [A, B] (A - B) or [A, B, C, D] ((A - B) - (C - D)); pt = true: |delta| < 0.07 is never "different" (1 pretrain seed)

const BOOT_DELTA = 0.03   # equivalence margin
const BOOT_PT_MIN = 0.07

function load_preds(src::AbstractString)
    path, key = occursin('#', src) ? String.(split(src, '#'; limit=2)) : (String(src), "preds_final")
    isdir(path) && (path = joinpath(path, "predstrues.jld2"))
    key in ("final", "best") && (key = "preds_" * key)
    s = load(path)
    haskey(s, key) || error("$path has no key $key (keys: $(join(keys(s), ", ")))")
    tk = haskey(s, "trues") ? "trues" : "all_trues"
    return Int.(s[key]), Int.(s[tk]), get(s, "test_sample_idx", nothing), path * "#" * key
end

function run_bootstrap(out::AbstractString, arms::Vector{Pair{String,String}}; contrasts = Tuple{Vector{String},Bool}[],
                       seed_gap::Real = 0.0, all_pairs::Bool = false, B::Int = 2000)
    isempty(arms) && error("no arms")
    L = [(n, load_preds(s)...) for (n, s) in arms]
    y = L[1][3]; tsi = nothing
    for (n, p, t, ix, path) in L
        length(p) == length(y) || error("$n: $(length(p)) preds vs $(length(y)) test samples")
        t == y || error("$n: trues differ from $(L[1][1]) (different test set or order): $path")
        if !isnothing(ix)
            isnothing(tsi) ? (tsi = ix) : (ix == tsi || error("$n: test_sample_idx differs: $path"))
        end
    end
    println("$(length(L)) arms, $(length(y)) test samples, test_sample_idx checked on $(count(x -> !isnothing(x[4]), L)) arms")

    # clusters = true classes; per-arm correct counts per class
    C = sort(unique(y)); cid = Dict(c => j for (j, c) in enumerate(C)); nc = length(C)
    n = zeros(Int, nc); for t in y; n[cid[t]] += 1; end
    corr = Dict(nm => (c = zeros(Int, nc); for (pp, t) in zip(p, y); c[cid[t]] += (pp == t); end; c) for (nm, p, _, _, _) in L)
    @printf("%d clusters, %d-%d samples each\n", nc, minimum(n), maximum(n))

    # shared resample: multiplicity of each cluster in each draw (depends only on nc, B and the seed)
    rng = MersenneTwister(2026)
    W = zeros(Int32, nc, B)
    for b in 1:B, j in rand(rng, 1:nc, nc); W[j, b] += 1; end
    den = vec(n' * W)
    bacc(c) = vec(c' * W) ./ den
    pct(v, q) = quantile(v, q)
    balacc(p) = mean([mean(p[y .== c] .== c) for c in C])

    mkpath(dirname(abspath(out)))
    arm_rows = NamedTuple[]
    for (nm, p, _, _, path) in L
        v = bacc(corr[nm]); a = mean(p .== y)
        @printf("%-28s acc %.4f [%.4f, %.4f] bal %.4f\n", nm, a, pct(v, .025), pct(v, .975), balacc(p))
        push!(arm_rows, (name = nm, acc = a, lo95 = pct(v, .025), hi95 = pct(v, .975), boot_sd = std(v), bal_acc = balacc(p),
                         n_test = length(y), n_cpd = nc, B = B, source = path))
    end
    write_curve(out * "_arms.tsv", arm_rows)

    cs = copy(contrasts)
    all_pairs && for a in 1:length(L), b in a+1:length(L); push!(cs, ([L[a][1], L[b][1]], false)); end
    con_rows = NamedTuple[]
    for (nm, pt) in cs
        all(x -> haskey(corr, x), nm) || error("contrast $(join(nm, ':')) names an unknown arm")
        ws = length(nm) == 2 ? [1, -1] : [1, -1, -1, 1]
        v = sum(w .* bacc(corr[x]) for (w, x) in zip(ws, nm)); dlt = sum(w * sum(corr[x]) for (w, x) in zip(ws, nm)) / length(y)
        lo, hi = pct(v, .025), pct(v, .975)
        dec = (lo > 0 || hi < 0) && abs(dlt) > seed_gap ? "different" : (lo >= -BOOT_DELTA && hi <= BOOT_DELTA ? "equivalent" : "inconclusive")
        (pt && dec == "different" && abs(dlt) < BOOT_PT_MIN) && (dec = "inconclusive")
        lab = length(nm) == 2 ? "$(nm[1]) - $(nm[2])" : "($(nm[1]) - $(nm[2])) - ($(nm[3]) - $(nm[4]))"
        @printf("%s: %+.4f [%+.4f, %+.4f] %s\n", lab, dlt, lo, hi, dec)
        cd_ = length(nm) == 2 ? ["-", "-"] : nm[3:4]
        push!(con_rows, (A = nm[1], B = nm[2], C = cd_[1], D = cd_[2], delta = dlt, lo95 = lo, hi95 = hi, p_le0 = mean(v .<= 0),
                         decision = dec, pt_rule = pt ? "pt" : "-", seed_gap = Float64(seed_gap)))
    end
    isempty(con_rows) || write_curve(out * "_contrasts.tsv", con_rows)
    println("wrote $(out)_arms.tsv" * (isempty(con_rows) ? "" : " and $(out)_contrasts.tsv"))
    return arm_rows, con_rows
end
