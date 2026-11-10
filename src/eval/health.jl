# health checks of a pretrained encoder vs its own init/ checkpoint: pooled / token sd, variance shares of h_L, weight change, pretext vs trivial
# subset: nh samples of the pretrain val split (bad plates dropped), MersenneTwister(3); one row per call in <results_dir>/health.tsv

# flat Float64 vector of all trainable arrays
_flat(m) = (l = [vec(Float64.(Array(x))) for x in Flux.trainables(m)]; isempty(l) ? Float64[] : reduce(vcat, l))
_relchg(w, w0) = norm(w .- w0) / norm(w0)
# rms change of the seed-independent init entries (equal in two inits of different seeds: LayerNorm, zero biases)
function _detchg(b, b0, br)
    w, w0, wr = _flat(b), _flat(b0), _flat(br); k = w0 .== wr
    return any(k) ? norm(w[k] .- w0[k]) / sqrt(count(k)) : NaN
end
_blocks(e::Encoder) = vcat([("value", e.value), ("gene", e.gene)], [("L$i", l) for (i, l) in enumerate(e.layers.layers)])

# last-layer token-state stats streamed over batches: total / sample / gene / token x sample sums of squares (balanced two-way split),
# plus the share of rank-position means (ranks by x / median, the old RTF position axis)
function state_stats(enc::Encoder, Xin::AbstractMatrix, R::AbstractMatrix{<:Integer}; bs::Int = 50)
    m = eval_device(enc); D = embed_dim(enc); T, N = size(Xin)
    A = zeros(Float64, D, N); Pg = zeros(Float64, D, T); Pr = zeros(Float64, D, T); s1 = zeros(Float64, D); s2 = zeros(Float64, D)
    for s in 1:bs:N
        b = s:min(s + bs - 1, N)
        h = Float64.(Array(token_states(m, dev(Xin[:, b]))[end]))
        A[:, b] .= dropdims(mean(h, dims=2), dims=2)
        Pg .+= dropdims(sum(h, dims=3), dims=3)
        for (jj, j) in enumerate(b); Pr[:, R[:, j]] .+= h[:, :, jj]; end
        s1 .+= vec(sum(h, dims=(2, 3))); s2 .+= vec(sum(h .^ 2, dims=(2, 3)))
    end
    mu = s1 ./ (N * T)
    sst = s2 .- N * T .* mu .^ 2; ssa = T .* vec(sum((A .- mu) .^ 2, dims=2))
    ssb = N .* vec(sum((Pg ./ N .- mu) .^ 2, dims=2)); ssc = N .* vec(sum((Pr ./ N .- mu) .^ 2, dims=2)); ssr = sst .- ssa .- ssb
    tot = sum(sst)
    return (pool_sd = median(vec(std(A, dims=2))), tok_sd = median(sqrt.(max.(sst, 0) ./ (N * T - 1))), frac_sample = sum(ssa) / tot,
            frac_gene = sum(ssb) / tot, frac_rank = sum(ssc) / tot, frac_tokxsmp = sum(ssr) / tot, var_tot = tot / (D * T * N),
            var_tokxsmp = sum(ssr) / (D * T * N))
end

# pretext value and trivial of a checkpoint from the run's log.tsv: init = step 0, best = its saved step, else the last row
function pretext_from_log(ck::AbstractString, ex::Dict)
    lp = joinpath(dirname(ck), "log.tsv")
    isfile(lp) || return NaN, NaN
    rows = [split(l, '\t') for l in readlines(lp)]; h = rows[1]; rows = rows[2:end]
    isempty(rows) && return NaN, NaN
    col(k) = findfirst(==(k), h)
    st = basename(ck) == "init" ? 0 : (basename(ck) == "best" ? get(ex, "step", -1) : -1)
    r = st >= 0 ? something(findlast(x -> parse(Int, x[col("step")]) == st, rows), length(rows)) : length(rows)
    return parse(Float64, rows[r][col("val")]), parse(Float64, rows[r][col("trivial")])
end

function run_health(cfg::Dict; data::Union{Nothing,LincsData} = nothing)
    t_start = time()
    ck = String(rstrip(string(cfg["ckpt"]), '/')); name = string(get(cfg, "name", join(splitpath(ck)[end-2:end], "/")))
    nh = get(cfg, "nh", 5000); bs = get(cfg, "probe_bs", 50)
    d = isnothing(data) ? load_data(cfg["data_path"]) : data
    vs = sort(shuffle(MersenneTwister(3), pretrain_split(d).val)[1:min(nh, length(pretrain_split(d).val))])
    Xraw = d.X[:, vs]
    println("health subset: $(length(vs)) pretrain-val samples")

    # checkpoint, its run's init/ and an unrelated init (seed + 1) of the same arch and scheme
    is_ref = basename(ck) == "init"
    initdir = joinpath(dirname(ck), "init")
    m, pcfg, ex = load_encoder(ck)
    m0, _, _ = load_encoder(initdir)
    seed = Int(pcfg["seed"])
    mr, _, _ = load_encoder(initdir; random=true, init_seed=seed + 1)
    mre, _, _ = load_encoder(initdir; random=true, init_seed=seed)
    rebuild = maximum(abs.(_flat(mre) .- _flat(m0)))
    @printf("init rebuild from seed %d: max |diff| vs saved init %.3g\n", seed, rebuild)
    input = parse_input(pcfg["input"])
    Xin = model_input(input, Xraw, d, pcfg); R = gene_ranks(Xraw, d.medians)
    d = nothing; Xraw = nothing; GC.gc()

    B, B0, BR = _blocks(m), _blocks(m0), _blocks(mr)
    chg = Dict(n => _relchg(_flat(b), _flat(b0)) for ((n, b), (_, b0)) in zip(B, B0))
    ref = Dict(n => _relchg(_flat(b), _flat(b0)) for ((n, b), (_, b0)) in zip(BR, B0))
    det = Dict(n => _detchg(b, b0, br) for ((n, b), (_, b0), (_, br)) in zip(B, B0, BR))
    Ls = ["L$i" for i in 1:length(m.layers.layers)]; Llast = Ls[end]

    t0 = time()
    S = state_stats(m, Xin, R; bs); GC.gc(); CUDA.functional() && CUDA.reclaim()
    S0 = state_stats(m0, Xin, R; bs)
    t_fwd = time() - t0

    # pretext vs trivial (log.tsv, or pretext= / trivial= overrides)
    pv, tv = pretext_from_log(ck, ex)
    haskey(cfg, "pretext") && (pv = Float64(cfg["pretext"])); haskey(cfg, "trivial") && (tv = Float64(cfg["trivial"]))
    ratio = (isfinite(pv) && isfinite(tv)) ? pv / tv : NaN
    learned = isnan(ratio) ? "na" : ratio < 0.97 ? "yes" : "no"

    # drug probe of the same name, if present
    pdrug = NaN; pt = joinpath(cfg["results_dir"], "probe.tsv")
    if isfile(pt)
        L = [split(l, '\t') for l in readlines(pt)]; c = findfirst(==("drug_h2_test"), L[1])
        for f in L[2:end]; f[1] == name && (pdrug = parse(Float64, f[c])); end
    end

    # verdict, rules checked in order
    x_pool, x_tok, x_txs, x_tot = S.pool_sd / S0.pool_sd, S.tok_sd / S0.tok_sd, S.var_tokxsmp / S0.var_tokxsmp, S.var_tot / S0.var_tot
    mov = maximum(chg[l] for l in Ls); fpos = max(S.frac_gene, S.frac_rank)
    verdict, vnote = if is_ref
        "reference", "init checkpoint (seed $seed)"
    elseif mov < 0.1
        "near_init", @sprintf("transformer weights moved %.3g < 0.1 of init", mov)
    elseif x_txs < 0.1 && fpos > 0.98
        "token_uniform_collapse", @sprintf("h_L position-constant (gene share %.3f rank share %.3f, tokxsmp var x%.3g init)", S.frac_gene, S.frac_rank, x_txs)
    elseif x_pool < 0.1
        "scale_collapse", @sprintf("pooled sd x%.3g init", x_pool)
    elseif x_pool > 1000
        "scale_blowup", @sprintf("pooled sd x%.3g init", x_pool)
    elseif learned == "no"
        "pretext_not_learned", @sprintf("pretext %.4g / trivial %.4g = %.3f >= 0.97", pv, tv, ratio)
    else
        "healthy", learned == "na" ? "no trivial baseline" : @sprintf("pretext %.3f of trivial", ratio)
    end
    x_tot > 100 && (vnote *= @sprintf("; h_L variance x%.3g init", x_tot))

    row = (name = name, input = pcfg["input"], objective = pcfg["objective"], verdict = verdict, pool_drug_test = pdrug,
           pool_sd_med = S.pool_sd, pool_sd_x_init = x_pool, tok_sd_med = S.tok_sd, tok_sd_x_init = x_tok,
           frac_sample = S.frac_sample, frac_gene = S.frac_gene, frac_rank = S.frac_rank, frac_tokxsmp = S.frac_tokxsmp,
           var_tokxsmp = S.var_tokxsmp, var_tokxsmp_x_init = x_txs, var_tot_x_init = x_tot,
           chg_value = chg["value"], chg_gene = chg["gene"], chg_L1 = chg["L1"], chg_L2 = chg[Llast], ref_L1 = ref["L1"], ref_L2 = ref[Llast],
           det_value = det["value"], det_L1 = det["L1"], det_L2 = det[Llast], init_seed = seed, step = get(ex, "step", -1),
           pretext = pv, trivial = tv, pretext_ratio = ratio, pretext_learned = learned, n_val = length(vs),
           verdict_note = vnote * @sprintf("; fwd %.0fs total %.0fs", t_fwd, time() - t_start), ckpt = ck)
    append_row(joinpath(cfg["results_dir"], "health.tsv"), row)
    println(join(["$k=$(v isa AbstractFloat ? @sprintf("%.4g", v) : v)" for (k, v) in pairs(row)], " | "))
    println("appended to ", joinpath(cfg["results_dir"], "health.tsv"))
    return row
end
