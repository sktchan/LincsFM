# frozen probe of an encoder: mean-pooled h0 / h1 / h2 and h2 token sd -> shrinkage LDA on lvl2 (drug, drug after cell-line centring, cell line)
# ckpt = <pretrain run>/<init|best|final|final_student>, "random[:seed]" (default arch and init) or "input" (input = etf z / rtf rank / raw reference)
# sets: lvl2 train subsample (MersenneTwister(1)), val subsample (MersenneTwister(2)) for alpha, full lvl2 test; columns in that order

const PROBE_COLS = ["drug_h2", "drug_clc_h2", "drug_meanstd", "cell_h2", "drug_all", "drug_clc_all", "cell_all", "drug_h0", "drug_h1"]

# encoder to probe and its input name; random[:seed] = default arch, Random.seed!(seed), default init (as the old random:seed)
function probe_encoder(ck::AbstractString, input_name::AbstractString, G::Int; random::Bool = false, init_seed::Int = 0)
    if startswith(ck, "random")
        seed = occursin(':', ck) ? parse(Int, split(ck, ':')[2]) : 0
        Random.seed!(seed)
        enc = Encoder(parse_input(input_name); n_genes=G, embed_dim=DEFAULTS["embed_dim"], hidden_dim=DEFAULTS["hidden_dim"],
                      n_heads=DEFAULTS["n_heads"], n_layers=DEFAULTS["n_layers"], drop_prob=0.1)
        return enc, input_name, DEFAULTS
    end
    enc, pcfg, _ = load_encoder(ck; random, init_seed)
    return enc, pcfg["input"], pcfg
end

# pooled token states of every layer and the last layer's token sd (D x n each)
function pooled_states(enc::Encoder, Xin::AbstractMatrix; bs::Int = 50)
    m = eval_device(enc); D = embed_dim(enc); n = size(Xin, 2); L = length(enc.layers.layers)
    pools = [zeros(Float32, D, n) for _ in 0:L]; sdL = zeros(Float32, D, n)
    for s in 1:bs:n
        b = s:min(s + bs - 1, n)
        hs = token_states(m, dev(Xin[:, b]))
        for l in eachindex(hs); pools[l][:, b] .= Array(meanpool(hs[l])); end
        sdL[:, b] .= Array(dropdims(std(hs[end], dims=2, corrected=false), dims=2))
    end
    return pools, sdL
end

function run_probe(cfg::Dict; data::Union{Nothing,LincsData} = nothing)
    t_start = time()
    name = string(cfg["name"]); ck = String(rstrip(string(cfg["ckpt"]), '/'))
    ntr = get(cfg, "ntr", 60000); nva = get(cfg, "nva", 10000); bs = get(cfg, "probe_bs", 50); notes = string(get(cfg, "notes", ""))
    is_input = ck == "input"
    d = isnothing(data) ? load_data(cfg["data_path"]) : data
    G = n_genes(d)

    # lvl2 split exactly as finetune; cell line = lvl1 label of the same samples
    idx, y, names = labels(d, "lvl2")
    sp = finetune_split(d, idx)
    @printf("lvl2 split: train %d val %d test %d (%d drugs)\n", length(sp.train), length(sp.val), length(sp.test), length(names))
    str = sort(randperm(MersenneTwister(1), length(sp.train))[1:min(ntr, length(sp.train))])
    sva = sort(randperm(MersenneTwister(2), length(sp.val))[1:min(nva, length(sp.val))])
    pos = vcat(sp.train[str], sp.val[sva], sp.test); cols = idx[pos]
    ydr = y[pos]; Kd = length(names)
    cls = d.meta.cell_iname[cols]; CL = sort(unique(cls)); cid = Dict(c => i for (i, c) in enumerate(CL)); ycl = [cid[c] for c in cls]; Kc = length(CL)
    n1, n2, n3 = length(str), length(sva), length(sp.test)
    TR, VA, TE = 1:n1, n1+1:n1+n2, n1+n2+1:n1+n2+n3
    @printf("probe sizes: train %d val %d test %d | drugs %d cell lines %d\n", n1, n2, n3, Kd, Kc)

    gpu = "cpu"; t_fwd = 0.0
    if is_input
        input = string(cfg["input"])
        F = if input == "etf"
            Xt = d.X[:, idx[sp.train]]
            zmu = vec(mean(Xt, dims=2)); zsd = vec(std(Xt, dims=2)); zsd[zsd .< 1f-6] .= 1f0; Xt = nothing
            (d.X[:, cols] .- zmu) ./ zsd
        elseif input == "rtf"
            rank_features(gene_ranks(d.X[:, cols], d.medians), d.X[:, cols])
        elseif input == "raw"
            d.X[:, cols]
        else
            error("input reference must be etf (z), rtf (rank) or raw, got $input")
        end
        P = pca_project(F, TR, 256)
        pools = [P, P, P]; sdL = nothing
        println("input reference ($input): $(size(F, 1)) dims; h2 columns = PCA256 fit on probe train, *_all columns = all dims")
    else
        enc, input, pcfg = probe_encoder(ck, string(get(cfg, "input", "")), G; random=get(cfg, "random", false), init_seed=get(cfg, "init_seed", 0))
        Xin = model_input(parse_input(input), d.X[:, cols], d, pcfg)
        t0 = time()
        pools, sdL = pooled_states(enc, Xin; bs)
        t_fwd = time() - t0; gpu = gpu_name()
        @printf("forward %.0f s (%.2f ms/sample) on %s\n", t_fwd, 1000 * t_fwd / size(Xin, 2), gpu); flush(stdout)
        Xin = nothing
        em = joinpath(cfg["results_dir"], "probe_emb"); mkpath(em)
        jldsave(joinpath(em, "$name.jld2"); pools, std_last=sdL, ydrug=ydr, ycell=ycl, cell_lines=CL, sample_idx=cols, ntr=n1, nva=n2, nte=n3, ckpt=ck, input)
    end
    d = nothing; GC.gc()

    # readouts
    t1 = time()
    rd(F, yy, K) = lda_readout(F, yy, K, TR, VA, TE)
    function clc(F)
        Z = lda_zprep(F, TR); Mc, _ = lda_group_means(Z[:, TR], ycl[TR], Kc); R = Z .- Mc[:, ycl]
        s = std(R[:, TR], dims=2); s[s .< 1e-12] .= 1; R ./= s
        return shrink_lda(R[:, TR], ydr[TR], R[:, VA], ydr[VA], R[:, TE], ydr[TE], Kd)
    end
    NA = (NaN, NaN, NaN, Int[])
    hL = pools[end]
    r = if is_input
        Dict("drug_h2" => rd(hL, ydr, Kd), "drug_clc_h2" => clc(hL), "drug_meanstd" => NA, "cell_h2" => rd(hL, ycl, Kc),
             "drug_all" => rd(F, ydr, Kd), "drug_clc_all" => clc(F), "cell_all" => rd(F, ycl, Kc), "drug_h0" => NA, "drug_h1" => NA)
    else
        ALL = vcat(pools..., sdL)
        Dict("drug_h2" => rd(hL, ydr, Kd), "drug_clc_h2" => clc(hL), "drug_meanstd" => rd(vcat(hL, sdL), ydr, Kd),
             "cell_h2" => rd(hL, ycl, Kc), "drug_all" => rd(ALL, ydr, Kd), "drug_clc_all" => clc(ALL), "cell_all" => rd(ALL, ycl, Kc),
             "drug_h0" => rd(pools[1], ydr, Kd), "drug_h1" => rd(pools[2], ydr, Kd))
    end
    t_lda = time() - t1
    println(name, " | ", join([@sprintf("%s %.3f (val %.3f)", c, r[c][2], r[c][1]) for c in PROBE_COLS], " | "))

    ks = vcat([:name, :input, :ckpt], Symbol.(PROBE_COLS .* "_test"), [:ntr, :nva, :nte, :val_accs, :gpu, :fwd_s, :lda_s, :total_s, :notes])
    vs = vcat(Any[name, is_input ? "input_" * input : input, ck], Any[r[c][2] for c in PROBE_COLS], Any[n1, n2, n3],
              Any[join([@sprintf("%.4f", r[c][1]) for c in PROBE_COLS], "/"), replace(gpu, ' ' => '_'),
                  round(Int, t_fwd), round(Int, t_lda), round(Int, time() - t_start), isempty(notes) ? "-" : notes])
    row = NamedTuple{Tuple(ks)}(Tuple(vs))
    append_row(joinpath(cfg["results_dir"], "probe.tsv"), row)
    println("appended to ", joinpath(cfg["results_dir"], "probe.tsv"))
    return row
end
