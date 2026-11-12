# LDA refit on a finetuned classifier's pooled features and head hidden layer, compared with its own head

# rebuild a Classifier skeleton from its saved state (arch read from the arrays and nheads)
function classifier_from_state(st, input::Input, norm::AbstractString)
    se = st.encoder; l1 = se.layers.layers[1]
    D, G = size(se.gene.weight)
    enc = Encoder(input; n_genes=G, embed_dim=D, hidden_dim=size(l1.mlp.layers[1].weight, 1), n_heads=l1.mha.nheads,
                  n_layers=length(se.layers.layers), norm)
    m = Classifier(enc, size(st.head.layers[end].weight, 1); hidden_dim=size(st.head.layers[1].weight, 1))
    Flux.loadmodel!(m, (encoder = st.encoder, head = st.head))
    return m
end

function run_ldarefit(cfg::Dict; data::Union{Nothing,LincsData} = nothing)
    t_start = time()
    run = String(rstrip(string(cfg["run"]), '/')); name = string(get(cfg, "name", join(splitpath(run)[end-1:end], "/")))
    ckpts = split(string(get(cfg, "ckpts", "final,best")), ','); ntr_req = get(cfg, "ntr", 0); bs = get(cfg, "lda_bs", 100)
    M = load(joinpath(run, "model.jld2")); fcfg = M["config"]; PT = load(joinpath(run, "predstrues.jld2"))
    input = parse_input(fcfg["input"])
    d = isnothing(data) ? load_data(cfg["data_path"]) : data
    idx, y, names = labels(d, fcfg["level"]); sp = finetune_split(d, idx); K = length(names)
    Xtr, Xva, Xte = finetune_inputs(fcfg, d, idx, sp)
    ytr_all, yva, yte = y[sp.train], y[sp.val], y[sp.test]
    tsi = idx[sp.test]
    (PT["trues"] == yte && PT["test_sample_idx"] == tsi && PT["label_names"] == names) || error("predstrues.jld2 test set differs from the rebuilt split ($run)")
    d = nothing; GC.gc()
    @printf("%s: %s %s, %d classes, train %d val %d test %d\n", name, fcfg["input"], fcfg["level"], K, length(ytr_all), length(yva), length(yte))

    # train columns: all, or a class-stratified subsample (MersenneTwister(7))
    trc = if ntr_req <= 0 || ntr_req >= length(ytr_all)
        collect(1:length(ytr_all))
    else
        rng = MersenneTwister(7); f = ntr_req / length(ytr_all)
        sort(reduce(vcat, [(ix = shuffle(rng, findall(==(k), ytr_all)); ix[1:max(1, round(Int, f * length(ix)))]) for k in 1:K if any(==(k), ytr_all)]))
    end
    ytr = ytr_all[trc]

    g = nothing; gpu = gpu_name(); rows = NamedTuple[]
    for ck in ckpts
        haskey(M, ck) && !isnothing(M[ck]) || (@warn "no $ck state in $run/model.jld2, skipping"; continue)
        t0 = time()
        m = classifier_from_state(M[ck], input, get(fcfg, "norm", "pre"))
        # one device copy for all checkpoints, weights copied in place
        isnothing(g) ? (g = eval_device(m)) : Flux.testmode!(Flux.loadmodel!(g, M[ck]))
        D = embed_dim(m.encoder)
        function pooled(X, cols)
            E = zeros(Float32, D, length(cols))
            for s in 1:bs:length(cols)
                b = s:min(s + bs - 1, length(cols))
                E[:, b] .= Array(features(g, dev(X[:, cols[b]])))
            end
            return E
        end
        Ptr = pooled(Xtr, trc); Pva = pooled(Xva, 1:length(yva)); Pte = pooled(Xte, 1:length(yte))
        t_fwd = time() - t0
        # head on cpu (dropout off); hidden = post-gelu first head layer
        h = Flux.testmode!(m.head); d1 = h.layers[1]
        am(L) = vec(getindex.(argmax(L, dims=1), 1))
        ph = am(h(Pte)); head_te = mean(ph .== yte); head_va = mean(am(h(Pva)) .== yva)
        saved = PT["preds_$ck"]; saved_te = mean(saved .== yte); agree = mean(saved .== ph)
        ntr, nva, nte = length(ytr), length(yva), length(yte)
        TR, VA, TE = 1:ntr, ntr+1:ntr+nva, ntr+nva+1:ntr+nva+nte
        yall = vcat(ytr, yva, yte)
        rp = lda_readout(hcat(Ptr, Pva, Pte), yall, K, TR, VA, TE)
        rh = lda_readout(hcat(d1(Ptr), d1(Pva), d1(Pte)), yall, K, TR, VA, TE)
        dimsd = median(vec(std(Ptr, dims=2)))
        @printf("%s %s | head test %.4f (saved %.4f, agree %.4f) val %.4f | LDA pool test %.4f val %.4f a=%g | LDA hid test %.4f val %.4f a=%g | dim-sd med %.3g | fwd %.0fs\n",
                name, ck, head_te, saved_te, agree, head_va, rp[2], rp[1], rp[3], rh[2], rh[1], rh[3], dimsd, t_fwd)
        agree < 0.97 && @warn "head preds agree with the saved preds on only $(round(agree, digits=4)) of test samples"
        jldsave(joinpath(run, "ldarefit_$ck.jld2"); trues=yte, preds_head=ph, preds_pool=rp[4], preds_hid=rh[4], test_sample_idx=tsi,
                label_names=names, alpha_pool=rp[3], alpha_hid=rh[3], ntr)
        row = (name = name, ckpt = String(ck), input = fcfg["input"], mode = fcfg["mode"], level = fcfg["level"], head_test = head_te,
               saved_test = saved_te, agree_saved = agree, head_val = head_va, lda_pool_test = rp[2], lda_pool_val = rp[1], alpha_pool = rp[3],
               lda_hid_test = rh[2], lda_hid_val = rh[1], alpha_hid = rh[3], pool_minus_head = rp[2] - head_te, pool_dimsd_med = dimsd,
               ntr = ntr, nva = nva, nte = nte, gpu = replace(gpu, ' ' => '_'), fwd_s = round(Int, t_fwd), total_s = round(Int, time() - t_start), run = run)
        append_row(joinpath(cfg["results_dir"], "ldarefit.tsv"), row); push!(rows, row)
        println("wrote $run/ldarefit_$ck.jld2, appended to ", joinpath(cfg["results_dir"], "ldarefit.tsv")); flush(stdout)
    end
    return rows
end
