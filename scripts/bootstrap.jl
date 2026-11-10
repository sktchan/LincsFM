# usage: julia scripts/bootstrap.jl <out_prefix> name=<source> ... [--contrast A:B[:pt]] ... [--did A:B:C:D[:pt]] ... [--seed_gap g] [--all_pairs 1] [--B 2000]
# source = finetune run dir (preds_final), <dir>#best, or <file.jld2>#key (e.g. <run>/ldarefit_final.jld2#preds_pool)
# a relative out_prefix goes under results/bootstrap/; writes <out_prefix>_arms.tsv and <out_prefix>_contrasts.tsv
include(joinpath(@__DIR__, "setup.jl"))
out = isabspath(ARGS[1]) ? ARGS[1] : joinpath(LINCS_DIR, "results", "bootstrap", ARGS[1])
arms = Pair{String,String}[]; contrasts = Tuple{Vector{String},Bool}[]; seed_gap = 0.0; all_pairs = false; B = 2000
i = 2
while i <= length(ARGS)
    a = ARGS[i]
    if a == "--contrast" || a == "--did"
        p = String.(split(ARGS[i+1], ':')); k = a == "--did" ? 4 : 2
        length(p) >= k || error("$a needs $(k == 2 ? "A:B" : "A:B:C:D")")
        push!(contrasts, (p[1:k], length(p) > k && p[k+1] == "pt")); global i += 2
    elseif a == "--seed_gap"
        global seed_gap = parse(Float64, ARGS[i+1]); global i += 2
    elseif a == "--all_pairs"
        global all_pairs = ARGS[i+1] == "1"; global i += 2
    elseif a == "--B"
        global B = parse(Int, ARGS[i+1]); global i += 2
    else
        k = findfirst('=', a); isnothing(k) && error("arm must be name=source: $a")
        push!(arms, a[1:k-1] => a[k+1:end]); global i += 1
    end
end
LincsFM.run_bootstrap(out, arms; contrasts, seed_gap, all_pairs, B)
