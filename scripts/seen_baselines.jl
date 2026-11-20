# ID-only / Ridge / linear baselines on the seen-target task for every drug split -> results/seen/<task>/baselines.tsv
using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM, Statistics, Printf
const M = LincsFM

opts = Dict(String(k) => String(v) for (k, v) in (split(a, "="; limit=2) for a in ARGS))
t = M.load_seen_task(get(opts, "task", M.default_seen_path()))
out = joinpath(M.PKG_DIR, "results", "seen", t.reference); mkpath(out)
rows = Any[["seed", "model", "target", "n_test", "median_r", "median_r_drug", "mse", "ceiling_drug", "alpha"]]
for s in eachindex(t.seeds)
    b = M.seen_linear_baselines(t, s)
    ceil = M.seen_drug_ceiling(t, s)
    for name in ("ids", "ridge", "linear"), x in M.score_seen(t, b.test, b.pred[name]; base=b.base)
        c = get(ceil, x["target"], [NaN])
        push!(rows, Any[t.seeds[s], name, x["target"], x["n"], x["median_r"], name == "ids" ? NaN : x["median_r_drug"], x["mse"], M.nanmedian(c), b.alpha])
    end
    sm(name, key) = mean(r[key] for r in M.score_seen(t, b.test, b.pred[name]; base=b.base))
    @printf("seed %s alpha %g | raw: ids %.3f ridge %.3f linear %.3f | drug: ridge %.3f linear %.3f ceiling %.3f\n", t.seeds[s], b.alpha,
            sm("ids", "median_r"), sm("ridge", "median_r"), sm("linear", "median_r"), sm("ridge", "median_r_drug"),
            sm("linear", "median_r_drug"), mean(M.nanmedian(v) for v in values(ceil)))
end
open(joinpath(out, "baselines.tsv"), "w") do io
    for r in rows
        println(io, join(r, '\t'))
    end
end
println(joinpath(out, "baselines.tsv"))
