# bootstrap CIs of the final_* runs (best-val checkpoint) per level and seed, with the standard contrasts -> results/bootstrap/final_<level>_s<seed>
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM

tsv = joinpath(LincsFM.PKG_DIR, "results", "finetune.tsv")
lines = filter(!isempty, readlines(tsv)); head = split(lines[1], '\t')
rows = [Dict(String(h) => String(v) for (h, v) in zip(head, split(l, '\t'))) for l in lines[2:end] if split(l, '\t') != head]

# latest row per (level, name, seed) among final_* runs; arm = name without final_ / _<level>
latest = Dict{Tuple{String,String,String},String}()
for r in rows
    startswith(r["name"], "final_") && (latest[(r["level"], r["name"], r["seed"])] = r["dir"])
end
arm(name, lvl) = replace(chopsuffix(chopprefix(name, "final_"), "_" * lvl), "pretrained_" => "", "_z" => "")

for lvl in ("lvl1", "lvl2"), seed in ("1", "2", "3")
    arms = sort([arm(n, l) => d * "#best" for ((l, n, s), d) in latest if l == lvl && s == seed])
    names = Set(first.(arms))
    length(arms) < 2 && continue
    c(a, b, pt = false) = (a in names && b in names) ? ["--contrast", "$a:$b" * (pt ? ":pt" : "")] : String[]
    cons = String[]
    for i in ("etf", "rtf"), o in ("mlm", "erecon", "lrecon")
        append!(cons, c("$(o)_$i", "scratch_$i", true))     # pretraining vs none, same input
    end
    for m in ("scratch", "mlm", "erecon", "lrecon")
        append!(cons, c("$(m)_etf", "$(m)_rtf", m != "scratch"))   # expression vs rank
    end
    append!(cons, c("emlp", "rmlp")); append!(cons, c("elog", "rlog"))
    append!(cons, c("emlp", "scratch_etf")); append!(cons, c("rmlp", "scratch_rtf"))
    append!(cons, c("elog", "scratch_etf")); append!(cons, c("rlog", "scratch_rtf"))
    args = vcat(["final_$(lvl)_s$(seed)"], ["$a=$s" for (a, s) in arms], cons)
    println("\n## $lvl seed $seed: $(length(arms)) arms")
    run(`julia --threads=1,0 $(joinpath(@__DIR__, "bootstrap.jl")) $args`)
end
