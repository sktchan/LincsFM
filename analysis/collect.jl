# main results table: mean ± sd test accuracy per arm and level
using Statistics, Printf

const LINCS = dirname(@__DIR__)
rdir = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(LINCS, "results")
out = length(ARGS) >= 2 ? ARGS[2] : joinpath(rdir, "main_table.tsv")

# tsv -> vector of Dict(column => string)
function read_tsv(path)
    isfile(path) || return Dict{String,String}[]
    lines = filter(!isempty, readlines(path))
    isempty(lines) && return Dict{String,String}[]
    head = split(lines[1], '\t')
    rows = Dict{String,String}[]
    for l in lines[2:end]
        f = split(l, '\t')
        f == head && continue                      # repeated header
        length(f) == length(head) || (@warn "skipping malformed row in $path: $l"; continue)
        push!(rows, Dict(String(h) => String(v) for (h, v) in zip(head, f)))
    end
    return rows
end

num(s) = something(tryparse(Float64, s), NaN)
sd(v) = length(v) > 1 ? std(v) : NaN
pm(m, s) = isnan(s) ? @sprintf("%.4f", m) : @sprintf("%.4f ± %.4f", m, s)

runs = read_tsv(joinpath(rdir, "finetune.tsv"))
isempty(runs) && @warn "no runs in $(joinpath(rdir, "finetune.tsv"))"

# a rerun of the same arm and seed replaces the earlier row
latest = Dict{Tuple{String,String,String},Dict{String,String}}()
for r in runs
    latest[(r["level"], r["name"], r["seed"])] = r
end

arms = Dict{Tuple{String,String},Vector{Dict{String,String}}}()
for ((lvl, name, _), r) in latest
    push!(get!(arms, (lvl, name), Dict{String,String}[]), r)
end

const MODE_ORDER = Dict("baseline" => 1, "scratch" => 2, "random" => 3, "pretrained" => 4)
table = NamedTuple[]
for ((lvl, name), rs) in arms
    fin = num.(getindex.(rs, "final_test")); best = num.(getindex.(rs, "best_test")); bva = num.(getindex.(rs, "best_val_acc"))
    r1 = rs[1]
    push!(table, (level = lvl, arm = name, mode = r1["mode"], objective = r1["objective"], input = r1["input"], n_seeds = length(rs),
                  seeds = join(sort(getindex.(rs, "seed")), ","), steps = join(unique(getindex.(rs, "steps")), ","),
                  final_mean = mean(fin), final_sd = sd(fin), best_mean = mean(best), best_sd = sd(best), best_val_acc = mean(bva)))
end

# trivial label baselines (one row each; latest wins)
triv = Dict{Tuple{String,String},Float64}()
for r in read_tsv(joinpath(rdir, "trivial.tsv"))
    triv[(r["level"], r["baseline"])] = num(r["acc"])
end
for ((lvl, b), acc) in triv
    push!(table, (level = lvl, arm = b, mode = "trivial", objective = b, input = "-", n_seeds = 1, seeds = "-", steps = "-",
                  final_mean = acc, final_sd = NaN, best_mean = acc, best_sd = NaN, best_val_acc = NaN))
end

isempty(table) && (println("nothing to collect in $rdir"); exit())
sort!(table, by = t -> (t.level, get(MODE_ORDER, t.mode, 0), t.objective, t.input, t.arm))

open(out, "w") do io
    println(io, join(string.(keys(table[1])), '\t'))
    for t in table
        println(io, join([v isa AbstractFloat ? @sprintf("%.6g", v) : string(v) for v in values(t)], '\t'))
    end
end

for lvl in unique(getfield.(table, :level))
    println("\n== $lvl")
    @printf("%-34s %-10s %-9s %-5s %5s  %-18s %-18s\n", "arm", "mode", "objective", "input", "seeds", "final test", "best-val test")
    for t in filter(t -> t.level == lvl, table)
        @printf("%-34s %-10s %-9s %-5s %5d  %-18s %-18s\n", t.arm, t.mode, t.objective, t.input, t.n_seeds, pm(t.final_mean, t.final_sd), pm(t.best_mean, t.best_sd))
    end
end
println("\nwrote $out")
