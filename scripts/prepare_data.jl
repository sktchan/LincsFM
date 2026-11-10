# builds data/lincs.jld2 from the LINCS 2020 files; usage: julia --threads=8 scripts/prepare_data.jl [raw_dir=...] [cache=...] [out=...]
# download into raw_dir:
#      wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_all_n3026460x12328.gctx
#      wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt
#      wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt
#      wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
#      wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsProject, DataFrames, JLD2, Random, Statistics
include(joinpath(dirname(@__DIR__), "src", "LincsFM.jl"))

const LINCS_DATA = "/home/golem/scratch/chans/lincs/data"
opts = Dict(String(k) => String(v) for (k, v) in (split(a, "="; limit=2) for a in ARGS))
raw_dir = get(opts, "raw_dir", joinpath(LINCS_DATA, "lincs_loading_files"))
cache = get(opts, "cache", joinpath(LINCS_DATA, "lincs_data.jld2"))
out = get(opts, "out", joinpath(dirname(@__DIR__), "data", "lincs.jld2"))
const GCTX = "level3_beta_all_n3026460x12328.gctx"

const FILTER = Dict(:qc_pass => [Symbol("1")], :pert_type => [:ctl_untrt, :ctl_vehicle, :trt_cp])
const BAD_PLATES = ["REP.A010_JURKAT_24H_X3_B32"]
const SPLIT_SEED = 42

# seeded 80/10/10 split, same as the old tvsplit
function sample_split(N::Int, seed::Int)
    idx = shuffle(Xoshiro(seed), 1:N)
    n_test = floor(Int, N * 0.1); n_val = floor(Int, N * 0.1)
    s_test = N - n_test; s_val = s_test - n_val
    return (train = idx[1:s_val], val = idx[s_val+1:s_test], test = idx[s_test+1:end])
end

t0 = time()
for f in (GCTX, "geneinfo_beta.txt", "instinfo_beta.txt", "compoundinfo_beta.txt")
    isfile(joinpath(raw_dir, f)) || isfile(cache) || error("missing $(joinpath(raw_dir, f)) (see the wget lines at the top)")
end
println("raw_dir $raw_dir | cache $cache | out $out | $(Threads.nthreads()) threads")
lincs = LincsProject.Lincs(joinpath(raw_dir, ""), GCTX, cache)
println("all profiles: $(size(lincs.expr, 1)) genes x $(size(lincs.expr, 2)) samples ($(round(time() - t0, digits=1)) s)")

keep = LincsProject.create_filter(lincs, FILTER)
inst = lincs.inst[keep, :]
X = lincs.expr[:, keep]
str(v) = String.(string.(v))
genes = (symbol = str(lincs.gene.gene_symbol), id = str(lincs.gene.gene_id))
lincs = nothing; GC.gc()
G, N = size(X)
println("filtered (qc_pass 1, ctl_untrt / ctl_vehicle / trt_cp): $G x $N")

meta = (cell = str(inst.cell_mfc_name),
        cell_iname = str(inst.cell_iname),
        pert_id = str(inst.pert_id),
        pert_type = str(inst.pert_type),
        pert_dose = str(inst.pert_dose),
        pert_time = str(inst.pert_time),
        det_plate = str(inst.det_plate),
        sample_id = str(inst.sample_id))
all(meta.cell_iname .!= "") && all(meta.det_plate .!= "") || error("empty metadata")

sp = sample_split(N, SPLIT_SEED)
println("split sizes ", length.((sp.train, sp.val, sp.test)), " (seed $SPLIT_SEED)")

bad_plates = BAD_PLATES
println("bad plate samples: ", count(in(Set(bad_plates)), meta.det_plate))

# per-gene stats on the pretrain train split (bad plates dropped): RTF medians, ETF mean / sd
bp = Set(bad_plates)
st = LincsFM.gene_stats(X, [i for i in sp.train if !(meta.det_plate[i] in bp)])
medians, gene_mean, gene_sd = st.medians, st.mean, st.sd
println("train medians: range $(extrema(medians)), $(count(==(1f0), medians)) genes at 1")

mkpath(dirname(out))
jldsave(out; X, meta, genes, medians, gene_mean, gene_sd, split=sp, bad_plates, source=raw_dir)
println("wrote $out ($(round(filesize(out) / 2^30, digits=2)) GiB) in $(round(time() - t0, digits=1)) s")
