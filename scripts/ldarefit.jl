# usage: julia scripts/ldarefit.jl run=<finetune run dir> [name=... ckpts=final,best ntr=0 lda_bs=100 results_dir=...]
#        -> rows in <results_dir>/ldarefit.tsv, preds in <run>/ldarefit_<ckpt>.jld2
include(joinpath(@__DIR__, "setup.jl"))
LincsFM.run_ldarefit(LincsFM.load_config(nothing, LincsFM.parse_overrides(ARGS)))
