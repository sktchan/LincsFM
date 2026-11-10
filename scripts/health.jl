# usage: julia scripts/health.jl ckpt=<pretrain run>/<final|best|init|final_student> [name=... nh=5000 pretext=v trivial=v results_dir=...]
#        -> row in <results_dir>/health.tsv (pretext and trivial default to the run's log.tsv)
include(joinpath(@__DIR__, "setup.jl"))
LincsFM.run_health(LincsFM.load_config(nothing, LincsFM.parse_overrides(ARGS)))
