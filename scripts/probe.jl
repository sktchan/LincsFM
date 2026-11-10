# usage: julia scripts/probe.jl name=<row name> ckpt=<pretrain run>/<final|best|init|final_student> [random=true init_seed=k]
#        ckpt=random[:seed] input=etf|rtf (default arch and init) | ckpt=input input=etf|rtf|raw (z / rank / raw input reference)
#        [ntr=60000 nva=10000 probe_bs=50 notes=... results_dir=...]  -> row in <results_dir>/probe.tsv, features in <results_dir>/probe_emb/<name>.jld2
include(joinpath(@__DIR__, "setup.jl"))
LincsFM.run_probe(LincsFM.load_config(nothing, LincsFM.parse_overrides(ARGS)))
