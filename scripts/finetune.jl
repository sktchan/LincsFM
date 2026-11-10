# usage: julia scripts/finetune.jl <config.toml> [key=value ...]  (pretrained / random modes need ckpt=<pretrain run>/final)
include(joinpath(@__DIR__, "setup.jl"))
cfg = LincsFM.load_config(ARGS[1], LincsFM.parse_overrides(ARGS[2:end]))
LincsFM.finetune(cfg)
