# usage: julia scripts/pretrain.jl <config.toml> [key=value ...]
include(joinpath(@__DIR__, "setup.jl"))
cfg = LincsFM.load_config(ARGS[1], LincsFM.parse_overrides(ARGS[2:end]))
LincsFM.pretrain(cfg)
