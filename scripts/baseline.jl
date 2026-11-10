# usage: julia scripts/baseline.jl <config.toml> [key=value ...]  (model = elog | rlog | emlp | rmlp | elda | rlda; trivial = true runs the trivial baselines for level)
include(joinpath(@__DIR__, "setup.jl"))
cfg = LincsFM.load_config(ARGS[1], LincsFM.parse_overrides(ARGS[2:end]))
if get(cfg, "trivial", false)
    LincsFM.trivial_baselines(cfg["level"]; results_dir=cfg["results_dir"])
else
    LincsFM.baseline(cfg)
end
