# train a no-pretraining baseline, or the trivial label baselines, from a config
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM
cfg = LincsFM.cli_config(ARGS)
get(cfg, "trivial", false) ? trivial_baselines(cfg["level"]; results_dir=cfg["results_dir"]) : baseline(cfg)
