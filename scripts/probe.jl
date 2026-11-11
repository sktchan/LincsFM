# frozen lvl2 LDA probe of an encoder checkpoint or a reference
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM
run_probe(LincsFM.cli_config(ARGS))
