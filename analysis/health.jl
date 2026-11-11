# health check of a pretrain checkpoint against its init
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM
run_health(LincsFM.cli_config(ARGS))
