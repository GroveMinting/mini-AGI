# JetPack does not ship cuSPARSELt, but the NVIDIA torch wheel links it.
# Install it user-locally (no sudo needed), then source this first:
#
#   curl -LO https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz
#   mkdir -p "$HOME/.local/lib/cusparselt"
#   tar -xf libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz \
#       -C "$HOME/.local/lib/cusparselt" --strip-components=1
#
#   . scripts/jetson_env.sh
_cusparselt_lib="$(find "${HOME}/.local/lib/cusparselt" -name 'libcusparseLt.so.0' -printf '%h\n' 2>/dev/null | head -1)"
if [ -n "${_cusparselt_lib}" ]; then
    export LD_LIBRARY_PATH="${_cusparselt_lib}:${LD_LIBRARY_PATH:-}"
fi
unset _cusparselt_lib
