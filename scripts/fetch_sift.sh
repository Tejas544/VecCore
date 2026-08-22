#!/usr/bin/env bash
# Fetch SIFT1M (ANN_SIFT1M, TEXMEX corpus) and verify it structurally.
#
# CONTEXT.md D3: SIFT1M is the headline dataset because it has *published
# ground truth*, which is what makes recall measurable rather than asserted.
#
# BUGS.md P-01: .fvecs stores, per vector, a little-endian int32 dimension count
# FOLLOWED BY the values -- the dimension is repeated for every single vector,
# not written once as a header.  A reader that assumes a flat float array gets
# data that is wrong but numerically plausible: no NaNs, no crash, just quietly
# low recall that looks like an algorithm bug.  The exact-size assertions below
# are the cheap guard against having downloaded something with a different
# layout than the reader assumes.
#
# L-02: every path is quoted.  The source tree lives under a directory with a
# space in its name.

set -euo pipefail

DATA_DIR="${VECCORE_DATA:-${HOME}/veccore-data}"
SIFT_DIR="${DATA_DIR}/sift"
TARBALL="${DATA_DIR}/sift.tar.gz"
URL="${SIFT_URL:-ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz}"

# name:expected_bytes.  n * (4 + d*4) exactly -- see the P-01 note above.
#   base   1000000 * (4 + 128*4) = 516,000,000
#   learn   100000 * (4 + 128*4) =  51,600,000
#   query    10000 * (4 + 128*4) =   5,160,000
#   gt       10000 * (4 + 100*4) =   4,040,000
EXPECTED=(
  "sift_base.fvecs:516000000"
  "sift_learn.fvecs:51600000"
  "sift_query.fvecs:5160000"
  "sift_groundtruth.ivecs:4040000"
)

log() { printf '[fetch_sift] %s\n' "$*"; }
die() { printf '[fetch_sift] ERROR: %s\n' "$*" >&2; exit 1; }

file_size() { stat -c %s "$1"; }

verify() {
  local ok=1
  for entry in "${EXPECTED[@]}"; do
    local name="${entry%%:*}"
    local want="${entry##*:}"
    local path="${SIFT_DIR}/${name}"
    if [[ ! -f "${path}" ]]; then
      log "MISSING  ${name}"
      ok=0
      continue
    fi
    local got
    got="$(file_size "${path}")"
    if [[ "${got}" != "${want}" ]]; then
      log "BAD SIZE ${name}: got ${got}, expected ${want}"
      ok=0
    else
      log "ok       ${name} (${got} bytes)"
    fi
  done
  return $(( ok == 1 ? 0 : 1 ))
}

mkdir -p "${DATA_DIR}"

if verify 2>/dev/null; then
  log "SIFT1M already present and verified at ${SIFT_DIR}"
  exit 0
fi

if [[ ! -f "${TARBALL}" ]]; then
  log "downloading ~168 MB from ${URL}"
  log "(if this URL has rotted, set SIFT_URL to a mirror and re-run)"
  wget --progress=dot:giga -O "${TARBALL}.part" "${URL}" \
    || die "download failed. Set SIFT_URL to a mirror and re-run."
  mv "${TARBALL}.part" "${TARBALL}"
else
  log "tarball already downloaded: ${TARBALL}"
fi

log "extracting to ${DATA_DIR}"
tar -xzf "${TARBALL}" -C "${DATA_DIR}"

[[ -d "${SIFT_DIR}" ]] || die "expected ${SIFT_DIR} after extraction; archive layout changed?"

log "verifying"
verify || die "verification failed -- do NOT build an index on this data. \
A wrong file layout produces plausible-looking but wrong recall (P-01)."

log "SIFT1M ready at ${SIFT_DIR}"
log "next: python3 scripts/make_fixture.py"
