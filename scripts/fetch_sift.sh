#!/usr/bin/env bash
# Fetch SIFT1M (ANN_SIFT1M, TEXMEX corpus) and verify it structurally.
#
# CONTEXT.md D3: SIFT1M is the headline dataset because it has *published ground
# truth*, which is what makes recall measurable rather than asserted.
#
# ---------------------------------------------------------------------------
# On sources, and why there are two
# ---------------------------------------------------------------------------
# The canonical distribution (corpus-texmex.irisa.fr) offers FTP only. On many
# networks -- including this one -- outbound port 21 is blocked, and the failure
# presents as an indefinite hang writing a 0-byte file with no diagnostic
# whatsoever. So: the canonical source is tried FIRST and with a short connect
# timeout, and a mirror serving the identical files over HTTPS is the fallback.
#
# ---------------------------------------------------------------------------
# On trusting a mirror
# ---------------------------------------------------------------------------
# A third-party mirror deserves suspicion. Three things answer it, in order of
# strength:
#
#   1. Exact byte counts. Each file must be exactly n * (4 + 4*d) for the known
#      SIFT1M geometry. This is asserted below and it is not a soft check.
#   2. Per-record structure. The C++ reader (src/xvecs.cpp) verifies that every
#      single record repeats the same dimension -- see P-01.
#   3. **Phase 1's gate is itself the authenticity check.** Brute-force search
#      over the base vectors must reproduce the *published* ground truth
#      exactly. Data that satisfies that is either the real SIFT1M or an
#      internally consistent forgery of it, and the second is not a threat
#      model that applies to a benchmark corpus.
#
# BUGS.md P-01: .fvecs stores, per vector, a little-endian int32 dimension count
# FOLLOWED BY the values -- repeated for every vector, not a one-time header. A
# reader that assumes a flat float array gets numerically plausible garbage: no
# crash, no NaN, just quietly low recall that looks like an algorithm bug.
#
# L-02: every path is quoted. The source tree lives under a directory with a
# space in its name.

set -euo pipefail

DATA_DIR="${VECCORE_DATA:-${HOME}/veccore-data}"
SIFT_DIR="${DATA_DIR}/sift"

FTP_TARBALL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
MIRROR_BASE="https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main"

# name:expected_bytes -- n * (4 + d*4), exactly.
#   base   1000000 * (4 + 128*4) = 516,000,000
#   query    10000 * (4 + 128*4) =   5,160,000
#   gt       10000 * (4 + 100*4) =   4,040,000
REQUIRED=(
  "sift_base.fvecs:516000000"
  "sift_query.fvecs:5160000"
  "sift_groundtruth.ivecs:4040000"
)

log()  { printf '[fetch_sift] %s\n' "$*"; }
die()  { printf '[fetch_sift] ERROR: %s\n' "$*" >&2; exit 1; }

verify() {
  local ok=1 entry name want path got
  for entry in "${REQUIRED[@]}"; do
    name="${entry%%:*}"
    want="${entry##*:}"
    path="${SIFT_DIR}/${name}"
    if [[ ! -f "${path}" ]]; then
      log "MISSING  ${name}"
      ok=0
      continue
    fi
    got="$(stat -c %s "${path}")"
    if [[ "${got}" != "${want}" ]]; then
      log "BAD SIZE ${name}: got ${got}, expected ${want}"
      ok=0
    else
      log "ok       ${name} (${got} bytes)"
    fi
  done
  [[ "${ok}" == 1 ]]
}

mkdir -p "${SIFT_DIR}"

if verify >/dev/null 2>&1; then
  log "SIFT1M already present and verified at ${SIFT_DIR}"
  verify
  exit 0
fi

# --- attempt 1: the canonical source, failing fast rather than hanging -------
log "trying canonical source (FTP, 15s connect timeout)"
if curl -fsS --connect-timeout 15 --max-time 1800 -o "${DATA_DIR}/sift.tar.gz.part" "${FTP_TARBALL}"; then
  mv "${DATA_DIR}/sift.tar.gz.part" "${DATA_DIR}/sift.tar.gz"
  log "extracting"
  tar -xzf "${DATA_DIR}/sift.tar.gz" -C "${DATA_DIR}"
else
  rm -f "${DATA_DIR}/sift.tar.gz.part"
  log "canonical FTP source unreachable (port 21 is commonly blocked)"
  log "falling back to HTTPS mirror: ${MIRROR_BASE}"
  log "downloading ~525 MB across 3 files"

  curl -fL --progress-bar -o "${SIFT_DIR}/sift_base.fvecs.part" \
       "${MIRROR_BASE}/sift_base.fvecs" || die "mirror download failed: sift_base.fvecs"
  mv "${SIFT_DIR}/sift_base.fvecs.part" "${SIFT_DIR}/sift_base.fvecs"

  curl -fL --progress-bar -o "${SIFT_DIR}/sift_query.fvecs.part" \
       "${MIRROR_BASE}/sift_query.fvecs" || die "mirror download failed: sift_query.fvecs"
  mv "${SIFT_DIR}/sift_query.fvecs.part" "${SIFT_DIR}/sift_query.fvecs"

  curl -fL --progress-bar -o "${SIFT_DIR}/sift_groundtruth.ivecs.part" \
       "${MIRROR_BASE}/sift_groundtruth.ivecs" || die "mirror download failed: sift_groundtruth.ivecs"
  mv "${SIFT_DIR}/sift_groundtruth.ivecs.part" "${SIFT_DIR}/sift_groundtruth.ivecs"
fi

log "verifying"
verify || die "verification failed -- do NOT build an index on this data. \
A wrong file layout produces plausible-looking but wrong recall (P-01)."

log "SIFT1M ready at ${SIFT_DIR}"
log "next: ~/veccore-venv/bin/python scripts/make_fixture.py"
