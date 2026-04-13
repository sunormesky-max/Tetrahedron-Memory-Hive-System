#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# openclaw-integrate.sh
# Idempotent script to integrate TetraMem-XL as the memory backend for OpenClaw.
# Safe to run multiple times; handles fresh installs and re-patching after upgrades.
# ---------------------------------------------------------------------------

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BACKUP_DIR="${SCRIPT_DIR}/openclaw-backups/$(date +%Y%m%d_%H%M%S)"
readonly TETRAMEM_MANAGER_TEMPLATE="${SCRIPT_DIR}/tetramem-manager-template.js"
readonly TETRAMEM_API_HOST="127.0.0.1"
readonly TETRAMEM_API_PORT="8000"
readonly TETRAMEM_API_URL="http://${TETRAMEM_API_HOST}:${TETRAMEM_API_PORT}"

# ---------- colours / helpers ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
log_warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
log_error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
log_step()  { printf "\n${CYAN}==> STEP %s${NC}: %s\n" "$1" "$2"; }

die() {
  log_error "$@"
  exit 1
}

command_exists() {
  command -v "$1" &>/dev/null
}

# ---------- STEP 1: Detect OpenClaw installation ----------
detect_openclaw_path() {
  log_step "1" "Detecting OpenClaw installation path"

  local candidates=()

  # pnpm global paths
  if command_exists pnpm; then
    local pnpm_root
    pnpm_root="$(pnpm root -g 2>/dev/null || true)"
    if [[ -n "${pnpm_root}" ]]; then
      candidates+=("${pnpm_root}/openclaw")
      candidates+=("${pnpm_root}/@opencode/openclaw")
    fi
  fi

  # npm global paths
  if command_exists npm; then
    local npm_root
    npm_root="$(npm root -g 2>/dev/null || true)"
    if [[ -n "${npm_root}" ]]; then
      candidates+=("${npm_root}/openclaw")
      candidates+=("${npm_root}/@opencode/openclaw")
    fi
  fi

  # pnpm .pnpm store paths (actual install location with hash)
  local pnpm_store="${HOME}/.local/share/pnpm/global/5/.pnpm"
  if [[ -d "${pnpm_store}" ]]; then
    for d in "${pnpm_store}"/openclaw@*/node_modules/openclaw; do
      candidates+=("$d")
    done
  fi

  # Common locations
  candidates+=(
    "/usr/local/lib/node_modules/openclaw"
    "/usr/local/lib/node_modules/@opencode/openclaw"
    "${HOME}/.local/share/pnpm/global/5/node_modules/openclaw"
    "${HOME}/.local/share/pnpm/global/5/node_modules/@opencode/openclaw"
  )

  # Allow override via environment variable
  if [[ -n "${OPENCLAW_PATH:-}" ]]; then
    candidates=("${OPENCLAW_PATH}" "${candidates[@]}")
  fi

  for dir in "${candidates[@]}"; do
    if [[ -d "${dir}" && -d "${dir}/dist" ]]; then
      OPENCLAW_PATH="${dir}"
      log_info "Found OpenClaw at: ${OPENCLAW_PATH}"
      return 0
    fi
  done

  die "Could not detect OpenClaw installation. Set OPENCLAW_PATH to the installation directory."
}

# ---------- STEP 2: Verify OpenClaw version ----------
verify_version() {
  log_step "2" "Verifying OpenClaw version"

  local pkg_json="${OPENCLAW_PATH}/package.json"
  if [[ ! -f "${pkg_json}" ]]; then
    die "package.json not found at ${OPENCLAW_PATH}"
  fi

  OPENCLAW_VERSION="$(grep -oP '"version"\s*:\s*"\K[^"]+' "${pkg_json}" 2>/dev/null || true)"
  if [[ -z "${OPENCLAW_VERSION}" ]]; then
    # Fallback: try python or node to parse
    OPENCLAW_VERSION="$(node -e "console.log(require('${pkg_json}').version)" 2>/dev/null || true)"
  fi

  if [[ -z "${OPENCLAW_VERSION}" ]]; then
    log_warn "Could not determine OpenClaw version — continuing anyway"
  else
    log_info "OpenClaw version: ${OPENCLAW_VERSION}"
  fi
}

# ---------- STEP 3: Detect dist/ directory and hash-based files ----------
detect_dist_files() {
  log_step "3" "Detecting dist/ directory with hash-based filenames"

  DIST_DIR="${OPENCLAW_PATH}/dist"
  if [[ ! -d "${DIST_DIR}" ]]; then
    die "dist/ directory not found at ${OPENCLAW_PATH}"
  fi

  log_info "dist/ directory: ${DIST_DIR}"

  # Find the specific files we need to patch
  ZOD_SCHEMA_FILE="$(find "${DIST_DIR}" -maxdepth 1 -name 'zod-schema-*.js' -not -name '*.map' | head -n 1 || true)"
  RUNTIME_SCHEMA_FILE="$(find "${DIST_DIR}" -maxdepth 1 -name 'runtime-schema-*.js' -not -name '*.map' | head -n 1 || true)"
  BACKEND_CONFIG_FILE="$(find "${DIST_DIR}" -maxdepth 1 -name 'backend-config-*.js' -not -name '*.map' | head -n 1 || true)"
  # memory-*.js but NOT memory-core-*.js and NOT memory-host-*.js
  MEMORY_FILE="$(find "${DIST_DIR}" -maxdepth 1 -name 'memory-*.js' -not -name 'memory-core-*' -not -name 'memory-host-*' -not -name '*.map' | head -n 1 || true)"

  local missing=()
  [[ -z "${ZOD_SCHEMA_FILE}" ]]      && missing+=("zod-schema-*.js")
  [[ -z "${RUNTIME_SCHEMA_FILE}" ]]   && missing+=("runtime-schema-*.js")
  [[ -z "${BACKEND_CONFIG_FILE}" ]]   && missing+=("backend-config-*.js")
  [[ -z "${MEMORY_FILE}" ]]           && missing+=("memory-*.js (main loader)")

  if [[ ${#missing[@]} -gt 0 ]]; then
    die "Missing dist files: ${missing[*]}"
  fi

  log_info "zod-schema:      $(basename "${ZOD_SCHEMA_FILE}")"
  log_info "runtime-schema:  $(basename "${RUNTIME_SCHEMA_FILE}")"
  log_info "backend-config:  $(basename "${BACKEND_CONFIG_FILE}")"
  log_info "memory-loader:   $(basename "${MEMORY_FILE}")"
}

# ---------- STEP 4: Backup all files ----------
backup_files() {
  log_step "4" "Backing up files before modification"

  mkdir -p "${BACKUP_DIR}"

  local files_to_backup=(
    "${ZOD_SCHEMA_FILE}"
    "${RUNTIME_SCHEMA_FILE}"
    "${BACKEND_CONFIG_FILE}"
    "${MEMORY_FILE}"
  )

  # Also backup config if it exists
  local config_candidates=(
    "${OPENCLAW_PATH}/openclaw.json"
    "${OPENCLAW_PATH}/config/openclaw.json"
    "${OPENCLAW_PATH}/.openclaw/openclaw.json"
    "${HOME}/.config/openclaw/openclaw.json"
    "${HOME}/.openclaw/openclaw.json"
  )

  for cfg in "${config_candidates[@]}"; do
    if [[ -f "${cfg}" ]]; then
      files_to_backup+=("${cfg}")
      OPENCLAW_CONFIG="${cfg}"
      break
    fi
  done

  for f in "${files_to_backup[@]}"; do
    if [[ -f "${f}" ]]; then
      cp -a "${f}" "${BACKUP_DIR}/"
      log_info "Backed up: $(basename "${f}")"
    fi
  done

  log_info "Backup directory: ${BACKUP_DIR}"
}

# ---------- STEP 5: Patch files ----------

# Helper: check if a file already has tetramem patches
is_patched() {
  local file="$1"
  grep -q "tetramem" "${file}" 2>/dev/null
}

patch_zod_schema() {
  log_info "Patching zod-schema..."
  local file="${ZOD_SCHEMA_FILE}"

  if is_patched "${file}"; then
    log_info "  Already patched — skipping"
    return 0
  fi

  # Add "tetramem" to the backend union:
  #   z.union([z.literal("builtin"), z.literal("qmd")])
  #   -> z.union([z.literal("builtin"), z.literal("qmd"), z.literal("tetramem")])
  sed -i 's/z\.union(\[z\.literal("builtin"),z\.literal("qmd")/z.union([z.literal("builtin"),z.literal("qmd"),z.literal("tetramem")/' "${file}"

  # Also handle the case with spaces
  sed -i 's/z\.union(\[z\.literal("builtin"), z\.literal("qmd")/z.union([z.literal("builtin"), z.literal("qmd"), z.literal("tetramem")/' "${file}"

  # Handle .strict() schema: add tetramem config property before .strict()
  # We look for the memory-related strict schema and add the tetramem property
  if ! grep -q 'tetramem:' "${file}"; then
    # Add tetramem config as an optional z object before .strict() in the memory schema area
    # Pattern: find the memory config schema with .strict() and add tetramem property
    sed -i '/\.strict()/i\    tetramem: z.object({ url: z.string().optional(), timeout: z.number().optional(), apiKey: z.string().optional() }).optional(),' "${file}" || true
  fi

  log_info "  Done"
}

patch_runtime_schema() {
  log_info "Patching runtime-schema..."
  local file="${RUNTIME_SCHEMA_FILE}"

  if is_patched "${file}"; then
    log_info "  Already patched — skipping"
    return 0
  fi

  # Add "tetramem" to the anyOf array in JSON Schema format
  # Pattern: {"type":"string","const":"builtin"} ... {"type":"string","const":"qmd"}
  # We add {"type":"string","const":"tetramem"} after the qmd entry
  sed -i 's/{"type":"string","const":"qmd"}/{"type":"string","const":"qmd"},{"type":"string","const":"tetramem"}/g' "${file}"

  # Handle with spaces
  sed -i 's/{ "type": "string", "const": "qmd" }/{ "type": "string", "const": "qmd" }, { "type": "string", "const": "tetramem" }/g' "${file}"

  # Add tetramem property definition to the properties block (before closing })
  # Look for the backend config properties and add tetramem
  if ! grep -q '"tetramem"' "${file}"; then
    # Insert tetramem property definition near other backend config properties
    sed -i '/"qmd"/a\                                                    ,"tetramem":{"type":"object","properties":{"url":{"type":"string"},"timeout":{"type":"number"},"apiKey":{"type":"string"}},"additionalProperties":false}' "${file}" || true
  fi

  log_info "  Done"
}

patch_backend_config() {
  log_info "Patching backend-config..."
  local file="${BACKEND_CONFIG_FILE}"

  if is_patched "${file}"; then
    log_info "  Already patched — skipping"
    return 0
  fi

  # Add a tetramem branch in resolveMemoryBackendConfig() before the fallback:
  #   if (backend !== "qmd") return {backend: "builtin"...}
  # We insert before this line:
  #   if (backend === "tetramem") return {backend: "tetramem", url: config?.tetramem?.url || "http://127.0.0.1:8000", ...}
  sed -i 's/if (backend !== "qmd")/if (backend === "tetramem") return { backend: "tetramem", url: (config \&\& config.tetramem \&\& config.tetramem.url) || "http:\/\/127.0.0.1:8000", timeout: (config \&\& config.tetramem \&\& config.tetramem.timeout) || 30000, apiKey: (config \&\& config.tetramem \&\& config.tetramem.apiKey) || null };\n  if (backend !== "qmd")/' "${file}"

  # Alternative pattern with different formatting
  if ! grep -q 'backend === "tetramem"' "${file}"; then
    sed -i 's/if(backend!=="qmd")/if(backend==="tetramem")return{backend:"tetramem",url:config?.tetramem?.url||"http:\/\/127.0.0.1:8000",timeout:config?.tetramem?.timeout||3e4,apiKey:config?.tetramem?.apiKey||null};if(backend!=="qmd")/' "${file}"
  fi

  log_info "  Done"
}

patch_memory_loader() {
  log_info "Patching memory loader..."
  local file="${MEMORY_FILE}"

  if is_patched "${file}"; then
    log_info "  Already patched — skipping"
    return 0
  fi

  # Find the pattern where after resolving backend config, if "qmd" loads QmdMemoryManager,
  # otherwise falls back to builtin. We add a tetramem branch.
  #
  # Pattern: if (backend === "qmd") { ... load Qmd ... }
  #          We add before the else/fallback:
  #          if (backend === "tetramem") { ... load TetraMem ... }

  local tetramem_loader_block='if(backend==="tetramem"){const{createTetraMemManager}=require("./tetramem-manager.js");return createTetraMemManager(resolvedConfig,params);}'

  # Insert tetramem branch — try several common patterns
  # Pattern 1: } else { for builtin fallback after qmd block
  if grep -q 'backend === "qmd"' "${file}" || grep -q 'backend==="qmd"' "${file}"; then
    # Add our branch right after the qmd block closes, before the builtin fallback
    sed -i "/backend.*===.*[\"']qmd[\"']/a\\
  else if(backend===\"tetramem\"){const{createTetraMemManager}=require(\"./tetramem-manager.js\");return createTetraMemManager(resolvedConfig,params);}" "${file}" 2>/dev/null || true

    # If the simple insert didn't create the tetramem check, try a broader approach
    if ! grep -q 'backend.*===.*"tetramem"' "${file}"; then
      # Prepend the tetramem import/require and branch at the top of the function
      # by inserting near the module requires
      sed -i '1s/^/\/\* TetraMem integration \*\/\n/' "${file}"

      # Add tetramem branch before the final builtin fallback return
      # Find "return builtin" or "return new BuiltinMemory" or similar and insert before
      sed -i '/return.*[Bb]uiltin[Mm]emory/i\  if (backend === "tetramem") { const { createTetraMemManager } = require("./tetramem-manager.js"); return createTetraMemManager(resolvedConfig, params); }\n' "${file}" 2>/dev/null || true
    fi
  fi

  # Verify the patch took
  if ! grep -q 'tetramem' "${file}"; then
    log_warn "  Could not auto-patch memory loader. Manual patching may be required."
    log_warn "  Look for the backend dispatch logic in $(basename "${file}") and add:"
    log_warn '    if (backend === "tetramem") {'
    log_warn '      const { createTetraMemManager } = require("./tetramem-manager.js");'
    log_warn '      return createTetraMemManager(resolvedConfig, params);'
    log_warn '    }'
  fi

  log_info "  Done"
}

patch_all_files() {
  log_step "5" "Patching files for TetraMem integration"
  patch_zod_schema
  patch_runtime_schema
  patch_backend_config
  patch_memory_loader
}

# ---------- STEP 6: Deploy TetraMem manager module ----------
deploy_tetramem_manager() {
  log_step "6" "Deploying TetraMem manager module"

  if [[ ! -f "${TETRAMEM_MANAGER_TEMPLATE}" ]]; then
    die "TetraMem manager template not found at ${TETRAMEM_MANAGER_TEMPLATE}"
  fi

  cp "${TETRAMEM_MANAGER_TEMPLATE}" "${DIST_DIR}/tetramem-manager.js"
  log_info "Deployed tetramem-manager.js to ${DIST_DIR}/"
}

# ---------- STEP 7: Update openclaw.json ----------
update_openclaw_config() {
  log_step "7" "Updating openclaw.json"

  # If we didn't find a config earlier, search again or create one
  if [[ -z "${OPENCLAW_CONFIG:-}" ]]; then
    local config_candidates=(
      "${OPENCLAW_PATH}/openclaw.json"
      "${OPENCLAW_PATH}/config/openclaw.json"
      "${OPENCLAW_PATH}/.openclaw/openclaw.json"
      "${HOME}/.config/openclaw/openclaw.json"
      "${HOME}/.openclaw/openclaw.json"
    )
    for cfg in "${config_candidates[@]}"; do
      if [[ -f "${cfg}" ]]; then
        OPENCLAW_CONFIG="${cfg}"
        break
      fi
    done
  fi

  if [[ -z "${OPENCLAW_CONFIG:-}" ]]; then
    # Create a new config in the OpenClaw directory
    OPENCLAW_CONFIG="${OPENCLAW_PATH}/openclaw.json"
    log_info "No existing config found — creating ${OPENCLAW_CONFIG}"
    echo '{}' > "${OPENCLAW_CONFIG}"
  fi

  # Use node to update the JSON (available since OpenClaw requires node)
  node -e '
    const fs = require("fs");
    const path = process.argv[1];
    const cfg = JSON.parse(fs.readFileSync(path, "utf8"));

    if (!cfg.memory) cfg.memory = {};
    cfg.memory.backend = "tetramem";
    cfg.memory.tetramem = {
      url: "http://127.0.0.1:8000",
      timeout: 30000
    };

    fs.writeFileSync(path, JSON.stringify(cfg, null, 2) + "\n");
    console.log("Updated " + path);
  ' "${OPENCLAW_CONFIG}"

  log_info "Set memory.backend = \"tetramem\" in ${OPENCLAW_CONFIG}"
}

# ---------- STEP 8: Start TetraMem API service ----------
start_tetramem_api() {
  log_step "8" "Starting TetraMem API service"

  # Check if already running
  if curl -sf "http://${TETRAMEM_API_HOST}:${TETRAMEM_API_PORT}/api/v1/status" &>/dev/null; then
    log_info "TetraMem API already running on ${TETRAMEM_API_URL}"
    return 0
  fi

  if ! command_exists pip; then
    if command_exists pip3; then
      PIP_CMD="pip3"
    else
      die "pip/pip3 not found — required to install tetrahedron-memory"
    fi
  else
    PIP_CMD="pip"
  fi

  log_info "Installing tetrahedron-memory[api]..."
  "${PIP_CMD}" install "tetrahedron-memory[api]" 2>&1 | tail -5 || {
    log_warn "pip install failed — the TetraMem API may already be installed or needs manual setup"
  }

  log_info "Starting TetraMem API on ${TETRAMEM_API_HOST}:${TETRAMEM_API_PORT}..."

  # Start in background, nohup
  nohup python -m tetrahedron_memory.api \
    --host "${TETRAMEM_API_HOST}" \
    --port "${TETRAMEM_API_PORT}" \
    > "${SCRIPT_DIR}/tetramem-api.log" 2>&1 &
  TETRAMEM_PID=$!

  log_info "TetraMem API PID: ${TETRAMEM_PID}"

  # Wait for it to come up
  local retries=30
  while (( retries > 0 )); do
    if curl -sf "http://${TETRAMEM_API_HOST}:${TETRAMEM_API_PORT}/api/v1/status" &>/dev/null; then
      log_info "TetraMem API is up at ${TETRAMEM_API_URL}"
      return 0
    fi
    retries=$((retries - 1))
    sleep 1
  done

  log_warn "TetraMem API did not respond within 30s — check ${SCRIPT_DIR}/tetramem-api.log"
  log_warn "You can start it manually: python -m tetrahedron_memory.api --host ${TETRAMEM_API_HOST} --port ${TETRAMEM_API_PORT}"
}

# ---------- STEP 9: Restart OpenClaw gateway ----------
restart_openclaw() {
  log_step "9" "Restarting OpenClaw gateway"

  if command_exists pnpm; then
    # Try to find and restart via pnpm
    if pnpm list -g openclaw &>/dev/null; then
      log_info "Restarting via pnpm..."
      pnpm restart openclaw 2>/dev/null && {
        log_info "OpenClaw gateway restarted"
        return 0
      }
    fi
  fi

  # Try npm
  if command_exists npm; then
    if npm list -g openclaw &>/dev/null; then
      log_info "Restarting via npm..."
      npm restart -g openclaw 2>/dev/null && {
        log_info "OpenClaw gateway restarted"
        return 0
      }
    fi
  fi

  # Try to find and kill the process, then the user can restart
  local oc_pid
  oc_pid="$(pgrep -f "openclaw" 2>/dev/null || true)"
  if [[ -n "${oc_pid}" ]]; then
    log_info "Found OpenClaw process(es): ${oc_pid}"
    log_info "Sending SIGHUP to reload..."
    kill -HUP ${oc_pid} 2>/dev/null || true
  else
    log_warn "Could not auto-restart OpenClaw. Please restart it manually."
    log_warn "  pnpm start openclaw   OR   npm start -g openclaw"
  fi
}

# ---------- STEP 10: Verify ----------
verify_integration() {
  log_step "10" "Verifying integration"

  local errors=0

  # 10a. Check tetramem-manager.js exists in dist
  if [[ -f "${DIST_DIR}/tetramem-manager.js" ]]; then
    log_info "[OK] tetramem-manager.js deployed"
  else
    log_error "[FAIL] tetramem-manager.js missing from dist/"
    errors=$((errors + 1))
  fi

  # 10b. Check patched files contain "tetramem"
  for f in "${ZOD_SCHEMA_FILE}" "${RUNTIME_SCHEMA_FILE}" "${BACKEND_CONFIG_FILE}" "${MEMORY_FILE}"; do
    local bn
    bn="$(basename "${f}")"
    if grep -q "tetramem" "${f}" 2>/dev/null; then
      log_info "[OK] ${bn} patched"
    else
      log_error "[FAIL] ${bn} not patched"
      errors=$((errors + 1))
    fi
  done

  # 10c. Check openclaw.json config
  if [[ -f "${OPENCLAW_CONFIG:-}" ]]; then
    if grep -q '"tetramem"' "${OPENCLAW_CONFIG}" 2>/dev/null; then
      log_info "[OK] openclaw.json configured"
    else
      log_error "[FAIL] openclaw.json not configured"
      errors=$((errors + 1))
    fi
  fi

  # 10d. Check TetraMem API
  if curl -sf "http://${TETRAMEM_API_HOST}:${TETRAMEM_API_PORT}/api/v1/status" &>/dev/null; then
    log_info "[OK] TetraMem API reachable"
  else
    log_warn "[WARN] TetraMem API not reachable at ${TETRAMEM_API_URL}"
  fi

  echo ""
  if [[ ${errors} -eq 0 ]]; then
    log_info "All checks passed! TetraMem-XL integration complete."
  else
    log_error "${errors} check(s) failed — review output above"
    exit 1
  fi
}

# ---------- main ----------
main() {
  echo "============================================"
  echo " TetraMem-XL  <-->  OpenClaw Integration"
  echo "============================================"

  detect_openclaw_path
  verify_version
  detect_dist_files
  backup_files
  patch_all_files
  deploy_tetramem_manager
  update_openclaw_config
  start_tetramem_api
  restart_openclaw
  verify_integration

  echo ""
  log_info "Done. Backups are in ${BACKUP_DIR}"
}

main "$@"
