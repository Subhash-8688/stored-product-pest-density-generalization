#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT}/vendor/ultralytics"
PIN="eec4148e7b976cbbe1378aeee03f52337c79479e"

# Keep the generated upstream checkout outside version control.
if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  mkdir -p "${ROOT}/vendor"
  git clone https://github.com/ultralytics/ultralytics.git "${VENDOR_DIR}"
fi

git -C "${VENDOR_DIR}" fetch --all --tags
git -C "${VENDOR_DIR}" checkout --detach "${PIN}"
# Allow the setup command to be rerun without stacking the patch twice.
if git -C "${VENDOR_DIR}" apply --reverse --check "${ROOT}/patches/ultralytics.patch" >/dev/null 2>&1; then
  printf 'Ultralytics patch is already applied at %s\n' "${VENDOR_DIR}"
  exit 0
fi
git -C "${VENDOR_DIR}" apply --check "${ROOT}/patches/ultralytics.patch"
git -C "${VENDOR_DIR}" apply "${ROOT}/patches/ultralytics.patch"

printf 'Patched Ultralytics at %s\n' "${VENDOR_DIR}"
printf 'Run: export PYTHONPATH="%s:$PYTHONPATH"\n' "${VENDOR_DIR}"
