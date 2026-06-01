#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT}/vendor/ultralytics"
PIN="eec4148e7b976cbbe1378aeee03f52337c79479e"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  mkdir -p "${ROOT}/vendor"
  git clone https://github.com/ultralytics/ultralytics.git "${VENDOR_DIR}"
fi

git -C "${VENDOR_DIR}" fetch --all --tags
git -C "${VENDOR_DIR}" checkout --detach "${PIN}"
if git -C "${VENDOR_DIR}" apply --reverse --check "${ROOT}/patches/ultralytics_chapter2.patch" >/dev/null 2>&1; then
  printf 'Ultralytics patch is already applied at %s\n' "${VENDOR_DIR}"
  exit 0
fi
git -C "${VENDOR_DIR}" apply --check "${ROOT}/patches/ultralytics_chapter2.patch"
git -C "${VENDOR_DIR}" apply "${ROOT}/patches/ultralytics_chapter2.patch"

printf 'Patched Ultralytics at %s\n' "${VENDOR_DIR}"
printf 'Run: export PYTHONPATH="%s:$PYTHONPATH"\n' "${VENDOR_DIR}"
