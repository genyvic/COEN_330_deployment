#!/usr/bin/env bash
# install.sh - installs the power-saver check tool on Ubuntu (laptop or VMware VM).
#
# One line, from the internet:
#   curl -sSL https://raw.githubusercontent.com/<team>/<repo>/main/deployment/install.sh | sudo bash
# From a copy of the repo:
#   sudo bash deployment/install.sh
# Add --autostart to start watch mode at every boot (popups without typing anything):
#   curl -sSL .../install.sh | sudo bash -s -- --autostart
#
# What it does: installs perf, Python and desktop notifications, copies the tool
# to /opt/powersaver, and adds the command `powersaver-check`.
set -euo pipefail

REPO_RAW="${REPO_RAW:-https://raw.githubusercontent.com/<team>/<repo>/main/deployment}"
DIR=/opt/powersaver
FILES="check_app.py features.py model.py dataset_B_final.csv requirements-helper.txt"
AUTOSTART=0
for a in "$@"; do [ "$a" = "--autostart" ] && AUTOSTART=1; done

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run with sudo."
    exit 1
fi

echo "1/4 Installing perf, Python and notifications..."
apt-get update -qq
apt-get install -y -qq linux-tools-common linux-tools-generic python3-venv libnotify-bin curl
apt-get install -y -qq "linux-tools-$(uname -r)" 2>/dev/null \
    || echo "    (no linux-tools package for kernel $(uname -r); the generic one is used)"

echo "2/4 Copying the tool to $DIR..."
mkdir -p "$DIR"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || echo .)"
for f in $FILES; do
    if [ -f "$HERE/$f" ]; then
        cp "$HERE/$f" "$DIR/$f"                       # run from a copy of the repo
    else
        curl -fsSL "$REPO_RAW/$f" -o "$DIR/$f"        # run with curl | bash
    fi
done

echo "3/4 Installing the Python libraries (a few minutes)..."
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install -q --upgrade pip
"$DIR/venv/bin/pip" install -q -r "$DIR/requirements-helper.txt"
"$DIR/venv/bin/python" -c "import sys; sys.path.insert(0, '$DIR'); import model; model.load_model()"

echo "4/4 Adding the command powersaver-check..."
cat > /usr/local/bin/powersaver-check <<EOF
#!/bin/sh
exec $DIR/venv/bin/python $DIR/check_app.py "\$@"
EOF
chmod +x /usr/local/bin/powersaver-check

if [ "$AUTOSTART" -eq 1 ]; then
    cat > /etc/systemd/system/powersaver-watch.service <<EOF
[Unit]
Description=Power-saver check (watch mode)
After=graphical.target

[Service]
ExecStart=/usr/local/bin/powersaver-check --watch
Restart=on-failure

[Install]
WantedBy=graphical.target
EOF
    systemctl daemon-reload
    systemctl enable --now powersaver-watch.service
    echo "Watch mode starts automatically at every boot."
fi

# Quick check: can perf count cycles? (needed for the CPU speed)
PERF_TEST="$(perf stat -x, -e cycles -- true 2>&1 || true)"
if echo "$PERF_TEST" | grep -q "not supported"; then
    echo
    echo "WARNING: perf can't count CPU cycles on this computer, so CPU speed (the model's"
    echo "main input) can't be measured. In VMware: power off the VM, then"
    echo "VM Settings > Processors > 'Virtualize CPU performance counters'."
fi

echo
echo "Done! Try it:"
echo "  sudo powersaver-check gzip -k <some big file>"
echo "  sudo powersaver-check --watch"
