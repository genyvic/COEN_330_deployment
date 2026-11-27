#!/usr/bin/env python3
"""
check_app.py - measures the first 5 seconds of a program (Linux), predicts if
power-saver mode is worth it, shows a popup and opens the website.

Two modes:
  sudo python3 check_app.py gzip -k big.txt   # starts the program and measures it
  sudo python3 check_app.py --watch           # waits for programs from a list to start

It measures exactly like our data collection script (collection/collect_data.py):
same 13 perf counters, read every second for 5 seconds, plus the program's
threads, memory and I/O sampled every 0.25 s. Then features.py turns that into
the 23 model inputs, like the preprocessing notebook.

Needs: Linux, perf, sudo. RAPL (Intel) for power; without it (e.g. in a virtual
machine), the training average is used for power and temperature.
"""
import argparse
import glob
import os
import pwd
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import FEATURES, WINDOW_S, compute_features  # noqa: E402

# Change this to your Streamlit link after deploying (or set POWERSAVER_URL)
WEBSITE_URL = os.environ.get("POWERSAVER_URL", "https://<your-app>.streamlit.app")

# Same counters as collect_data.py (they fit in the CPU's hardware counters)
PERF_EVENTS = [
    "task-clock", "context-switches", "cpu-migrations",
    "page-faults", "minor-faults", "major-faults",
    "cycles", "instructions", "ref-cycles",
    "cache-references", "cache-misses", "branch-misses", "L1-dcache-load-misses",
]

# Programs that watch mode reacts to (process names, as shown by `ps`)
WATCH_LIST = {
    "7z", "7za", "7zz", "magick", "convert", "blender", "soffice.bin", "mpv",
    "gzip", "bzip2", "xz", "zstd", "pigz", "brotli", "lz4", "ffmpeg", "x265",
    "make", "sqlite3", "octave", "octave-cli", "ngspice", "iverilog", "vvp", "yosys",
    "gmsh", "tesseract", "verilator", "rg", "sha256sum", "dot",
}


def col(event):
    """perf event name -> column name, e.g. 'L1-dcache-load-misses' -> 'l1_dcache_load_misses'"""
    return event.lower().replace("-", "_")


# ------------------------------------------------------------------ power and temperature
# (copied from collect_data.py)
def find_rapl_domains():
    domains = {}
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        try:
            name = open(f"{d}/name").read().strip()
            open(f"{d}/energy_uj").read()
        except OSError:
            continue
        if name in domains:
            name = f"{name}_{d.split(':', 1)[1]}"
        domains[name] = d
    return domains


def read_rapl(domains):
    out = {}
    for name, d in domains.items():
        with open(f"{d}/energy_uj") as f:
            e = int(f.read())
        with open(f"{d}/max_energy_range_uj") as f:
            m = int(f.read())
        out[name] = (e, m)
    return out


def package_watts(start, end, seconds):
    """CPU package power (W) between two RAPL readings."""
    joules = 0.0
    for name, (e0, m) in start.items():
        if name.startswith("package"):
            diff = end[name][0] - e0
            if diff < 0:                      # the counter went back to 0
                diff += m
            joules += diff / 1e6
    return joules / seconds


def cpu_temp():
    zones = glob.glob("/sys/class/thermal/thermal_zone*")
    for z in zones:                           # prefer the CPU package sensor
        try:
            if open(f"{z}/type").read().strip() == "x86_pkg_temp":
                return int(open(f"{z}/temp").read()) / 1000
        except OSError:
            pass
    for z in zones:
        try:
            return int(open(f"{z}/temp").read()) / 1000
        except OSError:
            pass
    return None


def measure_idle(domains, seconds):
    """Power (W) and CPU use of the whole computer (%) before the program starts."""
    psutil.cpu_percent(None)
    e0, t0 = (read_rapl(domains) if domains else None), time.monotonic()
    time.sleep(seconds)
    e1, t1 = (read_rapl(domains) if domains else None), time.monotonic()
    power = package_watts(e0, e1, t1 - t0) if domains else None
    return power, psutil.cpu_percent(None)


class IdleMonitor:
    """Watch mode: keeps measuring the computer's power and CPU use every second,
    so we know what they were just BEFORE a program started."""

    def __init__(self, domains, keep=3):
        self.domains, self.keep, self.samples = domains, keep, []
        psutil.cpu_percent(None)
        self.last = (read_rapl(domains) if domains else None, time.monotonic())

    def tick(self):
        e0, t0 = self.last
        if time.monotonic() - t0 < 1.0:
            return
        e1, t1 = (read_rapl(self.domains) if self.domains else None), time.monotonic()
        power = package_watts(e0, e1, t1 - t0) if self.domains else None
        self.samples = (self.samples + [(power, psutil.cpu_percent(None))])[-self.keep:]
        self.last = (e1, t1)

    def value(self):
        if not self.samples:
            return None, None
        powers = [p for p, _ in self.samples if p is not None]
        return ((sum(powers) / len(powers)) if powers else None,
                sum(c for _, c in self.samples) / len(self.samples))


# ------------------------------------------------------------------ perf
def parse_perf(path):
    """Reads `perf stat -I 1000 -x,` output (same as collect_data.py).
    Returns {column name: [s1..s5]} (None if the counter isn't supported)
    and the last full second found."""
    per_sec = {col(e): [0.0] * WINDOW_S for e in PERF_EVENTS}
    supported, last_sec = set(), 0
    try:
        lines = open(path).read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        try:
            t, value = float(p[0]), float(p[1])
        except ValueError:                    # "<not counted>" / "<not supported>"
            continue
        event = p[3]
        if "/" in event:                      # Intel hybrid CPUs: cpu_core/cycles/
            event = event.split("/")[-2]
        event = col(event.split(":")[0])
        sec = int(round(t))
        if event in per_sec and 1 <= sec <= WINDOW_S:
            per_sec[event][sec - 1] += value
            supported.add(event)
            last_sec = max(last_sec, sec)
    per_sec = {e: (v if e in supported else [None] * WINDOW_S) for e, v in per_sec.items()}
    return per_sec, last_sec


def temp_csv():
    """An empty temporary file for perf's output (closed, so perf can write to it)."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    return path


def perf_check():
    """Checks that perf works, which counters exist, and if --interval-count is known."""
    out = temp_csv()
    try:
        r = subprocess.run(["perf", "stat", "-x,", "-e", ",".join(PERF_EVENTS), "-o", out,
                            "--", "sleep", "0.1"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        sys.exit("perf is not installed. Run install.sh, or: sudo apt install linux-tools-generic")
    if r.returncode != 0:
        sys.exit("perf doesn't work:\n" + r.stderr.strip()[:500] + "\nTry running with sudo.")
    text = open(out).read()
    os.remove(out)
    missing = [e for e in PERF_EVENTS
               if not any(f",{e}," in line or f"/{e}/" in line for line in text.splitlines()
                          if "<not" not in line)]
    r2 = subprocess.run(["perf", "stat", "-I", "100", "--interval-count", "1", "-x,",
                         "-e", "task-clock", "--", "sleep", "0.2"],
                        capture_output=True, text=True, timeout=20)
    return missing, r2.returncode == 0


# ------------------------------------------------------------------ the program's processes
def sample_processes(procs, acc):
    """One sample of the program's processes (same as collect_data.py)."""
    rss = threads = 0
    for c in procs:
        try:
            with c.oneshot():
                io = c.io_counters()
                rss += c.memory_info().rss
                threads += c.num_threads()
            for key, val in (("read_chars", io.read_chars), ("write_chars", io.write_chars)):
                acc[key][c.pid] = max(acc[key].get(c.pid, 0), val)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
            pass
    acc["rss_max"] = max(acc["rss_max"], rss)
    acc["threads_max"] = max(acc["threads_max"], threads)


def family(proc, include_self):
    try:
        kids = proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    return ([proc] if include_self else []) + kids


def sample_for_window(target, include_self, done):
    """Samples the program's processes every 0.25 s for 5 s (while perf counts).
    `done()` says if the program has ended. Returns the samples and how long it ran."""
    acc = {"read_chars": {}, "write_chars": {}, "rss_max": 0, "threads_max": 0}
    t0 = time.monotonic()
    while time.monotonic() - t0 < WINDOW_S and not done():
        sample_processes(family(target, include_self), acc)
        time.sleep(0.25)
    return acc, time.monotonic() - t0


def build_raw(per_sec, acc, temp, idle_power, background):
    return {
        "per_sec": per_sec,
        "cpu_temp_c": temp,
        "idle_power_w": idle_power,
        "background_cpu_pct": background,
        "threads_max": acc["threads_max"],
        "mem_rss_max_mb": acc["rss_max"] / 2**20,
        "io_read_chars": sum(acc["read_chars"].values()),
        "io_write_chars": sum(acc["write_chars"].values()),
    }


def wait_for_perf(out, until_sec, still_running, timeout=3.0):
    """perf writes one block of lines per second; wait until second 5 is there."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        per_sec, last = parse_perf(out)
        if last >= until_sec or not still_running():
            return per_sec, last
        time.sleep(0.2)
    return parse_perf(out)


# ------------------------------------------------------------------ the person using the desktop
def desktop_user():
    """The normal user (not root), to show popups and open the browser on their screen."""
    name = os.environ.get("SUDO_USER") or os.environ.get("POWERSAVER_USER")
    if name and name != "root":
        try:
            return pwd.getpwnam(name)
        except KeyError:
            pass
    if os.geteuid() != 0:
        return pwd.getpwuid(os.geteuid())
    for d in sorted(glob.glob("/run/user/*")):          # e.g. started by systemd at boot
        try:
            uid = int(os.path.basename(d))
            if uid >= 1000:
                return pwd.getpwuid(uid)
        except (ValueError, KeyError):
            pass
    return None


def as_user(cmd):
    """Runs a command as the desktop user (with their screen), even if we are root."""
    pw = desktop_user()
    if os.geteuid() != 0 or pw is None or pw.pw_uid == 0:
        return cmd
    uid, run = pw.pw_uid, f"/run/user/{pw.pw_uid}"
    env = {"HOME": pw.pw_dir, "USER": pw.pw_name, "LOGNAME": pw.pw_name,
           "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
           "XDG_RUNTIME_DIR": run, "DBUS_SESSION_BUS_ADDRESS": f"unix:path={run}/bus",
           "DISPLAY": os.environ.get("DISPLAY", ":0")}
    if os.path.exists(f"{run}/wayland-0"):
        env["WAYLAND_DISPLAY"] = "wayland-0"
    xauth = os.environ.get("XAUTHORITY") or next(iter(
        glob.glob(f"{run}/.mutter-Xwaylandauth.*") + glob.glob(f"{run}/gdm/Xauthority")), None)
    if xauth:
        env["XAUTHORITY"] = xauth
    return (["setpriv", f"--reuid={uid}", f"--regid={pw.pw_gid}", "--init-groups", "env", "-i"]
            + [f"{k}={v}" for k, v in env.items()] + cmd)


def open_url(url):
    try:
        subprocess.Popen(as_user(["xdg-open", url]), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except OSError:
        pass


def popup(title, body, url=None):
    """Desktop notification. With `url`, it gets an 'Open details' button that opens
    the website (used in watch mode, which keeps running to handle the click)."""
    base = ["notify-send", "--app-name=Power-saver check", "--icon=battery", title, body]
    if url is None:
        try:
            subprocess.run(as_user(base), capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return

    def run():
        try:
            r = subprocess.run(as_user(base[:1] + ["--action=open=Open details", "--wait"]
                                       + base[1:]), capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                if r.stdout.strip() == "open":
                    open_url(url)
                return
        except (OSError, subprocess.TimeoutExpired):
            return
        try:                                   # older notify-send without buttons
            subprocess.run(as_user(base), capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass

    threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------ prediction and output
def website_link(name, features, missing):
    q = {"app": name}
    for k in FEATURES:
        v = features.get(k)
        q[k] = "nan" if v is None else f"{v:.6g}"
    if missing:
        q["missing"] = ",".join(missing)
    return WEBSITE_URL.rstrip("/") + "/?" + urllib.parse.urlencode(q)


def report(name, raw, args, watch_mode):
    feats = compute_features(raw)
    missing = [k for k in FEATURES if feats[k] is None]
    link = website_link(name, feats, missing)
    print(f"\n=== {name} (first {WINDOW_S} seconds) ===")
    print(f"CPU speed: {feats['cpu_freq_mhz_mean'] or 0:.0f} MHz | "
          f"cores busy: {(feats['task_clock_mean'] or 0) / 1000:.1f} | "
          f"threads: {feats['threads_max']}")
    try:
        import model as m
        mdl = m.load_model()
        filled, _ = m.fill_missing(feats)
        energy, extra = m.predict(mdl, filled)
        label, emoji, why = m.verdict(energy, extra)
        title = f"{name}: {emoji} {label}"
        body = f"Energy saved: {energy:.1f} %\nExtra time: {extra:+.1f} %\n{why}"
        print(f"Energy saved: {energy:.1f} %\nExtra time:   {extra:+.1f} %")
        print(f"Power-saving mode: {emoji} {label}. {why}")
    except Exception as e:                      # e.g. scikit-learn not installed
        title, body = f"{name}: measured", "Click to see the prediction on the website."
        print("(No local prediction:", e, ")")
    if missing:
        print("Couldn't measure (training average used):", ", ".join(missing))
        if "cpu_freq_mhz_mean" in missing:
            print("!! CPU speed is missing, so the prediction is NOT reliable.")
    print("Details:", link)
    if "<your-app>" in WEBSITE_URL:             # website not deployed yet
        print("(The website link isn't set yet: edit WEBSITE_URL in check_app.py or use --url.)")
        if not args.no_popup or watch_mode:
            popup(title, body)
        return
    if watch_mode:                              # popup with a button to the website
        popup(title, body, link)
        return
    if not args.no_popup:
        popup(title, body)
    if not args.no_browser:
        open_url(link)


# ------------------------------------------------------------------ mode 1: one program
def run_one(cmd, args, domains, interval_count_ok):
    name = os.path.basename(cmd[0])
    print(f"Measuring the computer before starting {name} (3 s)...")
    temp = cpu_temp()
    idle_power, background = measure_idle(domains, 3)

    out = temp_csv()
    perf_cmd = (["perf", "stat", "-I", "1000"]
                + (["--interval-count", str(WINDOW_S)] if interval_count_ok else [])
                + ["-x,", "-e", ",".join(PERF_EVENTS), "-o", out, "--"]
                + (cmd if args.as_root else as_user(cmd)))
    print(f"Starting {name} and measuring its first {WINDOW_S} seconds...")
    proc = subprocess.Popen(perf_cmd)
    acc, ran_for = sample_for_window(psutil.Process(proc.pid), False,
                                     done=lambda: proc.poll() is not None)
    per_sec, last = wait_for_perf(out, WINDOW_S, lambda: proc.poll() is None)
    if last == 0 and proc.poll() is None:
        print("perf hasn't written its numbers yet; waiting for the program to finish...")
        proc.wait()
        per_sec, last = parse_perf(out)

    if ran_for < WINDOW_S - 0.5:
        print(f"\n{name} finished in {ran_for:.1f} s: too short to predict (the model needs "
              f"{WINDOW_S} s). Power-saver mode makes little difference for such short programs.")
    else:
        report(name, build_raw(per_sec, acc, temp, idle_power, background), args,
               watch_mode=False)
    try:
        if proc.poll() is None:
            print(f"\n{name} is still running (Ctrl+C to stop it).")
        proc.wait()
    except KeyboardInterrupt:
        proc.wait()
    finally:
        if os.path.exists(out):
            os.remove(out)


# ------------------------------------------------------------------ mode 2: watch
def watch(args, domains, interval_count_ok):
    names = set(args.watch_list.split(",")) if args.watch_list else WATCH_LIST
    print("Watch mode: waiting for one of these programs to start (Ctrl+C to stop):")
    print("  " + ", ".join(sorted(names)))
    seen = set(psutil.pids())
    idle = IdleMonitor(domains)
    try:
        while True:
            idle.tick()
            for p in psutil.process_iter(["pid", "name", "ppid", "create_time"]):
                if p.pid in seen:
                    continue
                seen.add(p.pid)
                if p.info["name"] not in names or time.time() - p.info["create_time"] > 1.0:
                    continue
                try:                              # skip children of a program already measured
                    if psutil.Process(p.info["ppid"]).name() in names:
                        continue
                except psutil.NoSuchProcess:
                    pass
                watch_one(p, args, idle, interval_count_ok)
                seen.update(psutil.pids())        # ignore what started during the measurement
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")


def watch_one(p, args, idle, interval_count_ok):
    name = p.info["name"].replace(".bin", "")
    print(f"\n{name} started (pid {p.pid}), measuring {WINDOW_S} seconds...")
    temp = cpu_temp()
    idle_power, background = idle.value()
    out = temp_csv()
    perf_cmd = (["perf", "stat", "-I", "1000"]
                + (["--interval-count", str(WINDOW_S)] if interval_count_ok else [])
                + ["-x,", "-e", ",".join(PERF_EVENTS), "-o", out, "-p", str(p.pid)])
    perf = subprocess.Popen(perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    acc, ran_for = sample_for_window(p, True, done=lambda: not p.is_running())
    if perf.poll() is None:
        try:
            perf.wait(timeout=2)
        except subprocess.TimeoutExpired:
            perf.send_signal(2)                   # Ctrl+C: perf writes what it has and stops
            perf.wait()
    per_sec, _ = parse_perf(out)
    os.remove(out)
    if ran_for < WINDOW_S - 0.5:
        print(f"{name} finished in {ran_for:.1f} s: too short to predict.")
        return
    report(name, build_raw(per_sec, acc, temp, idle_power, background), args, watch_mode=True)


# ------------------------------------------------------------------ main
def main():
    global WEBSITE_URL
    ap = argparse.ArgumentParser(
        description="Is power-saver mode worth it for this program? (measures its first 5 s)",
        usage="sudo %(prog)s [options] program [arguments]\n       sudo %(prog)s --watch")
    ap.add_argument("--watch", action="store_true",
                    help="wait in the background for programs from the watch list")
    ap.add_argument("--watch-list", help="comma-separated process names for --watch")
    ap.add_argument("--url", help=f"website link (default: {WEBSITE_URL})")
    ap.add_argument("--no-popup", action="store_true", help="no desktop notification")
    ap.add_argument("--no-browser", action="store_true", help="don't open the website")
    ap.add_argument("--as-root", action="store_true",
                    help="run the program as root (default: as the normal user)")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="program to measure")
    args = ap.parse_args()
    if args.url:
        WEBSITE_URL = args.url
    if not args.watch and not args.command:
        ap.print_help()
        sys.exit(1)
    if os.geteuid() != 0:
        print("Note: not running as root. perf and power readings usually need sudo.")

    missing, interval_count_ok = perf_check()
    if missing:
        print("perf can't count:", ", ".join(missing))
        if "cycles" in missing:
            print("!! Without 'cycles' there is no CPU speed, the model's main input.\n"
                  "   In VMware: power off the VM, VM Settings > Processors >\n"
                  "   'Virtualize CPU performance counters', then start it again.")
    domains = find_rapl_domains()
    if not domains:
        print("No RAPL power readings (normal in a virtual machine or on AMD): "
              "the training average is used for idle power.")

    if args.watch:
        watch(args, domains, interval_count_ok)
    else:
        run_one(args.command, args, domains, interval_count_ok)


if __name__ == "__main__":
    main()
