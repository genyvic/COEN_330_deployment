"""
app.py - the website (Streamlit).

Run locally:   streamlit run app.py
Hosted on:     Streamlit Community Cloud (see README.md)

Three ways to use it:
  1. Open a link made by check_app.py (Linux): the 23 measurements are in the link.
  2. Pick one of the 5 test apps (works from any computer, Windows included).
  3. Move the sliders to see how the prediction changes.
"""
import os

import pandas as pd
import streamlit as st

import model as m
from features import FEATURES

st.set_page_config(page_title="Power-saver check", page_icon="🔋", layout="centered")

APP_NAMES = {
    "mpv_play_video": "mpv: playing a video",
    "libreoffice_to_pdf": "LibreOffice: converting documents to PDF",
    "7zip_compress": "7-Zip: compressing files",
    "imagemagick_resize": "ImageMagick: resizing photos",
    "blender_render": "Blender: rendering a 3D scene",
}
INSTALL_CMD = ("curl -sSL https://raw.githubusercontent.com/genyvic/COEN_330_deployment/main/"
               "deployment/install.sh | sudo bash")


@st.cache_resource
def get_model():
    return m.load_model()


@st.cache_data
def get_means():
    return m.training_means()


@st.cache_data
def get_examples():
    return pd.read_csv(os.path.join(m.HERE, "examples.csv"))


@st.cache_data
def get_ranges():
    df = pd.read_csv(m.TRAIN_CSV)
    return df[FEATURES].min().to_dict(), df[FEATURES].max().to_dict()


def show_result(energy, time, actual=None):
    """The result box: 2 numbers + the answer."""
    label, emoji, why = m.verdict(energy, time)
    c1, c2 = st.columns(2)
    c1.metric("Energy saved", f"{energy:.1f} %")
    c2.metric("Extra time", f"{time:+.1f} %")
    box = {"✅": st.success, "⚖️": st.warning, "❌": st.error}[emoji]
    box(f"**Power-saving mode: {emoji} {label}**  \n{why}")
    st.caption(f"Typical error of the model on new programs: about ±{m.TEST_MAE_ENERGY} points "
               f"for energy and ±{m.TEST_MAE_TIME} points for extra time.")
    if actual is not None:
        real_label, real_emoji, _ = m.verdict(*actual)
        st.markdown(f"**Real result (measured):** energy saved {actual[0]:.1f} %, "
                    f"extra time {actual[1]:+.1f} % → {real_emoji} {real_label}")


def key_numbers(f):
    """The few measurements that matter most, in plain words."""
    st.markdown(
        f"- **CPU speed:** {f['cpu_freq_mhz_mean']:.0f} MHz "
        f"(the model's most important input: power-saver mostly lowers this)\n"
        f"- **CPU cores busy:** {f['task_clock_mean'] / 1000:.1f} (CPU time per second)\n"
        f"- **Threads:** {f['threads_max']:.0f}\n"
        f"- **Instructions per cycle:** {f['instructions_per_cycle']:.2f}")


st.title("🔋 Is power-saver mode worth it?")
st.write("From the **first 5 seconds** of a program, this predicts how much energy Ubuntu's "
         "power-saver mode would save, and how much slower the program would get.")

model = get_model()

# ------------------------------------------------------------------ 1. link from check_app.py
params = st.query_params
if all(k in params for k in FEATURES):
    measured = {}
    for k in FEATURES:
        try:
            measured[k] = float(params[k])
        except ValueError:
            measured[k] = None
    filled, missing = m.fill_missing(measured, get_means())
    missing += [k for k in params.get("missing", "").split(",") if k and k not in missing]
    name = params.get("app", "your program")

    st.subheader(f"Your program: {name}")
    energy, time = m.predict(model, filled)
    show_result(energy, time)
    if missing:
        st.info("Some values couldn't be measured on this computer (for example power and "
                "temperature inside a virtual machine), so the training average was used: "
                + ", ".join(f"`{k}`" for k in missing)
                + (". **CPU speed is missing, so this prediction is not reliable.**"
                   if "cpu_freq_mhz_mean" in missing else ""))
    with st.expander("What was measured"):
        key_numbers(filled)
        st.dataframe(pd.DataFrame({"value": filled}), use_container_width=True)
    st.divider()

# ------------------------------------------------------------------ 2. examples / own program / about
tab_ex, tab_own, tab_about = st.tabs(["Try an example", "Measure your own program", "About"])

with tab_ex:
    st.write("These 5 programs were measured on our Linux laptop but **never used to train** "
             "the model, so you can compare the prediction with what really happened.")
    ex = get_examples()
    app = st.selectbox("Program", list(APP_NAMES), format_func=APP_NAMES.get)
    run = st.radio("Run", [1, 2, 3], horizontal=True,
                   help="Each program was measured 3 times.")
    row = ex[(ex["workload"] == app) & (ex["run"] == run)].iloc[0]
    f = {k: float(row[k]) for k in FEATURES}

    energy, time = m.predict(model, f)
    show_result(energy, time, actual=(row["energy_saved_pct"], row["extra_time_pct"]))
    with st.expander("What was measured"):
        key_numbers(f)

    with st.expander("Change the measurements and see what happens"):
        lo, hi = get_ranges()
        f2 = dict(f)
        f2["cpu_freq_mhz_mean"] = st.slider(
            "CPU speed (MHz)", int(lo["cpu_freq_mhz_mean"]), int(hi["cpu_freq_mhz_mean"]),
            int(f["cpu_freq_mhz_mean"]), step=50)
        cores = st.slider("CPU cores busy", 0.0, round(hi["task_clock_mean"] / 1000, 1),
                          round(f["task_clock_mean"] / 1000, 1), step=0.1)
        f2["task_clock_mean"] = cores * 1000
        f2["threads_max"] = st.slider("Threads", int(lo["threads_max"]),
                                      int(hi["threads_max"]), int(f["threads_max"]))
        # keep the rates consistent with the new CPU time
        f2["context_switch_rate"] = f["context_switches_mean"] / (f2["task_clock_mean"] + 1e-10)
        f2["page_fault_rate"] = f["page_faults_mean"] / (f2["task_clock_mean"] + 1e-10)
        e2, t2 = m.predict(model, f2)
        show_result(e2, t2)

with tab_own:
    st.write("Measuring a program needs **Linux** (the CPU counters come from `perf`), "
             "an **Intel** CPU for power readings, and admin rights (`sudo`).")
    st.markdown("**1. Install** (one time):")
    st.code(INSTALL_CMD, language="bash")
    st.markdown("**2. Check one program:** start it through the tool. It measures the first "
                "5 seconds, shows a popup, and opens this page with the result.")
    st.code("sudo powersaver-check gzip -k big_file.txt", language="bash")
    st.markdown("**Or watch mode:** the tool waits in the background, and every time a "
                "program from its list starts (gzip, blender, ffmpeg...), a popup appears.")
    st.code("sudo powersaver-check --watch", language="bash")
    st.info("**No Linux laptop?** Use a VMware virtual machine with Ubuntu, and turn on "
            "*Virtualize CPU performance counters* (VM Settings → Processors). The README "
            "explains it step by step. In a virtual machine the tool runs, but the "
            "predictions are less accurate (see About).")

with tab_about:
    st.markdown(f"""
**Model:** a Random Forest (400 trees, max depth 5), chosen because it had the lowest
validation and test error of our 5 models. It was trained on **252 runs of 32 programs**,
measured on one laptop (Dell Inspiron 5502, Intel i7-1165G7, Ubuntu, plugged in).

**Accuracy on 5 new programs:** energy saved ±{m.TEST_MAE_ENERGY} points,
extra time ±{m.TEST_MAE_TIME} points (mean absolute error).

**The answer:**

| Answer | Rule |
|---|---|
| ✅ Worth it | energy saved ≥ {m.ENERGY_MIN}% and extra time ≤ {m.TIME_FREE}% |
| ⚖️ Worth it if you're not in a hurry | energy saved ≥ {m.ENERGY_MIN}% and extra time {m.TIME_FREE}–{m.TIME_MAX}% |
| ❌ Not worth it | energy saved < {m.ENERGY_MIN}% or extra time > {m.TIME_MAX}% |

**Limitations:**
- The model learned from one laptop. It is only reliable on that hardware.
- It relies mostly on CPU speed. In a virtual machine, CPU speed comes from the host
  computer, and power/temperature can't be read, so predictions are less accurate.
- It only sees the first 5 seconds. Programs that change behaviour later are harder.
- Small dataset: 32 programs. Programs very different from these may be predicted badly.
""")
