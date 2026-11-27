# Power-saver check: deployment

**Is Ubuntu's power-saver mode worth it for this program?** From the first 5 seconds of a
program, our model predicts how much energy power-saver mode would save and how much slower
the program would get, then answers:

```
Energy saved:   15.6 %
Extra time:    +62.0 %
Power-saving mode: ⚖️ Worth it if you're not in a hurry
```

**Website:** https://testcoen330deployment.streamlit.app

It has two parts:

| Part | File | Runs on | What it does |
|---|---|---|---|
| Website | `app.py` | Any browser (hosted on Streamlit Cloud) | Shows the prediction. Has 5 example programs, so it works without Linux. |
| Linux tool | `check_app.py` | Ubuntu (laptop or VMware VM) | Measures a real program's first 5 s, shows a popup, opens the website with the result. |

```
program starts → check_app.py measures 5 s (perf) → 23 features → popup + website link
                                                                    → Random Forest → answer
```

## 1. Use the website (any computer)

Open the link above and pick one of the 5 test programs (mpv, LibreOffice, 7-Zip,
ImageMagick, Blender). These were measured on our laptop but never used for training, so
the page shows the prediction **and** the real measured result.

**Example input/output:** 7-Zip compressing files (run 1) → predicted energy saved 15.6 %,
extra time +62.0 % → ⚖️ *Worth it if you're not in a hurry*. Real result: 13.7 % / +52.8 %,
same answer.

## 2. Measure your own program (Linux)

Needs Ubuntu, `sudo`, and ideally an Intel CPU (for power readings).

**Install** (one time, about 3 minutes):
```bash
curl -sSL https://raw.githubusercontent.com/genyvic/COEN_330_deployment/main/deployment/install.sh | sudo bash
```
Or from a copy of this repo: `sudo bash deployment/install.sh`.
The installer is short: it installs `perf`, Python and `notify-send`, copies the tool to
`/opt/powersaver` and adds the command `powersaver-check`. Read it before running it.

**Check one program:** start it through the tool.
```bash
sudo powersaver-check gzip -k big_file.txt
sudo powersaver-check blender -b scene.blend -f 1
```
The tool measures the computer for 3 s, starts the program (as your normal user, not root),
measures its first 5 seconds, prints the result, shows a popup and opens the website.
The program then keeps running normally.

**Watch mode (use your apps normally, by clicking):** the tool waits in the background and
shows a popup with the answer and an *Open details* button when:
- **an open app starts working hard** (more than 80% CPU for 1 second): for example you
  click *Render* in Blender, export a video, apply a big filter in GIMP, or compress a folder
  with the Archive Manager. It measures the next 5 seconds. (Just opening an app doesn't
  count: loading a window isn't the app's real work.)
- **a program from its list starts** (gzip, xz, 7z, ffmpeg, make...): measured from its
  start, exactly like our data collection.
```bash
sudo powersaver-check --watch
sudo powersaver-check --watch --busy-cpu 150      # only react above 1.5 cores busy
sudo powersaver-check --watch --no-busy           # only react to programs from the list
sudo powersaver-check --watch --watch-list gzip,xz,blender   # your own list
```
Each app gets at most one popup every 2 minutes.

**Always on:** install with `--autostart` (see below) and watch mode starts at every boot and
waits for you to log in. It uses about 1% of the CPU. For accurate predictions, keep the laptop
in the same conditions as our data: **Balanced** power mode (it skips measuring while the
laptop is already in power-saver mode), plugged in, no other heavy programs.
To start watch mode automatically at every boot, install with
`curl -sSL .../install.sh | sudo bash -s -- --autostart`.

Other options: `--no-popup`, `--no-browser`, `--url <website>`, `--as-root`.

## 3. Test in a virtual machine (VMware)

For someone without a Linux laptop:

1. Install **VMware Workstation Pro** (free for personal use). VirtualBox does **not** work:
   it can't give the VM the CPU counters `perf` needs.
2. Create an Ubuntu VM. **Before starting it:** VM Settings → Processors → check
   **Virtualize CPU performance counters**. Without it, `perf` can't count cycles, so the
   CPU speed (the model's main input) can't be measured.
3. In the VM, run the install command above, then `sudo powersaver-check gzip -k <file>`.

In a VM the tool runs, but the **predictions are less accurate**: the CPU speed comes from
the host computer (not our laptop), and power and temperature can't be read (the training
average is used instead; the website says so).

## Files

| File | What it is |
|---|---|
| `app.py` | The website (Streamlit). |
| `check_app.py` | The Linux tool: measuring, popup, link to the website. Uses the same perf counters as `collection/collect_data.py`. |
| `features.py` | Turns 5 s of measurements into the 23 inputs, like the preprocessing notebook. |
| `model.py` | The final Random Forest (400 trees, max depth 5, min 5 samples per leaf) and the 3-class answer. |
| `rf_model.joblib` | The trained model. If it was saved with another scikit-learn version, `model.py` retrains it from `dataset_B_final.csv` (2 s, same settings). |
| `dataset_B_final.csv` | The 252 training rows. |
| `examples.csv` | The 15 test runs (5 programs × 3) shown on the website. |
| `verify.py` + `tests/` | Checks that the features and the model match the report (see below). |
| `install.sh`, `requirements.txt`, `requirements-helper.txt` | Installation. |

## The answer (3 classes)

| Answer | Rule | Why |
|---|---|---|
| ✅ Worth it | energy saved ≥ 5 % and extra time ≤ 15 % | below 15 %, the slowdown is within the model's error |
| ⚖️ Worth it if you're not in a hurry | energy saved ≥ 5 % and extra time 15–100 % | saves energy, up to 2× slower |
| ❌ Not worth it | energy saved < 5 % or extra time > 100 % | saves almost nothing (within the model's error), or more than 2× slower |

## Checking that it matches the report

```bash
pip install -r requirements.txt
python verify.py
```
1. The 23 features computed by `features.py` from the raw test measurements are identical
   to `dataset_B_test_final.csv` (the preprocessing notebook's output).
2. The test MAE is 3.87 / 6.60 (report: 3.85 / 6.64). The small difference comes from
   the scikit-learn version (Colab's was different), which builds slightly different trees.

On the 15 test runs, the answer is right 13 times. The 2 misses are mpv runs whose real
energy saving was negative (−0.3 % and −15.2 %), which no model predicted.

## Run the website on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
2. Choose the repo `genyvic/COEN_330_deployment`, branch `main`, main file `deployment/app.py`.
3. In **App URL**, type `testcoen330deployment`, then **Deploy**.
   (If you choose another name, change `WEBSITE_URL` in `check_app.py` and the top of this README.)

## Limitations

- **One laptop:** the model was trained on a Dell Inspiron 5502 (Intel i7-1165G7). Other
  computers have different CPU speeds, so predictions are only reliable on that hardware.
- **CPU speed matters most:** the model relies mostly on it. In a VM it reflects the host,
  and without performance counters it can't be measured at all.
- **First 5 seconds only:** programs that change behaviour later are harder to predict.
  Programs shorter than 5 s aren't predicted.
- **Watch mode starts about 0.1 s late** and only counts the threads that exist when it
  attaches (simple mode counts everything, like our data collection).
- **"App gets busy" is an approximation:** the model learned from the first 5 s after a
  program *starts*; here it sees the first 5 s after an app *gets busy*, which is similar
  but not identical. Interactive apps with no fixed job (web browsers, games) can't really
  get a meaningful "extra time".
- **Small dataset:** 252 runs of 32 programs. Very different programs may be predicted badly.
- **Linux only** for measuring, and it needs `sudo` (for `perf` and power readings).
