from datetime import datetime
from short_interval import ShortIntervalPlotter
from data_paths import DATA_DIR, PICKLE_ROOT, INTERVAL_MANIFEST  # per-platform E:/ vs /mnt/e

# Windows' multiprocessing 'spawn' start method (used by ShortIntervalPlotter's
# ProcessPoolExecutor/attempt_download machinery) re-imports this script from
# scratch in every worker process. Without this guard, each worker would
# re-run every plotter.plot(...) call below -- including spawning its own
# nested download/FFT worker processes -- instead of just executing the one
# task it was actually given. (Missing since the MMS-REFACTOR baseline
# rewrite; this is what was crashing PyCharm.)
if __name__ == '__main__':
	plotter = ShortIntervalPlotter(
		data_dir=DATA_DIR,
		pickle_root=PICKLE_ROOT,
		manifest_dir=INTERVAL_MANIFEST,
	)

	# ~12-second window around the reconnection event in the paper
	plotter.plot(
		key='B',
		pad=0.1,  # 0.2 s FFT windows, Δf = 5 Hz
		start=datetime(2019, 8, 16, 9, 31, 58),
		end=datetime(2019, 8, 16, 9, 32, 10),
	)

	plotter.plot(
		key='E',
		pad=0.1,
		start=datetime(2019, 8, 16, 9, 31, 58),
		end=datetime(2019, 8, 16, 9, 32, 10),
		lpf_lim=500,  # EDP has a lower useful ceiling
		display_dt=0.02,  # finer time resolution
	)

	plotter.plot(
		key='B',
		pad=0.2,  # 0.4 s windows, Δf = 2.5 Hz — better low-freq resolution
		start=datetime(2019, 8, 16, 9, 30, 0),
		end=datetime(2019, 8, 16, 9, 35, 0),
		display_dt=0.5,  # 0.5 s display bins keeps memory reasonable over 5 min
	)
