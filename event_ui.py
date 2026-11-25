"""
Event-export UI for the MMS spectrogram pipeline.   Run:  python event_ui.py

Pick a UTC range, B/E and probes; list the quality-good events in it; select
any of them; each one is written to
<save folder>/<request folder>/MMS<n>/<field>/<event>/ as ffts.csv + graph.png.
The window is event_window.py; the work is event_export.py.

This file is deliberately almost empty. On Windows every worker process of the
FFT pools starts by re-importing the script that launched the program, so
whatever this file imports at the top is imported again in every worker -- the
lesson recorded in ffttest.py above SAMPLE_RATE. The window and the pipeline are
imported only inside the guard, which workers never run.
"""
import os

if __name__ == '__main__':
	os.environ.setdefault('MPLBACKEND', 'Agg')      # images are saved, never shown
	import multiprocessing
	multiprocessing.freeze_support()
	from event_window import main
	raise SystemExit(main())
