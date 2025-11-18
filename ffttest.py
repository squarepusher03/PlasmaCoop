from datetime import datetime

import numpy as np
from scipy import fft
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram
from scipy.signal.windows import hann

SCHB_FS = 16_384

sfn = './mms_data/mms1/scm/brst/l2/schb/2020/09/02/mms1_scm_brst_l2_schb_20200902062933_v2.2.0.cdf'
if __name__ == "__main__":
	df = pd.read_pickle('.cache/scm/schb/20200902062933.pkl')
	df = df.dropna().reset_index(drop=True)
	
	start = datetime(2020, 9, 2, 6, 30, 55)
	end = start + pd.Timedelta(seconds=1)
	df = df[(df['time'] > start) & (df['time'] < end)]
	
	bz = df['Bz']
	bz = bz - bz.mean()
	time = df['time']
	
	var_time = np.var(bz)
	
	# Length and window
	n = len(bz)
	# Apply a Hann window (FFT-oriented, periodic form) to reduce spectral leakage
	# Use SciPy's hann(..., sym=False) to match periodogram's default
	w = hann(n, sym=False)
	bz_win = bz * w

	# FFT and frequency axis (one-sided)
	bz_fft = fft.rfft(bz_win)
	bz_fft_freq = np.fft.rfftfreq(n, d=1 / SCHB_FS)  # Hz

	# Hann window power normalization for PSD (units: nT^2/Hz)
	U = np.sum(w ** 2)  # window power of Hann
	bz_psd = (np.abs(bz_fft) ** 2) / (SCHB_FS * U)

	# One-sided correction (conserve variance)
	if n % 2 == 0:
		# even n: DC at 0, Nyquist present at last index
		if bz_psd.size > 2:
			bz_psd[1:-1] *= 2
	else:
		# odd n: no Nyquist bin
		if bz_psd.size > 1:
			bz_psd[1:] *= 2

	# Frequency mask — keep only (0, 2000] Hz and apply to BOTH arrays
	mask = (bz_fft_freq > 0) & (bz_fft_freq <= 2000)
	f_sel = bz_fft_freq[mask]
	P_sel = bz_psd[mask]

	# Plot (convert Hz to kHz for x-axis)
	plt.ylabel(r'Power $\left(\frac{\text{nT}^2}{\text{Hz}}\right)$')
	plt.xlabel('Frequency (Hz)')
	plt.yscale('log')
	plt.title(r'$B_z$ Power Spectrum')
	plt.plot(f_sel, P_sel)
	plt.grid(True)
	plt.show()

	# Integrate PSD over the selected band to get band-limited variance (nT^2)
#	total_w = np.trapezoid(bz_psd, bz_fft_freq)
#
#	# Discrete rectangle-rule (explicit endpoint half-weights) — should match trapz
#	if bz_fft_freq.size > 1:
#		df = bz_fft_freq[1] - bz_fft_freq[0]
#	else:
#		df = SCHB_FS / n
#	var_rect = df * (bz_psd[0] / 2.0 + bz_psd[1:-1].sum() + bz_psd[-1] / 2.0)
#
#	# SciPy reference periodogram configured to match our manual construction
#	f_ref, P_ref = periodogram(
#		bz.values if hasattr(bz, 'values') else np.asarray(bz),
#		fs=SCHB_FS,
#		# pass the exact same window array to ensure elementwise agreement
#		window=w,
#		detrend=False,           # already removed mean above
#		scaling='density',
#		return_onesided=True,
#	)
	# Use np.trapezoid if available (NumPy >= 2.0), else fallback to np.trapz
#	var_ref = np.trapezoid(P_ref, f_ref) if hasattr(np, 'trapezoid') else np.trapz(P_ref, f_ref)

	# Optional: elementwise comparison to detect subtle construction mismatches
#	psd_match = np.allclose(bz_psd, P_ref, rtol=1e-6, atol=0) and np.allclose(bz_fft_freq, f_ref)

	# Window-weighted time variance should match the PSD integrals for a single windowed record
#	x_arr = bz.values if hasattr(bz, 'values') else np.asarray(bz)
#	var_time_weighted = np.sum((x_arr ** 2) * (w ** 2)) / np.sum(w ** 2)

#	print('time variance        =', var_time)
#	print('your integral trapz  =', total_w)
#	print('your integral rect   =', var_rect)
#	print('scipy integral trapz =', var_ref)
#	print('ratios: your/time =', total_w / var_time_weighted,
#	      '  rect/time =', var_rect / var_time_weighted,
#	      '  scipy/time =', var_ref / var_time_weighted)
#	print('weighted variance    =', var_time_weighted,
#	      '  weighted/time =', total_w / var_time_weighted)
#	print('manual_psd_matches_scipy =', psd_match)
