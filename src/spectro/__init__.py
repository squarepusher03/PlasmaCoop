"""
spectro — low-coupling spectrogram pipeline carved out of ffttest.py.

Layer 0 (pure values / specs, no I/O):
	instrument_spec   per-instrument constants and dispatch
	cache_layout      on-disk path schema for pickles and FFT results
	freq_grid         Df grid math and rfft-bin snapping

Layers 1-5 follow in subsequent commits per the design PRD at
MEMORY/WORK/20260512-190743_low-coupling-fft-architecture/PRD.md.
"""
