"""Parameter groups passed as one object instead of one argument at a time.

Three frozen dataclasses, split by WHAT EACH ONE INVALIDATES rather than by what
it is nominally "about". That split is the whole point of grouping them:

    FftSpec     decides the CONTENT of an FFT pickle.  Change any field and
                every cached pickle written under the old meaning is wrong --
                so adding a field here means bumping ffttest.FFT_PICKLE_VERSION
                in the same commit, or stale entries stay readable and wrong.
    BinSpec     decides how already-computed rows are reduced for display and
                export.  Touches no cache at all; change it as often as you
                like, nothing is recomputed.
    PowerScale  is purely visual: the colour-scale limits, nothing else.

So "will this force a recompute?" is answerable by looking at which class the
field lives in, instead of by tracing it through the pipeline.

Frozen because these travel into worker processes. A mutable options object
that one worker edits is a bug that only shows up under parallelism; `replace`
(from dataclasses) gives a modified copy instead:

    wide = replace(bins, lpf_lim=4096)

Nothing here imports ffttest or short_interval -- those import this, not the
other way round.
"""
from dataclasses import dataclass
from typing import Optional

try:
	from .instrument_spec import SAMPLE_RATE
except ImportError:                      # flat-path import (spectro on sys.path)
	from instrument_spec import SAMPLE_RATE


# What a binned cell holds when NO FFT row landed in it.
#
# Deliberately NOT part of PowerScale. It was historically the same number as
# the colour-scale floor, which quietly made two unrelated things one: the
# per-event CSV writer decides "did data land in this cell" by comparing against
# it, so making the floor a user-facing display option would have silently
# changed which CSV rows get dropped whenever someone re-scaled a plot. The
# marker is fixed here; the display range moves freely.
EMPTY_FILL = {'B': 1e-6, 'E': 1e-4}

# Colour-scale defaults per field -- the values the pipeline used when the range
# was hardcoded, so leaving PowerScale alone reproduces today's images exactly.
_SCALE_DEFAULTS = {'B': (1e-6, 1e-2), 'E': (1e-4, 10.0)}


def empty_fill_for(key: str) -> float:
	"""The no-data marker for a field key ('B', 'Bz', 'E', 'Ex', ...)."""
	return EMPTY_FILL[key[0]]


@dataclass(frozen=True)
class FftSpec:
	"""How a per-file FFT is computed, and therefore what its cache entry holds.

	pad  -- half-window in seconds; the full window is 2*pad and Df = 1/(2*pad).
	step -- spacing between window CENTRES, in seconds.
	fs   -- the instrument's sample rate. SCM burst and EDP burst are both
	        SAMPLE_RATE; see instrument_spec.BurstRate for why it is 8192 and
	        not the 8191.887 the timestamps actually step at.
	"""
	pad:  float = 0.25
	step: float = 0.05
	fs:   float = SAMPLE_RATE

	@property
	def window_s(self) -> float:
		"""Full FFT window length in seconds."""
		return 2.0 * self.pad

	@property
	def df(self) -> float:
		"""Frequency resolution in Hz -- the grid every bin is snapped onto."""
		return 1.0 / self.window_s

	@property
	def n_expected(self) -> int:
		"""Samples a complete window should hold, at the nominal rate."""
		return int(self.fs * self.window_s)

	@property
	def nyquist(self) -> float:
		return self.fs / 2.0

	@property
	def top_row_hz(self) -> int:
		"""Highest frequency the snapped grid actually emits.

		rfft tops out at fs/2, and _compute_file_ffts rounds every bin onto the
		Df grid, so this is that grid's last row -- 4096 Hz at pad 0.25 and
		fs 8192. It is the Nyquist row itself, which is why the highest ceiling
		worth offering is one step BELOW it (see ceiling_max).
		"""
		return int(round(self.nyquist / self.df) * self.df)

	@property
	def ceiling_max(self) -> int:
		"""Highest usable frequency ceiling: the Nyquist row itself.

		This stopped one step below the Nyquist row while _compute_file_ffts
		doubled an even window's Nyquist bin -- excluding an overstated bin was
		the right call then. That doubling is gone, so the row is ordinary
		correctly-scaled data and there is no reason to withhold it.
		"""
		return self.top_row_hz

	@property
	def cache_token(self) -> str:
		"""Directory name identifying this spec's FFT cache.

		_full records that the pickle spans DC to Nyquist with no ceiling baked
		in. It must never collapse to a bare pad{pad} -- that was the pre-ceiling
		layout, and a truncated legacy entry would be indistinguishable from a
		full-spectrum one on disk.
		"""
		return f'pad{self.pad:.3f}_full'


@dataclass(frozen=True)
class BinSpec:
	"""How computed FFT rows are reduced to a display/export grid.

	lpf_lim    -- frequency ceiling in Hz. Applied where rows are binned, NOT
	              where they are computed, so raising it costs no recompute.
	num_bins   -- log-frequency channel count requested.
	display_dt -- width of one time cell in seconds; None means "use the FFT
	              step", and it is widened automatically for a long window.
	"""
	lpf_lim:    float = 3000
	num_bins:   int = 25
	display_dt: Optional[float] = None

	@property
	def bin_factor(self) -> float:
		"""Ratio between consecutive log-frequency bin edges."""
		return self.lpf_lim ** (1.0 / self.num_bins)


@dataclass(frozen=True)
class PowerScale:
	"""Colour-scale limits for one panel. Purely visual.

	Nothing downstream of the image reads these: they set LogNorm's vmin/vmax
	and nothing else. In particular they do NOT decide what an empty cell holds
	(see EMPTY_FILL) and they do NOT affect the exported numbers.

	Both must be positive -- the spectrogram is drawn on a log colour scale.
	"""
	vmin: float
	vmax: float

	def __post_init__(self):
		if not (self.vmin > 0 and self.vmax > 0):
			raise ValueError('Power limits are drawn on a log scale, so both must be above zero.')
		if self.vmin >= self.vmax:
			raise ValueError(f'The power minimum ({self.vmin:g}) must be below the maximum ({self.vmax:g}).')

	@classmethod
	def for_field(cls, key: str) -> 'PowerScale':
		"""The default range for a field key -- what the pipeline used when this
		range was hardcoded, so the defaults reproduce today's images."""
		return cls(*_SCALE_DEFAULTS[key[0]])

	def as_tuple(self):
		"""(vmin, vmax) -- for the call sites that still pass the pair around."""
		return self.vmin, self.vmax
