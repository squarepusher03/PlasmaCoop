"""Guard: every call to a watched helper matches that helper's real signature.

Why this exists. Removing the frequency ceiling from the FFT compute/cache chain
changed several signatures, and one live caller kept passing `lpf_lim=` to a
function that no longer took it. `py_compile` cannot see a stale keyword
argument, and no unit test exercised that path, so 115 tests passed with a
guaranteed TypeError sitting in the middle of the plotting pipeline. Only a
manual grep found it.

So this checks the source rather than the behaviour: it parses every module in
the project, finds each call to one of the watched functions, and binds the
keywords it passes against the parameters the function actually declares. It is
deliberately keyword-only -- positional counts are ambiguous for methods (`self`
is implicit at the call site) and keywords are where this class of bug lives.

functools.partial is handled too: `partial(fn, key=...)` passes its keywords
through to fn, and one of the call sites the refactor changed had exactly that
shape.
"""
import ast
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLBACKEND', 'Agg')

import ffttest
from short_interval import ShortIntervalPlotter

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name as written at the call site -> the object it resolves to
WATCHED = {
	'_fft_pickle_path':      ffttest._fft_pickle_path,
	'_compute_file_ffts':    ffttest._compute_file_ffts,
	'_compute_and_save_fft': ffttest._compute_and_save_fft,
	'get_or_compute_fft':    ffttest.get_or_compute_fft,
	'collect_fft_data':      ffttest.collect_fft_data,
	'_segments_from_pickle': ffttest._segments_from_pickle,
	# The direct-CDF readers that replaced the data pickles.
	'_cdf_to_df':            ffttest._cdf_to_df,
	'read_burst_data':       ffttest.read_burst_data,
	'_segments_from_cdf':    ffttest._segments_from_cdf,
	'_segments_via_sidecar': ffttest._segments_via_sidecar,
	'_good_time_bounds':     ffttest._good_time_bounds,
	'find_local_cdfs':       ffttest.find_local_cdfs,
	'check_cdfs':            ffttest.check_cdfs,
	'_bin_fft_files':        ShortIntervalPlotter._bin_fft_files,
	'_bin_fft_dataframe':    ShortIntervalPlotter._bin_fft_dataframe,
	'_write_bins_readme_for': ShortIntervalPlotter._write_bins_readme_for,
	'_analytic_log_bins':    ShortIntervalPlotter._analytic_log_bins,
}

# Directories whose modules are part of the project (slop/ is a dead copy).
SCAN_DIRS = [PROJECT, os.path.join(PROJECT, 'src', 'spectro'), os.path.join(PROJECT, 'tests')]


def _accepted(fn):
	"""Parameter names fn accepts by keyword, or None if it takes **kwargs."""
	params = inspect.signature(fn).parameters
	if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
		return None
	return {name for name, p in params.items()
	        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
	                      inspect.Parameter.KEYWORD_ONLY)}


def _called_name(node):
	"""The bare name a Call node targets: f(...) or obj.f(...)."""
	func = node.func
	if isinstance(func, ast.Name):
		return func.id
	if isinstance(func, ast.Attribute):
		return func.attr
	return None


def _project_files():
	for folder in SCAN_DIRS:
		if not os.path.isdir(folder):
			continue
		for entry in sorted(os.listdir(folder)):
			if entry.endswith('.py'):
				yield os.path.join(folder, entry)


def _call_sites():
	"""(path, lineno, watched_name, [keyword names]) for every watched call."""
	for path in _project_files():
		with open(path, encoding='utf-8') as fh:
			source = fh.read()
		try:
			tree = ast.parse(source, filename=path)
		except SyntaxError:                      # not ours to police
			continue
		for node in ast.walk(tree):
			if not isinstance(node, ast.Call):
				continue
			name = _called_name(node)
			keywords = [kw.arg for kw in node.keywords if kw.arg is not None]

			# partial(fn, **kw) forwards its keywords to fn.
			if name in ('partial', 'partialmethod') and node.args:
				inner = _called_name(ast.Call(func=node.args[0], args=[], keywords=[])) \
					if isinstance(node.args[0], (ast.Name, ast.Attribute)) else None
				if inner in WATCHED:
					yield path, node.lineno, inner, keywords
				continue

			if name in WATCHED:
				yield path, node.lineno, name, keywords


def test_there_are_call_sites_to_check():
	"""If this fails the walker is broken, and every other check here is vacuous."""
	assert len(list(_call_sites())) >= 8


@pytest.mark.parametrize('path, lineno, name, keywords', list(_call_sites()),
                         ids=lambda v: os.path.basename(v) if isinstance(v, str) else str(v))
def test_call_site_keywords_exist_on_the_callee(path, lineno, name, keywords):
	accepted = _accepted(WATCHED[name])
	if accepted is None:
		return
	unknown = [k for k in keywords if k not in accepted]
	assert not unknown, (
		f'{os.path.relpath(path, PROJECT)}:{lineno} calls {name}() with '
		f'{unknown}, which it does not accept. Accepted: {sorted(accepted)}'
	)


def test_the_ceiling_is_absent_from_the_cache_path():
	"""The ceiling must not creep back into anything that decides cache CONTENT.

	It lived there once: pickles were truncated at lpf_lim, which forced the
	ceiling into the cache key and meant raising it rebuilt the whole cache.
	Pickles now span DC to Nyquist and the ceiling is applied where rows are
	binned, so if one of these grows an lpf_lim again, the cache key has to grow
	with it or entries written under two different ceilings collide silently.
	"""
	for name in ('_fft_pickle_path', '_compute_file_ffts', '_compute_and_save_fft',
	             'get_or_compute_fft'):
		assert 'lpf_lim' not in inspect.signature(WATCHED[name]).parameters, (
			f'{name}() takes lpf_lim again -- see _fft_pickle_path.__doc__ before '
			f'putting it back, and bump FFT_PICKLE_VERSION if you do.'
		)


def test_the_ceiling_is_still_present_where_rows_are_reduced():
	"""...and the other half of that contract: the binners must still take it.

	The binners now receive the ceiling inside a BinSpec rather than as a loose
	argument, so what has to hold is that it still REACHES them -- not that it
	arrives under one particular parameter name. collect_fft_data sits on the
	ffttest side of the line and still takes it directly.
	"""
	from short_interval import BinSpec

	assert 'lpf_lim' in inspect.signature(WATCHED['collect_fft_data']).parameters, (
		'collect_fft_data() no longer takes lpf_lim -- the ceiling has to be '
		'applied somewhere, and this is where.'
	)
	assert 'lpf_lim' in BinSpec.__dataclass_fields__, (
		'BinSpec lost lpf_lim -- the binners take their ceiling from it now, so '
		'losing it there silently removes the ceiling from every binning path.'
	)
	for name in ('_bin_fft_files', '_bin_fft_dataframe'):
		assert 'binning' in inspect.signature(WATCHED[name]).parameters, (
			f'{name}() no longer takes a BinSpec -- the ceiling has to be applied '
			f'somewhere, and this is where.'
		)
