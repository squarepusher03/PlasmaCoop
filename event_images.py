"""
Event images: every requested probe stacked on one time axis, one PNG per
group of selected events, per field. Written next to the MMS<n> folders:

  <request folder>/event_images/<B|E>/MMS<probe>_<start>.png

Which images (Graham, 2026-09-14):
  * Events of one field whose start times lie within SAME_START_S (5 s) of an
    image's first event share that image. It is named after that first event
    and runs to the latest end among its events.
  * If one event in such a group lasts at least LONG_RATIO (2x) as long as the
    next longest, it ALSO gets an image of its own over its whole duration,
    showing everything the probes recorded inside it; events running past its
    end are cut off there. The group's image is still written without it: it
    ends where the next longest ends and is named after its earliest remaining
    event, so the two names cannot collide.
  * Rows: every requested probe, MMS1 at the top, each drawing that probe's
    FFTs across the whole image window. A probe with nothing there gets a
    'No data' row. The f_ce line is drawn when the request has it on.

Drawn from the same pieces graph.png uses (the grids event_export.prepare_export
bins, ShortIntervalPlotter._draw_onto, event_export's title) on a bare Figure,
so it is safe off the main thread. The FFTs for every image window must already
be computed: pass fft_windows(...) to event_export.prepare_export.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

import event_export as ee

IMAGES_DIR   = 'event_images'
SAME_START_S = 5.0
LONG_RATIO   = 2.0
ROW_HEIGHT_IN = 3.2


@dataclass(frozen=True)
class ImagePlan:
	field: str
	anchor: ee.Event            # the event the image is based on, and named after
	start: datetime
	end: datetime
	kind: str                   # 'group' or 'long'
	members: Tuple[ee.Event, ...]

	@property
	def name(self):
		return f'MMS{self.anchor.probe}_{ee.name_ts(self.anchor.start)}.png'


def _plans_for_group(fld, group, field_events, long_ratio):
	by_length = sorted(group, key=lambda e: (-e.duration_s, e.start, e.probe))
	if len(group) >= 2 and by_length[0].duration_s >= long_ratio * by_length[1].duration_s:
		long = by_length[0]
		rest = [e for e in group if e is not long]
		inside = tuple(e for e in field_events if e.start < long.end and e.end > long.start)
		return [ImagePlan(fld, long, long.start, long.end, 'long', inside),
		        ImagePlan(fld, rest[0], rest[0].start, max(e.end for e in rest), 'group', tuple(rest))]
	return [ImagePlan(fld, group[0], group[0].start, max(e.end for e in group), 'group', tuple(group))]


def plan_images(events, same_start_s=SAME_START_S, long_ratio=LONG_RATIO):
	"""Every ImagePlan for the selected events, ordered by field then start."""
	plans = []
	for fld in ee.FIELDS:
		# By start only, and stable: events sharing a start keep the order they
		# came in, so the event that started a grouped row (listed first by
		# EventGroup.as_events) names the image, just as it would ungrouped.
		field_events = sorted((e for e in events if e.field == fld), key=lambda e: e.start)
		group = []
		for ev in field_events:
			if group and (ev.start - group[0].start).total_seconds() <= same_start_s:
				group.append(ev)
				continue
			if group:
				plans.extend(_plans_for_group(fld, group, field_events, long_ratio))
			group = [ev]
		if group:
			plans.extend(_plans_for_group(fld, group, field_events, long_ratio))
	plans.sort(key=lambda p: (ee.FIELDS.index(p.field), p.start, p.anchor.probe))
	return plans


def fft_windows(plans, probes):
	"""{(field, probe): [(start, end, False), ...]}: every probe over every
	image window -- the extra FFTs the images need, in prewarm's window form."""
	out = {}
	for plan in plans:
		for probe in probes:
			out.setdefault((plan.field, probe), []).append((plan.start, plan.end, False))
	return out


def draw_image(path, plan, probes, settings, plotter, log=print, binned=None):
	"""Draw one plan to `path`. Returns the path, or None (nothing written) when
	no probe has any FFT rows in the window. binned: the request's
	event_export.BinnedFfts (default: the last prepare_export's), which must hold
	every probe over the plan's window (fft_windows)."""
	store = ee.prepared_ffts(binned)
	rows = len(probes)
	fig = Figure(figsize=(ee.CHUNK_FIG_W, ROW_HEIGHT_IN * rows + 1.2), layout='constrained')
	FigureCanvasAgg(fig)
	axes = fig.subplots(nrows=rows, ncols=1, squeeze=False)
	span = (plan.end - plan.start).total_seconds()
	last = None
	for row, probe in enumerate(probes):
		ax = axes[row][0]
		grid = store.get(plan.field, probe, plan.start, plan.end)
		if grid is None:
			ax.text(0.5, 0.5, f'No data\nMMS{probe}', ha='center', va='center', transform=ax.transAxes)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_ylabel(f'MMS{probe}')
			continue
		mean_g, time_bins, ye, bins, n_chan, vmi, vma, dd_used = grid
		fce = None
		if settings.fce:
			fce = ee.collect_fce(ee.DATA_DIR, ee.PICKLE_ROOT, plan.start, plan.end, probe=probe, download=False)
		last = plotter._draw_onto(
			ax=ax, mean_g=mean_g, time_bins=time_bins, ye=ye, bins=bins, n_chan=n_chan,
			vmi=vmi, vma=vma, key=plan.field, pad=ee.PAD, start=plan.start, display_dt=dd_used,
			is_top_row=False, is_bottom_row=(row == rows - 1), is_left_col=True, probe=probe, fce=fce,
		)
		if span > 0:
			ax.set_xlim(0, span)
	if last is None:
		return None
	axes[-1][0].set_xlabel(f'Seconds since {ee.label_ts(plan.start)} UTC')
	end_txt = ee.label_ts(plan.end) if plan.end.date() != plan.start.date() else ee.label_ts(plan.end)[11:]
	basis = ('its whole duration' if plan.kind == 'long'
	         else f'events starting within {SAME_START_S:g} s of it')
	fig.suptitle(f'{ee._figure_title(plan.field)}\n{ee.label_ts(plan.start)} – {end_txt} UTC   '
	             f'(based on MMS{plan.anchor.probe}: {basis})')
	cbar = fig.colorbar(last, ax=axes, location='right', shrink=0.9)
	cbar.formatter = FuncFormatter(lambda x, pos: f'{int(np.log10(x))}')
	fig.savefig(path, dpi=150, bbox_inches='tight', format='png')     # format: path may be a .partial
	return path


def write_event_images(settings, plans, probes, cancelled=lambda: False, log=print, binned=None):
	"""Draw every plan into <request folder>/event_images/<field>/. Stops between
	images once `cancelled()` is true. Returns [(plan, path or None), ...] for
	the plans it finished. binned: as draw_image.

	Each image is drawn as a .partial and moved into place once whole; one
	being drawn when the cancel comes is deleted (an older copy, if any, stays)
	and left out of the result (Graham, 2026-11-24)."""
	plotter = ee.ShortIntervalPlotter(data_dir=ee.DATA_DIR, pickle_root=ee.PICKLE_ROOT)
	probes = sorted(probes)
	done = []
	for plan in plans:
		if cancelled():
			break
		folder = os.path.join(settings.top_dir, IMAGES_DIR, plan.field)
		os.makedirs(folder, exist_ok=True)
		final = os.path.join(folder, plan.name)
		part = ee._partial(final)
		try:
			path = draw_image(part, plan, probes, settings, plotter, log=log, binned=binned)
			if path is None or cancelled():
				ee._drop_partials([part])
				if path is not None:
					log(f'  event image {plan.field} {plan.name}: cancelled while it was being drawn; deleted')
					break
			else:
				ee._commit_partials([(part, final)])
				path = final
		except BaseException:
			ee._drop_partials([part])
			raise
		log(f'  event image {plan.field} {plan.name} ({plan.kind}): '
		    + ('no FFT rows on any probe -- not written' if path is None else path))
		done.append((plan, path))
	return done
