"""Read one row of a SQLite table straight from the drive, past the Windows file cache.

Why (2026-10-29/30): bytes read from E: through the Windows cache have twice come back wrong while the drive held the
right ones (memory: project_validation_verdict_flips). The preprocessed archive replaces the CDFs it reads, so before
a CDF is deleted its archive row must be checked against what the DRIVE holds, not a cached copy. Python's sqlite3
always reads through the cache, so this reads the row's pages itself: the table's b-tree from its root page down to
the row's leaf cell, then the cell's overflow chain, then the record's columns. All of it is SQLite's documented file
format (https://www.sqlite.org/fileformat2.html), unchanged since 2004.

Only the Windows venv can read past the cache (FILE_FLAG_NO_BUFFERING through ctypes). WSL reaches E: through the
Windows cache whatever it asks for (O_DIRECT on /mnt/e is accepted but still served from the cache, measured
2026-10-29), so off Windows this reads the file plainly: the walk is the same, the bytes may be cached ones.

Measured on E: (venv, 20 real archive rows of 0.9-2.9 MB, E:/PlasmaCoop/cache_bypass_study_2026-10-30/): a row's
drive read takes 15 ms median (6-196 ms). The prototype and its proof (every row of four test databases read both
ways and compared, in WSL and in the venv) are _wsl_scratch/sqlite_drive_read.py.

CALLER RULES
  - The database must hold the row in its MAIN file: in WAL mode, checkpoint first. Only the main file is read.
  - Keep a SQLite connection to the file open while calling. Opening an uncached handle while nothing else holds the
    file open makes Windows drop the WHOLE file from its cache (measured 2026-10-30); with a cached handle open it
    drops nothing.
  - The root page comes from the caller's SQL, through the cache. Were it wrong, the walk could not end at a row
    holding the expected bytes, so a wrong root cannot make a bad row pass.

Stdlib only.
"""
import os
import struct

ALIGN = 4096                     # whole sectors on a 512-byte or a 4 KiB-sector drive alike (E: has 512 B sectors)
LOCK_BYTE = 0x40000000           # SQLite's lock-byte page holds no data; Windows locks bytes in it, refusing reads


class DriveFile:
	"""A file read past the Windows file cache (FILE_FLAG_NO_BUFFERING): each read is widened to whole sectors and
	read into page-aligned memory, as that flag requires. Off Windows it reads plainly."""

	def __init__(self, path):
		self.path = path
		if os.name != 'nt':
			self.fh = open(path, 'rb')
			return
		import ctypes
		from ctypes import wintypes
		self.ctypes, self.wintypes = ctypes, wintypes
		k32 = self.k32 = ctypes.WinDLL('kernel32', use_last_error=True)
		k32.CreateFileW.restype = wintypes.HANDLE
		k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
		                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
		k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong),
		                                 wintypes.DWORD]
		k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
		                         wintypes.LPVOID]
		k32.VirtualAlloc.restype = wintypes.LPVOID
		k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
		k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
		k32.CloseHandle.argtypes = [wintypes.HANDLE]
		# GENERIC_READ; share read, write and delete (SQLite keeps the file open); OPEN_EXISTING; FILE_FLAG_NO_BUFFERING
		self.h = k32.CreateFileW(path, 0x80000000, 0x7, None, 3, 0x20000000, None)
		if self.h in (None, wintypes.HANDLE(-1).value):
			raise OSError(ctypes.get_last_error(), 'CreateFileW failed', path)

	def read(self, offset, length):
		if os.name != 'nt':
			self.fh.seek(offset)
			return self.fh.read(length)
		ctypes, k32 = self.ctypes, self.k32
		start = offset - offset % ALIGN
		size = -(-(offset + length - start) // ALIGN) * ALIGN
		buf = k32.VirtualAlloc(None, size, 0x3000, 0x04)       # commit + reserve, read/write: page aligned
		if not buf:
			raise MemoryError(f'VirtualAlloc of {size} bytes failed')
		try:
			if not k32.SetFilePointerEx(self.h, start, None, 0):
				raise OSError(ctypes.get_last_error(), 'SetFilePointerEx failed', self.path)
			got, total = self.wintypes.DWORD(0), 0
			while total < size:
				want = min(size - total, 1 << 30)
				if not k32.ReadFile(self.h, buf + total, want, ctypes.byref(got), None):
					raise OSError(ctypes.get_last_error(), 'ReadFile failed', self.path)
				total += got.value
				if got.value < want:                              # the end of the file
					break
			data = ctypes.string_at(buf, total)
		finally:
			k32.VirtualFree(buf, 0, 0x8000)                       # release
		return data[offset - start:offset - start + length]

	def close(self):
		if os.name != 'nt':
			self.fh.close()
		else:
			self.k32.CloseHandle(self.h)

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		self.close()


# ---------------------------------------------------------------------------
# the SQLite file format, as far as one row of a table needs it
# ---------------------------------------------------------------------------

def _varint(b, i):
	"""(value, next index) of the varint at b[i]: 7 bits a byte, the high bit set on all but the last; a ninth byte
	gives all 8 of its bits."""
	v = 0
	for k in range(8):
		c = b[i + k]
		v = (v << 7) | (c & 0x7F)
		if c < 0x80:
			return v, i + k + 1
	return (v << 8) | b[i + 8], i + 9


def _u16(b, i):
	return (b[i] << 8) | b[i + 1]


def _u32(b, i):
	return int.from_bytes(b[i:i + 4], 'big')


def _key(p, ptr, interior):
	"""The rowid of the cell the cell pointer at p[ptr] points to: an interior cell is (left child, rowid), a leaf cell
	(payload size, rowid, payload)."""
	off = _u16(p, ptr)
	if interior:
		return _varint(p, off + 4)[0]
	return _varint(p, _varint(p, off)[1])[0]


def _first_at_or_above(p, first, ncell, rowid, interior):
	"""Index of the first cell whose rowid is >= rowid, or ncell: a page keeps its cells in rowid order."""
	lo, hi = 0, ncell
	while lo < hi:
		mid = (lo + hi) // 2
		if _key(p, first + 2 * mid, interior) < rowid:
			lo = mid + 1
		else:
			hi = mid
	return lo


def _record(payload):
	"""The column values of a record, in the table's column order: its header (its own length, then a serial type per
	column), then the values."""
	end, j = _varint(payload, 0)
	types = []
	while j < end:
		t, j = _varint(payload, j)
		types.append(t)
	out, k = [], end
	for t in types:
		if t == 0:
			out.append(None)
		elif t <= 6:
			w = (0, 1, 2, 3, 4, 6, 8)[t]
			out.append(int.from_bytes(payload[k:k + w], 'big', signed=True))
			k += w
		elif t == 7:
			out.append(struct.unpack('>d', payload[k:k + 8])[0])
			k += 8
		elif t in (8, 9):
			out.append(t - 8)
		elif t >= 12:
			w = (t - 12) >> 1
			v = bytes(payload[k:k + w])
			out.append(v if t % 2 == 0 else v.decode('utf-8'))
			k += w
		else:
			raise ValueError(f'serial type {t} is reserved')
	if k != len(payload):
		raise ValueError(f'the record holds {k} bytes, its cell says {len(payload)}')
	return out


class _Db:
	"""The pages of one database file, read through f."""

	def __init__(self, f):
		head = f.read(0, 100)
		if head[:16] != b'SQLite format 3\x00':
			raise ValueError(f'{f.path} is not a SQLite database')
		if _u32(head, 56) not in (0, 1):
			raise ValueError(f'{f.path} does not keep its text as UTF-8')
		size = _u16(head, 16)
		self.f, self.size = f, 65536 if size == 1 else size
		self.usable = self.size - head[20]                       # less the bytes reserved at each page's end
		self.lock_page = LOCK_BYTE // self.size + 1

	def pages(self, n, count=1):
		"""count pages from page n, but never into the lock-byte page."""
		if n < self.lock_page < n + count:
			count = self.lock_page - n
		return self.f.read((n - 1) * self.size, count * self.size)

	def payload(self, p, off):
		"""A table leaf cell's payload: the part kept on its page, then its overflow pages, each the next one's number
		and usable - 4 bytes. The overflow pages are read in runs: a row written at the end of the file has them one
		after another, so its whole chain is usually one read."""
		total, j = _varint(p, off)
		j = _varint(p, j)[1]                                     # past the rowid
		u = self.usable
		x = u - 35
		if total <= x:
			return p[j:j + total]
		m = (u - 12) * 32 // 255 - 23
		k = m + (total - m) % (u - 4)
		local = k if k <= x else m
		parts, left, nxt = [p[j:j + local]], total - local, _u32(p, j + local)
		run_first, run = 0, b''
		for _ in range(left // (u - 4) + 1):
			if nxt == 0:
				raise ValueError('the overflow chain ends before the payload does')
			i = nxt - run_first
			if not (run and 0 <= i < len(run) // self.size):
				run_first, i, run = nxt, 0, self.pages(nxt, -(-left // (u - 4)))
			q = run[i * self.size:(i + 1) * self.size]
			take = min(left, u - 4)
			parts.append(q[4:4 + take])
			left -= take
			nxt = _u32(q, 0)
			if left == 0:
				break
		if left or nxt:
			raise ValueError('the overflow chain does not end where the payload does')
		return b''.join(parts)

	def row(self, root, rowid):
		"""The column values of rowid's row in the table b-tree whose root is page root."""
		n = root
		for _ in range(64):
			p = self.pages(n)
			h = 100 if n == 1 else 0                             # page 1 starts with the file header
			kind, ncell = p[h], _u16(p, h + 3)
			if kind == 0x0D:                                     # a table leaf page
				i = _first_at_or_above(p, h + 8, ncell, rowid, False)
				if i == ncell or _key(p, h + 8 + 2 * i, False) != rowid:
					raise KeyError(f'rowid {rowid} is not on its leaf page {n}')
				return _record(self.payload(p, _u16(p, h + 8 + 2 * i)))
			if kind != 0x05:
				raise ValueError(f'page {n} is not a table b-tree page (type {kind:#x})')
			i = _first_at_or_above(p, h + 12, ncell, rowid, True)  # an interior page: down the left child of the
			n = _u32(p, h + 8) if i == ncell else _u32(p, _u16(p, h + 12 + 2 * i))  # first key >= rowid, else right
		raise ValueError('the b-tree is more than 64 pages deep')


def root_page(conn, table):
	"""The root page of a table, from SQL. The schema table is rooted at page 1 and not listed."""
	if table == 'sqlite_master':
		return 1
	return conn.execute("SELECT rootpage FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()[0]


def row(db_path, root, rowid):
	"""The column values (table column order) of rowid's row in the table rooted at page root, as the drive holds the
	database's main file. Raises KeyError when the main file has no such row, ValueError when its pages do not parse.
	See CALLER RULES above."""
	with DriveFile(db_path) as f:
		return _Db(f).row(root, rowid)


def rows(db_path, root, rowids):
	"""row() for several rows of one table through ONE uncached handle (the archive's batched store, 2026-11-03):
	{rowid: column values, or the KeyError / ValueError / OSError that reading that row raised}. Each row is walked
	from the root and read from the drive by itself, exactly as row() reads it; only the open and the file header are
	shared. Raises OSError when the file cannot be opened or its header read, ValueError when the header is not
	SQLite's. Same CALLER RULES."""
	out = {}
	with DriveFile(db_path) as f:
		db = _Db(f)
		for rowid in rowids:
			try:
				out[rowid] = db.row(root, rowid)
			except (KeyError, ValueError, OSError) as exc:
				out[rowid] = exc
	return out
