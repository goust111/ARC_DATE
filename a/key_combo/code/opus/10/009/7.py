from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8

EDGES = ("left", "right", "top", "bottom")
PROFILE_SIZES = (5, 6, 7)
GRID_MIN = 13
GRID_MAX = 20
MAX_ATTEMPTS = 400

_CALL_COUNT = 0
_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _profile_catalog() -> list[tuple[int, ...]]:
    profiles: list[tuple[int, ...]] = []
    for height in PROFILE_SIZES:
        for peak in range(3, 6):
            if height == 5:
                for a in range(1, peak):
                    for b in range(1, a + 1):
                        prof = (peak, a, b, a, peak)
                        if 8 <= sum(prof) <= 15 and 2 * sum(peak - x for x in prof) >= 3:
                            profiles.append(prof)
            elif height == 6:
                for a in range(1, peak):
                    for b in range(1, a + 1):
                        prof = (peak, a, b, b, a, peak)
                        if 8 <= sum(prof) <= 15 and 2 * sum(peak - x for x in prof) >= 3:
                            profiles.append(prof)
            else:
                for a in range(1, peak):
                    for b in range(1, a + 1):
                        for c in range(1, b + 1):
                            prof = (peak, a, b, c, b, a, peak)
                            if 8 <= sum(prof) <= 15 and 2 * sum(peak - x for x in prof) >= 3:
                                profiles.append(prof)
    return profiles


HALF_PROFILES = _profile_catalog()


def _find_marker(grid: np.ndarray) -> tuple[int, int, str]:
    marker_cells = np.argwhere(grid == TEAL)
    assert marker_cells.shape == (1, 2), "expected exactly one teal edge marker"
    r, c = map(int, marker_cells[0])
    h, w = grid.shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "marker must be on exactly one edge"
    assert (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}, "marker must not be in a corner"
    if on_left:
        return r, c, "left"
    if on_right:
        return r, c, "right"
    if on_top:
        return r, c, "top"
    return r, c, "bottom"


def _compress_half(grid: np.ndarray, edge: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)

    if edge == "left":
        for r in range(h):
            count = int(np.count_nonzero(g[r, :] == BLUE))
            if count:
                out[r, :count] = BLUE
    elif edge == "right":
        for r in range(h):
            count = int(np.count_nonzero(g[r, :] == BLUE))
            if count:
                out[r, w - count :] = BLUE
    elif edge == "top":
        for c in range(w):
            count = int(np.count_nonzero(g[:, c] == BLUE))
            if count:
                out[:count, c] = BLUE
    else:
        for c in range(w):
            count = int(np.count_nonzero(g[:, c] == BLUE))
            if count:
                out[h - count :, c] = BLUE
    return out


def _mirror_half(half: np.ndarray, edge: str) -> np.ndarray:
    half = np.asarray(half, dtype=int)
    coords = np.argwhere(half == BLUE)
    assert coords.size > 0, "expected compacted blue pixels"
    h, w = half.shape
    out = half.copy()

    if edge == "left":
        axis = int(coords[:, 1].max())
        for r, c in coords:
            mc = 2 * axis + 1 - int(c)
            assert 0 <= mc < w
            out[int(r), mc] = BLUE
    elif edge == "right":
        axis = int(coords[:, 1].min())
        for r, c in coords:
            mc = 2 * axis - 1 - int(c)
            assert 0 <= mc < w
            out[int(r), mc] = BLUE
    elif edge == "top":
        axis = int(coords[:, 0].max())
        for r, c in coords:
            mr = 2 * axis + 1 - int(r)
            assert 0 <= mr < h
            out[mr, int(c)] = BLUE
    else:
        axis = int(coords[:, 0].min())
        for r, c in coords:
            mr = 2 * axis - 1 - int(r)
            assert 0 <= mr < h
            out[mr, int(c)] = BLUE

    return out


def _fill_interior(outline: np.ndarray, fill_color: int) -> np.ndarray:
    out = np.asarray(outline, dtype=int).copy()
    h, _w = out.shape
    for r in range(h):
        cols = np.flatnonzero(out[r, :] == BLUE)
        if cols.size >= 2:
            left = int(cols[0])
            right = int(cols[-1])
            if left + 1 < right:
                span = out[r, left + 1 : right]
                span[span == BG] = fill_color
                out[r, left + 1 : right] = span
    return out


def _components8(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
    return count


def _render_without_compaction(grid: np.ndarray, edge: str, fill_color: int) -> np.ndarray | None:
    raw = np.zeros_like(grid)
    raw[grid == BLUE] = BLUE
    coords = np.argwhere(raw == BLUE)
    if coords.size == 0:
        return None

    h, w = raw.shape
    mirrored = raw.copy()
    try:
        if edge == "left":
            axis = int(coords[:, 1].max())
            for r, c in coords:
                mc = 2 * axis + 1 - int(c)
                if not (0 <= mc < w):
                    return None
                mirrored[int(r), mc] = BLUE
        elif edge == "right":
            axis = int(coords[:, 1].min())
            for r, c in coords:
                mc = 2 * axis - 1 - int(c)
                if not (0 <= mc < w):
                    return None
                mirrored[int(r), mc] = BLUE
        elif edge == "top":
            axis = int(coords[:, 0].max())
            for r, c in coords:
                mr = 2 * axis + 1 - int(r)
                if not (0 <= mr < h):
                    return None
                mirrored[mr, int(c)] = BLUE
        else:
            axis = int(coords[:, 0].min())
            for r, c in coords:
                mr = 2 * axis - 1 - int(r)
                if not (0 <= mr < h):
                    return None
                mirrored[mr, int(c)] = BLUE
    except AssertionError:
        return None
    return _fill_interior(mirrored, fill_color)


def main(grid: np.ndarray) -> np.ndarray:
    """Compact the blue pixels toward the teal edge marker, mirror the emergent half-outline, and fill its interior by marker parity.

    Input structure:
    A black grid contains scattered blue foreground pixels and exactly one teal marker on a non-corner edge cell.
    The marker's edge determines whether blue pixels compact left, right, up, or down; the marker's coordinates determine the fill color.

    Rule:
    Compress every row or column of blue pixels toward the marked edge while clearing their original locations.
    Treat the compacted blue set as a half-outline, mirror it across the axis parallel to the marked edge that lies just beyond the compacted half's far boundary, then fill each row's strict interior between its leftmost and rightmost blue pixels.
    Use red when the marker coordinate sum is even, green when it is odd, and finally erase the teal marker.

    Key insight:
    The scattered blue pixels are not meaningful until edge-directed compaction turns per-row or per-column counts into a coherent half-shape; only after that does mirroring close the contour, while the fill color comes from the marker's checkerboard parity rather than from the shape itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_r, marker_c, edge = _find_marker(g)
    fill_color = RED if (marker_r + marker_c) % 2 == 0 else GREEN

    compacted = _compress_half(g, edge)
    mirrored = _mirror_half(compacted, edge)
    out = _fill_interior(mirrored, fill_color)
    out[marker_r, marker_c] = BG
    return out


def _choose_dims(edge: str, profile: tuple[int, ...]) -> tuple[int, int]:
    peak = int(profile[0])
    scatter_slack = random.randint(5, 9)
    if edge in ("left", "right"):
        h = random.randint(max(GRID_MIN, len(profile) + 4), GRID_MAX)
        w = random.randint(max(GRID_MIN, 2 * peak + scatter_slack), GRID_MAX)
    else:
        h = random.randint(max(GRID_MIN, 2 * peak + scatter_slack), GRID_MAX)
        w = random.randint(max(GRID_MIN, len(profile) + 4), GRID_MAX)
    return h, w


def _build_half(h: int, w: int, edge: str, profile: tuple[int, ...]) -> tuple[np.ndarray, int]:
    half = np.zeros((h, w), dtype=int)
    if edge in ("left", "right"):
        top = random.randint(1, h - len(profile) - 2)
        for i, length in enumerate(profile):
            r = top + i
            if edge == "left":
                half[r, :length] = BLUE
            else:
                half[r, w - length :] = BLUE
        return half, top

    left = random.randint(1, w - len(profile) - 2)
    for i, length in enumerate(profile):
        c = left + i
        if edge == "top":
            half[:length, c] = BLUE
        else:
            half[h - length :, c] = BLUE
    return half, left


def _pick_marker(h: int, w: int, edge: str, occupied_band: int, span: int, want_even: bool) -> tuple[int, int] | None:
    if edge in ("left", "right"):
        candidates = []
        for r in range(1, h - 1):
            if occupied_band <= r < occupied_band + span:
                continue
            c = 0 if edge == "left" else w - 1
            if ((r + c) % 2 == 0) == want_even:
                candidates.append((r, c))
        return random.choice(candidates) if candidates else None

    candidates = []
    for c in range(1, w - 1):
        if occupied_band <= c < occupied_band + span:
            continue
        r = 0 if edge == "top" else h - 1
        if ((r + c) % 2 == 0) == want_even:
            candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _pick_nonadjacent(values: list[int], k: int) -> list[int] | None:
    if k == 0:
        return []

    def rec(pool: list[int], need: int, start: int, chosen: list[int]) -> list[int] | None:
        if need == 0:
            return chosen[:]
        for i in range(start, len(pool)):
            value = pool[i]
            if chosen and value - chosen[-1] <= 1:
                continue
            chosen.append(value)
            got = rec(pool, need - 1, i + 1, chosen)
            if got is not None:
                return got
            chosen.pop()
        return None

    random.shuffle(values)
    values = sorted(values)
    return rec(values, k, 0, [])


def _scatter_rows(inp: np.ndarray, profile: tuple[int, ...], top: int, edge: str, marker: tuple[int, int]) -> bool:
    _h, w = inp.shape
    peak = int(profile[0])
    marker_r, marker_c = marker
    for i, count in enumerate(profile):
        r = top + i
        if edge == "left":
            base = list(range(peak + 2, w))
        else:
            base = list(range(0, w - peak - 2))
        if marker_r == r:
            base = [c for c in base if c != marker_c]
        random.shuffle(base)

        good = _pick_nonadjacent(base[:], count)
        if good is None:
            return False
        inp[r, good] = BLUE
    return True


def _scatter_cols(inp: np.ndarray, profile: tuple[int, ...], left: int, edge: str, marker: tuple[int, int]) -> bool:
    h, w = inp.shape
    peak = int(profile[0])
    marker_r, marker_c = marker
    for i, count in enumerate(profile):
        c = left + i
        if edge == "top":
            base = list(range(peak + 2, h))
        else:
            base = list(range(0, h - peak - 2))
        if marker_c == c:
            base = [r for r in base if r != marker_r]
        random.shuffle(base)

        good = _pick_nonadjacent(base[:], count)
        if good is None:
            return False
        inp[good, c] = BLUE
    return True


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _semantic_checks(
    inp: np.ndarray,
    half: np.ndarray,
    out: np.ndarray,
    edge: str,
    marker: tuple[int, int],
) -> bool:
    marker_r, marker_c = marker
    fill_color = RED if (marker_r + marker_c) % 2 == 0 else GREEN

    if np.count_nonzero(inp == BLUE) < 8 or np.count_nonzero(inp == BLUE) > 15:
        return False
    if _components8(inp == BLUE) <= 1:
        return False
    if np.array_equal(inp == BLUE, half == BLUE):
        return False
    if not np.array_equal(_compress_half(inp, edge), half):
        return False
    if not np.array_equal(main(inp), out):
        return False
    naive = _render_without_compaction(inp, edge, fill_color)
    if naive is not None and np.array_equal(naive, out):
        return False

    filled_cells = int(np.count_nonzero(out == fill_color))
    if filled_cells < 3:
        return False
    if np.any(out == TEAL):
        return False

    if edge in ("left", "right"):
        blue_rows = [r for r in range(inp.shape[0]) if np.any(inp[r, :] == BLUE)]
        if not blue_rows or max(blue_rows) - min(blue_rows) + 1 != len(blue_rows):
            return False
        for r in blue_rows:
            cols = np.flatnonzero(inp[r, :] == BLUE)
            if cols.size >= 2 and np.all(np.diff(cols) == 1):
                return False
    else:
        blue_cols = [c for c in range(inp.shape[1]) if np.any(inp[:, c] == BLUE)]
        if not blue_cols or max(blue_cols) - min(blue_cols) + 1 != len(blue_cols):
            return False
        for c in blue_cols:
            rows = np.flatnonzero(inp[:, c] == BLUE)
            if rows.size >= 2 and np.all(np.diff(rows) == 1):
                return False

    return True


def _attempt_generate() -> np.ndarray | None:
    global _CALL_COUNT

    edge = EDGES[_CALL_COUNT % len(EDGES)]
    want_even = (_CALL_COUNT % 2) == 0

    profile_choices = [p for p in HALF_PROFILES if len(p) == PROFILE_SIZES[_CALL_COUNT % len(PROFILE_SIZES)]]
    if not profile_choices:
        profile_choices = HALF_PROFILES
    profile = random.choice(profile_choices)

    h, w = _choose_dims(edge, profile)
    half, band_start = _build_half(h, w, edge, profile)
    marker = _pick_marker(h, w, edge, band_start, len(profile), want_even)
    if marker is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    marker_r, marker_c = marker
    inp[marker_r, marker_c] = TEAL

    if edge in ("left", "right"):
        if not _scatter_rows(inp, profile, band_start, edge, marker):
            return None
    else:
        if not _scatter_cols(inp, profile, band_start, edge, marker):
            return None

    out = main(np.where(inp == TEAL, TEAL, inp))
    if not _semantic_checks(inp, half, out, edge, marker):
        return None

    in_sig = _signature(inp)
    out_sig = _signature(out)
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None

    _SEEN_INPUTS.add(in_sig)
    _SEEN_OUTPUTS.add(out_sig)
    _CALL_COUNT += 1
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
