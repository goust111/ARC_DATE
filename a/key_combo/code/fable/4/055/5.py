from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_pieces(grid: np.ndarray) -> list[np.ndarray]:
    bg = _get_bg(grid)
    pieces: list[np.ndarray] = []
    for coords in _components(grid != bg):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        patch_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        assert bool(patch_mask.all()), "each fragment must be a solid rectangle"
        pieces.append(np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy())
    assert 3 <= len(pieces) <= 4, "expected three or four rectangular fragments"
    return pieces


def _overlap_slices(
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[slice, slice, slice, slice] | None:
    ha, wa = map(int, shape_a)
    hb, wb = map(int, shape_b)
    ra0 = max(0, dr)
    ca0 = max(0, dc)
    rb0 = max(0, -dr)
    cb0 = max(0, -dc)
    overlap_h = min(ha - ra0, hb - rb0)
    overlap_w = min(wa - ca0, wb - cb0)
    if overlap_h <= 0 or overlap_w <= 0:
        return None
    return (
        slice(ra0, ra0 + overlap_h),
        slice(ca0, ca0 + overlap_w),
        slice(rb0, rb0 + overlap_h),
        slice(cb0, cb0 + overlap_w),
    )


def _matching_offsets(a: np.ndarray, b: np.ndarray) -> list[Coord]:
    ha, wa = a.shape
    hb, wb = b.shape
    offsets: list[Coord] = []
    for dr in range(-hb + 1, ha):
        for dc in range(-wb + 1, wa):
            overlap = _overlap_slices(a.shape, b.shape, dr, dc)
            if overlap is None:
                continue
            ars, acs, brs, bcs = overlap
            if np.array_equal(a[ars, acs], b[brs, bcs]):
                offsets.append((dr, dc))
    return offsets


def _position_valid(
    pieces: list[np.ndarray],
    placed: dict[int, Coord],
    idx: int,
    pos: Coord,
) -> bool:
    pr, pc = pos
    piece = pieces[idx]
    has_overlap = False
    for other_idx, (orow, ocol) in placed.items():
        other = pieces[other_idx]
        overlap = _overlap_slices(other.shape, piece.shape, pr - orow, pc - ocol)
        if overlap is None:
            continue
        has_overlap = True
        ors, ocs, prs, pcs = overlap
        if not np.array_equal(other[ors, ocs], piece[prs, pcs]):
            return False
    return has_overlap


def _render_full_rectangle(
    pieces: list[np.ndarray],
    placed: dict[int, Coord],
) -> np.ndarray | None:
    min_r = min(r for r, _ in placed.values())
    min_c = min(c for _, c in placed.values())
    max_r = max(r + pieces[i].shape[0] for i, (r, _c) in placed.items())
    max_c = max(c + pieces[i].shape[1] for i, (_r, c) in placed.items())
    out = np.full((max_r - min_r, max_c - min_c), -1, dtype=int)
    cover = np.zeros(out.shape, dtype=bool)

    for i, (r, c) in placed.items():
        patch = pieces[i]
        rr = r - min_r
        cc = c - min_c
        h, w = patch.shape
        window = out[rr : rr + h, cc : cc + w]
        overlap = cover[rr : rr + h, cc : cc + w]
        if np.any(overlap & (window != patch)):
            return None
        window[~overlap] = patch[~overlap]
        cover[rr : rr + h, cc : cc + w] = True

    if not bool(cover.all()):
        return None
    return out


def _assemble(pieces: list[np.ndarray]) -> np.ndarray:
    n = len(pieces)
    pair_offsets = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_offsets[i][j] = _matching_offsets(pieces[i], pieces[j])

    solutions: dict[tuple[bytes, tuple[int, int]], np.ndarray] = {}
    placed: dict[int, Coord] = {0: (0, 0)}

    def search() -> None:
        if len(solutions) > 1:
            return
        if len(placed) == n:
            out = _render_full_rectangle(pieces, placed)
            if out is None:
                return
            key = (out.tobytes(), tuple(map(int, out.shape)))
            solutions[key] = out
            return

        best_idx = None
        best_positions: list[Coord] | None = None
        for idx in range(n):
            if idx in placed:
                continue
            candidates: set[Coord] = set()
            for other_idx, (orow, ocol) in placed.items():
                for dr, dc in pair_offsets[other_idx][idx]:
                    candidates.add((orow + dr, ocol + dc))
            if not candidates:
                continue
            ordered = sorted(candidates)
            if best_positions is None or len(ordered) < len(best_positions):
                best_idx = idx
                best_positions = ordered

        if best_idx is None or best_positions is None:
            return

        for pos in best_positions:
            if not _position_valid(pieces, placed, best_idx, pos):
                continue
            placed[best_idx] = pos
            search()
            del placed[best_idx]
            if len(solutions) > 1:
                return

    search()
    assert len(solutions) == 1, "expected a unique assembly"
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """Align the scattered rectangular fragments by overlap consistency and recover their hidden full rectangle.

    Input structure:
    - A dominant outer background contains three or four disconnected rectangular fragments.
    - Each fragment is a solid rectangle filled with exactly two motif colors and is an unrotated crop of one hidden pattern.
    - Whenever two source windows overlap in that hidden pattern, the overlapping cells agree exactly.

    Rule:
    - Ignore the fragments' input positions and extract each rectangular crop.
    - Search for relative translations that make all overlapping cells between placed fragments identical.
    - Keep only assemblies whose union is a single fully covered rectangle, then output that completed rectangle.

    Key insight:
    - The coordinates in the input are irrelevant; the real placement signal is the content agreement inside overlaps, which uniquely determines the missing global alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _assemble(_extract_pieces(g))


GEN_BG = 0
GEN_COLORS = tuple(range(1, 10))
GEN_DIM_RANGE = (6, 11)
GEN_COUNT_OPTIONS = (3, 4)
GEN_MARGIN_RANGE = (2, 5)
GEN_GAP_RANGE = (2, 5)
GEN_ATTEMPTS = 500


def _random_composition(total: int, parts: int, min_part: int) -> list[int]:
    remain = total - parts * min_part
    cuts = sorted(random.sample(range(remain + parts - 1), parts - 1)) if parts > 1 else []
    out: list[int] = []
    prev = -1
    for cut in cuts + [remain + parts - 1]:
        stars = cut - prev - 1
        out.append(min_part + stars)
        prev = cut
    return out


def _make_pattern(h: int, w: int, colors: tuple[int, int]) -> np.ndarray:
    density = random.uniform(0.35, 0.65)
    mask = np.random.random((h, w)) < density
    if not mask.any() or bool(mask.all()):
        return _make_pattern(h, w, colors)
    return np.where(mask, colors[0], colors[1]).astype(int)


def _stripe_intervals(length: int, count: int) -> list[tuple[int, int]] | None:
    if length < 2 * count:
        return None
    base = _random_composition(length, count, 2)
    starts = [0] * count
    ends = [0] * count
    cut = 0
    overlaps: list[int] = []
    for i in range(count - 1):
        cut += base[i]
        max_overlap = min(2, base[i] - 1, base[i + 1] - 1)
        overlaps.append(random.randint(1, max_overlap))
    cut = 0
    starts[0] = 0
    for i in range(count - 1):
        cut += base[i]
        ends[i] = cut + overlaps[i] - 1
        starts[i + 1] = cut - overlaps[i]
    ends[-1] = length - 1
    return [(starts[i], ends[i]) for i in range(count)]


def _build_windows(pattern: np.ndarray, orientation: str, count: int) -> list[np.ndarray] | None:
    h, w = pattern.shape
    if orientation == "vertical":
        intervals = _stripe_intervals(w, count)
        if intervals is None:
            return None
        windows = [pattern[:, c0 : c1 + 1].copy() for c0, c1 in intervals]
    else:
        intervals = _stripe_intervals(h, count)
        if intervals is None:
            return None
        windows = [pattern[r0 : r1 + 1, :].copy() for r0, r1 in intervals]

    if any(len(np.unique(win)) < 2 for win in windows):
        return None

    coverage = np.zeros(pattern.shape, dtype=int)
    if orientation == "vertical":
        for c0, c1 in intervals:
            coverage[:, c0 : c1 + 1] += 1
    else:
        for r0, r1 in intervals:
            coverage[r0 : r1 + 1, :] += 1
    if not bool((coverage > 0).all()):
        return None
    if orientation == "vertical":
        for c0, c1 in intervals:
            if not bool(np.any(coverage[:, c0 : c1 + 1] == 1)):
                return None
    else:
        for r0, r1 in intervals:
            if not bool(np.any(coverage[r0 : r1 + 1, :] == 1)):
                return None
    return windows


def _scatter_pieces(pieces: list[np.ndarray]) -> np.ndarray | None:
    order = list(range(len(pieces)))
    random.shuffle(order)
    pieces = [pieces[i] for i in order]
    layouts_by_count = {
        3: [(3,), (1, 2), (2, 1)],
        4: [(4,), (2, 2), (1, 3), (3, 1)],
    }
    layouts = list(layouts_by_count[len(pieces)])
    random.shuffle(layouts)

    for layout in layouts:
        rows: list[list[np.ndarray]] = []
        start = 0
        for size in layout:
            rows.append(pieces[start : start + size])
            start += size

        row_gaps = [random.randint(*GEN_GAP_RANGE) for _ in range(max(0, len(rows) - 1))]
        piece_gaps = [[random.randint(*GEN_GAP_RANGE) for _ in range(max(0, len(row) - 1))] for row in rows]
        row_heights = [max(piece.shape[0] for piece in row) for row in rows]
        row_widths = [
            sum(piece.shape[1] for piece in row) + sum(gaps)
            for row, gaps in zip(rows, piece_gaps)
        ]
        top = random.randint(*GEN_MARGIN_RANGE)
        left = random.randint(*GEN_MARGIN_RANGE)
        bottom = random.randint(*GEN_MARGIN_RANGE)
        right = random.randint(*GEN_MARGIN_RANGE)
        canvas_h = top + sum(row_heights) + sum(row_gaps) + bottom
        canvas_w = left + max(row_widths) + right
        if canvas_h > 30 or canvas_w > 30:
            continue

        grid = np.full((canvas_h, canvas_w), GEN_BG, dtype=int)
        y = top
        for row_idx, (row, gaps, row_h, row_w) in enumerate(zip(rows, piece_gaps, row_heights, row_widths)):
            x_base_slack = canvas_w - right - left - row_w
            x = left + (0 if x_base_slack == 0 else random.randint(0, x_base_slack))
            for i, piece in enumerate(row):
                ph, pw = piece.shape
                y_offset = 0 if row_h == ph else random.randint(0, row_h - ph)
                grid[y + y_offset : y + y_offset + ph, x : x + pw] = piece
                x += pw
                if i < len(gaps):
                    x += gaps[i]
            y += row_h
            if row_idx < len(row_gaps):
                y += row_gaps[row_idx]

        if _get_bg(grid) != GEN_BG:
            continue
        return grid
    return None


def _candidate_windows() -> tuple[np.ndarray, list[np.ndarray]] | None:
    count = random.choice(GEN_COUNT_OPTIONS)
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        h = random.randint(*GEN_DIM_RANGE)
        w = random.randint(max(GEN_DIM_RANGE[0], 2 * count), GEN_DIM_RANGE[1] - 1)
    else:
        h = random.randint(max(GEN_DIM_RANGE[0], 2 * count), GEN_DIM_RANGE[1] - 1)
        w = random.randint(*GEN_DIM_RANGE)
    colors = tuple(random.sample(GEN_COLORS, 2))
    pattern = _make_pattern(h, w, colors)
    windows = _build_windows(pattern, orientation, count)
    if windows is None:
        return None
    return pattern, windows


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _candidate_windows()
        if candidate is None:
            continue
        pattern, windows = candidate
        grid = _scatter_pieces(windows)
        if grid is None:
            continue
        try:
            restored = main(grid)
        except AssertionError:
            continue
        if np.array_equal(restored, pattern):
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
