from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SEED_COLOR = 4
BG_FALLBACK = 5
LEGEND_COLOR = 6
NON_SPECIAL_COLORS = (0, 1, 2, 3, 7, 8, 9)
PATTERN_LIBRARY = (
    np.array([[1, 0], [0, 0]], dtype=bool),
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
)
GEN_RETRIES = 400
FRAME_PLACE_TRIES = 250


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the magenta legend, map seed counts to binary patterns, then tile the
    matching pattern inside each colored rectangular fence using the fence color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend_bbox = _find_legend_bbox(g)
    mapping = _read_legend(g, legend_bbox, bg)

    out = g.copy()
    for color in np.unique(g):
        color = int(color)
        if color in (bg, SEED_COLOR, LEGEND_COLOR):
            continue
        for comp in _components(g == color):
            bbox = _bbox(comp)
            if _bbox_inside(bbox, legend_bbox):
                continue
            if not _is_rect_frame(comp, bbox):
                continue
            r0, c0, r1, c1 = bbox
            assert r1 - r0 >= 2 and c1 - c0 >= 2
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_count = int(np.count_nonzero(interior == SEED_COLOR))
            assert seed_count in mapping
            out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(
                mapping[seed_count], interior.shape, color, bg
            )

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    entry_count = rng.randint(2, 4)
    counts = rng.sample((1, 2, 3, 4, 5), entry_count)
    patterns = rng.sample(PATTERN_LIBRARY, entry_count)
    legend_entries = []
    sample_colors = rng.sample(NON_SPECIAL_COLORS, entry_count)
    for count, pattern, sample_color in zip(counts, patterns, sample_colors):
        legend_entries.append(_make_legend_entry(rng, count, pattern, sample_color))
    rng.shuffle(legend_entries)

    legend_inner_h = sum(entry.shape[0] for entry in legend_entries) + (entry_count - 1)
    legend_inner_w = max(entry.shape[1] for entry in legend_entries)
    legend_h = legend_inner_h + 2
    legend_w = legend_inner_w + 2

    h_min = max(18, legend_h + 5)
    w_min = max(18, legend_w + 8)
    if h_min > 30 or w_min > 34:
        return None
    h = rng.randint(h_min, 30)
    w = rng.randint(w_min, 34)
    bg = BG_FALLBACK
    grid = np.full((h, w), bg, dtype=int)

    legend_bbox = _place_legend(grid, legend_entries, legend_inner_h, legend_inner_w, rng)
    occupied = np.zeros((h, w), dtype=bool)
    _mark_with_margin(occupied, legend_bbox, h, w)

    fence_count = rng.randint(2, min(4, len(NON_SPECIAL_COLORS)))
    fence_colors = rng.sample(NON_SPECIAL_COLORS, fence_count)
    region_counts = [rng.choice(counts) for _ in range(fence_count)]
    if fence_count > 1 and len(set(counts)) > 1 and len(set(region_counts)) == 1:
        region_counts[-1] = rng.choice([c for c in counts if c != region_counts[0]])

    for fence_color, seed_count in zip(fence_colors, region_counts):
        placed = False
        for _ in range(FRAME_PLACE_TRIES):
            inner_h = rng.randint(3, 8)
            inner_w = rng.randint(3, 8)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 >= h or outer_w + 2 >= w:
                continue
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _rect_clear(occupied, bbox):
                continue
            _draw_frame(grid, bbox, fence_color)
            interior_cells = [
                (r, c)
                for r in range(r0 + 1, r0 + outer_h - 1)
                for c in range(c0 + 1, c0 + outer_w - 1)
            ]
            for r, c in rng.sample(interior_cells, seed_count):
                grid[r, c] = SEED_COLOR
            _mark_with_margin(occupied, bbox, h, w)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_inside(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0] <= inner[2] <= outer[2]
        and outer[1] <= inner[1] <= inner[3] <= outer[3]
    )


def _is_rect_frame(points: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(points) != expected:
        return False
    point_set = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in point_set):
                return False
    return True


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in _components(grid == LEGEND_COLOR):
        bbox = _bbox(comp)
        if not _is_rect_frame(comp, bbox):
            continue
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if np.count_nonzero(interior != _get_bg(grid)) == 0:
            continue
        candidates.append(bbox)
    assert len(candidates) == 1
    return candidates[0]


def _read_legend(
    grid: np.ndarray, legend_bbox: tuple[int, int, int, int], bg: int
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = legend_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    active_rows = [r for r in range(interior.shape[0]) if np.any(interior[r] != bg)]
    assert active_rows

    bands = []
    start = active_rows[0]
    prev = active_rows[0]
    for row in active_rows[1:]:
        if row == prev + 1:
            prev = row
            continue
        bands.append((start, prev))
        start = prev = row
    bands.append((start, prev))

    mapping: dict[int, np.ndarray] = {}
    for r_start, r_end in bands:
        band = interior[r_start : r_end + 1]
        seed_count = int(np.count_nonzero(band == SEED_COLOR))
        pattern_mask = (band != bg) & (band != SEED_COLOR)
        assert seed_count > 0
        assert np.any(pattern_mask)
        coords = np.argwhere(pattern_mask)
        pr0, pc0 = np.min(coords, axis=0)
        pr1, pc1 = np.max(coords, axis=0)
        pattern = pattern_mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
        assert seed_count not in mapping
        mapping[seed_count] = pattern
    assert mapping
    return mapping


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int], color: int, bg: int) -> np.ndarray:
    ph, pw = pattern.shape
    out = np.full(shape, bg, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            if pattern[r % ph, c % pw]:
                out[r, c] = color
    return out


def _make_legend_entry(
    rng: random.Random, count: int, pattern: np.ndarray, sample_color: int
) -> np.ndarray:
    band = np.full((3, 4 + count), BG_FALLBACK, dtype=int)
    ph, pw = pattern.shape
    row_choices = [r for r in range(3 - ph + 1) if r <= 1 < r + ph]
    off_r = rng.choice(row_choices)
    off_c = rng.randint(0, 3 - pw)
    patch = band[off_r : off_r + ph, off_c : off_c + pw]
    patch[pattern] = sample_color
    band[1, 4 : 4 + count] = SEED_COLOR
    return band


def _place_legend(
    grid: np.ndarray,
    entries: list[np.ndarray],
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    legend_h = inner_h + 2
    legend_w = inner_w + 2
    corner = rng.choice(("TL", "TR", "BL", "BR"))
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, w - legend_w
    elif corner == "BL":
        r0, c0 = h - legend_h, 0
    else:
        r0, c0 = h - legend_h, w - legend_w
    r1 = r0 + legend_h - 1
    c1 = c0 + legend_w - 1
    _draw_frame(grid, (r0, c0, r1, c1), LEGEND_COLOR)

    cur_r = r0 + 1
    for idx, entry in enumerate(entries):
        eh, ew = entry.shape
        grid[cur_r : cur_r + eh, c0 + 1 : c0 + 1 + ew] = entry
        cur_r += eh
        if idx != len(entries) - 1:
            cur_r += 1
    return r0, c0, r1, c1


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _mark_with_margin(
    occupied: np.ndarray, bbox: tuple[int, int, int, int], h: int, w: int
) -> None:
    r0, c0, r1, c1 = bbox
    occupied[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _rect_clear(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not np.any(occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2])
