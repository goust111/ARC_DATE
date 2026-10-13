from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BAND = 5
CONTROL = 7
PALETTE_POOL = [1, 2, 3, 4, 6, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500
GRID_MIN = 18
GRID_MAX = 24
PALETTE_MIN = 2
PALETTE_MAX = 4
BAND_COUNT_MIN = 2
BAND_COUNT_MAX = 4
WIDTH_MIN = 3
WIDTH_MAX = 5
LENGTH_MIN = 8
LENGTH_MAX = 12
CONTROL_MIN = 2
CONTROL_MAX = 5
PLACEMENT_TRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for row in (0, h - 1):
        cols = [c for c in range(w) if int(grid[row, c]) not in (BG, BAND, CONTROL)]
        if len(cols) < 2:
            continue
        groups: list[list[int]] = []
        current = [cols[0]]
        for c in cols[1:]:
            if c == current[-1] + 1:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
        groups.append(current)
        valid_groups = []
        for group in groups:
            colors = [int(grid[row, c]) for c in group]
            if len(group) >= 2 and len(set(colors)) == len(colors):
                valid_groups.append(colors)
        if len(valid_groups) == 1:
            candidates.append((row, valid_groups[0]))
    assert len(candidates) == 1, "expected one palette row on the top or bottom edge"
    row, palette = candidates[0]
    assert PALETTE_MIN <= len(palette) <= PALETTE_MAX
    return row, palette


def _band_depth_map(coords: np.ndarray) -> dict[tuple[int, int], int]:
    rows: dict[int, list[int]] = {}
    for r, c in coords:
        rows.setdefault(int(r), []).append(int(c))
    r0 = min(rows)
    r1 = max(rows)
    top_center = (min(rows[r0]) + max(rows[r0])) / 2.0
    bottom_center = (min(rows[r1]) + max(rows[r1])) / 2.0
    assert top_center != bottom_center, "expected a slanted diagonal band"

    if bottom_center > top_center:
        strip_values = sorted({int(c) - int(r) for r, c in coords})
        strip_index = {value: i for i, value in enumerate(strip_values)}
        max_index = len(strip_values) - 1
        return {
            (int(r), int(c)): min(strip_index[int(c) - int(r)], max_index - strip_index[int(c) - int(r)])
            for r, c in coords
        }

    strip_values = sorted({int(r) + int(c) for r, c in coords})
    strip_index = {value: i for i, value in enumerate(strip_values)}
    max_index = len(strip_values) - 1
    return {
        (int(r), int(c)): min(strip_index[int(r) + int(c)], max_index - strip_index[int(r) + int(c)])
        for r, c in coords
    }


def _analyze_bands(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    components = _find_components_of_color(grid, BAND)
    assert components, "expected at least one grey diagonal band"
    depth_by_pos: dict[tuple[int, int], int] = {}
    band_id_by_pos: dict[tuple[int, int], int] = {}
    for band_id, coords in enumerate(components):
        band_depths = _band_depth_map(coords)
        for pos, depth in band_depths.items():
            depth_by_pos[pos] = depth
            band_id_by_pos[pos] = band_id
    return depth_by_pos, band_id_by_pos


def _first_hit(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = start[0] + step[0], start[1] + step[1]
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) == BAND:
            return int(r), int(c)
        r += step[0]
        c += step[1]
    return None


def _ordered_controls(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in zip(*np.where(grid == CONTROL))]
    coords.sort()
    return coords


def _hits_by_control(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    hits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for control in _ordered_controls(grid):
        for step in DIRS:
            hit = _first_hit(grid, control, step)
            if hit is not None:
                hits.append((control, hit))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast orthogonal rays from orange controls onto grey diagonal bands and stamp each hit with the palette color indexed by band depth.

    Input structure:
    - The background is dominant black.
    - One edge row contains a contiguous horizontal palette of 2-4 distinct colors.
    - The foreground also contains several grey 45-degree diagonal bands and 2-5 isolated orange control pixels.

    Rule:
    - Keep the palette row unchanged and keep every non-hit grey band pixel unchanged.
    - For each orange control pixel in row-major order, cast rays up, down, left, and right until the first grey band pixel in that direction.
    - For every hit, compute the pixel's depth inside its band as the distance to the nearer side strip measured perpendicular to the band's diagonal axis.
    - Recolor the hit pixel with palette[depth mod palette_length], then remove all orange control pixels to background.

    Key insight:
    - The orange controls choose positions, but the stamp colors come from the hit band's internal strip depth rather than from the control or the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    palette_row, palette = _extract_palette(g)
    assert bg == BG, "expected black background"
    assert np.count_nonzero(g == CONTROL) >= CONTROL_MIN, "expected orange control pixels"

    depth_by_pos, _band_id_by_pos = _analyze_bands(g)
    out = g.copy()
    for _control, hit in _hits_by_control(g):
        depth = depth_by_pos[hit]
        out[hit] = palette[depth % len(palette)]

    for r, c in _ordered_controls(g):
        out[r, c] = bg

    assert palette_row in (0, g.shape[0] - 1)
    return out


def _expanded8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = True
    return out


def _band_cells(length: int, width: int, slope: int, r0: int, c0: int) -> np.ndarray:
    cells = []
    for u in range(length):
        for v in range(width):
            r = r0 + u
            c = c0 + u + v if slope > 0 else c0 - u + v
            cells.append((r, c))
    return np.array(cells, dtype=int)


def _place_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    row = random.choice([0, h - 1])
    size = random.randint(PALETTE_MIN, PALETTE_MAX)
    palette = random.sample(PALETTE_POOL, size)
    start_min = 1 if w - size >= 3 else 0
    start_max = w - size - 1 if w - size >= 3 else w - size
    start = random.randint(start_min, start_max)
    grid[row, start : start + size] = np.array(palette, dtype=int)
    return row, palette


def _place_bands(grid: np.ndarray, palette_row: int, band_specs: list[tuple[int, int, int]]) -> bool:
    h, w = grid.shape
    occupied = grid != BG
    for length, width, slope in sorted(band_specs, key=lambda item: item[0] * item[1], reverse=True):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            if slope > 0:
                r0_lo = 2 if palette_row == 0 else 1
                r0_hi = h - length - (2 if palette_row == h - 1 else 1)
                if r0_lo > r0_hi:
                    return False
                r0 = random.randint(r0_lo, r0_hi)
                c0 = random.randint(0, w - (length + width - 1))
            else:
                r0_lo = 2 if palette_row == 0 else 1
                r0_hi = h - length - (2 if palette_row == h - 1 else 1)
                if r0_lo > r0_hi:
                    return False
                r0 = random.randint(r0_lo, r0_hi)
                c0 = random.randint(length - 1, w - width)

            cells = _band_cells(length, width, slope, r0, c0)
            blocked = _expanded8(occupied)
            if np.any(blocked[cells[:, 0], cells[:, 1]]):
                continue
            grid[cells[:, 0], cells[:, 1]] = BAND
            occupied[cells[:, 0], cells[:, 1]] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _cell_is_isolated_background(grid: np.ndarray, r: int, c: int, palette_row: int) -> bool:
    h, w = grid.shape
    if int(grid[r, c]) != BG or r == palette_row:
        return False
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr != r or cc != c) and int(grid[rr, cc]) != BG:
                return False
    return True


def _control_candidates(
    grid: np.ndarray,
    palette_row: int,
    depth_by_pos: dict[tuple[int, int], int],
    band_id_by_pos: dict[tuple[int, int], int],
) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not _cell_is_isolated_background(grid, r, c, palette_row):
                continue
            hits = {}
            hit_depths = set()
            hit_bands = set()
            for step in DIRS:
                hit = _first_hit(grid, (r, c), step)
                if hit is None:
                    continue
                hits[step] = hit
                hit_depths.add(depth_by_pos[hit])
                hit_bands.add(band_id_by_pos[hit])
            if hits:
                candidates.append(
                    {
                        "pos": (r, c),
                        "hits": hits,
                        "depths": hit_depths,
                        "bands": hit_bands,
                    }
                )
    return candidates


def _far_from_selected(pos: tuple[int, int], selected: list[dict]) -> bool:
    return all(max(abs(pos[0] - item["pos"][0]), abs(pos[1] - item["pos"][1])) > 1 for item in selected)


def _select_controls(candidates: list[dict], target_count: int) -> list[dict] | None:
    interior = [item for item in candidates if any(depth > 0 for depth in item["depths"])]
    if not interior:
        return None

    first = random.choice(interior)
    selected = [first]
    pool = candidates[:]
    random.shuffle(pool)
    for item in pool:
        if item is first:
            continue
        if _far_from_selected(item["pos"], selected):
            selected.append(item)
            if len(selected) == target_count:
                break
    if len(selected) != target_count:
        return None
    return selected


def _validate_generated(grid: np.ndarray, palette: list[int]) -> bool:
    depth_by_pos, band_id_by_pos = _analyze_bands(grid)
    hits = _hits_by_control(grid)
    if len(hits) < 2:
        return False
    if len({hit for _control, hit in hits}) < 2:
        return False
    if not any(depth_by_pos[hit] > 0 for _control, hit in hits):
        return False
    if len({band_id_by_pos[hit] for _control, hit in hits}) < 2:
        return False

    out = main(grid)
    stamp_mask = (grid == BAND) & (out != BAND)
    if int(stamp_mask.sum()) < 2:
        return False
    stamp_colors = {int(out[r, c]) for r, c in zip(*np.where(stamp_mask))}
    if len(stamp_colors) < 2:
        return False
    if not stamp_colors.issubset(set(palette)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    band_count = random.randint(BAND_COUNT_MIN, BAND_COUNT_MAX)
    size_boost = 1 if band_count >= 3 else 0
    h = random.randint(GRID_MIN + size_boost, GRID_MAX)
    w = random.randint(GRID_MIN + size_boost, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    palette_row, palette = _place_palette(grid)
    specs = [
        (
            random.randint(LENGTH_MIN, min(LENGTH_MAX, h - 4)),
            random.randint(WIDTH_MIN, WIDTH_MAX),
            random.choice([-1, 1]),
        )
        for _ in range(band_count)
    ]
    if not _place_bands(grid, palette_row, specs):
        return None

    depth_by_pos, band_id_by_pos = _analyze_bands(grid)
    candidates = _control_candidates(grid, palette_row, depth_by_pos, band_id_by_pos)
    if len(candidates) < CONTROL_MIN:
        return None

    target_count = random.randint(CONTROL_MIN, min(CONTROL_MAX, len(candidates)))
    selected = _select_controls(candidates, target_count)
    if selected is None:
        return None

    for item in selected:
        r, c = item["pos"]
        grid[r, c] = CONTROL

    return grid if _validate_generated(grid, palette) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-band ray-stamp input")
