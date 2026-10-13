from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        value = int(grid[0, c])
        if value == bg:
            continue
        if np.all(grid[:, c] == value):
            candidates.append((c, value))
    assert candidates, "expected a full-height separator column"
    for c, value in candidates:
        if 0 < c < w - 1 and np.any(grid[:, :c] != bg) and np.any(grid[:, c + 1 :] != bg):
            return c, value
    return candidates[0]


def _extract_legend(panel: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for comp in _components(panel != bg):
        r0, c0, r1, c1 = _bbox(comp)
        sub = panel[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h not in (3, 4) or w != h + 1:
            continue
        left = sub[:, 0] != bg
        if int(left.sum()) != 1:
            continue
        pattern = sub[:, 1:]
        if pattern.shape != (h, h) or np.any(pattern == bg):
            continue
        index_row = int(np.flatnonzero(left)[0])
        index_color = int(sub[index_row, 0])
        mapping[index_color] = pattern.copy()
    assert mapping, "legend panel could not be parsed"
    return mapping


def _enclosed_regions(canvas: np.ndarray, wall_color: int) -> list[list[tuple[int, int]]]:
    h, w = canvas.shape
    passable = canvas != wall_color
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    interior = passable & (~seen)
    return _components(interior)


def _fill_canvas(canvas: np.ndarray, bg: int, wall_color: int, mapping: dict[int, np.ndarray]) -> np.ndarray:
    out = canvas.copy()
    for region in _enclosed_regions(canvas, wall_color):
        marker_cells = [(r, c) for r, c in region if int(canvas[r, c]) != bg]
        assert len(marker_cells) == 1, "each enclosed region must contain one marker"
        mr, mc = marker_cells[0]
        marker_color = int(canvas[mr, mc])
        pattern = mapping[marker_color]
        ph, pw = pattern.shape
        for r, c in region:
            out[r, c] = int(pattern[(r - mr) % ph, (c - mc) % pw])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed region with the legend pattern keyed by its interior marker color.

    Input structure:
    - The background is black, with a full-height grey separator column splitting a left legend panel from a right canvas.
    - Each legend entry is one colored index pixel directly left of a solid 3x3 or 4x4 multi-color pattern block.
    - The canvas contains several grey-walled closed regions, and each region has exactly one interior marker pixel whose color matches one legend index.

    Rule:
    - Read the legend as a mapping from index color to its adjacent pattern block.
    - Find every enclosed non-grey region on the right canvas and locate its unique non-background marker pixel.
    - Use that marker as the pattern's phase anchor: tile the matched pattern periodically across the whole enclosed region, replacing the marker itself while leaving the grey walls and the legend unchanged.

    Key insight:
    - The left panel is a color-to-pattern lookup table, and the marker's position inside a region determines not just which pattern to use but also the tiling phase that fills the region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    sep_col, wall_color = _find_separator(g, bg)
    legend = _extract_legend(g[:, :sep_col], bg)

    out = g.copy()
    canvas = g[:, sep_col + 1 :]
    out[:, sep_col + 1 :] = _fill_canvas(canvas, bg, wall_color, legend)
    return out


def _has_full_period(pattern: np.ndarray) -> bool:
    h, w = pattern.shape
    if len(set(map(int, pattern.ravel()))) < 2:
        return False
    for rp in range(1, h):
        if all(np.array_equal(pattern[r], pattern[r % rp]) for r in range(h)):
            return False
    for cp in range(1, w):
        if all(np.array_equal(pattern[:, c], pattern[:, c % cp]) for c in range(w)):
            return False
    return True


def _random_pattern(size: int, colors: tuple[int, ...]) -> np.ndarray:
    usable = list(colors)
    for _ in range(200):
        n_colors = random.randint(2, min(4, len(usable)))
        chosen = random.sample(usable, n_colors)
        cells = [random.choice(chosen) for _ in range(size * size)]
        for idx, color in enumerate(chosen):
            cells[idx] = color
        random.shuffle(cells)
        pattern = np.array(cells, dtype=int).reshape(size, size)
        if _has_full_period(pattern):
            return pattern
    raise RuntimeError("failed to generate a non-degenerate pattern")


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _region_dims(area_min: int, base_min: int) -> tuple[int, int] | None:
    choices = [
        (h, w)
        for h in range(base_min, 9)
        for w in range(base_min, 9)
        if h * w >= area_min
    ]
    return random.choice(choices) if choices else None


def _make_rectangle(area_min: int) -> np.ndarray | None:
    dims = _region_dims(area_min, max(5, int(np.ceil(np.sqrt(area_min)))))
    if dims is None:
        return None
    h, w = dims
    return np.ones((h, w), dtype=bool)


def _make_notched(area_min: int) -> np.ndarray | None:
    for _ in range(120):
        dims = _region_dims(area_min + 4, max(6, int(np.ceil(np.sqrt(area_min)))))
        if dims is None:
            return None
        h, w = dims
        mask = np.ones((h, w), dtype=bool)
        notch_h = random.randint(1, h - 2)
        notch_w = random.randint(1, w - 2)
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[:notch_h, :notch_w] = False
        elif corner == "tr":
            mask[:notch_h, w - notch_w :] = False
        elif corner == "bl":
            mask[h - notch_h :, :notch_w] = False
        else:
            mask[h - notch_h :, w - notch_w :] = False
        if int(mask.sum()) >= area_min:
            return mask
    return None


def _make_union(area_min: int) -> np.ndarray | None:
    for _ in range(120):
        dims = _region_dims(area_min, max(6, int(np.ceil(np.sqrt(area_min)))))
        if dims is None:
            return None
        h, w = dims
        mask = np.zeros((h, w), dtype=bool)
        stem_w = random.randint(2, w - 2)
        bar_h = random.randint(2, h - 2)
        if random.random() < 0.5:
            c0 = 0 if random.random() < 0.5 else w - stem_w
            r0 = 0 if random.random() < 0.5 else h - bar_h
        else:
            c0 = random.randint(0, w - stem_w)
            r0 = random.randint(0, h - bar_h)
        mask[:, c0 : c0 + stem_w] = True
        mask[r0 : r0 + bar_h, :] = True
        mask = _trim_mask(mask)
        if int(mask.sum()) >= area_min and np.any(~mask):
            return mask
    return None


def _make_region_mask(area_min: int) -> np.ndarray | None:
    families = (
        _make_rectangle,
        _make_notched,
        _make_notched,
        _make_union,
        _make_union,
    )
    for _ in range(200):
        mask = random.choice(families)(area_min)
        if mask is not None and int(mask.sum()) >= area_min:
            return mask
    return None


def _make_patch(inner: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = inner.shape
    inner_patch = np.zeros((h + 2, w + 2), dtype=bool)
    inner_patch[1 : h + 1, 1 : w + 1] = inner
    boundary = np.zeros_like(inner_patch)
    for r, c in np.argwhere(inner_patch):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not inner_patch[nr, nc]:
                boundary[nr, nc] = True
    return inner_patch, boundary, inner_patch | boundary


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def _place_regions(
    canvas: np.ndarray,
    region_specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    h, w = canvas.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    order = sorted(
        range(len(region_specs)),
        key=lambda i: int(np.sum(region_specs[i]["inner"])),
        reverse=True,
    )
    placed: list[dict[str, object] | None] = [None] * len(region_specs)

    for idx in order:
        spec = region_specs[idx]
        inner = spec["inner"]
        pattern = spec["pattern"]
        index_color = int(spec["index_color"])
        inner_patch, boundary, footprint = _make_patch(inner)
        ph, pw = footprint.shape
        if ph + 2 > h or pw + 2 > w:
            return None
        positions = [
            (top, left)
            for top in range(1, h - ph)
            for left in range(1, w - pw)
        ]
        random.shuffle(positions)
        success = False
        for top, left in positions:
            footprint_coords = [
                (top + int(r), left + int(c))
                for r, c in np.argwhere(footprint)
            ]
            if any(blocked[r, c] for r, c in footprint_coords):
                continue

            for r, c in np.argwhere(boundary):
                canvas[top + int(r), left + int(c)] = GREY

            cells = [
                (top + int(r), left + int(c))
                for r, c in np.argwhere(inner_patch)
            ]
            marker = random.choice(cells)
            canvas[marker] = index_color

            placed[idx] = {
                "cells": cells,
                "marker": marker,
                "pattern": pattern,
                "index_color": index_color,
            }
            _mark_blocked(blocked, footprint_coords)
            success = True
            break
        if not success:
            return None

    return [spec for spec in placed if spec is not None]


def _distribute(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _place_legend(
    grid: np.ndarray,
    entries: list[tuple[int, np.ndarray]],
    panel_width: int,
    height: int,
) -> None:
    sizes = [pattern.shape[0] for _, pattern in entries]
    baseline = sum(sizes) + (len(entries) - 1) + 2
    extra = height - baseline
    gaps = [1 + add for add in _distribute(extra, len(entries) + 1)]
    row = gaps[0]
    for i, (index_color, pattern) in enumerate(entries):
        size = int(pattern.shape[0])
        marker_row = random.randrange(size)
        grid[row + marker_row, 0] = index_color
        grid[row : row + size, 1 : 1 + size] = pattern
        row += size + gaps[i + 1]
    grid[:, panel_width] = GREY


def _expected_output(
    grid: np.ndarray,
    sep_col: int,
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    canvas = out[:, sep_col + 1 :]
    for spec in regions:
        pattern = spec["pattern"]
        cells = spec["cells"]
        mr, mc = spec["marker"]
        ph, pw = pattern.shape
        for r, c in cells:
            canvas[r, c] = int(pattern[(r - mr) % ph, (c - mc) % pw])
    return out


def _gen_attempt() -> np.ndarray | None:
    entry_count = random.randint(2, 4)
    region_count = random.randint(2, 3)
    index_colors = random.sample(PALETTE_COLORS, entry_count)
    pattern_pool = tuple(color for color in PALETTE_COLORS if color not in index_colors)

    sizes = [random.choice((3, 4)) for _ in range(entry_count)]
    if entry_count >= 2 and random.random() < 0.5:
        sizes[0] = 3
        sizes[1] = 4
        random.shuffle(sizes)

    entries: list[tuple[int, np.ndarray]] = []
    legend_map: dict[int, np.ndarray] = {}
    for index_color, size in zip(index_colors, sizes):
        pattern = _random_pattern(size, pattern_pool)
        entries.append((index_color, pattern))
        legend_map[index_color] = pattern
    random.shuffle(entries)

    region_specs: list[dict[str, object]] = []
    for _ in range(region_count):
        color = random.choice(index_colors)
        pattern = legend_map[color]
        area_min = 2 * int(pattern.shape[0]) * int(pattern.shape[1])
        inner = _make_region_mask(area_min)
        if inner is None:
            return None
        region_specs.append(
            {
                "index_color": color,
                "pattern": pattern,
                "inner": inner,
            }
        )

    panel_width = max(int(pattern.shape[0]) for _, pattern in entries) + 1
    canvas_min_w = max(int(spec["inner"].shape[1]) + 5 for spec in region_specs)
    canvas_max_w = 30 - panel_width - 1
    if canvas_min_w > canvas_max_w:
        return None

    panel_min_h = sum(int(pattern.shape[0]) for _, pattern in entries) + (len(entries) - 1) + 2
    region_min_h = max(int(spec["inner"].shape[0]) + 5 for spec in region_specs)
    height_min = max(16, panel_min_h, region_min_h)
    if height_min > 30:
        return None

    canvas_width = random.randint(canvas_min_w, canvas_max_w)
    width = panel_width + 1 + canvas_width
    height = random.randint(height_min, 30)

    grid = np.full((height, width), BLACK, dtype=int)
    _place_legend(grid, entries, panel_width, height)
    canvas = grid[:, panel_width + 1 :]
    placed_regions = _place_regions(canvas, region_specs)
    if placed_regions is None or len(placed_regions) != region_count:
        return None

    counts = Counter(map(int, grid.ravel()))
    if counts[BLACK] != max(counts.values()):
        return None

    expected = _expected_output(grid, panel_width, placed_regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
