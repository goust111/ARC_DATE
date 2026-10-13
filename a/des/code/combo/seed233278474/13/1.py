from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, FRAME))
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_bbox(mask: np.ndarray, points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(points) != expected:
        return None
    for r, c in points:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    if not np.all(mask[r0, c0 : c1 + 1]):
        return None
    if not np.all(mask[r1, c0 : c1 + 1]):
        return None
    if not np.all(mask[r0 : r1 + 1, c0]):
        return None
    if not np.all(mask[r0 : r1 + 1, c1]):
        return None
    return r0, c0, r1, c1


def _find_frame_color(grid: np.ndarray, bg: int) -> int:
    best: tuple[int, int, int] | None = None
    best_color = -1
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = grid == color
        rings = []
        for pts in _components(mask):
            box = _ring_bbox(mask, pts)
            if box is not None:
                r0, c0, r1, c1 = box
                rings.append((r1 - r0 + 1) * (c1 - c0 + 1))
        if not rings:
            continue
        score = (len(rings), sum(rings), int(mask.sum()))
        if best is None or score > best:
            best = score
            best_color = color
    assert best is not None, "expected at least one rectangular frame color"
    return best_color


def _find_regions(grid: np.ndarray, bg: int, frame: int) -> list[tuple[int, int, int, int, int]]:
    regions: list[tuple[int, int, int, int, int]] = []
    mask = grid == frame
    for pts in _components(mask):
        box = _ring_bbox(mask, pts)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        non_bg = inner[(inner != bg) & (inner != frame)]
        if non_bg.size != 1:
            continue
        regions.append((r0, c0, r1, c1, int(non_bg[0])))
    assert regions, "expected at least one enclosed region with a unique seed"
    return regions


def _contiguous_run(indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size == 0:
        return None
    if np.any(np.diff(indices) != 1):
        return None
    return int(indices[0]), int(indices[-1])


def _legend_mapping(grid: np.ndarray, legend_top: int, bg: int) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for r in range(legend_top, grid.shape[0]):
        row = grid[r]
        nz = np.flatnonzero(row != bg)
        run = _contiguous_run(nz)
        if run is None:
            continue
        c0, c1 = run
        if c1 - c0 + 1 < 3:
            continue
        segment = row[c0 : c1 + 1]
        seed = int(segment[0])
        pattern = np.array(segment[1:], dtype=int)
        mapping[seed] = pattern
    assert mapping, "expected legend rows below the framed regions"
    return mapping


def _tile_pattern(pattern: np.ndarray, width: int) -> np.ndarray:
    tiled = np.empty(width, dtype=int)
    plen = int(pattern.size)
    for idx in range(width):
        tiled[idx] = int(pattern[idx % plen])
    return tiled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle using the stripe unit from the bottom legend row whose leading
    color matches the region's seed pixel, then remove the entire legend area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = int(BG if np.any(g == BG) else np.bincount(g.ravel()).argmax())
    frame = _find_frame_color(g, bg)
    regions = _find_regions(g, bg, frame)
    legend_top = max(r1 for _, _, r1, _, _ in regions) + 1
    legend = _legend_mapping(g, legend_top, bg)

    out = g[:legend_top, :].copy()
    for r0, c0, r1, c1, seed in regions:
        pattern = legend[seed]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        row = _tile_pattern(pattern, inner_w)
        out[r0 + 1 : r1, c0 + 1 : c1] = np.tile(row, (inner_h, 1))
    return out


def _random_pattern() -> np.ndarray:
    length = random.randint(2, 5)
    palette_size = random.randint(2, min(4, len(COLOR_POOL)))
    palette = random.sample(COLOR_POOL, palette_size)
    values = [random.choice(palette) for _ in range(length)]
    if len(set(values)) == 1:
        swap_idx = random.randrange(length)
        alt = random.choice([color for color in palette if color != values[0]])
        values[swap_idx] = alt
    return np.array(values, dtype=int)


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(used.shape[0], r0 + h + 1)
    cc1 = min(used.shape[1], c0 + w + 1)
    return not np.any(used[rr0:rr1, cc0:cc1])


def _place_regions(
    specs: list[dict[str, int | np.ndarray]], canvas_h: int, canvas_w: int
) -> list[tuple[int, int]] | None:
    used = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    ordered = specs[:]
    random.shuffle(ordered)
    positions_by_id: dict[int, tuple[int, int]] = {}
    for spec in ordered:
        outer_h = int(spec["inner_h"]) + 2
        outer_w = int(spec["inner_w"]) + 2
        placed = False
        for _ in range(250):
            r0 = random.randint(0, canvas_h - outer_h)
            c0 = random.randint(0, canvas_w - outer_w)
            if not _can_place(used, r0, c0, outer_h, outer_w):
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(canvas_h, r0 + outer_h + 1)
            cc1 = min(canvas_w, c0 + outer_w + 1)
            used[rr0:rr1, cc0:cc1] = True
            positions_by_id[id(spec)] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    for spec in specs:
        placements.append(positions_by_id[id(spec)])
    return placements


def _draw_region(
    grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, seed: int
) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    seed_r = r0 + 1 + random.randrange(inner_h)
    seed_c = c0 + 1 + random.randrange(inner_w)
    grid[seed_r, seed_c] = seed


def _expected_output(
    top_h: int, width: int, specs: list[dict[str, int | np.ndarray]], placements: list[tuple[int, int]]
) -> np.ndarray:
    out = np.full((top_h, width), BG, dtype=int)
    for spec, (r0, c0) in zip(specs, placements, strict=False):
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        seed = int(spec["seed"])
        pattern = np.asarray(spec["pattern"], dtype=int)
        _draw_region(out, r0, c0, inner_h, inner_w, seed)
        row = _tile_pattern(pattern, inner_w)
        out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = np.tile(row, (inner_h, 1))
    return out


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    region_count = random.randint(2, 5)
    seeds = random.sample(COLOR_POOL, region_count)
    specs: list[dict[str, int | np.ndarray]] = []
    for seed in seeds:
        specs.append(
            {
                "seed": seed,
                "inner_h": random.randint(1, 6),
                "inner_w": random.randint(1, 6),
                "pattern": _random_pattern(),
            }
        )

    min_width = max(max(int(spec["inner_w"]) + 4 for spec in specs), max(len(spec["pattern"]) + 2 for spec in specs))
    canvas_w = random.randint(min_width + 1, max(min_width + 1, 26))
    canvas_h = random.randint(max(9, region_count * 3), 18)
    placements = _place_regions(specs, canvas_h, canvas_w)
    if placements is None:
        return None

    top_h = max(r0 + int(spec["inner_h"]) + 2 for spec, (r0, _c0) in zip(specs, placements, strict=False))

    max_entry_w = max(len(spec["pattern"]) + 1 for spec in specs)
    legend_left = random.randint(0, canvas_w - max_entry_w)
    top_pad = random.randint(0, 1)
    bottom_pad = random.randint(0, 1)
    gaps = [random.randint(1, 2) for _ in range(region_count - 1)]
    legend_h = top_pad + bottom_pad + region_count + sum(gaps)
    total_h = top_h + legend_h

    inp = np.full((total_h, canvas_w), BG, dtype=int)
    for spec, (r0, c0) in zip(specs, placements, strict=False):
        if r0 + int(spec["inner_h"]) + 2 > top_h:
            return None
        _draw_region(inp, r0, c0, int(spec["inner_h"]), int(spec["inner_w"]), int(spec["seed"]))

    legend_entries = [(int(spec["seed"]), np.asarray(spec["pattern"], dtype=int)) for spec in specs]
    random.shuffle(legend_entries)
    row = top_h + top_pad
    for idx, (seed, pattern) in enumerate(legend_entries):
        inp[row, legend_left] = seed
        inp[row, legend_left + 1 : legend_left + 1 + len(pattern)] = pattern
        row += 1
        if idx < len(gaps):
            row += gaps[idx]

    expected = _expected_output(top_h, canvas_w, specs, placements)
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
