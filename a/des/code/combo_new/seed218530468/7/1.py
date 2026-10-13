import random
from collections import Counter

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _color_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    current = int(row[0])
    for i in range(1, len(row) + 1):
        if i == len(row) or int(row[i]) != current:
            runs.append((current, i - start))
            if i < len(row):
                start = i
                current = int(row[i])
    return runs


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, dict[int, int]]:
    h, _ = grid.shape
    fallback = None
    for r in range(h):
        cols = np.flatnonzero(grid[r] != bg)
        if cols.size == 0:
            continue
        c0, c1 = int(cols[0]), int(cols[-1])
        if np.any(grid[r, c0 : c1 + 1] == bg):
            continue
        runs = _color_runs(grid[r, c0 : c1 + 1])
        if len(runs) < 2:
            continue
        if fallback is None:
            fallback = (r, c0, c1, {color: length for color, length in runs})
        if r + 1 < h and np.all(grid[r + 1, c0 : c1 + 1] == bg):
            return r, c0, c1, {color: length for color, length in runs}
    assert fallback is not None, "legend row not found"
    return fallback


def _find_border_color(
    grid: np.ndarray, bg: int, legend_colors: set[int], legend_mask: np.ndarray
) -> int:
    h, w = grid.shape
    scores: Counter[int] = Counter()
    for color in legend_colors:
        for r, c in np.argwhere((grid == color) & (~legend_mask)):
            counts: Counter[int] = Counter()
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    value = int(grid[nr, nc])
                    if value != bg and value not in legend_colors:
                        counts[value] += 1
            for candidate, ct in counts.items():
                if ct >= 2:
                    scores[candidate] += ct
    if scores:
        return max(
            scores,
            key=lambda color: (scores[color], int(np.count_nonzero((grid == color) & (~legend_mask)))),
        )

    non_bg = [
        int(v)
        for v in grid[~legend_mask].ravel()
        if int(v) != bg and int(v) not in legend_colors
    ]
    assert non_bg, "border color not found"
    return Counter(non_bg).most_common(1)[0][0]


def _marker_orientations(grid: np.ndarray, r: int, c: int, border_color: int) -> list[str]:
    h, w = grid.shape
    options = {
        "tl": ((0, 1), (1, 0)),
        "tr": ((0, -1), (1, 0)),
        "bl": ((0, 1), (-1, 0)),
        "br": ((0, -1), (-1, 0)),
    }
    hits: list[str] = []
    for name, ((dr1, dc1), (dr2, dc2)) in options.items():
        r1, c1 = r + dr1, c + dc1
        r2, c2 = r + dr2, c + dc2
        if not (0 <= r1 < h and 0 <= c1 < w and 0 <= r2 < h and 0 <= c2 < w):
            continue
        if int(grid[r1, c1]) == border_color and int(grid[r2, c2]) == border_color:
            hits.append(name)
    return hits


def _scan_frame_bounds(
    grid: np.ndarray, r: int, c: int, orientation: str, border_color: int
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    if orientation == "tl":
        r0, c0 = r, c
        c1 = c
        while c1 + 1 < w and int(grid[r0, c1 + 1]) == border_color:
            c1 += 1
        r1 = r
        while r1 + 1 < h and int(grid[r1 + 1, c0]) == border_color:
            r1 += 1
    elif orientation == "tr":
        r0, c1 = r, c
        c0 = c
        while c0 - 1 >= 0 and int(grid[r0, c0 - 1]) == border_color:
            c0 -= 1
        r1 = r
        while r1 + 1 < h and int(grid[r1 + 1, c1]) == border_color:
            r1 += 1
    elif orientation == "bl":
        r1, c0 = r, c
        c1 = c
        while c1 + 1 < w and int(grid[r1, c1 + 1]) == border_color:
            c1 += 1
        r0 = r
        while r0 - 1 >= 0 and int(grid[r0 - 1, c0]) == border_color:
            r0 -= 1
    else:
        r1, c1 = r, c
        c0 = c
        while c0 - 1 >= 0 and int(grid[r1, c0 - 1]) == border_color:
            c0 -= 1
        r0 = r
        while r0 - 1 >= 0 and int(grid[r0 - 1, c1]) == border_color:
            r0 -= 1
    return r0, c0, r1, c1


def _valid_frame(
    grid: np.ndarray, bounds: tuple[int, int, int, int], border_color: int, marker_color: int
) -> bool:
    r0, c0, r1, c1 = bounds
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    marker_hits = 0
    for rr, cc in corners:
        value = int(grid[rr, cc])
        if value == marker_color:
            marker_hits += 1
        elif value != border_color:
            return False
    if marker_hits != 1:
        return False

    if np.any(grid[r0, c0 + 1 : c1] != border_color):
        return False
    if np.any(grid[r1, c0 + 1 : c1] != border_color):
        return False
    if np.any(grid[r0 + 1 : r1, c0] != border_color):
        return False
    if np.any(grid[r0 + 1 : r1, c1] != border_color):
        return False
    return True


def _extract_seed_patch(interior: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(interior != bg)
    assert pts.size > 0, "region must contain a seed pattern"
    max_r = int(pts[:, 0].max())
    max_c = int(pts[:, 1].max())
    return interior[: max_r + 1, : max_c + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top color band as scale factors, match them to corner markers, and enlarge each region's seed pattern.

    Input structure:
    - The background is the most common color.
    - A one-row horizontal color-band legend sits near the top, with each legend color appearing in one contiguous run of length N.
    - Lower in the grid are several grey rectangular frames; each frame has exactly one colored corner marker on its border and a small multicolor seed patch near the frame interior's top-left corner.

    Rule:
    - Interpret each legend run length N as the integer scale factor for that run's color.
    - For every framed region, find its colored corner marker, look up that color in the legend, and obtain the matching scale factor.
    - Take the region's seed patch, enlarge it by that factor using block scaling, write the scaled patch into the frame interior from the top-left, and leave any remaining interior cells as background.
    - Remove the legend from the output by replacing its cells with the background color.

    Key insight:
    - The puzzle is solved in two linked steps: the legend converts visible run lengths into numeric scale factors, and the corner-marker color chooses which factor to apply to each enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    legend_row, legend_c0, legend_c1, scale_by_color = _find_legend(g, bg)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[legend_row, legend_c0 : legend_c1 + 1] = True

    work = g.copy()
    work[legend_mask] = bg

    legend_colors = set(scale_by_color)
    border_color = _find_border_color(work, bg, legend_colors, legend_mask)

    out = work.copy()
    for marker_color, scale in scale_by_color.items():
        candidates: list[tuple[int, tuple[int, int, int, int]]] = []
        for r, c in np.argwhere((work == marker_color) & (~legend_mask)):
            for orientation in _marker_orientations(work, int(r), int(c), border_color):
                bounds = _scan_frame_bounds(work, int(r), int(c), orientation, border_color)
                if _valid_frame(work, bounds, border_color, marker_color):
                    r0, c0, r1, c1 = bounds
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    candidates.append((area, bounds))
        assert candidates, f"frame for marker color {marker_color} not found"
        _, (r0, c0, r1, c1) = max(candidates, key=lambda item: item[0])

        interior = work[r0 + 1 : r1, c0 + 1 : c1]
        seed = _extract_seed_patch(interior, bg)
        scaled = np.repeat(np.repeat(seed, scale, axis=0), scale, axis=1)

        filled = np.full(interior.shape, bg, dtype=int)
        hh = min(filled.shape[0], scaled.shape[0])
        ww = min(filled.shape[1], scaled.shape[1])
        filled[:hh, :ww] = scaled[:hh, :ww]
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

    return out


def _random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    path = ["D"] * (h - 1) + ["R"] * (w - 1)
    rng.shuffle(path)
    r = c = 0
    mask[r, c] = True
    for step in path:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    target = rng.randint(int(mask.sum()), h * w)
    while int(mask.sum()) < target:
        frontier = []
        for rr in range(h):
            for cc in range(w):
                if mask[rr, cc]:
                    continue
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((rr, cc))
                        break
        if not frontier:
            break
        rr, cc = rng.choice(frontier)
        mask[rr, cc] = True
    return mask


def _paint_seed_patch(
    rng: random.Random, mask: np.ndarray, bg: int, seed_palette: list[int]
) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    occupied = [(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if mask[r, c]]
    color_count = min(len(seed_palette), max(2, rng.randint(2, 4)))
    chosen = rng.sample(seed_palette, color_count)
    used: set[int] = set()
    for r, c in occupied:
        color = rng.choice(chosen)
        patch[r, c] = color
        used.add(color)
    if len(used) == 1 and len(chosen) >= 2:
        rr, cc = occupied[-1]
        patch[rr, cc] = chosen[1]
    return patch


def _make_region_spec(
    rng: random.Random, scale: int, marker_color: int, bg: int, border_color: int, seed_palette: list[int]
) -> dict:
    max_dim = 4 if scale <= 3 else 3
    seed_h = rng.randint(2, max_dim)
    seed_w = rng.randint(2, max_dim)
    seed_mask = _random_connected_mask(rng, seed_h, seed_w)
    seed_patch = _paint_seed_patch(rng, seed_mask, bg, seed_palette)

    extra_h = rng.randint(0, 3)
    extra_w = rng.randint(0, 3)
    interior_h = seed_h * scale + extra_h
    interior_w = seed_w * scale + extra_w

    return {
        "scale": scale,
        "marker_color": marker_color,
        "corner": rng.choice(["tl", "tr", "bl", "br"]),
        "seed_patch": seed_patch,
        "border_color": border_color,
        "bg": bg,
        "outer_h": interior_h + 2,
        "outer_w": interior_w + 2,
    }


def _choose_canvas(rng: random.Random, specs: list[dict]) -> tuple[int, int] | None:
    legend_w = sum(spec["scale"] for spec in specs)
    area = sum(spec["outer_h"] * spec["outer_w"] for spec in specs)
    max_h = max(spec["outer_h"] for spec in specs)
    max_w = max(spec["outer_w"] for spec in specs)

    min_w = max(legend_w + 2, max_w + 2, int(np.ceil(np.sqrt(area))) + 2, 14)
    if min_w > 30:
        return None
    w = rng.randint(min_w, 30)

    min_h = max(max_h + 4, int(np.ceil(area / max(1, w - 2))) + 5, 12)
    if min_h > 30:
        return None
    h = rng.randint(min_h, 30)
    return h, w


def _place_regions(rng: random.Random, h: int, w: int, specs: list[dict]) -> list[tuple[int, int]] | None:
    occ = np.zeros((h, w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)

    for idx in order:
        outer_h = specs[idx]["outer_h"]
        outer_w = specs[idx]["outer_w"]
        options: list[tuple[int, int]] = []
        for r in range(3, h - outer_h + 1):
            for c in range(0, w - outer_w + 1):
                rr0 = max(0, r - 1)
                rr1 = min(h, r + outer_h + 1)
                cc0 = max(0, c - 1)
                cc1 = min(w, c + outer_w + 1)
                if not occ[rr0:rr1, cc0:cc1].any():
                    options.append((r, c))
        if not options:
            return None
        placements[idx] = rng.choice(options)
        r, c = placements[idx]
        occ[r : r + outer_h, c : c + outer_w] = True

    return [placements[i] for i in range(len(specs))]


def _draw_region(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    border_color = spec["border_color"]
    marker_color = spec["marker_color"]
    bg = spec["bg"]

    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color

    if spec["corner"] == "tl":
        grid[r0, c0] = marker_color
    elif spec["corner"] == "tr":
        grid[r0, c1] = marker_color
    elif spec["corner"] == "bl":
        grid[r1, c0] = marker_color
    else:
        grid[r1, c1] = marker_color

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[:, :] = bg
    seed_patch = spec["seed_patch"]
    sh, sw = seed_patch.shape
    interior[:sh, :sw] = seed_patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(2, 4)
    scale_pool = [2, 3, 4, 5] if region_count <= 3 else [2, 3, 4]

    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    border_color = rng.choice(colors)
    colors.remove(border_color)
    rng.shuffle(colors)
    marker_colors = colors[:region_count]
    seed_palette = colors[region_count:]
    if len(seed_palette) < 2:
        return None

    specs = [
        _make_region_spec(rng, rng.choice(scale_pool), marker_colors[i], bg, border_color, seed_palette)
        for i in range(region_count)
    ]

    dims = _choose_canvas(rng, specs)
    if dims is None:
        return None
    h, w = dims

    placements = _place_regions(rng, h, w, specs)
    if placements is None:
        return None

    legend_width = sum(spec["scale"] for spec in specs)
    if legend_width > w:
        return None

    grid = np.full((h, w), bg, dtype=int)
    legend_order = list(range(region_count))
    rng.shuffle(legend_order)
    legend_left = rng.randint(0, w - legend_width)
    cursor = legend_left
    for idx in legend_order:
        scale = specs[idx]["scale"]
        color = specs[idx]["marker_color"]
        grid[0, cursor : cursor + scale] = color
        cursor += scale

    for spec, (top, left) in zip(specs, placements):
        _draw_region(grid, spec, top, left)

    body = grid.copy()
    body[0, legend_left : legend_left + legend_width] = bg
    counts = Counter(int(v) for v in body.ravel() if int(v) != bg)
    if not counts or counts.most_common(1)[0][0] != border_color:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
