from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEGEND_THICKNESS = 3
LEGEND_GAP = 1
MIN_REGIONS = 2
MAX_REGIONS = 4
MIN_INNER = 3
MAX_INNER_SMALL = 5
MAX_INNER_LARGE = 6
GRID_MIN = 20
GRID_MAX = 28
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400
COLORS = tuple(range(10))


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_regions(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    regions: list[dict[str, int]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            frame = _perimeter_cells(r0, c0, r1, c1)
            if set(comp) != frame:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            non_bg = np.argwhere(interior != bg)
            if non_bg.shape[0] != 1:
                continue
            sr = int(non_bg[0, 0] + r0 + 1)
            sc = int(non_bg[0, 1] + c0 + 1)
            seed_color = int(grid[sr, sc])
            if seed_color == color:
                continue

            regions.append(
                {
                    "border_color": int(color),
                    "seed_color": int(seed_color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "seed_r": int(sr),
                    "seed_c": int(sc),
                }
            )
    return regions


def _edge_strip(shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[:LEGEND_THICKNESS, :] = True
    elif side == "bottom":
        mask[h - LEGEND_THICKNESS :, :] = True
    elif side == "left":
        mask[:, :LEGEND_THICKNESS] = True
    else:
        mask[:, w - LEGEND_THICKNESS :] = True
    return mask


def _legend_mapping(grid: np.ndarray, bg: int, seed_colors: set[int]) -> dict[int, tuple[int, int]]:
    candidates: list[tuple[tuple[int, int], dict[int, tuple[int, int]]]] = []

    for side in ("top", "bottom", "left", "right"):
        strip = _edge_strip(grid.shape, side)
        comps = _components((grid != bg) & strip)
        if not comps:
            continue
        if any(len(comp) != 2 for comp in comps):
            continue

        strip_colors = [int(grid[r, c]) for comp in comps for r, c in comp]
        counts = Counter(strip_colors)
        shared = [color for color, count in counts.items() if count == len(comps)]

        mapping: dict[int, tuple[int, int]] = {}
        valid = True
        for comp in comps:
            (r0, c0), (r1, c1) = comp
            color0 = int(grid[r0, c0])
            color1 = int(grid[r1, c1])

            key_cell: tuple[int, int] | None = None
            other_cell: tuple[int, int] | None = None
            if (color0 in seed_colors) ^ (color1 in seed_colors):
                key_cell = (r0, c0) if color0 in seed_colors else (r1, c1)
                other_cell = (r1, c1) if color0 in seed_colors else (r0, c0)
            elif len(shared) == 1:
                marker = shared[0]
                if color0 == marker and color1 != marker:
                    key_cell = (r1, c1)
                    other_cell = (r0, c0)
                elif color1 == marker and color0 != marker:
                    key_cell = (r0, c0)
                    other_cell = (r1, c1)

            if key_cell is None or other_cell is None:
                valid = False
                break

            dr = int(other_cell[0] - key_cell[0])
            dc = int(other_cell[1] - key_cell[1])
            if abs(dr) + abs(dc) != 1:
                valid = False
                break

            seed = int(grid[key_cell[0], key_cell[1]])
            if seed in mapping:
                valid = False
                break
            mapping[seed] = (dr, dc)

        if valid and seed_colors <= set(mapping):
            score = (len(seed_colors & set(mapping)), -abs(len(mapping) - len(seed_colors)))
            candidates.append((score, mapping))

    assert candidates, "expected a valid edge legend"
    best = max(candidates, key=lambda item: item[0])[1]
    return {seed: best[seed] for seed in seed_colors}


def _paint_half(out: np.ndarray, region: dict[str, int], direction: tuple[int, int]) -> None:
    r0 = region["r0"]
    c0 = region["c0"]
    r1 = region["r1"]
    c1 = region["c1"]
    sr = region["seed_r"]
    sc = region["seed_c"]
    border = region["border_color"]
    dr, dc = direction

    if dr == -1:
        out[r0 + 1 : sr + 1, c0 + 1 : c1] = border
    elif dr == 1:
        out[sr:r1, c0 + 1 : c1] = border
    elif dc == -1:
        out[r0 + 1 : r1, c0 + 1 : sc + 1] = border
    else:
        out[r0 + 1 : r1, sc:c1] = border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a seed-color legend on the grid edge to decide which half of each hollow rectangle to fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    regions = _find_regions(g, bg)
    assert regions, "expected at least one closed rectangular region"

    seed_colors = {region["seed_color"] for region in regions}
    mapping = _legend_mapping(g, bg, seed_colors)

    out = g.copy()
    for region in regions:
        _paint_half(out, region, mapping[region["seed_color"]])
    return out


def _body_bounds(h: int, w: int, side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = 1, 1, h - 2, w - 2
    if side == "top":
        r0 = LEGEND_THICKNESS + LEGEND_GAP
    elif side == "bottom":
        r1 = h - LEGEND_THICKNESS - LEGEND_GAP - 1
    elif side == "left":
        c0 = LEGEND_THICKNESS + LEGEND_GAP
    else:
        c1 = w - LEGEND_THICKNESS - LEGEND_GAP - 1
    return r0, c0, r1, c1


def _sample_spaced_positions(length: int, count: int) -> list[int] | None:
    choices = list(range(3, length - 3))
    random.shuffle(choices)
    picked: list[int] = []
    for value in choices:
        if all(abs(value - prev) >= 4 for prev in picked):
            picked.append(int(value))
            if len(picked) == count:
                return sorted(picked)
    return None


def _place_legend(
    grid: np.ndarray,
    side: str,
    marker_color: int,
    entries: list[tuple[int, tuple[int, int]]],
) -> bool:
    h, w = grid.shape
    axis_len = w if side in {"top", "bottom"} else h
    positions = _sample_spaced_positions(axis_len, len(entries))
    if positions is None:
        return False

    for (seed_color, (dr, dc)), pos in zip(entries, positions):
        if side == "top":
            key = (1, pos)
        elif side == "bottom":
            key = (h - 2, pos)
        elif side == "left":
            key = (pos, 1)
        else:
            key = (pos, w - 2)

        marker = (key[0] + dr, key[1] + dc)
        if not (0 <= marker[0] < h and 0 <= marker[1] < w):
            return False
        if grid[key] != grid.flat[0] or grid[marker] != grid.flat[0]:
            return False
        grid[key] = int(seed_color)
        grid[marker] = int(marker_color)
    return True


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _build_regions(
    grid: np.ndarray,
    side: str,
    border_colors: list[int],
    seed_colors: list[int],
    directions: list[tuple[int, int]],
) -> list[dict[str, int]] | None:
    h, w = grid.shape
    body_r0, body_c0, body_r1, body_c1 = _body_bounds(h, w, side)
    region_count = len(border_colors)
    max_inner = MAX_INNER_LARGE if region_count <= 3 else MAX_INNER_SMALL

    specs = []
    for border_color, seed_color, direction in zip(border_colors, seed_colors, directions):
        inner_h = random.randint(MIN_INNER, max_inner)
        inner_w = random.randint(MIN_INNER, max_inner)
        specs.append(
            {
                "border_color": int(border_color),
                "seed_color": int(seed_color),
                "dr": int(direction[0]),
                "dc": int(direction[1]),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
            }
        )
    specs.sort(key=lambda spec: (spec["outer_h"] * spec["outer_w"]), reverse=True)

    placed_boxes: list[tuple[int, int, int, int]] = []
    regions: list[dict[str, int]] = []
    bg = int(grid[0, 0])

    for spec in specs:
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        if body_r1 - body_r0 + 1 < outer_h or body_c1 - body_c0 + 1 < outer_w:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(body_r0, body_r1 - outer_h + 1)
            c0 = random.randint(body_c0, body_c1 - outer_w + 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            box = (r0, c0, r1, c1)
            if any(_boxes_touch(box, other) for other in placed_boxes):
                continue

            seed_r = r0 + 1 + random.randint(0, spec["inner_h"] - 1)
            seed_c = c0 + 1 + random.randint(0, spec["inner_w"] - 1)

            grid[r0 : r1 + 1, c0] = spec["border_color"]
            grid[r0 : r1 + 1, c1] = spec["border_color"]
            grid[r0, c0 : c1 + 1] = spec["border_color"]
            grid[r1, c0 : c1 + 1] = spec["border_color"]
            grid[seed_r, seed_c] = spec["seed_color"]

            placed_boxes.append(box)
            regions.append(
                {
                    "border_color": spec["border_color"],
                    "seed_color": spec["seed_color"],
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "seed_r": int(seed_r),
                    "seed_c": int(seed_c),
                    "dr": spec["dr"],
                    "dc": spec["dc"],
                }
            )
            placed = True
            break

        if not placed:
            grid[:, :] = bg
            return None

    return regions


def _expected_output(inp: np.ndarray, regions: list[dict[str, int]]) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        _paint_half(out, region, (region["dr"], region["dc"]))
    return out


def _valid_background_dominance(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = counts.pop(int(bg), 0)
    return bool(counts) and bg_count > max(counts.values())


def _generate_once() -> np.ndarray | None:
    region_count = random.randint(MIN_REGIONS, MAX_REGIONS)
    side = random.choice(("top", "bottom", "left", "right"))

    min_axis = max(GRID_MIN, 4 * region_count + 4)
    if side in {"top", "bottom"}:
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(min_axis, GRID_MAX)
    else:
        h = random.randint(min_axis, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)

    color_pool = list(COLORS)
    random.shuffle(color_pool)
    bg = int(color_pool.pop())
    marker_color = int(color_pool.pop())
    border_colors = [int(color_pool.pop()) for _ in range(region_count)]
    seed_colors = [int(color_pool.pop()) for _ in range(region_count)]
    directions = random.sample(DIRECTIONS, region_count)

    grid = np.full((h, w), bg, dtype=int)
    regions = _build_regions(grid, side, border_colors, seed_colors, directions)
    if regions is None:
        return None

    entries = list(zip(seed_colors, directions))
    random.shuffle(entries)
    if not _place_legend(grid, side, marker_color, entries):
        return None

    if not _valid_background_dominance(grid, bg):
        return None

    expected = _expected_output(grid, regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    found_regions = _find_regions(grid, bg)
    if len(found_regions) != region_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
