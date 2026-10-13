from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 8
ALL_COLORS = tuple(range(10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Generator settings.
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
FENCE_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
FENCE_H_RANGE = (4, 9)
FENCE_W_RANGE = (4, 9)
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400
MAX_FILL_COLORS = 3


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expand_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1)
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1)
    }


def _ray_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    rr, cc = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _ray_buffer(ray: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(ray)
    for r, c in ray:
        for dr, dc in DIRS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _gap_direction(
    gap: tuple[int, int],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[int, int] | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return SIDE_TO_DIR["top"]
    if gr == r1 and c0 < gc < c1:
        return SIDE_TO_DIR["bottom"]
    if gc == c0 and r0 < gr < r1:
        return SIDE_TO_DIR["left"]
    if gc == c1 and r0 < gr < r1:
        return SIDE_TO_DIR["right"]
    return None


def _analyze_fence_component(
    grid: np.ndarray,
    color: int,
    coords: list[tuple[int, int]],
    bg: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = _border_cells(r0, c0, r1, c1)
    comp = set(coords)
    if len(comp) != len(border) - 1 or not comp.issubset(border):
        return None

    missing = list(border - comp)
    if len(missing) != 1:
        return None
    gap = missing[0]
    direction = _gap_direction(gap, r0, c0, r1, c1)
    if direction is None or int(grid[gap]) != bg:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(interior == color):
        return None

    filled = interior[interior != bg]
    if filled.size == 0:
        return None

    counts = Counter(map(int, filled.tolist()))
    top_count = max(counts.values())
    winners = [fill_color for fill_color, cnt in counts.items() if cnt == top_count]
    if len(winners) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "count": int(filled.size),
        "ray_color": int(winners[0]),
    }


def _find_fences(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    fences: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            info = _analyze_fence_component(grid, color, comp, bg)
            if info is not None:
                fences.append(info)
    fences.sort(key=lambda item: item["bbox"])
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect gapped rectangular fences, count their interior colored pixels, and
    pour the dominant interior color outward as a ray through the gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = BG_COLOR if np.any(g == BG_COLOR) else _most_common_color(g)

    fences = _find_fences(g, bg)
    assert fences, "expected at least one gapped rectangular fence"

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        gap_r, gap_c = fence["gap"]
        dr, dc = fence["direction"]
        ray_color = int(fence["ray_color"])
        ray_length = int(fence["count"])

        out[r0 + 1 : r1, c0 + 1 : c1] = bg

        rr, cc = gap_r, gap_c
        for _ in range(ray_length):
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = ray_color
            rr += dr
            cc += dc

    return out


def _biased_int(lo: int, hi: int, toward_low: bool) -> int:
    a = random.randint(lo, hi)
    b = random.randint(lo, hi)
    return min(a, b) if toward_low else max(a, b)


def _sample_bbox(
    h: int,
    w: int,
    fence_h: int,
    fence_w: int,
    side: str,
    want_truncated: bool,
) -> tuple[int, int, int, int] | None:
    r_lo, r_hi = 1, h - fence_h - 1
    c_lo, c_hi = 1, w - fence_w - 1
    if r_lo > r_hi or c_lo > c_hi:
        return None

    if side == "top":
        r0 = _biased_int(r_lo, r_hi, toward_low=want_truncated)
        c0 = random.randint(c_lo, c_hi)
    elif side == "bottom":
        r0 = _biased_int(r_lo, r_hi, toward_low=not want_truncated)
        c0 = random.randint(c_lo, c_hi)
    elif side == "left":
        r0 = random.randint(r_lo, r_hi)
        c0 = _biased_int(c_lo, c_hi, toward_low=want_truncated)
    else:
        r0 = random.randint(r_lo, r_hi)
        c0 = _biased_int(c_lo, c_hi, toward_low=not want_truncated)

    return r0, c0, r0 + fence_h - 1, c0 + fence_w - 1


def _gap_for_side(
    bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        gap_c = random.randint(c0 + 1, c1 - 1)
        gap = (r0 if side == "top" else r1, gap_c)
    else:
        gap_r = random.randint(r0 + 1, r1 - 1)
        gap = (gap_r, c0 if side == "left" else c1)
    return gap, SIDE_TO_DIR[side]


def _available_ray_length(
    bbox: tuple[int, int, int, int],
    side: str,
    h: int,
    w: int,
) -> int:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 + 1
    if side == "bottom":
        return h - r1
    if side == "left":
        return c0 + 1
    return w - c1


def _sample_fill_spec(count: int, fill_pool: list[int]) -> tuple[int, list[tuple[int, int]]]:
    max_palette = min(MAX_FILL_COLORS, len(fill_pool), count - (count // 2))
    if count <= 2 or max_palette <= 1:
        palette_size = 1
    else:
        palette_size = random.randint(1, max_palette)

    colors = random.sample(fill_pool, palette_size)
    if palette_size == 1:
        return colors[0], [(colors[0], count)]

    major = random.randint(count // 2 + 1, count - (palette_size - 1))
    remainder = count - major
    other_counts = [1] * (palette_size - 1)
    for _ in range(remainder - (palette_size - 1)):
        other_counts[random.randrange(palette_size - 1)] += 1

    assignments = [(colors[0], major)]
    assignments.extend((color, cnt) for color, cnt in zip(colors[1:], other_counts, strict=False))
    return colors[0], assignments


def _expected_output(grid: np.ndarray, fences: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        gap_r, gap_c = fence["gap"]
        dr, dc = fence["direction"]
        ray_color = int(fence["ray_color"])
        ray_length = int(fence["count"])

        out[r0 + 1 : r1, c0 + 1 : c1] = BG_COLOR
        rr, cc = gap_r, gap_c
        for _ in range(ray_length):
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = ray_color
            rr += dr
            cc += dc
    return out


def _generate_scene() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    fence_count = random.choice(FENCE_COUNT_CHOICES)

    fence_palette = [color for color in ALL_COLORS if color != BG_COLOR]
    if fence_count > len(fence_palette):
        return None
    fence_colors = random.sample(fence_palette, fence_count)
    fill_pool = [color for color in fence_palette if color not in fence_colors]
    if not fill_pool:
        return None

    blocked: set[tuple[int, int]] = set()
    fence_specs: list[dict[str, object]] = []
    for fence_color in fence_colors:
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            side = random.choice(tuple(SIDE_TO_DIR))
            want_truncated = random.choice([False, True])
            fence_h = random.randint(FENCE_H_RANGE[0], min(FENCE_H_RANGE[1], h - 2))
            fence_w = random.randint(FENCE_W_RANGE[0], min(FENCE_W_RANGE[1], w - 2))
            bbox = _sample_bbox(h, w, fence_h, fence_w, side, want_truncated)
            if bbox is None:
                continue

            area = (fence_h - 2) * (fence_w - 2)
            available = _available_ray_length(bbox, side, h, w)
            if want_truncated:
                if area <= available:
                    continue
                fill_count = random.randint(available + 1, area)
            else:
                if min(area, available) < 1:
                    continue
                fill_count = random.randint(1, min(area, available))

            gap, direction = _gap_for_side(bbox, side)
            drawn_ray = _ray_cells(gap, direction, min(fill_count, available), h, w)
            footprint = _expand_bbox(*bbox, h, w) | _ray_buffer(drawn_ray, h, w)
            if footprint & blocked:
                continue

            ray_color, color_counts = _sample_fill_spec(fill_count, fill_pool)
            placed = {
                "bbox": bbox,
                "gap": gap,
                "direction": direction,
                "count": fill_count,
                "ray_color": ray_color,
                "fence_color": fence_color,
                "color_counts": color_counts,
            }
            blocked |= footprint
            break

        if placed is None:
            return None
        fence_specs.append(placed)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for fence in fence_specs:
        r0, c0, r1, c1 = fence["bbox"]
        gap = fence["gap"]
        fence_color = int(fence["fence_color"])

        for cell in _border_cells(r0, c0, r1, c1):
            if cell != gap:
                grid[cell] = fence_color

        interior_cells = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
        ]
        chosen = random.sample(interior_cells, int(fence["count"]))
        random.shuffle(chosen)
        idx = 0
        for fill_color, amount in fence["color_counts"]:
            for cell in chosen[idx : idx + amount]:
                grid[cell] = fill_color
            idx += amount

    expected = _expected_output(grid, fence_specs)

    if np.count_nonzero(grid == BG_COLOR) <= grid.size // 2:
        return None
    if len(_find_fences(grid, BG_COLOR)) != fence_count:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
