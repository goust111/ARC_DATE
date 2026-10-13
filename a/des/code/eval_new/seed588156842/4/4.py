from __future__ import annotations

from collections import Counter, deque
from itertools import product
import random

import numpy as np


BG = 0
RING_COLOR = 3
PALETTE_CHOICES = [1, 2, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")

GRID_MIN = 14
GRID_MAX = 30
LAYER_MIN = 2
LAYER_MAX = 5
INNER_MIN = 5
SPACING_CHOICES = (2, 3)
GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat)
    return int(max(counts, key=counts.get))


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    top, left, bottom, right = bbox
    coords: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        coords.append((top, c))
    for r in range(top + 1, bottom):
        coords.append((r, right))
    for c in range(right, left - 1, -1):
        coords.append((bottom, c))
    for r in range(bottom - 1, top, -1):
        coords.append((r, left))
    return coords


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    top, left, bottom, right = bbox
    if r == top and left < c < right:
        return "top"
    if r == bottom and left < c < right:
        return "bottom"
    if c == left and top < r < bottom:
        return "left"
    if c == right and top < r < bottom:
        return "right"
    raise ValueError("gap must lie on a non-corner border cell")


def _straight_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("bridge endpoints must be axis-aligned")


def _extract_palette(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    counts = Counter(int(v) for v in grid.flat)
    ring_candidates = [color for color, count in counts.items() if color != bg and count > 1]
    if len(ring_candidates) != 1:
        raise ValueError("expected exactly one repeated non-background color")
    ring_color = int(ring_candidates[0])

    palette_cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != bg) & (grid != ring_color)):
        palette_cells.append((int(r), int(c), int(grid[r, c])))
    if not palette_cells:
        raise ValueError("missing palette strip")

    rows = {r for r, _, _ in palette_cells}
    if len(rows) != 1:
        raise ValueError("palette strip must be horizontal")
    palette_cells.sort(key=lambda item: item[1])
    cols = [c for _, c, _ in palette_cells]
    if any(cols[i + 1] != cols[i] + 1 for i in range(len(cols) - 1)):
        raise ValueError("palette strip must be contiguous")
    return ring_color, [color for _, _, color in palette_cells]


def _find_rings(grid: np.ndarray, ring_color: int) -> list[dict]:
    rings: list[dict] = []
    for comp in _components(grid == ring_color):
        bbox = _bbox(comp)
        border = _border_coords(bbox)
        border_set = set(border)
        comp_set = set(comp)
        if not comp_set.issubset(border_set):
            continue
        if len(comp_set) != len(border) - 1:
            continue
        missing = list(border_set - comp_set)
        if len(missing) != 1:
            continue
        gap = missing[0]
        rings.append(
            {
                "coords": comp,
                "bbox": bbox,
                "gap": gap,
                "side": _gap_side(gap, bbox),
            }
        )
    rings.sort(
        key=lambda ring: (
            -(ring["bbox"][2] - ring["bbox"][0] + 1) * (ring["bbox"][3] - ring["bbox"][1] + 1),
            ring["bbox"][0],
            ring["bbox"][1],
        )
    )
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular ring tracks by layer and bridge aligned gaps across adjacent rings.

    Input structure:
    - The background is black.
    - One repeated foreground color forms several strictly nested 1-pixel rectangular rings, with each ring missing exactly one non-corner border cell.
    - A horizontal palette strip sits in a corner: distinct single pixels ordered left-to-right give the colors for outer-to-inner rings.

    Rule:
    - Identify the nested ring components and sort them from outermost to innermost.
    - Recolor every ring's existing track pixels with the corresponding palette color at the same layer index.
    - For each adjacent ring pair, if their missing cells are on the same side and are aligned on one straight row or column, draw the bridge between those two gaps using the outer ring's color.

    Key insight:
    - The task combines two hidden codes at once: the corner palette specifies layer order, while each ring's single missing border cell specifies both bridge eligibility and bridge direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    ring_color, palette = _extract_palette(g, bg)
    rings = _find_rings(g, ring_color)
    if len(rings) != len(palette):
        raise ValueError("palette length must match ring count")

    out = g.copy()
    for ring, color in zip(rings, palette):
        rows, cols = zip(*ring["coords"])
        out[rows, cols] = int(color)

    for idx in range(len(rings) - 1):
        outer = rings[idx]
        inner = rings[idx + 1]
        if outer["side"] != inner["side"]:
            continue
        if outer["side"] in ("top", "bottom") and outer["gap"][1] != inner["gap"][1]:
            continue
        if outer["side"] in ("left", "right") and outer["gap"][0] != inner["gap"][0]:
            continue
        for r, c in _straight_line(outer["gap"], inner["gap"]):
            out[r, c] = int(palette[idx])

    return out


def _valid_gap_values(bbox: tuple[int, int, int, int], side: str) -> list[int]:
    top, left, bottom, right = bbox
    if side in ("top", "bottom"):
        return list(range(left + 1, right))
    return list(range(top + 1, bottom))


def _gap_from_value(bbox: tuple[int, int, int, int], side: str, value: int) -> tuple[int, int]:
    top, left, bottom, right = bbox
    if side == "top":
        return top, value
    if side == "bottom":
        return bottom, value
    if side == "left":
        return value, left
    return value, right


def _build_side_sequence(n_layers: int) -> list[str]:
    sides: list[str] = []
    for i in range(n_layers):
        if i == 0:
            sides.append(random.choice(SIDES))
            continue
        can_repeat = not (i >= 2 and sides[i - 1] == sides[i - 2])
        if can_repeat and random.random() < 0.5:
            sides.append(sides[i - 1])
        else:
            sides.append(random.choice([side for side in SIDES if side != sides[i - 1]]))
    return sides


def _build_input(
    shape: tuple[int, int],
    corner: str,
    bboxes: list[tuple[int, int, int, int]],
    gaps: list[tuple[int, int]],
    palette: list[int],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)

    row = 0 if corner.startswith("top") else h - 1
    start = 0 if corner.endswith("left") else w - len(palette)
    for idx, color in enumerate(palette):
        grid[row, start + idx] = int(color)

    for bbox, gap in zip(bboxes, gaps):
        for cell in _border_coords(bbox):
            if cell != gap:
                grid[cell] = RING_COLOR
    return grid


def _expected_output(
    inp: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    gaps: list[tuple[int, int]],
    sides: list[str],
    palette: list[int],
) -> np.ndarray:
    out = inp.copy()
    for bbox, gap, color in zip(bboxes, gaps, palette):
        for cell in _border_coords(bbox):
            if cell != gap:
                out[cell] = int(color)
    for i in range(len(bboxes) - 1):
        if sides[i] != sides[i + 1]:
            continue
        for r, c in _straight_line(gaps[i], gaps[i + 1]):
            out[r, c] = int(palette[i])
    return out


def _sample_spec() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    corner = random.choice(CORNERS)

    top_min = 2 if corner.startswith("top") else 1
    bottom_min = 2 if corner.startswith("bottom") else 1
    left_min = 1
    right_min = 1
    avail_h = h - top_min - bottom_min
    avail_w = w - left_min - right_min

    feasible_layers: list[int] = []
    for n_layers in range(LAYER_MIN, LAYER_MAX + 1):
        min_outer = INNER_MIN + 4 * (n_layers - 1)
        if min_outer <= avail_h and min_outer <= avail_w:
            feasible_layers.append(n_layers)
    if not feasible_layers:
        return None

    n_layers = random.choice(feasible_layers)
    max_inset = min((avail_h - INNER_MIN) // 2, (avail_w - INNER_MIN) // 2)
    spacing_options = [
        combo
        for combo in product(SPACING_CHOICES, repeat=n_layers - 1)
        if sum(combo) <= max_inset
    ]
    if not spacing_options:
        return None
    spacings = list(random.choice(spacing_options))
    total_inset = sum(spacings)

    inner_h_max = avail_h - 2 * total_inset
    inner_w_max = avail_w - 2 * total_inset
    if inner_h_max < INNER_MIN or inner_w_max < INNER_MIN:
        return None
    inner_h = random.randint(INNER_MIN, inner_h_max)
    inner_w = random.randint(INNER_MIN, inner_w_max)
    outer_h = inner_h + 2 * total_inset
    outer_w = inner_w + 2 * total_inset

    top = random.randint(top_min, h - outer_h - bottom_min)
    left = random.randint(left_min, w - outer_w - right_min)
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    offsets = [0]
    running = 0
    for spacing in spacings:
        running += spacing
        offsets.append(running)
    bboxes = [
        (top + inset, left + inset, bottom - inset, right - inset)
        for inset in offsets
    ]

    sides = _build_side_sequence(n_layers)
    gap_values: list[int] = []
    gaps: list[tuple[int, int]] = []
    for i in range(n_layers):
        side = sides[i]
        if i > 0 and side == sides[i - 1]:
            value = gap_values[i - 1]
        else:
            candidates = _valid_gap_values(bboxes[i], side)
            if i + 1 < n_layers and sides[i + 1] == side:
                inner_candidates = set(_valid_gap_values(bboxes[i + 1], side))
                candidates = [value for value in candidates if value in inner_candidates]
            if not candidates:
                return None
            value = random.choice(candidates)
        gap_values.append(value)
        gaps.append(_gap_from_value(bboxes[i], side, value))

    palette = random.sample(PALETTE_CHOICES, n_layers)
    inp = _build_input((h, w), corner, bboxes, gaps, palette)
    out = _expected_output(inp, bboxes, gaps, sides, palette)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _sample_spec()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
