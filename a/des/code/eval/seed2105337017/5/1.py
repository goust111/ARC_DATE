from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))

    if set(coords) != expected:
        return None
    return r0, c0, r1, c1


def _tile_to_shape(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    ph, pw = pattern.shape
    reps_r = (height + ph - 1) // ph
    reps_c = (width + pw - 1) // pw
    return np.tile(pattern, (reps_r, reps_c))[:height, :width]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular fence with the tiled pattern from the same-colored palette block.

    Input structure:
    - bg is the most common color.
    - Several closed one-pixel-thick rectangular fences enclose bg interiors.
    - A palette area contains small framed blocks whose border color matches a fence color.
      The palette interior is a 2x2 or 3x3 non-bg pattern.

    Rule:
    - Detect monochrome rectangular frames.
    - Frames whose interior is all bg are fences.
    - Frames whose interior is a non-bg 2x2 or 3x3 pattern are palettes.
    - For each fence, tile the matching palette pattern into the fence interior with
      top-left alignment, truncating on the right/bottom as needed.
    - Erase the palette blocks to bg.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    palettes: dict[int, np.ndarray] = {}
    palette_boxes: list[tuple[int, int, int, int]] = []
    fences: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components_mask(g == color):
            bbox = _rect_frame_bbox(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue

            if np.all(inner == bg):
                fences[color].append(bbox)
                continue

            inner_colors = {int(v) for v in np.unique(inner)}
            if bg in inner_colors:
                continue
            if inner.shape[0] == inner.shape[1] and inner.shape[0] in (2, 3) and len(inner_colors) == 2:
                palettes[color] = inner.copy()
                palette_boxes.append(bbox)

    out = g.copy()
    for color, boxes in fences.items():
        assert color in palettes
        pattern = palettes[color]
        for r0, c0, r1, c1 in boxes:
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            out[r0 + 1 : r1, c0 + 1 : c1] = _tile_to_shape(pattern, inner_h, inner_w)

    for r0, c0, r1, c1 in palette_boxes:
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    return out


# --------------- generator ---------------

SIDES = ("top", "bottom", "left", "right")
GRID_MIN = 20
GRID_MAX = 32
PAIR_COUNT_RANGE = (2, 4)
FENCE_INNER_MIN = 4
FENCE_INNER_MAX = 8
PALETTE_GAP = 1
PLACEMENT_TRIES = 300
GENERATION_RETRIES = 400


def _random_pattern(size: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    for _ in range(200):
        pattern = np.random.choice([a, b], size=(size, size))
        if len(set(map(int, pattern.ravel()))) != 2:
            continue
        # Prefer patterns that are not trivially constant by row and column.
        if all(np.all(pattern[r] == pattern[0]) for r in range(size)):
            continue
        if all(np.all(pattern[:, c] == pattern[:, 0]) for c in range(size)):
            continue
        return np.asarray(pattern, dtype=int)
    pattern = np.full((size, size), a, dtype=int)
    pattern[0, 0] = b
    pattern[-1, -1] = b
    return pattern


def _sample_fence_inner(tile_size: int) -> tuple[int, int]:
    force_partial = random.random() < 0.5
    for _ in range(100):
        inner_h = random.randint(max(FENCE_INNER_MIN, tile_size + 1), FENCE_INNER_MAX)
        inner_w = random.randint(max(FENCE_INNER_MIN, tile_size + 1), FENCE_INNER_MAX + 1)
        if force_partial and inner_h % tile_size == 0 and inner_w % tile_size == 0:
            continue
        return inner_h, inner_w
    return tile_size + 2, tile_size + 3


def _random_starts_1d(total: int, sizes: list[int], gap: int) -> list[int] | None:
    if not sizes:
        return []
    needed = sum(sizes) + gap * (len(sizes) - 1)
    if needed > total:
        return None
    slack = total - needed
    extras = np.random.multinomial(slack, np.ones(len(sizes) + 1) / (len(sizes) + 1))
    pos = int(extras[0])
    starts: list[int] = []
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if i < len(sizes) - 1:
            pos += gap + int(extras[i + 1])
    return starts


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_palette(grid: np.ndarray, bbox: tuple[int, int, int, int], border: int, pattern: np.ndarray) -> None:
    _draw_frame(grid, bbox, border)
    r0, c0, r1, c1 = bbox
    grid[r0 + 1 : r1, c0 + 1 : c1] = pattern


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _expected_output(grid: np.ndarray, bg: int, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["fence_bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_to_shape(spec["pattern"], r1 - r0 - 1, c1 - c0 - 1)
    for spec in specs:
        r0, c0, r1, c1 = spec["palette_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out


def _generate_case() -> np.ndarray | None:
    bg = random.randrange(10)
    side = random.choice(SIDES)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    border_colors = random.sample([c for c in range(10) if c != bg], pair_count)
    pattern_pool = [c for c in range(10) if c != bg and c not in border_colors]
    if len(pattern_pool) < 2:
        return None

    specs: list[dict] = []
    for border in border_colors:
        tile_size = random.choice((2, 3))
        inner_h, inner_w = _sample_fence_inner(tile_size)
        pattern_colors = tuple(random.sample(pattern_pool, 2))
        pattern = _random_pattern(tile_size, pattern_colors)
        specs.append(
            {
                "border": int(border),
                "pattern": pattern,
                "palette_h": tile_size + 2,
                "palette_w": tile_size + 2,
                "fence_h": inner_h + 2,
                "fence_w": inner_w + 2,
            }
        )

    total_fence_area = sum(int(spec["fence_h"] * spec["fence_w"]) for spec in specs)
    max_palette_h = max(int(spec["palette_h"]) for spec in specs)
    max_palette_w = max(int(spec["palette_w"]) for spec in specs)
    max_fence_h = max(int(spec["fence_h"]) for spec in specs)
    max_fence_w = max(int(spec["fence_w"]) for spec in specs)

    if side in ("top", "bottom"):
        palette_span = sum(int(spec["palette_w"]) for spec in specs) + PALETTE_GAP * (pair_count - 1)
        min_h = max(GRID_MIN, max_palette_h + max_fence_h + 5, int(np.ceil(np.sqrt(total_fence_area * 1.8))))
        min_w = max(GRID_MIN, palette_span + 2, int(np.ceil(np.sqrt(total_fence_area * 2.4))))
    else:
        palette_span = sum(int(spec["palette_h"]) for spec in specs) + PALETTE_GAP * (pair_count - 1)
        min_h = max(GRID_MIN, palette_span + 2, int(np.ceil(np.sqrt(total_fence_area * 2.4))))
        min_w = max(GRID_MIN, max_palette_w + max_fence_w + 5, int(np.ceil(np.sqrt(total_fence_area * 1.8))))

    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    if side in ("top", "bottom"):
        starts = _random_starts_1d(width - 2, [int(spec["palette_w"]) for spec in specs], PALETTE_GAP)
        if starts is None:
            return None
        for spec, local_c0 in zip(specs, starts, strict=False):
            c0 = 1 + int(local_c0)
            r0 = 0 if side == "top" else height - int(spec["palette_h"])
            bbox = (r0, c0, r0 + int(spec["palette_h"]) - 1, c0 + int(spec["palette_w"]) - 1)
            spec["palette_bbox"] = bbox
            _place_palette(grid, bbox, int(spec["border"]), spec["pattern"])
            _mark_bbox(occupied, bbox)
    else:
        starts = _random_starts_1d(height - 2, [int(spec["palette_h"]) for spec in specs], PALETTE_GAP)
        if starts is None:
            return None
        for spec, local_r0 in zip(specs, starts, strict=False):
            r0 = 1 + int(local_r0)
            c0 = 0 if side == "left" else width - int(spec["palette_w"])
            bbox = (r0, c0, r0 + int(spec["palette_h"]) - 1, c0 + int(spec["palette_w"]) - 1)
            spec["palette_bbox"] = bbox
            _place_palette(grid, bbox, int(spec["border"]), spec["pattern"])
            _mark_bbox(occupied, bbox)

    body_row_min = 1
    body_row_max = height - 2
    body_col_min = 1
    body_col_max = width - 2
    if side == "top":
        body_row_min = max_palette_h + 1
    elif side == "bottom":
        body_row_max = height - max_palette_h - 2
    elif side == "left":
        body_col_min = max_palette_w + 1
    else:
        body_col_max = width - max_palette_w - 2

    ordered_specs = sorted(specs, key=lambda spec: int(spec["fence_h"] * spec["fence_w"]), reverse=True)
    for spec in ordered_specs:
        fh = int(spec["fence_h"])
        fw = int(spec["fence_w"])
        max_r0 = body_row_max - fh + 1
        max_c0 = body_col_max - fw + 1
        if body_row_min > max_r0 or body_col_min > max_c0:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(body_row_min, max_r0)
            c0 = random.randint(body_col_min, max_c0)
            if not _can_place_bbox(occupied, r0, c0, fh, fw, pad=1):
                continue
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            spec["fence_bbox"] = bbox
            _draw_frame(grid, bbox, int(spec["border"]))
            _mark_bbox(occupied, bbox)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, bg, specs)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if _most_common_color(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
