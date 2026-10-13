from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Generator configuration.
GEN_RETRIES = 400
RING_COUNT_RANGE = (2, 4)
THICKNESS_RANGE = (1, 2)
HOLE_SIZE_BY_CONTENT = {
    0: (1, 3),
    1: (3, 5),
    2: (5, 7),
    3: (6, 8),
}
MAX_OUTER_SIDE = 26
GRID_SIDE_MAX = 32
SLOT_GAP_RANGE = (2, 5)
OUTER_MARGIN_RANGE = (1, 4)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _count_non_bg_components(window: np.ndarray, bg: int) -> int:
    h, w = window.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(window[r, c]) == bg:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(window[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _leading_full_rows(mask: np.ndarray) -> int:
    rows = 0
    while rows < mask.shape[0] and bool(mask[rows].all()):
        rows += 1
    return rows


def _leading_full_cols(mask: np.ndarray) -> int:
    cols = 0
    while cols < mask.shape[1] and bool(mask[:, cols].all()):
        cols += 1
    return cols


def _ring_mask(outer_h: int, outer_w: int, top: int, left: int, hole_h: int, hole_w: int) -> np.ndarray:
    mask = np.ones((outer_h, outer_w), dtype=bool)
    mask[top : top + hole_h, left : left + hole_w] = False
    return mask


def _parse_ring_component(component: dict) -> dict | None:
    coords = component["coords"]
    r0, c0, r1, c1 = _bbox(coords)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 3 or outer_w < 3:
        return None

    mask = np.zeros((outer_h, outer_w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    top = _leading_full_rows(mask)
    bottom = _leading_full_rows(mask[::-1])
    left = _leading_full_cols(mask)
    right = _leading_full_cols(mask[:, ::-1])
    if min(top, bottom, left, right) <= 0:
        return None
    if top != bottom or left != right:
        return None

    hole_h = outer_h - top - bottom
    hole_w = outer_w - left - right
    if hole_h <= 0 or hole_w <= 0:
        return None

    expected = _ring_mask(outer_h, outer_w, top, left, hole_h, hole_w)
    if not np.array_equal(mask, expected):
        return None

    return {
        "color": int(component["color"]),
        "bbox": (r0, c0, r1, c1),
        "outer_h": outer_h,
        "outer_w": outer_w,
        "top": top,
        "left": left,
        "hole_h": hole_h,
        "hole_w": hole_w,
        "mask": expected,
    }


def _extract_rings(grid: np.ndarray, bg: int) -> list[dict]:
    rings: list[dict] = []
    for component in _monochromatic_components(grid, bg):
        ring = _parse_ring_component(component)
        if ring is None:
            continue
        r0, c0, _r1, _c1 = ring["bbox"]
        hole = grid[
            r0 + ring["top"] : r0 + ring["top"] + ring["hole_h"],
            c0 + ring["left"] : c0 + ring["left"] + ring["hole_w"],
        ]
        ring["content_count"] = _count_non_bg_components(hole, bg)
        rings.append(ring)
    return rings


def _render_nested_output(rings: list[dict], bg: int) -> np.ndarray:
    assert rings
    outer_h = int(rings[0]["outer_h"])
    outer_w = int(rings[0]["outer_w"])
    out = np.full((outer_h, outer_w), bg, dtype=int)

    for ring in rings:
        delta_h = outer_h - int(ring["outer_h"])
        delta_w = outer_w - int(ring["outer_w"])
        assert delta_h >= 0 and delta_w >= 0
        assert delta_h % 2 == 0 and delta_w % 2 == 0
        off_r = delta_h // 2
        off_c = delta_w // 2
        sub = out[off_r : off_r + ring["outer_h"], off_c : off_c + ring["outer_w"]]
        overlap = ring["mask"] & (sub != bg)
        assert not bool(overlap.any())
        sub[ring["mask"]] = int(ring["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder colored rectangular rings by the number of objects inside their holes
    and place them as concentric nested layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    rings = _extract_rings(g, bg)
    assert rings, "expected at least one rectangular ring"

    rings.sort(
        key=lambda ring: (
            -int(ring["content_count"]),
            -(int(ring["outer_h"]) * int(ring["outer_w"])),
            -(int(ring["hole_h"]) * int(ring["hole_w"])),
            int(ring["bbox"][0]),
            int(ring["bbox"][1]),
        )
    )
    return _render_nested_output(rings, bg)


def _make_ring_specs(rng: random.Random) -> list[dict] | None:
    ring_count = rng.randint(*RING_COUNT_RANGE)
    content_counts = sorted(rng.sample([0, 1, 2, 3], ring_count), reverse=True)

    inner_count = content_counts[-1]
    hole_lo, hole_hi = HOLE_SIZE_BY_CONTENT[inner_count]
    inner_hole_caps = {2: 6, 3: 4, 4: 2}
    hole_hi = min(hole_hi, inner_hole_caps[ring_count])
    if hole_lo > hole_hi:
        return None
    next_hole_h = rng.randint(hole_lo, hole_hi)
    next_hole_w = rng.randint(hole_lo, hole_hi)
    outer_caps = {2: 26, 3: 18, 4: 16}

    specs_inner_to_outer: list[dict] = []
    child_outer_h = None
    child_outer_w = None

    for content_count in reversed(content_counts):
        if child_outer_h is None:
            hole_h = next_hole_h
            hole_w = next_hole_w
        else:
            hole_h = child_outer_h + 2
            hole_w = child_outer_w + 2

        if ring_count >= 4:
            top = 1
            left = 1
        elif ring_count == 3:
            top = 1 if rng.random() < 0.75 else 2
            left = 1 if rng.random() < 0.75 else 2
        else:
            top = rng.randint(*THICKNESS_RANGE)
            left = rng.randint(*THICKNESS_RANGE)
        outer_h = hole_h + 2 * top
        outer_w = hole_w + 2 * left
        if max(outer_h, outer_w) > min(MAX_OUTER_SIDE, outer_caps[ring_count]):
            return None

        min_hole_side = HOLE_SIZE_BY_CONTENT[content_count][0]
        if hole_h < min_hole_side or hole_w < min_hole_side:
            return None

        specs_inner_to_outer.append(
            {
                "content_count": content_count,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "top": top,
                "left": left,
                "hole_h": hole_h,
                "hole_w": hole_w,
            }
        )
        child_outer_h = outer_h
        child_outer_w = outer_w

    specs = list(reversed(specs_inner_to_outer))
    colors = rng.sample(PALETTE, len(specs))
    for spec, color in zip(specs, colors):
        spec["color"] = color
        spec["mask"] = _ring_mask(
            int(spec["outer_h"]),
            int(spec["outer_w"]),
            int(spec["top"]),
            int(spec["left"]),
            int(spec["hole_h"]),
            int(spec["hole_w"]),
        )
    return specs


def _place_content_rectangles(hole_h: int, hole_w: int, count: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    if count == 0:
        return []
    blocked = np.zeros((hole_h, hole_w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    rects: list[tuple[int, int, int, int]] = []

    for _ in range(count):
        candidates: list[tuple[int, int, int, int]] = []
        max_rect_h = min(2, hole_h - 2)
        max_rect_w = min(3, hole_w - 2)
        for rect_h in range(1, max_rect_h + 1):
            for rect_w in range(1, max_rect_w + 1):
                for r0 in range(1, hole_h - rect_h):
                    for c0 in range(1, hole_w - rect_w):
                        if blocked[r0 : r0 + rect_h, c0 : c0 + rect_w].any():
                            continue
                        candidates.append((r0, c0, rect_h, rect_w))
        if not candidates:
            return None
        rect = rng.choice(candidates)
        r0, c0, rect_h, rect_w = rect
        rects.append(rect)
        blocked[max(0, r0 - 1) : min(hole_h, r0 + rect_h + 1), max(0, c0 - 1) : min(hole_w, c0 + rect_w + 1)] = True

    return rects


def _render_input_ring(spec: dict, rng: random.Random) -> np.ndarray | None:
    sprite = np.full((int(spec["outer_h"]), int(spec["outer_w"])), BG, dtype=int)
    sprite[spec["mask"]] = int(spec["color"])

    rects = _place_content_rectangles(int(spec["hole_h"]), int(spec["hole_w"]), int(spec["content_count"]), rng)
    if rects is None:
        return None

    choices = [color for color in PALETTE if color != int(spec["color"])]
    rng.shuffle(choices)
    for idx, (r0, c0, rect_h, rect_w) in enumerate(rects):
        color = choices[idx % len(choices)]
        rr0 = int(spec["top"]) + r0
        cc0 = int(spec["left"]) + c0
        sprite[rr0 : rr0 + rect_h, cc0 : cc0 + rect_w] = color
    return sprite


def _layout_sprites(sprites: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    n = len(sprites)
    indices = list(range(n))
    rng.shuffle(indices)

    layouts = []
    for rows in range(1, n + 1):
        cols = (n + rows - 1) // rows
        layouts.append((rows, cols))
    layouts = list(dict.fromkeys(layouts))
    layouts.sort(key=lambda layout: (abs(layout[0] - layout[1]), layout[0] * layout[1], layout[0]))

    for rows, cols in layouts:
        row_heights = [0] * rows
        col_widths = [0] * cols
        for slot, idx in enumerate(indices):
            sr = slot // cols
            sc = slot % cols
            row_heights[sr] = max(row_heights[sr], int(sprites[idx].shape[0]))
            col_widths[sc] = max(col_widths[sc], int(sprites[idx].shape[1]))

        if n == 4:
            margin_range = (0, 2)
            gap_range = (1, 2)
        elif n == 3:
            margin_range = (1, 3)
            gap_range = (1, 3)
        else:
            margin_range = OUTER_MARGIN_RANGE
            gap_range = SLOT_GAP_RANGE

        top_margin = rng.randint(*margin_range)
        bottom_margin = rng.randint(*margin_range)
        left_margin = rng.randint(*margin_range)
        right_margin = rng.randint(*margin_range)
        row_gaps = [rng.randint(*gap_range) for _ in range(max(0, rows - 1))]
        col_gaps = [rng.randint(*gap_range) for _ in range(max(0, cols - 1))]

        height = top_margin + bottom_margin + sum(row_heights) + sum(row_gaps)
        width = left_margin + right_margin + sum(col_widths) + sum(col_gaps)
        if height > GRID_SIDE_MAX or width > GRID_SIDE_MAX:
            continue

        extra_h = GRID_SIDE_MAX - height
        extra_w = GRID_SIDE_MAX - width
        add_top = rng.randint(0, extra_h)
        extra_h -= add_top
        add_bottom = rng.randint(0, extra_h)
        add_left = rng.randint(0, extra_w)
        extra_w -= add_left
        add_right = rng.randint(0, extra_w)
        top_margin += add_top
        bottom_margin += add_bottom
        left_margin += add_left
        right_margin += add_right

        height = top_margin + bottom_margin + sum(row_heights) + sum(row_gaps)
        width = left_margin + right_margin + sum(col_widths) + sum(col_gaps)
        if height > GRID_SIDE_MAX or width > GRID_SIDE_MAX:
            continue

        grid = np.full((height, width), BG, dtype=int)
        row_starts = [top_margin]
        for gap, row_h in zip(row_gaps, row_heights[:-1]):
            row_starts.append(row_starts[-1] + row_h + gap)
        col_starts = [left_margin]
        for gap, col_w in zip(col_gaps, col_widths[:-1]):
            col_starts.append(col_starts[-1] + col_w + gap)

        for slot, idx in enumerate(indices):
            sr = slot // cols
            sc = slot % cols
            sprite = sprites[idx]
            cell_h = row_heights[sr]
            cell_w = col_widths[sc]
            off_r = rng.randint(0, cell_h - sprite.shape[0])
            off_c = rng.randint(0, cell_w - sprite.shape[1])
            r0 = row_starts[sr] + off_r
            c0 = col_starts[sc] + off_c
            sub = grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]]
            assert not bool((sub != BG).any())
            sub[sprite != BG] = sprite[sprite != BG]

        return grid

    return None


def _counts_are_valid(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = 0
    best_non_bg = 0
    for value, count in zip(values, counts):
        if int(value) == BG:
            bg_count = int(count)
        else:
            best_non_bg = max(best_non_bg, int(count))
    return bg_count > best_non_bg


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _make_ring_specs(rng)
    if specs is None:
        return None

    sprites: list[np.ndarray] = []
    for spec in specs:
        sprite = _render_input_ring(spec, rng)
        if sprite is None:
            return None
        sprites.append(sprite)

    inp = _layout_sprites(sprites, rng)
    if inp is None or not _counts_are_valid(inp):
        return None

    expected = _render_nested_output(specs, BG)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
