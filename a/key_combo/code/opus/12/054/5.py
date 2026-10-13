import random
from collections import deque

import numpy as np


BG = 0
SPRITE_COLORS = (1, 2, 3)
INDICATOR_COLORS = (4, 6, 7)
FENCE_COLOR = 5
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

SHAPE_LIBRARY = [
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
]


def _normalize(coords):
    coords = list(coords)
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _shape_variants():
    variants = set()
    for base in SHAPE_LIBRARY:
        pts = np.array(base, dtype=int)
        for k in range(4):
            rot = pts.copy()
            for _ in range(k):
                rot = np.column_stack((rot[:, 1], -rot[:, 0]))
            variants.add(_normalize(map(tuple, rot)))
            flipped = rot.copy()
            flipped[:, 1] *= -1
            variants.add(_normalize(map(tuple, flipped)))
    return sorted(variants)


ALL_SHAPES = _shape_variants()


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r, c):
    for dr, dc in FOUR_DIRS:
        yield r + dr, c + dc


def _find_color_components(grid, colors):
    H, W = grid.shape
    components = []
    for color in colors:
        seen = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if seen[r, c] or int(grid[r, c]) != int(color):
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                            seen[nr, nc] = True
                            q.append((nr, nc))
                components.append(
                    {
                        "color": int(color),
                        "coords": tuple(sorted(coords)),
                        "bbox": _bbox(coords),
                        "area": len(coords),
                    }
                )
    components.sort(key=lambda comp: comp["bbox"])
    return components


def _detect_fences(grid):
    fence_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == FENCE_COLOR)]
    assert len(fence_cols) >= 2, "expected at least two full-height fence columns"
    diffs = [b - a for a, b in zip(fence_cols, fence_cols[1:])]
    assert len(set(diffs)) == 1, "fence spacing must be periodic"
    spacing = diffs[0]
    assert 4 <= spacing <= 6, "fence spacing must be 4..6"
    return fence_cols, spacing


def _intervals_from_fences(fence_cols):
    return [(fence_cols[i] + 1, fence_cols[i + 1] - 1) for i in range(len(fence_cols) - 1)]


def _interval_index(intervals, bbox):
    _, c0, _, c1 = bbox
    for idx, (left, right) in enumerate(intervals):
        if left <= c0 and c1 <= right:
            return idx
    raise AssertionError("sprite must lie entirely inside one fenced interval")


def _indicator_side(indicator, bbox):
    r, c = indicator
    r0, c0, r1, c1 = bbox
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    if r == r0 - 1 and c0 <= c <= c1:
        return "up"
    if r == r1 + 1 and c0 <= c <= c1:
        return "down"
    raise AssertionError("indicator must sit just outside one sprite side")


def _mirrored_targets(sprite_coords, indicator, side):
    ir, ic = indicator
    if side in ("left", "right"):
        return [(r, 2 * ic - c) for r, c in sprite_coords]
    return [(2 * ir - r, c) for r, c in sprite_coords]


def _parse_scene(grid):
    H, W = grid.shape
    fence_cols, spacing = _detect_fences(grid)
    intervals = _intervals_from_fences(fence_cols)
    sprites = _find_color_components(grid, SPRITE_COLORS)
    indicators = {(r, c): int(grid[r, c]) for r, c in np.argwhere(np.isin(grid, INDICATOR_COLORS))}

    parsed = []
    for sprite in sprites:
        bbox = sprite["bbox"]
        interval_idx = _interval_index(intervals, bbox)
        sprite_set = set(sprite["coords"])
        adjacent = []
        for r, c in sprite["coords"]:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in indicators:
                    adjacent.append((nr, nc))
        adjacent = sorted(set(adjacent))
        assert len(adjacent) <= 1, "each sprite can have at most one indicator"
        indicator = adjacent[0] if adjacent else None
        side = _indicator_side(indicator, bbox) if indicator is not None else None
        parsed.append(
            {
                "color": sprite["color"],
                "coords": sprite["coords"],
                "coord_set": sprite_set,
                "bbox": bbox,
                "area": sprite["area"],
                "interval_idx": interval_idx,
                "interval": intervals[interval_idx],
                "indicator": indicator,
                "indicator_color": indicators[indicator] if indicator is not None else None,
                "side": side,
            }
        )

    for left, right in intervals:
        count = sum(1 for obj in parsed if obj["interval"] == (left, right))
        assert count <= 1, "each interval may contain at most one sprite"
        for c in range(left, right + 1):
            assert 0 <= c < W
    return {
        "height": H,
        "width": W,
        "spacing": spacing,
        "fence_cols": fence_cols,
        "intervals": intervals,
        "objects": parsed,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each indicated sprite across its attached side marker, but keep only the part of the reflection that remains inside the same fenced interval.

    Input structure:
    - The grid has black background plus periodic full-height gray vertical fence lines.
    - Each fenced interval contains at most one 4-connected monochrome sprite in blue, red, or green.
    - Some sprites have exactly one adjacent single-pixel indicator in yellow, magenta, or orange; other sprites have no indicator.

    Rule:
    - For every sprite with an indicator, determine whether the indicator is above, below, left, or right of the sprite bbox.
    - Left/right indicators define a vertical mirror axis at the indicator column; up/down indicators define a horizontal mirror axis at the indicator row.
    - Reflect the whole sprite across that axis, recolor the reflected copy with the indicator color, and discard reflected pixels that leave the sprite's own fenced interval or the grid.
    - Keep all original sprites, indicators, and fences unchanged; reflected pixels are drawn only onto background cells.

    Key insight:
    - The indicator supplies both the mirror axis and the new color, while the periodic fence lines define a hard local visibility window that can truncate horizontal reflections.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    out = g.copy()

    for obj in scene["objects"]:
        if obj["indicator"] is None:
            continue
        left, right = obj["interval"]
        protected = obj["coord_set"]
        for (src_r, src_c), (dst_r, dst_c) in zip(
            obj["coords"],
            _mirrored_targets(obj["coords"], obj["indicator"], obj["side"]),
        ):
            del src_r, src_c
            if not (0 <= dst_r < scene["height"] and left <= dst_c <= right):
                continue
            if int(out[dst_r, dst_c]) != BG:
                continue
            out[dst_r, dst_c] = int(obj["indicator_color"])
        assert protected  # keeps object parsing non-empty and intentional

    return out


def _attachment_choices(shape, side):
    if side == "left":
        edge = min(c for _, c in shape)
        return [r for r, c in shape if c == edge]
    if side == "right":
        edge = max(c for _, c in shape)
        return [r for r, c in shape if c == edge]
    if side == "up":
        edge = min(r for r, _ in shape)
        return [c for r, c in shape if r == edge]
    edge = max(r for r, _ in shape)
    return [c for r, c in shape if r == edge]


def _candidate_shape_pool(role, interval_width):
    pool = []
    for shape in ALL_SHAPES:
        r0, c0, r1, c1 = _bbox(shape)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = len(shape)
        if not (3 <= area <= 7):
            continue
        if w > interval_width:
            continue
        if role == "plain":
            pool.append(shape)
        elif role in ("left_trunc", "right_trunc"):
            if interval_width >= 4 and 2 <= w <= min(3, interval_width - 1):
                pool.append(shape)
        elif role in ("left_full", "right_full"):
            if w <= max(1, (interval_width - 1) // 2):
                pool.append(shape)
        else:
            if h <= 4:
                pool.append(shape)
    return pool


def _evaluate_reflection(shape_cells, indicator, side, interval, H, W):
    visible = []
    left, right = interval
    for rr, cc in _mirrored_targets(shape_cells, indicator, side):
        if 0 <= rr < H and left <= cc <= right:
            visible.append((rr, cc))
    return visible


def _enumerate_role_candidates(shape, role, interval, H, W):
    left, right = interval
    interval_width = right - left + 1
    r0, c0, r1, c1 = _bbox(shape)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    row_positions = range(0, H - h + 1)
    col_positions = range(left, right - w + 1 + 1)
    candidates = []

    if role == "plain":
        for base_r in row_positions:
            for base_c in col_positions:
                cells = [(base_r + r, base_c + c) for r, c in shape]
                candidates.append({"cells": cells, "indicator": None, "side": None})
        return candidates

    side = role.split("_")[0]
    for base_r in row_positions:
        for base_c in col_positions:
            cells = [(base_r + r, base_c + c) for r, c in shape]
            for attach in _attachment_choices(shape, side):
                if side == "left":
                    indicator = (base_r + attach, base_c - 1)
                elif side == "right":
                    indicator = (base_r + attach, base_c + w)
                elif side == "up":
                    indicator = (base_r - 1, base_c + attach)
                else:
                    indicator = (base_r + h, base_c + attach)

                ir, ic = indicator
                if not (0 <= ir < H and left <= ic <= right):
                    continue
                reflected = _evaluate_reflection(cells, indicator, side, interval, H, W)
                if len(reflected) == 0:
                    continue
                if role.endswith("trunc") and len(reflected) >= len(cells):
                    continue
                if role.endswith("full") and len(reflected) != len(cells):
                    continue
                if side in ("up", "down") and len(reflected) != len(cells):
                    continue
                candidates.append(
                    {
                        "cells": cells,
                        "indicator": indicator,
                        "side": side,
                        "visible_reflection": reflected,
                    }
                )
    return candidates


def _place_interval_content(grid, interval, role):
    H, W = grid.shape
    left, right = interval
    pool = _candidate_shape_pool(role, right - left + 1)
    random.shuffle(pool)
    for shape in pool[:]:
        candidates = _enumerate_role_candidates(shape, role, interval, H, W)
        random.shuffle(candidates)
        for cand in candidates:
            cells = cand["cells"]
            indicator = cand["indicator"]
            if any(int(grid[r, c]) != BG for r, c in cells):
                continue
            if indicator is not None and int(grid[indicator]) != BG:
                continue
            sprite_color = random.choice(SPRITE_COLORS)
            for r, c in cells:
                grid[r, c] = sprite_color
            indicator_color = None
            if indicator is not None:
                indicator_color = random.choice(INDICATOR_COLORS)
                grid[indicator] = indicator_color
            return {
                "role": role,
                "cells": tuple(cells),
                "indicator": indicator,
                "indicator_color": indicator_color,
                "sprite_color": sprite_color,
            }
    return None


def _build_role_list(interval_count, spacing):
    num_objects = random.randint(2, interval_count)
    roles = []
    if spacing >= 5:
        roles.append(random.choice(("left_trunc", "right_trunc")))
    else:
        roles.append(random.choice(("left_full", "right_full")))
    if num_objects >= 2:
        roles.append(random.choice(("up", "down", "plain")))
    while len(roles) < num_objects:
        pool = ["plain", "up", "down", "left_full", "right_full"]
        if spacing >= 5:
            pool += ["left_trunc", "right_trunc"]
        roles.append(random.choice(pool))
    random.shuffle(roles)
    return roles


def _validate_generated(grid, planned_roles):
    scene = _parse_scene(grid)
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "at least one reflection must be visible"

    indicated = 0
    truncated = 0
    for obj in scene["objects"]:
        if obj["indicator"] is None:
            continue
        indicated += 1
        reflected = _evaluate_reflection(
            obj["coords"], obj["indicator"], obj["side"], obj["interval"], scene["height"], scene["width"]
        )
        assert 0 < len(reflected) <= obj["area"]
        if len(reflected) < obj["area"]:
            truncated += 1
        for r, c in reflected:
            assert int(out[r, c]) in (BG, obj["indicator_color"])
    assert indicated >= 1
    if any(role.endswith("trunc") for role in planned_roles):
        assert truncated >= 1


def _attempt_generate():
    H = random.randint(13, 17)
    spacing = random.choice((4, 5, 6))
    max_intervals = min(5, (30 - 1) // spacing)
    interval_count = random.randint(3, max(3, max_intervals))
    W = interval_count * spacing + 1

    grid = np.zeros((H, W), dtype=int)
    fence_cols = list(range(0, W, spacing))
    for c in fence_cols:
        grid[:, c] = FENCE_COLOR
    intervals = _intervals_from_fences(fence_cols)

    roles = _build_role_list(len(intervals), spacing)
    chosen_intervals = random.sample(range(len(intervals)), len(roles))
    placed = []
    for interval_idx, role in zip(chosen_intervals, roles):
        info = _place_interval_content(grid, intervals[interval_idx], role)
        if info is None:
            return None
        placed.append(info)

    _validate_generated(grid, roles)
    assert len(placed) == len(roles)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-mirror sample")
