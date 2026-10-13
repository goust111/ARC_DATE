from __future__ import annotations

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _draw_ring(canvas: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    canvas[r0, c0 : c0 + width] = color
    canvas[r0 + height - 1, c0 : c0 + width] = color
    canvas[r0 : r0 + height, c0] = color
    canvas[r0 : r0 + height, c0 + width - 1] = color


def _ring_info(grid: np.ndarray, coords: np.ndarray) -> dict[str, int]:
    colors = np.unique(grid[coords[:, 0], coords[:, 1]])
    assert len(colors) == 1, "each fragment must be monochromatic"
    color = int(colors[0])

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height >= 3 and width >= 3, "each fragment must be a hollow rectangle"

    rel = coords - np.array([r0, c0])
    mask = np.zeros((height, width), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.array_equal(mask, border), "each fragment must be a 1-pixel rectangular ring"

    return {
        "color": color,
        "height": height,
        "width": width,
        "area": int(height * width),
    }


def _find_corner_marker(grid: np.ndarray, bg: int) -> tuple[str, int, set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[str, int, set[tuple[int, int]]]] = []
    for comp in _components(grid != bg):
        if len(comp) != 2:
            continue
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 1):
            continue

        top = (r0, c0)
        bottom = (r1, c1)
        if int(grid[top]) == int(grid[bottom]):
            continue

        if r0 == 0 and c0 == 0:
            anchor = "tl"
        elif r0 == 0 and c0 == w - 1:
            anchor = "tr"
        elif r1 == h - 1 and c0 == 0:
            anchor = "bl"
        elif r1 == h - 1 and c1 == w - 1:
            anchor = "br"
        else:
            continue

        candidates.append((anchor, int(grid[top]), {top, bottom}))

    assert len(candidates) == 1, "expected exactly one 2x1 corner marker"
    return candidates[0]


def _split_gap(total: int, favor_near_side: bool) -> tuple[int, int]:
    assert total >= 2, "nested rings need at least one black cell on each side"
    small = total // 2
    large = total - small
    return (small, large) if favor_near_side else (large, small)


def _render_nested(rings: list[dict[str, int]], anchor: str, bg: int) -> np.ndarray:
    assert rings, "expected at least one ring"
    outer = rings[0]
    out = np.full((int(outer["height"]), int(outer["width"])), bg, dtype=int)

    positions: list[tuple[int, int]] = [(0, 0)]
    _draw_ring(out, 0, 0, int(outer["height"]), int(outer["width"]), int(outer["color"]))

    top_favored = anchor in {"tl", "tr"}
    left_favored = anchor in {"tl", "bl"}

    for idx in range(1, len(rings)):
        parent = rings[idx - 1]
        child = rings[idx]
        parent_r, parent_c = positions[idx - 1]

        total_v = int(parent["height"] - child["height"] - 2)
        total_h = int(parent["width"] - child["width"] - 2)
        top_gap, _bottom_gap = _split_gap(total_v, top_favored)
        left_gap, _right_gap = _split_gap(total_h, left_favored)

        child_r = int(parent_r + 1 + top_gap)
        child_c = int(parent_c + 1 + left_gap)
        positions.append((child_r, child_c))
        _draw_ring(
            out,
            child_r,
            child_c,
            int(child["height"]),
            int(child["width"]),
            int(child["color"]),
        )

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular rings into a nested multi-layer frame.

    Input structure:
    - The background is black.
    - Several disconnected monochromatic objects are hollow rectangular rings with 1-pixel borders.
    - One corner contains a unique 2x1 vertical marker whose upper pixel names the outermost ring color.
    - The marker's corner fixes how center-rounding ambiguities are resolved when the rings are nested.

    Rule:
    - Remove the corner marker from consideration and identify every ring fragment by its rectangular bbox.
    - Sort the rings by enclosed rectangle area from largest to smallest; the largest ring must match the marker's upper color.
    - Rebuild the rings as a nested structure from outer to inner, keeping a black band between consecutive rings and centering each child as closely as possible inside its parent.
    - If centering leaves a one-cell parity ambiguity, bias the placement toward the marker's corner, then crop tightly to the reconstructed outer ring.

    Key insight:
    - The nesting order is encoded by geometric size, while the corner marker simultaneously verifies the outermost ring and breaks the otherwise ambiguous center alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    anchor, outer_color, marker_cells = _find_corner_marker(g, bg)

    ring_mask = g != bg
    for r, c in marker_cells:
        ring_mask[r, c] = False

    rings = [_ring_info(g, comp) for comp in _components(ring_mask)]
    assert rings, "expected at least one ring fragment"
    rings.sort(key=lambda item: int(item["area"]), reverse=True)

    areas = [int(item["area"]) for item in rings]
    assert len(set(areas)) == len(areas), "ring areas must define a unique nesting order"
    assert int(rings[0]["color"]) == int(outer_color), "marker must identify the outermost ring"

    return _render_nested(rings, anchor, bg)


GEN_ATTEMPTS = 400
GEN_COUNT_CHOICES = [2, 3]
GEN_INNER_RANGE_BY_COUNT = {2: (3, 8), 3: (3, 5)}
GEN_STEP_CHOICES = [4, 5]
GEN_ANCHORS = ["tl", "tr", "bl", "br"]
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_PLACE_TRIES = 300


def _build_rings(rng: np.random.Generator) -> list[dict[str, int]] | None:
    count = int(rng.choice(np.array(GEN_COUNT_CHOICES, dtype=int)))
    inner_lo, inner_hi = GEN_INNER_RANGE_BY_COUNT[count]
    inner_h = int(rng.integers(inner_lo, inner_hi + 1))
    inner_w = int(rng.integers(inner_lo, inner_hi + 1))

    heights = [inner_h]
    widths = [inner_w]
    steps: list[tuple[int, int]] = []
    for _ in range(count - 1):
        dh = int(rng.choice(np.array(GEN_STEP_CHOICES, dtype=int)))
        dw = int(rng.choice(np.array(GEN_STEP_CHOICES, dtype=int)))
        steps.append((dh, dw))
        heights.append(heights[-1] + dh)
        widths.append(widths[-1] + dw)

    outer_h = heights[-1]
    outer_w = widths[-1]
    if outer_h > 15 or outer_w > 15:
        return None

    if count == 3 and not any(dh == 5 or dw == 5 for dh, dw in steps) and float(rng.random()) < 0.75:
        idx = int(rng.integers(0, len(steps)))
        if float(rng.random()) < 0.5:
            for pos in range(idx + 1, len(widths)):
                widths[pos] += 1
        else:
            for pos in range(idx + 1, len(heights)):
                heights[pos] += 1
        if heights[-1] > 15 or widths[-1] > 15:
            return None

    ring_colors = list(map(int, rng.choice(np.array(GEN_COLORS, dtype=int), size=count, replace=False)))
    rings: list[dict[str, int]] = []
    for color, height, width in zip(ring_colors, reversed(heights), reversed(widths)):
        rings.append(
            {
                "color": int(color),
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )
    return rings


def _ring_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    _draw_ring(patch, 0, 0, height, width, color)
    return patch


def _marker_cells_for(anchor: str, height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if anchor == "tl":
        return (0, 0), (1, 0)
    if anchor == "tr":
        return (0, width - 1), (1, width - 1)
    if anchor == "bl":
        return (height - 2, 0), (height - 1, 0)
    assert anchor == "br"
    return (height - 2, width - 1), (height - 1, width - 1)


def _reserve_marker(blocked: np.ndarray, anchor: str) -> None:
    h, w = blocked.shape
    if anchor == "tl":
        blocked[:3, :2] = True
    elif anchor == "tr":
        blocked[:3, max(0, w - 2) :] = True
    elif anchor == "bl":
        blocked[max(0, h - 3) :, :2] = True
    else:
        blocked[max(0, h - 3) :, max(0, w - 2) :] = True


def _place_patch_bbox(
    blocked: np.ndarray,
    patch_h: int,
    patch_w: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    if patch_h > h or patch_w > w:
        return None
    for _ in range(GEN_PLACE_TRIES):
        r0 = int(rng.integers(0, h - patch_h + 1))
        c0 = int(rng.integers(0, w - patch_w + 1))
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + patch_h + 1)
        cc1 = min(w, c0 + patch_w + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        blocked[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _scatter_input(
    rng: np.random.Generator,
    rings: list[dict[str, int]],
    anchor: str,
    marker_color: int,
) -> np.ndarray | None:
    largest_dim = max(max(int(r["height"]), int(r["width"])) for r in rings)
    total_bbox_area = sum(int(r["height"] * r["width"]) for r in rings)
    min_side = max(GEN_GRID_MIN, largest_dim + 4, int(np.ceil(np.sqrt(total_bbox_area))) + 6)
    if min_side > GEN_GRID_MAX:
        return None

    grid_h = int(rng.integers(min_side, GEN_GRID_MAX + 1))
    grid_w = int(rng.integers(min_side, GEN_GRID_MAX + 1))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    _reserve_marker(blocked, anchor)

    placements: list[tuple[dict[str, int], int, int]] = []
    for ring in sorted(rings, key=lambda item: int(item["area"]), reverse=True):
        pos = _place_patch_bbox(blocked, int(ring["height"]), int(ring["width"]), rng)
        if pos is None:
            return None
        placements.append((ring, pos[0], pos[1]))

    top_cell, bottom_cell = _marker_cells_for(anchor, grid_h, grid_w)
    grid[top_cell] = int(rings[0]["color"])
    grid[bottom_cell] = marker_color

    for ring, r0, c0 in placements:
        patch = _ring_patch(int(ring["height"]), int(ring["width"]), int(ring["color"]))
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        rings = _build_rings(rng)
        if rings is None:
            continue

        anchor = str(rng.choice(np.array(GEN_ANCHORS, dtype=object)))
        used_colors = {int(ring["color"]) for ring in rings}
        marker_options = [color for color in GEN_COLORS if color not in used_colors]
        marker_color = int(rng.choice(np.array(marker_options, dtype=int)))

        grid = _scatter_input(rng, rings, anchor, marker_color)
        if grid is None:
            continue

        expected = _render_nested(rings, anchor, 0)
        try:
            produced = main(grid)
        except Exception:
            continue
        if np.array_equal(produced, expected):
            return grid

    raise RuntimeError("failed to generate a valid nested-ring input")
