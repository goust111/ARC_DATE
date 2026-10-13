from __future__ import annotations

import random

import numpy as np


NEIGHBORS4: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
CARDINALS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
PALETTE: tuple[int, ...] = tuple(range(10))

# A minimal 4-cell arrow glyph represented as a path.
# Removing the tip leaves a straight 3-cell shaft, which keeps the direction visible.
ARROW_PATHS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((1, 0), (1, 1), (1, 2), (0, 2)),   # right
    (0, -1): ((1, 2), (1, 1), (1, 0), (2, 0)),  # left
    (-1, 0): ((2, 1), (1, 1), (0, 1), (0, 2)),  # up
    (1, 0): ((0, 1), (1, 1), (2, 1), (2, 0)),   # down
}

LAYER2_SEEDS: tuple[np.ndarray, ...] = (
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
)

LAYER3_SEED = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=bool,
)

GEN_RETRIES = 400
TEMPLATE_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                out[rr, cc] = True
                break
    return out


def _peel_layers(mask: np.ndarray) -> list[np.ndarray]:
    work = mask.copy()
    layers: list[np.ndarray] = []
    while work.any():
        ring = _boundary(work)
        layers.append(np.argwhere(ring))
        work[ring] = False
    return layers


def _is_hole_free(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return not np.any(bg & ~seen)


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    del h
    full_cols = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    for i, c1 in enumerate(full_cols):
        for c2 in full_cols[i + 1 :]:
            left = int(c1)
            middle = int(c2 - c1 - 1)
            right = int(w - c2 - 1)
            if left > 0 and left == middle == right:
                return int(c1), int(c2), int(left)
    raise ValueError("failed to locate the two equal-width divider columns")


def _decode_arrow(signal_zone: np.ndarray, bg: int) -> tuple[tuple[int, int], list[int]]:
    coords = [tuple(map(int, cell)) for cell in np.argwhere(signal_zone != bg)]
    if len(coords) != 4:
        raise ValueError("signal zone must contain exactly four arrow pixels")

    coord_set = set(coords)
    neighbors = {
        cell: [other for other in coords if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1]
        for cell in coords
    }
    endpoints = [cell for cell, adj in neighbors.items() if len(adj) == 1]
    if len(endpoints) != 2:
        raise ValueError("arrow must be a 4-connected path")

    tip: tuple[int, int] | None = None
    for cell in endpoints:
        shaft = [other for other in coords if other != cell]
        rows = {r for r, _c in shaft}
        cols = {c for _r, c in shaft}
        if len(rows) == 1 or len(cols) == 1:
            tip = cell
            break
    if tip is None:
        raise ValueError("failed to identify arrow tip")

    order = [tip]
    prev: tuple[int, int] | None = None
    cur = tip
    while True:
        nxts = [cell for cell in neighbors[cur] if cell != prev]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt

    if len(order) != 4:
        raise ValueError("arrow path length mismatch")

    front = order[1]
    back = order[-1]
    dr = int(np.sign(front[0] - back[0]))
    dc = int(np.sign(front[1] - back[1]))
    if (dr, dc) not in CARDINALS:
        raise ValueError("failed to recover a cardinal arrow direction")

    colors_tip_to_tail = [int(signal_zone[r, c]) for r, c in order]

    # Exclude pathological graphs where the tip is not attached to a shaft endpoint.
    if len({front[0], order[2][0], back[0]}) not in (1, 3) and len({front[1], order[2][1], back[1]}) not in (1, 3):
        raise ValueError("arrow shaft is not straight")
    if coord_set != set(coords):
        raise ValueError("invalid arrow coordinates")
    return (dr, dc), colors_tip_to_tail


def _render_canvas(
    template_mask: np.ndarray,
    direction: tuple[int, int],
    colors_tip_to_tail: list[int],
    canvas_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    layers = _peel_layers(template_mask)
    h, w = template_mask.shape
    out = np.full(canvas_shape, bg, dtype=int)
    base_r = (canvas_shape[0] - h) // 2
    base_c = (canvas_shape[1] - w) // 2

    for idx, coords in enumerate(layers):
        color = int(colors_tip_to_tail[idx])
        shift_r = idx * direction[0]
        shift_c = idx * direction[1]
        for r, c in coords:
            rr = int(base_r + r + shift_r)
            cc = int(base_c + c + shift_c)
            if 0 <= rr < canvas_shape[0] and 0 <= cc < canvas_shape[1]:
                out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the input into template / signal / canvas regions, peel the template into
    onion-like contour layers, then print those layers onto the right canvas using
    the arrow's direction and the tip-to-tail color order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    div1, div2, zone_w = _find_dividers(g, bg)

    template_zone = g[:, :div1]
    signal_zone = g[:, div1 + 1 : div2]
    canvas_zone = g[:, div2 + 1 :]
    assert template_zone.shape[1] == signal_zone.shape[1] == canvas_zone.shape[1] == zone_w

    template_mask = template_zone != bg
    template_mask = _trim_mask(template_mask)
    if template_mask.size == 0 or not template_mask.any():
        raise ValueError("template zone is empty")

    direction, colors_tip_to_tail = _decode_arrow(signal_zone, bg)
    return _render_canvas(template_mask, direction, colors_tip_to_tail, canvas_zone.shape, bg)


def _random_polyomino(max_h: int, max_w: int, area: int) -> np.ndarray | None:
    if max_h <= 0 or max_w <= 0 or area <= 0 or area > max_h * max_w:
        return None
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    if area > h * w:
        return None

    start = (random.randint(0, h - 1), random.randint(0, w - 1))
    cells = {start}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in NEIGHBORS4:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != area:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim_mask(mask)


def _embed_seed(seed: np.ndarray, work_h: int, work_w: int) -> np.ndarray:
    out = np.zeros((work_h, work_w), dtype=bool)
    r0 = (work_h - seed.shape[0]) // 2
    c0 = (work_w - seed.shape[1]) // 2
    out[r0 : r0 + seed.shape[0], c0 : c0 + seed.shape[1]] = seed
    return out


def _boundary_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                candidates.add((nr, nc))
    return list(candidates)


def _grow_from_seed(seed: np.ndarray, target_layers: int, max_h: int, max_w: int) -> np.ndarray | None:
    work_h = max(max_h + 2, seed.shape[0] + 4)
    work_w = max(max_w + 2, seed.shape[1] + 4)
    mask = _embed_seed(seed, work_h, work_w)
    max_extra = min(20 - int(seed.sum()), 8)
    steps = random.randint(0, max_extra)

    for _ in range(steps):
        candidates = _boundary_candidates(mask)
        random.shuffle(candidates)
        added = False
        for cell in candidates:
            trial = mask.copy()
            trial[cell] = True
            trimmed = _trim_mask(trial)
            if (
                trimmed.shape[0] <= max_h
                and trimmed.shape[1] <= max_w
                and int(trimmed.sum()) <= 20
                and _is_hole_free(trimmed)
            ):
                mask = trial
                added = True
                break
        if not added:
            break

    trimmed = _trim_mask(mask)
    if (
        trimmed.size > 0
        and trimmed.shape[0] <= max_h
        and trimmed.shape[1] <= max_w
        and int(trimmed.sum()) <= 20
        and _is_hole_free(trimmed)
        and len(_peel_layers(trimmed)) == target_layers
    ):
        return trimmed
    return None


def _generate_template_mask(max_h: int, max_w: int) -> np.ndarray:
    possible_layers = [1, 2, 3]
    if max_h < 5 or max_w < 5:
        possible_layers = [1, 2]
    target_layers = random.choice(possible_layers)

    for _ in range(TEMPLATE_RETRIES):
        if target_layers == 1:
            area_hi = min(20, max_h * max_w, 12)
            area_lo = min(area_hi, 4)
            area = random.randint(area_lo, area_hi)
            mask = _random_polyomino(max_h, max_w, area)
        elif target_layers == 2:
            seed = random.choice(LAYER2_SEEDS)
            mask = _grow_from_seed(seed, 2, max_h, max_w)
        else:
            mask = _grow_from_seed(LAYER3_SEED, 3, max_h, max_w)

        if mask is None or mask.size == 0:
            continue
        if not _is_hole_free(mask):
            continue
        if len(_peel_layers(mask)) != target_layers:
            continue
        return mask

    if target_layers == 1:
        return np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    if target_layers == 2:
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    return LAYER3_SEED.copy()


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        zone_h = random.randint(10, 14)
        zone_w = random.randint(10, 14)
        direction = random.choice(CARDINALS)

        # Centered canvas printing with at most 3 used layers needs two free cells of slack.
        max_shift = 2
        max_template_h = max(5, zone_h - 2 * max_shift)
        max_template_w = max(5, zone_w - 2 * max_shift)
        template_mask = _generate_template_mask(max_template_h, max_template_w)
        bbox_h, bbox_w = template_mask.shape
        layers = _peel_layers(template_mask)
        if not (1 <= len(layers) <= 4):
            continue
        if bbox_h > zone_h - 2 * (len(layers) - 1) or bbox_w > zone_w - 2 * (len(layers) - 1):
            continue

        colors = random.sample(PALETTE, 7)
        bg, divider, template_color = colors[:3]
        arrow_tail_to_tip = colors[3:7]

        grid = np.full((zone_h, 3 * zone_w + 2), bg, dtype=int)
        grid[:, zone_w] = divider
        grid[:, 2 * zone_w + 1] = divider

        top_min = 1
        left_min = 1
        if zone_h - bbox_h - 1 < top_min or zone_w - bbox_w - 1 < left_min:
            continue
        template_top = random.randint(top_min, zone_h - bbox_h - 1)
        template_left = random.randint(left_min, zone_w - bbox_w - 1)
        _paint_mask(grid[:, :zone_w], template_mask, template_top, template_left, template_color)

        arrow_shape = ARROW_PATHS[direction]
        arrow_rows = [r for r, _c in arrow_shape]
        arrow_cols = [c for _r, c in arrow_shape]
        arrow_h = max(arrow_rows) - min(arrow_rows) + 1
        arrow_w = max(arrow_cols) - min(arrow_cols) + 1
        arrow_r0 = random.randint(1, zone_h - arrow_h - 1)
        arrow_c0 = random.randint(1, zone_w - arrow_w - 1)
        row_shift = -min(arrow_rows)
        col_shift = -min(arrow_cols)
        for idx, (r, c) in enumerate(arrow_shape):
            rr = arrow_r0 + r + row_shift
            cc = zone_w + 1 + arrow_c0 + c + col_shift
            grid[rr, cc] = arrow_tail_to_tip[idx]

        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != (zone_h, zone_w):
            continue
        if np.array_equal(out, np.full_like(out, bg)):
            continue
        return grid

    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
