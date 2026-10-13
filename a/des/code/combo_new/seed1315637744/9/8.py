from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

H_ONLY = 0
V_ONLY = 1
BOTH_AXES = 2
NO_SYMMETRY = 3

LEGEND_ORDER = [RED, BLUE, GREEN, YELLOW]
SLOT_PITCH = 8
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    return len(_components(mask)) == 1


def _symmetry_class(mask: np.ndarray) -> int:
    h_sym = bool(np.array_equal(mask, np.flipud(mask)))
    v_sym = bool(np.array_equal(mask, np.fliplr(mask)))
    if h_sym and v_sym:
        return BOTH_AXES
    if h_sym:
        return H_ONLY
    if v_sym:
        return V_ONLY
    return NO_SYMMETRY


def _orbit_groups(h: int, w: int, keep_h: bool, keep_v: bool) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    groups: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {(r, c)}
            if keep_h:
                orbit.add((h - 1 - r, c))
            if keep_v:
                orbit.add((r, w - 1 - c))
            if keep_h and keep_v:
                orbit.add((h - 1 - r, w - 1 - c))
            group = sorted(orbit)
            seen.update(group)
            groups.append(group)
    return groups


def _orbit_adjacency(groups: list[list[tuple[int, int]]]) -> list[list[int]]:
    cell_to_group: dict[tuple[int, int], int] = {}
    for idx, group in enumerate(groups):
        for cell in group:
            cell_to_group[cell] = idx

    adj: list[set[int]] = [set() for _ in groups]
    for idx, group in enumerate(groups):
        for r, c in group:
            for nr, nc in _neighbors4(r, c):
                other = cell_to_group.get((nr, nc))
                if other is not None and other != idx:
                    adj[idx].add(other)
    return [sorted(neis) for neis in adj]


def _sample_connected_orbit_mask(
    h: int, w: int, keep_h: bool, keep_v: bool, rng: random.Random
) -> np.ndarray:
    groups = _orbit_groups(h, w, keep_h=keep_h, keep_v=keep_v)
    adjacency = _orbit_adjacency(groups)
    start = rng.randrange(len(groups))
    target = rng.randint(1, len(groups))

    selected = {start}
    frontier = list(adjacency[start])
    while frontier and len(selected) < target:
        pick_idx = rng.randrange(len(frontier))
        node = frontier.pop(pick_idx)
        if node in selected:
            continue
        selected.add(node)
        for nxt in adjacency[node]:
            if nxt not in selected:
                frontier.append(nxt)

    mask = np.zeros((h, w), dtype=bool)
    for idx in selected:
        for r, c in groups[idx]:
            mask[r, c] = True
    return mask


def _random_shape(category: int, rng: random.Random) -> np.ndarray:
    for _ in range(600):
        if category == H_ONLY:
            h = rng.randint(2, 5)
            w = rng.randint(3, 5)
            mask = _sample_connected_orbit_mask(h, w, keep_h=True, keep_v=False, rng=rng)
        elif category == V_ONLY:
            h = rng.randint(3, 5)
            w = rng.randint(2, 5)
            mask = _sample_connected_orbit_mask(h, w, keep_h=False, keep_v=True, rng=rng)
        elif category == BOTH_AXES:
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
            mask = _sample_connected_orbit_mask(h, w, keep_h=True, keep_v=True, rng=rng)
        else:
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
            mask = _sample_connected_orbit_mask(h, w, keep_h=False, keep_v=False, rng=rng)

        if not mask.any():
            continue
        mask = _crop_to_bbox(mask)
        if min(mask.shape) < 2 or mask.sum() < 3:
            continue
        if not _is_connected(mask):
            continue
        if _symmetry_class(mask) != category:
            continue

        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.rot90(mask, 2)
        return mask.astype(bool)

    raise RuntimeError(f"failed to sample a shape for category {category}")


def _draw_outer_border(
    out: np.ndarray, bbox: tuple[int, int, int, int], color: int, bg: int
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)

    for c in range(cc0, cc1 + 1):
        if out[rr0, c] == bg:
            out[rr0, c] = color
        if out[rr1, c] == bg:
            out[rr1, c] = color
    for r in range(rr0 + 1, rr1):
        if out[r, cc0] == bg:
            out[r, cc0] = color
        if out[r, cc1] == bg:
            out[r, cc1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify grey objects by reflection symmetry and surround each one with the legend-selected border color.

    Input structure:
    - The background is black.
    - Above the last row are several disjoint grey 4-connected objects, each fitting inside a compact bbox no larger than 5x5.
    - The bottom row is a legend of four single colored pixels separated by one black cell, ordered left-to-right as the colors for horizontal-only, vertical-only, both-axis, and no symmetry.

    Rule:
    - Remove the legend row from the output canvas.
    - For each grey object, crop to its minimal bounding box and test whether that mask matches its top-bottom reflection and/or its left-right reflection.
    - Use the legend order to map the detected symmetry class to a border color.
    - Draw a one-pixel rectangular border on the bbox expanded by one cell, only painting background cells and never overwriting the original grey object.

    Key insight:
    - All objects share the same fill color, so the discriminating feature is geometric symmetry rather than color, size, or position; the legend converts that geometric class into the visible border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = int(np.bincount(g.ravel()).argmax())
    legend_row = g[-1]
    legend_positions = np.flatnonzero(legend_row != bg)
    assert len(legend_positions) == 4
    legend_colors = [int(legend_row[c]) for c in legend_positions]

    work = g[:-1].copy()
    non_bg_colors = sorted(set(map(int, work.ravel())) - {bg})
    assert len(non_bg_colors) == 1
    obj_color = int(non_bg_colors[0])

    out = work.copy()
    for coords in _components(work == obj_color):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        mask = work[r0 : r1 + 1, c0 : c1 + 1] == obj_color
        border_color = legend_colors[_symmetry_class(mask)]
        _draw_outer_border(out, (r0, c0, r1, c1), border_color, bg)

    return out


def _slot_layout(count: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int]]]:
    candidates = []
    for rows, cols in ((2, 2), (2, 3), (3, 2)):
        if rows * cols < count:
            continue
        for margins in (
            (1, 1, 1, 1),
            (1, 2, 1, 2),
            (2, 1, 2, 1),
            (2, 2, 1, 1),
            (1, 1, 2, 2),
            (2, 2, 2, 2),
            (3, 1, 2, 1),
            (1, 3, 1, 2),
        ):
            top, left, bottom, right = margins
            work_h = top + rows * SLOT_PITCH - 1 + bottom
            work_w = left + cols * SLOT_PITCH - 1 + right
            if work_h + 1 <= 30 and work_w <= 30:
                candidates.append((rows, cols, top, left, bottom, right, work_h, work_w))

    rows, cols, top, left, _bottom, _right, work_h, work_w = rng.choice(candidates)
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    chosen = slots[:count]
    slot_origins = [(top + r * SLOT_PITCH, left + c * SLOT_PITCH) for r, c in chosen]
    return work_h, work_w, slot_origins


def _build_case(rng: random.Random) -> np.ndarray:
    categories = [H_ONLY, V_ONLY, BOTH_AXES, NO_SYMMETRY]
    extra_pool = categories[:]
    rng.shuffle(extra_pool)
    categories.extend(extra_pool[: rng.randint(0, 2)])
    rng.shuffle(categories)

    shapes = [_random_shape(category, rng) for category in categories]
    work_h, work_w, slot_origins = _slot_layout(len(shapes), rng)

    grid = np.full((work_h + 1, work_w), BLACK, dtype=int)

    for mask, (slot_r, slot_c) in zip(shapes, slot_origins):
        sh, sw = mask.shape
        top = rng.randint(slot_r + 1, slot_r + 6 - sh)
        left = rng.randint(slot_c + 1, slot_c + 6 - sw)
        patch = grid[top : top + sh, left : left + sw]
        patch[mask] = GREY

    legend_start = rng.randint(0, work_w - 7)
    for idx, color in enumerate(LEGEND_ORDER):
        grid[work_h, legend_start + 2 * idx] = color

    # Validate the generated sample against the solver and its intended branch coverage.
    out = main(grid)
    assert out.shape == (work_h, work_w)
    seen_classes = set()
    for coords in _components(grid[:-1] == GREY):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        seen_classes.add(_symmetry_class(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY))
    assert seen_classes == {H_ONLY, V_ONLY, BOTH_AXES, NO_SYMMETRY}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        try:
            return _build_case(rng)
        except AssertionError:
            continue
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid symmetry-border ARC input")
