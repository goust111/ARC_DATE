from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RAY_COLORS = (2, 3, 4)
MAROON = 9
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PANEL_H_RANGE = (5, 7)
PANEL_W_RANGE = (11, 15)
PANEL_FG_RANGE = (4, 6)
COLLISION_COUNT_OPTIONS = (1, 2)
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _chebyshev_far_enough(
    cell: tuple[int, int], others: list[tuple[int, int]] | set[tuple[int, int]], min_gap: int = 1
) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > min_gap for rr, cc in others)


def _find_divider_row(grid: np.ndarray) -> int:
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLUE)]
    assert len(divider_rows) == 1, "expected exactly one full-width blue divider row"
    return int(divider_rows[0])


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    divider = _find_divider_row(grid)
    top = grid[:divider, :]
    bottom = grid[divider + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "divider must split two equal non-empty panels"
    assert not np.any(top == BLUE) and not np.any(bottom == BLUE), "blue is reserved for the divider"
    return top, bottom


def _overlay_panels(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    top_fg = top != BLACK
    bottom_fg = bottom != BLACK
    collisions = top_fg & bottom_fg

    out = np.zeros_like(top)
    out[top_fg & ~bottom_fg] = top[top_fg & ~bottom_fg]
    out[bottom_fg & ~top_fg] = bottom[bottom_fg & ~top_fg]
    out[collisions] = MAROON
    return out


def _extend_rays(base: np.ndarray) -> np.ndarray:
    out = base.copy()
    blockers = base != BLACK
    for r0, c0 in np.argwhere(base == MAROON):
        r0 = int(r0)
        c0 = int(c0)
        for dr, dc in DIRS:
            r, c = r0 + dr, c0 + dc
            while _in_bounds(base.shape, r, c) and not blockers[r, c]:
                out[r, c] = MAROON
                r += dr
                c += dc
    return out


def _ray_segments(base: np.ndarray) -> list[dict[str, object]]:
    blockers = base != BLACK
    segments: list[dict[str, object]] = []
    for r0, c0 in np.argwhere(base == MAROON):
        start = (int(r0), int(c0))
        for dr, dc in DIRS:
            r, c = start[0] + dr, start[1] + dc
            length = 0
            while _in_bounds(base.shape, r, c) and not blockers[r, c]:
                length += 1
                r += dr
                c += dc
            stop = "edge" if not _in_bounds(base.shape, r, c) else "obstacle"
            blocker_color = None if stop == "edge" else int(base[r, c])
            segments.append(
                {
                    "start": start,
                    "dir": (dr, dc),
                    "length": length,
                    "stop": stop,
                    "blocker_color": blocker_color,
                }
            )
    return segments


def _render_input(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    h, w = top.shape
    inp = np.zeros((2 * h + 1, w), dtype=int)
    inp[:h, :] = top
    inp[h, :] = BLUE
    inp[h + 1 :, :] = bottom
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two panel drawings, turn overlap collisions into maroon emitters, and extend their rays.

    Input structure:
    - The background is black, and a single full-width blue row divides the grid into
      two equal-height panels.
    - The upper and lower panels contain sparse colored foreground pixels at local panel
      coordinates, with possible overlaps between the two panels.
    - Overlaps represent collisions between the aligned panels rather than ordinary
      preserved foreground cells.

    Rule:
    - Delete the blue divider row and align the lower panel directly over the upper panel.
    - Keep lone foreground pixels in their original colors, but recolor every aligned
      foreground collision to maroon.
    - From each maroon collision, extend rays up, down, left, and right through black
      cells only; stop just before the first non-black cell or the grid edge, and recolor
      every traversed ray cell maroon.

    Key insight:
    - The panel overlay is only the first stage: collision cells become new functional
      sources whose ray lengths depend on the merged post-overlay obstacle layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, bottom = _split_panels(g)
    base = _overlay_panels(top, bottom)
    return _extend_rays(base)


def _edge_distances(shape: tuple[int, int], cell: tuple[int, int]) -> dict[tuple[int, int], int]:
    h, w = shape
    r, c = cell
    return {
        (-1, 0): r,
        (1, 0): h - 1 - r,
        (0, -1): c,
        (0, 1): w - 1 - c,
    }


def _ray_cells(
    start: tuple[int, int], direction: tuple[int, int], distance: int
) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(start[0] + dr * step, start[1] + dc * step) for step in range(1, distance + 1)]


def _choose_anchor(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(interior)
    for anchor in interior:
        dists = _edge_distances(shape, anchor)
        obstacle_dirs = [d for d, dist in dists.items() if dist >= 3]
        edge_dirs = [d for d, dist in dists.items() if dist >= 2]
        if not obstacle_dirs or len(edge_dirs) < 2:
            continue
        obstacle_dir = random.choice(obstacle_dirs)
        edge_choices = [d for d in edge_dirs if d != obstacle_dir]
        if not edge_choices:
            continue
        edge_dir = random.choice(edge_choices)
        return anchor, obstacle_dir, edge_dir
    return None


def _pick_sparse_cells(
    candidates: list[tuple[int, int]],
    count: int,
    reserved: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    placed = list(reserved)
    for cell in candidates:
        if not _chebyshev_far_enough(cell, placed, min_gap=1):
            continue
        chosen.append(cell)
        placed.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _semantic_checks(inp: np.ndarray) -> bool:
    top, bottom = _split_panels(inp)
    top_fg = top != BLACK
    bottom_fg = bottom != BLACK
    collisions = top_fg & bottom_fg
    collision_coords = [tuple(map(int, rc)) for rc in np.argwhere(collisions)]
    if len(collision_coords) not in COLLISION_COUNT_OPTIONS:
        return False
    if not np.all(top[collisions] != bottom[collisions]):
        return False

    top_count = int(np.count_nonzero(top_fg))
    bottom_count = int(np.count_nonzero(bottom_fg))
    if not (PANEL_FG_RANGE[0] <= top_count <= PANEL_FG_RANGE[1]):
        return False
    if not (PANEL_FG_RANGE[0] <= bottom_count <= PANEL_FG_RANGE[1]):
        return False

    overlay = _overlay_panels(top, bottom)
    out = main(inp)
    if int(np.count_nonzero(out == MAROON)) <= len(collision_coords):
        return False

    overlay_positions = [tuple(map(int, rc)) for rc in np.argwhere(overlay != BLACK)]
    if not all(_chebyshev_far_enough(cell, [other for other in overlay_positions if other != cell], min_gap=0) for cell in overlay_positions):
        return False

    segments = _ray_segments(overlay)
    by_start: dict[tuple[int, int], list[dict[str, object]]] = {}
    for segment in segments:
        by_start.setdefault(segment["start"], []).append(segment)
    if not all(any(int(seg["length"]) > 0 for seg in group) for group in by_start.values()):
        return False

    has_edge_stop = any(int(seg["length"]) >= 1 and seg["stop"] == "edge" for seg in segments)
    has_fg_blocker = any(
        int(seg["length"]) >= 1
        and seg["stop"] == "obstacle"
        and int(seg["blocker_color"]) in RAY_COLORS
        for seg in segments
    )
    has_long_ray = any(int(seg["length"]) >= 2 for seg in segments)
    return has_edge_stop and has_fg_blocker and has_long_ray


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    width = random.randint(*PANEL_W_RANGE)
    shape = (panel_h, width)

    anchor_info = _choose_anchor(shape)
    if anchor_info is None:
        return None
    anchor, obstacle_dir, edge_dir = anchor_info
    edge_dists = _edge_distances(shape, anchor)
    blocker_distance = random.randint(2, edge_dists[obstacle_dir])
    blocker = _ray_cells(anchor, obstacle_dir, blocker_distance)[-1]

    collision_count = random.choice(COLLISION_COUNT_OPTIONS)
    top_count = random.randint(*PANEL_FG_RANGE)
    bottom_count = random.randint(*PANEL_FG_RANGE)
    top_only_count = top_count - collision_count
    bottom_only_count = bottom_count - collision_count

    protected_empty = set(_ray_cells(anchor, edge_dir, edge_dists[edge_dir]))
    protected_empty.update(_ray_cells(anchor, obstacle_dir, blocker_distance - 1))
    protected_empty.discard(blocker)

    collisions = [anchor]
    reserved = [anchor, blocker]

    if collision_count == 2:
        interior = [
            (r, c)
            for r in range(1, panel_h - 1)
            for c in range(1, width - 1)
            if (r, c) not in protected_empty and (r, c) not in reserved
        ]
        second = _pick_sparse_cells(interior, 1, reserved)
        if second is None:
            return None
        collisions.extend(second)
        reserved.extend(second)

    total_overlay = top_only_count + bottom_only_count + collision_count
    extra_needed = total_overlay - len(collisions)
    candidates = [
        (r, c)
        for r in range(panel_h)
        for c in range(width)
        if (r, c) not in protected_empty and (r, c) not in reserved
    ]
    extras = _pick_sparse_cells(candidates, extra_needed, reserved)
    if extras is None:
        return None

    top = np.zeros(shape, dtype=int)
    bottom = np.zeros(shape, dtype=int)

    for cell in collisions:
        top_color, bottom_color = random.sample(RAY_COLORS, 2)
        top[cell] = top_color
        bottom[cell] = bottom_color

    blocker_owner = "top" if top_only_count > 0 and (bottom_only_count == 0 or random.random() < 0.5) else "bottom"
    if blocker_owner == "top":
        top[blocker] = random.choice(RAY_COLORS)
        top_only_count -= 1
    else:
        bottom[blocker] = random.choice(RAY_COLORS)
        bottom_only_count -= 1

    random.shuffle(extras)
    top_extras = extras[:top_only_count]
    bottom_extras = extras[top_only_count : top_only_count + bottom_only_count]
    if len(top_extras) != top_only_count or len(bottom_extras) != bottom_only_count:
        return None

    for cell in top_extras:
        top[cell] = random.choice(RAY_COLORS)
    for cell in bottom_extras:
        bottom[cell] = random.choice(RAY_COLORS)

    inp = _render_input(top, bottom)
    return inp if _semantic_checks(inp) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid collision-ray ARC input")
