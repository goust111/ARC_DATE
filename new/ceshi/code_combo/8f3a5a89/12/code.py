# Combo variant of 8f3a5a89 (index 12)
# combo_spec: D9/1+D12/0
# Name: 仅输出边界环+小可达区域

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D9/1 + D12/0 combo
# a) Concrete behavior and trace:
#    The input still uses the original vocabulary: azure background, one unique
#    magenta corner marker at bottom-left, blue wall-colored pixels, and optional
#    blue decorative blobs. The solver must still separate border-anchored
#    structural walls from non-structural decorations before drawing the orange
#    inner boundary of the marker-reachable side. The D9 part changes only the
#    emitted scene: instead of keeping the whole processed layout, the output
#    becomes a sparse scaffold that preserves only structural blue walls and the
#    orange boundary ring. The D12 part changes the scene geometry seen during
#    training: the marker-reachable area is no longer a broad open side but a
#    compact bottom-left room covering only 15-35% of the grid.
#    A concrete trace is a 14x16 grid with a 5x5 reachable room in the
#    bottom-left. A thick blue vertical barrier stands just to the room's right,
#    a thick blue horizontal barrier sits just above the room, and extra blue
#    masses are attached on the unreachable side so the wall structure visibly
#    dominates the canvas. The original task would output the full processed
#    scene, including any visible blue decoration placed inside the room. The
#    combo output instead keeps only the structural blue scaffold and the orange
#    room boundary; the room interior, the unreachable side, the marker, and the
#    visible decorations all become background.
# b) Complexity calibration:
#    A purely thin-line room would satisfy the ratio constraint but would not
#    look visually distinct enough from the source task, so the generator adds
#    attached wall masses on the unreachable side. That keeps the reachable
#    region small and makes the output read as "small orange loop embedded inside
#    a larger wall construction". Variation is kept to a few repeated degrees of
#    freedom: room width/height, wall thickness, and the size of top/right wall
#    masses. This is enough to avoid trivial memorization without creating too
#    many unrelated patterns.
# c) Implicit assumptions review:
#    - Marker detection still assumes one unique non-background corner pixel.
#      The generator fixes the marker at bottom-left and keeps it unique.
#    - Wall-color inference still assumes a single dominant non-bg non-marker
#      color. Only blue walls are used.
#    - Structural-wall recovery still assumes that every true structural wall is
#      part of a border-touching wall component adjacent to the first flood. The
#      generator makes one connected border-touching wall component whose base
#      faces the small room, so the whole component is recoverable.
#    - The second flood still depends on non-structural blue decorations being
#      traversable. Visible decorations are placed only in the room interior, and
#      hidden decorations are kept non-border and separated from structural walls.
#    - Because D9 outputs only the scaffold, writing into a copy of the input
#      would incorrectly preserve marker and decoration pixels. The solver must
#      emit a fresh background canvas.
#    - The background must remain the most common color for `get_bg(...)` to be
#      valid. The generator explicitly rejects layouts where wall pixels would
#      outnumber background pixels.
# d) Placement legality:
#    The structural wall mask is built from rectangles fully inside bounds. The
#    small room is always at least 4x4 so it can contain an interior distinct
#    from the 1-cell ring. Visible decorations are placed only on room-interior
#    background cells, which keeps them disconnected from structural walls and
#    away from the ring. Hidden decorations are placed only in unreachable
#    background cells after removing border cells and a 4-neighborhood around the
#    structural wall component, so they cannot be mistaken for structural walls.
# e) Visual observability:
#    From a few examples, a human can see two stable signals at once: the orange
#    output is only a compact loop near the bottom-left, and extra blue pixels
#    that were visible inside the room disappear in the output. This makes both
#    the sparse-output rule and the small-room meta change visible without relying
#    on invisible numeric properties.
# f) Visual cleanliness:
#    The generator uses one dominant wall construction rather than many unrelated
#    barriers, so the scene reads as a single room plus surrounding walls. The
#    orange ring is required to be one 4-connected component, and visible
#    decorations are small and confined to the room interior, avoiding clutter on
#    the boundary itself.
# g) Change inferability:
#    No new colors are introduced. Blue still means "wall-like thing" in the
#    input, orange still marks the reachable-side inner boundary, and the output
#    keeps blue only for structural walls. The rule is inferable from structure:
#    pixels disappear because they are not part of the true room boundary, not
#    because of any hidden color ordering.


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

REACH_RATIO_RANGE = (0.15, 0.35)
WALL_RATIO_MAX = 0.44

ROOM_MIN_DIM = 4
WALL_THICKNESS_RANGE = (1, 2)

NUM_VISIBLE_DECOR_RANGE = (1, 2)
NUM_HIDDEN_DECOR_RANGE = (0, 2)
DECOR_AREA_RANGE = (1, 2)
DECOR_PADDING = 1

GEN_ATTEMPTS = 500


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _boundary_fill_mask(
    bg_cells: np.ndarray,
    reach: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    fill = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        if not bg_cells[r, c]:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            fill[r, c] = True
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            fill[r, c] = True
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            fill[r, c] = True
    return fill


def _recover_walls_and_reach(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    other_cols = [col for col in counts if col not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda col: counts[col])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls.update(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )
    return bg, wall_col, marker, walls, reach


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _paint_output(
    g: np.ndarray,
    bg: int,
    wall_col: int,
    walls: set[tuple[int, int]],
    reach: set[tuple[int, int]],
) -> np.ndarray:
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)
    for r, c in walls:
        out[r, c] = wall_col
    fill = _boundary_fill_mask(g == bg, reach, walls, h, w)
    out[fill] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the compact room boundary scaffold around the marker-side region.

    Input structure:
    - The dominant color is background, a unique magenta corner pixel marks the flood origin, and blue pixels mix true structural walls with small decorative blobs.
    - Border-touching structural walls, together with the outer frame, isolate a small marker-reachable room near the bottom-left while larger wall masses dominate the unreachable side.
    - Decorative blue pixels may appear inside the room or behind the structural barrier, but they do not define the room boundary.

    Rule:
    - Locate the unique corner marker, infer the blue wall color, and flood from the marker while treating all blue pixels as blocked to identify which border-touching wall components genuinely face the reachable side.
    - Flood again while allowing travel through non-structural blue pixels so decorations do not change the geometry of the reachable room.
    - Output only the boundary scaffold: keep structural wall pixels blue and paint orange on reachable background pixels that lie on the room's inner boundary; erase the marker, decorations, reachable interior, and unreachable area to background.

    Key insight:
    - The crucial distinction is not "all blue survives" but "only boundary-defining blue survives": the solver must recover the real structural wall component first, then emit a sparse wall-plus-ring mask for the small reachable room.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_col, _marker, walls, reach = _recover_walls_and_reach(g)
    return _paint_output(g, bg, wall_col, walls, reach)


def _build_structural_mask(h: int, w: int, room_h: int, room_w: int, thickness: int, rng: random.Random) -> np.ndarray:
    structural = np.zeros((h, w), dtype=bool)
    room_r0 = h - room_h

    # Base L-shaped barrier that makes the marker-reachable room exactly the
    # bottom-left rectangle [room_r0:h, 0:room_w).
    structural[:, room_w : room_w + thickness] = True
    structural[room_r0 - thickness : room_r0, : room_w + thickness] = True

    max_top_h = room_r0 - thickness
    if max_top_h > 0:
        top_h = rng.randint(1, max_top_h)
        top_w_min = min(w, room_w + thickness)
        top_w = rng.randint(top_w_min, w)
        structural[room_r0 - thickness - top_h : room_r0 - thickness, :top_w] = True

    max_right_w = w - (room_w + thickness)
    if max_right_w > 0:
        right_w = rng.randint(1, max_right_w)
        right_h = rng.randint(max(2, h // 3), h)
        r0 = rng.randint(0, h - right_h)
        structural[r0 : r0 + right_h, room_w + thickness : room_w + thickness + right_w] = True

    return structural


def _pick_room_dims(h: int, w: int, thickness: int, rng: random.Random) -> tuple[int, int] | None:
    max_room_h = h - thickness - 1
    max_room_w = w - thickness - 1
    if max_room_h < ROOM_MIN_DIM or max_room_w < ROOM_MIN_DIM:
        return None

    for _ in range(120):
        room_h = rng.randint(ROOM_MIN_DIM, max_room_h)
        room_w = rng.randint(ROOM_MIN_DIM, max_room_w)
        reach_ratio = (room_h * room_w) / float(h * w)
        if REACH_RATIO_RANGE[0] <= reach_ratio <= REACH_RATIO_RANGE[1]:
            return room_h, room_w
    return None


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed

    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 2:
            shapes.append((1, 1))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=0,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)
    thickness = rng.randint(*WALL_THICKNESS_RANGE)

    room_dims = _pick_room_dims(h, w, thickness, rng)
    if room_dims is None:
        return None
    room_h, room_w = room_dims

    structural = _build_structural_mask(h, w, room_h, room_w, thickness, rng)
    if structural[marker_rc]:
        return None

    wall_ratio = structural.mean()
    if wall_ratio > WALL_RATIO_MAX:
        return None

    room_r0 = h - room_h
    expected_reach = np.zeros((h, w), dtype=bool)
    expected_reach[room_r0:h, :room_w] = True
    reach_ratio = expected_reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_ratio <= REACH_RATIO_RANGE[1]):
        return None

    base_grid = np.full((h, w), BG, dtype=int)
    base_grid[structural] = WALL
    base_grid[marker_rc] = MARKER

    bg, wall_col, _marker, recovered_walls, recovered_reach = _recover_walls_and_reach(base_grid)
    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in recovered_walls:
        wall_mask[r, c] = True
    reach_mask = np.zeros((h, w), dtype=bool)
    for r, c in recovered_reach:
        reach_mask[r, c] = True

    if bg != BG or wall_col != WALL:
        return None
    if not np.array_equal(wall_mask, structural):
        return None
    if not np.array_equal(reach_mask, expected_reach):
        return None

    out = _paint_output(base_grid, bg, wall_col, recovered_walls, recovered_reach)
    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    visible_allowed = reach_mask & ~(fill_mask | structural)
    if visible_allowed.sum() < 2:
        return None
    visible_count = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    visible_placed = _place_decor(visible_allowed, visible_count, rng)
    if visible_placed is None or visible_placed.sum() < 1:
        return None

    hidden_allowed = (~reach_mask) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False
    hidden_allowed &= ~_dilate4(structural)
    hidden_count = rng.randint(*NUM_HIDDEN_DECOR_RANGE)
    hidden_placed = _place_decor(hidden_allowed, hidden_count, rng)
    if hidden_placed is None:
        return None

    inp = base_grid.copy()
    inp[visible_placed] = WALL
    inp[hidden_placed] = WALL

    if np.count_nonzero(inp == BG) <= np.count_nonzero(inp == WALL):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    if np.count_nonzero(inp == WALL) <= np.count_nonzero(out == WALL):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant 12")
