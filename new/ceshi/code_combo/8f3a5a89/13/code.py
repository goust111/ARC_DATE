# Combo variant of 8f3a5a89 (index 13)
# combo_spec: D9/2+D17/0
# Name: 二值化输出+双色背景
#
# Design analysis
# =================
# 1) Concrete behavior trace
#    - Base task: flood from the corner marker, identify the border-touching blue
#      walls that actually block motion, erase hidden blue decorations, and paint
#      only a thin orange boundary on the reachable side.
#    - This combo keeps the same reachability reasoning but changes the visual
#      target. The input already uses two region colors: azure on the marker-
#      reachable background side and black on the blocked side. Blue pixels still
#      mix structural walls, reachable decorations, and hidden decorations.
#    - Example trace on a small room: if the bottom-left marker can reach an
#      azure corridor plus one isolated interior blue blob, then the output paints
#      both the corridor and that blob azure, while the structural wall and the
#      black-side cells all become black. The orange-border step disappears.
#    - Difference from the base task: the solver now has to recover the whole
#      flood-fill partition, not just its 1-cell inner rim.
#
# 2) Complexity calibration
#    - If the input only differed by recoloring the empty background, the task
#      would collapse into a direct color copy. To prevent that, every generated
#      sample contains at least one reachable blue decoration that must turn into
#      the reachable color in the output.
#    - Structural blue walls remain border-touching and hidden blue decorations
#      may appear on the blocked side, so blue cannot be mapped to a single
#      output color.
#    - Both region colors are forced to occupy substantial area, keeping the
#      binary partition obvious to a human while preserving enough blue ambiguity
#      that the flood-fill logic still matters.
#
# 3) Implicit-assumption review for main()
#    - The original `get_bg(...)` assumption no longer holds because there are two
#      background colors. The new solver instead detects the marker first, then
#      assumes the least frequent remaining color is the blue wall/decor color.
#    - That requires both region colors to each be more common than blue. The
#      generator enforces reachability ratios and small decoration budgets so this
#      remains true.
#    - Structural-wall recovery still depends on the original invariant: true
#      barriers are the wall-color components that touch the border and are
#      adjacent to the initial reach computed while all blue cells are blocked.
#    - The final binary output assumes there are exactly two non-wall region
#      colors left after removing marker and wall roles. The generator uses only
#      black and azure for that pair and always includes both.
#
# 4) Placement legality
#    - Structural shapes are placed as non-overlapping border-touching 4-connected
#      blue components that never cover the bottom-left marker.
#    - Reachable decorations are placed only inside the reachable region and kept
#      off the border so the solver cannot misread them as structural walls.
#    - Hidden decorations are placed only inside the blocked region and also kept
#      off the border; overlaps and out-of-bounds placements are rejected by the
#      placement mask plus `random_free_location(...)`.
#
# 5) Visual observability / cleanliness
#    - The two background colors make the coarse partition immediately visible in
#      a few examples, while the blue reachable decorations show that the output
#      is not just "copy the background colors as-is".
#    - The marker remains a single magenta corner pixel, so the human cue for the
#      flood origin is still present.
#    - The output uses exactly two colors, which makes the D9/2 binary target easy
#      to see and clearly different from the base task's orange-lined output.
#
# 6) Fusion-specific metric
#    - A generated sample is considered non-degenerate only if it contains both
#      kinds of blue pixels in the input: at least one blue cell that becomes the
#      reachable color in the output and at least one blue cell that becomes the
#      blocked color. The validation script at the end of this task measures that.
#
# 7) New assumptions audited against the generator
#    - `main()` assumes the marker color appears exactly once on a corner. The
#      generator fixes it at bottom-left.
#    - `main()` assumes at least one reachable non-marker cell exists so the
#      reachable region color can be inferred from the flood result. The generator
#      always leaves space around the marker.
#    - `main()` assumes blue components that touch the border but are not adjacent
#      to the marker-side initial reach do not exist. The generator validates each
#      structural component against the same adjacency rule the solver uses.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, neighbors4, random_free_location


REACH_BG = 8
BLOCKED_BG = 0
WALL = 1
MARKER = 6

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.36, 0.78)
MIN_REGION_CELLS = 18

VISIBLE_DECOR_RANGE = (1, 3)
HIDDEN_DECOR_RANGE = (0, 2)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 400


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


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _shape_vertical_line(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(2, max(2, w // RECT_MAX_DIM_DIV))
    anchor = rng.choice(["top", "right", "tr", "top_mid", "right_mid", "bottom_right"])
    if anchor == "top":
        r0, c0 = 0, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "right":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), w - rw
    elif anchor == "tr":
        r0, c0 = 0, w - rw
    elif anchor == "top_mid":
        r0, c0 = 0, rng.randint(2, max(2, w - rw - 2))
    elif anchor == "right_mid":
        r0, c0 = rng.randint(2, max(2, h - rh - 2)), w - rw
    else:
        r0, c0 = h - rh, w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]


def _shape_left_diamond(h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 3:
        return None
    size = rng.randint(3, max_size)
    r_center = rng.randint(size, h - size - 1)
    top = r_center - (size - 1)
    cells: list[tuple[int, int]] = []
    for i in range(size):
        cells.extend((top + i, c) for c in range(i + 1))
    for i in range(size - 1):
        cells.extend((top + size + i, c) for c in range(size - 1 - i))
    return cells


def _shape_top_triangle(h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells: list[tuple[int, int]] = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    max_size = min(TRI_MAX_SIZE, h // 2, w // 2)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    r_start = rng.randint(0, max(0, h - size - 1))
    cells: list[tuple[int, int]] = []
    for i in range(size):
        length = i + 1
        cells.extend((r_start + i, c) for c in range(w - length, w))
    return cells


SHAPE_FACTORIES = [
    _shape_vertical_line,
    _shape_horizontal_line,
    _shape_rectangle,
    _shape_left_diamond,
    _shape_top_triangle,
    _shape_right_triangle,
]


def _infer_roles(g: np.ndarray) -> tuple[tuple[int, int], int, int, set[int]]:
    h, w = g.shape
    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    non_marker = [c for c in counts if c != marker_col]
    assert len(non_marker) == 3, "expected exactly two region colors and one wall color"

    wall_col = min(non_marker, key=lambda c: counts[c])
    region_cols = {c for c in non_marker if c != wall_col}
    assert len(region_cols) == 2, "expected exactly two region colors"
    return (mr, mc), marker_col, wall_col, region_cols


def _solve_partition(grid: np.ndarray) -> tuple[set[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker_rc, marker_col, wall_col, region_cols = _infer_roles(g)

    reach0 = _flood(
        marker_rc,
        lambda r, c: int(g[r, c]) in region_cols or int(g[r, c]) == marker_col,
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = {(int(r), int(c)) for r, c in comp_arr}
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(
            0 <= nr < h and 0 <= nc < w and (nr, nc) in reach0
            for r, c in comp
            for nr, nc in neighbors4(r, c)
        )
        if adjacent_to_reach0:
            walls |= comp

    reach = _flood(
        marker_rc,
        lambda r, c: (
            int(g[r, c]) in region_cols
            or int(g[r, c]) == marker_col
            or (int(g[r, c]) == wall_col and (r, c) not in walls)
        ),
        h,
        w,
    )

    reach_region_counts = Counter(int(g[r, c]) for r, c in reach if int(g[r, c]) in region_cols)
    assert reach_region_counts, "reachable region has no background-colored cells"
    reach_col = max(reach_region_counts, key=reach_region_counts.get)
    blocked_col = next(c for c in region_cols if c != reach_col)
    return reach, reach_col, blocked_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse the wall-and-marker scene into a full reachable-vs-blocked partition map.

    Input structure:
    - A unique corner marker still indicates the flood-fill start.
    - Blue pixels are a mixture of border-touching structural walls and smaller
      decorative wall-colored blobs.
    - Unlike the base task, the empty space already uses two colors: one on the
      marker-reachable side and another on the blocked side.

    Rule:
    - Flood from the marker while treating both region colors as open space and
      initially treating every blue pixel as blocked.
    - Reclassify as structural only those blue components that touch the border
      and border the marker-side initial reach; other blue pixels are passable
      decorations, just as in the source task.
    - Flood again through the non-structural blue decorations to recover the full
      reachable set, then paint every reachable cell with the reachable region
      color and every remaining cell with the blocked region color.

    Key insight: The orange boundary from the base task was only a partial view of
    the flood result. Here the correct output is the entire two-way partition, so
    blue cells must be classified by reachability rather than copied by color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    reach, reach_col, blocked_col = _solve_partition(g)
    out = np.full((h, w), blocked_col, dtype=int)
    for r, c in reach:
        out[r, c] = reach_col
    return out


def _place_structural_walls(h: int, w: int, marker_rc: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _ in range(STRUCT_SHAPE_RETRIES):
            factory = rng.choice(SHAPE_FACTORIES)
            cells = factory(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells or any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    return structural if placed > 0 else None


def _validate_structural(structural: np.ndarray, reach: np.ndarray, h: int, w: int) -> bool:
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
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
                    border=DECOR_BORDER,
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


def _build_input(
    reach: np.ndarray,
    structural: np.ndarray,
    visible_placed: np.ndarray,
    hidden_placed: np.ndarray,
    marker_rc: tuple[int, int],
) -> np.ndarray:
    inp = np.where(reach, REACH_BG, BLOCKED_BG).astype(int)
    inp[structural] = WALL
    inp[visible_placed] = WALL
    inp[hidden_placed] = WALL
    inp[marker_rc] = MARKER
    return inp


def _binary_output(reach: np.ndarray) -> np.ndarray:
    return np.where(reach, REACH_BG, BLOCKED_BG).astype(int)


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None
    if int(reach.sum()) < MIN_REGION_CELLS or int((~reach).sum()) < MIN_REGION_CELLS:
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    visible_allowed = reach & ~structural
    visible_allowed[marker_rc] = False
    visible_allowed[0, :] = False
    visible_allowed[-1, :] = False
    visible_allowed[:, 0] = False
    visible_allowed[:, -1] = False

    hidden_allowed = (~reach) & ~structural
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    n_visible = rng.randint(*VISIBLE_DECOR_RANGE)
    n_hidden = rng.randint(*HIDDEN_DECOR_RANGE)

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None or not visible_placed.any():
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    inp = _build_input(reach, structural, visible_placed, hidden_placed, marker_rc)
    out = _binary_output(reach)

    counts = Counter(int(v) for v in inp.ravel())
    if not (counts[REACH_BG] > counts[WALL] and counts[BLOCKED_BG] > counts[WALL]):
        return None

    if not np.array_equal(main(inp), out):
        return None

    blue_to_reach = int((out[inp == WALL] == REACH_BG).sum())
    blue_to_blocked = int((out[inp == WALL] == BLOCKED_BG).sum())
    if blue_to_reach <= 0 or blue_to_blocked <= 0:
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
    raise RuntimeError("failed to generate valid input for combo 8f3a5a89/13")
