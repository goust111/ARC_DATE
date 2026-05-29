# Variant of 8f3a5a89
# Dimension: D4 — 终止/边界判定条件 (dst)
# Change: 洪填被结构墙和网格边缘阻挡——结构墙定义为触碰网格边缘且4-邻接可达区域的墙色连通分量
#   → 特定颜色像素作为目标边界

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D4 variant
# ==================================
# a) Concrete behavior and trace
#    - Base scene stays the same: azure background, blue wall color, magenta marker in the
#      bottom-left corner, orange inner border in the output.
#    - New visible object: red single-pixel targets placed in deep interior cells of the
#      marker-reachable side. They are not walls and they keep their red color in the output.
#    - Solver change: the fill treats a target like a terminal contact. When flood reaches a
#      target from a passable side, that target pixel is marked as reached but is not expanded.
#      Because it belongs to the reached set, the four neighboring background cells do not become
#      orange just for touching that target.
#    - Small trace:
#        old logic on ...BG BG BG... with a red pixel in the center:
#          red is "not reached", so its four orthogonal bg neighbors see an unreachable neighbor
#          and turn orange.
#        new logic:
#          red is added into the reached set as a terminal, so those four bg neighbors remain bg
#          unless some real boundary condition also applies.
#        difference:
#          walls still create borders; targets only stop motion locally and keep their own color.
# b) Complexity calibration
#    - Targets are always present and always on the reachable side.
#    - They are placed only at strong interior centers whose four orthogonal neighbors are also
#      interior cells. This makes the old/new difference visually obvious: each target suppresses
#      a plus-shaped orange halo that the original logic would have painted.
#    - Target color is fixed to one color to avoid turning the task into an arbitrary color-lookup
#      problem. The solver still infers it structurally as "the non-wall extra color".
# c) Hidden assumptions audited against the original solver
#    - Original wall inference "most frequent non-bg non-marker color" still works because the
#      generator keeps wall pixels much more numerous than targets.
#    - Original second flood assumed every non-structural wall pixel was passable. That remains
#      true for wall-colored decorations. Targets are the new exception, so they are handled by a
#      separate terminal predicate instead of being folded into passable or unreachable cells.
#    - Original border painting used "4-neighbor not in reach" as a boundary test. That would
#      wrongly paint around targets in the variant. The fix is not a special-case subtraction;
#      the target cells are included in the reached set, so the existing test remains valid.
# d) Placement legality
#    - Targets never overlap marker, walls, or decorations.
#    - Targets are isolated from one another by at least Chebyshev distance 2, so each one has an
#      independent visible effect.
#    - Visible decorations are kept away from targets so they do not mask the target-induced gap.
#    - Hidden decorations are kept off the border so they cannot be misidentified as structural.
# e) Visual observability
#    - In input/output pairs the user sees red pixels inside the open region that do not turn
#      orange and do not generate orange borders around themselves, unlike true walls.
#    - Structural walls still produce the familiar orange lining, so the contrast "wall vs target"
#      is readable from a few examples.
# f) Visual cleanliness
#    - Targets are single pixels, not blobs, which avoids ambiguous partial-terminal components.
#    - Decorations are few and small; target neighborhoods are reserved to keep the motif clean.
# g) Color-rule inference
#    - Only one extra non-wall color is introduced. Its role is supported by repeated placement in
#      interior space, not by a hidden numeric property of the color id.
# Source-task note
#    - The prompt cites 195c6913 as the source of the terminal-stop idea. In this workspace there
#      is an evaluation JSON for 195c6913 but no standalone solver file to inspect, so the variant
#      adopts the minimal collision-stop mechanism stated in the prompt: contact the target cell,
#      mark it, and stop there without crossing it.


def _flood(
    start: tuple[int, int],
    passable,
    terminal,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in vis:
                continue
            if passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
            elif terminal(nr, nc):
                vis.add((nr, nc))
    return vis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the inner border of the marker-reachable side while preserving terminal target pixels.

    Input structure:
    - The background is the most common color; blue components contain both structural walls and
      decorative wall fragments.
    - A unique magenta corner pixel marks the flood start, and one extra non-wall color appears as
      interior target pixels on the reachable side.
    - Structural walls are the blue components that touch the grid edge and face the marker side;
      target pixels are separate blockers that are not themselves border-forming walls.

    Rule:
    - Detect the wall color, identify the structural wall components, and flood from the marker.
    - During the flood, ordinary reachable cells can be crossed, decorative blue pixels can be
      crossed on the marker side, but a target pixel is only marked as reached and never crossed.
    - Remove unreachable blue decorations hidden behind structural walls, then paint orange on
      reachable background cells that lie on the true inner boundary of the marker side.

    Key insight: Walls define the boundary, but target pixels only terminate the flood locally; the
    target cells stay in their own color and do not create orange border cells around themselves.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

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

    other_cols = [c for c in counts if c not in (bg, marker_col)]
    assert len(other_cols) >= 2, "need both wall color and target color"
    wall_col = max(other_cols, key=lambda c: counts[c])
    target_cols = set(other_cols) - {wall_col}
    assert target_cols, "no target color found"

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        lambda r, c: False,
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
            walls |= set(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        lambda r, c: int(g[r, c]) in target_cols,
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    return out


BG = 8
WALL = 1
TARGET = 2
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

TARGET_COUNT_RANGE = (1, 3)
TARGET_MIN_GAP = 2

GEN_ATTEMPTS = 500


def _shape_vertical_line(h, w, rng):
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h, w, rng):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h, w, rng):
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


def _shape_left_diamond(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 3:
        return None
    size = rng.randint(3, max_size)
    r_center = rng.randint(size, h - size - 1)
    top = r_center - (size - 1)
    cells = []
    for i in range(size):
        cells.extend((top + i, c) for c in range(i + 1))
    for i in range(size - 1):
        cells.extend((top + size + i, c) for c in range(size - 1 - i))
    return cells


def _shape_top_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // 2, w // 2)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    r_start = rng.randint(0, max(0, h - size - 1))
    cells = []
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


def _flood_mask(start, blocked, h, w):
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


def _flood_mask_terminal(start, passable, terminal, h, w):
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or reach[nr, nc]:
                continue
            if passable[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
            elif terminal[nr, nc]:
                reach[nr, nc] = True
    return reach


def _touches_border(mask):
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def _dilate4(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask):
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _place_structural_walls(h, w, marker_rc, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells or any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    return structural if placed > 0 else None


def _validate_structural(structural, reach, h, w):
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _compute_fill_mask(structural, reach, bg_mask):
    candidate = reach & bg_mask

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return candidate & (edge | near_struct | near_unreach)


def _strong_target_centers(structural, reach_open, marker_rc):
    h, w = structural.shape
    bg_mask = np.ones((h, w), dtype=bool)
    bg_mask[marker_rc] = False
    candidate = reach_open & bg_mask

    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    candidate &= ~edge
    candidate &= ~_dilate8(structural)
    candidate &= ~_dilate4(~reach_open)

    strong = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(candidate):
        ok = True
        for nr, nc in neighbors4(int(r), int(c)):
            if not (0 <= nr < h and 0 <= nc < w and candidate[nr, nc]):
                ok = False
                break
        if ok:
            strong[r, c] = True
    return strong


def _pick_spaced_cells(mask, count, rng, min_gap):
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    rng.shuffle(cells)
    chosen = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) > min_gap for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _place_decor(allowed_mask, count, rng):
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
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


def _render_pair(structural, targets, visible_decor, hidden_decor, marker_rc):
    h, w = structural.shape
    inp = np.full((h, w), BG, dtype=int)
    inp[structural | visible_decor | hidden_decor] = WALL
    inp[targets] = TARGET
    inp[marker_rc] = MARKER

    passable = (~structural) & (~targets)
    reach = _flood_mask_terminal(marker_rc, passable, targets, h, w)

    out = inp.copy()
    nonstruct_walls = (inp == WALL) & (~structural)
    out[nonstruct_walls & (~reach)] = BG

    fill = _compute_fill_mask(structural, reach, out == BG)
    out[fill] = FILL
    return inp, out, reach


def _counterfactual_fill_delta(structural, targets, visible_decor, hidden_decor, marker_rc):
    inp, out, _ = _render_pair(structural, targets, visible_decor, hidden_decor, marker_rc)
    h, w = structural.shape
    blocked_old = structural | targets
    reach_old = _flood_mask(marker_rc, blocked_old, h, w)
    old_fill = _compute_fill_mask(structural, reach_old, out == BG)
    new_fill = out == FILL
    return int(np.count_nonzero(old_fill & (~new_fill)))


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach_open = _flood_mask(marker_rc, structural, h, w)
    reach_frac = reach_open.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach_open, h, w):
        return None

    strong = _strong_target_centers(structural, reach_open, marker_rc)
    n_strong = int(strong.sum())
    if n_strong == 0:
        return None

    max_targets = min(TARGET_COUNT_RANGE[1], n_strong)
    n_targets = rng.randint(TARGET_COUNT_RANGE[0], max_targets)
    target_cells = _pick_spaced_cells(strong, n_targets, rng, TARGET_MIN_GAP)
    if target_cells is None:
        return None

    targets = np.zeros((h, w), dtype=bool)
    for r, c in target_cells:
        targets[r, c] = True

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    target_buffer = _dilate8(targets)
    interior_bg = strong & (~targets)
    visible_allowed = interior_bg & (~target_buffer)
    hidden_allowed = (~reach_open) & (~structural) & (~targets)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    inp, out, reach = _render_pair(structural, targets, visible_placed, hidden_placed, marker_rc)

    if not np.all(reach[targets]):
        return None

    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    if _counterfactual_fill_delta(structural, targets, visible_placed, hidden_placed, marker_rc) < 4 * n_targets:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D4 variant")
