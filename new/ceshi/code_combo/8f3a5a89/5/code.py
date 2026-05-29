# Combo variant of 8f3a5a89 (index 5)
# combo_spec: D1/0+D4/1
# Name: 射线扫描+目标像素碰撞

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior trace
#    - Base palette stays close to the source task: bg=8, wall=1, marker=6,
#      fill=7, plus one target color TARGET=2.
#    - The marker is still the unique corner pixel at bottom-left.
#    - Reachability changes twice at once:
#      1) D1/0: the open side is no longer found by 4-neighbor flood-fill.
#         Instead the marker emits a one-bend ray fan: first sweep along the
#         marker row/column, then each swept seed cell emits one orthogonal ray.
#      2) D4/1: if a ray meets a target pixel, that target cell is included in
#         the reached set but the ray stops there and does not continue beyond.
#    - Small trace:
#        bottom-left marker -> bottom-row seed ray reaches column 5.
#        From that seed cell an upward ray climbs through bg cells.
#        original task: flood-fill could turn around corners and would ignore
#        any special non-wall single pixel.
#        D1-only: the upward ray would continue until a wall/edge.
#        combo: if a red target sits midway on that upward ray, the target cell
#        becomes reached, keeps its red color, and all cells above it on that
#        column stay unreachable. Orange is painted only on true bg boundary
#        cells of the final ray-swept region.
#
# b) Complexity calibration
#    - A literal four-direction scan from a fixed bottom-left corner would be a
#      thin L-shape. The generator therefore uses the same one-bend fan-out as
#      the D1 reference so the swept set is still a 2D region.
#    - Targets are restricted to interior intersections of row/column sweeps.
#      This makes their role visible from examples: they are clearly not walls,
#      but they also clearly truncate a ray.
#    - Structural walls are top/right anchored so the ray region is observably
#      smaller than the ordinary flood-fill region on the same grid.
#
# c) Hidden-assumption audit
#    - Original `reach0`-based wall detection no longer fits: with rays there is
#      no initial flood bootstrap. The solver now treats all border-touching
#      wall-color components as structural. Generator guarantee: decorative wall
#      fragments never touch the border.
#    - Original connectivity assumption is invalid. Every place that depended on
#      flood reach now depends on ray reach.
#    - Treating targets as ordinary blocked cells would wrongly create orange
#      halos around them. The solver instead inserts target cells into `reach`
#      without allowing further propagation. Generator guarantee: targets are
#      placed in open interior spots where this distinction is visible.
#
# d) Placement legality
#    - Structural wall shapes are non-overlapping and stay on the top/right
#      borders, never covering the marker.
#    - Targets are single pixels placed on base ray paths, away from borders,
#      walls, and one another; no two chosen targets share a row or column, so
#      one target cannot shadow another target's reachability.
#    - Visible decorations are placed only in final reachable bg cells and kept
#      away from targets. Hidden decorations are placed strictly off-border and
#      outside final reach.
#
# e) Visual observability
#    - The generator rejects samples unless ordinary flood-fill reaches
#      substantially more cells than the open ray fan. This exposes D1/0.
#    - It also rejects samples unless removing target-terminal semantics changes
#      the orange output around targets, and unless ignoring targets entirely
#      would extend the ray region. This exposes both parts of D4/1.
#
# f) Visual cleanliness
#    - Only five colors appear, with a single extra target color.
#    - Targets are isolated single pixels and decorations are sparse, so the
#      viewer can distinguish walls, targets, and orange border cells.
#    - The final orange mask is required to be a single 4-connected component to
#      keep the result legible.
#
# g) Rule inferability
#    - There is only one non-wall extra color, repeated as the terminal target
#      type. Its meaning is inferred structurally from repeated placement on the
#      reachable side where rays stop but the pixel itself is preserved.
#
# New solver assumptions and generator guarantees
#    - Solver assumption: the most frequent non-bg, non-marker color is the wall
#      color. Generator guarantee: structural/decoration wall pixels outnumber
#      the targets.
#    - Solver assumption: every other non-bg, non-marker color is a target.
#      Generator guarantee: only one target color is introduced.
#    - Solver assumption: non-structural wall pixels are traversable by the ray
#      scan. Generator guarantee: decorations are small isolated blobs that do
#      not touch the border and therefore cannot be mistaken for structural
#      blockers.
# -----------------------------------------------------------------------------


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


def _ray_cast(
    start: tuple[int, int],
    dr: int,
    dc: int,
    passable,
    terminal,
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    cells: list[tuple[int, int]] = []
    r, c = start
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells, None
        if passable(r, c):
            cells.append((r, c))
            continue
        if terminal(r, c):
            return cells, (r, c)
        return cells, None


def _ray_region_details(
    start: tuple[int, int],
    passable,
    terminal,
    h: int,
    w: int,
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]],
]:
    horizontal_seed = {start}
    vertical_seed = {start}
    reach = {start}
    secondary_paths: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []

    for dc in (-1, 1):
        cells, hit = _ray_cast(start, 0, dc, passable, terminal, h, w)
        horizontal_seed.update(cells)
        reach.update(cells)
        if hit is not None:
            reach.add(hit)

    for dr in (-1, 1):
        cells, hit = _ray_cast(start, dr, 0, passable, terminal, h, w)
        vertical_seed.update(cells)
        reach.update(cells)
        if hit is not None:
            reach.add(hit)

    for cell in list(horizontal_seed):
        for dr in (-1, 1):
            cells, hit = _ray_cast(cell, dr, 0, passable, terminal, h, w)
            reach.update(cells)
            if hit is not None:
                reach.add(hit)
            if cells:
                secondary_paths.append((cell, (dr, 0), cells))

    for cell in list(vertical_seed):
        for dc in (-1, 1):
            cells, hit = _ray_cast(cell, 0, dc, passable, terminal, h, w)
            reach.update(cells)
            if hit is not None:
                reach.add(hit)
            if cells:
                secondary_paths.append((cell, (0, dc), cells))

    return reach, horizontal_seed, vertical_seed, secondary_paths


def _ray_mask_from_booleans(
    start: tuple[int, int],
    structural: np.ndarray,
    targets: np.ndarray | None,
) -> np.ndarray:
    h, w = structural.shape
    target_mask = np.zeros_like(structural) if targets is None else targets
    reach = np.zeros((h, w), dtype=bool)
    region, _, _, _ = _ray_region_details(
        start,
        lambda r, c: (not structural[r, c]) and (not target_mask[r, c]),
        lambda r, c: bool(target_mask[r, c]),
        h,
        w,
    )
    for r, c in region:
        reach[r, c] = True
    return reach


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the inner border of the corner-origin ray-swept region while preserving terminal targets.

    Input structure:
    - The background is the most common color; blue cells contain both border-touching structural
      blockers and small decorative wall fragments.
    - A unique magenta corner pixel marks the scan origin, and one extra non-wall color appears as
      single-pixel targets on the open side.
    - Border-touching blue components are true walls; target pixels are separate terminal stoppers
      that keep their own color.

    Rule:
    - Infer the wall color, treat every border-touching wall component as structural, and start a
      one-bend ray fan from the marker: first scan along the marker row/column, then let those
      swept seed cells emit one orthogonal straight ray each.
    - Rays may cross background and non-structural blue decorations, stop before structural walls,
      and when they hit a target they mark that target cell but do not continue through it.
    - Erase decorative blue pixels that were never swept, then paint orange on background cells on
      the inner boundary of the final swept region.

    Key insight: The open side is determined by line-of-sight rather than flood connectivity, and
    target pixels are terminal contacts rather than walls: they belong to the swept set, keep their
    color, and stop the ray without generating orange borders around themselves.
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

    structural = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            for r, c in comp:
                structural[r, c] = True

    reach, _, _, _ = _ray_region_details(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and not structural[r, c]),
        lambda r, c: int(g[r, c]) in target_cols,
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and not structural[r, c] and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any(structural[nr, nc] for nr, nc in neighbors8(r, c) if 0 <= nr < h and 0 <= nc < w):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w):
            out[r, c] = fill_col

    return out


# --------------- generator ---------------

BG = 8
WALL = 1
TARGET = 2
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (2, 4)
STRUCT_RETRIES = 12

RAY_REACH_RATIO_RANGE = (0.24, 0.84)
MIN_FLOOD_ADVANTAGE = 10
MIN_COLLISION_STOP_DELTA = 3
MIN_TARGET_SEMANTICS_DELTA = 1

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

TARGET_COUNT_RANGE = (1, 1)
TARGET_MIN_GAP = 2

GEN_ATTEMPTS = 1200


def _shape_top_rect(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    rh = rng.randint(2, max(2, min(6, h - 3)))
    rw = rng.randint(2, max(2, min(6, w - 3)))
    c0 = rng.randint(1, max(1, w - rw - 1))
    return [(r, c) for r in range(rh) for c in range(c0, c0 + rw)]


def _shape_top_drop(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    depth = rng.randint(3, max(3, h - 3))
    c = rng.randint(1, w - 2)
    return [(r, c) for r in range(depth)]


def _shape_right_rect(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    rh = rng.randint(2, max(2, min(6, h - 3)))
    rw = rng.randint(2, max(2, min(6, w - 3)))
    r0 = rng.randint(1, max(1, h - rh - 1))
    c0 = w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, w)]


def _shape_right_bar(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    length = rng.randint(3, max(3, w - 3))
    r = rng.randint(1, h - 2)
    return [(r, c) for c in range(w - length, w)]


def _shape_top_right_corner(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    rh = rng.randint(2, max(2, min(5, h - 3)))
    rw = rng.randint(2, max(2, min(5, w - 3)))
    return [(r, c) for r in range(rh) for c in range(w - rw, w)]


SHAPE_FACTORIES = [
    _shape_top_rect,
    _shape_top_drop,
    _shape_right_rect,
    _shape_right_bar,
    _shape_top_right_corner,
]


def _place_structural_walls(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    structural = np.zeros((h, w), dtype=bool)
    num = rng.randint(*NUM_STRUCTS_RANGE)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    if placed == 0:
        return None
    ratio = float(structural.mean())
    if not (0.04 <= ratio <= 0.22):
        return None
    return structural


def _validate_structural(structural: np.ndarray, h: int, w: int) -> bool:
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
    return True


def _compute_fill_mask(structural: np.ndarray, reach: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    candidate = reach & bg_mask

    edge = np.zeros_like(candidate)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return candidate & (edge | near_struct | near_unreach)


def _target_candidate_mask(structural: np.ndarray, marker_rc: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = structural.shape
    reach_open = _ray_mask_from_booleans(marker_rc, structural, None)
    _, horizontal_seed, vertical_seed, _ = _ray_region_details(
        marker_rc,
        lambda r, c: not structural[r, c],
        lambda r, c: False,
        h,
        w,
    )

    vert_tail = np.full((h, w), -1, dtype=int)
    horiz_tail = np.full((h, w), -1, dtype=int)

    for cell in horizontal_seed:
        for dr in (-1, 1):
            cells, _ = _ray_cast(cell, dr, 0, lambda r, c: not structural[r, c], lambda r, c: False, h, w)
            for i, (r, c) in enumerate(cells):
                vert_tail[r, c] = max(vert_tail[r, c], len(cells) - i - 1)

    for cell in vertical_seed:
        for dc in (-1, 1):
            cells, _ = _ray_cast(cell, 0, dc, lambda r, c: not structural[r, c], lambda r, c: False, h, w)
            for i, (r, c) in enumerate(cells):
                horiz_tail[r, c] = max(horiz_tail[r, c], len(cells) - i - 1)

    interior = reach_open.copy()
    interior[marker_rc] = False
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    interior &= ~edge
    interior &= ~_dilate8(structural)
    interior &= ~_dilate4(~reach_open)

    candidate = interior & (vert_tail >= 0) & (horiz_tail >= 0) & ((vert_tail >= 2) | (horiz_tail >= 2))
    return candidate, reach_open


def _pick_targets(mask: np.ndarray, rng: random.Random, count: int) -> list[tuple[int, int]] | None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for r, c in cells:
        if r in used_rows or c in used_cols:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= TARGET_MIN_GAP for rr, cc in chosen):
            continue
        chosen.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
        if len(chosen) == count:
            return chosen
    return None


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
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
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _render_pair(
    structural: np.ndarray,
    targets: np.ndarray,
    visible_decor: np.ndarray,
    hidden_decor: np.ndarray,
    marker_rc: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = structural.shape
    inp = np.full((h, w), BG, dtype=int)
    inp[structural | visible_decor | hidden_decor] = WALL
    inp[targets] = TARGET
    inp[marker_rc] = MARKER

    reach = _ray_mask_from_booleans(marker_rc, structural, targets)

    out = inp.copy()
    nonstruct_walls = (inp == WALL) & (~structural)
    out[nonstruct_walls & (~reach)] = BG

    fill = _compute_fill_mask(structural, reach, out == BG)
    out[fill] = FILL
    return inp, out, reach


def _render_counterfactual_target_blocked(
    structural: np.ndarray,
    targets: np.ndarray,
    visible_decor: np.ndarray,
    hidden_decor: np.ndarray,
    marker_rc: tuple[int, int],
) -> np.ndarray:
    inp = np.full(structural.shape, BG, dtype=int)
    inp[structural | visible_decor | hidden_decor] = WALL
    inp[targets] = TARGET
    inp[marker_rc] = MARKER

    blocked = structural | targets
    reach = _ray_mask_from_booleans(marker_rc, blocked, None)

    out = inp.copy()
    nonstruct_walls = (inp == WALL) & (~structural)
    out[nonstruct_walls & (~reach)] = BG

    fill = _compute_fill_mask(structural, reach, out == BG)
    out[fill] = FILL
    return out


def _counterfactual_metrics(
    structural: np.ndarray,
    targets: np.ndarray,
    visible_decor: np.ndarray,
    hidden_decor: np.ndarray,
    marker_rc: tuple[int, int],
) -> tuple[int, int, int]:
    inp, out, reach = _render_pair(structural, targets, visible_decor, hidden_decor, marker_rc)
    reach_open = _ray_mask_from_booleans(marker_rc, structural, None)
    flood_open = _flood_mask(marker_rc, structural, *structural.shape)
    out_blocked = _render_counterfactual_target_blocked(structural, targets, visible_decor, hidden_decor, marker_rc)

    flood_advantage = int(flood_open.sum() - reach_open.sum())
    collision_stop_delta = int(reach_open.sum() - reach.sum())
    target_semantics_delta = int(np.count_nonzero((out_blocked == FILL) & (out != FILL)))
    return flood_advantage, collision_stop_delta, target_semantics_delta


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, rng)
    if structural is None or structural[marker_rc]:
        return None
    if not _validate_structural(structural, h, w):
        return None

    candidate_targets, reach_open = _target_candidate_mask(structural, marker_rc)
    reach_frac = float(reach_open.mean())
    if not (RAY_REACH_RATIO_RANGE[0] <= reach_frac <= RAY_REACH_RATIO_RANGE[1]):
        return None

    flood_open = _flood_mask(marker_rc, structural, h, w)
    flood_advantage = int(flood_open.sum() - reach_open.sum())
    if flood_advantage < max(MIN_FLOOD_ADVANTAGE, int(0.10 * h * w)):
        return None

    n_candidates = int(candidate_targets.sum())
    if n_candidates == 0:
        return None

    max_targets = min(TARGET_COUNT_RANGE[1], n_candidates)
    n_targets = rng.randint(TARGET_COUNT_RANGE[0], max_targets)
    target_cells = _pick_targets(candidate_targets, rng, n_targets)
    if target_cells is None:
        return None

    targets = np.zeros((h, w), dtype=bool)
    for r, c in target_cells:
        targets[r, c] = True

    final_reach = _ray_mask_from_booleans(marker_rc, structural, targets)
    if not np.all(final_reach[targets]):
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    target_buffer = _dilate8(targets)
    visible_allowed = final_reach & (~structural) & (~targets) & (~target_buffer)
    hidden_allowed = (~final_reach) & (~structural) & (~targets)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    inp, out, reach = _render_pair(structural, targets, visible_placed, hidden_placed, marker_rc)

    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    remaining_bg = (out == BG) & reach
    if int(remaining_bg.sum()) < 4:
        return None

    flood_advantage2, collision_stop_delta, target_semantics_delta = _counterfactual_metrics(
        structural, targets, visible_placed, hidden_placed, marker_rc
    )
    if flood_advantage2 < max(MIN_FLOOD_ADVANTAGE, int(0.10 * h * w)):
        return None
    if collision_stop_delta < MIN_COLLISION_STOP_DELTA * n_targets:
        return None
    if target_semantics_delta < MIN_TARGET_SEMANTICS_DELTA * n_targets:
        return None

    if get_bg(inp) != BG:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant 5")
