# Combo variant of 53fb4810 (index 31)
# combo_spec: D1/0+D3/0+D5/0+D6/0
# Name: 对角大瓦片间歇遇障停止

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Original 53fb4810: one small marker tile sits on a single arm of a plus, then
#      repeats in a straight cardinal direction all the way to the boundary.
#    - This combo changes four linked stages at once:
#      1. D1: the propagation direction is diagonal, not axial.
#      2. D3: only every (skip_gap+1)-th full tile slot is painted; intermediate slots stay bg.
#      3. D5: the seed is a large filled square tile (3x3 or 4x4) with a richer multicolor texture.
#      4. D6: a dedicated obstacle square blocks the next painted slot, so repetition stops before it.
#    - To make the diagonal direction explicit without hidden conventions, the seed now occupies one
#      empty corner cell of the plus bbox and expands outward from that corner. Example:
#        plus bbox corner = NE empty corner, seed side = 3, skip_gap = 1
#        seed at slot 0, visible copies at slots 0/2/4, blank slots 1/3, obstacle at slot 6.
#        The output is therefore a diagonal chain of 3x3 textured squares separated by 3x3 bg gaps,
#        ending in a solid obstacle square of one shared blocker color.
#
# b) Complexity calibration
#    - skip_gap is limited to {1, 2}; this is visually obvious and globally shared inside one grid.
#    - tile side is limited to {3, 4}; this is large enough to look unlike the base task, while still
#      allowing several copies plus an obstacle in a medium grid.
#    - every seed tile uses the same small global seed palette for the whole grid, and every tile must
#      contain every seed color at least once. This gives repeated evidence while also making obstacle
#      detection structurally stable.
#
# c) Hidden assumptions from the original solver that break here
#    - Original adjacency logic expected exactly one cardinal side to determine direction. Here the seed
#      is attached at a bbox corner, so direction is inferred from which seed corner touches the plus.
#    - Original extension logic assumed a single connected strip. Here the demonstrated ray is a set of
#      disconnected full-tile components with blank square slots in between.
#    - Original plus-color detection by "largest big components" can be confused by large obstacle
#      squares. The combo solver therefore identifies plus color by repeated symmetric non-rectangular
#      shapes with empty bbox corners.
#    - Original termination at the boundary is no longer valid. The new painter must inspect future
#      diagonal slots and stop when the next visible slot is a blocker square.
#
# d) Placement legality in the generator
#    - Generation is output-first: place one plus, its seed, every reserved slot on its diagonal lane,
#      and one obstacle square, then move to the next plus.
#    - The generator reserves every slot up to the obstacle, including skipped bg slots, so no later plus
#      or lane can intrude and invalidate slot-based inference.
#    - Pluses are also kept away from existing reserved lanes by a small 4-neighbor buffer so that no
#      pre-extended demo copy can accidentally become adjacent to a different plus.
#
# e) Visual observability
#    - The input still has exactly one long demonstrated ray. That ray now simultaneously teaches:
#      diagonal direction, large-tile period, dashed spacing, and stopping at a solid blocker square.
#    - Other pluses show only compact corner-attached seeds plus the same obstacle-color square farther
#      out, which gives enough repeated evidence to infer that the stopping condition is not just boundary clipping.
#
# f) Visual cleanliness
#    - Whole-slot reservation prevents overlapping diagonals and keeps skipped squares genuinely empty.
#    - Obstacle squares use one dedicated color outside the seed palette, so blockers stay visually distinct
#      from texture colors even though the grid contains large multicolor tiles.
#
# g) Color inference
#    - No hidden color ordering is used. Seed colors only define visible tile texture; the blocker color is
#      globally shared and never appears inside seeds. This lets the solver recover the blocker structurally
#      instead of via numeric color ids.
#
# h) Post-change solver assumptions and generator support
#    - main() assumes every plus has exactly one 4-connected filled square seed component adjacent to it,
#      and that the touching cell is exactly one seed corner. The generator enforces that geometry.
#    - main() assumes one global skip_gap and one global obstacle color per grid. The generator samples one
#      pair and uses it for every ray.
#    - main() assumes full tile slots along a lane are either: the seed texture, bg, or a solid obstacle
#      square. The generator reserves the entire lane so no fourth state appears.


_CARDINALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_CONTACT_CORNER_TO_DIR = {
    (0, 0): (1, 1),
    (0, -1): (1, -1),
    (-1, 0): (-1, 1),
    (-1, -1): (-1, -1),
}


def _shape_signature(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8), axis=None).tobytes() + bytes([mask.shape[0], mask.shape[1]])


def detect_plus_color(grid: np.ndarray, *, bg: int | None = None) -> int | None:
    """Find the repeated plus color by shape, not by raw area."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=4)
        if int(n) < 2:
            continue

        sigs: list[bytes] = []
        total_area = 0
        ok = True
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                ok = False
                break
            r0, c0, r1, c1 = bbox_coords(pts)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                ok = False
                break
            local = (lab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(cid))
            if bool(local[0, 0]) or bool(local[0, -1]) or bool(local[-1, 0]) or bool(local[-1, -1]):
                ok = False
                break
            if np.all(local):
                ok = False
                break
            if not np.array_equal(local, local[::-1, :]):
                ok = False
                break
            if not np.array_equal(local, local[:, ::-1]):
                ok = False
                break
            sigs.append(_shape_signature(local))
            total_area += int(pts.shape[0])

        if not ok or len(set(sigs)) != 1:
            continue

        key = (int(n), int(total_area))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def _slot_origin(
    r0: int,
    c0: int,
    side: int,
    direction: tuple[int, int],
    slot_idx: int,
) -> tuple[int, int]:
    """Top-left corner of one full diagonal tile slot."""
    dr, dc = map(int, direction)
    return int(r0 + slot_idx * dr * side), int(c0 + slot_idx * dc * side)


def _slot_fully_in_bounds(H: int, W: int, r0: int, c0: int, side: int) -> bool:
    return 0 <= int(r0) and 0 <= int(c0) and int(r0 + side) <= int(H) and int(c0 + side) <= int(W)


def _paint_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    protect = set() if protect is None else set(map(int, protect))
    side = int(tile.shape[0])
    for i in range(side):
        for j in range(side):
            v = int(tile[i, j])
            if v == int(bg):
                continue
            rr = int(r0 + i)
            cc = int(c0 + j)
            if int(out[rr, cc]) in protect:
                continue
            out[rr, cc] = int(v)


def _adjacent_component_ids(mark_lab: np.ndarray, plus_mask: np.ndarray) -> set[int]:
    """Return ids of 4-connected payload components touching one plus."""
    H, W = mark_lab.shape
    ids: set[int] = set()
    for r, c in np.argwhere(plus_mask):
        r = int(r)
        c = int(c)
        for dr, dc in _CARDINALS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                ids.add(int(mark_lab[rr, cc]))
    return ids


def _parse_seed_component(
    grid: np.ndarray,
    mark_lab: np.ndarray,
    cid: int,
    plus_mask: np.ndarray,
) -> dict | None:
    """Validate one adjacent payload as a filled square seed with one corner contact."""
    pts = np.argwhere(mark_lab == int(cid))
    if pts.size == 0:
        return None

    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 3:
        return None
    if int(pts.shape[0]) != int(h * w):
        return None

    touch_cells: set[tuple[int, int]] = set()
    H, W = plus_mask.shape
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in _CARDINALS:
            rr = int(pr + dr)
            cc = int(pc + dc)
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                touch_cells.add((int(pr), int(pc)))
                break
    if len(touch_cells) != 1:
        return None

    tr, tc = next(iter(touch_cells))
    local = (int(tr - r0), int(tc - c0))
    if local == (0, 0):
        direction = (1, 1)
    elif local == (0, int(w - 1)):
        direction = (1, -1)
    elif local == (int(h - 1), 0):
        direction = (-1, 1)
    elif local == (int(h - 1), int(w - 1)):
        direction = (-1, -1)
    else:
        return None

    n_plus_neighbors = 0
    for dr, dc in _CARDINALS:
        rr = int(tr + dr)
        cc = int(tc + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
            n_plus_neighbors += 1
    if int(n_plus_neighbors) < 1:
        return None

    tile = crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))
    return {
        "tile": tile,
        "r0": int(r0),
        "c0": int(c0),
        "side": int(h),
        "direction": (int(direction[0]), int(direction[1])),
    }


def _inspect_ray_pattern(
    grid: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    obstacle_col: int,
) -> dict | None:
    """Read one diagonal lane slot-by-slot."""
    H, W = grid.shape
    side = int(tile.shape[0])
    occupied_slots = [0]
    stop_slot: int | None = None
    slot_idx = 1

    while True:
        sr, sc = _slot_origin(int(r0), int(c0), side, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, side):
            break
        patch = grid[int(sr) : int(sr + side), int(sc) : int(sc + side)]
        if np.array_equal(patch, tile):
            occupied_slots.append(int(slot_idx))
        elif np.all(patch == int(bg)):
            pass
        elif np.all(patch == int(obstacle_col)):
            stop_slot = int(slot_idx)
            break
        else:
            return None
        slot_idx += 1

    if stop_slot is None:
        return None

    inferred_gap: int | None = None
    if len(occupied_slots) >= 3:
        diffs = [int(b - a) for a, b in zip(occupied_slots, occupied_slots[1:])]
        if len(set(diffs)) == 1 and int(diffs[0]) >= 2 and int(stop_slot) == int(occupied_slots[-1] + diffs[0]):
            inferred_gap = int(diffs[0] - 1)

    return {
        "occupied_slots": occupied_slots,
        "visible_count": int(len(occupied_slots)),
        "stop_slot": int(stop_slot),
        "inferred_gap": inferred_gap,
    }


def _collect_markers(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    *,
    bg: int,
    plus_col: int,
    obstacle_col: int,
) -> tuple[np.ndarray, list[dict]] | None:
    """Remove one candidate obstacle color and parse one seed for every plus."""
    non_mask = (grid != int(bg)) & (grid != int(plus_col)) & (grid != int(obstacle_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    markers: list[dict] = []
    for pk in range(1, int(plus_lab.max()) + 1):
        plus_mask = plus_lab == int(pk)
        adj_ids = _adjacent_component_ids(mark_lab, plus_mask)
        if len(adj_ids) != 1:
            return None
        cid = next(iter(adj_ids))
        parsed = _parse_seed_component(grid, mark_lab, int(cid), plus_mask)
        if parsed is None:
            return None
        markers.append(parsed)
    return mark_lab, markers


def _validate_candidate_gap(
    grid: np.ndarray,
    markers: list[dict],
    *,
    bg: int,
    obstacle_col: int,
    skip_gap: int,
) -> bool:
    """Check that every lane is compatible with one shared skip gap."""
    H, W = grid.shape
    step_slots = int(skip_gap) + 1
    for m in markers:
        side = int(m["side"])
        stop_slot = int(m["pattern"]["stop_slot"])
        if stop_slot < int(step_slots) or stop_slot % int(step_slots) != 0:
            return False
        for slot_idx in range(1, int(stop_slot)):
            sr, sc = _slot_origin(int(m["r0"]), int(m["c0"]), side, m["direction"], int(slot_idx))
            if not _slot_fully_in_bounds(H, W, sr, sc, side):
                return False
            patch = grid[int(sr) : int(sr + side), int(sc) : int(sc + side)]
            if int(slot_idx) % int(step_slots) == 0:
                if not (np.array_equal(patch, m["tile"]) or np.all(patch == int(bg))):
                    return False
            else:
                if not np.all(patch == int(bg)):
                    return False
        sr, sc = _slot_origin(int(m["r0"]), int(m["c0"]), side, m["direction"], int(stop_slot))
        patch = grid[int(sr) : int(sr + side), int(sc) : int(sc + side)]
        if not np.all(patch == int(obstacle_col)):
            return False
    return True


def _infer_obstacle_color_and_skip_gap(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> tuple[int, int, list[dict]] | None:
    """Recover the shared blocker color and dashed spacing from the clue ray."""
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(plus_col)})
    best_score: tuple[int, int] | None = None
    best: tuple[int, int, list[dict]] | None = None

    for obstacle_col in colors:
        collected = _collect_markers(
            grid,
            plus_lab,
            bg=int(bg),
            plus_col=int(plus_col),
            obstacle_col=int(obstacle_col),
        )
        if collected is None:
            continue
        _mark_lab, markers = collected

        demo_gap: int | None = None
        demo_visible = -1
        valid = True
        for m in markers:
            pattern = _inspect_ray_pattern(
                grid,
                m["tile"],
                int(m["r0"]),
                int(m["c0"]),
                m["direction"],
                bg=int(bg),
                obstacle_col=int(obstacle_col),
            )
            if pattern is None:
                valid = False
                break
            m["pattern"] = pattern
            if pattern["inferred_gap"] is not None and int(pattern["visible_count"]) > int(demo_visible):
                demo_gap = int(pattern["inferred_gap"])
                demo_visible = int(pattern["visible_count"])
        if not valid or demo_gap is None:
            continue
        if not _validate_candidate_gap(
            grid,
            markers,
            bg=int(bg),
            obstacle_col=int(obstacle_col),
            skip_gap=int(demo_gap),
        ):
            continue

        score = (int(demo_visible), int(len(markers)))
        if best_score is None or score > best_score:
            best_score = score
            best = (int(obstacle_col), int(demo_gap), markers)

    return best


def extend_diagonal_tile_until_obstacle(
    out: np.ndarray,
    base: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    skip_gap: int,
    bg: int,
    obstacle_col: int,
    protect: set[int] | None = None,
) -> None:
    """Repeat a square tile on diagonal slots, skipping regular gaps and halting at the blocker."""
    H, W = out.shape
    side = int(tile.shape[0])
    slot_idx = int(skip_gap) + 1
    step_slots = int(skip_gap) + 1
    while True:
        sr, sc = _slot_origin(int(r0), int(c0), side, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, side):
            break
        patch = base[int(sr) : int(sr + side), int(sc) : int(sc + side)]
        if np.all(patch == int(obstacle_col)):
            break
        _paint_tile(out, tile, sr, sc, bg=int(bg), protect=protect)
        slot_idx += int(step_slots)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend large corner-attached marker tiles along dashed diagonal rays until a blocker square.

    Input structure:
    - The background is the most common color.
    - One repeated non-background color forms several identical plus-like anchors with empty bbox corners.
    - Each plus has exactly one filled multicolor square seed occupying one bbox corner outside the plus mass.
    - One seed is already shown as a longer dashed diagonal ray, and one dedicated color appears only in solid obstacle squares farther along the rays.

    Rule:
    - Detect the plus color by its repeated symmetric shape, then locate one adjacent square seed for each plus.
    - Use which corner of the seed touches the plus to determine the outward diagonal direction.
    - Read the long demonstrated ray to infer the shared skip spacing in whole square-tile slots.
    - Repaint every seed on that diagonal, drawing only every visible slot and leaving skipped slots blank, and stop when the next visible slot is the solid obstacle square.

    Key insight:
    - The clue ray jointly encodes four things at once: corner-based diagonal direction, whole-square period, global dashed spacing, and the fact that obstacle squares terminate the propagation before the boundary.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_plus_color(g, bg=int(bg))
    if plus_col is None:
        return out

    plus_lab, _ = label_mask(g == int(plus_col), connectivity=4)
    inferred = _infer_obstacle_color_and_skip_gap(
        g,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_col),
    )
    if inferred is None:
        return out
    obstacle_col, skip_gap, markers = inferred

    for m in markers:
        extend_diagonal_tile_until_obstacle(
            out,
            g,
            m["tile"],
            int(m["r0"]),
            int(m["c0"]),
            m["direction"],
            skip_gap=int(skip_gap),
            bg=int(bg),
            obstacle_col=int(obstacle_col),
            protect={int(plus_col), int(obstacle_col)},
        )

    return out


# --------------- generator (output-first combo variant) ---------------

_H_RANGE = (28, 39)
_W_RANGE = (28, 39)
_N_PLUS_RANGE = (2, 4)
_TILE_SIDE_CHOICES = [3]
_SKIP_GAP_CHOICES = [1, 2]
_SEED_COLOR_COUNT_CHOICES = [2, 3]
_MIN_VISIBLE_COPIES = 2
_DEMO_MIN_VISIBLE = 3
_PLACEMENT_TRIES = 220
_TARGETED_RETRIES = 320
_GEN_RETRIES = 700

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
]


def _tile_has_rich_texture(tile: np.ndarray) -> bool:
    if len(set(map(int, tile.flatten()))) < 2:
        return False
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False
    if np.array_equal(tile, tile[::-1, ::-1]):
        return False
    return True


def _build_seed_tile(side: int, palette: list[int], rng: np.random.Generator) -> np.ndarray | None:
    """Build one large square tile that uses the full shared seed palette."""
    for _ in range(40):
        vals = list(map(int, palette))
        vals += list(map(int, rng.choice(palette, size=side * side - len(palette), replace=True)))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(side, side)
        if all(int(col) in set(map(int, tile.flatten())) for col in palette) and _tile_has_rich_texture(tile):
            return tile
    return None


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    side: int,
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, -1):
        return int(pr0 - side + 1), int(pc0 - side + 1)
    if (dr, dc) == (-1, 1):
        return int(pr0 - side + 1), int(pc1)
    if (dr, dc) == (1, -1):
        return int(pr1), int(pc0 - side + 1)
    if (dr, dc) == (1, 1):
        return int(pr1), int(pc1)
    raise ValueError("direction must be diagonal")


def _full_slot_count(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    side: int,
    direction: tuple[int, int],
) -> int:
    count = 0
    slot_idx = 0
    while True:
        sr, sc = _slot_origin(int(seed_r0), int(seed_c0), side, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, side):
            break
        count += 1
        slot_idx += 1
    return int(count)


def _slot_cells(seed_r0: int, seed_c0: int, side: int, direction: tuple[int, int], slot_idx: int) -> list[tuple[int, int]]:
    sr, sc = _slot_origin(int(seed_r0), int(seed_c0), side, direction, int(slot_idx))
    return [(int(sr + i), int(sc + j)) for i in range(side) for j in range(side)]


def _plus_cells_from_shape(r0: int, c0: int, plus_shape: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = plus_shape.shape
    return [
        (int(r0 + i), int(c0 + j))
        for i in range(ph)
        for j in range(pw)
        if int(plus_shape[i, j]) == 1
    ]


def _plus_touch_cells(plus_cells: list[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    touched: set[tuple[int, int]] = set()
    for r, c in plus_cells:
        for rr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= rr < H and 0 <= cc < W:
                touched.add((int(rr), int(cc)))
    return touched


def _mark_buffer(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    H, W = mask.shape
    for r, c in cells:
        for rr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= rr < H and 0 <= cc < W:
                mask[int(rr), int(cc)] = 1


def _ray_layout(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile: np.ndarray,
    skip_gap: int,
    visible_count: int,
    obstacle_color: int,
) -> dict | None:
    """Build all reserved, painted, and obstacle cells for one diagonal dashed ray."""
    side = int(tile.shape[0])
    seed_r0, seed_c0 = _seed_origin(plus_bbox, direction, side)
    step_slots = int(skip_gap) + 1
    stop_slot = int(visible_count) * int(step_slots)

    lane_cells: set[tuple[int, int]] = set()
    tile_cells: list[tuple[int, int, int]] = []
    obstacle_cells: list[tuple[int, int, int]] = []

    for slot_idx in range(0, int(stop_slot) + 1):
        sr, sc = _slot_origin(int(seed_r0), int(seed_c0), side, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, side):
            return None
        for i in range(side):
            for j in range(side):
                rr = int(sr + i)
                cc = int(sc + j)
                lane_cells.add((int(rr), int(cc)))
                if int(slot_idx) == int(stop_slot):
                    obstacle_cells.append((int(rr), int(cc), int(obstacle_color)))
                elif int(slot_idx) % int(step_slots) == 0:
                    tile_cells.append((int(rr), int(cc), int(tile[i, j])))

    return {
        "seed_r0": int(seed_r0),
        "seed_c0": int(seed_c0),
        "side": int(side),
        "stop_slot": int(stop_slot),
        "lane_cells": sorted(lane_cells),
        "tile_cells": tile_cells,
        "obstacle_cells": obstacle_cells,
        "visible_count": int(visible_count),
        "direction": (int(direction[0]), int(direction[1])),
    }


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    tile_palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_buffer: np.ndarray,
    solid_buffer: np.ndarray,
    out: np.ndarray,
    *,
    plus_color: int,
    obstacle_color: int,
    skip_gap: int,
    min_visible_count: int,
    prefer_visible_count: int,
) -> dict | None:
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph - 1))
        c0 = int(rng.integers(1, W - pw - 1))
        plus_cells = _plus_cells_from_shape(int(r0), int(c0), plus_shape)
        plus_touch = _plus_touch_cells(plus_cells, H, W)

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(occ[r, c]) != 0 for r, c in plus_touch):
            continue
        if any(int(solid_buffer[r, c]) != 0 for r, c in plus_touch):
            continue

        side = int(rng.choice(_TILE_SIDE_CHOICES))
        tile = _build_seed_tile(side, tile_palette, rng)
        if tile is None:
            continue

        dirs = list(_DIAGONALS)
        rng.shuffle(dirs)
        plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        for direction in dirs:
            seed_r0, seed_c0 = _seed_origin(plus_bbox, direction, int(side))
            full_slots = _full_slot_count(H, W, seed_r0, seed_c0, int(side), direction)
            if int(full_slots) <= int(min_visible_count * (skip_gap + 1)):
                continue

            candidates = list(range(int(min_visible_count), max(int(min_visible_count), full_slots // (skip_gap + 1)) + 1))
            candidates = [v for v in candidates if int(v * (skip_gap + 1)) < int(full_slots)]
            candidates.sort(key=lambda v: (abs(int(v) - int(prefer_visible_count)), int(v)))
            for visible_count in candidates:
                layout = _ray_layout(
                    H,
                    W,
                    plus_bbox,
                    direction,
                    tile,
                    int(skip_gap),
                    int(visible_count),
                    int(obstacle_color),
                )
                if layout is None:
                    continue
                if any(int(occ[r, c]) != 0 for r, c in layout["lane_cells"]):
                    continue
                if any(int(plus_buffer[r, c]) != 0 for r, c in layout["lane_cells"]):
                    continue
                solid_cells = [(int(r), int(c)) for r, c, _ in layout["tile_cells"] + layout["obstacle_cells"]]
                if any(int(solid_buffer[r, c]) != 0 for r, c in solid_cells):
                    continue

                for r, c in plus_cells:
                    occ[int(r), int(c)] = 1
                    out[int(r), int(c)] = int(plus_color)
                for r, c in plus_touch:
                    plus_buffer[int(r), int(c)] = 1
                for r, c in layout["lane_cells"]:
                    occ[int(r), int(c)] = 1
                for r, c, v in layout["tile_cells"]:
                    out[int(r), int(c)] = int(v)
                for r, c, v in layout["obstacle_cells"]:
                    out[int(r), int(c)] = int(v)
                _mark_buffer(solid_buffer, solid_cells)

                return {
                    "tile": tile,
                    "seed_r0": int(layout["seed_r0"]),
                    "seed_c0": int(layout["seed_c0"]),
                    "side": int(layout["side"]),
                    "direction": layout["direction"],
                    "lane_cells": layout["lane_cells"],
                    "tile_cells": layout["tile_cells"],
                    "obstacle_cells": layout["obstacle_cells"],
                    "visible_count": int(layout["visible_count"]),
                    "stop_slot": int(layout["stop_slot"]),
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one full demo ray, collapse the others back to their seed square only."""
    candidates = [i for i, p in enumerate(pluses) if int(p["visible_count"]) >= _DEMO_MIN_VISIBLE]
    if not candidates:
        raise ValueError("no sufficiently long demonstration ray")

    best_visible = max(int(pluses[i]["visible_count"]) for i in candidates)
    demo_choices = [i for i in candidates if int(pluses[i]["visible_count"]) == int(best_visible)]
    pre_idx = int(rng.choice(demo_choices))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        seed_set = {
            (int(p["seed_r0"] + rr), int(p["seed_c0"] + cc))
            for rr in range(int(p["side"]))
            for cc in range(int(p["side"]))
        }
        for r, c, _ in p["tile_cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(
    rng: np.random.Generator,
    *,
    return_meta: bool = False,
    force_skip_gap: int | None = None,
) -> np.ndarray | dict | None:
    if force_skip_gap is None:
        skip_gap = int(rng.choice(_SKIP_GAP_CHOICES))
    else:
        skip_gap = int(force_skip_gap)
    if int(skip_gap) == 2:
        H = int(rng.integers(max(_H_RANGE[0], 35), _H_RANGE[1]))
        W = int(rng.integers(max(_W_RANGE[0], 35), _W_RANGE[1]))
    else:
        H = int(rng.integers(*_H_RANGE))
        W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != int(bg)]
    plus_color, obstacle_color = map(int, rng.choice(others, size=2, replace=False))
    seed_color_count = int(rng.choice(_SEED_COLOR_COUNT_CHOICES))
    seed_palette = list(map(int, rng.choice([c for c in others if c not in {plus_color, obstacle_color}], size=seed_color_count, replace=False)))

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = 2 if int(skip_gap) == 2 else int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_buffer = np.zeros((H, W), dtype=int)
    solid_buffer = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    demo_placed = False
    for plus_idx in range(n_plus):
        min_visible_count = _DEMO_MIN_VISIBLE if not demo_placed and int(plus_idx) == 0 else _MIN_VISIBLE_COPIES
        prefer_visible_count = _DEMO_MIN_VISIBLE if not demo_placed and int(plus_idx) == 0 else _MIN_VISIBLE_COPIES
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            seed_palette,
            H,
            W,
            occ,
            plus_buffer,
            solid_buffer,
            out,
            plus_color=int(plus_color),
            obstacle_color=int(obstacle_color),
            skip_gap=int(skip_gap),
            min_visible_count=int(min_visible_count),
            prefer_visible_count=int(prefer_visible_count),
        )
        if info is None:
            return None
        pluses.append(info)
        if int(info["visible_count"]) >= _DEMO_MIN_VISIBLE:
            demo_placed = True

    if not demo_placed:
        return None

    try:
        inp = _build_input_from_output(out, pluses, int(bg), rng)
    except ValueError:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    if return_meta:
        return {
            "input": inp,
            "output": out,
            "bg": int(bg),
            "plus_color": int(plus_color),
            "obstacle_color": int(obstacle_color),
            "skip_gap": int(skip_gap),
            "seed_palette": seed_palette,
            "pluses": pluses,
        }
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    target_gap = int(rng.choice(_SKIP_GAP_CHOICES))
    for _ in range(_TARGETED_RETRIES):
        g = _gen_attempt(rng, force_skip_gap=int(target_gap))
        if g is not None:
            return g
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    total_tries = int(_TARGETED_RETRIES + _GEN_RETRIES)
    raise RuntimeError(f"generate_input: failed after {total_tries} attempts")
