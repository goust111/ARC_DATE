# Combo variant of 53fb4810 (index 2)
# combo_spec: D1/0+D6/0
# Name: 对角线遇障停止

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask


# Design analysis for the D1/0 + D6/0 combo
# ------------------------------------------
# a) Concrete behavior:
#    The original task attached one small seed tile to each '+' and extended that
#    tile axially until the boundary. This combo changes two linked stages at
#    once:
#    - D1: the path is now diagonal, so each copy is shifted by (±1, ±1).
#    - D6: the ray no longer ends at the frame; it stops when the next diagonal
#      copy would collide with a dedicated obstacle color.
#    A workable concrete geometry is to use the empty corner notches inside the
#    '+' bounding box. Each seed is a 2-cell domino that includes exactly one
#    notch cell, so the occupied notch identifies one of four diagonals:
#      NW notch -> ray goes up-left
#      NE notch -> ray goes up-right
#      SW notch -> ray goes down-left
#      SE notch -> ray goes down-right
#    Example: if a seed uses the NE notch, the output shows the same 2-cell
#    domino repeated on a staircase moving (-1,+1), and a monochrome blocker
#    domino sits exactly where the next copy would land.
# b) Complexity calibration:
#    Restricting seeds to 2-cell dominos is deliberate. A 1x1 seed would turn
#    the clue ray into sparse diagonal dots, while a 2x2 seed would self-overlap
#    under unit diagonal shifts and create color-consistency ambiguity. Dominos
#    keep the diagonal repetition visually obvious and still allow both 1-color
#    and 2-color payloads.
# c) Hidden assumptions from the original main():
#    - The old direction parser assumed the marker was edge-attached on one of
#      four axial sides. That is false here; direction must be inferred from the
#      occupied corner notch instead.
#    - The old bbox-crop logic was harmless for a long axial clue because a
#      boundary-complete ray needed no further growth. Here a full diagonal clue
#      has a long staircase bbox, so the solver must reconstruct the local seed
#      from the notch neighborhood rather than cropping the whole component.
#    - The old write loop assumed future cells were either bg or protected '+'.
#      After D6, obstacle pixels are intentional foreign occupancies, so the loop
#      must stop when the next translated copy hits any non-matching non-bg cell.
# d) Placement legality:
#    The generator stays output-first. It places one '+' at a time together with
#    a diagonal domino ray and a blocker domino. Exact overlap is forbidden, and
#    a 4-neighbor guard band is also enforced between payloads so component
#    labeling never merges two different rays.
# e) Visual observability:
#    In each sample, one ray is already expanded as a long diagonal staircase and
#    terminates flush against the shared obstacle color. The remaining pluses show
#    only the corner seed plus the same kind of blocker farther out, which makes
#    both "use the corner to choose a diagonal" and "stop at the blocker" visible
#    from a few examples.
# f) Visual cleanliness:
#    Only one seed touches each '+', foreign payloads are kept away from all plus
#    neighborhoods, and distinct rays are separated by a moat. This avoids merged
#    components and keeps the clue/example ray readable.
# g) Color inferability:
#    There is no color remapping. Seed colors repeat unchanged; the obstacle uses
#    one globally shared color excluded from every seed palette, so humans can
#    treat it as a structural blocker rather than a second payload source.
# h) New solver assumptions audited against the generator:
#    - Every plus shape has empty bbox corners; the provided templates satisfy
#      that and the generator never swaps in other shapes.
#    - Exactly one corner notch belongs to the adjacent payload component of each
#      plus, and exactly one outward orthogonal neighbor of that notch is occupied.
#      The generator enforces this by using only 2-cell dominos anchored at the
#      notch and by preventing unrelated payloads from entering plus neighborhoods.
#    - The obstacle occupies the next translated domino position, so "blocked"
#      really differs from "boundary reached". The generator records a positive
#      slack `full_units - stop_units` for every ray.


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the non-bg color with the strongest population of large components."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            if int(r1 - r0 + 1) >= int(min_h) and int(c1 - c0 + 1) >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_DIR_TO_NAME = {
    (-1, -1): "NW",
    (-1, 1): "NE",
    (1, -1): "SW",
    (1, 1): "SE",
}


def _adjacent_component_ids(mark_lab: np.ndarray, plus_mask: np.ndarray) -> set[int]:
    """Return non-zero payload component ids that touch one plus by 4-neighborhood."""
    H, W = mark_lab.shape
    adj_ids: set[int] = set()
    for r, c in np.argwhere(plus_mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                adj_ids.add(int(mark_lab[rr, cc]))
    return adj_ids


def _corner_specs(plus_mask: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    """Return per-diagonal notch data for one plus component."""
    coords = np.argwhere(plus_mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return {
        (-1, -1): {
            "notch": (int(r0), int(c0)),
            "seconds": [(int(r0 - 1), int(c0)), (int(r0), int(c0 - 1))],
        },
        (-1, 1): {
            "notch": (int(r0), int(c1)),
            "seconds": [(int(r0 - 1), int(c1)), (int(r0), int(c1 + 1))],
        },
        (1, -1): {
            "notch": (int(r1), int(c0)),
            "seconds": [(int(r1 + 1), int(c0)), (int(r1), int(c0 - 1))],
        },
        (1, 1): {
            "notch": (int(r1), int(c1)),
            "seconds": [(int(r1 + 1), int(c1)), (int(r1), int(c1 + 1))],
        },
    }


def _parse_marker_seed(
    grid: np.ndarray,
    pts: np.ndarray,
    plus_mask: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> tuple[tuple[int, int], int, int, np.ndarray] | None:
    """
    Recover the diagonal direction and the local 2-cell seed domino.

    The adjacent payload must contain exactly one corner-notch cell of the plus
    bbox and exactly one of that notch's two outward orthogonal neighbors.
    """
    if pts.size == 0:
        return None
    H, W = grid.shape
    pt_set = {(int(r), int(c)) for r, c in pts}
    specs = _corner_specs(plus_mask)

    matching_dirs = [d for d, spec in specs.items() if spec["notch"] in pt_set]
    if len(matching_dirs) != 1:
        return None
    direction = matching_dirs[0]
    notch = specs[direction]["notch"]
    second_hits = [
        (int(r), int(c))
        for r, c in specs[direction]["seconds"]
        if 0 <= int(r) < H and 0 <= int(c) < W and (int(r), int(c)) in pt_set
    ]
    if len(second_hits) != 1:
        return None

    nr, nc = notch
    sr, sc = second_hits[0]
    r0 = min(int(nr), int(sr))
    c0 = min(int(nc), int(sc))
    r1 = max(int(nr), int(sr))
    c1 = max(int(nc), int(sc))
    tile = np.array(grid[int(r0) : int(r1 + 1), int(c0) : int(c1 + 1)], dtype=int)
    if tile.shape not in {(2, 1), (1, 2)}:
        return None
    if np.any(tile == int(bg)) or np.any(tile == int(plus_col)):
        return None
    return (int(direction[0]), int(direction[1])), int(r0), int(c0), tile


def _infer_obstacle_color_and_mark_lab(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> tuple[int | None, np.ndarray]:
    """
    Recover the blocker color by removing one candidate color at a time and
    asking whether every plus then has exactly one valid adjacent payload.
    """
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(plus_col)})
    n_plus = int(plus_lab.max())

    for obstacle_col in colors:
        non_mask = (grid != int(bg)) & (grid != int(plus_col)) & (grid != int(obstacle_col))
        mark_lab, _ = label_mask(non_mask, connectivity=4)
        ok = True
        for pk in range(1, n_plus + 1):
            plus_mask = plus_lab == int(pk)
            adj_ids = _adjacent_component_ids(mark_lab, plus_mask)
            if len(adj_ids) != 1:
                ok = False
                break
            cid = next(iter(adj_ids))
            pts = np.argwhere(mark_lab == int(cid))
            parsed = _parse_marker_seed(
                grid,
                pts,
                plus_mask,
                bg=int(bg),
                plus_col=int(plus_col),
            )
            if parsed is None:
                ok = False
                break
        if ok:
            return int(obstacle_col), mark_lab

    non_mask = (grid != int(bg)) & (grid != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)
    return None, mark_lab


def extend_repeating_tile_diagonally_until_blocked(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    """Repeat a 2-cell seed by unit diagonal translations until the next copy is blocked."""
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    step = 1

    while True:
        rr0 = int(r0 + step * dr)
        cc0 = int(c0 + step * dc)
        cells: list[tuple[int, int, int]] = []
        for i in range(h):
            for j in range(w):
                v = int(tile[i, j])
                if v == int(bg):
                    continue
                rr = int(rr0 + i)
                cc = int(cc0 + j)
                if not (0 <= rr < H and 0 <= cc < W):
                    return
                cur = int(out[rr, cc])
                if cur not in {int(bg), int(v)}:
                    return
                cells.append((rr, cc, v))
        for rr, cc, v in cells:
            out[rr, cc] = int(v)
        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend corner-anchored marker dominos along diagonal rays, stopping at blockers.

    Input structure:
    - The background is the dominant color.
    - One dominant non-bg color forms several identical '+' anchors whose bbox
      corners remain empty.
    - Each plus has exactly one small 2-cell seed touching one empty bbox corner,
      and one extra color appears only in detached obstacle dominos farther out on
      the corresponding diagonals.

    Rule:
    - For each payload touching a '+', use the occupied bbox corner notch to infer
      one of the four diagonal directions.
    - Reconstruct the local 2-cell seed from that notch and its single outward
      orthogonal neighbor, then copy the seed by repeated unit diagonal shifts.
    - Stop as soon as the next translated copy would hit a foreign non-bg cell,
      which is the visible obstacle in the generated tasks.

    Key insight: The seed's occupied corner notch determines the diagonal, while
    the obstacle color provides the endpoint; the long clue ray is only evidence
    for humans, not a template that needs to be cropped wholesale.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    _obstacle_col, mark_lab = _infer_obstacle_color_and_mark_lab(
        g,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_col),
    )

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        adj_ids = _adjacent_component_ids(mark_lab, plus_mask)
        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            parsed = _parse_marker_seed(
                g,
                pts,
                plus_mask,
                bg=int(bg),
                plus_col=int(plus_col),
            )
            if parsed is None:
                continue
            direction, seed_r0, seed_c0, tile = parsed
            extend_repeating_tile_diagonally_until_blocked(
                out,
                tile,
                int(seed_r0),
                int(seed_c0),
                direction,
                bg=int(bg),
            )

    return out


# --------------- generator (output-first, diagonal + collision-stop) ---------------

_H_RANGE = (15, 30)
_W_RANGE = (15, 25)
_N_PLUS_RANGE = (2, 4)
_SEED_MAX_COLORS = 2
_MIN_STOP_UNITS = 3
_PRE_EXT_IDEAL_UNITS = 6
_PLACEMENT_TRIES = 180
_GEN_RETRIES = 600

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
        dtype=int,
    ),
]

_SEED_ORIENTS = ["vertical", "horizontal"]


def _corner_notch_from_bbox(
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = plus_bbox
    if tuple(map(int, direction)) == (-1, -1):
        return int(pr0), int(pc0)
    if tuple(map(int, direction)) == (-1, 1):
        return int(pr0), int(pc1)
    if tuple(map(int, direction)) == (1, -1):
        return int(pr1), int(pc0)
    if tuple(map(int, direction)) == (1, 1):
        return int(pr1), int(pc1)
    raise ValueError("direction must be diagonal")


def _build_seed_tile(
    palette: list[int],
    orient: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a 2-cell domino with 1-2 colors."""
    shape = (2, 1) if orient == "vertical" else (1, 2)
    n_cells = int(shape[0] * shape[1])
    n_colors = int(rng.integers(1, min(_SEED_MAX_COLORS, len(palette), n_cells) + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = chosen + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(shape)


def _seed_origin_from_notch(
    notch: tuple[int, int],
    direction: tuple[int, int],
    orient: str,
) -> tuple[int, int]:
    nr, nc = map(int, notch)
    dr, dc = map(int, direction)
    if orient == "vertical":
        return (int(nr - 1) if dr == -1 else int(nr), int(nc))
    return (int(nr), int(nc - 1) if dc == -1 else int(nc))


def _project_tile_coords_for_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    units: int,
) -> list[tuple[int, int, int]]:
    """Cells for the first `units` diagonal copies, including the seed at k=0."""
    dr, dc = map(int, direction)
    h, w = tile.shape
    cells: list[tuple[int, int, int]] = []
    for k in range(int(units)):
        rr0 = int(seed_r0 + k * dr)
        cc0 = int(seed_c0 + k * dc)
        for i in range(h):
            for j in range(w):
                cells.append((int(rr0 + i), int(cc0 + j), int(tile[i, j])))
    return cells


def _obstacle_cells_at_offset(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    offset: int,
    obstacle_color: int,
) -> list[tuple[int, int, int]]:
    """A monochrome blocker domino occupying the next translated-copy position."""
    dr, dc = map(int, direction)
    h, w = tile.shape
    rr0 = int(seed_r0 + offset * dr)
    cc0 = int(seed_c0 + offset * dc)
    return [
        (int(rr0 + i), int(cc0 + j), int(obstacle_color))
        for i in range(h)
        for j in range(w)
    ]


def _all_in_bounds(H: int, W: int, cells: list[tuple[int, int, int]]) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c, _ in cells)


def _max_units_before_boundary(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> int:
    """Maximum number of full diagonal copies that fit before touching the frame."""
    dr, dc = map(int, direction)
    h, w = tile.shape
    row_room = (H - (int(seed_r0) + h)) if dr == 1 else int(seed_r0)
    col_room = (W - (int(seed_c0) + w)) if dc == 1 else int(seed_c0)
    return 1 + int(min(row_room, col_room))


def _mark_guard(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Mark each cell and its 4-neighborhood in a guard mask."""
    H, W = mask.shape
    for r, c in cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= int(rr) < H and 0 <= int(cc) < W:
                mask[int(rr), int(cc)] = True


def _place_plus_ray_and_obstacle(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_guard: np.ndarray,
    payload_guard: np.ndarray,
    out: np.ndarray,
    *,
    plus_color: int,
    obstacle_color: int,
    direction_choices: list[tuple[int, int]] | None = None,
) -> dict | None:
    """Place one plus, one diagonal ray, and one blocker, or return None."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]
        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(bool(plus_guard[r, c]) or bool(payload_guard[r, c]) for r, c in plus_cells):
            continue

        plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        dirs = list(_DIAGONALS if direction_choices is None else direction_choices)
        rng.shuffle(dirs)
        for direction in dirs:
            orient_choices = list(_SEED_ORIENTS)
            rng.shuffle(orient_choices)
            notch = _corner_notch_from_bbox(plus_bbox, direction)
            for orient in orient_choices:
                tile = _build_seed_tile(palette, orient, rng)
                seed_r0, seed_c0 = _seed_origin_from_notch(notch, direction, orient)
                full_units = _max_units_before_boundary(H, W, seed_r0, seed_c0, direction, tile)
                if full_units - 1 < _MIN_STOP_UNITS:
                    continue

                candidates = list(range(_MIN_STOP_UNITS, full_units))
                rng.shuffle(candidates)
                for stop_units in candidates:
                    ray_cells = _project_tile_coords_for_units(seed_r0, seed_c0, direction, tile, stop_units)
                    obstacle_cells = _obstacle_cells_at_offset(
                        seed_r0,
                        seed_c0,
                        direction,
                        tile,
                        stop_units,
                        int(obstacle_color),
                    )
                    if not _all_in_bounds(H, W, ray_cells) or not _all_in_bounds(H, W, obstacle_cells):
                        continue
                    payload_cells = [(int(r), int(c)) for r, c, _ in ray_cells + obstacle_cells]
                    if any(int(occ[r, c]) != 0 for r, c in payload_cells):
                        continue
                    if any(bool(plus_guard[r, c]) or bool(payload_guard[r, c]) for r, c in payload_cells):
                        continue

                    for r, c in plus_cells:
                        occ[r, c] = 1
                        out[r, c] = int(plus_color)
                    for r, c, v in ray_cells:
                        occ[r, c] = 2
                        out[r, c] = int(v)
                    for r, c, v in obstacle_cells:
                        occ[r, c] = 3
                        out[r, c] = int(v)

                    _mark_guard(plus_guard, plus_cells)
                    _mark_guard(payload_guard, payload_cells)

                    return {
                        "tile": tile,
                        "cells": ray_cells,
                        "seed_r0": int(seed_r0),
                        "seed_c0": int(seed_c0),
                        "direction": (int(direction[0]), int(direction[1])),
                        "direction_name": _DIR_TO_NAME[(int(direction[0]), int(direction[1]))],
                        "orient": orient,
                        "units": int(stop_units),
                        "full_units": int(full_units),
                        "obstacle_cells": obstacle_cells,
                    }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Keep one full clue ray and erase the rest back down to their seed only."""
    pre_idx = max(range(len(pluses)), key=lambda i: pluses[i]["units"])
    if pluses[pre_idx]["units"] < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, p in enumerate(pluses) if int(p["units"]) >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        seed_set = {
            (int(r), int(c))
            for r, c, _ in _project_tile_coords_for_units(
                int(p["seed_r0"]),
                int(p["seed_c0"]),
                tuple(p["direction"]),
                np.array(p["tile"], dtype=int),
                1,
            )
        }
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp, int(pre_idx)


def _gen_attempt(
    rng: np.random.Generator,
    *,
    return_meta: bool = False,
) -> np.ndarray | dict | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color, obstacle_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {plus_color, obstacle_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))
    direction_plan = list(_DIAGONALS)
    rng.shuffle(direction_plan)
    direction_plan = direction_plan[:N]

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_guard = np.zeros((H, W), dtype=bool)
    payload_guard = np.zeros((H, W), dtype=bool)

    pluses: list[dict] = []
    for idx in range(N):
        info = _place_plus_ray_and_obstacle(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_guard,
            payload_guard,
            out,
            plus_color=int(plus_color),
            obstacle_color=int(obstacle_color),
            direction_choices=[direction_plan[idx]],
        )
        if info is None:
            return None
        pluses.append(info)

    if max(int(p["units"]) for p in pluses) < _PRE_EXT_IDEAL_UNITS:
        return None

    inp, pre_idx = _build_input_from_output(out, pluses, bg, rng)

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
            "pluses": pluses,
            "clue_index": int(pre_idx),
        }
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
