# Combo variant of 53fb4810 (index 11)
# combo_spec: D6/1+D18/0
# Name: 边缘编码距离射线

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo D6/1 + D18/0
# a) Concrete behavior and trace:
#    - Keep the original object vocabulary: one shared '+' color, one attached payload per plus, and one
#      already-finished ray preserved in the input as a clue.
#    - D18 changes the direction source: the payload's touching side is no longer the ray direction.
#      Instead, a thin border strip in one dedicated control color says up/down/left/right by which edge it
#      occupies. The strip is matched to a payload by exact row-span or column-span agreement.
#    - D6 then changes the stopping rule: once the direction is known, the ray occupies exactly K tile-units,
#      where K is the number of distinct visible non-bg colors inside that payload tile. Example: a 2x2 seed
#      attached on the right of a plus, with a left-edge strip on the same rows and colors {2,4,7}, must fire
#      left for exactly 3 copies. The original task would have fired right and continued to the border.
# b) Complexity calibration:
#    - Direction still has only four categories, and distance has only three practical values here: 2, 3, 4.
#      That is enough variety to show the fusion without creating a large lookup burden.
#    - The generator freezes payloads to 2x2 tiles and uses only plus shapes with nozzle width 2. This keeps
#      the color-count distance code multi-valued and makes a finished clue ray visibly longer than any compact
#      seed, which the solver needs in order to skip the already-complete example.
# c) Broken assumptions from the original solver:
#    - "Touching side = ray direction" is false; direction must be decoded from border controls instead.
#    - "Extend until boundary" is false; rays stop after an encoded number of whole-tile copies.
#    - "It is harmless to treat the long demonstration ray as another seed" is false; the solver must detect
#      that the preserved example component is already longer than a compact 2x2 seed along its decoded axis
#      and leave it unchanged.
# d) Placement legality:
#    - All pluses stay away from the border so the control color is the unique border-only non-bg color.
#    - Payload/ray components never overlap, never become 4-neighbor-adjacent to other payload/ray components,
#      and never enter another plus's neighborhood; otherwise the boolean component labeling in `main()` would
#      merge objects and break ownership.
#    - Control strips are kept disjoint on the border so the solver sees one component per instruction.
# e) Visual observability:
#    - The two hidden rays in each generated sample have different encoded lengths, so the stop rule cannot be
#      mistaken for a constant. At least one hidden ray also has a direction different from its attachment side,
#      so the border instruction is genuinely necessary for completing the output.
#    - Because the clue ray itself also stops short of the border, the task visibly differs from the original
#      border-filling behavior in both the input and output examples.
# f) Visual cleanliness:
#    - Only one extra symbol family is introduced: thin strips on the border. They never overlap the central
#      scene, and all payloads remain one cell away from the border, so the interior stays easy to parse.
#    - Object spacing is conservative: pluses, payloads, rays, and strips are separated by background gaps to
#      avoid accidental merges or ambiguous ownership.
# g) Inferability of the changed signal:
#    - Distance is encoded by counting distinct hues inside a visible 2x2 payload, not by hidden numeric color
#      ids. Direction is encoded by control-strip position, not by color ordering. Both cues are directly human-
#      observable from a few input/output pairs.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: exactly one border-only control color exists. Guaranteed by reserving one color for strips and
#      forcing every payload ray to stop before the border.
#    - Assumption: each plus has exactly one adjacent non-control component. Guaranteed by giving each plus one
#      payload and forbidding 4-neighbor contact between different payload/ray components.
#    - Assumption: each payload bbox matches exactly one control strip by exact row-span or column-span.
#      Guaranteed by making all payload row spans unique and all payload column spans unique inside a grid.


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Pick the non-bg color with the strongest collection of large components."""
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
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def detect_border_only_control_color(
    grid: np.ndarray,
    *,
    bg: int,
    exclude: set[int] | None = None,
) -> int | None:
    """Pick the unique non-bg color whose every occurrence lies on the border."""
    exclude = set() if exclude is None else set(map(int, exclude))
    H, W = grid.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    candidates: list[int] = []
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)} - exclude):
        pts = np.argwhere(grid == int(col))
        if pts.size == 0:
            continue
        if all(bool(border[int(r), int(c)]) for r, c in pts):
            candidates.append(int(col))

    return int(candidates[0]) if len(candidates) == 1 else None


def encoded_ray_units(tile: np.ndarray, *, bg: int) -> int:
    """Distance code: number of distinct non-bg colors inside the payload tile."""
    cols = {int(v) for v in tile.flatten() if int(v) != int(bg)}
    return max(1, int(len(cols)))


def extend_repeating_tile_exact(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    total_units: int,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Paint additional whole-tile copies until the encoded unit count is reached."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    bg = int(bg)

    def put(rr: int, cc: int, v: int) -> None:
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if int(v) == int(bg):
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    for unit_idx in range(1, int(total_units)):
        base_r = int(r0 + unit_idx * dr * h)
        base_c = int(c0 + unit_idx * dc * w)
        for i in range(h):
            for j in range(w):
                put(base_r + i, base_c + j, int(tile[i, j]))


def _collect_control_specs(ctrl_mask: np.ndarray) -> list[dict]:
    """Describe each border strip as a direction plus its alignment span."""
    H, W = ctrl_mask.shape
    ctrl_lab, cn = label_mask(ctrl_mask, connectivity=4)
    specs: list[dict] = []
    for k in range(1, int(cn) + 1):
        pts = np.argwhere(ctrl_lab == int(k))
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if int(r0) == 0 and int(r1) == 0:
            specs.append({"direction": (-1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(r0) == int(H - 1) and int(r1) == int(H - 1):
            specs.append({"direction": (1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(c0) == 0 and int(c1) == 0:
            specs.append({"direction": (0, -1), "span": tuple(range(int(r0), int(r1) + 1))})
        elif int(c0) == int(W - 1) and int(c1) == int(W - 1):
            specs.append({"direction": (0, 1), "span": tuple(range(int(r0), int(r1) + 1))})
    return specs


def _along_span_for_direction(
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = map(int, direction)
    return int(r1 - r0 + 1) if int(dc) == 0 else int(c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each payload tile from its current position using border-coded direction and payload-coded distance.

    Input structure:
    - The background is the dominant color.
    - One dominant non-bg color forms several '+' anchors.
    - Each plus has one adjacent 2x2 payload tile, and one small border strip of a dedicated control color.
      Exactly one payload is already shown as a completed repeated ray that stops short of the border.

    Rule:
    - Match each payload component to the unique border strip aligned with its bbox: top/bottom strips match by
      columns, left/right strips by rows. The strip's border side gives the ray direction.
    - Count the payload's distinct non-bg colors. That count is the total number of 2x2 tile-units the ray should
      occupy, including the visible seed tile itself.
    - Compact payloads are extended by that exact number of copies; the preserved long clue ray is left unchanged.
      '+' pixels and control-strip pixels are never overwritten.

    Key insight:
    - The adjacent side of the payload no longer tells you where to fire, and the border no longer tells you where to
      stop. Direction comes from the edge control strip, while stopping distance comes from the payload's own color set.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    ctrl_color = detect_border_only_control_color(g, bg=bg, exclude={int(plus_col)})
    if ctrl_color is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    ctrl_specs = _collect_control_specs(g == int(ctrl_color))

    non_mask = (g != int(bg)) & (g != int(plus_col)) & (g != int(ctrl_color))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        adj_ids: set[int] = set()
        for r, c in plus_coords:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
        pts = np.argwhere(mark_lab == int(cid))
        if pts.size == 0:
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        tile = crop_bbox(g, bbox)
        span_rows = tuple(range(int(r0), int(r1) + 1))
        span_cols = tuple(range(int(c0), int(c1) + 1))

        dirs: list[tuple[int, int]] = []
        for spec in ctrl_specs:
            dr, dc = spec["direction"]
            if int(dc) == 0 and tuple(spec["span"]) == span_cols:
                dirs.append((int(dr), int(dc)))
            elif int(dr) == 0 and tuple(spec["span"]) == span_rows:
                dirs.append((int(dr), int(dc)))
        dirs = sorted(set(dirs))
        if len(dirs) != 1:
            continue

        direction = dirs[0]
        if _along_span_for_direction(bbox, direction) > 2:
            continue

        total_units = encoded_ray_units(tile, bg=int(bg))
        extend_repeating_tile_exact(
            out,
            tile,
            int(r0),
            int(c0),
            direction,
            total_units=int(total_units),
            bg=int(bg),
            protect={int(plus_col), int(ctrl_color)},
        )

    return out


# --------------- generator ---------------

_H_RANGE = (14, 29)
_W_RANGE = (14, 23)
_N_PLUS = 3
_SEED_SIZE = 2
_SEED_MAX_COLORS = 4
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 800
_PRECLUE_MIN_ALONG_SPAN = 4

_PLUS_SHAPES = [
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
        dtype=int,
    ),
]
_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span for one side of the plus shape."""
    dr, dc = direction
    if int(dc) == 0:
        row = plus_mask.shape[0] - 1 if int(dr) == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if int(dc) == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _build_seed_tile(
    plus_mask: np.ndarray,
    attach_direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
    desired_units_total: int | None = None,
) -> tuple[np.ndarray, int] | None:
    """Build a 2x2 payload tile with 2-4 distinct colors."""
    s, e = _nozzle_span(plus_mask, attach_direction)
    cross = int(e - s + 1)
    if cross != int(_SEED_SIZE):
        return None

    n_cells = int(_SEED_SIZE * _SEED_SIZE)
    max_k = min(int(_SEED_MAX_COLORS), n_cells, len(palette))
    if max_k < 2:
        return None

    if desired_units_total is None:
        n_colors = int(rng.integers(2, max_k + 1))
    else:
        n_colors = int(desired_units_total)
        if not 2 <= int(n_colors) <= int(max_k):
            return None
    chosen = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
    vals = chosen.copy()
    if n_cells > n_colors:
        vals.extend(list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True))))
    rng.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(_SEED_SIZE, _SEED_SIZE)
    return tile, int(n_colors)


def _anchor_seed_tile(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    attach_direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Place the payload flush against one side of the plus."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, attach_direction)
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, attach_direction)

    if int(dc) == 0:
        c0 = int(pc0 + s)
        r0 = int(pr1 + 1) if int(dr) == 1 else int(pr0 - th)
    else:
        r0 = int(pr0 + s)
        c0 = int(pc1 + 1) if int(dc) == 1 else int(pc0 - tw)
    return int(r0), int(c0)


def _project_tile_coords_exact(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    units_total: int,
) -> tuple[list[tuple[int, int, int]], int, int] | None:
    """Return the exact stopped ray, plus its along-axis span and border gap."""
    th, tw = tile.shape
    dr, dc = map(int, direction)

    cells: list[tuple[int, int, int]] = []
    for unit_idx in range(int(units_total)):
        base_r = int(seed_r0 + unit_idx * dr * th)
        base_c = int(seed_c0 + unit_idx * dc * tw)
        if base_r < 0 or base_c < 0 or base_r + th > H or base_c + tw > W:
            return None
        for i in range(th):
            for j in range(tw):
                cells.append((int(base_r + i), int(base_c + j), int(tile[i, j])))

    if int(dc) == 0:
        last_r0 = int(seed_r0 + dr * th * (units_total - 1))
        lo_r = min(int(seed_r0), int(last_r0))
        hi_r = max(int(seed_r0 + th - 1), int(last_r0 + th - 1))
        along_span = int(hi_r - lo_r + 1)
        boundary_gap = int(H - 1 - hi_r) if int(dr) == 1 else int(lo_r)
    else:
        last_c0 = int(seed_c0 + dc * tw * (units_total - 1))
        lo_c = min(int(seed_c0), int(last_c0))
        hi_c = max(int(seed_c0 + tw - 1), int(last_c0 + tw - 1))
        along_span = int(hi_c - lo_c + 1)
        boundary_gap = int(W - 1 - hi_c) if int(dc) == 1 else int(lo_c)

    return cells, int(along_span), int(boundary_gap)


def _build_control_cells(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """Place a border strip whose span matches the payload cross-axis span."""
    th, tw = tile.shape
    dr, dc = map(int, direction)
    if int(dc) == 0:
        row = 0 if int(dr) == -1 else int(H - 1)
        return [(int(row), int(seed_c0 + j)) for j in range(tw)]
    col = 0 if int(dc) == -1 else int(W - 1)
    return [(int(seed_r0 + i), int(col)) for i in range(th)]


def _allowed_ray_directions(attach_direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Allow same-side firing or a sideways turn, but never back through the plus."""
    opposite = (-int(attach_direction[0]), -int(attach_direction[1]))
    return [d for d in _DIRECTIONS if d != opposite]


def _direction_orientation(direction: tuple[int, int]) -> str:
    """Classify a direction as horizontal or vertical."""
    return "vertical" if int(direction[1]) == 0 else "horizontal"


def _choose_case_specs(rng: np.random.Generator) -> tuple[list[dict], int]:
    """
    Choose three role specs.

    Two roles are hidden rays that deliberately cover one vertical and one horizontal branch, while also
    splitting between a turned ray and a straight ray. The third role is the preserved clue ray.
    """
    vertical_dir = tuple(map(int, _DIRECTIONS[int(rng.choice([0, 1]))]))
    horizontal_dir = tuple(map(int, _DIRECTIONS[int(rng.choice([2, 3]))]))
    hidden_units = list(map(int, rng.choice(np.array([2, 3, 4]), size=2, replace=False)))

    if bool(rng.integers(0, 2)):
        hidden_specs = [
            {"ray_direction": vertical_dir, "must_turn": True, "units_total": hidden_units[0]},
            {"ray_direction": horizontal_dir, "must_turn": False, "units_total": hidden_units[1]},
        ]
    else:
        hidden_specs = [
            {"ray_direction": vertical_dir, "must_turn": False, "units_total": hidden_units[0]},
            {"ray_direction": horizontal_dir, "must_turn": True, "units_total": hidden_units[1]},
        ]

    clue_direction = tuple(map(int, rng.choice(_DIRECTIONS)))
    clue_units = int(rng.choice([3, 4]))
    clue_spec = {
        "ray_direction": clue_direction,
        "must_turn": None,
        "units_total": clue_units,
        "keep_full_in_input": True,
    }

    role_specs = hidden_specs + [clue_spec]
    rng.shuffle(role_specs)
    pre_idx = next(i for i, spec in enumerate(role_specs) if bool(spec.get("keep_full_in_input")))
    return role_specs, int(pre_idx)


def _has_neighbor(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
            return True
    return False


def _plus_position_is_clear(
    plus_cells: list[tuple[int, int]],
    occ: np.ndarray,
    mark_occ: np.ndarray,
    ctrl_occ: np.ndarray,
    plus_occ: np.ndarray,
) -> bool:
    """Future pluses must stay separated from previously placed payloads and controls."""
    cell_set = {(int(r), int(c)) for r, c in plus_cells}
    for r, c in cell_set:
        if int(occ[r, c]) != 0 or bool(plus_occ[r, c]):
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if (rr, cc) in cell_set:
                continue
            if 0 <= rr < occ.shape[0] and 0 <= cc < occ.shape[1]:
                if bool(mark_occ[rr, cc]) or bool(ctrl_occ[rr, cc]):
                    return False
    return True


def _component_is_clear(
    proposed: set[tuple[int, int]],
    occ: np.ndarray,
    mark_occ: np.ndarray,
    plus_occ: np.ndarray,
) -> bool:
    """Payload/ray cells must not overlap or touch earlier payload components or plus neighborhoods."""
    for r, c in proposed:
        if int(occ[r, c]) != 0 or bool(plus_occ[r, c]):
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < occ.shape[0] and 0 <= cc < occ.shape[1]:
                if (rr, cc) not in proposed and bool(mark_occ[rr, cc]):
                    return False
    return True


def _control_is_clear(
    proposed: set[tuple[int, int]],
    occ: np.ndarray,
    ctrl_occ: np.ndarray,
) -> bool:
    """Control strips must be separate border components."""
    for r, c in proposed:
        if int(occ[r, c]) != 0 or bool(ctrl_occ[r, c]):
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < occ.shape[0] and 0 <= cc < occ.shape[1]:
                if (rr, cc) not in proposed and bool(ctrl_occ[rr, cc]):
                    return False
    return True


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    mark_occ: np.ndarray,
    ctrl_occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    ctrl_color: int,
    used_row_spans: set[tuple[int, ...]],
    used_col_spans: set[tuple[int, ...]],
    desired_ray_direction: tuple[int, int] | None = None,
    desired_turn: bool | None = None,
    desired_units_total: int | None = None,
) -> dict | None:
    """Place one plus, one payload, its border control strip, and the exact output ray."""
    ph, pw = plus_shape.shape
    min_r = 2
    max_r = H - ph - 2
    min_c = 2
    max_c = W - pw - 2
    if max_r < min_r or max_c < min_c:
        return None

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(min_r, max_r + 1))
        c0 = int(rng.integers(min_c, max_c + 1))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) != 0
        ]
        if not _plus_position_is_clear(plus_cells, occ, mark_occ, ctrl_occ, plus_occ):
            continue

        attach_dirs = list(_DIRECTIONS)
        rng.shuffle(attach_dirs)
        for attach_direction in attach_dirs:
            allowed = _allowed_ray_directions(attach_direction)
            if desired_ray_direction is not None and tuple(map(int, desired_ray_direction)) not in allowed:
                continue
            if desired_turn is True and tuple(map(int, attach_direction)) == tuple(map(int, desired_ray_direction)):
                continue
            if desired_turn is False and tuple(map(int, attach_direction)) != tuple(map(int, desired_ray_direction)):
                continue

            built = _build_seed_tile(
                plus_shape,
                attach_direction,
                palette,
                rng,
                desired_units_total=desired_units_total,
            )
            if built is None:
                continue
            tile, units_total = built
            seed_r0, seed_c0 = _anchor_seed_tile(
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                attach_direction,
                tile,
            )
            th, tw = tile.shape
            if seed_r0 < 1 or seed_c0 < 1 or seed_r0 + th > H - 1 or seed_c0 + tw > W - 1:
                continue

            seed_rows = tuple(range(int(seed_r0), int(seed_r0 + th)))
            seed_cols = tuple(range(int(seed_c0), int(seed_c0 + tw)))
            if seed_rows in used_row_spans or seed_cols in used_col_spans:
                continue

            if desired_ray_direction is not None:
                ray_dirs = [tuple(map(int, desired_ray_direction))]
            else:
                sideways = [d for d in allowed if d != tuple(map(int, attach_direction))]
                rng.shuffle(sideways)
                ray_dirs = sideways + [tuple(map(int, attach_direction))]
            for ray_direction in ray_dirs:
                projected = _project_tile_coords_exact(
                    H,
                    W,
                    tile,
                    int(seed_r0),
                    int(seed_c0),
                    ray_direction,
                    int(units_total),
                )
                if projected is None:
                    continue
                cells, along_span, boundary_gap = projected
                if int(boundary_gap) < 1:
                    continue

                proposed_cells = {(int(r), int(c)) for r, c, _ in cells}
                if not _component_is_clear(proposed_cells, occ, mark_occ, plus_occ):
                    continue

                control_cells = _build_control_cells(
                    H,
                    W,
                    tile,
                    int(seed_r0),
                    int(seed_c0),
                    ray_direction,
                )
                control_set = {(int(r), int(c)) for r, c in control_cells}
                if not _control_is_clear(control_set, occ, ctrl_occ):
                    continue

                for r, c in plus_cells:
                    occ[r, c] = 1
                    out[r, c] = int(plus_color)
                    plus_occ[r, c] = True
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr = int(r + dr2)
                        cc = int(c + dc2)
                        if 0 <= rr < H and 0 <= cc < W:
                            plus_occ[rr, cc] = True

                for r, c, v in cells:
                    occ[r, c] = 2
                    mark_occ[r, c] = True
                    out[r, c] = int(v)

                for r, c in control_set:
                    occ[r, c] = 3
                    ctrl_occ[r, c] = True
                    out[r, c] = int(ctrl_color)

                used_row_spans.add(seed_rows)
                used_col_spans.add(seed_cols)
                return {
                    "tile": tile,
                    "cells": cells,
                    "control_set": control_set,
                    "seed_r0": int(seed_r0),
                    "seed_c0": int(seed_c0),
                    "seed_rows": seed_rows,
                    "seed_cols": seed_cols,
                    "units_total": int(units_total),
                    "along_span": int(along_span),
                    "boundary_gap": int(boundary_gap),
                    "attach_direction": tuple(map(int, attach_direction)),
                    "ray_direction": tuple(map(int, ray_direction)),
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    pre_idx: int,
) -> np.ndarray:
    """Keep one long clue ray; collapse the others back to their compact seeds."""
    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)

    return inp


def _generate_case(rng: np.random.Generator) -> dict | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    ctrl_color = int(rng.choice([c for c in others if c != plus_color]))
    palette = [c for c in others if c not in {plus_color, ctrl_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    mark_occ = np.zeros((H, W), dtype=bool)
    ctrl_occ = np.zeros((H, W), dtype=bool)
    plus_occ = np.zeros((H, W), dtype=bool)
    used_row_spans: set[tuple[int, ...]] = set()
    used_col_spans: set[tuple[int, ...]] = set()
    role_specs, pre_idx = _choose_case_specs(rng)

    pluses: list[dict] = []
    for idx in range(int(_N_PLUS)):
        spec = role_specs[idx]
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            mark_occ,
            ctrl_occ,
            plus_occ,
            out,
            plus_color,
            ctrl_color,
            used_row_spans,
            used_col_spans,
            desired_ray_direction=tuple(map(int, spec["ray_direction"])),
            desired_turn=spec["must_turn"],
            desired_units_total=int(spec["units_total"]),
        )
        if info is None:
            return None
        pluses.append(info)

    clue = pluses[int(pre_idx)]
    if int(clue["along_span"]) < int(_PRECLUE_MIN_ALONG_SPAN):
        return None
    if int(clue["units_total"]) < 3:
        return None

    inp = _build_input_from_output(out, pluses, bg, int(pre_idx))

    hidden = [p for i, p in enumerate(pluses) if i != int(pre_idx)]
    if len({int(p["units_total"]) for p in hidden}) < 2:
        return None
    if len({tuple(map(int, p["ray_direction"])) for p in hidden}) < 2:
        return None
    if len({_direction_orientation(tuple(map(int, p["ray_direction"]))) for p in hidden}) < 2:
        return None
    if len({bool(tuple(map(int, p["attach_direction"])) != tuple(map(int, p["ray_direction"]))) for p in hidden}) < 2:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    return {
        "input": inp,
        "output": out,
        "bg": int(bg),
        "plus_color": int(plus_color),
        "ctrl_color": int(ctrl_color),
        "pluses": pluses,
        "pre_idx": int(pre_idx),
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return np.array(case["input"], dtype=int)
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
