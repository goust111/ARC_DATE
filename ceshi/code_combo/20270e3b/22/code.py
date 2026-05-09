# Combo variant of 20270e3b (index 22)
# combo_spec: D3/1+D6/0+D8/2
# Name: 多连接器面积判定一对多

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
Design analysis for combo D3/1 + D6/0 + D8/2
=============================================

a) Concrete behavior trace
--------------------------
- Original task:
  there are exactly two sprites, each sprite has one horizontal ctrl bar, the origin-anchored
  sprite is base, and one plug is translated once so the two bars align vertically.
- Combo task:
  there are exactly three sprites: one larger base and two smaller plugs.
  Every sprite has exactly two horizontal ctrl bars.
  The base's two bars both face downward and become the two docking ports.
  Each plug's two bars both face upward, but only one of the two matches one base port.
  The other plug bar is a decoy whose length matches no base port.
- Small concrete trace:
  base connector lengths = {1, 3}; both are on the same base.
  plug A connector lengths = {1, 2}; its length-1 bar is active and length-2 is decoy.
  plug B connector lengths = {3, 2}; its length-3 bar is active and length-2 is decoy.
  Solver must first count areas to find the base, then split ctrl pixels into components,
  then find the unique matching bar inside each plug, then perform two independent dockings.

b) Complexity calibration
-------------------------
- The changed structure is visually obvious: inputs now show three sprites and six ctrl bars.
- To keep the matching problem readable rather than ambiguous, connector lengths stay in {1,2,3}
  and base uses two distinct lengths. The unused third length becomes the shared decoy length.
- This means each plug has exactly one valid match and one invalid candidate, which is enough to
  activate D3 without exploding the search space.

c) Implicit assumption audit against the original `main()`
----------------------------------------------------------
- `assert len(best_comps) == 2` breaks: combo needs exactly three components.
- `base = component closest to origin` breaks: combo base is defined by strict area maximum.
- `conn(sprite)` returning all ctrl pixels as one set breaks: every sprite now has two ctrl bars.
- `assert shpA == shpB` breaks: matching is no longer global, it is per-plug against base ports.
- Single `(tr, tc)` translation breaks: combo requires one translation per plug.
- Overlap logic becomes stricter: with several plugs, we must reject plug-plug overlap and only
  allow plug-base overlap on the matched 2-row docking seam.

d) Placement legality
---------------------
- Generator uses output-first construction.
- It first builds the docked base with two separated slot regions.
- Then it builds one plug for each slot, with the active connector at local top-left so docking
  never pushes the plug above or left of base.
- The two docked plugs occupy disjoint horizontal slots, so they cannot overlap each other.
- Finally, input placement pastes one smaller plug at the origin and scatters the larger base and
  the other plug elsewhere with padding=1 under 8-connectivity, guaranteeing exactly 3 components.

e) Visual observability
-----------------------
- A human looking at 2-4 examples can see that:
  1) the origin sprite is not the stationary one,
  2) one large multi-port sprite receives two smaller plugs,
  3) each plug carries an extra unused orange bar.
- Because the output shows both plugs attached and all orange bars recolored away, the unique
  active-vs-decoy distinction is recoverable from examples.

f) Visual cleanliness
---------------------
- Shapes are thin scaffold-like sprites, not dense blocks.
- Connector bars are on separate rows, so the six orange bars remain readable.
- Docking seams are separated by at least one empty column in the assembled state, avoiding
  confusing multi-layer overlaps.

g) Color inference
------------------
- Colors are unchanged from the original task: bg=1, obj=4, ctrl=7.
- The combo changes structure, counting, and multiplicity only; it does not add a hidden
  color-remapping rule.

h) New solver assumptions and generator guarantees
--------------------------------------------------
- Solver assumes exactly 3 foreground components. Generator always places 1 base + 2 plugs.
- Solver assumes the largest component is unique. Generator enforces base area > each plug area.
- Solver assumes every sprite has exactly 2 horizontal ctrl components.
  Generator validates this with the same connector parser used by `main()`.
- Solver assumes each plug has exactly one matching connector among its two bars and that the two
  matched base bars are different. Generator constructs lengths to guarantee this.
- Solver assumes docked plugs do not overlap each other and only overlap base on the seam.
  Generator reserves disjoint horizontal slots and checks the assembled geometry explicitly.
"""


BG = 1
OBJ = 4
CTRL = 7


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _sprite_pts(grid: np.ndarray, bg: int = BG) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid != int(bg))}


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (int(r0), int(c0))


def _scene_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    lab, n = label_mask(grid != BG, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    assert len(comps) == 3
    return comps


def _connector_infos(grid: np.ndarray, sprite: set[tuple[int, int]]) -> list[dict[str, object]]:
    ctrl_mask = np.zeros_like(grid, dtype=bool)
    for r, c in sprite:
        if int(grid[int(r), int(c)]) == CTRL:
            ctrl_mask[int(r), int(c)] = True

    lab, n = label_mask(ctrl_mask, connectivity=4)
    infos: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        assert coords.size > 0
        rows = sorted({int(r) for r, _c in coords})
        assert len(rows) == 1
        row = int(rows[0])
        cols = sorted(int(c) for _r, c in coords)
        assert cols == list(range(cols[0], cols[-1] + 1))

        pts = {(int(row), int(c)) for c in cols}
        body_above = all((int(row - 1), int(c)) in sprite and int(grid[int(row - 1), int(c)]) == OBJ for c in cols)
        body_below = all((int(row + 1), int(c)) in sprite and int(grid[int(row + 1), int(c)]) == OBJ for c in cols)
        free_above = all((int(row - 1), int(c)) not in sprite for c in cols)
        free_below = all((int(row + 1), int(c)) not in sprite for c in cols)

        if body_above and free_below and not body_below:
            orientation = "down"
        elif body_below and free_above and not body_above:
            orientation = "up"
        else:
            orientation = "other"

        shape, anchor = _norm_pts(pts)
        infos.append(
            {
                "pts": pts,
                "row": int(row),
                "c0": int(cols[0]),
                "length": int(len(cols)),
                "shape": shape,
                "anchor": anchor,
                "orientation": orientation,
            }
        )
    return infos


def _build_scene_from_base_index(grid: np.ndarray, base_idx: int) -> dict[str, object]:
    comps = _scene_components(grid)
    areas = [int(len(comp)) for comp in comps]
    boxes = [_bbox_of_pts(comp) for comp in comps]

    base = comps[int(base_idx)]
    base_infos = _connector_infos(grid, base)
    assert len(base_infos) == 2
    assert all(info["orientation"] == "down" for info in base_infos)

    base_map: dict[frozenset[tuple[int, int]], dict[str, object]] = {}
    for info in base_infos:
        shape = info["shape"]
        assert shape not in base_map
        base_map[shape] = info

    assignments: list[dict[str, object]] = []
    used_shapes: set[frozenset[tuple[int, int]]] = set()
    for idx, plug in enumerate(comps):
        if idx == int(base_idx):
            continue
        plug_infos = _connector_infos(grid, plug)
        assert len(plug_infos) == 2
        assert all(info["orientation"] == "up" for info in plug_infos)

        matches = [info for info in plug_infos if info["shape"] in base_map]
        assert len(matches) == 1
        active_plug = matches[0]
        active_shape = active_plug["shape"]
        assert active_shape not in used_shapes
        used_shapes.add(active_shape)
        assignments.append(
            {
                "plug": plug,
                "plug_idx": int(idx),
                "plug_infos": plug_infos,
                "plug_conn": active_plug,
                "base_conn": base_map[active_shape],
            }
        )

    assert len(assignments) == 2
    assert len(used_shapes) == len(base_map) == 2
    return {
        "components": comps,
        "areas": areas,
        "boxes": boxes,
        "base_idx": int(base_idx),
        "base": base,
        "base_infos": base_infos,
        "assignments": assignments,
    }


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    comps = _scene_components(grid)
    areas = [int(len(comp)) for comp in comps]
    max_area = max(areas)
    assert areas.count(max_area) == 1
    base_idx = max(range(len(comps)), key=lambda idx: int(areas[idx]))
    return _build_scene_from_base_index(grid, int(base_idx))


def _render_solution(grid: np.ndarray, scene: dict[str, object]) -> np.ndarray:
    base = scene["base"]
    shifted_union: set[tuple[int, int]] = set()
    shifted_plugs: list[tuple[set[tuple[int, int]], int, int]] = []

    for item in scene["assignments"]:
        plug = item["plug"]
        plug_conn = item["plug_conn"]
        base_conn = item["base_conn"]
        pr0, pc0 = plug_conn["anchor"]
        dr0, dc0 = base_conn["anchor"]

        tr = int(dr0 - 1 - pr0)
        tc = int(dc0 - pc0)
        plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

        dock_cols = range(int(dc0), int(dc0 + base_conn["length"]))
        dock_zone = {(int(dr0 - 1), int(c)) for c in dock_cols}
        dock_zone |= {(int(dr0), int(c)) for c in dock_cols}

        for pt in plug_shift:
            if pt in base:
                assert pt in dock_zone
        assert shifted_union.isdisjoint(plug_shift)
        shifted_union |= plug_shift
        shifted_plugs.append((plug, int(tr), int(tc)))

    pts_all = set(base) | shifted_union
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    br0, bc0, _br1, _bc1 = _bbox_of_pts(base)
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(grid[int(r), int(c)])

    for plug, tr, tc in shifted_plugs:
        for r, c in plug:
            rr = int(r + tr)
            cc = int(c + tc)
            val = int(grid[int(r), int(c)])
            cur = int(out[int(rr - rr0), int(cc - cc0)])
            if cur != BG and val != BG:
                assert cur == CTRL or val == CTRL
            out[int(rr - rr0), int(cc - cc0)] = int(val)

    out[out == CTRL] = OBJ
    nz = np.argwhere(out != BG)
    if nz.size == 0:
        return np.full((0, 0), BG, dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    scene = _parse_scene(g)
    return _render_solution(g, scene)


def _origin_counterfactual(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    comps = _scene_components(g)
    boxes = [_bbox_of_pts(comp) for comp in comps]
    origin_idx = min(range(len(comps)), key=lambda idx: int(boxes[idx][0] + boxes[idx][1]))
    scene = _build_scene_from_base_index(g, int(origin_idx))
    return _render_solution(g, scene)


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    comps = _scene_components(g)
    areas = [int(len(comp)) for comp in comps]
    boxes = [_bbox_of_pts(comp) for comp in comps]
    origin_idx = min(range(len(comps)), key=lambda idx: int(boxes[idx][0] + boxes[idx][1]))
    scene = _parse_scene(g)

    origin_rule_same = False
    try:
        wrong = _origin_counterfactual(g)
        origin_rule_same = bool(np.array_equal(wrong, _render_solution(g, scene)))
    except Exception:
        origin_rule_same = False

    per_component_connector_counts = [len(_connector_infos(g, comp)) for comp in comps]
    plug_match_counts = []
    plug_length_pairs = []
    base_lengths = sorted(int(info["length"]) for info in scene["base_infos"])
    for item in scene["assignments"]:
        lengths = sorted(int(info["length"]) for info in item["plug_infos"])
        plug_length_pairs.append(tuple(lengths))
        match_count = sum(1 for info in item["plug_infos"] if info["shape"] in {base_info["shape"] for base_info in scene["base_infos"]})
        plug_match_counts.append(int(match_count))

    return {
        "areas": tuple(areas),
        "base_area": int(areas[scene["base_idx"]]),
        "origin_idx": int(origin_idx),
        "base_idx": int(scene["base_idx"]),
        "origin_is_base": bool(origin_idx == scene["base_idx"]),
        "origin_rule_same": bool(origin_rule_same),
        "connectors_per_component": tuple(per_component_connector_counts),
        "base_lengths": tuple(base_lengths),
        "plug_length_pairs": tuple(plug_length_pairs),
        "plug_match_counts": tuple(plug_match_counts),
    }


def _build_base(base_lengths: list[int], decoy_len: int) -> tuple[np.ndarray, list[dict[str, int]]] | None:
    assert len(base_lengths) == 2
    slot_widths = [int(length + decoy_len + _rng.randint(0, 1)) for length in base_lengths]
    left_pad = _rng.randint(0, 1)
    between_gap = _rng.randint(1, 2)
    width = int(left_pad + slot_widths[0] + between_gap + slot_widths[1])
    if width > 13:
        return None

    conn_row = _rng.randint(3, 4)
    height = int(conn_row + 1)
    body = np.zeros((height, width), dtype=bool)

    body[0, :] = True
    body[0:conn_row, 0] = True
    if conn_row >= 3:
        shelf_row = _rng.randint(1, conn_row - 2)
        shelf_end = width - 1 - _rng.randint(0, 1)
        body[int(shelf_row), 0 : int(shelf_end) + 1] = True

    starts = [
        int(left_pad),
        int(left_pad + slot_widths[0] + between_gap),
    ]
    ports: list[dict[str, int]] = []
    for start, length, slot_width in zip(starts, base_lengths, slot_widths):
        for c in range(int(start), int(start + length)):
            body[0:conn_row, int(c)] = True
        ports.append(
            {
                "row": int(conn_row),
                "c0": int(start),
                "length": int(length),
                "slot_width": int(slot_width),
            }
        )

    grid = np.full((height, width), BG, dtype=int)
    grid[body] = OBJ
    for port in ports:
        c0 = int(port["c0"])
        for c in range(c0, c0 + int(port["length"])):
            grid[int(port["row"]), int(c)] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    infos = _connector_infos(grid, _sprite_pts(grid))
    if len(infos) != 2 or any(info["orientation"] != "down" for info in infos):
        return None
    if sorted(int(info["length"]) for info in infos) != sorted(int(length) for length in base_lengths):
        return None
    return grid, ports


def _build_plug(active_len: int, decoy_len: int, slot_width: int) -> np.ndarray | None:
    if slot_width < active_len + decoy_len:
        return None

    height = _rng.randint(4, 6)
    width = int(slot_width)
    decoy_c0 = int(width - decoy_len)
    if decoy_c0 < active_len:
        return None
    decoy_row = _rng.randint(2, height - 2)

    mask = np.zeros((height, width), dtype=bool)
    mask[height - 1, :] = True

    mask[0, 0:active_len] = True
    mask[1, 0:active_len] = True
    mask[1 : height - 1, 0] = True

    mask[decoy_row, decoy_c0 : decoy_c0 + decoy_len] = True
    mask[decoy_row + 1, decoy_c0 : decoy_c0 + decoy_len] = True
    mask[decoy_row + 1 : height - 1, decoy_c0] = True

    extra_lo = max(1, decoy_row + 1)
    extra_hi = height - 2
    if _rng.random() < 0.5 and width >= 3 and extra_lo <= extra_hi:
        extra_row = _rng.randint(extra_lo, extra_hi)
        mask[int(extra_row), 0 : int(width - _rng.randint(0, 1))] = True

    grid = np.full((height, width), BG, dtype=int)
    grid[mask] = OBJ
    grid[0, 0:active_len] = CTRL
    grid[decoy_row, decoy_c0 : decoy_c0 + decoy_len] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    infos = _connector_infos(grid, _sprite_pts(grid))
    if len(infos) != 2 or any(info["orientation"] != "up" for info in infos):
        return None
    if sorted(int(info["length"]) for info in infos) != sorted([int(active_len), int(decoy_len)]):
        return None
    return grid


def _assembled_geometry_ok(
    base_grid: np.ndarray,
    ports: list[dict[str, int]],
    plug_grids: list[np.ndarray],
) -> bool:
    base_pts = _sprite_pts(base_grid)
    occupied: set[tuple[int, int]] = set()

    for port, plug_grid in zip(ports, plug_grids):
        plug_pts = _sprite_pts(plug_grid)
        tr = int(port["row"] - 1)
        tc = int(port["c0"])
        shifted = {(int(r + tr), int(c + tc)) for r, c in plug_pts}

        dock_zone = {(int(port["row"] - 1), int(c)) for c in range(int(port["c0"]), int(port["c0"] + port["length"]))}
        dock_zone |= {(int(port["row"]), int(c)) for c in range(int(port["c0"]), int(port["c0"] + port["length"]))}

        for pt in shifted:
            if pt in base_pts and pt not in dock_zone:
                return False
        if not occupied.isdisjoint(shifted):
            return False
        occupied |= shifted

    return True


def _place_input_sprites(base_sprite: np.ndarray, plug_sprites: list[np.ndarray]) -> np.ndarray | None:
    assert len(plug_sprites) == 2
    areas = [int(np.sum(sprite != BG)) for sprite in plug_sprites]
    origin_idx = min(range(2), key=lambda idx: int(areas[idx]))
    origin_plug = plug_sprites[int(origin_idx)]
    other_plug = plug_sprites[1 - int(origin_idx)]

    max_h = max(int(base_sprite.shape[0]), int(origin_plug.shape[0]), int(other_plug.shape[0]))
    max_w = max(int(base_sprite.shape[1]), int(origin_plug.shape[1]), int(other_plug.shape[1]))
    min_h = max(9, int(max_h + 3))
    min_w = max(13, int(max_w + 4), int(base_sprite.shape[1] + other_plug.shape[1] + 2))
    if min_h > 13 or min_w > 18:
        return None

    for _ in range(80):
        canvas_h = _rng.randint(min_h, 13)
        canvas_w = _rng.randint(min_w, 18)
        canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
        paste(canvas, origin_plug, 0, 0, transparent=BG)
        try:
            br, bc = random_free_location(canvas, base_sprite, transparent=BG, padding=1, padding_connectivity=8)
            paste(canvas, base_sprite, br, bc, transparent=BG)
            pr, pc = random_free_location(canvas, other_plug, transparent=BG, padding=1, padding_connectivity=8)
            paste(canvas, other_plug, pr, pc, transparent=BG)
        except ValueError:
            continue

        lab, n = label_mask(canvas != BG, connectivity=8)
        if int(n) != 3:
            continue
        return canvas
    return None


def _gen_attempt() -> np.ndarray | None:
    base_lengths = _rng.sample([1, 2, 3], 2)
    decoy_len = next(length for length in [1, 2, 3] if length not in base_lengths)

    built_base = _build_base(base_lengths, int(decoy_len))
    if built_base is None:
        return None
    base_grid, ports = built_base

    plug_grids: list[np.ndarray] = []
    for port in ports:
        plug_grid = _build_plug(int(port["length"]), int(decoy_len), int(port["slot_width"]))
        if plug_grid is None:
            return None
        plug_grids.append(plug_grid)

    if not _assembled_geometry_ok(base_grid, ports, plug_grids):
        return None

    base_area = int(np.sum(base_grid != BG))
    plug_areas = [int(np.sum(plug_grid != BG)) for plug_grid in plug_grids]
    if not all(base_area >= area + 3 for area in plug_areas):
        return None

    base_sprite = _crop_non_bg(base_grid, BG)
    if base_sprite is None:
        return None
    plug_sprites = [_crop_non_bg(plug_grid, BG) for plug_grid in plug_grids]
    if any(sprite is None for sprite in plug_sprites):
        return None

    canvas = _place_input_sprites(base_sprite, [sprite for sprite in plug_sprites if sprite is not None])
    if canvas is None:
        return None
    if np.sum(canvas == BG) <= canvas.size // 2:
        return None

    try:
        out = main(canvas)
        metrics = _fusion_metrics(canvas)
    except Exception:
        return None

    if out.size == 0 or not np.any(out != BG):
        return None
    if metrics["origin_is_base"]:
        return None
    if metrics["origin_rule_same"]:
        return None
    if metrics["connectors_per_component"] != (2, 2, 2):
        return None
    if metrics["plug_match_counts"] != (1, 1):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 600 attempts")
