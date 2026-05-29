# Combo variant of 271d71e2 (index 6)
# combo_spec: D3/3+D8/0
# Name: 灰色消失容器缩小
#
# Design analysis
# ---------------
# fused_rule:
#   - Keep the original scene grammar: magenta background, black rectangular containers,
#     grey/orange interiors, one same-colored head line adjacent to a container side,
#     and one matching destination line farther along the movement axis.
#   - Keep the original matching logic: each container moves toward the nearest matching
#     destination on the head axis.
#   - Replace the original "charge grey -> orange" step with a fused per-step process:
#       1. erase one head-most grey interior pixel to background;
#       2. advance the whole container by one cell toward the destination;
#       3. shrink the container by dropping one trailing row/column of body.
#   - Repeat until either the destination distance is exhausted, no grey pixels remain,
#     or the container can no longer shrink while keeping a non-empty interior axis.
#
# concrete_trace:
#   - Example: move right, inner row `[5, 5, 5, 5]`, distance = 2.
#   - Original task: output inner becomes `[5, 5, 7, 7]` and the box keeps the same size.
#   - This combo:
#       step 1: erase head-most grey -> `[5, 5, 5, 6]`, then trim trailing side -> `[5, 5, 6]`
#       step 2: erase head-most grey -> `[5, 6, 6]`, then trim trailing side -> `[6, 6]`
#     The final box is two cells narrower and the surviving interior contains two bg holes.
#   - Difference from D3-only: holes remain, but the frame also contracts.
#   - Difference from D8-only: consumed material does not become an orange external trail;
#     it disappears, and the damage is visible inside the surviving smaller container.
#
# complexity_calibration:
#   - The solver still uses the original head/destination parsing, so the object matching
#     problem stays stable and recognizable.
#   - The generator keeps 2-4 containers and mixes movers with 0-1 full-orange decoys.
#   - Mover interiors are built with at most two initial orange cells, so the erased holes
#     remain visually obvious instead of being hidden behind a large pre-charged orange slab.
#   - At least one mover is forced to take >= 2 steps, making the fused behavior easy to see.
#
# hidden_assumption_audit:
#   - Original container detection assumes input interiors contain only grey/orange.
#     This still holds because bg holes appear only in the output.
#   - Original per-pixel recolor assumed the box size stayed fixed. That no longer holds,
#     so the solver recomputes the container box step-by-step instead of pasting a fixed patch.
#   - D8-only used grey *layers* as fuel. That would over-constrain this combo because D3/3
#     explicitly talks about grey pixels disappearing one by one. Here the real fuel is
#     current grey existence, while geometry adds an additional shrinkability cap.
#   - The new solver assumes a trailing shrink always removes the side opposite the head.
#     The generator guarantees this matches the motion axis used by the head/destination pair.
#   - The new solver assumes the erased pixel should survive the shrink. Erasing from the
#     head side before trimming the trailing side makes that invariant true.
#
# placement_legality:
#   - Assemblies reserve the original corridor: destination + gap + head + container.
#     Because the combo only shrinks the moved container and leaves no external trail,
#     every output stays inside that same reserved envelope.
#   - Placement still enforces a 1-cell background moat between assemblies, preventing
#     accidental head/destination ambiguity from touching objects.
#
# visual_observability:
#   - Humans can infer two simultaneous invariants from 2-4 examples:
#       1. moved distance is tied to how many interior holes appear;
#       2. the moved box is smaller along the movement axis by exactly that many steps.
#   - This is structurally visible and does not depend on hidden numeric ordering tricks.
#
# cleanliness:
#   - No extra colors or overlay artifacts are introduced.
#   - Damage is confined to the surviving smaller container, so the result reads as one
#     coherent "self-consuming" motion instead of unrelated noise.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _ordered_cells(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if direction == (0, 1):
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif direction == (0, -1):
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif direction == (1, 0):
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _grey_order(inner: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    return [p for p in _ordered_cells(inner.shape[0], inner.shape[1], direction) if int(inner[p]) == GREY]


def _trim_inner_one_step(inner: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    if direction == (-1, 0):
        return inner[:-1, :].copy()
    if direction == (1, 0):
        return inner[1:, :].copy()
    if direction == (0, -1):
        return inner[:, :-1].copy()
    return inner[:, 1:].copy()


def _shrink_box_one_step(box: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    r0 += direction[0]
    c0 += direction[1]
    r1 += direction[0]
    c1 += direction[1]
    if direction == (-1, 0):
        r1 -= 1
    elif direction == (1, 0):
        r0 += 1
    elif direction == (0, -1):
        c1 -= 1
    else:
        c0 += 1
    return int(r0), int(c0), int(r1), int(c1)


def _head_cells_from_box(box: tuple[int, int, int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if direction == (-1, 0):
        return [(r0 - 1, c) for c in range(c0, c1 + 1)]
    if direction == (1, 0):
        return [(r1 + 1, c) for c in range(c0, c1 + 1)]
    if direction == (0, -1):
        return [(r, c0 - 1) for r in range(r0, r1 + 1)]
    return [(r, c1 + 1) for r in range(r0, r1 + 1)]


def _stamp_container(out: np.ndarray, box: tuple[int, int, int, int], inner: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = BORDER
    out[r0 + 1 : r1, c0 + 1 : c1] = inner


def _simulate_combo(
    box: tuple[int, int, int, int], inner: np.ndarray, direction: tuple[int, int], distance: int
) -> tuple[tuple[int, int, int, int], np.ndarray, int]:
    cur_box = tuple(int(v) for v in box)
    cur_inner = np.array(inner, dtype=int).copy()
    steps = 0
    for _ in range(int(distance)):
        axis_depth = int(cur_inner.shape[0] if direction[0] != 0 else cur_inner.shape[1])
        if axis_depth <= 1:
            break
        order = _grey_order(cur_inner, direction)
        if not order:
            break
        rr, cc = order[0]
        cur_inner[rr, cc] = BG
        cur_box = _shrink_box_one_step(cur_box, direction)
        cur_inner = _trim_inner_one_step(cur_inner, direction)
        steps += 1
    return cur_box, cur_inner, int(steps)


def _head_info_for_container(
    g: np.ndarray,
    box_: tuple[int, int, int, int],
    bg: int,
    comp_by_color: dict[int, list[dict]],
) -> dict | None:
    H, W = g.shape
    r0, c0, r1, c1 = box_
    head_seeds = []
    for r, c in _rect_border_coords(r0, c0, r1, c1):
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                val = int(g[rr, cc])
                if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                    head_seeds.append((rr, cc, val))
    if not head_seeds:
        return None

    hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
    seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(hc)}
    comp0 = None
    for comp in comp_by_color.get(hc, []):
        if comp["pts"] & seed_pts:
            comp0 = comp
            break
    if comp0 is None:
        return None

    border_pts = list(_rect_border_coords(r0, c0, r1, c1))
    best_d = None
    dist = {}
    for pr, pc in comp0["pts"]:
        d = min(abs(int(pr - br)) + abs(int(pc - bc)) for br, bc in border_pts)
        dist[(int(pr), int(pc))] = int(d)
        if best_d is None or d < best_d:
            best_d = int(d)
    if best_d is None:
        return None
    head_pts = {p for p, d in dist.items() if int(d) == int(best_d)}

    coords = np.array(list(head_pts), dtype=int)
    hr0, hc0, hr1, hc1 = bbox_coords(coords)
    if hc1 < c0:
        direction = (0, -1)
    elif hc0 > c1:
        direction = (0, 1)
    elif hr1 < r0:
        direction = (-1, 0)
    elif hr0 > r1:
        direction = (1, 0)
    else:
        direction = (0, 0)

    r00, c00 = int(hr0), int(hc0)
    head_shape = {(int(r - r00), int(c - c00)) for r, c in head_pts}
    return {
        "color": hc,
        "head_pts": set(head_pts),
        "head_shape": head_shape,
        "head_tl": (r00, c00),
        "comp_pts": set(comp0["pts"]),
        "dir": direction,
    }


def _pick_destination(
    g: np.ndarray,
    box_: tuple[int, int, int, int],
    head_info: dict,
    comp_by_color: dict[int, list[dict]],
) -> dict | None:
    H, W = g.shape
    hc = int(head_info["color"])
    head_pts = set(head_info["head_pts"])
    head_shape = head_info["head_shape"]
    head_tl = head_info["head_tl"]
    comp_pts = set(head_info["comp_pts"])
    direction = head_info["dir"]

    dests = []
    for d in comp_by_color.get(hc, []):
        if d["shape"] != head_shape:
            continue
        if d["pts"] == comp_pts:
            continue
        dests.append(d)

    rem = comp_pts - head_pts
    if rem:
        mask_rem = np.zeros((H, W), dtype=bool)
        for rr, cc in rem:
            mask_rem[int(rr), int(cc)] = True
        labr, nr = label(mask_rem, structure=STRUCT4)
        for k in range(1, int(nr) + 1):
            pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
            if not pts:
                continue
            adj = False
            for r, c in pts:
                for dr, dc in DIR4:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                        adj = True
                        break
                if adj:
                    break
            if adj:
                continue
            rs = [r for r, _c in pts]
            cs = [c for _r, c in pts]
            r00, c00 = int(min(rs)), int(min(cs))
            shp = {(int(r - r00), int(c - c00)) for r, c in pts}
            if shp != head_shape:
                continue
            dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})

    best = None
    best_dest = None
    for d in dests:
        tr0 = int(d["tl"][0] - head_tl[0])
        tc0 = int(d["tl"][1] - head_tl[1])
        if direction[0] != 0 and tc0 != 0:
            continue
        if direction[1] != 0 and tr0 != 0:
            continue
        if direction[0] == 1 and tr0 <= 0:
            continue
        if direction[0] == -1 and tr0 >= 0:
            continue
        if direction[1] == 1 and tc0 <= 0:
            continue
        if direction[1] == -1 and tc0 >= 0:
            continue
        dist = abs(tr0) + abs(tc0)
        if best is None or dist < best:
            best = dist
            best_dest = d
    return best_dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Combo rule:
    - Move toward the matched destination exactly as in the original task.
    - Each successful step erases one head-side grey pixel to background.
    - The container simultaneously advances by one cell and shrinks by one
      trailing row/column of body, so the moved frame becomes smaller.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != int(BORDER) for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    for cont in containers:
        box_ = cont["box"]
        r0, c0, r1, c1 = box_
        head_info = _head_info_for_container(g, box_, bg, comp_by_color)
        if head_info is None or head_info["dir"] == (0, 0):
            continue
        best_dest = _pick_destination(g, box_, head_info, comp_by_color)
        if best_dest is None:
            continue

        head_pts = set(head_info["head_pts"])
        hc = int(head_info["color"])
        head_tl = head_info["head_tl"]
        direction = head_info["dir"]
        distance = abs(int(best_dest["tl"][0] - head_tl[0])) + abs(int(best_dest["tl"][1] - head_tl[1]))
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        final_box, final_inner, steps = _simulate_combo(box_, inner, direction, distance)
        if steps <= 0:
            continue

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        _stamp_container(out, final_box, final_inner)
        for r, c in _head_cells_from_box(final_box, direction):
            if 0 <= r < H and 0 <= c < W:
                out[int(r), int(c)] = int(hc)

    return out


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    return _ordered_cells(inner_h, inner_w, SIDE_TO_DIR[head_side])


def _build_input_interior(inner_h: int, inner_w: int, head_side: int, initial_orange: int) -> np.ndarray:
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for idx_cell, (r, c) in enumerate(_interior_fill_order(inner_h, inner_w, head_side)):
        if idx_cell >= int(initial_orange):
            break
        interior[r, c] = ORANGE
    return interior


def _assembly_layout(outer_h: int, outer_w: int, head_side: int, gap: int) -> dict[str, int]:
    if head_side == 0:
        return {"total_h": 1 + gap + 1 + outer_h, "total_w": outer_w, "cont_r": gap + 2, "cont_c": 0}
    if head_side == 1:
        return {"total_h": outer_h + 1 + gap + 1, "total_w": outer_w, "cont_r": 0, "cont_c": 0}
    if head_side == 2:
        return {"total_h": outer_h, "total_w": 1 + gap + 1 + outer_w, "cont_r": 0, "cont_c": gap + 2}
    return {"total_h": outer_h, "total_w": outer_w + 1 + gap + 1, "cont_r": 0, "cont_c": 0}


def _build_spec(is_decoy: bool) -> dict:
    head_side = random.randint(0, 3)
    if is_decoy:
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        initial_orange = inner_h * inner_w
        gap = random.randint(0, 6)
    else:
        if head_side in (0, 1):
            inner_h = random.randint(2, 8)
            inner_w = random.randint(1, 5)
        else:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(2, 5)
        area = inner_h * inner_w
        initial_orange = random.randint(0, min(2, area - 1))
        gap = random.randint(1, 10)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    layout = _assembly_layout(outer_h, outer_w, head_side, gap)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "initial_orange": initial_orange,
        "is_decoy": is_decoy,
        **layout,
    }


def _draw_input_assembly(grid: np.ndarray, spec: dict, pr: int, pc: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    cr = pr + spec["cont_r"]
    cc = pc + spec["cont_c"]
    box = (cr, cc, cr + oh - 1, cc + ow - 1)
    inner = _build_input_interior(spec["inner_h"], spec["inner_w"], hs, spec["initial_orange"])

    _stamp_container(grid, box, inner)
    for r, c in _head_cells_from_box(box, SIDE_TO_DIR[hs]):
        grid[r, c] = HEAD_COLOR

    if hs == 0:
        grid[cr - 2 - spec["gap"], cc : cc + ow] = HEAD_COLOR
    elif hs == 1:
        grid[cr + oh + 1 + spec["gap"], cc : cc + ow] = HEAD_COLOR
    elif hs == 2:
        grid[cr : cr + oh, cc - 2 - spec["gap"]] = HEAD_COLOR
    else:
        grid[cr : cr + oh, cc + ow + 1 + spec["gap"]] = HEAD_COLOR

    return box, inner


def _expected_output(grid: np.ndarray, specs: list[dict], placements: list[tuple[int, int]]) -> tuple[np.ndarray, list[int], list[int]]:
    expected = grid.copy()
    step_counts: list[int] = []
    hole_counts: list[int] = []
    for spec, (pr, pc) in zip(specs, placements):
        oh, ow = spec["outer_h"], spec["outer_w"]
        hs = spec["head_side"]
        direction = SIDE_TO_DIR[hs]
        cr = pr + spec["cont_r"]
        cc = pc + spec["cont_c"]
        box = (cr, cc, cr + oh - 1, cc + ow - 1)
        inner = _build_input_interior(spec["inner_h"], spec["inner_w"], hs, spec["initial_orange"])
        distance = spec["gap"] + 1
        final_box, final_inner, steps = _simulate_combo(box, inner, direction, distance)
        step_counts.append(int(steps))
        hole_counts.append(int(np.count_nonzero(final_inner == BG)) if steps > 0 else 0)
        if steps <= 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = BG
        for r, c in _head_cells_from_box(box, direction):
            expected[r, c] = BG
        _stamp_container(expected, final_box, final_inner)
        for r, c in _head_cells_from_box(final_box, direction):
            expected[r, c] = HEAD_COLOR

    return expected, step_counts, hole_counts


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = [_build_spec(i >= n_movers) for i in range(n_containers)]

    total_area = sum(int(s["total_h"]) * int(s["total_w"]) for s in specs)
    min_side = max(8, int(total_area**0.5) + 3)
    max_asm_h = max(int(s["total_h"]) for s in specs)
    max_asm_w = max(int(s["total_w"]) for s in specs)
    lo_h = max(8, max_asm_h + 2)
    lo_w = max(8, max_asm_w + 2)
    hi_h = min(26, max(lo_h, min_side + 6))
    hi_w = min(26, max(lo_w, min_side + 6))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for spec in specs:
        th, tw = int(spec["total_h"]), int(spec["total_w"])
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                occupied[r0 : r0 + th, c0 : c0 + tw] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, (pr, pc) in zip(specs, placements):
        _draw_input_assembly(grid, spec, pr, pc)

    expected, step_counts, hole_counts = _expected_output(grid, specs, placements)
    mover_steps = [step_counts[i] for i, s in enumerate(specs) if not s["is_decoy"]]
    mover_holes = [hole_counts[i] for i, s in enumerate(specs) if not s["is_decoy"]]
    if not mover_steps or max(mover_steps) < 2:
        return None
    if any(h != st for h, st in zip(mover_holes, mover_steps) if st > 0):
        return None

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 500 attempts")
