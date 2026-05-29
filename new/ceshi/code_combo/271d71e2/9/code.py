# Combo variant of 271d71e2 (index 9)
# combo_spec: D6/2+D7/1
# Name: 颜色匹配配对加无目标干扰

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste

# Design analysis for the combo variant
# -------------------------------------
# Base task recap:
# - Magenta background 6.
# - Black rectangular containers around grey/orange interiors.
# - A 1-cell-thick head line sits flush with one side of the container.
# - A same-shaped target line lies farther ahead on the movement axis.
# - Steps = min(head-to-target distance, grey count). The closest grey cells from the head side
#   turn orange, and the container+head translate by that many cells.
#
# Combo change:
# - D6/2: heads and targets are no longer globally maroon. Each mover uses its own visible marker
#   color, and the target must match that color.
# - D7/1: some containers now have a head and grey cells but no target line at all. Those are
#   decoys and must remain unchanged.
#
# a) Concretization and behavior trace
# - Example sketch:
#   1) Blue head above container A and blue target farther above -> A moves upward exactly as in
#      the base task, but only toward the blue line.
#   2) Green head beside container B and green target farther right -> B moves right.
#   3) Red head beside container C, grey cells inside, but no red target anywhere ahead -> C is a
#      decoy and does not move.
# - Relative to the base task, the changed decision point is the matching stage:
#   old rule = "find eligible target line";
#   new rule = "find eligible target line of the head's own color, otherwise stay put".
#
# b) Complexity calibration
# - To keep the color-pairing rule readable, marker colors come from a small fixed palette and are
#   unique per container inside one grid.
# - Each generated grid contains exactly two true movers plus one decoy. One mover and the decoy
#   share the same line shape and lie on the same axis, but the only visible target in front of the
#   decoy is the mover's target of a different color. This makes the color test operational rather
#   than decorative: ignoring color would move the decoy incorrectly.
# - The decoy still has grey cells, so the "missing same-colored target means stay put" rule is
#   visible in every sample instead of relying on occasional edge cases.
#
# c) Implicit-assumption audit of main()
# - Original code assumed any non-bg/non-border marker component of the same color could be a
#   target. That breaks once another container head can share the same representation type.
# - The solver therefore distinguishes:
#   * head components: marker components touching a black border,
#   * external targets: same-colored same-shaped components that do not touch any black border.
# - Gap=0 still needs special handling because head and target can be one connected same-colored
#   component; we split off the border-adjacent slice as the head and treat the remaining
#   non-border-adjacent slice as the target candidate.
# - Decoys must still expose a valid head even though they have no destination, so head picking
#   cannot require an existing target. The generator guarantees only one marker color is adjacent to
#   any container border, making "strongest border adjacency" sufficient.
#
# d) Placement legality
# - The anchor mover and the decoy are placed as one combined group. The anchor keeps its normal
#   head-gap-target corridor; beyond that target, after a short background spacer, sits the decoy
#   with the opposite-facing head. This guarantees the decoy has a legal wrong-color target ahead
#   without overlapping any object.
# - The second mover is placed as a normal standalone assembly with a 1-cell background moat from
#   the special group.
# - Every mover's output remains inside the footprint already reserved in front of its head.
#
# e) Visual observability
# - A human sees that movers pair by identical visible color, not by "any reachable target line",
#   because the decoy has a reachable target-looking line ahead of it but still does not move.
# - The decoy is visually chargeable because it still has grey cells and a head, so the absence of
#   a matching-color distant line is the only cue explaining why it stays still.
# - With exactly two movers plus one decoy, the sample stays readable while still showing both
#   sub-rules directly inside a single grid.
#
# f) Visual cleanliness
# - Geometry stays identical to the source task: only straight 1-cell marker lines are added.
# - Color count stays small and the objects remain separated by background, so the scene is easy to
#   parse without stacked annotations or overlapping overlays.
#
# g) Change inferability
# - Colors are not used numerically or for hidden ordering. They serve only as a visible pairing
#   key between head and target, and the generator ensures that ignoring this key would produce the
#   wrong movement on the chargeable decoy.
# - The palette is small enough that humans can compare colors across 2-4 examples without noise.
#
# New solver assumptions and generator guarantees
# - Assumption: every container has exactly one adjacent marker color touching its border.
#   Guarantee: each assembly draws exactly one head line flush with the container; the moat prevents
#   foreign markers from touching the same border.
# - Assumption: external targets never touch a black border.
#   Guarantee: only heads are drawn adjacent to containers; standalone targets are separated by at
#   least one background cell unless gap=0, in which case the split logic recovers the target slice.
# - Assumption: movers have one legal aligned same-colored target and decoys have none.
#   Guarantee: generator assigns unique colors per container, draws one target only for movers, and
#   validates the finished grid by re-running the same analyzer used by main().
# - Assumption: color really matters for at least one object.
#   Guarantee: the decoy is constructed so that a wrong-color target is legally reachable on its
#   axis, and validation explicitly checks that this target exists while a same-color one does not.

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
MARKER_COLORS = [1, 2, 3, 4, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _dominant_non_border_bg(g: np.ndarray) -> int:
    cnt = Counter(map(int, g.flatten()))
    return int(max((c for c in cnt if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _touches_border(g: np.ndarray, pts: set[tuple[int, int]]) -> bool:
    h, w = g.shape
    for r, c in pts:
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BORDER:
                return True
    return False


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _ordered_greys(grey_coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_coords, key=lambda p: (p[0], p[1]))


def _find_containers(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})
    containers.sort(key=lambda d: d["box"])
    return containers


def _collect_marker_components(g: np.ndarray, bg: int) -> dict[int, list[dict]]:
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    marker_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]
    for col in marker_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append(
                {
                    "pts": pts,
                    "shape": shape,
                    "tl": (int(r0), int(c0)),
                    "touches_border": _touches_border(g, pts),
                }
            )
    return comp_by_color


def _split_head_from_component(
    box_: tuple[int, int, int, int],
    comp_pts: set[tuple[int, int]],
) -> dict | None:
    r0, c0, r1, c1 = box_
    border_pts = list(_rect_border_coords(r0, c0, r1, c1))
    best_d = None
    dist = {}
    for pr, pc in comp_pts:
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
        head_side = 2
    elif hc0 > c1:
        direction = (0, 1)
        head_side = 3
    elif hr1 < r0:
        direction = (-1, 0)
        head_side = 0
    elif hr0 > r1:
        direction = (1, 0)
        head_side = 1
    else:
        direction = (0, 0)
        head_side = -1
    if direction == (0, 0):
        return None

    r00, c00 = int(hr0), int(hc0)
    head_shape = {(int(r - r00), int(c - c00)) for r, c in head_pts}
    return {
        "head_pts": head_pts,
        "head_shape": head_shape,
        "head_tl": (r00, c00),
        "dir": direction,
        "head_side": head_side,
    }


def _aligned_distance(direction: tuple[int, int], head_tl: tuple[int, int], dest_tl: tuple[int, int]) -> int | None:
    tr0 = int(dest_tl[0] - head_tl[0])
    tc0 = int(dest_tl[1] - head_tl[1])
    if direction[0] != 0 and tc0 != 0:
        return None
    if direction[1] != 0 and tr0 != 0:
        return None
    if direction[0] == 1 and tr0 <= 0:
        return None
    if direction[0] == -1 and tr0 >= 0:
        return None
    if direction[1] == 1 and tc0 <= 0:
        return None
    if direction[1] == -1 and tc0 >= 0:
        return None
    return abs(tr0) + abs(tc0)


def _corridor_clear(
    g: np.ndarray,
    bg: int,
    head_pts: set[tuple[int, int]],
    direction: tuple[int, int],
    distance: int,
) -> bool:
    for r, c in head_pts:
        for step in range(1, int(distance)):
            rr = int(r + direction[0] * step)
            cc = int(c + direction[1] * step)
            if int(g[rr, cc]) != bg:
                return False
    return True


def _find_best_destination(
    g: np.ndarray,
    bg: int,
    comp_by_color: dict[int, list[dict]],
    head_info: dict,
) -> dict | None:
    h, w = g.shape
    color = int(head_info["color"])
    head_shape = head_info["head_shape"]
    head_tl = head_info["head_tl"]
    comp_pts = set(head_info["comp_pts"])
    head_pts = set(head_info["head_pts"])
    direction = tuple(head_info["dir"])
    if direction == (0, 0):
        return None

    dests = []
    seen = set()

    for comp in comp_by_color.get(color, []):
        if comp["shape"] != head_shape or comp["pts"] == comp_pts or comp["touches_border"]:
            continue
        dist = _aligned_distance(direction, head_tl, comp["tl"])
        if dist is None:
            continue
        if not _corridor_clear(g, bg, head_pts, direction, dist):
            continue
        key = tuple(sorted(comp["pts"]))
        if key not in seen:
            seen.add(key)
            dests.append(comp)

    rem = comp_pts - head_pts
    if rem:
        mask_rem = np.zeros((h, w), dtype=bool)
        for rr, cc in rem:
            mask_rem[int(rr), int(cc)] = True
        labr, nr = label(mask_rem, structure=STRUCT4)
        for k in range(1, int(nr) + 1):
            pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
            if not pts or _touches_border(g, pts):
                continue
            rs = [r for r, _c in pts]
            cs = [c for _r, c in pts]
            r00, c00 = int(min(rs)), int(min(cs))
            shp = {(int(r - r00), int(c - c00)) for r, c in pts}
            if shp != head_shape:
                continue
            dist = _aligned_distance(direction, head_tl, (r00, c00))
            if dist is None:
                continue
            if not _corridor_clear(g, bg, head_pts, direction, dist):
                continue
            key = tuple(sorted(pts))
            if key not in seen:
                seen.add(key)
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00), "touches_border": False})

    best_dist = None
    best_dest = None
    for dest in dests:
        dist = _aligned_distance(direction, head_tl, dest["tl"])
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_dest = dest
    return best_dest


def _pick_head_for_container(
    box_: tuple[int, int, int, int],
    g: np.ndarray,
    bg: int,
    comp_by_color: dict[int, list[dict]],
) -> dict | None:
    h, w = g.shape
    r0, c0, r1, c1 = box_
    seed_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r, c in _rect_border_coords(r0, c0, r1, c1):
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                val = int(g[rr, cc])
                if val not in (bg, BORDER, GREY, ORANGE):
                    seed_by_color[int(val)].add((int(rr), int(cc)))

    best = None
    best_key = None
    for color, seed_pts in seed_by_color.items():
        for comp in comp_by_color.get(int(color), []):
            if not (comp["pts"] & seed_pts):
                continue
            split = _split_head_from_component(box_, set(comp["pts"]))
            if split is None:
                continue
            key = (len(seed_pts), int(color))
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "color": int(color),
                    "head_pts": set(split["head_pts"]),
                    "head_shape": split["head_shape"],
                    "head_tl": split["head_tl"],
                    "comp_pts": set(comp["pts"]),
                    "dir": split["dir"],
                    "head_side": int(split["head_side"]),
                }
            break
    return best


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = _dominant_non_border_bg(g)
    containers = _find_containers(g)
    assert containers
    comp_by_color = _collect_marker_components(g, bg)

    infos = []
    for cont in containers:
        box_ = cont["box"]
        r0, c0, r1, c1 = box_
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        head_info = _pick_head_for_container(box_, g, bg, comp_by_color)
        dest = None
        distance = None
        steps = 0
        if head_info is not None:
            dest = _find_best_destination(g, bg, comp_by_color, head_info)
            if dest is not None:
                distance = _aligned_distance(head_info["dir"], head_info["head_tl"], dest["tl"])
                steps = min(int(distance), int(len(grey_coords)))
        infos.append(
            {
                "box": box_,
                "grey_coords": grey_coords,
                "head_info": head_info,
                "dest": dest,
                "has_dest": dest is not None,
                "distance": distance,
                "steps": steps,
            }
        )
    return {"grid": g, "bg": bg, "containers": infos, "comp_by_color": comp_by_color}


def _has_closer_wrong_color_target(analysis: dict, cont: dict) -> bool:
    head_info = cont["head_info"]
    dest = cont["dest"]
    distance = cont["distance"]
    if head_info is None or dest is None or distance is None:
        return False

    g = analysis["grid"]
    bg = int(analysis["bg"])
    comp_by_color = analysis["comp_by_color"]
    color = int(head_info["color"])
    head_shape = head_info["head_shape"]
    head_tl = head_info["head_tl"]
    head_pts = set(head_info["head_pts"])
    direction = tuple(head_info["dir"])
    correct_dist = int(distance)

    for other_color, comps in comp_by_color.items():
        if int(other_color) == color:
            continue
        for comp in comps:
            if comp["touches_border"] or comp["shape"] != head_shape:
                continue
            dist = _aligned_distance(direction, head_tl, comp["tl"])
            if dist is None or int(dist) >= correct_dist:
                continue
            if _corridor_clear(g, bg, head_pts, direction, int(dist)):
                return True
    return False


def _has_any_wrong_color_target(analysis: dict, cont: dict) -> bool:
    head_info = cont["head_info"]
    if head_info is None:
        return False

    g = analysis["grid"]
    bg = int(analysis["bg"])
    comp_by_color = analysis["comp_by_color"]
    color = int(head_info["color"])
    head_shape = head_info["head_shape"]
    head_tl = head_info["head_tl"]
    head_pts = set(head_info["head_pts"])
    direction = tuple(head_info["dir"])

    for other_color, comps in comp_by_color.items():
        if int(other_color) == color:
            continue
        for comp in comps:
            if comp["touches_border"] or comp["shape"] != head_shape:
                continue
            dist = _aligned_distance(direction, head_tl, comp["tl"])
            if dist is None:
                continue
            if _corridor_clear(g, bg, head_pts, direction, int(dist)):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only containers whose head has a same-colored aligned destination line.

    Containers with a head but no matching destination are decoys and remain unchanged.
    """
    analysis = _analyze_grid(grid)
    g = analysis["grid"]
    bg = int(analysis["bg"])
    out = g.copy()

    for cont in analysis["containers"]:
        head_info = cont["head_info"]
        if head_info is None or cont["dest"] is None or int(cont["steps"]) <= 0:
            continue

        r0, c0, r1, c1 = cont["box"]
        color = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        direction = tuple(head_info["dir"])
        steps = int(cont["steps"])
        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        to_fill = set(_ordered_greys(cont["grey_coords"], direction)[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == color:
                out[int(r), int(c)] = bg

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color

    return out


def _sample_marker_colors(n_containers: int) -> list[int] | None:
    if n_containers > len(MARKER_COLORS):
        return None
    colors = random.sample(MARKER_COLORS, n_containers)
    if len(set(colors)) != n_containers:
        return None
    return colors


def _assembly_layout(spec: dict) -> tuple[int, int, int, int]:
    oh, ow = int(spec["outer_h"]), int(spec["outer_w"])
    hs = int(spec["head_side"])
    if spec["is_decoy"]:
        pad = int(spec["forward_pad"])
        if hs == 0:
            return pad + 1 + oh, ow, pad + 1, 0
        if hs == 1:
            return oh + 1 + pad, ow, 0, 0
        if hs == 2:
            return oh, pad + 1 + ow, 0, pad + 1
        return oh, ow + 1 + pad, 0, 0

    gap = int(spec["gap"])
    if hs == 0:
        return 1 + gap + 1 + oh, ow, gap + 2, 0
    if hs == 1:
        return oh + 1 + gap + 1, ow, 0, 0
    if hs == 2:
        return oh, 1 + gap + 1 + ow, 0, gap + 2
    return oh, ow + 1 + gap + 1, 0, 0

def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True

def _draw_one(grid: np.ndarray, spec: dict) -> None:
    oh, ow = int(spec["outer_h"]), int(spec["outer_w"])
    hs = int(spec["head_side"])
    inner_h, inner_w = int(spec["inner_h"]), int(spec["inner_w"])
    initial_orange = int(spec["initial_orange"])
    marker_color = int(spec["marker_color"])
    cr = int(spec["cr"])
    cc = int(spec["cc"])

    grid[cr, cc : cc + ow] = BORDER
    grid[cr + oh - 1, cc : cc + ow] = BORDER
    grid[cr : cr + oh, cc] = BORDER
    grid[cr : cr + oh, cc + ow - 1] = BORDER

    fill_order = _interior_fill_order(inner_h, inner_w, hs)
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for idx_cell in range(initial_orange):
        r, c = fill_order[idx_cell]
        interior[r, c] = ORANGE
    grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

    if hs == 0:
        grid[cr - 1, cc : cc + ow] = marker_color
    elif hs == 1:
        grid[cr + oh, cc : cc + ow] = marker_color
    elif hs == 2:
        grid[cr : cr + oh, cc - 1] = marker_color
    else:
        grid[cr : cr + oh, cc + ow] = marker_color

    if spec["is_decoy"]:
        return

    gap = int(spec["gap"])
    if hs == 0:
        grid[cr - 2 - gap, cc : cc + ow] = marker_color
    elif hs == 1:
        grid[cr + oh + 1 + gap, cc : cc + ow] = marker_color
    elif hs == 2:
        grid[cr : cr + oh, cc - 2 - gap] = marker_color
    else:
        grid[cr : cr + oh, cc + ow + 1 + gap] = marker_color


def _draw_input(grid: np.ndarray, specs: list[dict]) -> dict:
    expected_by_box = {}
    for spec in specs:
        _draw_one(grid, spec)
        cr = int(spec["cr"])
        cc = int(spec["cc"])
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        expected_by_box[(cr, cc, cr + oh - 1, cc + ow - 1)] = spec
    return expected_by_box


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for spec in specs:
        if spec["is_decoy"] or int(spec["steps"]) <= 0:
            continue

        oh, ow = int(spec["outer_h"]), int(spec["outer_w"])
        hs = int(spec["head_side"])
        dr, dc = SIDE_TO_DIR[hs]
        inner_h, inner_w = int(spec["inner_h"]), int(spec["inner_w"])
        initial_orange = int(spec["initial_orange"])
        steps = int(spec["steps"])
        marker_color = int(spec["marker_color"])
        cr = int(spec["cr"])
        cc = int(spec["cc"])

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        nr = int(cr + dr * steps)
        nc = int(cc + dc * steps)
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = marker_color
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = marker_color
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = marker_color
        else:
            expected[nr : nr + oh, nc + ow] = marker_color
    return expected


def _build_special_specs(colors: list[int]) -> list[dict] | None:
    axis = random.choice(["horizontal", "vertical"])
    anchor_head = random.choice([2, 3] if axis == "horizontal" else [0, 1])
    decoy_head = {0: 1, 1: 0, 2: 3, 3: 2}[anchor_head]

    inner_h = random.randint(2, 5)
    inner_w = random.randint(2, 4)
    area = inner_h * inner_w
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    anchor_gap = random.randint(1, 5)
    wrong_gap = random.randint(1, min(4, area - 2))
    if wrong_gap + 1 >= area:
        return None

    anchor_grey = random.randint(1, area)
    decoy_grey = random.randint(wrong_gap + 2, area)

    extra_inner_h = random.randint(1, 8)
    extra_inner_w = random.randint(1, 5)
    extra_area = extra_inner_h * extra_inner_w
    extra_gap = random.randint(0, 8)
    extra_grey = random.randint(1, extra_area)

    return [
        {
            "role": "anchor",
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": anchor_head,
            "grey": anchor_grey,
            "initial_orange": area - anchor_grey,
            "gap": anchor_gap,
            "steps": min(anchor_grey, anchor_gap + 1),
            "forward_pad": None,
            "is_decoy": False,
            "marker_color": int(colors[0]),
        },
        {
            "role": "decoy",
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": decoy_head,
            "grey": decoy_grey,
            "initial_orange": area - decoy_grey,
            "gap": None,
            "steps": 0,
            "forward_pad": wrong_gap,
            "is_decoy": True,
            "marker_color": int(colors[1]),
        },
        {
            "role": "extra",
            "inner_h": extra_inner_h,
            "inner_w": extra_inner_w,
            "outer_h": extra_inner_h + 2,
            "outer_w": extra_inner_w + 2,
            "head_side": random.randint(0, 3),
            "grey": extra_grey,
            "initial_orange": extra_area - extra_grey,
            "gap": extra_gap,
            "steps": min(extra_grey, extra_gap + 1),
            "forward_pad": None,
            "is_decoy": False,
            "marker_color": int(colors[2]),
        },
    ]


def _special_group_dims(anchor: dict, decoy: dict) -> tuple[int, int]:
    oh = int(anchor["outer_h"])
    ow = int(anchor["outer_w"])
    gap = int(anchor["gap"])
    wrong_gap = int(decoy["forward_pad"])
    if int(anchor["head_side"]) in (2, 3):
        return oh, 2 * ow + gap + wrong_gap + 3
    return 2 * oh + gap + wrong_gap + 3, ow


def _place_special_specs(specs: list[dict]) -> tuple[int, int] | None:
    anchor = next(spec for spec in specs if spec["role"] == "anchor")
    decoy = next(spec for spec in specs if spec["role"] == "decoy")
    extra = next(spec for spec in specs if spec["role"] == "extra")

    group_h, group_w = _special_group_dims(anchor, decoy)
    extra_h, extra_w, extra_cont_r, extra_cont_c = _assembly_layout(extra)

    total_area = group_h * group_w + extra_h * extra_w
    lo_h = max(16, max(group_h, extra_h) + 2)
    lo_w = max(16, max(group_w, extra_w) + 2)
    hi_h = min(26, max(lo_h, int(total_area**0.5) + 8))
    hi_w = min(26, max(lo_w, int(total_area**0.5) + 8))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    group_r = group_c = None
    for _ in range(300):
        r0 = random.randint(0, grid_h - group_h)
        c0 = random.randint(0, grid_w - group_w)
        if _can_place(occupied, r0, c0, group_h, group_w):
            group_r, group_c = r0, c0
            _reserve(occupied, r0, c0, group_h, group_w)
            break
    if group_r is None or group_c is None:
        return None

    oh = int(anchor["outer_h"])
    ow = int(anchor["outer_w"])
    gap = int(anchor["gap"])
    wrong_gap = int(decoy["forward_pad"])
    head_side = int(anchor["head_side"])

    if head_side == 3:
        anchor["cr"], anchor["cc"] = group_r, group_c
        decoy["cr"], decoy["cc"] = group_r, group_c + ow + gap + wrong_gap + 3
    elif head_side == 2:
        decoy["cr"], decoy["cc"] = group_r, group_c
        anchor["cr"], anchor["cc"] = group_r, group_c + ow + wrong_gap + gap + 3
    elif head_side == 1:
        anchor["cr"], anchor["cc"] = group_r, group_c
        decoy["cr"], decoy["cc"] = group_r + oh + gap + wrong_gap + 3, group_c
    else:
        decoy["cr"], decoy["cc"] = group_r, group_c
        anchor["cr"], anchor["cc"] = group_r + oh + wrong_gap + gap + 3, group_c

    placed = False
    for _ in range(300):
        r0 = random.randint(0, grid_h - extra_h)
        c0 = random.randint(0, grid_w - extra_w)
        if _can_place(occupied, r0, c0, extra_h, extra_w):
            _reserve(occupied, r0, c0, extra_h, extra_w)
            extra["cr"] = int(r0 + extra_cont_r)
            extra["cc"] = int(c0 + extra_cont_c)
            placed = True
            break
    if not placed:
        return None

    return grid_h, grid_w


def _gen_attempt() -> np.ndarray | None:
    colors = _sample_marker_colors(3)
    if colors is None:
        return None

    specs = _build_special_specs(colors)
    if specs is None:
        return None
    grid_shape = _place_special_specs(specs)
    if grid_shape is None:
        return None

    grid_h, grid_w = grid_shape
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected_by_box = _draw_input(grid, specs)

    try:
        analysis = _analyze_grid(grid)
    except Exception:
        return None

    if len(analysis["containers"]) != len(specs):
        return None

    seen_decoy = False
    mover_count = 0
    decoy_with_wrong_color_target = 0
    for info in analysis["containers"]:
        spec = expected_by_box.get(info["box"])
        if spec is None:
            return None
        head_info = info["head_info"]
        if head_info is None:
            return None
        if int(head_info["head_side"]) != int(spec["head_side"]):
            return None
        if int(head_info["color"]) != int(spec["marker_color"]):
            return None

        if spec["is_decoy"]:
            seen_decoy = True
            if info["has_dest"] or int(len(info["grey_coords"])) <= 0 or int(info["steps"]) != 0:
                return None
            if _has_any_wrong_color_target(analysis, info):
                decoy_with_wrong_color_target += 1
        else:
            mover_count += 1
            if not info["has_dest"]:
                return None
            if int(info["distance"]) != int(spec["gap"]) + 1:
                return None
            if int(info["steps"]) != int(spec["steps"]):
                return None

    if mover_count != 2 or not seen_decoy or decoy_with_wrong_color_target != 1:
        return None

    expected = _expected_output(grid, specs)
    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 600 attempts")
