# Combo variant of 271d71e2 (index 18)
# combo_spec: D7/3+D16/1
# Name: 多型干扰项棋盘格背景

from __future__ import annotations

"""
Design analysis for the combo variant.

a) Concretization and behavior trace
   - Base objects stay the same: black rectangular frames, grey/orange interior,
     a 1-cell-thick maroon head line on one side, and a same-shaped maroon
     destination line farther along that axis.
   - D16/1 changes the global background from solid 6 to a 6/8 checkerboard.
     Every vacated source cell must restore the local checker color by parity.
   - D7/3 changes the object set: every grid includes 2-3 decoys chosen from
     three stationary types:
       * full_orange: normal headed container, but no grey charge remains.
       * no_head: border+interior plus destination line, but the adjacent head
         line is missing, so direction cannot be acquired.
       * empty_frame: only the black frame exists; its interior is untouched
         checkerboard background, while head and destination lines still exist.
   - Small trace:
       * Original solver detected containers from GREY/ORANGE connected
         components and erased old positions with a single background color.
       * In this combo, an empty_frame has no GREY/ORANGE at all, so that entry
         point fails; the solver must detect black rectangular rings instead.
       * Suppose a right-moving real container vacates a 4x3 box. The source box
         cannot be flooded with 6; each cell must restore either 6 or 8 from
         its board position. A full_orange decoy with the same marker grammar
         still stays in place because `steps = min(distance, grey_count) = 0`.

b) Complexity calibration
   - Each sample has 1-2 true movers plus 2-3 decoys, so the extra object
     classes are visually obvious rather than rare corner cases.
   - Decoy types are limited to three repeated categories. Every sample shows at
     least two types, and 3-decoy samples show all three.
   - Marker shapes remain unique per orientation among all marked assemblies, so
     head/destination pairing stays unambiguous despite the clutter.

c) Implicit assumption audit against the original solver
   - `bg = dominant non-border color` no longer works because the background has
     two colors. The solver now infers the even/odd checker colors first.
   - Detecting containers from GREY/ORANGE components assumed every relevant
     frame had colored interior content. Empty frames break this; detection now
     starts from black border components and validates rectangular rings whose
     interior uses only `{checker colors, GREY, ORANGE}`.
   - Excluding only one background color when searching marker colors would
     misclassify the second checker color as a marker candidate.
   - Erasing a source rectangle with a scalar background value would destroy the
     checkerboard. Source restoration is now coordinate-aware.
   - Missing heads are no longer malformed noise. `pick_head_for_container()`
     returning `None` is a legitimate stationary branch used by `no_head`.
   - Zero-step cases now arise from two different visible causes: no grey charge
     in full_orange, and no grey content at all in empty_frame.

d) Placement legality
   - Each assembly reserves the entire corridor from destination to container.
     Because movement never exceeds head-to-destination distance, both input and
     output fit inside the same reserved box.
   - A 1-cell moat is enforced between assembly boxes, which prevents black
     borders from merging and keeps the border-component detector valid.
   - The board is checkerboard-filled before any objects are stamped, so empty
     frame interiors and later source restoration are both well defined.

e) Visual observability
   - Across 2-4 ARC examples, a human can see a global 6/8 checkerboard and can
     also see that moved-away regions restore that pattern exactly.
   - The three stationary decoy types are visually distinct: already-full,
     missing-head, and empty-shell. Real movers still display the original
     charge-and-translate rule, so the added clutter is contrastive.

f) Visual cleanliness
   - No new decorative marks or extra colors beyond the checkerboard are added.
   - Empty frames are readable because the black ring cleanly encloses visible
     checkerboard cells instead of mixed overlays.
   - The no_head decoy leaves only one maroon line on its axis, avoiding marker
     collisions with the frame itself.

g) Change inferability
   - Color semantics stay structured: 6/8 are background-only and arranged by a
     simple parity rule; 0 is frame, 5 is available charge, 7 is charged, 9 is
     head/destination. No hidden color lookup is required.

Post-change solver assumptions guaranteed by the generator
   - Background colors are exactly {6, 8} and follow one global checker phase.
   - Every black border component is an isolated rectangular ring with no black
     pixels inside its bbox interior.
   - Assemblies with a present head have exactly one nearest same-shaped marker
     farther along that head axis.
   - At least one mover has positive steps, and all movers stay in-bounds both
     before and after motion.
"""

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, paste


BORDER = 0
GREY = 5
BG_A = 6
ORANGE = 7
BG_B = 8
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
DECOY_TYPES = ("full_orange", "no_head", "empty_frame")


def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _checker_color(r: int, c: int, even_color: int, odd_color: int) -> int:
    return int(even_color if ((int(r) + int(c)) & 1) == 0 else odd_color)


def _paint_checkerboard_rect(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    even_color: int,
    odd_color: int,
) -> None:
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            grid[r, c] = _checker_color(r, c, even_color, odd_color)


def _checkerboard_grid(h: int, w: int, phase: int) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = BG_A if ((r + c + phase) & 1) == 0 else BG_B
    return grid


def _infer_checkerboard_colors(g: np.ndarray) -> tuple[int, int]:
    object_colors = {BORDER, GREY, ORANGE, HEAD_COLOR}
    cnt = Counter(map(int, g.flatten()))
    bg_candidates = [int(c) for c in cnt if int(c) not in object_colors]
    bg_candidates.sort(key=lambda c: (-cnt[c], c))
    if not bg_candidates:
        return BG_A, BG_B
    if len(bg_candidates) == 1:
        only = int(bg_candidates[0])
        return only, only

    a, b = int(bg_candidates[0]), int(bg_candidates[1])
    match_ab = 0
    match_ba = 0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            val = int(g[r, c])
            if val != a and val != b:
                continue
            expected_ab = a if ((r + c) & 1) == 0 else b
            expected_ba = b if ((r + c) & 1) == 0 else a
            match_ab += int(val == expected_ab)
            match_ba += int(val == expected_ba)
    return (a, b) if match_ab >= match_ba else (b, a)


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


def _marker_axis(head_side: int) -> str:
    return "h" if head_side in (0, 1) else "v"


def _marker_length(outer_h: int, outer_w: int, head_side: int) -> int:
    return outer_w if head_side in (0, 1) else outer_h


def _find_containers(g: np.ndarray, bg_colors: set[int]) -> list[dict]:
    h, w = g.shape
    lab, ncomp = label(g == BORDER, structure=STRUCT4)
    containers = []
    allowed_inner = set(int(c) for c in bg_colors) | {GREY, ORANGE}

    for k in range(1, int(ncomp) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        pts = {(int(r), int(c)) for r, c in coords}
        border_pts = set(rect_border_coords(int(r0), int(c0), int(r1), int(c1)))
        if pts != border_pts:
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        if not bool(np.isin(inner, list(allowed_inner)).all()):
            continue
        if np.any(inner == BORDER):
            continue
        containers.append({"box": (int(r0), int(c0), int(r1), int(c1))})

    containers.sort(key=lambda item: item["box"])
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    bg_even, bg_odd = _infer_checkerboard_colors(g)
    bg_colors = {int(bg_even), int(bg_odd)}
    containers = _find_containers(g, bg_colors)
    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude=bg_colors | {BORDER, GREY, ORANGE})]

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

    def pick_head_for_container(box_: tuple[int, int, int, int]) -> dict | None:
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    val = int(g[rr, cc])
                    if val not in bg_colors | {BORDER, GREY, ORANGE}:
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

        border_pts = list(rect_border_coords(r0, c0, r1, c1))
        best_d = None
        dist: dict[tuple[int, int], int] = {}
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
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
                continue
            dests.append(d)

        rem = comp_pts - head_pts
        if rem:
            mask_rem = np.zeros((h, w), dtype=bool)
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
                        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BORDER:
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

        if not dests:
            continue

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
        if best_dest is None:
            continue

        tr0 = int(best_dest["tl"][0] - head_tl[0])
        tc0 = int(best_dest["tl"][1] - head_tl[1])
        steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_coords)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:
            order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
        to_fill = set(order[:steps])

        _paint_checkerboard_rect(out, r0, c0, r1, c1, bg_even, bg_odd)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = _checker_color(int(r), int(c), bg_even, bg_odd)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < h and 0 <= ccc < w:
                out[rrr, ccc] = int(hc)

    return out


def _build_spec(kind: str, used_marker_shapes: set[tuple[str, int]]) -> dict | None:
    for _ in range(200):
        inner_h = random.randint(1, 6)
        inner_w = random.randint(1, 4)
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        area = inner_h * inner_w
        gap = random.randint(0, 8)
        marker_shape = (_marker_axis(head_side), _marker_length(outer_h, outer_w, head_side))

        if kind == "mover":
            grey = random.randint(1, area)
            initial_orange = area - grey
            marker_mode = "head_dest"
        elif kind == "full_orange":
            grey = 0
            initial_orange = area
            marker_mode = "head_dest"
        elif kind == "no_head":
            grey = random.randint(1, area)
            initial_orange = area - grey
            marker_mode = "dest_only"
        elif kind == "empty_frame":
            grey = 0
            initial_orange = 0
            marker_mode = "head_dest"
        else:
            return None

        if marker_shape in used_marker_shapes:
            continue

        h2d = gap + 1
        steps = min(grey, h2d) if kind == "mover" else 0
        spec = {
            "kind": kind,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "grey": grey,
            "initial_orange": initial_orange,
            "steps": steps,
            "marker_mode": marker_mode,
            "marker_shape": marker_shape,
        }
        used_marker_shapes.add(marker_shape)
        return spec
    return None


def _assembly_dims(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    gap = spec["gap"]
    if hs == 0:
        total_h = 1 + gap + 1 + oh
        total_w = ow
        cont_r = gap + 2
        cont_c = 0
    elif hs == 1:
        total_h = oh + 1 + gap + 1
        total_w = ow
        cont_r = 0
        cont_c = 0
    elif hs == 2:
        total_h = oh
        total_w = 1 + gap + 1 + ow
        cont_r = 0
        cont_c = gap + 2
    else:
        total_h = oh
        total_w = ow + 1 + gap + 1
        cont_r = 0
        cont_c = 0
    return {**spec, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c}


def _draw_container(grid: np.ndarray, spec: dict, cr: int, cc: int) -> None:
    oh, ow = spec["outer_h"], spec["outer_w"]
    inner_h, inner_w = spec["inner_h"], spec["inner_w"]
    head_side = spec["head_side"]

    grid[cr, cc : cc + ow] = BORDER
    grid[cr + oh - 1, cc : cc + ow] = BORDER
    grid[cr : cr + oh, cc] = BORDER
    grid[cr : cr + oh, cc + ow - 1] = BORDER

    if spec["kind"] == "empty_frame":
        return

    fill_order = _interior_fill_order(inner_h, inner_w, head_side)
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for idx in range(spec["initial_orange"]):
        r, c = fill_order[idx]
        interior[r, c] = ORANGE
    grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior


def _draw_markers(grid: np.ndarray, spec: dict, cr: int, cc: int) -> None:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    gap = spec["gap"]
    mode = spec["marker_mode"]

    if mode == "head_dest":
        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR

    if mode in ("head_dest", "dest_only"):
        if hs == 0:
            grid[cr - 2 - gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh + 1 + gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 2 - gap] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow + 1 + gap] = HEAD_COLOR


def _gen_attempt() -> np.ndarray | None:
    n_movers = random.randint(1, 2)
    n_decoys = random.randint(2, 3)
    decoy_types = list(random.sample(DECOY_TYPES, k=n_decoys))

    specs = []
    used_marker_shapes: set[tuple[str, int]] = set()
    for _ in range(n_movers):
        spec = _build_spec("mover", used_marker_shapes)
        if spec is None:
            return None
        specs.append(spec)
    for kind in decoy_types:
        spec = _build_spec(kind, used_marker_shapes)
        if spec is None:
            return None
        specs.append(spec)
    random.shuffle(specs)

    assemblies = [_assembly_dims(spec) for spec in specs]
    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(10, int(total_area**0.5) + 3)
    max_side = min(26, min_side + 7)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)
    lo_h = max(10, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 3))
    lo_w = max(10, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 3))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []

    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
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
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + th, c0 : c0 + tw] = True
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    phase = random.randint(0, 1)
    grid = _checkerboard_grid(grid_h, grid_w, phase)

    for a, (pr, pc) in zip(assemblies, placements):
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]
        _draw_container(grid, a, cr, cc)
        _draw_markers(grid, a, cr, cc)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    bg_even = BG_A if (phase & 1) == 0 else BG_B
    bg_odd = BG_B if (phase & 1) == 0 else BG_A

    for a, (pr, pc) in zip(assemblies, placements):
        if a["kind"] != "mover" or a["steps"] == 0:
            continue

        oh, ow = a["outer_h"], a["outer_w"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        hs = a["head_side"]
        steps = a["steps"]
        dr, dc = SIDE_TO_DIR[hs]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        _paint_checkerboard_rect(expected, cr, cc, cr + oh - 1, cc + ow - 1, bg_even, bg_odd)
        if hs == 0:
            _paint_checkerboard_rect(expected, cr - 1, cc, cr - 1, cc + ow - 1, bg_even, bg_odd)
        elif hs == 1:
            _paint_checkerboard_rect(expected, cr + oh, cc, cr + oh, cc + ow - 1, bg_even, bg_odd)
        elif hs == 2:
            _paint_checkerboard_rect(expected, cr, cc - 1, cr + oh - 1, cc - 1, bg_even, bg_odd)
        else:
            _paint_checkerboard_rect(expected, cr, cc + ow, cr + oh - 1, cc + ow, bg_even, bg_odd)

        nr = cr + dr * steps
        nc = cc + dc * steps
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx in range(a["initial_orange"] + steps):
            r, c = fill_order[idx]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr : nr + oh, nc + ow] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 500 attempts")
