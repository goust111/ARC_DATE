# Variant of 271d71e2
# Dimension: D7 — 干扰项（decoy）设计 (obstacle)
# Change: 0-1个全橙色容器作为干扰项，不移动（灰色像素数=0，步数=0）... → 多个干扰项且具有不同特征

from __future__ import annotations

"""
Design analysis for the D7 variant.

variant_spec:
  core_change:
    - Keep the original mover rule unchanged for valid headed containers.
    - Expand decoys from "0-1 fully-orange headed containers" to "2-3 decoys
      per grid, with at least two different decoy types visible in every grid".
    - Decoy types:
      1. full_orange: normal head+dest markers exist, but the interior has no
         grey cells, so movement steps collapse to 0.
      2. no_head: the container still has a same-shaped destination line on the
         movement axis, but the adjacent head line is missing, so the container
         never acquires a direction and stays put.
      3. empty_frame: only the black border exists inside the assembly; head and
         destination markers are present, but the frame has no grey content to
         charge, so movement steps are again 0.

  a_concretization_and_trace:
    - Pixel form:
      * movers: identical to the original task.
      * full_orange decoy: looks exactly like a normal assembly except the
        interior is already fully orange.
      * no_head decoy: the border+interior exists, a distant maroon target line
        exists on one side, but the 1-cell-thick maroon line adjacent to the
        border is absent.
      * empty_frame decoy: black rectangle with magenta interior, plus normal
        maroon head/dest lines.
    - Flow impact example:
      * Original solver finds containers by connected GREY/ORANGE interiors.
        An empty frame is invisible because its interior is all background.
      * In the variant, container detection must instead start from black
        rectangular border components. Once a box is known, movement is applied
        only if a valid head exists and matched destination exists.
      * Example: an empty frame with a top head and top destination used to be
        skipped before detection. Now it is detected, its grey count is 0, and
        steps = min(distance, 0) = 0, so the output keeps it unchanged.

  b_complexity_calibration:
    - Every generated grid has 1-2 movers plus 2-3 decoys, so the D7 change is
      visually obvious.
    - At least two distinct decoy types appear in each grid; if there are three
      decoys, all three types appear.
    - Marker shapes are kept unique per orientation across assemblies that have
      maroon markers, preventing ambiguous head/destination pairing.

  c_implicit_assumption_audit:
    - Old assumption: every relevant container has GREY/ORANGE interior cells.
      Broken by empty_frame; fixed by black-border-based detection.
    - Old assumption: lack of detected head only happens on malformed noise.
      Broken by no_head decoys; now "head missing" is a legitimate stationary
      branch and must leave the object unchanged.
    - Old assumption: zero-step cases only arise from fully-orange interiors.
      Broken by empty_frame; now zero grey can mean either all orange or all bg.
    - Old erase-and-restamp logic remains safe because a zero-step headed decoy
      is stamped back onto the same coordinates with the same patch.

  d_placement_legality:
    - Assemblies reserve the whole corridor from container through the missing
      or present head slot to the destination line, so input and output both fit
      within the same non-overlapping bounding box.
    - A 1-cell background moat is enforced between assemblies.
    - Movers remain the only objects with positive steps.

  e_visual_observability:
    - From a few examples, a solver can see three kinds of stationary objects:
      "already full", "missing head", and "empty shell".
    - The moving objects still provide the original rule support, so the new
      decoys are contrastive rather than replacing the task structure.

  f_cleanliness:
    - Decoys reuse the same assembly grammar as movers instead of adding new
      colors or decorative marks.
    - The no_head decoy keeps the head slot empty, which is easy to read and
      avoids overlap with the destination line.

  g_color_inference:
    - No new colors are introduced. Color semantics remain: 6 bg, 0 border,
      5 grey charge, 7 orange charged, 9 head/destination markers.

  main_generator_contract:
    - main() assumes every real frame is an isolated black rectangular ring
      whose interior uses only {bg, grey, orange}; generate_input() guarantees
      that.
    - main() assumes marked assemblies with a head have a unique matching marker
      shape along the movement axis; generate_input() guarantees unique marker
      lengths per orientation.
"""

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
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


def _find_containers(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    lab, ncomp = label(g == BORDER, structure=STRUCT4)
    containers = []
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
        if not bool(np.isin(inner, [int(bg), BORDER, GREY, ORANGE]).all()):
            continue
        if np.any(inner == BORDER):
            continue
        containers.append(
            {
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "inner": inner.copy(),
                "shape": (int(H), int(W)),
            }
        )
    containers.sort(key=lambda item: item["box"])
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))
    containers = _find_containers(g, bg)
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

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), BORDER, GREY, ORANGE):
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

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

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
            if 0 <= rrr < H and 0 <= ccc < W:
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

        if marker_mode != "none" and marker_shape in used_marker_shapes:
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
        if marker_mode != "none":
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


def _draw_container(grid: np.ndarray, spec: dict, cr: int, cc: int):
    oh, ow = spec["outer_h"], spec["outer_w"]
    inner_h, inner_w = spec["inner_h"], spec["inner_w"]
    head_side = spec["head_side"]

    grid[cr, cc : cc + ow] = BORDER
    grid[cr + oh - 1, cc : cc + ow] = BORDER
    grid[cr : cr + oh, cc] = BORDER
    grid[cr : cr + oh, cc + ow - 1] = BORDER

    if spec["kind"] == "empty_frame":
        interior = np.full((inner_h, inner_w), BG, dtype=int)
    else:
        fill_order = _interior_fill_order(inner_h, inner_w, head_side)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx in range(spec["initial_orange"]):
            r, c = fill_order[idx]
            interior[r, c] = ORANGE
    grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior


def _draw_markers(grid: np.ndarray, spec: dict, cr: int, cc: int):
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
    min_side = max(10, int(total_area ** 0.5) + 3)
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

    grid = np.full((grid_h, grid_w), BG, dtype=int)
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

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

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
    raise RuntimeError("Failed to generate valid D7 variant after 500 attempts")
