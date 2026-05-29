# Combo variant of 271d71e2 (index 14)
# combo_spec: D9/1+D10/0
# Name: 仅输出条纹内部

"""
Design analysis for the combo variant
=====================================

a) Concretization and behavior trace
------------------------------------
- Base scene objects stay the same as task 271d71e2:
  magenta background, black rectangular container, one maroon head line on a
  border side, and one matching maroon destination line on the same axis.
- D10 changes the interior semantics:
  the inside is not an orange block plus a grey block anymore. Instead it is a
  family of 1-cell stripes orthogonal to motion. The stripe on the head side is
  grey, the next stripe is orange, then grey again, and so on.
- A move step now means:
  1. find the head-to-destination distance as before (`gap + 1`);
  2. cap by the number of grey stripes, not grey pixels;
  3. convert that many whole grey stripes from the head side to orange.
- D9 then changes only the rendering stage:
  after computing each container's post-move state, output only the final
  interior matrix, drop the border/head/destination, and pack all interiors
  vertically in row-major order of the input containers.
- Small trace:
  top-head container, inner size 4x3, base stripes by row:
    [[5,5,5],
     [7,7,7],
     [5,5,5],
     [7,7,7]]
  if head-to-destination distance is 2, the first two grey stripes from the top
  are just row 0 and row 2, so final interior becomes:
    [[7,7,7],
     [7,7,7],
     [7,7,7],
     [7,7,7]]
  Original task would redraw the moved container on the full canvas and would
  have charged only two pixels, not two full stripes. The combo instead outputs
  only the 4x3 final matrix above inside a packed dashboard panel.

b) Complexity calibration
-------------------------
- To keep the stripe rule visible, movers are generated with at least 3 stripe
  layers along the movement axis and at least 3 cells per stripe.
- To keep the crop rule visible, every sample has 2-4 containers and the output
  panel stacks multiple interiors with one magenta separator row.
- Freedom stays controlled:
  one stripe family per container, one fixed packing order, one head color.

c) Implicit-assumption audit of the original solver
---------------------------------------------------
- Container detection over `(GREY | ORANGE)` still works because the interior is
  still a solid rectangle even though colors alternate.
- Head/destination detection and direction inference still work unchanged.
- The original assumptions that break are:
  - `steps` capped by grey pixel count;
  - grey pixels can be sorted individually from the head side;
  - the answer lives on the full input-sized canvas.
- In the combo:
  - `steps` must be capped by grey stripe count;
  - recoloring must happen stripe-by-stripe;
  - the final render must be a packed panel of interiors only.

d) Placement legality
---------------------
- The generator keeps the original assembly layout
  `destination + gap + head + container` on one axis.
- Since `steps <= gap + 1`, the translated box and head stay inside that same
  reserved strip, so non-overlap of input assemblies also guarantees legal
  motion.
- The packed output height is bounded explicitly to stay within 26 rows.

e) Visual observability
-----------------------
- The stripe pattern is visible directly in the input because all movers start
  from the alternating base pattern.
- The output visibly drops the surrounding scene and keeps only the internal
  "readings", so both dimensions of the combo are human-observable.
- We avoid fully-orange decoys because they weaken the D10 signal and can make
  the crop panel look like arbitrary rectangles.

f) Visual cleanliness
---------------------
- Containers are separated by at least one background cell in the input.
- The output panel uses a single fixed layout: vertical stack, left-aligned,
  one background row between readings.
- No extra colors or markers are introduced.

g) Color-change inferability
----------------------------
- Only colors 0, 5, 6, 7, 9 are used.
- Grey-to-orange change is explained structurally by stripe position relative to
  the head, not by hidden color-id ordering.
- Across samples, the stable cue is:
  "the nearest grey stripes to the head disappear one stripe at a time, and the
  output keeps only the final striped dashboard readings."

New solver assumptions introduced by this combo
===============================================
- Interiors are generated as full-width/full-height stripes aligned to motion.
- The stripe nearest the head is grey in the input, then colors alternate.
- Each container has exactly one valid head direction and one aligned
  destination line of matching shape.
- The generator guarantees these assumptions by construction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label


ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "arc_utils.py").exists():
        ROOT = parent
        break
    if (parent / "solvers" / "arc_utils.py").exists():
        ROOT = parent
        break
if ROOT is None:
    raise RuntimeError("Could not locate project root containing arc_utils.py")
for extra in (ROOT, ROOT / "solvers"):
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.append(extra_str)

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


def _stripe_layers(
    inner_h: int,
    inner_w: int,
    *,
    head_side: int | None = None,
    direction: tuple[int, int] | None = None,
) -> list[list[tuple[int, int]]]:
    """Return stripe layers in head-to-far order using local inner coords."""
    if direction is None:
        if head_side is None:
            raise ValueError("Need either head_side or direction")
        direction = SIDE_TO_DIR[int(head_side)]
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return [[(r, c) for c in range(inner_w)] for r in range(inner_h)]
    if (dr, dc) == (1, 0):
        return [[(r, c) for c in range(inner_w)] for r in range(inner_h - 1, -1, -1)]
    if (dr, dc) == (0, -1):
        return [[(r, c) for r in range(inner_h)] for c in range(inner_w)]
    if (dr, dc) == (0, 1):
        return [[(r, c) for r in range(inner_h)] for c in range(inner_w - 1, -1, -1)]
    raise ValueError(f"Unsupported direction: {direction}")


def _build_striped_interior(
    inner_h: int, inner_w: int, head_side: int, converted_grey_stripes: int = 0
) -> np.ndarray:
    """Build the base alternating stripe pattern, then convert the first k grey stripes."""
    interior = np.full((inner_h, inner_w), ORANGE, dtype=int)
    grey_layers = _stripe_layers(inner_h, inner_w, head_side=head_side)[::2]
    for layer in grey_layers:
        for r, c in layer:
            interior[r, c] = GREY
    for layer in grey_layers[:converted_grey_stripes]:
        for r, c in layer:
            interior[r, c] = ORANGE
    return interior


def _grey_stripe_count(inner_h: int, inner_w: int, head_side: int) -> int:
    return len(_stripe_layers(inner_h, inner_w, head_side=head_side)[::2])


def _pack_interiors(readings: list[dict], bg: int) -> np.ndarray:
    assert readings
    total_h = sum(int(item["inner"].shape[0]) for item in readings) + (len(readings) - 1)
    max_w = max(int(item["inner"].shape[1]) for item in readings)
    out = np.full((total_h, max_w), int(bg), dtype=int)
    rr = 0
    for idx, item in enumerate(readings):
        inner = item["inner"]
        h, w = inner.shape
        out[rr : rr + h, :w] = inner
        rr += h
        if idx + 1 < len(readings):
            rr += 1
    return out


def _old_rule_fill_order(
    grey_cells: list[tuple[int, int]], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_cells, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_cells, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_cells, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_cells, key=lambda p: (p[0], p[1]))


def _compute_head_components(g: np.ndarray, bg: int) -> dict[int, list[dict]]:
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in colors_present(g, exclude={bg, BORDER, GREY, ORANGE}):
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})
    return comp_by_color


def _pick_head_for_container(
    g: np.ndarray,
    bg: int,
    comp_by_color: dict[int, list[dict]],
    box_: tuple[int, int, int, int],
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
        "head_pts": head_pts,
        "head_shape": head_shape,
        "head_tl": (r00, c00),
        "comp_pts": set(comp0["pts"]),
        "dir": direction,
    }


def _candidate_destinations(
    g: np.ndarray,
    comp_by_color: dict[int, list[dict]],
    box_: tuple[int, int, int, int],
    head_info: dict,
) -> list[dict]:
    H, W = g.shape
    hc = int(head_info["color"])
    head_shape = head_info["head_shape"]
    head_pts = set(head_info["head_pts"])
    comp_pts = set(head_info["comp_pts"])

    dests = []
    for d in comp_by_color.get(hc, []):
        if d["shape"] != head_shape:
            continue
        if d["pts"] == comp_pts:
            continue
        dests.append(d)

    rem = comp_pts - head_pts
    if not rem:
        return dests

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
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(BORDER):
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
    return dests


def _best_destination(head_info: dict, dests: list[dict]) -> dict | None:
    direction = head_info["dir"]
    head_tl = head_info["head_tl"]
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


def _detect_final_interiors(grid: np.ndarray) -> tuple[int, list[dict]]:
    """
    Return `(bg, readings)` where each reading is the post-move interior only.

    The input scene is still the full original apparatus; only the output format
    changes. The row-major order is by the input container boxes.
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
        containers.append({"box": (r0, c0, r1, c1), "inner": inner.copy()})

    assert containers

    comp_by_color = _compute_head_components(g, bg)

    readings = []
    for cont in containers:
        box_ = cont["box"]
        final_inner = cont["inner"].copy()
        head_info = _pick_head_for_container(g, bg, comp_by_color, box_)
        if head_info is not None and head_info["dir"] != (0, 0):
            best_dest = _best_destination(head_info, _candidate_destinations(g, comp_by_color, box_, head_info))
            if best_dest is not None:
                tr0 = int(best_dest["tl"][0] - head_info["head_tl"][0])
                tc0 = int(best_dest["tl"][1] - head_info["head_tl"][1])
                steps = abs(tr0) + abs(tc0)

                layers = _stripe_layers(final_inner.shape[0], final_inner.shape[1], direction=head_info["dir"])
                grey_layers = [layer for layer in layers if any(int(final_inner[r, c]) == GREY for r, c in layer)]
                steps = min(int(steps), int(len(grey_layers)))
                for layer in grey_layers[:steps]:
                    for r, c in layer:
                        final_inner[int(r), int(c)] = int(ORANGE)

        readings.append({"box": box_, "inner": final_inner})

    readings.sort(key=lambda item: (int(item["box"][0]), int(item["box"][1])))
    return bg, readings


def _old_rule_panel(grid: np.ndarray) -> np.ndarray:
    """
    Build the D9-style packed panel using the original per-cell charging rule.

    The generator uses this to reject degenerate inputs where the stripe-wise combo
    output would accidentally match the source task's cell-wise output.
    """
    g = np.array(grid, dtype=int)
    bg, combo_readings = _detect_final_interiors(g)
    comp_by_color = _compute_head_components(g, bg)

    readings = []
    for item in combo_readings:
        box_ = item["box"]
        r0, c0, r1, c1 = box_
        inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        head_info = _pick_head_for_container(g, bg, comp_by_color, box_)
        if head_info is not None and head_info["dir"] != (0, 0):
            best_dest = _best_destination(head_info, _candidate_destinations(g, comp_by_color, box_, head_info))
            if best_dest is not None:
                tr0 = int(best_dest["tl"][0] - head_info["head_tl"][0])
                tc0 = int(best_dest["tl"][1] - head_info["head_tl"][1])
                steps = abs(tr0) + abs(tc0)
                grey_cells = [
                    (rr, cc)
                    for rr in range(inner.shape[0])
                    for cc in range(inner.shape[1])
                    if int(inner[rr, cc]) == int(GREY)
                ]
                steps = min(int(steps), int(len(grey_cells)))
                for rr, cc in _old_rule_fill_order(grey_cells, head_info["dir"])[:steps]:
                    inner[int(rr), int(cc)] = int(ORANGE)
        readings.append({"box": box_, "inner": inner})

    readings.sort(key=lambda item: (int(item["box"][0]), int(item["box"][1])))
    return _pack_interiors(readings, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the combo variant:
    1. infer the stripe-wise movement state for each container;
    2. output only the final interiors in a packed dashboard panel.
    """
    bg, readings = _detect_final_interiors(np.array(grid, dtype=int))
    return _pack_interiors(readings, bg)


def _choose_inner_dims(head_side: int) -> tuple[int, int]:
    """
    Pick sizes that make stripes visible while keeping the packed panel compact.

    - stripe depth along motion axis >= 3
    - stripe length orthogonal to motion >= 3
    """
    if head_side in (0, 1):
        return random.randint(3, 6), random.randint(3, 5)
    return random.randint(3, 5), random.randint(3, 6)


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)

    specs = []
    for _ in range(n_containers):
        head_side = random.randint(0, 3)
        inner_h, inner_w = _choose_inner_dims(head_side)
        packed_h_if_added = sum(int(s["inner_h"]) for s in specs) + inner_h + len(specs)
        if packed_h_if_added > 26:
            return None

        grey_stripes = _grey_stripe_count(inner_h, inner_w, head_side)
        if grey_stripes <= 1:
            return None
        steps = random.randint(1, grey_stripes - 1)
        gap = steps - 1

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "head_side": head_side,
                "gap": gap,
                "steps": steps,
            }
        )

    packed_h = sum(int(s["inner_h"]) for s in specs) + (n_containers - 1)
    packed_w = max(int(s["inner_w"]) for s in specs)
    if packed_h > 26 or packed_w > 26:
        return None

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]
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
        assemblies.append({**s, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c})

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area**0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
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
            cr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            cr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[cr0:cr1, cc0:cc1].any():
                occupied[r0 : r0 + th, c0 : c0 + tw] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    panel_items = []

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        gap = a["gap"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        steps = a["steps"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = _build_striped_interior(
            inner_h, inner_w, hs, converted_grey_stripes=0
        )

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - gap] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + gap] = HEAD_COLOR

        panel_items.append(
            {
                "box": (cr, cc, cr + oh - 1, cc + ow - 1),
                "inner": _build_striped_interior(inner_h, inner_w, hs, converted_grey_stripes=steps),
            }
        )

    try:
        output = main(grid)
    except Exception:
        return None

    panel_items.sort(key=lambda item: (int(item["box"][0]), int(item["box"][1])))
    expected = _pack_interiors(panel_items, BG)
    if np.array_equal(expected, _old_rule_panel(grid)):
        return None
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
