# Variant of 271d71e2
# Dimension: D6 — 容器与目标的基数关系 (meta)
# Change: 1:1 — 每个容器恰好对应一条目标线，颜色相同（均为maroon/9）... → 一个容器对应多条目标线 (1:many)
#
# Design analysis for this D6 variant
# a) Concrete behavior:
#    Each container still has a full-edge head line and moves straight forward while
#    turning grey interior cells orange. The change is in front of the head: there
#    are now 2-3 candidate target lines, not one. Head colors are only blue(1) or
#    red(2). The valid target is the opposite color:
#      blue head -> nearest red target on the movement axis
#      red head  -> nearest blue target on the movement axis
#    A farther same-color line is always present, so the original same-color
#    assumption is no longer valid. Example: a blue-headed right-facing container
#    may see a red line 3 cells ahead and a blue line 6 cells ahead; the old logic
#    would move 6, the variant moves 3.
# b) Complexity calibration:
#    The selector palette is intentionally tiny (only 1 and 2) so the mapping is
#    learnable from repeated support across 2-4 containers. Candidate count is kept
#    at 2-3 lines to make the 1:many change obvious without clutter.
# c) Original solver assumptions that break:
#    - destination search grouped by the head color; this is false here because the
#      chosen target color is the opposite of the head color.
#    - "same-color candidate" implied "correct candidate"; now a same-color line is
#      deliberately present as a decoy.
#    - the original task could safely merge "head reaches destination" because both
#      colors matched; here, if the container reaches its target, the moved head
#      overwrites the opposite-colored line. The generator allows this and the
#      expected-output builder models it explicitly.
# d) Placement legality:
#    Each container and all its target lines are generated as one isolated assembly
#    whose bounding box already includes the farthest decoy. Assemblies are placed
#    with a 1-cell background moat, so solver detection remains local and movement
#    stays inside the reserved region.
# e) Visual observability:
#    Every moving sample contains both selector colors among movers when possible,
#    and each mover has at least two visible candidate lines ahead. A farther
#    same-color decoy makes the changed matching rule visible.
# f) Visual cleanliness:
#    Decoy lines are placed farther away than the chosen target, so the container
#    never has to pass through an unused target before reaching the correct one.
#    This avoids mid-path clutter; only the chosen target may be overwritten on
#    exact contact, which remains visually interpretable.
# g) Color inference:
#    The mapping is a fixed two-color swap, not an arbitrary many-color lookup:
#      1 <-> 2
#    This keeps the color rule inferable from a few examples.
from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLORS = (1, 2)
EXTRA_DECOY_COLORS = (8, 9)
HEAD_TO_TARGET = {1: 2, 2: 1}

# Direction mapping: head_side -> (dr, dc) movement direction
# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward the target line chosen by head-color swapping.

    Variant rule:
    - Heads are blue(1) or red(2).
    - Several candidate target lines can appear in front of one container.
    - The chosen target is the nearest aligned line whose color is the opposite
      of the head color: 1 -> 2, 2 -> 1.
    - Steps = min(distance to chosen target, grey cell count).
    - Grey cells become orange from the head-facing side inward.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

    def component_adjacent_to_border(pts: set[tuple[int, int]]) -> bool:
        for r, c in pts:
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(BORDER):
                    return True
        return False

    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()
    marker_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]
    comp_by_color: dict[int, list[dict]] = defaultdict(list)

    for col in marker_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append(
                {
                    "color": int(col),
                    "pts": pts,
                    "shape": shape,
                    "tl": (int(r0), int(c0)),
                    "adj_border": component_adjacent_to_border(pts),
                }
            )

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
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

        coords = np.array(list(comp0["pts"]), dtype=int)
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

        if direction == (0, 0):
            return None

        r00, c00 = int(hr0), int(hc0)
        head_shape = {(int(r - r00), int(c - c00)) for r, c in comp0["pts"]}
        return {
            "color": hc,
            "head_pts": set(comp0["pts"]),
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "dir": direction,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None:
            continue

        hc = int(head_info["color"])
        target_color = int(HEAD_TO_TARGET.get(hc, hc))
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        direction = head_info["dir"]

        best = None
        best_dest = None
        for d in comp_by_color.get(target_color, []):
            if d["adj_border"]:
                continue
            if d["shape"] != head_shape:
                continue
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
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


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


def _assembly_geometry(outer_h: int, outer_w: int, head_side: int, farthest_dist: int) -> tuple[int, int, int, int]:
    if head_side == 0:
        return farthest_dist + 1 + outer_h, outer_w, farthest_dist + 1, 0
    if head_side == 1:
        return outer_h + farthest_dist + 1, outer_w, 0, 0
    if head_side == 2:
        return outer_h, farthest_dist + 1 + outer_w, 0, farthest_dist + 1
    return outer_h, outer_w + farthest_dist + 1, 0, 0


def _line_slice(cr: int, cc: int, outer_h: int, outer_w: int, head_side: int, dist_from_head: int | None):
    if dist_from_head is None:
        if head_side == 0:
            return ("row", cr - 1, slice(cc, cc + outer_w))
        if head_side == 1:
            return ("row", cr + outer_h, slice(cc, cc + outer_w))
        if head_side == 2:
            return ("col", slice(cr, cr + outer_h), cc - 1)
        return ("col", slice(cr, cr + outer_h), cc + outer_w)

    if head_side == 0:
        return ("row", cr - 1 - dist_from_head, slice(cc, cc + outer_w))
    if head_side == 1:
        return ("row", cr + outer_h + dist_from_head, slice(cc, cc + outer_w))
    if head_side == 2:
        return ("col", slice(cr, cr + outer_h), cc - 1 - dist_from_head)
    return ("col", slice(cr, cr + outer_h), cc + outer_w + dist_from_head)


def _paint_line(grid: np.ndarray, geom, color: int) -> None:
    kind, a, b = geom
    if kind == "row":
        grid[int(a), b] = int(color)
    else:
        grid[a, int(b)] = int(color)


def _build_expected(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for item in placements:
        oh, ow = item["outer_h"], item["outer_w"]
        hs = item["head_side"]
        steps = item["steps"]
        if steps == 0:
            continue

        cr, cc = item["container_rc"]
        dr, dc = SIDE_TO_DIR[hs]
        expected[cr : cr + oh, cc : cc + ow] = BG
        _paint_line(expected, _line_slice(cr, cc, oh, ow, hs, None), BG)

        nr = cr + dr * steps
        nc = cc + dc * steps
        fill_order = _interior_fill_order(item["inner_h"], item["inner_w"], hs)
        interior = np.full((item["inner_h"], item["inner_w"]), GREY, dtype=int)
        for idx in range(item["initial_orange"] + steps):
            r, c = fill_order[idx]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch
        _paint_line(expected, _line_slice(nr, nc, oh, ow, hs, None), item["head_color"])
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = 0 if n_containers == 2 else random.choice([0, 1])
    n_movers = n_containers - n_decoys
    if n_movers < 2:
        return None

    mover_colors = [HEAD_COLORS[i % 2] for i in range(n_movers)]
    random.shuffle(mover_colors)

    specs: list[dict] = []
    distance_capped_movers = 0
    for idx in range(n_containers):
        is_decoy = idx >= n_movers
        head_color = mover_colors[idx] if not is_decoy else random.choice(HEAD_COLORS)
        target_color = HEAD_TO_TARGET[head_color]
        head_side = random.randint(0, 3)

        inner_h = random.randint(1, 7)
        inner_w = random.randint(1, 5)
        if not is_decoy and inner_h * inner_w == 1:
            if random.random() < 0.5:
                inner_h += 1
            else:
                inner_w += 1
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        inner_area = inner_h * inner_w

        match_dist = random.randint(1, min(6, inner_area + 2))
        same_dist = match_dist + random.randint(2, 5)
        line_specs = [(match_dist, target_color), (same_dist, head_color)]
        if random.random() < 0.5:
            extra_dist = same_dist + random.randint(2, 4)
            line_specs.append((extra_dist, random.choice(EXTRA_DECOY_COLORS)))

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            steps = 0
        else:
            if inner_area >= match_dist and random.random() < 0.7:
                grey = random.randint(match_dist, inner_area)
                steps = match_dist
                distance_capped_movers += 1
            else:
                if match_dist == 1:
                    grey = random.randint(1, inner_area)
                    steps = 1
                    distance_capped_movers += 1
                else:
                    grey = random.randint(1, min(inner_area, match_dist - 1))
                    steps = grey
            initial_orange = inner_area - grey

        total_h, total_w, cont_r, cont_c = _assembly_geometry(outer_h, outer_w, head_side, max(d for d, _ in line_specs))
        if total_h > 26 or total_w > 26:
            return None

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "head_color": head_color,
                "target_color": target_color,
                "initial_orange": initial_orange,
                "grey": grey,
                "steps": steps,
                "line_specs": line_specs,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
                "is_decoy": is_decoy,
                "match_dist": match_dist,
                "same_dist": same_dist,
            }
        )

    if distance_capped_movers == 0:
        return None

    total_area = sum(s["total_h"] * s["total_w"] for s in specs)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_h = max(s["total_h"] for s in specs)
    max_w = max(s["total_w"] for s in specs)

    lo_h = max(8, max_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []
    for s in specs:
        th, tw = s["total_h"], s["total_w"]
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
    build_records = []
    for s, (pr, pc) in zip(specs, placements):
        oh, ow = s["outer_h"], s["outer_w"]
        cr = pr + s["cont_r"]
        cc = pc + s["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = np.full((s["inner_h"], s["inner_w"]), GREY, dtype=int)
        fill_order = _interior_fill_order(s["inner_h"], s["inner_w"], s["head_side"])
        for idx in range(s["initial_orange"]):
            r, c = fill_order[idx]
            interior[r, c] = ORANGE
        grid[cr + 1 : cr + 1 + s["inner_h"], cc + 1 : cc + 1 + s["inner_w"]] = interior

        _paint_line(grid, _line_slice(cr, cc, oh, ow, s["head_side"], None), s["head_color"])
        for dist, color in s["line_specs"]:
            _paint_line(grid, _line_slice(cr, cc, oh, ow, s["head_side"], dist), color)

        build_records.append(
            {
                **s,
                "container_rc": (cr, cc),
            }
        )

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _build_expected(grid, build_records)
    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
