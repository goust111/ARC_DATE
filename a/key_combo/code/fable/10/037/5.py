from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400


@dataclass(frozen=True)
class Tank:
    r0: int
    c0: int
    r1: int
    c1: int
    gap: tuple[int, int] | None
    is_source: bool = False

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def interior(self) -> tuple[slice, slice]:
        return slice(self.r0 + 1, self.r1), slice(self.c0 + 1, self.c1)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_of(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1))
    coords.extend((r1, c) for c in range(c1, c0 - 1, -1))
    coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _border_mask_for_tank(shape: tuple[int, int], tank: Tank) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[tank.r0, tank.c0 : tank.c1 + 1] = True
    mask[tank.r1, tank.c0 : tank.c1 + 1] = True
    mask[tank.r0 : tank.r1 + 1, tank.c0] = True
    mask[tank.r0 : tank.r1 + 1, tank.c1] = True
    if tank.gap is not None:
        mask[tank.gap] = False
    return mask


def _find_source_candidate(grid: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int, int, int]]:
    h, w = grid.shape
    colors = [int(x) for x in np.unique(grid) if int(x) != bg]
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for fill_color in colors:
        for comp in _mask_components(grid == fill_color):
            r0, c0, r1, c1 = _bbox_of(comp)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area != len(comp) or area < 4:
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == fill_color):
                continue
            tr0, tc0, tr1, tc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
            border_vals = [int(grid[rr, cc]) for rr, cc in _perimeter_coords(tr0, tc0, tr1, tc1)]
            border_palette = {v for v in border_vals if v != bg}
            if len(border_palette) != 1:
                continue
            border_color = next(iter(border_palette))
            if border_color == fill_color:
                continue
            if any(v != border_color for v in border_vals):
                continue
            candidates.append((fill_color, border_color, (tr0, tc0, tr1, tc1)))
    assert len(candidates) == 1, "expected exactly one filled source tank"
    return candidates[0]


def _scan_tanks(grid: np.ndarray, border_color: int, fill_color: int, bg: int) -> list[Tank]:
    h, w = grid.shape
    tanks: list[Tank] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            if int(grid[r0, c0]) != border_color:
                continue
            for r1 in range(r0 + 3, h):
                if int(grid[r1, c0]) != border_color:
                    continue
                for c1 in range(c0 + 3, w):
                    if int(grid[r0, c1]) != border_color or int(grid[r1, c1]) != border_color:
                        continue
                    box = (r0, c0, r1, c1)
                    if box in seen_boxes:
                        continue
                    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
                    inner = sub[1:-1, 1:-1]
                    if np.any(inner == border_color):
                        continue
                    inner_colors = {int(x) for x in np.unique(inner)}
                    if inner_colors == {bg}:
                        is_source = False
                    elif inner_colors == {fill_color}:
                        is_source = True
                    else:
                        continue

                    missing: list[tuple[int, int]] = []
                    bad_border = False
                    for rr, cc in _perimeter_coords(r0, c0, r1, c1):
                        v = int(grid[rr, cc])
                        if v == border_color:
                            continue
                        if v == bg:
                            missing.append((rr, cc))
                        else:
                            bad_border = True
                            break
                    if bad_border:
                        continue
                    if len(missing) > 1:
                        continue
                    if is_source and missing:
                        continue
                    gap = None
                    if missing:
                        gr, gc = missing[0]
                        if (gr, gc) in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                            continue
                        gap = (gr, gc)

                    tanks.append(Tank(r0, c0, r1, c1, gap, is_source))
                    seen_boxes.add(box)
    assert len([t for t in tanks if t.is_source]) == 1, "expected one source tank after scan"
    return sorted(tanks, key=lambda t: (t.r0, t.c0, t.r1, t.c1))


def _pipe_graph(grid: np.ndarray, border_color: int, tanks: list[Tank]) -> tuple[list[set[int]], list[set[int]]]:
    pipe_mask = grid == border_color
    for tank in tanks:
        pipe_mask &= ~_border_mask_for_tank(grid.shape, tank)

    comps = _mask_components(pipe_mask)
    comp_id = -np.ones(grid.shape, dtype=int)
    for cid, comp in enumerate(comps):
        for r, c in comp:
            comp_id[r, c] = cid

    tank_to_comps = [set() for _ in tanks]
    comp_to_tanks = [set() for _ in comps]
    h, w = grid.shape
    for tid, tank in enumerate(tanks):
        touches = set()
        for c in range(tank.c0, tank.c1 + 1):
            if tank.r0 - 1 >= 0 and comp_id[tank.r0 - 1, c] >= 0:
                touches.add(int(comp_id[tank.r0 - 1, c]))
            if tank.r1 + 1 < h and comp_id[tank.r1 + 1, c] >= 0:
                touches.add(int(comp_id[tank.r1 + 1, c]))
        for r in range(tank.r0, tank.r1 + 1):
            if tank.c0 - 1 >= 0 and comp_id[r, tank.c0 - 1] >= 0:
                touches.add(int(comp_id[r, tank.c0 - 1]))
            if tank.c1 + 1 < w and comp_id[r, tank.c1 + 1] >= 0:
                touches.add(int(comp_id[r, tank.c1 + 1]))
        tank_to_comps[tid] = touches
        for cid in touches:
            comp_to_tanks[cid].add(tid)
    return tank_to_comps, comp_to_tanks


def _draw_leak(out: np.ndarray, tank: Tank, liquid_color: int, bg: int) -> None:
    assert tank.gap is not None
    r, c = tank.gap
    rr = r + 1
    while rr < out.shape[0] and int(out[rr, c]) == bg:
        out[rr, c] = liquid_color
        rr += 1


def _propagate(grid: np.ndarray, tanks: list[Tank], tank_to_comps: list[set[int]], comp_to_tanks: list[set[int]], liquid_color: int, bg: int) -> tuple[np.ndarray, set[int]]:
    out = np.array(grid, copy=True, dtype=int)
    source_id = next(i for i, tank in enumerate(tanks) if tank.is_source)

    seen = {source_id}
    q = deque([source_id])
    while q:
        tid = q.popleft()
        tank = tanks[tid]
        if tank.gap is not None:
            _draw_leak(out, tank, liquid_color, bg)
            continue

        if not tank.is_source:
            rs, cs = tank.interior
            out[rs, cs] = liquid_color

        for cid in tank_to_comps[tid]:
            for nid in comp_to_tanks[cid]:
                if nid not in seen:
                    seen.add(nid)
                    q.append(nid)
    return out, seen


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate the liquid from the single full tank through the pipe network, filling only closed tanks while gap tanks leak downward and terminate their branch.

    Input structure:
    - A black background contains several axis-aligned rectangular tanks drawn in one shared border color.
    - Straight one-pixel-wide pipes of the same color touch tank exteriors and connect tanks into a network.
    - Exactly one closed tank already has its whole interior filled with the liquid color; other tank interiors are empty.
    - Some tanks are fully closed rectangles, while others have exactly one missing border cell.

    Rule:
    - Treat each tank as a node and the exterior pipe runs as the links between nodes.
    - Starting from the unique filled source tank, propagate the liquid through the connected pipe network.
    - When the liquid reaches a closed tank, fill its entire interior with the liquid color and continue propagating through that tank's other pipes.
    - When the liquid reaches a tank with a border gap, do not fill that tank; instead draw a vertical leak directly below the gap until the grid bottom or the cell above the first non-background obstacle, and stop propagation past that tank.

    Key insight:
    - The missing border cell is both a leak outlet and a gate: connectivity in the border-color pixels is not enough, because a gap tank blocks transfer to any downstream tanks even though the surrounding pipes still look connected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    liquid_color, border_color, _source_box = _find_source_candidate(g, bg)
    tanks = _scan_tanks(g, border_color, liquid_color, bg)
    tank_to_comps, comp_to_tanks = _pipe_graph(g, border_color, tanks)
    out, _ = _propagate(g, tanks, tank_to_comps, comp_to_tanks, liquid_color, bg)
    return out


def _draw_tank(grid: np.ndarray, tank: Tank, border_color: int, liquid_color: int, bg: int) -> None:
    grid[tank.r0, tank.c0 : tank.c1 + 1] = border_color
    grid[tank.r1, tank.c0 : tank.c1 + 1] = border_color
    grid[tank.r0 : tank.r1 + 1, tank.c0] = border_color
    grid[tank.r0 : tank.r1 + 1, tank.c1] = border_color
    if tank.gap is not None:
        grid[tank.gap] = bg
    if tank.is_source:
        rs, cs = tank.interior
        grid[rs, cs] = liquid_color


def _draw_hpipe(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    if c0 > c1:
        c0, c1 = c1, c0
    grid[row, c0 : c1 + 1] = color


def _draw_vpipe(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    if r0 > r1:
        r0, r1 = r1, r0
    grid[r0 : r1 + 1, col] = color


def _shift_tank(tank: Tank, dr: int, dc: int) -> Tank:
    gap = None if tank.gap is None else (tank.gap[0] + dr, tank.gap[1] + dc)
    return Tank(tank.r0 + dr, tank.c0 + dc, tank.r1 + dr, tank.c1 + dc, gap, tank.is_source)


def _mirror_tank(tank: Tank, width: int) -> Tank:
    gap = None if tank.gap is None else (tank.gap[0], width - 1 - tank.gap[1])
    return Tank(tank.r0, width - 1 - tank.c1, tank.r1, width - 1 - tank.c0, gap, tank.is_source)


def _scene_attempt() -> tuple[np.ndarray, dict] | None:
    border_color, liquid_color = random.sample(COLOR_POOL, 2)

    hs, ws = random.randint(4, 6), random.randint(4, 6)
    ha, wa = random.randint(4, 6), random.randint(4, 6)
    hg, wg = random.randint(4, 6), random.randint(4, 6)
    hb, wb = random.randint(4, 6), random.randint(4, 6)
    hd, wd = random.randint(4, 6), random.randint(4, 6)
    hx, wx = random.randint(4, 6), random.randint(4, 6)

    top_gap_1 = random.randint(2, 4)
    top_gap_2 = random.randint(2, 4)
    row_gap_1 = random.randint(4, 6)
    row_gap_2 = random.randint(4, 6)

    s = Tank(1, 1, hs, ws, None, True)
    a = Tank(1, s.c1 + top_gap_1, ha, s.c1 + top_gap_1 + wa - 1, None, False)
    g_gap_off = random.randint(1, wg - 2)
    g = Tank(1, a.c1 + top_gap_2, hg, a.c1 + top_gap_2 + wg - 1, None, False)
    g = Tank(g.r0, g.c0, g.r1, g.c1, (g.r1, g.c0 + g_gap_off), False)

    mid_row = max(s.r1, a.r1, g.r1) + row_gap_1
    b = Tank(mid_row, max(0, s.c0 + random.randint(-1, 1)), mid_row + hb - 1, max(0, s.c0 + random.randint(-1, 1)) + wb - 1, None, False)
    x_kind_gap = bool(random.getrandbits(1))
    x_row = max(b.r1, g.r1) + row_gap_2
    x_left = max(0, a.c0 + random.randint(-1, 2))
    x_gap = None
    if x_kind_gap:
        x_gap = (x_row + hx - 1, x_left + random.randint(1, wx - 2))
    x = Tank(x_row, x_left, x_row + hx - 1, x_left + wx - 1, x_gap, False)

    leak_hits_obstacle = bool(random.getrandbits(1))
    d_row = mid_row
    gap_col = g.gap[1]
    if leak_hits_obstacle:
        left_lo = max(g.c0 - wd + 1, gap_col - wd + 1)
        left_hi = min(g.c1, gap_col)
        if left_lo > left_hi:
            return None
        d_left = random.randint(left_lo, left_hi)
    else:
        d_left = gap_col + random.randint(2, 4)
    d = Tank(d_row, d_left, d_row + hd - 1, d_left + wd - 1, None, False)

    tanks_by_name = {"S": s, "A": a, "G": g, "B": b, "D": d, "X": x}

    boxes = list(tanks_by_name.values())
    for i, t1 in enumerate(boxes):
        for t2 in boxes[i + 1 :]:
            if not (t1.r1 + 1 < t2.r0 or t2.r1 + 1 < t1.r0 or t1.c1 + 1 < t2.c0 or t2.c1 + 1 < t1.c0):
                return None

    shared_cols = list(range(max(s.c0 + 1, b.c0 + 1), min(s.c1 - 1, b.c1 - 1) + 1))
    if not shared_cols:
        return None
    s_b_col = random.choice(shared_cols)
    top_pipe_row = 1 + random.randint(1, min(s.height, a.height, g.height) - 2)
    gd_trunk_col = max(g.c1, d.c1) + 1

    local_h = x.r1 + 3
    local_w = max(gd_trunk_col, d.c1, x.c1) + 3
    if local_h > 30 or local_w > 30:
        return None

    grid = np.full((local_h, local_w), BG, dtype=int)
    for tank in tanks_by_name.values():
        _draw_tank(grid, tank, border_color, liquid_color, BG)

    _draw_hpipe(grid, top_pipe_row, s.c1 + 1, a.c0 - 1, border_color)
    _draw_hpipe(grid, top_pipe_row, a.c1 + 1, g.c0 - 1, border_color)
    _draw_vpipe(grid, s_b_col, s.r1 + 1, b.r0 - 1, border_color)

    g_port_row = g.r0 + random.randint(1, g.height - 2)
    d_port_row = d.r0 + random.randint(1, d.height - 2)
    _draw_hpipe(grid, g_port_row, g.c1 + 1, gd_trunk_col, border_color)
    _draw_vpipe(grid, gd_trunk_col, g_port_row, d_port_row, border_color)
    _draw_hpipe(grid, d_port_row, d.c1 + 1, gd_trunk_col, border_color)

    edges = {
        "S": {"A", "B"},
        "A": {"S", "G"},
        "B": {"S"},
        "G": {"A", "D"},
        "D": {"G"},
        "X": set(),
    }

    if random.random() < 0.5:
        grid = grid[:, ::-1].copy()
        w = grid.shape[1]
        tanks_by_name = {name: _mirror_tank(tank, w) for name, tank in tanks_by_name.items()}

    pts = np.argwhere(grid != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    cropped = grid[r0 : r1 + 1, c0 : c1 + 1]
    tanks_by_name = {name: _shift_tank(tank, -int(r0), -int(c0)) for name, tank in tanks_by_name.items()}

    min_bottom_pad = 2 if tanks_by_name["X"].gap is not None else 0
    free_h = 30 - cropped.shape[0]
    free_w = 30 - cropped.shape[1]
    if free_h < min_bottom_pad or free_w < 0:
        return None

    top_pad = random.randint(0, free_h - min_bottom_pad)
    bottom_pad = random.randint(min_bottom_pad, free_h - top_pad)
    left_pad = random.randint(0, free_w)
    right_pad = random.randint(0, free_w - left_pad)

    final = np.full((cropped.shape[0] + top_pad + bottom_pad, cropped.shape[1] + left_pad + right_pad), BG, dtype=int)
    final[top_pad : top_pad + cropped.shape[0], left_pad : left_pad + cropped.shape[1]] = cropped
    tanks_by_name = {name: _shift_tank(tank, top_pad, left_pad) for name, tank in tanks_by_name.items()}

    meta = {
        "border_color": border_color,
        "liquid_color": liquid_color,
        "tanks": tanks_by_name,
        "edges": edges,
        "leak_hits_obstacle": leak_hits_obstacle,
        "isolated_gap": tanks_by_name["X"].gap is not None,
    }
    return final, meta


def _simulate_from_meta(grid: np.ndarray, meta: dict) -> tuple[np.ndarray, set[str], set[str]]:
    tanks = meta["tanks"]
    edges = meta["edges"]
    liquid = meta["liquid_color"]

    reached = {"S"}
    q = deque(["S"])
    while q:
        name = q.popleft()
        tank = tanks[name]
        if tank.gap is not None:
            continue
        for nxt in edges[name]:
            if nxt not in reached:
                reached.add(nxt)
                q.append(nxt)

    ungated = {"S"}
    q2 = deque(["S"])
    while q2:
        name = q2.popleft()
        for nxt in edges[name]:
            if nxt not in ungated:
                ungated.add(nxt)
                q2.append(nxt)

    tank_list = [tanks[name] for name in sorted(tanks)]
    tank_index = {name: idx for idx, name in enumerate(sorted(tanks))}
    tank_to_comps = [set() for _ in tank_list]
    comp_to_tanks: list[set[int]] = []
    for a_name, bs in edges.items():
        for b_name in bs:
            if a_name > b_name:
                continue
            comp_to_tanks.append({tank_index[a_name], tank_index[b_name]})
    cid = 0
    seen_pairs = set()
    for a_name, bs in edges.items():
        for b_name in bs:
            key = tuple(sorted((a_name, b_name)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            tank_to_comps[tank_index[a_name]].add(cid)
            tank_to_comps[tank_index[b_name]].add(cid)
            cid += 1

    expected, _ = _propagate(grid, tank_list, tank_to_comps, comp_to_tanks, liquid, BG)
    return expected, reached, ungated


def _validate_scene(grid: np.ndarray, meta: dict) -> bool:
    expected, reached, ungated = _simulate_from_meta(grid, meta)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    tanks = meta["tanks"]
    liquid = meta["liquid_color"]
    border = meta["border_color"]

    if "D" not in ungated or "D" in reached:
        return False
    if "G" not in reached or tanks["G"].gap is None:
        return False
    if "A" not in reached or "B" not in reached:
        return False

    a = tanks["A"]
    b = tanks["B"]
    d = tanks["D"]
    ars, acs = a.interior
    brs, bcs = b.interior
    drs, dcs = d.interior
    if not np.all(actual[ars, acs] == liquid):
        return False
    if not np.all(actual[brs, bcs] == liquid):
        return False
    if not np.all(actual[drs, dcs] == BG):
        return False

    g_gap = tanks["G"].gap
    assert g_gap is not None
    leak_len = 0
    rr = g_gap[0] + 1
    while rr < actual.shape[0] and int(actual[rr, g_gap[1]]) == liquid:
        leak_len += 1
        rr += 1
    if leak_len == 0:
        return False

    if meta["leak_hits_obstacle"]:
        if rr >= actual.shape[0] or int(grid[rr, g_gap[1]]) == BG:
            return False
    else:
        if rr != actual.shape[0]:
            return False

    x = tanks["X"]
    xrs, xcs = x.interior
    if x.gap is None:
        if not np.all(actual[xrs, xcs] == BG):
            return False
    else:
        if np.any(actual[x.gap[0] + 1 :, x.gap[1]] == liquid):
            return False

    non_bg = {int(x) for x in np.unique(grid)} - {BG}
    if non_bg != {border, liquid}:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _scene_attempt()
        if built is None:
            continue
        grid, meta = built
        if _validate_scene(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid leaking-tank input")
