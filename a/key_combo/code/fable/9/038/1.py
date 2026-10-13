from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
LIQUID_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                nxt = (nr, nc)
                if _in_bounds(grid.shape, nr, nc) and nxt not in seen and int(grid[nr, nc]) == color:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tube_from_component(
    grid: np.ndarray,
    wall_coords: list[tuple[int, int]],
    wall_color: int,
    bg: int,
) -> dict | None:
    r0, c0, r1, c1 = _bbox(wall_coords)
    corridor = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if int(grid[r, c]) != wall_color
    ]
    if len(corridor) < 2:
        return None

    corridor_set = set(corridor)
    boundary_cells = []
    degrees: dict[tuple[int, int], int] = {}
    for cell in corridor:
        r, c = cell
        deg = sum((r + dr, c + dc) in corridor_set for dr, dc in DIRS4)
        if deg not in (1, 2):
            return None
        degrees[cell] = deg
        if r in (r0, r1) or c in (c0, c1):
            boundary_cells.append(cell)

    if len(boundary_cells) != 1:
        return None
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return None
    opening = boundary_cells[0]
    if opening not in endpoints:
        return None

    open_sides = []
    if opening[0] == r0:
        open_sides.append((-1, 0))
    if opening[0] == r1:
        open_sides.append((1, 0))
    if opening[1] == c0:
        open_sides.append((0, -1))
    if opening[1] == c1:
        open_sides.append((0, 1))
    if len(open_sides) != 1:
        return None
    opening_dir = open_sides[0]
    outside = (opening[0] + opening_dir[0], opening[1] + opening_dir[1])
    if not _in_bounds(grid.shape, *outside) or int(grid[outside]) != bg:
        return None

    ordered = [opening]
    prev = None
    cur = opening
    while True:
        nxts = [
            (cur[0] + dr, cur[1] + dc)
            for dr, dc in DIRS4
            if (cur[0] + dr, cur[1] + dc) in corridor_set and (cur[0] + dr, cur[1] + dc) != prev
        ]
        if not nxts:
            break
        if len(nxts) != 1:
            return None
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    if len(ordered) != len(corridor):
        return None
    if ordered[-1] == opening or ordered[-1] in boundary_cells:
        return None

    colors = [int(grid[r, c]) for r, c in ordered]
    first_filled = next((i for i, color in enumerate(colors) if color != bg), len(colors))
    if any(color == bg for color in colors[first_filled:]):
        return None
    liquid_seq = colors[first_filled:]

    return {
        "bbox": (r0, c0, r1, c1),
        "path": ordered,
        "opening": opening,
        "opening_dir": opening_dir,
        "capacity": len(ordered),
        "filled_start": first_filled,
        "liquid_seq": liquid_seq,
        "filled_len": len(liquid_seq),
    }


def _extract_tubes(grid: np.ndarray) -> tuple[int, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates: list[tuple[int, list[dict]]] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        comps = _components_of_color(g, color)
        if len(comps) != 2:
            continue
        tubes = []
        ok = True
        for comp in comps:
            tube = _tube_from_component(g, comp, color, bg)
            if tube is None:
                ok = False
                break
            tubes.append(tube)
        if not ok:
            continue
        filled_counts = sorted(tube["filled_len"] for tube in tubes)
        if filled_counts == [0, 0] or filled_counts[0] > 0:
            continue
        candidates.append((color, tubes))

    assert len(candidates) == 1, "expected exactly one pair of tube-wall components"
    wall_color, tubes = candidates[0]
    return wall_color, tubes, bg


def _apply_transfer(
    grid: np.ndarray,
    source: dict,
    target: dict,
    bg: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in source["path"]:
        out[r, c] = bg
    for r, c in target["path"]:
        out[r, c] = bg

    seq = list(source["liquid_seq"])
    inside = min(len(seq), target["capacity"])
    if inside:
        dst_cells = target["path"][target["capacity"] - inside :]
        for (r, c), color in zip(dst_cells, reversed(seq[:inside])):
            out[r, c] = color

    dr, dc = target["opening_dir"]
    for step, color in enumerate(seq[inside:], start=1):
        rr = target["opening"][0] + dr * step
        cc = target["opening"][1] + dc * step
        assert _in_bounds(out.shape, rr, cc), "overflow ray left the grid"
        out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pour the source tube into the empty target tube, reversing segment order against the sealed end and extending any overflow straight out of the target opening.

    Input structure:
    - A black background contains exactly two gray wall components, each a solid rectangular shell with a one-cell-wide carved corridor.
    - In each corridor exactly one endpoint is a boundary opening cell, while the other endpoint is sealed inside the tube.
    - One tube is the source: a continuous suffix of corridor cells from the sealed end is filled with colored liquid segments; the other tube is the empty target.

    Rule:
    - Recover each corridor as the unique simple path of non-wall cells inside its wall bounding box, ordered from the opening to the sealed end.
    - Read the source liquid in pouring order from the first filled cell nearest the opening toward the sealed end, clear the source corridor, and drop that sequence into the target so the earliest poured cells settle at the target sealed end.
    - If more liquid remains after the target corridor is full, continue placing it outside the target opening along the opening's outward direction, with the earliest overflow cell adjacent to the opening.

    Key insight:
    - The geometry matters only to identify each tube's path order, sealed end, and opening direction: source liquid exits opening-first, but target storage is indexed from the sealed end, which reverses the visible segment order before any overflow ray continues outward.
    """
    _wall_color, tubes, bg = _extract_tubes(np.asarray(grid, dtype=int))
    source = next(tube for tube in tubes if tube["filled_len"] > 0)
    target = next(tube for tube in tubes if tube["filled_len"] == 0)
    return _apply_transfer(np.asarray(grid, dtype=int), source, target, bg)


def _turns(path: list[tuple[int, int]]) -> int:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return sum(a != b for a, b in zip(dirs, dirs[1:]))


def _on_boundary(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _path_candidates(
    current: tuple[int, int],
    visited: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = current[0] + dr, current[1] + dc
        cand = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if cand in visited:
            continue
        if _on_boundary(cand, h, w):
            continue
        visited_neighbors = [
            (nr + ddr, nc + ddc)
            for ddr, ddc in DIRS4
            if (nr + ddr, nc + ddc) in visited
        ]
        if visited_neighbors != [current] and set(visited_neighbors) != {current}:
            continue
        out.append(cand)
    random.shuffle(out)
    return out


def _build_path(
    h: int,
    w: int,
    opening_side: str,
    length: int,
    require_turn: bool,
) -> list[tuple[int, int]] | None:
    if length > (h - 2) * (w - 2) + 1:
        return None

    if opening_side == "up":
        starts = [(0, c) for c in range(1, w - 1)]
    elif opening_side == "down":
        starts = [(h - 1, c) for c in range(1, w - 1)]
    elif opening_side == "left":
        starts = [(r, 0) for r in range(1, h - 1)]
    else:
        starts = [(r, w - 1) for r in range(1, h - 1)]
    random.shuffle(starts)

    def dfs(path: list[tuple[int, int]], visited: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            if _on_boundary(path[-1], h, w):
                return None
            if require_turn and _turns(path) == 0:
                return None
            return list(path)

        for cand in _path_candidates(path[-1], visited, h, w):
            path.append(cand)
            visited.add(cand)
            ans = dfs(path, visited)
            if ans is not None:
                return ans
            visited.remove(cand)
            path.pop()
        return None

    for start in starts:
        path = [start]
        ans = dfs(path, {start})
        if ans is not None:
            return ans
    return None


def _gen_tube(
    capacity: int,
    opening_side: str,
    fill_seq: list[int] | None,
) -> dict | None:
    feasible_dims = [
        (h, w)
        for h in range(5, 10)
        for w in range(5, 10)
        if capacity <= (h - 2) * (w - 2) + 1
    ]
    random.shuffle(feasible_dims)
    require_turn = capacity >= 5 and random.random() < 0.7
    for h, w in feasible_dims:
        for _ in range(12):
            path = _build_path(h, w, opening_side, capacity, require_turn=require_turn)
            if path is None:
                continue
            grid = np.full((h, w), WALL, dtype=int)
            for r, c in path:
                grid[r, c] = BG
            if len(_components_of_color(grid, WALL)) != 1:
                continue
            if fill_seq:
                start = capacity - len(fill_seq)
                for (r, c), color in zip(path[start:], fill_seq):
                    grid[r, c] = color
            return {
                "local_grid": grid,
                "path_local": path,
                "opening_local": path[0],
                "opening_dir": SIDE_TO_VEC[opening_side],
                "capacity": capacity,
                "size": (h, w),
            }
    return None


def _split_total(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    lengths = []
    for cut in cuts + [total]:
        lengths.append(cut - prev)
        prev = cut
    return lengths


def _make_liquid_sequence(total: int) -> tuple[list[int], list[int]]:
    seg_count = random.randint(2, min(4, total))
    lengths = _split_total(total, seg_count)
    colors = random.sample(LIQUID_COLORS, seg_count)
    seq: list[int] = []
    for color, seg_len in zip(colors, lengths):
        seq.extend([color] * seg_len)
    return seq, colors


def _tube_margins(opening_dir: tuple[int, int], outward_len: int) -> tuple[int, int, int, int]:
    top = bottom = left = right = 1
    if opening_dir == (-1, 0):
        top = outward_len + 1
    elif opening_dir == (1, 0):
        bottom = outward_len + 1
    elif opening_dir == (0, -1):
        left = outward_len + 1
    else:
        right = outward_len + 1
    return top, bottom, left, right


def _all_positions(
    H: int,
    W: int,
    size: tuple[int, int],
    margins: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = size
    top, bottom, left, right = margins
    rows = range(top, H - bottom - h + 1)
    cols = range(left, W - right - w + 1)
    return [(r0, c0) for r0 in rows for c0 in cols]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)


def _place_tubes(
    H: int,
    W: int,
    source_tube: dict,
    target_tube: dict,
    overflow_len: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    target_positions = _all_positions(
        H,
        W,
        target_tube["size"],
        _tube_margins(target_tube["opening_dir"], overflow_len),
    )
    source_positions = _all_positions(
        H,
        W,
        source_tube["size"],
        _tube_margins(source_tube["opening_dir"], 0),
    )
    random.shuffle(target_positions)
    random.shuffle(source_positions)

    for tr0, tc0 in target_positions:
        th, tw = target_tube["size"]
        tbox = (tr0, tc0, tr0 + th - 1, tc0 + tw - 1)
        topen = (tr0 + target_tube["opening_local"][0], tc0 + target_tube["opening_local"][1])
        dr, dc = target_tube["opening_dir"]
        overflow_cells = [(topen[0] + dr * step, topen[1] + dc * step) for step in range(1, overflow_len + 1)]
        if not all(_in_bounds((H, W), r, c) for r, c in overflow_cells):
            continue

        for sr0, sc0 in source_positions:
            sh, sw = source_tube["size"]
            sbox = (sr0, sc0, sr0 + sh - 1, sc0 + sw - 1)
            if _boxes_overlap(_expand(tbox, 1), _expand(sbox, 1)):
                continue
            if any(sbox[0] - 1 <= r <= sbox[2] + 1 and sbox[1] - 1 <= c <= sbox[3] + 1 for r, c in overflow_cells):
                continue
            return (sr0, sc0), (tr0, tc0)
    return None


def _stamp(base: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    base[top : top + h, left : left + w] = patch


def _abs_path(path_local: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in path_local]


def _build_expected_from_scene(scene: dict) -> np.ndarray:
    inp = scene["input"]
    source = scene["source"]
    target = scene["target"]
    source_desc = {
        "path": source["path_abs"],
        "liquid_seq": scene["liquid_seq"],
    }
    target_desc = {
        "path": target["path_abs"],
        "capacity": target["capacity"],
        "opening": target["opening_abs"],
        "opening_dir": target["opening_dir"],
    }
    return _apply_transfer(
        inp,
        {
            "path": source_desc["path"],
            "liquid_seq": source_desc["liquid_seq"],
        },
        target_desc,
        BG,
    )


def _make_scene() -> dict | None:
    H = random.randint(20, 28)
    W = random.randint(20, 28)

    target_capacity = random.randint(4, 12)
    overflow_case = random.random() < 0.55
    if overflow_case:
        total = random.randint(target_capacity + 1, min(15, target_capacity + 4))
    else:
        upper = max(3, target_capacity - 1) if random.random() < 0.75 else target_capacity
        total = random.randint(3, upper)
    seq, segment_colors = _make_liquid_sequence(total)

    source_capacity = random.randint(total + (0 if random.random() < 0.3 else 1), min(16, total + 5))
    source_capacity = max(source_capacity, total)
    if source_capacity == total and random.random() < 0.6 and total < 16:
        source_capacity += 1
    overflow_len = max(0, total - target_capacity)
    inside = min(total, target_capacity)
    if len(set(seq[:inside])) < 2:
        return None

    source_side = random.choice(tuple(SIDE_TO_VEC))
    target_side = random.choice(tuple(SIDE_TO_VEC))

    source_tube = _gen_tube(source_capacity, source_side, seq)
    target_tube = _gen_tube(target_capacity, target_side, None)
    if source_tube is None or target_tube is None:
        return None

    placements = _place_tubes(H, W, source_tube, target_tube, overflow_len)
    if placements is None:
        return None
    (sr0, sc0), (tr0, tc0) = placements

    grid = np.full((H, W), BG, dtype=int)
    _stamp(grid, source_tube["local_grid"], sr0, sc0)
    _stamp(grid, target_tube["local_grid"], tr0, tc0)

    source_path_abs = _abs_path(source_tube["path_local"], sr0, sc0)
    target_path_abs = _abs_path(target_tube["path_local"], tr0, tc0)
    scene = {
        "input": grid,
        "liquid_seq": seq,
        "segment_colors": segment_colors,
        "source": {
            "capacity": source_capacity,
            "path_abs": source_path_abs,
            "opening_abs": (sr0 + source_tube["opening_local"][0], sc0 + source_tube["opening_local"][1]),
            "opening_dir": source_tube["opening_dir"],
        },
        "target": {
            "capacity": target_capacity,
            "path_abs": target_path_abs,
            "opening_abs": (tr0 + target_tube["opening_local"][0], tc0 + target_tube["opening_local"][1]),
            "opening_dir": target_tube["opening_dir"],
        },
    }
    return scene


def _validate_scene(scene: dict) -> bool:
    inp = scene["input"]
    seq = scene["liquid_seq"]
    source = scene["source"]
    target = scene["target"]
    target_cap = target["capacity"]
    inside = min(len(seq), target_cap)
    overflow = len(seq) - inside

    assert len(seq) >= 3
    assert len(scene["segment_colors"]) >= 2
    assert len(set(scene["segment_colors"])) == len(scene["segment_colors"])
    assert inside >= 2
    assert list(reversed(seq[:inside])) != seq[:inside]

    src_colors = [int(inp[r, c]) for r, c in source["path_abs"]]
    src_start = source["capacity"] - len(seq)
    assert all(color == BG for color in src_colors[:src_start])
    assert src_colors[src_start:] == seq
    assert all(int(inp[r, c]) == BG for r, c in target["path_abs"])

    dr, dc = target["opening_dir"]
    for step in range(1, overflow + 1):
        rr = target["opening_abs"][0] + dr * step
        cc = target["opening_abs"][1] + dc * step
        assert _in_bounds(inp.shape, rr, cc)
        assert int(inp[rr, cc]) == BG

    expected = _build_expected_from_scene(scene)
    got = main(inp)
    return np.array_equal(got, expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        scene = _make_scene()
        if scene is None:
            continue
        if _validate_scene(scene):
            return scene["input"]
    raise RuntimeError("failed to generate a valid double-tube transfer input")
