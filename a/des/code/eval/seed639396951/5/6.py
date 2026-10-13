from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _mode_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    others = g[g != bg]
    if others.size == 0:
        return bg, bg, bg
    grey = _mode_color(others)

    rem = others[others != grey]
    if rem.size == 0:
        return bg, grey, grey
    maroon = _mode_color(rem)
    return int(bg), int(grey), int(maroon)


def _build_lane_masks(grid: np.ndarray, grey: int, maroon: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    open_mask = (g != grey) & (g != maroon)
    lane_masks = [np.zeros((h, w), dtype=bool) for _ in range(3)]

    for r in range(h):
        for c in range(w):
            if not open_mask[r, c]:
                continue

            for off in range(3):
                top = r - off
                if 1 <= top <= h - 4:
                    if (
                        g[top - 1, c] == grey
                        and g[top + 3, c] == grey
                        and bool(open_mask[top : top + 3, c].all())
                    ):
                        lane_masks[off][r, c] = True

                left = c - off
                if 1 <= left <= w - 4:
                    if (
                        g[r, left - 1] == grey
                        and g[r, left + 3] == grey
                        and bool(open_mask[r, left : left + 3].all())
                    ):
                        lane_masks[off][r, c] = True

    return lane_masks


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)

    return comps


def _boundary_distance(shape: tuple[int, int], cell: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    return min(r, c, h - 1 - r, w - 1 - c)


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, grey, maroon = _infer_colors(g)
    if grey == bg or maroon == grey:
        return g.copy()

    lane_masks = _build_lane_masks(g, grey, maroon)
    seed_mask = (g != bg) & (g != grey) & (g != maroon)
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]

    seed_infos: list[tuple[int, int, int, int]] = []
    for r, c in seeds:
        offsets = [off for off in range(3) if lane_masks[off][r, c]]
        if not offsets:
            continue
        seed_infos.append((offsets[0], r, c, int(g[r, c])))

    best_dist = np.full((h, w), 10**9, dtype=int)
    best_color = g.copy()

    by_offset: list[list[tuple[int, int, int]]] = [[] for _ in range(3)]
    for off, r, c, color in seed_infos:
        by_offset[off].append((r, c, color))

    maroon_mask = g == maroon

    for off in range(3):
        mask = lane_masks[off]
        if not bool(mask.any()):
            continue

        seeds_here = {(r, c): color for r, c, color in by_offset[off]}
        if not seeds_here:
            continue

        for comp in _component_cells(mask):
            comp_set = set(comp)
            comp_seeds = [(r, c, seeds_here[(r, c)]) for r, c in comp if (r, c) in seeds_here]
            if not comp_seeds:
                continue

            nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for cell in comp:
                rr, cc = cell
                adj = []
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (rr + dr, cc + dc)
                    if nxt in comp_set:
                        adj.append(nxt)
                nbrs[cell] = adj

            leaves = [cell for cell in comp if len(nbrs[cell]) <= 1]
            maroon_leaves = []
            open_leaves = []
            for rr, cc in leaves:
                touches_maroon = False
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and maroon_mask[nr, nc]:
                        touches_maroon = True
                        break
                if touches_maroon:
                    maroon_leaves.append((rr, cc))
                else:
                    open_leaves.append((rr, cc))

            if open_leaves:
                root = min(open_leaves, key=lambda cell: (_boundary_distance((h, w), cell), cell))
            elif leaves:
                root = min(leaves, key=lambda cell: (_boundary_distance((h, w), cell), cell))
            else:
                root = comp[0]

            parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
            depth: dict[tuple[int, int], int] = {root: 0}
            children: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in comp}
            q = deque([root])
            while q:
                cell = q.popleft()
                for nxt in nbrs[cell]:
                    if nxt in depth:
                        continue
                    parent[nxt] = cell
                    depth[nxt] = depth[cell] + 1
                    children[cell].append(nxt)
                    q.append(nxt)

            for sr, sc, color in comp_seeds:
                stack = [((sr, sc), 0)]
                while stack:
                    cell, dist = stack.pop()
                    rr, cc = cell
                    if dist < best_dist[rr, cc]:
                        best_dist[rr, cc] = dist
                        best_color[rr, cc] = color
                    for nxt in children[cell]:
                        stack.append((nxt, dist + 1))

    out = g.copy()
    paint_mask = best_dist < 10**9
    out[paint_mask] = best_color[paint_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse colored seeds along 3-lane pipes toward maroon dead ends while preserving lane offset.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


BG = 0
GREY = 5
MAROON = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
GEN_ATTEMPTS = 400


def _draw_horizontal(grid: np.ndarray, r: int, c0: int, c1: int) -> None:
    if c0 > c1:
        c0, c1 = c1, c0
    grid[r - 2, c0 : c1 + 1] = GREY
    grid[r + 2, c0 : c1 + 1] = GREY
    interior = grid[r - 1 : r + 2, c0 : c1 + 1]
    interior[interior != GREY] = BG


def _draw_vertical(grid: np.ndarray, c: int, r0: int, r1: int) -> None:
    if r0 > r1:
        r0, r1 = r1, r0
    grid[r0 : r1 + 1, c - 2] = GREY
    grid[r0 : r1 + 1, c + 2] = GREY
    interior = grid[r0 : r1 + 1, c - 1 : c + 2]
    interior[interior != GREY] = BG


def _cap_horizontal(grid: np.ndarray, r: int, c: int) -> None:
    grid[r - 2, c] = GREY
    grid[r + 2, c] = GREY
    grid[r - 1 : r + 2, c] = MAROON


def _cap_vertical(grid: np.ndarray, r: int, c: int) -> None:
    grid[r, c - 2] = GREY
    grid[r, c + 2] = GREY
    grid[r, c - 1 : c + 2] = MAROON


def _pick_spaced(values: list[int], count: int, min_gap: int, rng: random.Random) -> list[int] | None:
    values = list(values)
    rng.shuffle(values)
    picked: list[int] = []
    for v in values:
        if all(abs(v - u) >= min_gap for u in picked):
            picked.append(v)
            if len(picked) == count:
                return sorted(picked)
    return None


def _rand_between(lo: int, hi: int, rng: random.Random) -> int | None:
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _segment_candidates(start: int, end: int, blocked: set[int], margin: int = 2) -> list[int]:
    lo = min(start, end) + margin
    hi = max(start, end) - margin
    return [x for x in range(lo, hi + 1) if x not in blocked]


def _gen_scene(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(17, 29)
    w = rng.randint(17, 29)
    grid = np.full((h, w), BG, dtype=int)

    orient = rng.choice(["horizontal", "vertical"])
    conflict_case = rng.random() < 0.5
    has_branch = rng.random() < 0.5

    hosts: dict[str, dict[str, object]] = {}
    branch_meta: list[tuple[str, int, int, int]] = []

    if orient == "horizontal":
        r = rng.randint(4, h - 5)
        root_side = rng.choice(["left", "right"])

        if root_side == "left":
            leaf_c = rng.randint(max(9, w // 2), w - 3)
            _draw_horizontal(grid, r, 0, leaf_c)
            _cap_horizontal(grid, r, leaf_c + 1)
            trunk_start, trunk_end = 0, leaf_c
            branch_axis_vals = list(range(7, leaf_c - 4))
            downstream_sign = 1
        else:
            leaf_c = rng.randint(2, min(w // 2, w - 10))
            _draw_horizontal(grid, r, leaf_c, w - 1)
            _cap_horizontal(grid, r, leaf_c - 1)
            trunk_start, trunk_end = w - 1, leaf_c
            branch_axis_vals = list(range(leaf_c + 5, w - 7))
            downstream_sign = -1

        n_branches = 0
        if has_branch:
            n_branches = 1 if rng.random() < 0.5 else 2
        branch_cols = _pick_spaced(branch_axis_vals, n_branches, 6, rng) if n_branches else []
        if n_branches and not branch_cols:
            return None

        blocked = set()
        for idx, bc in enumerate(branch_cols):
            side_choices = []
            if r >= 8:
                side_choices.append("up")
            if h - 1 - r >= 8:
                side_choices.append("down")
            if not side_choices:
                return None
            side = rng.choice(side_choices)
            if side == "up":
                top = _rand_between(1, r - 4, rng)
                if top is None:
                    return None
                _draw_vertical(grid, bc, top, r + 1)
                _cap_vertical(grid, top - 1, bc)
                branch_meta.append((f"branch{idx}", bc, top, r + 1))
                blocked.update(range(bc - 1, bc + 2))
                hosts[f"branch{idx}"] = {
                    "orientation": "vertical",
                    "fixed": bc,
                    "positions": list(range(top + 1, r - 1)),
                    "direction": -1,
                }
            else:
                bottom = _rand_between(r + 4, h - 2, rng)
                if bottom is None:
                    return None
                _draw_vertical(grid, bc, r - 1, bottom)
                _cap_vertical(grid, bottom + 1, bc)
                branch_meta.append((f"branch{idx}", bc, r - 1, bottom))
                blocked.update(range(bc - 1, bc + 2))
                hosts[f"branch{idx}"] = {
                    "orientation": "vertical",
                    "fixed": bc,
                    "positions": list(range(r + 2, bottom)),
                    "direction": 1,
                }

        trunk_positions = _segment_candidates(trunk_start, trunk_end, blocked)
        if len(trunk_positions) < 4:
            return None
        hosts["trunk"] = {
            "orientation": "horizontal",
            "fixed": r,
            "positions": trunk_positions,
            "direction": downstream_sign,
        }

    else:
        c = rng.randint(4, w - 5)
        root_side = rng.choice(["top", "bottom"])

        if root_side == "top":
            leaf_r = rng.randint(max(9, h // 2), h - 3)
            _draw_vertical(grid, c, 0, leaf_r)
            _cap_vertical(grid, leaf_r + 1, c)
            trunk_start, trunk_end = 0, leaf_r
            branch_axis_vals = list(range(7, leaf_r - 4))
            downstream_sign = 1
        else:
            leaf_r = rng.randint(2, min(h // 2, h - 10))
            _draw_vertical(grid, c, leaf_r, h - 1)
            _cap_vertical(grid, leaf_r - 1, c)
            trunk_start, trunk_end = h - 1, leaf_r
            branch_axis_vals = list(range(leaf_r + 5, h - 7))
            downstream_sign = -1

        n_branches = 0
        if has_branch:
            n_branches = 1 if rng.random() < 0.5 else 2
        branch_rows = _pick_spaced(branch_axis_vals, n_branches, 6, rng) if n_branches else []
        if n_branches and not branch_rows:
            return None

        blocked = set()
        for idx, br in enumerate(branch_rows):
            side_choices = []
            if c >= 8:
                side_choices.append("left")
            if w - 1 - c >= 8:
                side_choices.append("right")
            if not side_choices:
                return None
            side = rng.choice(side_choices)
            if side == "left":
                left = _rand_between(1, c - 4, rng)
                if left is None:
                    return None
                _draw_horizontal(grid, br, left, c + 1)
                _cap_horizontal(grid, br, left - 1)
                blocked.update(range(br - 1, br + 2))
                hosts[f"branch{idx}"] = {
                    "orientation": "horizontal",
                    "fixed": br,
                    "positions": list(range(left + 1, c - 1)),
                    "direction": -1,
                }
            else:
                right = _rand_between(c + 4, w - 2, rng)
                if right is None:
                    return None
                _draw_horizontal(grid, br, c - 1, right)
                _cap_horizontal(grid, br, right + 1)
                blocked.update(range(br - 1, br + 2))
                hosts[f"branch{idx}"] = {
                    "orientation": "horizontal",
                    "fixed": br,
                    "positions": list(range(c + 2, right)),
                    "direction": 1,
                }

        trunk_positions = _segment_candidates(trunk_start, trunk_end, blocked)
        if len(trunk_positions) < 4:
            return None
        hosts["trunk"] = {
            "orientation": "vertical",
            "fixed": c,
            "positions": trunk_positions,
            "direction": downstream_sign,
        }

    available_colors = list(SEED_COLORS)
    rng.shuffle(available_colors)
    seeds: list[tuple[int, int, int, str, int]] = []

    def place_seed(host_name: str, position: int, offset: int, color: int) -> None:
        host = hosts[host_name]
        if host["orientation"] == "horizontal":
            rr = int(host["fixed"]) - 1 + offset
            cc = int(position)
        else:
            rr = int(position)
            cc = int(host["fixed"]) - 1 + offset
        grid[rr, cc] = color
        seeds.append((rr, cc, color, host_name, offset))

    if conflict_case and available_colors:
        trunk_pos = list(hosts["trunk"]["positions"])
        trunk_dir = int(hosts["trunk"]["direction"])
        if len(trunk_pos) < 3:
            return None

        overlap_target = None
        if any(name.startswith("branch") for name in hosts):
            branch_names = [name for name in hosts if name.startswith("branch") and len(hosts[name]["positions"]) >= 2]
            if branch_names:
                overlap_target = rng.choice(branch_names)

        if overlap_target is not None:
            branch_fixed = int(hosts[overlap_target]["fixed"])
            if hosts["trunk"]["orientation"] == "horizontal":
                upstream = [p for p in trunk_pos if (branch_fixed - p) * trunk_dir > 2]
            else:
                upstream = [p for p in trunk_pos if (branch_fixed - p) * trunk_dir > 2]
            if not upstream:
                return None
            offset = rng.randint(0, 2)
            place_seed("trunk", rng.choice(upstream), offset, available_colors.pop())
            place_seed(overlap_target, rng.choice(list(hosts[overlap_target]["positions"])), offset, available_colors.pop())
        else:
            ordered = sorted(trunk_pos, reverse=trunk_dir < 0)
            if len(ordered) < 5:
                return None
            offset = rng.randint(0, 2)
            place_seed("trunk", ordered[1], offset, available_colors.pop())
            place_seed("trunk", ordered[-2], offset, available_colors.pop())

    target_seed_count = rng.randint(1, min(3, len(hosts)))
    if seeds:
        target_seed_count = max(target_seed_count, len(seeds))

    host_order = list(hosts.keys())
    rng.shuffle(host_order)
    used_hosts = {host for _r, _c, _color, host, _off in seeds}
    while len(seeds) < target_seed_count and available_colors:
        chosen = None
        for host_name in host_order:
            if host_name in used_hosts:
                continue
            positions = list(hosts[host_name]["positions"])
            if positions:
                chosen = (host_name, positions)
                break
        if chosen is None:
            break
        host_name, positions = chosen
        place_seed(host_name, rng.choice(positions), rng.randint(0, 2), available_colors.pop())
        used_hosts.add(host_name)

    if not seeds:
        return None

    counts = Counter(map(int, grid.flatten()))
    if counts.most_common(1)[0][0] != BG:
        return None
    if counts[GREY] <= counts[MAROON]:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    bg2, grey2, maroon2 = _infer_colors(grid)
    if (bg2, grey2, maroon2) != (BG, GREY, MAROON):
        return None

    lane_masks = _build_lane_masks(grid, GREY, MAROON)
    for rr, cc, _color, _host, _off in seeds:
        if sum(int(mask[rr, cc]) for mask in lane_masks) == 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
