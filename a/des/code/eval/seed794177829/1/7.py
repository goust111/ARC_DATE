import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SYMBOL_WALL = -1


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _parse_path_component(grid: np.ndarray, color: int, cells: list[tuple[int, int]]) -> dict:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    assert 2 <= len(cells) <= 4, "path segment must have length 2-4"

    if len(rows) == 1:
        r = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert cols == list(range(c0, c1 + 1)), "horizontal path must be contiguous"
        left_blocked = not _in_bounds(grid.shape, r, c0 - 1) or grid[r, c0 - 1] != BG
        right_blocked = not _in_bounds(grid.shape, r, c1 + 1) or grid[r, c1 + 1] != BG
        assert left_blocked != right_blocked, "path must have exactly one blocked tail side"
        if left_blocked:
            head = (r, c1)
            dir_idx = 1
        else:
            head = (r, c0)
            dir_idx = 3
    else:
        assert len(cols) == 1, "path segment must be straight"
        c = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert rows == list(range(r0, r1 + 1)), "vertical path must be contiguous"
        up_blocked = not _in_bounds(grid.shape, r0 - 1, c) or grid[r0 - 1, c] != BG
        down_blocked = not _in_bounds(grid.shape, r1 + 1, c) or grid[r1 + 1, c] != BG
        assert up_blocked != down_blocked, "path must have exactly one blocked tail side"
        if up_blocked:
            head = (r1, c)
            dir_idx = 2
        else:
            head = (r0, c)
            dir_idx = 0

    return {
        "color": int(color),
        "head": head,
        "dir_idx": int(dir_idx),
        "active": True,
    }


def _parse_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid is too small"
    assert np.any(g != BG), "expected foreground objects"

    path_states: list[dict] = []
    dye_pools: dict[tuple[int, int], int] = {}
    path_colors: set[int] = set()
    dye_colors: set[int] = set()

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, WALL)):
        for cells in _components(g == color):
            if len(cells) == 1:
                pos = cells[0]
                assert pos not in dye_pools, "duplicate dye position"
                dye_pools[pos] = color
                dye_colors.add(color)
            else:
                path_states.append(_parse_path_component(g, color, cells))
                path_colors.add(color)

    assert path_states, "expected at least one path segment"
    assert path_colors.isdisjoint(dye_colors), "dye colors must differ from path colors"
    return g, dye_pools, path_states


def _choose_move(grid: np.ndarray, state: dict):
    r, c = state["head"]
    dir_idx = state["dir_idx"]

    for delta, tag in ((0, "straight"), (1, "right"), (-1, "left")):
        ndir = (dir_idx + delta) % 4
        dr, dc = DIRS[ndir]
        nr, nc = r + dr, c + dc
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BG:
            if delta == 0:
                return nr, nc, ndir, tag
            return nr, nc, ndir, tag
        if delta == 0:
            break

    for delta, tag in ((1, "right"), (-1, "left")):
        ndir = (dir_idx + delta) % 4
        dr, dc = DIRS[ndir]
        nr, nc = r + dr, c + dc
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BG:
            return nr, nc, ndir, tag
    return None


def _simulate(grid: np.ndarray, dye_pools: dict[tuple[int, int], int], path_states: list[dict], record: bool):
    out = grid.copy()
    traces = [
        {
            "steps": 0,
            "right_turn": False,
            "left_turn": False,
            "color_change": False,
            "blocked_stop": False,
            "collision": False,
        }
        for _ in path_states
    ]

    while any(state["active"] for state in path_states):
        proposals: dict[int, tuple[int, int, int, str]] = {}
        for idx, state in enumerate(path_states):
            if not state["active"]:
                continue
            move = _choose_move(out, state)
            if move is None:
                state["active"] = False
                traces[idx]["blocked_stop"] = True
            else:
                proposals[idx] = move

        if not proposals:
            break

        target_counts: dict[tuple[int, int], int] = {}
        for nr, nc, _, _ in proposals.values():
            target_counts[(nr, nc)] = target_counts.get((nr, nc), 0) + 1

        survivors: list[tuple[int, int, int, int, str]] = []
        for idx, move in proposals.items():
            nr, nc, ndir, tag = move
            if target_counts[(nr, nc)] > 1:
                path_states[idx]["active"] = False
                traces[idx]["collision"] = True
                continue
            survivors.append((idx, nr, nc, ndir, tag))

        for idx, nr, nc, _, _ in survivors:
            out[nr, nc] = path_states[idx]["color"]

        for idx, nr, nc, ndir, tag in survivors:
            state = path_states[idx]
            state["head"] = (nr, nc)
            state["dir_idx"] = ndir
            traces[idx]["steps"] += 1
            if tag == "right":
                traces[idx]["right_turn"] = True
            elif tag == "left":
                traces[idx]["left_turn"] = True

            adjacent_pools = {
                dye_pools[(ar, ac)]
                for ar, ac in _neighbors4(nr, nc)
                if (ar, ac) in dye_pools
            }
            if adjacent_pools:
                assert len(adjacent_pools) == 1, "growth cell cannot touch multiple dye colors"
                next_color = adjacent_pools.pop()
                if next_color != state["color"]:
                    traces[idx]["color_change"] = True
                state["color"] = next_color

    return out, traces


def _solve(grid: np.ndarray, record: bool = False):
    g, dye_pools, path_states = _parse_grid(grid)
    return _simulate(g, dye_pools, path_states, record=record)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each short path one step per round with right-turn priority at blockers,
    recoloring later growth after passing next to a dye pool, and resolving same-cell
    simultaneous claims by stopping all claimants.
    """
    out, _ = _solve(grid, record=False)
    return out


def _dye_candidates(template: np.ndarray, future_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = template.shape
    candidates = set()
    corridor_mask = template >= 0
    for r, c in future_cells:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if template[nr, nc] != SYMBOL_WALL:
                continue
            adj_open = 0
            for ar, ac in _neighbors4(nr, nc):
                if 0 <= ar < h and 0 <= ac < w and corridor_mask[ar, ac]:
                    adj_open += 1
            if adj_open == 1:
                candidates.add((nr, nc))
    return sorted(candidates)


def _attach_optional_dye(template: np.ndarray, future_cells: list[tuple[int, int]], add_dye: bool) -> np.ndarray:
    if not add_dye:
        return template
    candidates = _dye_candidates(template, future_cells)
    assert candidates, "expected at least one dye candidate"
    rr, cc = random.choice(candidates)
    out = template.copy()
    out[rr, cc] = 11
    return out


def _make_straight_gadget(add_dye: bool) -> dict:
    seg_len = random.randint(2, 4)
    travel = random.randint(2, 5)
    height = 5
    width = seg_len + travel + 2
    row = 2
    grid = np.full((height, width), SYMBOL_WALL, dtype=int)
    grid[row, 1 : width - 1] = BG
    grid[row, 1 : seg_len + 1] = 1
    future = [(row, c) for c in range(seg_len + 1, width - 1)]
    grid = _attach_optional_dye(grid, future, add_dye)
    return {"grid": grid, "kind": "straight", "needs_dye": add_dye}


def _make_right_turn_gadget(add_dye: bool) -> dict:
    seg_len = random.randint(2, 4)
    htravel = random.randint(1, 4)
    vtravel = random.randint(2, 5)
    row = 1
    corner = seg_len + htravel
    height = vtravel + 3
    width = corner + 2
    grid = np.full((height, width), SYMBOL_WALL, dtype=int)
    grid[row, 1 : corner + 1] = BG
    grid[2 : vtravel + 2, corner] = BG
    grid[row, 1 : seg_len + 1] = 1
    future = [(row, c) for c in range(seg_len + 1, corner + 1)]
    future += [(r, corner) for r in range(2, vtravel + 2)]
    grid = _attach_optional_dye(grid, future, add_dye)
    return {"grid": grid, "kind": "right", "needs_dye": add_dye}


def _make_left_turn_gadget(add_dye: bool) -> dict:
    base = _make_right_turn_gadget(add_dye)
    base["grid"] = np.flipud(base["grid"])
    base["kind"] = "left"
    return base


def _make_collision_gadget() -> dict:
    left_len = random.randint(2, 4)
    right_len = random.randint(2, 4)
    approach = random.randint(2, 4)
    row = 2
    right_head = left_len + 2 * approach
    width = right_head + right_len + 1
    height = 5
    grid = np.full((height, width), SYMBOL_WALL, dtype=int)
    grid[row, 1 : width - 1] = BG
    grid[row, 1 : left_len + 1] = 1
    grid[row, right_head : right_head + right_len] = 2
    return {"grid": grid, "kind": "collision", "needs_dye": False}


def _rotate_gadget(spec: dict) -> dict:
    turns = random.randint(0, 3)
    grid = np.rot90(spec["grid"], turns).copy()
    return {"grid": grid, "kind": spec["kind"], "needs_dye": spec["needs_dye"]}


RECIPES = [
    ["straight", "right"],
    ["straight", "left"],
    ["right", "left"],
    ["collision", "straight"],
    ["collision", "right"],
    ["collision", "left"],
    ["straight", "right", "left"],
]


def _build_recipe_specs() -> tuple[list[dict], set[str]]:
    recipe = random.choice(RECIPES)
    single_indices = [i for i, kind in enumerate(recipe) if kind != "collision"]
    dyed = set()
    if single_indices and random.random() < 0.7:
        dyed.add(random.choice(single_indices))
    if len(single_indices) >= 2 and random.random() < 0.3:
        remaining = [i for i in single_indices if i not in dyed]
        if remaining:
            dyed.add(random.choice(remaining))

    specs: list[dict] = []
    expected = {"blocked_stop"}
    for idx, kind in enumerate(recipe):
        add_dye = idx in dyed
        if kind == "straight":
            spec = _make_straight_gadget(add_dye)
        elif kind == "right":
            spec = _make_right_turn_gadget(add_dye)
            expected.add("right_turn")
        elif kind == "left":
            spec = _make_left_turn_gadget(add_dye)
            expected.add("left_turn")
        else:
            spec = _make_collision_gadget()
            expected.add("collision")
        if add_dye:
            expected.add("color_change")
        specs.append(_rotate_gadget(spec))
    return specs, expected


def _find_positions(occupied: np.ndarray, gh: int, gw: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions = []
    for r0 in range(h - gh + 1):
        for c0 in range(w - gw + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + gh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + gw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _render_grid(specs: list[dict]) -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    path_total = 0
    dye_total = 0
    for spec in specs:
        vals = set(int(v) for v in np.unique(spec["grid"]))
        path_total += sum(1 for v in vals if 1 <= v < 10)
        dye_total += sum(1 for v in vals if v >= 10)
    if path_total + dye_total > len(PALETTE):
        return None

    path_colors = random.sample(PALETTE, path_total)
    remaining = [c for c in PALETTE if c not in path_colors]
    dye_colors = random.sample(remaining, dye_total)
    path_ptr = 0
    dye_ptr = 0

    for spec in sorted(specs, key=lambda item: item["grid"].shape[0] * item["grid"].shape[1], reverse=True):
        local = spec["grid"]
        gh, gw = local.shape
        positions = _find_positions(occupied, gh, gw)
        if not positions:
            return None
        r0, c0 = positions[0]

        pmap: dict[int, int] = {}
        dmap: dict[int, int] = {}
        for val in sorted(int(v) for v in np.unique(local)):
            if 1 <= val < 10:
                pmap[val] = path_colors[path_ptr]
                path_ptr += 1
            elif val >= 10:
                dmap[val] = dye_colors[dye_ptr]
                dye_ptr += 1

        for r in range(gh):
            for c in range(gw):
                val = int(local[r, c])
                if val == BG:
                    continue
                if val == SYMBOL_WALL:
                    grid[r0 + r, c0 + c] = WALL
                elif val in pmap:
                    grid[r0 + r, c0 + c] = pmap[val]
                else:
                    grid[r0 + r, c0 + c] = dmap[val]

        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + gh + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + gw + 1)
        occupied[rr0:rr1, cc0:cc1] = True

    return grid


def _trace_features(traces: list[dict]) -> set[str]:
    features = set()
    if any(trace["right_turn"] for trace in traces):
        features.add("right_turn")
    if any(trace["left_turn"] for trace in traces):
        features.add("left_turn")
    if any(trace["color_change"] for trace in traces):
        features.add("color_change")
    if any(trace["collision"] for trace in traces):
        features.add("collision")
    if any(trace["blocked_stop"] for trace in traces):
        features.add("blocked_stop")
    return features


def _gen_attempt() -> np.ndarray | None:
    specs, expected = _build_recipe_specs()
    grid = _render_grid(specs)
    if grid is None:
        return None
    if np.mean(grid == BG) < 0.4:
        return None

    try:
        out, traces = _solve(grid, record=True)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not expected.issubset(_trace_features(traces)):
        return None
    if sum(trace["steps"] for trace in traces) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
