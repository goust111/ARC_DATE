from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_PIPE = 5
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_PRIORITY = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _infer_bg_and_pipe(grid: np.ndarray) -> tuple[int, int | None]:
    vals, counts = np.unique(grid, return_counts=True)
    if len(vals) == 0:
        return BG, None

    bg = BG if int(BG) in set(map(int, vals)) else int(vals[np.argmax(counts)])
    non_bg = [int(v) for v in vals if int(v) != int(bg)]
    if not non_bg:
        return int(bg), None

    pipe = max(non_bg, key=lambda v: int(np.sum(grid == int(v))))
    return int(bg), int(pipe)


def _build_graph(mask: np.ndarray) -> tuple[set[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]], dict[tuple[int, int], int]]:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    degree: dict[tuple[int, int], int] = {}
    for cell in cells:
        nbrs = [nxt for nxt in _neighbors4(cell) if nxt in cells]
        neighbors[cell] = nbrs
        degree[cell] = len(nbrs)
    return cells, neighbors, degree


def _find_seeds(
    grid: np.ndarray,
    bg: int,
    pipe: int,
    degree: dict[tuple[int, int], int],
) -> list[dict[str, object]]:
    seeds: list[dict[str, object]] = []
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            col = int(grid[r, c])
            if col == int(bg) or col == int(pipe):
                continue
            adj = []
            for nr, nc in _neighbors4((r, c)):
                if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == int(pipe):
                    adj.append((nr, nc))
            assert len(adj) == 1, "each seed must touch exactly one pipe cell"
            leaf = adj[0]
            assert degree.get(leaf, 0) == 1, "seed must sit outside a pipe dead end"
            seeds.append(
                {
                    "seed_cell": (int(r), int(c)),
                    "leaf": leaf,
                    "color": int(col),
                }
            )
    seeds.sort(key=lambda x: (x["leaf"], x["seed_cell"]))  # stable simulation order
    return seeds


def _remaining_len(
    cell: tuple[int, int],
    prev: tuple[int, int] | None,
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    used_edges: set[tuple[tuple[int, int], tuple[int, int]]],
    memo: dict[tuple[tuple[int, int], tuple[int, int] | None], int],
) -> int:
    key = (cell, prev)
    if key in memo:
        return memo[key]

    options = [
        nxt
        for nxt in neighbors[cell]
        if nxt != prev and _edge(cell, nxt) not in used_edges
    ]
    if not options:
        memo[key] = 1
        return 1

    best = max(_remaining_len(nxt, cell, neighbors, used_edges, memo) for nxt in options)
    memo[key] = 1 + int(best)
    return memo[key]


def _choose_next(
    cell: tuple[int, int],
    prev: tuple[int, int] | None,
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    used_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[int, int] | None:
    options = [
        nxt
        for nxt in neighbors[cell]
        if nxt != prev and _edge(cell, nxt) not in used_edges
    ]
    if not options:
        return None
    if len(options) == 1:
        return options[0]

    memo: dict[tuple[tuple[int, int], tuple[int, int] | None], int] = {}
    ranked = []
    for nxt in options:
        dr = int(nxt[0] - cell[0])
        dc = int(nxt[1] - cell[1])
        ranked.append(
            (
                -int(_remaining_len(nxt, cell, neighbors, used_edges, memo)),
                int(DIR_PRIORITY[(dr, dc)]),
                nxt,
            )
        )
    ranked.sort()
    return ranked[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colored seeds through a tree-shaped grey pipe network.

    Rule summary:
    - Each seed starts at a dead-end leaf and advances one cell per step.
    - At a branch, the seed chooses the longest still-unused continuation;
      ties use Up > Right > Down > Left.
    - A pipe cell keeps the color of the first seed that reaches it; if multiple
      seeds first arrive there in the same step, that junction cell stays grey.
    - When a seed reaches its final dead end, it paints a 1-step cross marker on
      black background cells around that end.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, pipe = _infer_bg_and_pipe(g)
    if pipe is None:
        return out

    pipe_cells, neighbors, degree = _build_graph(g == int(pipe))
    if not pipe_cells:
        return out

    seeds = _find_seeds(g, int(bg), int(pipe), degree)
    if not seeds:
        return out

    fixed_color: dict[tuple[int, int], int | None] = {}
    used_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    agents = []
    for idx, seed in enumerate(seeds):
        leaf = seed["leaf"]
        color = int(seed["color"])
        if leaf not in fixed_color:
            fixed_color[leaf] = color
            out[leaf] = color
        agents.append(
            {
                "id": idx,
                "color": color,
                "curr": leaf,
                "prev": None,
                "done": False,
                "terminal": None,
            }
        )

    max_steps = max(1, len(pipe_cells) * (len(agents) + 2))
    for _ in range(max_steps):
        proposals: dict[int, tuple[int, int]] = {}
        for agent in agents:
            if agent["done"]:
                continue
            nxt = _choose_next(
                agent["curr"],
                agent["prev"],
                neighbors,
                used_edges,
            )
            if nxt is None:
                agent["done"] = True
                agent["terminal"] = agent["curr"]
            else:
                proposals[int(agent["id"])] = nxt

        if not proposals:
            break

        by_target: dict[tuple[int, int], list[int]] = {}
        for aid, target in proposals.items():
            by_target.setdefault(target, []).append(int(aid))

        for target in sorted(by_target):
            bidders = sorted(by_target[target])
            if target not in fixed_color:
                if len(bidders) == 1:
                    color = int(agents[bidders[0]]["color"])
                    fixed_color[target] = color
                    out[target] = color
                else:
                    fixed_color[target] = None
                    out[target] = int(pipe)

            for aid in bidders:
                agent = agents[aid]
                curr = agent["curr"]
                used_edges.add(_edge(curr, target))
                agent["prev"] = curr
                agent["curr"] = target
    else:
        raise AssertionError("pipe simulation did not converge")

    H, W = g.shape
    for agent in agents:
        terminal = agent["terminal"]
        if terminal is None:
            terminal = agent["curr"]
        if degree.get(terminal, 0) != 1:
            continue
        color = int(agent["color"])
        for dr, dc in DIRS4:
            rr = int(terminal[0] + dr)
            cc = int(terminal[1] + dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = color

    return out


def _rot(cell: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = cell
    if k % 4 == 0:
        return int(r), int(c)
    if k % 4 == 1:
        return int(-c), int(r)
    if k % 4 == 2:
        return int(-r), int(-c)
    return int(c), int(-r)


def _normalize_component(
    pipe_cells: set[tuple[int, int]],
    seed_slots: list[tuple[int, int]],
) -> dict[str, object]:
    all_cells = set(pipe_cells) | set(seed_slots)
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)
    sr = 1 - int(min_r)
    sc = 1 - int(min_c)

    pipe = {(int(r + sr), int(c + sc)) for r, c in pipe_cells}
    seeds = [(int(r + sr), int(c + sc)) for r, c in seed_slots]
    h = int(max_r - min_r + 3)
    w = int(max_c - min_c + 3)

    halo: set[tuple[int, int]] = set()
    for r, c in pipe | set(seeds):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))

    return {"pipe": pipe, "seeds": seeds, "h": h, "w": w, "halo": halo}


def _build_star_component() -> dict[str, object]:
    arm_dirs = random.sample(DIRS4, random.choice([3, 4]))
    if len(arm_dirs) == 4:
        n_seed = random.choice([1, 2])
    else:
        n_seed = 1
    seed_dirs = set(random.sample(arm_dirs, n_seed))
    target_dirs = [d for d in arm_dirs if d not in seed_dirs]

    seed_lengths = random.sample(range(2, 7), n_seed)
    lengths: dict[tuple[int, int], int] = {}
    for d, length in zip(sorted(seed_dirs), seed_lengths):
        lengths[d] = int(length)

    if len(target_dirs) >= 2 and random.random() < 0.5:
        tie_len = random.randint(2, 6)
        tied = random.sample(target_dirs, 2)
        for d in target_dirs:
            if d in tied:
                lengths[d] = tie_len
            else:
                lengths[d] = random.randint(2, 6)
    else:
        for d in target_dirs:
            lengths[d] = random.randint(2, 6)

    pipe_cells = {(0, 0)}
    seed_slots: list[tuple[int, int]] = []
    for dr, dc in arm_dirs:
        length = int(lengths[(dr, dc)])
        for step in range(1, length + 1):
            pipe_cells.add((int(dr * step), int(dc * step)))
        if (dr, dc) in seed_dirs:
            seed_slots.append((int(dr * (length + 1)), int(dc * (length + 1))))

    rot_k = random.randrange(4)
    pipe_rot = {_rot(cell, rot_k) for cell in pipe_cells}
    seeds_rot = [_rot(cell, rot_k) for cell in seed_slots]
    return _normalize_component(pipe_rot, seeds_rot)


def _build_chain_component() -> dict[str, object]:
    root = (0, 0)
    trunk_len = random.randint(2, 5)
    child = (0, trunk_len)

    root_side_dirs = [(-1, 0), (0, -1), (1, 0)]
    random.shuffle(root_side_dirs)
    n_seed = random.choice([1, 1, 2])
    seed_dirs = root_side_dirs[:n_seed]
    remaining_dirs = root_side_dirs[n_seed:]
    decoy_dirs: list[tuple[int, int]] = []
    if n_seed == 2:
        decoy_dirs = [random.choice(remaining_dirs)]
    elif remaining_dirs and random.random() < 0.7:
        decoy_dirs = [random.choice(remaining_dirs)]

    child_dirs_pool = [(-1, 0), (0, 1), (1, 0)]
    random.shuffle(child_dirs_pool)
    child_target_dirs = child_dirs_pool[: random.choice([2, 3])]

    seed_lengths = random.sample(range(2, 7), len(seed_dirs))
    child_lengths: dict[tuple[int, int], int] = {}
    if len(child_target_dirs) >= 2 and random.random() < 0.5:
        tie_len = random.randint(2, 6)
        tied = random.sample(child_target_dirs, 2)
        for d in child_target_dirs:
            if d in tied:
                child_lengths[d] = tie_len
            else:
                child_lengths[d] = random.randint(2, 6)
    else:
        for d in child_target_dirs:
            child_lengths[d] = random.randint(2, 6)

    best_child = max(child_lengths.values())
    trunk_score = int(trunk_len + best_child)
    decoy_lengths = {
        d: random.randint(2, max(2, trunk_score - 1))
        for d in decoy_dirs
    }

    pipe_cells = {root}
    seed_slots: list[tuple[int, int]] = []
    for step in range(1, trunk_len + 1):
        pipe_cells.add((0, int(step)))

    for d, length in zip(seed_dirs, seed_lengths):
        dr, dc = d
        for step in range(1, length + 1):
            pipe_cells.add((int(dr * step), int(dc * step)))
        seed_slots.append((int(dr * (length + 1)), int(dc * (length + 1))))

    for d, length in decoy_lengths.items():
        dr, dc = d
        for step in range(1, length + 1):
            pipe_cells.add((int(dr * step), int(dc * step)))

    for d, length in child_lengths.items():
        dr, dc = d
        for step in range(1, int(length) + 1):
            pipe_cells.add((int(child[0] + dr * step), int(child[1] + dc * step)))

    rot_k = random.randrange(4)
    pipe_rot = {_rot(cell, rot_k) for cell in pipe_cells}
    seeds_rot = [_rot(cell, rot_k) for cell in seed_slots]
    return _normalize_component(pipe_rot, seeds_rot)


def _render_single_component(grid: np.ndarray, comp: dict[str, object], colors: list[int]) -> None:
    H, W = grid.shape
    h, w = int(comp["h"]), int(comp["w"])
    max_r = H - h - 1
    max_c = W - w - 1
    assert max_r >= 1 and max_c >= 1
    r0 = random.randint(1, max_r)
    c0 = random.randint(1, max_c)
    for r, c in comp["pipe"]:
        grid[int(r0 + r), int(c0 + c)] = GEN_PIPE
    for (r, c), color in zip(comp["seeds"], colors):
        grid[int(r0 + r), int(c0 + c)] = int(color)


def _render_two_components(grid: np.ndarray, comps: list[dict[str, object]], color_chunks: list[list[int]]) -> bool:
    H, W = grid.shape
    comp_a, comp_b = comps
    colors_a, colors_b = color_chunks
    if random.random() < 0.5:
        gap = random.randint(3, 5)
        min_h = max(int(comp_a["h"]), int(comp_b["h"])) + 2
        min_w = int(comp_a["w"]) + int(comp_b["w"]) + gap + 2
        if min_h > H or min_w > W:
            return False
        extra_h = H - min_h
        extra_w = W - min_w
        top_pad = 1 + random.randint(0, extra_h)
        left_pad = 1 + random.randint(0, extra_w)
        mid_extra = random.randint(0, max(0, extra_w - (left_pad - 1)))
        gap_actual = gap + mid_extra
        ra = random.randint(1, H - int(comp_a["h"]) - 1)
        rb = random.randint(1, H - int(comp_b["h"]) - 1)
        ca = left_pad
        cb = ca + int(comp_a["w"]) + gap_actual
    else:
        gap = random.randint(3, 5)
        min_h = int(comp_a["h"]) + int(comp_b["h"]) + gap + 2
        min_w = max(int(comp_a["w"]), int(comp_b["w"])) + 2
        if min_h > H or min_w > W:
            return False
        extra_h = H - min_h
        extra_w = W - min_w
        top_pad = 1 + random.randint(0, extra_h)
        left_pad = 1 + random.randint(0, extra_w)
        mid_extra = random.randint(0, max(0, extra_h - (top_pad - 1)))
        gap_actual = gap + mid_extra
        ca = random.randint(1, W - int(comp_a["w"]) - 1)
        cb = random.randint(1, W - int(comp_b["w"]) - 1)
        ra = top_pad
        rb = ra + int(comp_a["h"]) + gap_actual

    for (comp, colors, r0, c0) in ((comp_a, colors_a, ra, ca), (comp_b, colors_b, rb, cb)):
        for r, c in comp["pipe"]:
            grid[int(r0 + r), int(c0 + c)] = GEN_PIPE
        for (r, c), color in zip(comp["seeds"], colors):
            grid[int(r0 + r), int(c0 + c)] = int(color)
    return True


def _build_components() -> tuple[list[dict[str, object]], list[list[int]]] | None:
    n_comp = random.choice([1, 1, 2])
    builders = [_build_star_component, _build_chain_component]
    comps = [random.choice(builders)() for _ in range(n_comp)]

    n_seeds_total = sum(len(comp["seeds"]) for comp in comps)
    if n_seeds_total == 0 or n_seeds_total > len(PALETTE):
        return None

    colors = random.sample(PALETTE, n_seeds_total)
    chunks: list[list[int]] = []
    idx = 0
    for comp in comps:
        count = len(comp["seeds"])
        chunks.append(colors[idx : idx + count])
        idx += count
    return comps, chunks


def _generate_attempt() -> np.ndarray | None:
    built = _build_components()
    if built is None:
        return None
    comps, color_chunks = built

    if len(comps) == 1:
        comp = comps[0]
        min_h = max(16, int(comp["h"]) + 2)
        min_w = max(16, int(comp["w"]) + 2)
        if min_h > 28 or min_w > 28:
            return None
        H = random.randint(min_h, min(28, min_h + 8))
        W = random.randint(min_w, min(28, min_w + 8))
    else:
        min_h = max(16, max(int(c["h"]) for c in comps) + 2)
        min_w = max(16, max(int(c["w"]) for c in comps) + 2)
        H = random.randint(min_h, 28)
        W = random.randint(min_w, 28)

    grid = np.full((H, W), BG, dtype=int)
    if len(comps) == 1:
        _render_single_component(grid, comps[0], color_chunks[0])
    else:
        if not _render_two_components(grid, comps, color_chunks):
            return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    pipe_recolored = np.any((grid == GEN_PIPE) & (out != GEN_PIPE))
    bg_recolored = np.any((grid == BG) & (out != BG))
    if not (pipe_recolored and bg_recolored):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-seed ARC sample")
