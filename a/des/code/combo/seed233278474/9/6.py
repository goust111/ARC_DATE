from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
CARGO_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4 if _inside(h, w, r + dr, c + dc)]


def _build_graph(grid: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    visible = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) != BG and int(grid[r, c]) != ORANGE
    }
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {node: set() for node in visible}

    for r, c in visible:
        for nr, nc in _neighbors4(h, w, r, c):
            if (nr, nc) in visible:
                adj[(r, c)].add((nr, nc))

    bridges: set[tuple[tuple[int, int], ...]] = set()
    for r, c in visible:
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if not _inside(h, w, rr, cc) or int(grid[rr, cc]) != ORANGE:
                continue
            chain = [(r, c)]
            while _inside(h, w, rr, cc) and int(grid[rr, cc]) == ORANGE:
                chain.append((rr, cc))
                rr += dr
                cc += dc
            if not _inside(h, w, rr, cc):
                continue
            if int(grid[rr, cc]) == BG or int(grid[rr, cc]) == ORANGE:
                continue
            chain.append((rr, cc))
            key = tuple(chain)
            rev = tuple(reversed(chain))
            bridges.add(min(key, rev))

    for chain in bridges:
        for node in chain:
            adj.setdefault(node, set())
        for a, b in zip(chain, chain[1:], strict=False):
            adj[a].add(b)
            adj[b].add(a)

    return adj


def _graph_components(adj: dict[tuple[int, int], set[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    parts: list[list[tuple[int, int]]] = []
    for start in adj:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        parts.append(comp)
    return parts


def _ordered_path(
    adj: dict[tuple[int, int], set[tuple[int, int]]], comp: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    endpoints = [node for node in comp if len(adj[node]) == 1]
    assert len(endpoints) == 2, "each conveyor must be an open chain"
    start = min(endpoints)
    end = max(endpoints)
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        nxts = [nxt for nxt in adj[cur] if nxt != prev]
        assert len(nxts) == 1, "path must not branch"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    adj = _build_graph(g)

    out = np.full((h, w), BG, dtype=int)
    out[g == ORANGE] = ORANGE

    for comp in _graph_components(adj):
        order = _ordered_path(adj, comp)

        k = 0
        in_wall = False
        for r, c in order:
            if int(g[r, c]) == ORANGE:
                if not in_wall:
                    k += 1
                    in_wall = True
            else:
                in_wall = False
                out[r, c] = GREY

        cargos: list[tuple[int, int]] = []
        for idx, (r, c) in enumerate(order):
            color = int(g[r, c])
            if color not in (BG, GREY, ORANGE):
                cargos.append((idx, color))

        plen = len(order)
        for idx, color in cargos:
            dst = (idx + k) % plen
            rr, cc = order[dst]
            if int(g[rr, cc]) != ORANGE:
                out[rr, cc] = color

    return out


def _can_use_start(
    h: int,
    w: int,
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if cell in occupied:
        return False
    for nr, nc in _neighbors4(h, w, r, c):
        if (nr, nc) in occupied:
            return False
    return True


def _can_step(
    h: int,
    w: int,
    nxt: tuple[int, int],
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    r, c = nxt
    if not _inside(h, w, r, c):
        return False
    if nxt in used or nxt in occupied:
        return False
    for nr, nc in _neighbors4(h, w, r, c):
        if (nr, nc) in occupied:
            return False
        if (nr, nc) in used and (nr, nc) != cur:
            return False
    return True


def _grow_path(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]] | None:
    free = [(r, c) for r in range(h) for c in range(w) if _can_use_start(h, w, (r, c), occupied)]
    if not free:
        return None

    target = random.randint(min_len, max_len)
    for _ in range(140):
        start = random.choice(free)
        path = [start]
        used = {start}
        prev_dir: tuple[int, int] | None = None

        while len(path) < target:
            r, c = path[-1]
            options: list[tuple[int, int]] = []
            weights: list[int] = []
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if not _can_step(h, w, nxt, path[-1], used, occupied):
                    continue
                options.append((dr, dc))
                if prev_dir is None:
                    weights.append(1)
                elif prev_dir == (dr, dc):
                    weights.append(4)
                else:
                    weights.append(2)
            if not options:
                break
            dr, dc = random.choices(options, weights=weights, k=1)[0]
            nxt = (r + dr, c + dc)
            path.append(nxt)
            used.add(nxt)
            prev_dir = (dr, dc)

        if len(path) >= min_len:
            return path
    return None


def _interval_candidates(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) < 3:
        return []
    step_dirs = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    runs: list[tuple[int, int]] = []
    run_start = 0
    for i in range(1, len(step_dirs)):
        if step_dirs[i] != step_dirs[i - 1]:
            runs.append((run_start, i))
            run_start = i
    runs.append((run_start, len(path) - 1))

    candidates: list[tuple[int, int]] = []
    for a, b in runs:
        if b - a + 1 < 3:
            continue
        for s in range(a + 1, b):
            max_e = min(b - 1, s + 3)
            for e in range(s, max_e + 1):
                candidates.append((s, e))
    return candidates


def _compatible(interval: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    s, e = interval
    for cs, ce in chosen:
        if not (e < cs - 1 or ce < s - 1):
            return False
    return True


def _pick_intervals(
    candidates: list[tuple[int, int]], force_nonzero: bool
) -> list[tuple[int, int]] | None:
    if not candidates:
        return None if force_nonzero else []
    target = random.randint(1 if force_nonzero else 0, min(3, len(candidates)))
    if target == 0:
        return []
    pool = candidates[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for interval in pool:
        if _compatible(interval, chosen):
            chosen.append(interval)
            if len(chosen) == target:
                break
    if force_nonzero and not chosen:
        return None
    return sorted(chosen)


def _render_input(
    h: int,
    w: int,
    paths: list[list[tuple[int, int]]],
    hidden_lists: list[list[tuple[int, int]]],
    cargo_maps: list[dict[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for path, hidden, cargos in zip(paths, hidden_lists, cargo_maps, strict=False):
        hidden_idx: set[int] = set()
        for s, e in hidden:
            hidden_idx.update(range(s, e + 1))
        for idx, (r, c) in enumerate(path):
            if idx in hidden_idx:
                grid[r, c] = ORANGE
            elif idx in cargos:
                grid[r, c] = cargos[idx]
            else:
                grid[r, c] = GREY
    return grid


def _render_expected(
    h: int,
    w: int,
    paths: list[list[tuple[int, int]]],
    hidden_lists: list[list[tuple[int, int]]],
    cargo_maps: list[dict[int, int]],
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for path, hidden, cargos in zip(paths, hidden_lists, cargo_maps, strict=False):
        hidden_idx: set[int] = set()
        for s, e in hidden:
            hidden_idx.update(range(s, e + 1))
        for idx, (r, c) in enumerate(path):
            if idx in hidden_idx:
                out[r, c] = ORANGE
            else:
                out[r, c] = GREY
        shift = len(hidden)
        plen = len(path)
        for idx, color in cargos.items():
            dst = (idx + shift) % plen
            if dst not in hidden_idx:
                rr, cc = path[dst]
                out[rr, cc] = color
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    max_paths = 3 if h * w >= 220 else 2
    num_paths = random.randint(1, max_paths)

    occupied: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []
    for _ in range(num_paths):
        path = _grow_path(h, w, occupied, min_len=8, max_len=min(20, max(8, (h * w) // 10 + 6)))
        if path is None:
            return None
        paths.append(path)
        occupied.update(path)

    candidate_lists = [_interval_candidates(path) for path in paths]
    eligible = [i for i, cands in enumerate(candidate_lists) if cands]
    if not eligible:
        return None

    forced = random.choice(eligible)
    hidden_lists: list[list[tuple[int, int]]] = []
    for i, cands in enumerate(candidate_lists):
        chosen = _pick_intervals(cands, force_nonzero=(i == forced))
        if chosen is None:
            return None
        hidden_lists.append(chosen)
    if sum(len(hidden) for hidden in hidden_lists) == 0:
        return None

    cargo_maps: list[dict[int, int]] = []
    for path, hidden in zip(paths, hidden_lists, strict=False):
        hidden_idx: set[int] = set()
        for s, e in hidden:
            hidden_idx.update(range(s, e + 1))
        visible_idx = [i for i in range(len(path)) if i not in hidden_idx]
        if not visible_idx:
            return None
        max_cargo = min(4, len(visible_idx))
        cargo_count = random.randint(1, max_cargo)
        chosen_idx = random.sample(visible_idx, cargo_count)
        cargos = {idx: random.choice(CARGO_COLORS) for idx in chosen_idx}
        cargo_maps.append(cargos)

    inp = _render_input(h, w, paths, hidden_lists, cargo_maps)
    out = _render_expected(h, w, paths, hidden_lists, cargo_maps)

    if np.array_equal(inp, out):
        return None

    non_bg = set(map(int, inp.ravel())) - {BG}
    if ORANGE not in non_bg or GREY not in non_bg:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate valid conveyor-wall input")
