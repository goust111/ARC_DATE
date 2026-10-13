from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
MARKER = 5
MOVER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return comps


def _find_squares(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = []
    for comp in _components_by_color(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        n = r1 - r0 + 1
        if n < 2 or n != c1 - c0 + 1 or int(comp["area"]) != n * n:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if bool((patch == int(comp["color"])).all()):
            candidates.append(
                {
                    "color": int(comp["color"]),
                    "top_left": (int(r0), int(c0)),
                    "n": int(n),
                }
            )

    by_size: dict[int, list[dict]] = {}
    for sq in candidates:
        by_size.setdefault(int(sq["n"]), []).append(sq)
    exact_two = [n for n, group in by_size.items() if len(group) == 2]
    if exact_two:
        chosen_n = max(exact_two)
    else:
        chosen_n = max(by_size, key=lambda n: (len(by_size[n]), n))
    squares = sorted(by_size[chosen_n], key=lambda sq: (int(sq["color"]), sq["top_left"]))
    assert len(squares) == 2
    assert squares[0]["color"] != squares[1]["color"]
    return squares


def _halo_from_markers(marker_mask: np.ndarray) -> np.ndarray:
    h, w = marker_mask.shape
    halo = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(marker_mask):
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        halo[r0:r1, c0:c1] = True
    return halo


def _feasible_positions(halo: np.ndarray, n: int) -> np.ndarray:
    h, w = halo.shape
    out = np.zeros((h - n + 1, w - n + 1), dtype=bool)
    for r in range(h - n + 1):
        for c in range(w - n + 1):
            out[r, c] = bool(halo[r : r + n, c : c + n].all())
    return out


def _bfs_dist(feasible: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = feasible.shape
    dist = -np.ones((h, w), dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and feasible[nr, nc] and int(dist[nr, nc]) < 0:
                dist[nr, nc] = int(dist[r, c]) + 1
                q.append((nr, nc))
    return dist


def _components_of_feasible(feasible: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = feasible.shape
    labels = -np.ones((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not feasible[r, c] or int(labels[r, c]) >= 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            nodes = []
            while q:
                rr, cc = q.popleft()
                nodes.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and feasible[nr, nc] and int(labels[nr, nc]) < 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps.append(nodes)
            next_id += 1
    return labels, comps


def _farthest_node(dist: np.ndarray, nodes: list[tuple[int, int]]) -> tuple[int, int]:
    return min(nodes, key=lambda rc: (-int(dist[rc]), rc[0], rc[1]))


def _component_info(feasible: np.ndarray, nodes: list[tuple[int, int]]) -> dict:
    a0 = nodes[0]
    dist0 = _bfs_dist(feasible, a0)
    a = _farthest_node(dist0, nodes)
    dist_a = _bfs_dist(feasible, a)
    b = _farthest_node(dist_a, nodes)
    dist_b = _bfs_dist(feasible, b)
    scores = np.minimum(dist_a, dist_b)
    return {
        "nodes": nodes,
        "a": a,
        "b": b,
        "dist_a": dist_a,
        "dist_b": dist_b,
        "scores": scores,
        "size": len(nodes),
    }


def _render_output(grid: np.ndarray, source: dict, stop: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0 = source["top_left"]
    n = source["n"]
    out[r0 : r0 + n, c0 : c0 + n] = BG
    rr, cc = stop
    out[rr : rr + n, cc : cc + n] = int(source["color"])
    return out


def _solve(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    squares = _find_squares(g, bg)
    n = int(squares[0]["n"])
    assert n == int(squares[1]["n"])

    marker_mask = g == MARKER
    traversable = _halo_from_markers(marker_mask)
    for sq in squares:
        r0, c0 = sq["top_left"]
        traversable[r0 : r0 + n, c0 : c0 + n] = True
    feasible = _feasible_positions(traversable, n)
    labels, comps = _components_of_feasible(feasible)
    infos = [_component_info(feasible, nodes) for nodes in comps]

    enriched = []
    for sq in squares:
        r0, c0 = sq["top_left"]
        assert feasible[r0, c0]
        comp_id = int(labels[r0, c0])
        info = infos[comp_id]
        score = int(info["scores"][r0, c0])
        enriched.append(
            {
                **sq,
                "comp_id": comp_id,
                "score": score,
                "comp_size": int(info["size"]),
            }
        )

    source_idx = min(
        range(2),
        key=lambda i: (
            int(enriched[i]["score"]),
            -int(enriched[i]["comp_size"]),
            enriched[i]["top_left"][0],
            enriched[i]["top_left"][1],
        ),
    )
    target_idx = 1 - source_idx
    source = enriched[source_idx]
    target = enriched[target_idx]

    source_node = tuple(source["top_left"])
    target_node = tuple(target["top_left"])
    dist = _bfs_dist(feasible, source_node)
    reachable = int(dist[target_node]) >= 0
    if reachable:
        stop = target_node
    else:
        src_nodes = infos[int(source["comp_id"])]["nodes"]
        stop = min(
            [node for node in src_nodes if int(dist[node]) >= 0],
            key=lambda rc: (
                -int(dist[rc]),
                abs(rc[0] - target_node[0]) + abs(rc[1] - target_node[1]),
                rc[0],
                rc[1],
            ),
        )

    return {
        "bg": bg,
        "n": n,
        "source": source,
        "target": target,
        "stop": stop,
        "reachable": reachable,
        "output": _render_output(g, source, stop),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the mover square through the implicit halo corridor generated by grey marker pixels.

    Input structure:
    - A black background contains many isolated grey marker pixels that trace a winding chain.
    - The chain mixes straight and diagonal progress, with uneven gaps between successive markers.
    - Two same-sized solid colored squares appear: one mover and one target.

    Rule:
    - Treat every grey marker as the center of an invisible 3x3 halo, and union all such halos into a corridor.
    - The mover square may slide one cell at a time between top-left positions whose full NxN footprint stays inside that halo corridor.
    - The square closer to a corridor endpoint is the mover. If it can reach the target square's position, it replaces the target; otherwise it stops at the farthest reachable position toward the target.
    - Clear the mover's original square to background and keep every visible grey marker unchanged.

    Key insight:
    - The corridor is never drawn directly: it must be reconstructed from the 3x3 halo union of the sparse marker chain, and diagonal or sparse stretches become bottlenecks that block 3x3 squares while still allowing 2x2 squares.
    """
    return _solve(grid)["output"]


def _marker_coverage(mask: np.ndarray, top_left: tuple[int, int], n: int) -> int:
    r0, c0 = top_left
    return int(mask[r0 : r0 + n, c0 : c0 + n].sum())


def _remove_markers(mask: np.ndarray, boxes: list[tuple[int, int, int]]) -> np.ndarray:
    out = mask.copy()
    for r0, c0, n in boxes:
        out[r0 : r0 + n, c0 : c0 + n] = False
    return out


def _step_choices() -> list[tuple[int, int, str]]:
    out = []
    for delta in (2, 3):
        out.extend(((delta, 0, "card"), (-delta, 0, "card"), (0, delta, "card"), (0, -delta, "card")))
    for dr in (-2, 2):
        for dc in (-2, 2):
            out.append((dr, dc, "diag"))
    return out


def _build_marker_chain(h: int, w: int) -> list[tuple[int, int]] | None:
    steps = _step_choices()
    target_len = random.randint(8, 12)
    for _ in range(120):
        start = (random.randint(2, h - 3), random.randint(2, w - 3))
        path = [start]
        kinds = []
        lengths = set()
        last_vec = None
        while len(path) < target_len:
            prev_r, prev_c = path[-1]
            candidates = []
            for dr, dc, kind in steps:
                nr, nc = prev_r + dr, prev_c + dc
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                    continue
                if (nr, nc) in path:
                    continue
                if last_vec is not None and (dr, dc) == (-last_vec[0], -last_vec[1]):
                    continue
                ok = True
                for idx, (rr, cc) in enumerate(path):
                    cheb = max(abs(rr - nr), abs(cc - nc))
                    if idx == len(path) - 1:
                        continue
                    if cheb <= 3:
                        ok = False
                        break
                if not ok:
                    continue
                turn = last_vec is not None and (dr, dc) != last_vec
                weight = 3 if turn else 1
                if kind == "diag":
                    weight += 1
                candidates.extend([((nr, nc), kind, max(abs(dr), abs(dc)))] * weight)
            if not candidates:
                break
            nxt, kind, length = random.choice(candidates)
            path.append(nxt)
            kinds.append(kind)
            lengths.add(length)
            last_vec = (nxt[0] - prev_r, nxt[1] - prev_c)
        if len(path) != target_len:
            continue
        if "card" not in kinds or "diag" not in kinds:
            continue
        if len(lengths) < 2:
            continue
        return path
    return None


def _analyze_mask(marker_mask: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    halo = _halo_from_markers(marker_mask)
    feasible = _feasible_positions(halo, n)
    labels, comps = _components_of_feasible(feasible)
    infos = [_component_info(feasible, nodes) for nodes in comps]
    return feasible, labels, infos


def _pick_positions(marker_mask: np.ndarray, n: int, want_reachable: bool) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    feasible, labels, infos = _analyze_mask(marker_mask, n)
    if not infos:
        return None

    source_candidates = []
    for comp_id, info in enumerate(infos):
        if int(info["size"]) < 5:
            continue
        for endpoint in {tuple(info["a"]), tuple(info["b"])}:
            src_dist = _bfs_dist(feasible, endpoint)
            maxd = max(int(src_dist[node]) for node in info["nodes"])
            far_nodes = [node for node in info["nodes"] if int(src_dist[node]) == maxd]
            if not want_reachable and len(far_nodes) != 1:
                continue
            source_candidates.append((comp_id, endpoint, src_dist, far_nodes[0] if far_nodes else endpoint))

    random.shuffle(source_candidates)
    for comp_id, source, src_dist, farthest in source_candidates:
        source_score = _marker_coverage(marker_mask, source, n)
        info = infos[comp_id]
        if want_reachable:
            targets = [
                node
                for node in info["nodes"]
                if node != source
                and int(info["scores"][node]) >= 1
                and int(src_dist[node]) >= max(4, n + 2)
            ]
            targets.sort(
                key=lambda rc: (
                    -int(src_dist[rc]),
                    -int(info["scores"][rc]),
                    _marker_coverage(marker_mask, rc, n),
                    rc[0],
                    rc[1],
                )
            )
            for target in targets[:10]:
                if source_score > _marker_coverage(marker_mask, target, n) + 3:
                    continue
                return source, target, target
        else:
            other_targets = []
            for other_id, other_info in enumerate(infos):
                if other_id == comp_id or int(other_info["size"]) < 2:
                    continue
                for node in other_info["nodes"]:
                    if int(other_info["scores"][node]) >= 1:
                        other_targets.append(
                            (
                                abs(node[0] - farthest[0]) + abs(node[1] - farthest[1]),
                                _marker_coverage(marker_mask, node, n),
                            node,
                        )
                        )
            if other_targets:
                other_targets.sort()
                return source, other_targets[0][2], farthest
    return None


def _render_input(
    h: int,
    w: int,
    marker_mask: np.ndarray,
    source: tuple[int, int],
    target: tuple[int, int],
    stop: tuple[int, int],
    n: int,
    mover_color: int,
    target_color: int,
) -> np.ndarray:
    visible = _remove_markers(marker_mask, [(source[0], source[1], n), (target[0], target[1], n), (stop[0], stop[1], n)])
    grid = np.zeros((h, w), dtype=int)
    grid[visible] = MARKER
    sr, sc = source
    tr, tc = target
    grid[sr : sr + n, sc : sc + n] = int(mover_color)
    grid[tr : tr + n, tc : tc + n] = int(target_color)
    return grid


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    n = random.choice((2, 3))
    want_reachable = random.random() < (0.7 if n == 2 else 0.5)

    chain = _build_marker_chain(h, w)
    if chain is None:
        return None
    marker_mask = np.zeros((h, w), dtype=bool)
    for r, c in chain:
        marker_mask[r, c] = True

    picked = _pick_positions(marker_mask, n, want_reachable)
    if picked is None:
        return None
    source, target, stop = picked

    mover_color, target_color = random.sample(MOVER_COLORS, 2)
    grid = _render_input(h, w, marker_mask, source, target, stop, n, mover_color, target_color)

    try:
        solved = _solve(grid)
    except Exception:
        return None

    marker_count = int((grid == MARKER).sum())
    if marker_count < 5:
        return None
    if np.array_equal(grid, solved["output"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
