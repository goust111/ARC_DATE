from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = ("tl", "tr", "bl", "br")
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_legend(grid: np.ndarray, bg: int, grey: int) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    h, w = grid.shape
    mask = (grid != bg) & (grid != grey)
    candidates: list[tuple[tuple[int, int, int, int], dict[int, int]]] = []
    for comp in _components(mask):
        if len(comp) < 2:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != 2 or len(comp) != 2 * width:
            continue
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        top = [int(x) for x in block[0]]
        bottom = [int(x) for x in block[1]]
        if len(set(top)) != width:
            continue
        mapping = {color_in: color_out for color_in, color_out in zip(top, bottom)}
        if set(mapping) != set(bottom):
            continue
        candidates.append(((r0, c0, r1, c1), mapping))
    assert len(candidates) == 1, f"expected one legend, got {len(candidates)}"
    return candidates[0]


def _find_sources(
    grid: np.ndarray,
    bg: int,
    grey: int,
    legend_bbox: tuple[int, int, int, int],
    turn_map: dict[int, int],
) -> list[dict[str, object]]:
    h, w = grid.shape
    r0, c0, r1, c1 = legend_bbox
    colored_mask = (grid != bg) & (grid != grey)
    colored_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(colored_mask)
    sources: list[dict[str, object]] = []
    for comp in comps:
        assert len(comp) == 1, "non-legend colored objects must be single-pixel sources"
        sr, sc = map(int, comp[0])
        color = int(grid[sr, sc])
        assert color in turn_map, f"source color {color} missing from legend"
        grey_neighbors = [(nr, nc) for nr, nc in _neighbors4(sr, sc, h, w) if int(grid[nr, nc]) == grey]
        assert len(grey_neighbors) == 1, f"source at {(sr, sc)} must touch exactly one grey endpoint"
        gr, gc = grey_neighbors[0]
        direction = (gr - sr, gc - sc)
        sources.append(
            {
                "source": (sr, sc),
                "start": (gr, gc),
                "dir": direction,
                "color": color,
            }
        )
    assert len(sources) >= 2, "expected at least two sources"
    sources.sort(key=lambda item: tuple(item["source"]))  # deterministic tie-break order
    for idx, src in enumerate(sources):
        src["id"] = idx
    return sources


def _solve(
    grid: np.ndarray,
    *,
    reject_ties: bool = False,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)
    grey = GREY
    assert np.any(g == grey), "expected a grey network"

    legend_bbox, turn_map = _detect_legend(g, bg, grey)
    sources = _find_sources(g, bg, grey, legend_bbox, turn_map)

    out = g.copy()
    claim_time = np.full((h, w), -1, dtype=int)
    claim_owner = np.full((h, w), -1, dtype=int)
    tie_cells = 0

    frontier: list[tuple[int, int, tuple[int, int], int, int]] = []
    for src in sources:
        sr, sc = src["start"]
        sid = int(src["id"])
        color = int(src["color"])
        claim_time[sr, sc] = 0
        claim_owner[sr, sc] = sid
        out[sr, sc] = color
        frontier.append((sr, sc, tuple(src["dir"]), color, sid))

    turned_claims = 0

    while frontier:
        layer_time = int(claim_time[frontier[0][0], frontier[0][1]])
        next_time = layer_time + 1
        attempts: dict[tuple[int, int], list[tuple[int, tuple[int, int], int]]] = {}
        for r, c, incoming_dir, color, sid in frontier:
            reverse = (-incoming_dir[0], -incoming_dir[1])
            for dr, dc in DIRS:
                if (dr, dc) == reverse:
                    continue
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != grey:
                    continue
                if claim_time[nr, nc] != -1 and claim_time[nr, nc] < next_time:
                    continue
                next_color = color if (dr, dc) == incoming_dir else turn_map[color]
                attempts.setdefault((nr, nc), []).append((sid, (dr, dc), int(next_color)))

        next_frontier: list[tuple[int, int, tuple[int, int], int, int]] = []
        for (nr, nc), moves in attempts.items():
            by_source: dict[int, list[tuple[tuple[int, int], int]]] = {}
            for sid, direction, color in moves:
                by_source.setdefault(sid, []).append((direction, color))

            current_owner = int(claim_owner[nr, nc])
            current_time = int(claim_time[nr, nc])
            if current_time == next_time:
                winner = current_owner
                if winner not in by_source:
                    continue
            elif current_time != -1:
                continue
            elif len(by_source) == 1:
                winner = next(iter(by_source))
            else:
                tie_cells += 1
                if reject_ties:
                    raise AssertionError(f"tie at {(nr, nc)}")
                winner = min(by_source)

            winner_moves = by_source[winner]
            winner_colors = {color for _direction, color in winner_moves}
            assert len(winner_colors) == 1, "winning source reached one pixel with inconsistent colors"
            winner_color = winner_colors.pop()

            if current_time == -1:
                parent_colors = []
                for sid, direction, color in moves:
                    if sid != winner:
                        continue
                    pr = nr - direction[0]
                    pc = nc - direction[1]
                    parent_colors.append(int(out[pr, pc]))
                if any(parent != winner_color for parent in parent_colors):
                    turned_claims += 1

                claim_time[nr, nc] = next_time
                claim_owner[nr, nc] = winner
                out[nr, nc] = winner_color

            seen_states: set[tuple[tuple[int, int], int]] = set()
            for direction, color in winner_moves:
                state_key = (direction, color)
                if state_key in seen_states:
                    continue
                seen_states.add(state_key)
                next_frontier.append((nr, nc, direction, color, winner))

        frontier = next_frontier

    assert not np.any(out == grey), "all grey pixels must be recolored"
    meta = {"tie_cells": tie_cells, "turned_claims": turned_claims}
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate color signals along a grey orthogonal network with turn-triggered recoloring.

    Input structure
    - bg is black.
    - Grey pixels form a 1-pixel-wide orthogonal network.
    - Several leaf endpoints have distinct colored source pixels attached.
    - A 2xN corner legend defines the color used after a 90-degree turn.

    Rule
    - Each source enters the adjacent grey endpoint and spreads through the network.
    - Continuing straight preserves the signal color.
    - Taking a turn changes the signal color using the legend mapping.
    - At T and cross nodes the signal forks into every non-backtracking direction.
    - When fronts collide, the earlier front keeps advancing and the later one stops.
    """
    return _solve(grid, reject_ties=False, return_meta=False)


def _add_edge(adj: dict[tuple[int, int], set[tuple[int, int]]], a: tuple[int, int], b: tuple[int, int]) -> None:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def _tree_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _leaf_count(adj: dict[tuple[int, int], set[tuple[int, int]]]) -> int:
    return sum(1 for nbrs in adj.values() if len(nbrs) == 1)


def _grow_tree(rng: random.Random, rows: int, cols: int, leaf_target: int) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    root = rng.choice(all_cells)
    first_options = _tree_neighbors(root, rows, cols)
    if not first_options:
        return None
    first = rng.choice(first_options)

    adj: dict[tuple[int, int], set[tuple[int, int]]] = {}
    _add_edge(adj, root, first)

    min_nodes = leaf_target + 2
    max_nodes = leaf_target + 6
    target_nodes = rng.randint(min_nodes, max_nodes)

    for _ in range(target_nodes - 2):
        current_leaves = [node for node, nbrs in adj.items() if len(nbrs) == 1]
        nodes_with_space = [node for node in adj if any(nbr not in adj for nbr in _tree_neighbors(node, rows, cols))]
        if not nodes_with_space:
            break

        if _leaf_count(adj) < leaf_target:
            preferred = [node for node in nodes_with_space if len(adj[node]) >= 2]
            if leaf_target == 4 and rng.random() < 0.45:
                preferred = [node for node in nodes_with_space if len(adj[node]) in (2, 3)] or preferred
            if not preferred:
                preferred = nodes_with_space
        else:
            preferred = [node for node in current_leaves if node in nodes_with_space]
            if not preferred:
                preferred = nodes_with_space

        parent = rng.choice(preferred)
        choices = [nbr for nbr in _tree_neighbors(parent, rows, cols) if nbr not in adj]
        if not choices:
            continue
        child = rng.choice(choices)
        _add_edge(adj, parent, child)

    if _leaf_count(adj) != leaf_target:
        return None
    if max(len(nbrs) for nbrs in adj.values()) < 3:
        return None
    return adj


def _rasterize_tree(
    adj: dict[tuple[int, int], set[tuple[int, int]]],
    step: int,
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    nodes = list(adj)
    min_r = min(r for r, _ in nodes)
    min_c = min(c for _, c in nodes)
    centers = {
        node: ((node[0] - min_r) * step, (node[1] - min_c) * step)
        for node in nodes
    }

    grey: set[tuple[int, int]] = set()
    for node, nbrs in adj.items():
        r0, c0 = centers[node]
        grey.add((r0, c0))
        for nbr in nbrs:
            if node > nbr:
                continue
            r1, c1 = centers[nbr]
            if r0 == r1:
                lo, hi = sorted((c0, c1))
                for cc in range(lo, hi + 1):
                    grey.add((r0, cc))
            else:
                lo, hi = sorted((r0, r1))
                for rr in range(lo, hi + 1):
                    grey.add((rr, c0))

    leaves: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for node, nbrs in adj.items():
        if len(nbrs) != 1:
            continue
        nbr = next(iter(nbrs))
        cr, cc = centers[node]
        dr = node[0] - nbr[0]
        dc = node[1] - nbr[1]
        source = (cr + dr, cc + dc)
        leaves.append((source, (cr, cc)))

    all_coords = list(grey) + [src for src, _start in leaves]
    min_rr = min(r for r, _ in all_coords)
    min_cc = min(c for _, c in all_coords)
    if min_rr < 0 or min_cc < 0:
        dr = -min_rr
        dc = -min_cc
        grey = {(r + dr, c + dc) for r, c in grey}
        leaves = [((sr + dr, sc + dc), (gr + dr, gc + dc)) for (sr, sc), (gr, gc) in leaves]
    return grey, leaves


def _choose_mapping(colors: list[int], rng: random.Random) -> dict[int, int]:
    ordered = colors[:]
    rng.shuffle(ordered)
    shift_choices = [shift for shift in range(1, len(ordered)) if shift % len(ordered) != 0]
    shift = rng.choice(shift_choices)
    return {
        ordered[i]: ordered[(i + shift) % len(ordered)]
        for i in range(len(ordered))
    }


def _legend_corner_ranges(
    corner: str,
    height: int,
    width: int,
    net_h: int,
    net_w: int,
    legend_w: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    row_pad = 4
    col_pad = legend_w + 4
    if corner == "tl":
        row_lo, row_hi = row_pad, height - net_h - 2
        col_lo, col_hi = col_pad, width - net_w - 2
    elif corner == "tr":
        row_lo, row_hi = row_pad, height - net_h - 2
        col_lo, col_hi = 2, width - net_w - col_pad
    elif corner == "bl":
        row_lo, row_hi = 2, height - net_h - row_pad
        col_lo, col_hi = col_pad, width - net_w - 2
    else:
        row_lo, row_hi = 2, height - net_h - row_pad
        col_lo, col_hi = 2, width - net_w - col_pad
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return (row_lo, row_hi), (col_lo, col_hi)


def _place_legend(grid: np.ndarray, mapping: dict[int, int], corner: str) -> None:
    colors = list(mapping)
    width = len(colors)
    legend = np.array([colors, [mapping[color] for color in colors]], dtype=int)
    h, w = grid.shape
    if corner == "tl":
        grid[:2, :width] = legend
    elif corner == "tr":
        grid[:2, w - width :] = legend
    elif corner == "bl":
        grid[h - 2 :, :width] = legend
    else:
        grid[h - 2 :, w - width :] = legend


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    source_count = rng.choice([3, 4])
    colors = rng.sample(COLOR_POOL, source_count)
    mapping = _choose_mapping(colors, rng)

    rows = rng.randint(4, 6)
    cols = rng.randint(4, 6)
    step = rng.choice([3, 4])
    tree = _grow_tree(rng, rows, cols, source_count)
    if tree is None:
        return None

    grey_rel, leaves_rel = _rasterize_tree(tree, step)
    net_h = max(r for r, _ in grey_rel) + 1
    net_w = max(c for _, c in grey_rel) + 1
    all_h = max(max(r for r, _ in grey_rel), max(r for (r, _c), _start in leaves_rel)) + 1
    all_w = max(max(c for _, c in grey_rel), max(c for (_r, c), _start in leaves_rel)) + 1
    net_h = max(net_h, all_h)
    net_w = max(net_w, all_w)

    corner = rng.choice(CORNERS)
    legend_w = len(mapping)
    min_h = max(net_h + 7, 18)
    min_w = max(net_w + legend_w + 7, 18)
    max_h = min_h + 8
    max_w = min_w + 8
    height = rng.randint(min_h, max_h)
    width = rng.randint(min_w, max_w)

    ranges = _legend_corner_ranges(corner, height, width, net_h, net_w, legend_w)
    if ranges is None:
        return None
    (row_lo, row_hi), (col_lo, col_hi) = ranges
    off_r = rng.randint(row_lo, row_hi)
    off_c = rng.randint(col_lo, col_hi)

    grid = np.full((height, width), BG, dtype=int)
    for r, c in grey_rel:
        grid[off_r + r, off_c + c] = GREY

    leaf_items = leaves_rel[:]
    rng.shuffle(leaf_items)
    rng.shuffle(colors)
    for color, (source_rc, _start_rc) in zip(colors, leaf_items):
        sr, sc = source_rc
        grid[off_r + sr, off_c + sc] = color

    _place_legend(grid, mapping, corner)

    try:
        out, meta = _solve(grid, reject_ties=True, return_meta=True)
    except AssertionError:
        return None

    if meta["tie_cells"] != 0:
        return None
    if meta["turned_claims"] == 0:
        return None
    if np.any(out == GREY):
        return None
    network_colors = set(map(int, np.unique(out[grid == GREY])))
    if len(network_colors) < min(2, len(mapping)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
