import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _compute_degrees(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(mask):
        count = 0
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            if bool(mask[nr, nc]):
                count += 1
        deg[int(r), int(c)] = count
    return deg


def _parse_network(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    nonblack = g != BG
    deg_all = _compute_degrees(nonblack)

    marker_cells = set()
    converter_targets = {}
    for r, c in np.argwhere(nonblack):
        r, c = int(r), int(c)
        if int(g[r, c]) == GREY or int(deg_all[r, c]) != 1:
            continue
        neigh = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if bool(nonblack[nr, nc])]
        if len(neigh) != 1:
            continue
        qr, qc = neigh[0]
        if int(g[qr, qc]) == GREY or int(deg_all[qr, qc]) < 4:
            continue
        marker_cells.add((r, c))
        if (qr, qc) in converter_targets:
            raise AssertionError("converter may only have one target marker")
        converter_targets[(qr, qc)] = int(g[r, c])

    network_mask = nonblack.copy()
    for r, c in marker_cells:
        network_mask[r, c] = False

    deg_net = _compute_degrees(network_mask)

    sources = []
    for r, c in np.argwhere(network_mask):
        r, c = int(r), int(c)
        if int(g[r, c]) == GREY:
            continue
        if (r, c) in converter_targets:
            continue
        if int(deg_net[r, c]) == 1:
            sources.append((r, c))

    return {
        "network_mask": network_mask,
        "marker_cells": marker_cells,
        "converter_targets": converter_targets,
        "sources": sorted(sources),
        "deg_net": deg_net,
    }


def _solve_with_meta(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    parsed = _parse_network(g)
    network_mask = parsed["network_mask"]
    converter_targets = parsed["converter_targets"]
    sources = parsed["sources"]

    assert sources, "expected at least one source"
    assert converter_targets, "expected at least one converter"

    dist = np.full((h, w), -1, dtype=int)
    arrival_color = np.full((h, w), -1, dtype=int)
    travel_color = np.full((h, w), -1, dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    parent = {}
    q = deque()

    for idx, (r, c) in enumerate(sources):
        dist[r, c] = 0
        arrival_color[r, c] = int(g[r, c])
        travel_color[r, c] = int(g[r, c])
        owner[r, c] = idx
        parent[(r, c)] = None
        q.append((r, c))

    while q:
        r, c = q.popleft()
        cur_after = int(travel_color[r, c])
        for nr, nc in _neighbors4(r, c, h, w):
            if not bool(network_mask[nr, nc]):
                continue
            nt = int(dist[r, c]) + 1
            if int(dist[nr, nc]) != -1 and int(dist[nr, nc]) <= nt:
                continue
            dist[nr, nc] = nt
            owner[nr, nc] = int(owner[r, c])
            parent[(nr, nc)] = (r, c)
            arrival_color[nr, nc] = cur_after
            travel_color[nr, nc] = int(converter_targets.get((nr, nc), cur_after))
            q.append((nr, nc))

    for r, c in np.argwhere(network_mask):
        assert int(dist[int(r), int(c)]) >= 0, "every path cell must be reachable from a source"

    children = {cell: [] for cell in parent}
    for cell, par in parent.items():
        if par is not None:
            children.setdefault(par, []).append(cell)

    out = g.copy()
    for r, c in parsed["marker_cells"]:
        out[r, c] = BG
    for r, c in np.argwhere(network_mask):
        r, c = int(r), int(c)
        if (r, c) in converter_targets:
            out[r, c] = int(g[r, c])
        else:
            out[r, c] = int(arrival_color[r, c])

    parsed.update(
        {
            "dist": dist,
            "arrival_color": arrival_color,
            "travel_color": travel_color,
            "owner": owner,
            "parent": parent,
            "children": children,
            "output": out,
        }
    )
    return out, parsed


def _distance_map(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not bool(mask[nr, nc]) or int(dist[nr, nc]) != -1:
                continue
            dist[nr, nc] = int(dist[r, c]) + 1
            q.append((nr, nc))
    return dist


def _unique_claims(parsed: dict) -> bool:
    mask = parsed["network_mask"]
    converters = set(parsed["converter_targets"])
    dmaps = [_distance_map(mask, src) for src in parsed["sources"]]
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        if cell in converters:
            continue
        vals = [int(dm[cell]) for dm in dmaps if int(dm[cell]) >= 0]
        best = min(vals)
        if sum(v == best for v in vals) != 1:
            return False
    return True


def _rand_with_parity(rng: random.Random, lo: int, hi: int, parity: int) -> int:
    choices = [v for v in range(lo, hi + 1) if v % 2 == parity]
    return rng.choice(choices)


def _transform_point(r: int, c: int, h: int, w: int, rot: int) -> tuple[int, int]:
    if rot == 0:
        return r, c
    if rot == 1:
        return c, h - 1 - r
    if rot == 2:
        return h - 1 - r, w - 1 - c
    return w - 1 - c, r


def _build_attempt(rng: random.Random):
    n_branches = rng.randint(2, 4)

    branch_lens = [rng.randint(2, 4) for _ in range(n_branches)]
    left_len = _rand_with_parity(rng, 4, 5, 1 - (branch_lens[0] % 2))
    right_len = _rand_with_parity(rng, 4, 5, 1 - (branch_lens[-1] % 2))

    gaps = []
    for i in range(n_branches - 1):
        delta_parity = (branch_lens[i + 1] - branch_lens[i]) & 1
        gaps.append(_rand_with_parity(rng, 4, 5, 1 - delta_parity))

    dirs = [rng.choice([-1, 1]) for _ in range(n_branches)]
    if n_branches >= 3 and len(set(dirs)) == 1:
        dirs[rng.randrange(n_branches)] *= -1

    converter_count = rng.randint(1, n_branches)
    if n_branches > 1 and converter_count == n_branches and rng.random() < 0.75:
        converter_count -= 1
    converter_ids = set(rng.sample(range(n_branches), converter_count))

    source_count = n_branches + 2
    source_colors = rng.sample(PALETTE, source_count)
    left_color = source_colors[0]
    right_color = source_colors[1]
    branch_colors = source_colors[2:]
    converter_palette = [c for c in PALETTE if c not in source_colors]
    if not converter_palette:
        return None

    x = left_len
    branch_xs = []
    for i in range(n_branches):
        branch_xs.append(x)
        if i < n_branches - 1:
            x += gaps[i]
    total_w = branch_xs[-1] + right_len + 1

    top_need = max(
        [branch_lens[i] for i in range(n_branches) if dirs[i] == -1] + [1 if any(i in converter_ids and dirs[i] == 1 for i in range(n_branches)) else 0]
    )
    bottom_need = max(
        [branch_lens[i] for i in range(n_branches) if dirs[i] == 1] + [1 if any(i in converter_ids and dirs[i] == -1 for i in range(n_branches)) else 0]
    )
    total_h = top_need + bottom_need + 1
    trunk_r = top_need

    path_cells = []
    for c in range(total_w):
        path_cells.append((trunk_r, c))
    for i, xc in enumerate(branch_xs):
        step = dirs[i]
        for k in range(1, branch_lens[i] + 1):
            path_cells.append((trunk_r + step * k, xc))

    left_source = (trunk_r, 0)
    right_source = (trunk_r, total_w - 1)
    branch_sources = [(trunk_r + dirs[i] * branch_lens[i], branch_xs[i]) for i in range(n_branches)]
    converters = [(trunk_r, branch_xs[i]) for i in range(n_branches) if i in converter_ids]
    markers = []
    marker_targets = {}
    converter_colors = {}
    for i in range(n_branches):
        if i not in converter_ids:
            continue
        mr = trunk_r - dirs[i]
        mc = branch_xs[i]
        markers.append((mr, mc))
        converter_colors[(trunk_r, branch_xs[i])] = rng.choice(converter_palette)
        target_choices = [c for c in source_colors if c != branch_colors[i]]
        marker_targets[(mr, mc)] = rng.choice(target_choices)

    base_h, base_w = total_h, total_w
    rot = rng.randrange(4)
    if rot % 2 == 0:
        total_h, total_w = base_h, base_w
    else:
        total_h, total_w = base_w, base_h

    transformed_path = [_transform_point(r, c, base_h, base_w, rot) for r, c in path_cells]
    transformed_markers = [_transform_point(r, c, base_h, base_w, rot) for r, c in markers]
    transformed_left = _transform_point(*left_source, base_h, base_w, rot)
    transformed_right = _transform_point(*right_source, base_h, base_w, rot)
    transformed_branch_sources = [_transform_point(r, c, base_h, base_w, rot) for r, c in branch_sources]

    converter_lookup = {}
    for old_conv in converters:
        converter_lookup[tuple(old_conv)] = _transform_point(old_conv[0], old_conv[1], base_h, base_w, rot)
    marker_lookup = {}
    for old_marker in markers:
        marker_lookup[tuple(old_marker)] = _transform_point(old_marker[0], old_marker[1], base_h, base_w, rot)

    final_h = total_h + rng.randint(0, min(8, 30 - total_h))
    final_w = total_w + rng.randint(0, min(8, 30 - total_w))
    row_off = rng.randint(0, final_h - total_h)
    col_off = rng.randint(0, final_w - total_w)

    grid = np.full((final_h, final_w), BG, dtype=int)

    for r, c in transformed_path:
        grid[row_off + r, col_off + c] = GREY

    grid[row_off + transformed_left[0], col_off + transformed_left[1]] = left_color
    grid[row_off + transformed_right[0], col_off + transformed_right[1]] = right_color
    for color, (r, c) in zip(branch_colors, transformed_branch_sources):
        grid[row_off + r, col_off + c] = color

    for old_conv, new_conv in converter_lookup.items():
        grid[row_off + new_conv[0], col_off + new_conv[1]] = converter_colors[old_conv]
    for old_marker, new_marker in marker_lookup.items():
        grid[row_off + new_marker[0], col_off + new_marker[1]] = marker_targets[old_marker]

    return grid


def _gen_attempt(rng: random.Random):
    grid = _build_attempt(rng)
    if grid is None:
        return None

    out, meta = _solve_with_meta(grid)
    if not _unique_claims(meta):
        return None

    converter_targets = meta["converter_targets"]
    children = meta["children"]
    arrival_color = meta["arrival_color"]
    for conv, target in converter_targets.items():
        if int(arrival_color[conv]) == int(target):
            return None
        if not children.get(conv):
            return None

    changed_path_cells = 0
    for r, c in np.argwhere(meta["network_mask"]):
        cell = (int(r), int(c))
        if cell in converter_targets:
            continue
        if int(grid[cell]) == GREY and int(out[cell]) != GREY:
            changed_path_cells += 1
    if changed_path_cells < 6:
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colored signals through a grey path network and recolor each path segment by the winning signal after any converter-triggered color changes.

    Input structure:
    - The background is black, and the path network is a one-pixel-wide orthogonal set of grey corridors.
    - Every leaf of the network is a colored endpoint source, while some T-junction centers are replaced by a distinct converter color.
    - Each converter has exactly one adjacent off-path marker pixel in the surrounding black area; that marker's color is the converter's target signal color.

    Rule:
    - Every source emits a signal from its endpoint and expands step by step along the connected path network.
    - Grey path pixels are recolored by whichever signal reaches them first; when fronts approach from opposite ends, the segment splits at the earliest-arrival boundary.
    - When a signal reaches a converter junction, the converter pixel itself keeps its original converter color, but the signal's traveling color changes to the converter's adjacent marker color before continuing onward.
    - All off-path marker pixels are removed to black in the output.

    Key insight:
    - The puzzle is a simultaneous shortest-path race on a thin network, except converter junctions alter the color label of the winning signal without blocking its motion.
    """
    out, _ = _solve_with_meta(np.asarray(grid, dtype=int))
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
