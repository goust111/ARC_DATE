from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PATH = 5
SIGNAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _mode(values: list[int]) -> int:
    counts = Counter(map(int, values))
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _infer_bg_and_grey(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    bg = _mode([int(g[0, 0]), int(g[0, w - 1]), int(g[h - 1, 0]), int(g[h - 1, w - 1])])
    counts = Counter(map(int, g.flat))
    non_bg = [color for color in counts if color != bg]
    assert non_bg, "expected non-background pixels"
    grey = max(non_bg, key=lambda color: (counts[color], -color))
    return int(bg), int(grey)


def _extract_square_and_source(mask: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], int]:
    pts = np.argwhere(mask)
    assert len(pts) >= 5, "signal color must contain a solid square plus one source pixel"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    area = int(len(pts))
    best = None

    for side in range(int(area**0.5), 1, -1):
        for r in range(int(r0), int(r1) - side + 2):
            for c in range(int(c0), int(c1) - side + 2):
                if not bool(mask[r : r + side, c : c + side].all()):
                    continue
                rem = mask.copy()
                rem[r : r + side, c : c + side] = False
                rem_pts = np.argwhere(rem)
                if len(rem_pts) != 1:
                    continue
                source = (int(rem_pts[0, 0]), int(rem_pts[0, 1]))
                square = {(rr, cc) for rr in range(r, r + side) for cc in range(c, c + side)}
                best = (square, source, int(side))
                break
            if best is not None:
                break
        if best is not None:
            break

    assert best is not None, "each signal color must contain one maximal solid square"
    return best


def _parse_signals(g: np.ndarray, bg: int, grey: int):
    h, w = g.shape
    signals = []
    square_mask = np.zeros_like(g, dtype=bool)
    seen_source_colors = set()

    colors = [int(color) for color in np.unique(g) if int(color) not in {bg, grey}]
    for color in colors:
        mask = g == color
        square, source, energy = _extract_square_and_source(mask)
        assert source not in square
        assert any(source == (nr, nc) for rr, cc in square for nr, nc in _neighbors4(rr, cc)), \
            "source must touch its encoding square"
        assert any(
            0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == grey
            for nr, nc in _neighbors4(*source)
        ), "source must lie on the grey path"
        for r, c in square:
            square_mask[r, c] = True
        signals.append(
            {
                "color": int(color),
                "source": (int(source[0]), int(source[1])),
                "energy": int(energy),
                "square": square,
            }
        )
        seen_source_colors.add(int(color))

    path_mask = g == grey
    for spec in signals:
        sr, sc = spec["source"]
        path_mask[sr, sc] = True

    recognized = (g == bg) | path_mask | square_mask
    assert bool(recognized.all()), "grid contains unrecognized pixels"
    assert len(seen_source_colors) == len(signals), "signal colors must be unique"
    return signals, path_mask, square_mask


def _distances(path_mask: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    h, w = path_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = source
    assert path_mask[sr, sc]
    dist[sr, sc] = 0
    q: deque[tuple[int, int]] = deque([source])

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _paint_signals(
    shape: tuple[int, int],
    bg: int,
    grey: int,
    path_mask: np.ndarray,
    square_mask: np.ndarray,
    signals: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    claim = np.full((h, w), -1, dtype=int)
    claimed = np.zeros((h, w), dtype=bool)
    conflict = np.zeros((h, w), dtype=bool)

    for spec in signals:
        dist = _distances(path_mask, spec["source"])
        reach = (dist >= 0) & (dist <= int(spec["energy"]))
        for r, c in np.argwhere(reach):
            rr = int(r)
            cc = int(c)
            if not claimed[rr, cc]:
                claimed[rr, cc] = True
                claim[rr, cc] = int(spec["color"])
            elif claim[rr, cc] != int(spec["color"]):
                conflict[rr, cc] = True

    out = np.full((h, w), bg, dtype=int)
    out[path_mask] = grey
    for r, c in np.argwhere(path_mask):
        rr = int(r)
        cc = int(c)
        if conflict[rr, cc]:
            out[rr, cc] = grey
        elif claimed[rr, cc]:
            out[rr, cc] = int(claim[rr, cc])
    out[square_mask] = bg
    return out, claimed, conflict


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colored signals along a maze path using nearby square sizes as travel budgets.

    Input structure:
    - The background is black.
    - Grey pixels form a single-pixel-wide 4-connected path network with straight runs, turns, and T/cross junctions.
    - Some path pixels are replaced by uniquely colored signal-source pixels.
    - Each source touches a same-colored solid square in the surrounding black area; the square side length encodes that signal's propagation budget.

    Rule:
    - Recover every source's energy from the side length of its adjacent solid square.
    - Treat the grey path plus the colored source pixels as the passable network, but do not allow travel through the encoding squares.
    - Each source colors every path pixel within its budgeted path distance from the source.
    - If exactly one signal reaches a path pixel, that pixel takes the signal color; if multiple different signals reach it, the pixel stays grey.
    - Erase every encoding square back to black in the output.

    Key insight:
    - The colored squares are not objects to keep; they are numeric encodings for how far each source may spread through the path graph before conflicts cancel colors back to grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, grey = _infer_bg_and_grey(g)
    signals, path_mask, square_mask = _parse_signals(g, bg, grey)
    out, _claimed, _conflict = _paint_signals(g.shape, bg, grey, path_mask, square_mask, signals)
    return out


def _node_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _steps_to_boundary(node: tuple[int, int], direction: tuple[int, int], rows: int, cols: int) -> int:
    r, c = node
    dr, dc = direction
    steps = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < rows and 0 <= c < cols):
            break
        steps += 1
    return steps


def _sample_path_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    total = rows * cols
    max_len = min(total, 8)
    for _ in range(200):
        start = (random.randrange(rows), random.randrange(cols))
        target_len = random.randint(5, max_len)
        nodes = [start]
        used = {start}
        while len(nodes) < target_len:
            cur = nodes[-1]
            nbrs = [nb for nb in _node_neighbors(cur, rows, cols) if nb not in used]
            if not nbrs:
                break
            random.shuffle(nbrs)
            nodes.append(nbrs[0])
            used.add(nbrs[0])
        if len(nodes) < 5:
            continue
        dirs = [(nodes[i + 1][0] - nodes[i][0], nodes[i + 1][1] - nodes[i][1]) for i in range(len(nodes) - 1)]
        if len(set(dirs)) < 2:
            continue
        return list(zip(nodes, nodes[1:], strict=False))
    return None


def _sample_t_or_cross_edges(rows: int, cols: int, want_cross: bool) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    if rows < 3 or cols < 3:
        return None
    center = (random.randint(1, rows - 2), random.randint(1, cols - 2))
    directions = DIR4[:] if want_cross else random.sample(DIR4, 3)
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    used_nodes = {center}
    arm_endpoints: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for dr, dc in directions:
        max_len = min(2, _steps_to_boundary(center, (dr, dc), rows, cols))
        if max_len < 1:
            return None
        length = random.randint(1, max_len)
        prev = center
        for _ in range(length):
            nxt = (prev[0] + dr, prev[1] + dc)
            edges.append((prev, nxt))
            used_nodes.add(nxt)
            prev = nxt
        arm_endpoints.append((prev, (dr, dc)))

    # Add a bent tail on one arm to make corners appear in branched layouts.
    if random.random() < 0.75:
        random.shuffle(arm_endpoints)
        for endpoint, (dr, dc) in arm_endpoints:
            options = [d for d in DIR4 if d != (dr, dc) and d != (-dr, -dc)]
            random.shuffle(options)
            placed = False
            for bdr, bdc in options:
                max_len = min(2, _steps_to_boundary(endpoint, (bdr, bdc), rows, cols))
                if max_len < 1:
                    continue
                trial = []
                prev = endpoint
                ok = True
                for _ in range(random.randint(1, max_len)):
                    nxt = (prev[0] + bdr, prev[1] + bdc)
                    if nxt in used_nodes:
                        ok = False
                        break
                    trial.append((prev, nxt))
                    prev = nxt
                if not ok:
                    continue
                for a, b in trial:
                    edges.append((a, b))
                    used_nodes.add(b)
                placed = True
                if placed:
                    break
            if placed:
                break

    return edges


def _render_path(rows: int, cols: int, spacing: int, margin: int, edges):
    height = 2 * margin + (rows - 1) * spacing + 1
    width = 2 * margin + (cols - 1) * spacing + 1
    node_cells = {(r, c): (margin + r * spacing, margin + c * spacing) for r in range(rows) for c in range(cols)}
    path = set()
    for a, b in edges:
        r0, c0 = node_cells[a]
        r1, c1 = node_cells[b]
        if r0 == r1:
            for cc in range(min(c0, c1), max(c0, c1) + 1):
                path.add((r0, cc))
        else:
            for rr in range(min(r0, r1), max(r0, r1) + 1):
                path.add((rr, c0))
    return height, width, path


def _attachment_cells(source: tuple[int, int], side: tuple[int, int], size: int) -> set[tuple[int, int]]:
    r, c = source
    dr, dc = side
    if dr == -1:
        r0, c0 = r - size, c - size // 2
    elif dr == 1:
        r0, c0 = r + 1, c - size // 2
    elif dc == -1:
        r0, c0 = r - size // 2, c - size
    else:
        r0, c0 = r - size // 2, c + 1
    return {(r0 + rr, c0 + cc) for rr in range(size) for cc in range(size)}


def _candidate_placements(path: set[tuple[int, int]], height: int, width: int):
    path_mask = np.zeros((height, width), dtype=bool)
    for r, c in path:
        path_mask[r, c] = True

    placements = []
    for source in path:
        for side in DIR4:
            for size in (2, 3):
                cells = _attachment_cells(source, side, size)
                if any(not (0 <= r < height and 0 <= c < width) for r, c in cells):
                    continue
                if any(path_mask[r, c] for r, c in cells):
                    continue
                touches = set()
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < height and 0 <= nc < width and path_mask[nr, nc]:
                            touches.add((nr, nc))
                if touches != {source}:
                    continue
                placements.append(
                    {
                        "source": source,
                        "side": side,
                        "size": size,
                        "cells": cells,
                    }
                )
    return placements


def _touches_existing_square(cells: set[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
    if cells & used:
        return True
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in used:
                return True
    return False


def _select_signals(
    path: set[tuple[int, int]],
    placements: list[dict],
    num_sources: int,
) -> list[dict] | None:
    for _ in range(250):
        shuffled = placements[:]
        random.shuffle(shuffled)
        chosen = []
        used_sources = set()
        used_square_cells = set()
        for cand in shuffled:
            if cand["source"] in used_sources:
                continue
            if any(abs(cand["source"][0] - sr) + abs(cand["source"][1] - sc) <= 1 for sr, sc in used_sources):
                continue
            if _touches_existing_square(cand["cells"], used_square_cells):
                continue
            chosen.append(cand)
            used_sources.add(cand["source"])
            used_square_cells |= cand["cells"]
            if len(chosen) == num_sources:
                break
        if len(chosen) == num_sources:
            colors = random.sample(SIGNAL_COLORS, num_sources)
            return [
                {
                    "color": int(colors[i]),
                    "source": chosen[i]["source"],
                    "energy": int(chosen[i]["size"]),
                    "square": set(chosen[i]["cells"]),
                }
                for i in range(num_sources)
            ]
    return None


def _build_grid(height: int, width: int, path: set[tuple[int, int]], signals: list[dict]) -> np.ndarray:
    g = np.full((height, width), BG, dtype=int)
    for r, c in path:
        g[r, c] = PATH
    for spec in signals:
        for r, c in spec["square"]:
            g[r, c] = int(spec["color"])
        sr, sc = spec["source"]
        g[sr, sc] = int(spec["color"])
    return g


def _path_has_degree(path: set[tuple[int, int]], degree: int) -> bool:
    for r, c in path:
        deg = sum((nr, nc) in path for nr, nc in _neighbors4(r, c))
        if deg == degree:
            return True
    return False


def _generate_candidate() -> np.ndarray | None:
    network_type = random.choice(["path", "t", "cross"])
    want_conflict = random.choice([False, True])
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    spacing = random.randint(4, 5)
    margin = 4

    if network_type == "path":
        edges = _sample_path_edges(rows, cols)
    elif network_type == "t":
        edges = _sample_t_or_cross_edges(rows, cols, want_cross=False)
    else:
        edges = _sample_t_or_cross_edges(rows, cols, want_cross=True)
    if edges is None:
        return None

    height, width, path = _render_path(rows, cols, spacing, margin, edges)
    if max(height, width) > 30:
        return None
    if len(path) < 15:
        return None
    if network_type == "t" and not _path_has_degree(path, 3):
        return None
    if network_type == "cross" and not _path_has_degree(path, 4):
        return None
    if network_type == "path" and (_path_has_degree(path, 3) or _path_has_degree(path, 4)):
        return None

    placements = _candidate_placements(path, height, width)
    distinct_sources = len({tuple(cand["source"]) for cand in placements})
    max_sources = min(4, distinct_sources)
    if max_sources < 2:
        return None

    for _ in range(300):
        num_sources = random.randint(2, max_sources)
        signals = _select_signals(path, placements, num_sources)
        if signals is None:
            continue
        grid = _build_grid(height, width, path, signals)
        square_mask = np.zeros_like(grid, dtype=bool)
        path_mask = np.zeros_like(grid, dtype=bool)
        for r, c in path:
            path_mask[r, c] = True
        for spec in signals:
            for r, c in spec["square"]:
                square_mask[r, c] = True
        out, _claimed, conflict = _paint_signals(grid.shape, BG, PATH, path_mask, square_mask, signals)
        has_conflict = bool(np.any(conflict & path_mask))
        if has_conflict != want_conflict:
            continue

        grey_count = int(np.count_nonzero(grid == PATH))
        largest_signal = max(int(np.count_nonzero(grid == spec["color"])) for spec in signals)
        if grey_count <= largest_signal:
            continue

        # Validate that the parser used by main recovers the same structure.
        parsed_signals, parsed_path_mask, parsed_square_mask = _parse_signals(grid, BG, PATH)
        reparsed_out, _claimed2, _conflict2 = _paint_signals(
            grid.shape,
            BG,
            PATH,
            parsed_path_mask,
            parsed_square_mask,
            parsed_signals,
        )
        if not np.array_equal(out, reparsed_out):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_candidate()
        if grid is None:
            continue
        return grid
    raise AssertionError("failed to generate a valid puzzle instance")
