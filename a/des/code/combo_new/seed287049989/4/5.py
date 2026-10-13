from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
PATH_COLOR = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_value(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _non_bg_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objs: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            objs.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return objs


def _is_straight_bar(obj: dict[str, object]) -> bool:
    coords = np.asarray(obj["coords"], dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return int(coords.shape[0]) == max(h, w) and min(h, w) == 1


def _neighbor_map(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = mask.shape
    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        cur: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                cur.append((nr, nc))
        nbrs[(rr, cc)] = cur
    return nbrs


def _extract_segments(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    nbrs = _neighbor_map(mask)
    if not nbrs:
        return []

    critical: set[tuple[int, int]] = set()
    for p, adj in nbrs.items():
        if len(adj) != 2:
            critical.add(p)
            continue
        a, b = adj
        if not ((a[0] == p[0] == b[0]) or (a[1] == p[1] == b[1])):
            critical.add(p)

    if not critical:
        only = sorted(nbrs)
        return [only]

    segments: list[list[tuple[int, int]]] = []
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for start in sorted(critical):
        for nxt in nbrs[start]:
            edge = (start, nxt) if start < nxt else (nxt, start)
            if edge in seen_edges:
                continue
            seg = [start, nxt]
            seen_edges.add(edge)
            prev = start
            cur = nxt
            while cur not in critical:
                options = [p for p in nbrs[cur] if p != prev]
                assert len(options) == 1, "non-critical path pixel must continue straight"
                nxt2 = options[0]
                edge = (cur, nxt2) if cur < nxt2 else (nxt2, cur)
                if edge in seen_edges:
                    break
                seen_edges.add(edge)
                seg.append(nxt2)
                prev, cur = cur, nxt2
            segments.append(seg)
    return segments


def _recolor_path(
    path_mask: np.ndarray, out_shape: tuple[int, int], bg: int, length_to_color: dict[int, int]
) -> np.ndarray:
    out = np.full(out_shape, bg, dtype=int)
    best_len = np.full(out_shape, -1, dtype=int)
    for seg in _extract_segments(path_mask):
        seg_len = len(seg)
        assert seg_len in length_to_color, f"missing palette entry for segment length {seg_len}"
        color = int(length_to_color[seg_len])
        for r, c in seg:
            if seg_len >= best_len[r, c]:
                best_len[r, c] = seg_len
                out[r, c] = color
    return out


def _degree_counts(mask: np.ndarray) -> dict[tuple[int, int], int]:
    return {p: len(adj) for p, adj in _neighbor_map(mask).items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor an orthogonal grey path by splitting it into straight segments and mapping segment lengths through a palette.

    Input structure:
    - The background is black.
    - One non-background component is a single-cell-wide grey orthogonal path with turns and one or more junctions.
    - A separate palette area sits on the right or bottom and contains multiple disconnected colored straight bars, stacked from top to bottom with strictly increasing lengths.

    Rule:
    - Identify the grey path component and the palette bars.
    - Split the path into maximal straight horizontal or vertical segments, using endpoints, turns, and junctions as segment boundaries.
    - Recolor each segment with the color of the palette bar having the same pixel length.
    - If multiple segments share a pixel, keep the color of the longest covering segment.
    - Remove the palette area from the output and keep only the recolored path canvas.

    Key insight:
    The path must be decomposed structurally before recoloring: the palette does not match whole connected components, only the straight segment lengths created by breaking the polyline at its direction changes and intersections.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_value(g)
    objects = _non_bg_objects(g, bg)
    assert objects, "grid must contain a path and palette"

    path_candidates = [obj for obj in objects if not _is_straight_bar(obj)]
    assert len(path_candidates) == 1, "expected exactly one non-bar path component"
    path_obj = path_candidates[0]
    palette = [obj for obj in objects if obj is not path_obj]
    assert palette, "palette must be non-empty"
    assert all(_is_straight_bar(obj) for obj in palette), "palette entries must be straight bars"

    length_to_color: dict[int, int] = {}
    for obj in palette:
        length = int(obj["area"])
        color = int(obj["color"])
        assert length not in length_to_color, "palette bar lengths must be unique"
        length_to_color[length] = color

    pr0, pc0, pr1, pc1 = path_obj["bbox"]
    palette_r0 = min(int(obj["bbox"][0]) for obj in palette)
    palette_c0 = min(int(obj["bbox"][1]) for obj in palette)

    if palette_r0 > pr1:
        out_shape = (palette_r0, g.shape[1])
    else:
        assert palette_c0 > pc1, "palette must be to the right or below the path"
        out_shape = (g.shape[0], palette_c0)

    path_mask = np.zeros(out_shape, dtype=bool)
    for r, c in np.asarray(path_obj["coords"], dtype=int):
        assert 0 <= int(r) < out_shape[0] and 0 <= int(c) < out_shape[1], "path must stay outside palette crop"
        path_mask[int(r), int(c)] = True

    return _recolor_path(path_mask, out_shape, bg, length_to_color)


def _adjacent_nodes(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _grow_graph(rows: int, cols: int, mode: str) -> set[tuple[tuple[int, int], tuple[int, int]]] | None:
    seed = (random.randint(1, rows - 2), random.randint(1, cols - 2))
    dirs = DIR4[:]
    if mode == "x":
        chosen = dirs
    else:
        chosen = random.sample(dirs, 3)

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    degree: dict[tuple[int, int], int] = {}

    def cap(node: tuple[int, int]) -> int:
        return 4 if mode == "x" and node == seed else 3

    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> bool:
        edge = (a, b) if a < b else (b, a)
        if edge in edges or degree.get(a, 0) >= cap(a) or degree.get(b, 0) >= cap(b):
            return False
        edges.add(edge)
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
        return True

    used = {seed}
    for dr, dc in chosen:
        nxt = (seed[0] + dr, seed[1] + dc)
        if not add_edge(seed, nxt):
            return None
        used.add(nxt)

    target = random.randint(len(used) + 1, min(rows * cols, 10))
    while len(used) < target:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for node in sorted(used):
            if degree.get(node, 0) >= cap(node):
                continue
            for nxt in _adjacent_nodes(node, rows, cols):
                if nxt in used:
                    continue
                if degree.get(nxt, 0) >= cap(nxt):
                    continue
                frontier.append((node, nxt))
        if not frontier:
            return None
        a, b = random.choice(frontier)
        if add_edge(a, b):
            used.add(b)

    return edges


def _render_graph(
    logical_rows: int,
    logical_cols: int,
    row_steps: list[int],
    col_steps: list[int],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    rows = [0]
    cols = [0]
    for step in row_steps:
        rows.append(rows[-1] + step)
    for step in col_steps:
        cols.append(cols[-1] + step)

    mask = np.zeros((rows[-1] + 1, cols[-1] + 1), dtype=bool)
    for (r0, c0), (r1, c1) in edges:
        rr0 = rows[r0]
        rr1 = rows[r1]
        cc0 = cols[c0]
        cc1 = cols[c1]
        if rr0 == rr1:
            lo, hi = sorted((cc0, cc1))
            mask[rr0, lo : hi + 1] = True
        else:
            lo, hi = sorted((rr0, rr1))
            mask[lo : hi + 1, cc0] = True
    return mask


def _shared_length_conflict(segments: list[list[tuple[int, int]]]) -> bool:
    seen: dict[tuple[int, int], set[int]] = {}
    for seg in segments:
        seg_len = len(seg)
        for p in seg:
            seen.setdefault(p, set()).add(seg_len)
    return any(len(lengths) >= 2 for lengths in seen.values())


def _path_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    return _bbox(np.argwhere(mask))


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["t", "x"])
    side = random.choice(["right", "bottom"])
    logical_rows = random.randint(3, 4)
    logical_cols = random.randint(3, 4)

    edge_lengths = random.sample([3, 4], k=random.randint(1, 2))
    row_steps = [random.choice(edge_lengths) - 1 for _ in range(logical_rows - 1)]
    col_steps = [random.choice(edge_lengths) - 1 for _ in range(logical_cols - 1)]

    edges = _grow_graph(logical_rows, logical_cols, mode)
    if edges is None:
        return None

    tight_mask = _render_graph(logical_rows, logical_cols, row_steps, col_steps, edges)
    segments = _extract_segments(tight_mask)
    if len(segments) < 4:
        return None

    degree_values = list(_degree_counts(tight_mask).values())
    if mode == "x":
        if 4 not in degree_values:
            return None
    else:
        if 3 not in degree_values or 4 in degree_values:
            return None

    unique_lengths = sorted({len(seg) for seg in segments})
    if len(unique_lengths) < 2 or len(unique_lengths) > len(PALETTE_COLORS):
        return None
    if max(unique_lengths) > 10:
        return None
    if not _shared_length_conflict(segments):
        return None

    palette_h = 2 * len(unique_lengths) - 1
    palette_w = max(unique_lengths)

    top_pad = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    tight_h, tight_w = tight_mask.shape

    if side == "right":
        main_h = max(tight_h + top_pad + bottom_pad, palette_h)
        main_w = tight_w + left_pad + right_pad
        final_h = main_h
        final_w = main_w + palette_w
    else:
        main_h = tight_h + top_pad + bottom_pad
        main_w = max(tight_w + left_pad + right_pad, palette_w)
        final_h = main_h + palette_h
        final_w = main_w

    if final_h > 30 or final_w > 30:
        return None

    main_grid = np.full((main_h, main_w), BG, dtype=int)
    path_mask = np.zeros((main_h, main_w), dtype=bool)
    path_mask[top_pad : top_pad + tight_h, left_pad : left_pad + tight_w] = tight_mask
    main_grid[path_mask] = PATH_COLOR

    path_r0, path_c0, path_r1, path_c1 = _path_bbox(path_mask)

    colors = random.sample(PALETTE_COLORS, len(unique_lengths))
    length_to_color = {length: color for length, color in zip(unique_lengths, colors)}
    expected = _recolor_path(path_mask, main_grid.shape, BG, length_to_color)

    if side == "right":
        grid = np.full((final_h, final_w), BG, dtype=int)
        grid[:, :main_w] = main_grid
        low = max(0, path_r0 - palette_h + 1)
        high = min(main_h - palette_h, path_r1)
        if low > high:
            return None
        palette_r0 = random.randint(low, high)
        for idx, length in enumerate(unique_lengths):
            rr = palette_r0 + 2 * idx
            grid[rr, main_w : main_w + length] = length_to_color[length]
    else:
        grid = np.full((final_h, final_w), BG, dtype=int)
        grid[:main_h, :] = main_grid
        palette_c0 = random.randint(0, main_w - palette_w)
        for idx, length in enumerate(unique_lengths):
            rr = main_h + 2 * idx
            grid[rr, palette_c0 : palette_c0 + length] = length_to_color[length]

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path/palette task")
