import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            queue = deque([(r, c)])
            labels[r, c] = next_id
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
            next_id += 1
    return labels, components


def _extract_fracture_segments(fracture: np.ndarray) -> list[tuple[str, int, int, int]]:
    h, w = fracture.shape
    segments: list[tuple[str, int, int, int]] = []

    for r in range(h):
        c = 0
        while c < w:
            if not fracture[r, c]:
                c += 1
                continue
            c0 = c
            while c + 1 < w and fracture[r, c + 1]:
                c += 1
            if c - c0 + 1 >= 2:
                segments.append(("h", r, c0, c))
            c += 1

    for c in range(w):
        r = 0
        while r < h:
            if not fracture[r, c]:
                r += 1
                continue
            r0 = r
            while r + 1 < h and fracture[r + 1, c]:
                r += 1
            if r - r0 + 1 >= 2:
                segments.append(("v", c, r0, r))
            r += 1

    return segments


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    blue = g == BLUE
    fracture = g == GREY
    labels, components = _label_components(blue)
    n_comp = len(components)
    assert n_comp >= 1, "expected at least one blue fragment"

    segments = _extract_fracture_segments(fracture)
    shifts = {idx + 1: [0, 0] for idx in range(n_comp)}
    graph: dict[int, set[int]] = {idx + 1: set() for idx in range(n_comp)}

    for orient, fixed, a0, a1 in segments:
        if orient == "h":
            r = fixed
            side_a = {int(labels[r - 1, c]) for c in range(a0, a1 + 1) if r - 1 >= 0 and labels[r - 1, c] > 0}
            side_b = {int(labels[r + 1, c]) for c in range(a0, a1 + 1) if r + 1 < g.shape[0] and labels[r + 1, c] > 0}
            for comp_id in side_a:
                shifts[comp_id][0] -= 1
            for comp_id in side_b:
                shifts[comp_id][0] += 1
        else:
            c = fixed
            side_a = {int(labels[r, c - 1]) for r in range(a0, a1 + 1) if c - 1 >= 0 and labels[r, c - 1] > 0}
            side_b = {int(labels[r, c + 1]) for r in range(a0, a1 + 1) if c + 1 < g.shape[1] and labels[r, c + 1] > 0}
            for comp_id in side_a:
                shifts[comp_id][1] -= 1
            for comp_id in side_b:
                shifts[comp_id][1] += 1

        assert side_a and side_b, "each fracture segment must separate blue pieces on both sides"
        for left_id in side_a:
            for right_id in side_b:
                if left_id != right_id:
                    graph[left_id].add(right_id)
                    graph[right_id].add(left_id)

    blue_coords = np.argwhere(blue)
    root_idx = np.lexsort((blue_coords[:, 1], blue_coords[:, 0]))[0]
    root_r, root_c = map(int, blue_coords[root_idx])
    root = int(labels[root_r, root_c])

    parity = {root: 0}
    queue = deque([root])
    while queue:
        comp_id = queue.popleft()
        for neighbor in graph[comp_id]:
            next_parity = 1 - parity[comp_id]
            if neighbor in parity:
                assert parity[neighbor] == next_parity, "fragment graph must be bipartite"
            else:
                parity[neighbor] = next_parity
                queue.append(neighbor)
    assert len(parity) == n_comp, "all fragments must be connected by fracture adjacency"

    moved_coords: dict[int, np.ndarray] = {}
    for comp_id, coords in enumerate(components, start=1):
        dr, dc = shifts[comp_id]
        moved_coords[comp_id] = coords + np.array([dr, dc], dtype=int)

    return {
        "labels": labels,
        "components": components,
        "segments": segments,
        "graph": graph,
        "shifts": shifts,
        "parity": parity,
        "moved_coords": moved_coords,
        "root": root,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a cracked blue object, slide each fragment away from its cracks, then bipartite-color the fragment graph.

    Input structure:
    - Black background surrounds one large blue object.
    - Grey one-cell fracture lines lie inside the object and run horizontally or vertically.
    - Removing the grey pixels separates the blue area into several 4-connected fragments.

    Rule:
    - Delete every grey fracture pixel.
    - For each horizontal fracture, move blue fragments touching its upper side up by 1 and fragments touching its lower side down by 1.
    - For each vertical fracture, move blue fragments touching its left side left by 1 and fragments touching its right side right by 1.
    - Connect two fragments in a graph whenever they touched opposite sides of the same fracture line before removal.
    - Color the fragment containing the original object's top-left blue pixel green, then alternate yellow/green across the fracture-adjacency graph.

    Key insight:
    - The same grey fracture pixels simultaneously define where the object splits, which way each fragment slides, and which fragment pairs are graph neighbors.
    """
    analysis = _analyze(grid)
    out = np.zeros_like(np.asarray(grid, dtype=int))
    for comp_id, coords in analysis["moved_coords"].items():
        color = GREEN if analysis["parity"][comp_id] == 0 else YELLOW
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "translated fragment left the grid"
            assert out[rr, cc] == BG, "translated fragments must not overlap"
            out[rr, cc] = color
    return out


def _split_lengths(rng: random.Random, total: int, count: int, minimum: int = 2) -> list[int]:
    parts = [minimum] * count
    extra = total - minimum * count
    if extra < 0:
        raise ValueError("total too small for requested partition")
    for _ in range(extra):
        parts[rng.randrange(count)] += 1
    return parts


def _generate_local_object(rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        pattern = rng.choices(["parallel_h", "parallel_v", "cross"], weights=[4, 4, 5], k=1)[0]

        if pattern == "cross":
            height = rng.randint(10, 15)
            width = rng.randint(10, 15)
            local = np.full((height, width), BLUE, dtype=int)
            cut_r = rng.randint(2, height - 3)
            cut_c = rng.randint(2, width - 3)
            local[cut_r, :] = GREY
            local[:, cut_c] = GREY
        else:
            n_lines = rng.choices([2, 3, 4], weights=[3, 4, 3], k=1)[0]
            if pattern == "parallel_h":
                height = rng.randint(max(10, 3 * n_lines + 2), 15)
                width = rng.randint(9, 15)
                local = np.full((height, width), BLUE, dtype=int)
                band_heights = _split_lengths(rng, height - n_lines, n_lines + 1, minimum=2)
                row = band_heights[0]
                for idx in range(n_lines):
                    local[row, :] = GREY
                    row += 1 + band_heights[idx + 1]
            else:
                height = rng.randint(9, 15)
                width = rng.randint(max(10, 3 * n_lines + 2), 15)
                local = np.full((height, width), BLUE, dtype=int)
                band_widths = _split_lengths(rng, width - n_lines, n_lines + 1, minimum=2)
                col = band_widths[0]
                for idx in range(n_lines):
                    local[:, col] = GREY
                    col += 1 + band_widths[idx + 1]

        try:
            analysis = _analyze(local)
        except AssertionError:
            continue
        component_areas = [int(coords.shape[0]) for coords in analysis["components"]]
        edge_count = sum(len(neighbors) for neighbors in analysis["graph"].values()) // 2
        mixed_shift = any(abs(dr) + abs(dc) >= 2 or (dr != 0 and dc != 0) for dr, dc in analysis["shifts"].values())
        axes_present = {segment[0] for segment in analysis["segments"]}
        if not (3 <= len(analysis["components"]) <= 5):
            continue
        if min(component_areas) < 4:
            continue
        if pattern == "cross":
            if len(analysis["components"]) != 4:
                continue
            if edge_count != 4:
                continue
        elif edge_count != len(analysis["components"]) - 1:
            continue
        if pattern != "cross" and len(axes_present) != 1:
            continue
        if pattern == "cross" and axes_present != {"h", "v"}:
            continue
        if len(axes_present) == 1 and len(analysis["components"]) >= 4 and rng.random() < 0.4:
            continue
        if pattern == "cross" and not mixed_shift:
            continue

        return local

    return None


def _embed_local(grid_shape: tuple[int, int], local: np.ndarray, top: int, left: int) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    h, w = local.shape
    grid[top : top + h, left : left + w] = local
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BG, BLUE, GREY})

    analysis = _analyze(g)
    output = main(g)
    assert set(np.unique(output)).issubset({BG, GREEN, YELLOW})
    assert len(analysis["components"]) in {3, 4, 5}
    assert min(int(coords.shape[0]) for coords in analysis["components"]) >= 4
    assert sum(len(neighbors) for neighbors in analysis["graph"].values()) // 2 == len(analysis["components"]) - 1
    assert output.shape == g.shape
    assert int((g == BLUE).sum()) == int((output != BG).sum())

    seen = set()
    for coords in analysis["moved_coords"].values():
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            assert (rr, cc) not in seen
            seen.add((rr, cc))

    root_color = GREEN
    for r, c in analysis["moved_coords"][analysis["root"]]:
        assert output[int(r), int(c)] == root_color


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        local = _generate_local_object(rng)
        if local is None:
            continue

        analysis = _analyze(local)
        shift_rows = [dr for dr, _dc in analysis["shifts"].values()]
        shift_cols = [dc for _dr, dc in analysis["shifts"].values()]
        min_dr, max_dr = min(shift_rows), max(shift_rows)
        min_dc, max_dc = min(shift_cols), max(shift_cols)

        pad = rng.randint(2, 4)
        extra_h = rng.randint(0, 4)
        extra_w = rng.randint(0, 4)
        full_h = local.shape[0] + (max_dr - min_dr) + 2 * pad + extra_h
        full_w = local.shape[1] + (max_dc - min_dc) + 2 * pad + extra_w
        if full_h > 30 or full_w > 30:
            continue

        top_base = pad - min_dr
        left_base = pad - min_dc
        top = top_base + rng.randint(0, extra_h)
        left = left_base + rng.randint(0, extra_w)
        grid = _embed_local((full_h, full_w), local, top, left)

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid fracture-split puzzle")
