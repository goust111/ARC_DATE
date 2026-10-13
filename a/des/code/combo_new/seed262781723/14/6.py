from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
LEGEND_FRAME = 5
LEGEND_CROSS = 3
COLOR_POOL = [1, 2, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400

FRAGMENT_TEMPLATES = {
    3: [[(0, 0), (1, 0), (1, 1)]],
    4: [[(0, 0), (1, 0), (1, 1), (2, 1)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
}


def neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    corners = [(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)]
    for r0, c0 in corners:
        if r0 < 0 or c0 < 0:
            continue
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        if patch.shape != (3, 3):
            continue
        corner_vals = {
            int(patch[0, 0]),
            int(patch[0, 2]),
            int(patch[2, 0]),
            int(patch[2, 2]),
        }
        cross_vals = {
            int(patch[1, 1]),
            int(patch[0, 1]),
            int(patch[1, 0]),
            int(patch[1, 2]),
            int(patch[2, 1]),
        }
        if len(corner_vals) != 1 or len(cross_vals) != 1:
            continue
        frame_color = next(iter(corner_vals))
        cross_color = next(iter(cross_vals))
        if frame_color == bg or cross_color == bg or frame_color == cross_color:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[r0 : r0 + 3, c0 : c0 + 3] = True
        return mask, frame_color, cross_color
    raise AssertionError("legend not found")


def _component_endpoints(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pset = set(points)
    endpoints = []
    for r, c in points:
        degree = sum((nr, nc) in pset for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if degree == 1:
            endpoints.append((r, c))
    return endpoints


def _draw_scaled_cross(out: np.ndarray, center: tuple[int, int], arm_len: int, color: int):
    h, w = out.shape
    r0, c0 = center
    if 0 <= r0 < h and 0 <= c0 < w:
        out[r0, c0] = color
    for dr, dc in NEIGHBORS4:
        for step in range(1, arm_len + 1):
            nr, nc = r0 + dr * step, c0 + dc * step
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair marker-linked path fragments, count each reconstructed path, and stamp scaled cross symbols at the free ends.

    Input structure:
    - The background is black.
    - Away from one corner legend, the foreground is split into short 4-connected chain fragments. Each fragment is one body color except that exactly one endpoint cell has been replaced by a shared marker color.
    - A 3x3 legend sits in a corner: grey corner cells frame a green cross template made of the center cell plus its four orthogonal neighbors.

    Rule:
    - Treat each connected fragment as a simple path, identify its two geometric endpoints, and use the endpoint whose color appears in exactly two fragments as the marker endpoint.
    - Two fragments sharing the same marker color belong to one full path. Their matched marker endpoints represent one overlapped join cell, so the joined path length is `len(fragment_a) + len(fragment_b) - 1`.
    - At each remaining free endpoint of that joined path, draw an enlarged copy of the legend cross centered on the endpoint, using the path body color and an arm length equal to the joined path length.
    - Keep the original fragments and the legend unchanged.

    Key insight:
    - The shared endpoint colors are pairing labels rather than part of the path body; after pairing, only the total joined length matters, and the corner legend supplies the cross-shaped template to scale.
    """
    g = np.asarray(grid, dtype=int)
    bg = Counter(map(int, g.ravel())).most_common(1)[0][0]
    legend_mask, _, _ = _find_legend(g, bg)

    active = (g != bg) & ~legend_mask
    color_counts = Counter(int(g[r, c]) for r, c in np.argwhere(active))
    marker_colors = {color for color, count in color_counts.items() if count == 2}

    fragments = []
    for pts in components_mask(active):
        endpoints = _component_endpoints(pts)
        assert len(endpoints) == 2, "each fragment must be a non-branching open chain"
        marker_candidates = [pt for pt in endpoints if int(g[pt]) in marker_colors]
        assert len(marker_candidates) == 1, "each fragment must have exactly one shared marker endpoint"
        marker_pt = marker_candidates[0]
        free_pt = endpoints[0] if endpoints[1] == marker_pt else endpoints[1]
        marker_color = int(g[marker_pt])
        body_colors = {int(g[r, c]) for r, c in pts if (r, c) != marker_pt}
        body_colors.discard(marker_color)
        assert len(body_colors) == 1, "each fragment body must be monochrome"
        body_color = next(iter(body_colors))
        fragments.append(
            {
                "marker_color": marker_color,
                "body_color": body_color,
                "size": len(pts),
                "free_pt": free_pt,
            }
        )

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for frag in fragments:
        groups[int(frag["marker_color"])].append(frag)

    out = g.copy()
    for marker_color, pair in groups.items():
        assert len(pair) == 2, f"marker color {marker_color} must connect exactly two fragments"
        body_colors = {int(item["body_color"]) for item in pair}
        assert len(body_colors) == 1, "paired fragments must share one body color"
        body_color = next(iter(body_colors))
        joined_len = int(pair[0]["size"]) + int(pair[1]["size"]) - 1
        for frag in pair:
            _draw_scaled_cross(out, tuple(frag["free_pt"]), joined_len, body_color)
    return out


def _legend_patch() -> np.ndarray:
    return np.array(
        [
            [LEGEND_FRAME, LEGEND_CROSS, LEGEND_FRAME],
            [LEGEND_CROSS, LEGEND_CROSS, LEGEND_CROSS],
            [LEGEND_FRAME, LEGEND_CROSS, LEGEND_FRAME],
        ],
        dtype=int,
    )


def _place_legend(grid: np.ndarray, corner: str) -> set[tuple[int, int]]:
    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 3
    elif corner == "bl":
        r0, c0 = h - 3, 0
    else:
        r0, c0 = h - 3, w - 3
    patch = _legend_patch()
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch
    return {(r0 + r, c0 + c) for r in range(3) for c in range(3)}


def _plus_cells(center: tuple[int, int], arm_len: int) -> set[tuple[int, int]]:
    r0, c0 = center
    pts = {(r0, c0)}
    for dr, dc in NEIGHBORS4:
        for step in range(1, arm_len + 1):
            pts.add((r0 + dr * step, c0 + dc * step))
    return pts


def _transform_path(points: list[tuple[int, int]], rot: int, reflect: bool) -> list[tuple[int, int]]:
    transformed: list[tuple[int, int]] = []
    for r, c in points:
        if reflect:
            c = -c
        for _ in range(rot):
            r, c = c, -r
        transformed.append((r, c))
    return transformed


def _choose_fragment_cells(
    rng: random.Random,
    anchor: tuple[int, int],
    length: int,
    shape: tuple[int, int],
    legend_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    own_footprint: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    options: list[list[tuple[int, int]]] = []
    for template in FRAGMENT_TEMPLATES[length]:
        for reflect in (False, True):
            for rot in range(4):
                rel = _transform_path(template, rot, reflect)
                cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in rel]
                if len(set(cells)) != len(cells):
                    continue
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if any((r, c) in legend_cells or (r, c) in occupied or (r, c) in blocked for r, c in cells):
                    continue
                if cells[-1] in own_footprint:
                    continue
                options.append(cells)
    if not options:
        return None
    return rng.choice(options)


def _paint_fragment(grid: np.ndarray, cells: list[tuple[int, int]], body_color: int, marker_color: int):
    for r, c in cells[:-1]:
        grid[r, c] = body_color
    mr, mc = cells[-1]
    grid[mr, mc] = marker_color


def _slot_centers(h: int, w: int) -> list[tuple[int, int]]:
    return [(7, 7), (7, w - 8), (h - 8, 7), (h - 8, w - 8)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(28, 30)
    w = rng.randint(28, 30)
    grid = np.full((h, w), BG, dtype=int)
    legend_cells = _place_legend(grid, rng.choice(["tl", "tr", "bl", "br"]))

    num_paths = rng.choice([1, 2])
    colors = rng.sample(COLOR_POOL, 2 * num_paths)
    body_colors = colors[:num_paths]
    marker_colors = colors[num_paths:]

    centers = _slot_centers(h, w)
    rng.shuffle(centers)
    if num_paths == 1:
        path_centers = [centers[:2]]
    else:
        path_centers = [centers[:2], centers[2:4]]

    length_options = [(3, 3), (3, 4), (4, 3)]
    path_specs = []
    reserved_footprints: dict[tuple[int, int], set[tuple[int, int]]] = {}

    for idx in range(num_paths):
        len_a, len_b = rng.choice(length_options)
        joined_len = len_a + len_b - 1
        c_a, c_b = path_centers[idx]
        reserved_footprints[c_a] = _plus_cells(c_a, joined_len)
        reserved_footprints[c_b] = _plus_cells(c_b, joined_len)
        path_specs.append(
            {
                "body_color": body_colors[idx],
                "marker_color": marker_colors[idx],
                "lengths": (len_a, len_b),
                "centers": (c_a, c_b),
                "joined_len": joined_len,
            }
        )

    footprint_sets = list(reserved_footprints.values())
    for i in range(len(footprint_sets)):
        for j in range(i + 1, len(footprint_sets)):
            if footprint_sets[i] & footprint_sets[j]:
                return None

    occupied = set(legend_cells)
    for spec in path_specs:
        body_color = int(spec["body_color"])
        marker_color = int(spec["marker_color"])
        centers_pair = spec["centers"]
        lengths_pair = spec["lengths"]
        for anchor, length in zip(centers_pair, lengths_pair):
            blocked = set()
            for other_anchor, other_fp in reserved_footprints.items():
                if other_anchor != anchor:
                    blocked.update(other_fp)
            own_fp = reserved_footprints[anchor]
            cells = _choose_fragment_cells(
                rng,
                anchor,
                int(length),
                grid.shape,
                legend_cells,
                occupied,
                blocked,
                own_fp,
            )
            if cells is None:
                return None
            _paint_fragment(grid, cells, body_color, marker_color)
            occupied.update(cells)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
