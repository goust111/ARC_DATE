from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
EDGES = ("top", "bottom", "left", "right")
DISTANCES = (1, 2, 3, 4)

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(vals[counts.argmax()])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_positions(h: int, w: int, corner: str) -> list[tuple[int, int]]:
    if corner == "top_left":
        return [(r, 0) for r in range(4)]
    if corner == "top_right":
        return [(r, w - 1) for r in range(4)]
    if corner == "bottom_left":
        return [(h - 4 + i, 0) for i in range(4)]
    if corner == "bottom_right":
        return [(h - 4 + i, w - 1) for i in range(4)]
    raise ValueError(f"unknown corner {corner}")


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    fg_mask = grid != bg
    edge_components = []
    for comp in _components4(fg_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            edge_components.append(comp)

    assert len(edge_components) == 1, "expected exactly one edge-touching legend component"
    legend = edge_components[0]
    assert len(legend) == 4, "legend must contain four cells"

    r0, c0, r1, c1 = _bbox(legend)
    rows = sorted({r for r, _ in legend})
    cols = sorted({c for _, c in legend})
    assert len(rows) == 4 and rows == list(range(r0, r1 + 1))
    assert len(cols) == 1
    assert r0 in (0, h - 4) and c0 in (0, w - 1) and c1 == c0

    legend_mask = np.zeros_like(fg_mask, dtype=bool)
    for r, c in legend:
        legend_mask[r, c] = True

    mapping = {idx + 1: int(grid[row, c0]) for idx, row in enumerate(rows)}
    return mapping, legend_mask


def _halo_layers(
    component: list[tuple[int, int]],
    bg_mask: np.ndarray,
    layers: int,
) -> list[list[tuple[int, int]]]:
    h, w = bg_mask.shape
    frontier = set(component)
    visited = set(component)
    result: list[list[tuple[int, int]]] = []

    for _ in range(layers):
        new_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if not bg_mask[nr, nc]:
                    continue
                cell = (nr, nc)
                if cell in visited:
                    continue
                visited.add(cell)
                new_frontier.add(cell)
        result.append(sorted(new_frontier))
        frontier = new_frontier
        if not frontier:
            break
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a same-color halo around each object.

    The halo depth equals the object's nearest-edge bbox distance d, and the
    halo color comes from the 4x1 corner legend row corresponding to d.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    distance_to_color, legend_mask = _detect_legend(g, bg)
    object_mask = (g != bg) & ~legend_mask
    components = _components4(object_mask)

    out = g.copy()
    bg_mask = g == bg
    h, w = g.shape

    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "objects must be monochrome"
        r0, c0, r1, c1 = _bbox(comp)
        d = min(r0, c0, h - 1 - r1, w - 1 - c1)
        assert 1 <= d <= 4, "object distance must be within 1..4"
        halo_color = distance_to_color[d]
        for layer in _halo_layers(comp, bg_mask, d):
            for r, c in layer:
                if out[r, c] == bg:
                    out[r, c] = halo_color
    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _reorient(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return _crop(out)


def _random_polyomino(rng: np.random.Generator, height: int, width: int, cells: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(rng.integers(height)), int(rng.integers(width)))
    mask[start] = True
    points = {start}
    while len(points) < cells:
        frontier = []
        for r, c in points:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True
        points.add((nr, nc))
    return _crop(mask)


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    style = str(rng.choice(["rect", "line", "ell", "tee", "poly"]))

    if style == "rect":
        h = int(rng.integers(1, 4))
        w = int(rng.integers(1, 5))
        mask = np.ones((h, w), dtype=bool)
        return _reorient(mask, rng)

    if style == "line":
        length = int(rng.integers(2, 6))
        if rng.random() < 0.5:
            mask = np.ones((1, length), dtype=bool)
        else:
            mask = np.ones((length, 1), dtype=bool)
        return _reorient(mask, rng)

    if style == "ell":
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        return _reorient(mask, rng)

    if style == "tee":
        w = int(rng.choice([3, 5]))
        h = int(rng.choice([2, 3]))
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, w // 2] = True
        return _reorient(mask, rng)

    h = int(rng.integers(2, 5))
    w = int(rng.integers(2, 5))
    cells = int(rng.integers(3, min(7, h * w) + 1))
    return _reorient(_random_polyomino(rng, h, w, cells), rng)


def _candidate_positions(
    grid_h: int,
    grid_w: int,
    shape_h: int,
    shape_w: int,
    edge: str,
    distance: int,
) -> list[tuple[int, int]]:
    if edge == "top":
        if grid_h - shape_h - distance < distance or grid_w - shape_w - distance < distance:
            return []
        return [(distance, c) for c in range(distance, grid_w - shape_w - distance + 1)]
    if edge == "bottom":
        r0 = grid_h - shape_h - distance
        if r0 < distance or grid_w - shape_w - distance < distance:
            return []
        return [(r0, c) for c in range(distance, grid_w - shape_w - distance + 1)]
    if edge == "left":
        if grid_w - shape_w - distance < distance or grid_h - shape_h - distance < distance:
            return []
        return [(r, distance) for r in range(distance, grid_h - shape_h - distance + 1)]
    if edge == "right":
        c0 = grid_w - shape_w - distance
        if c0 < distance or grid_h - shape_h - distance < distance:
            return []
        return [(r, c0) for r in range(distance, grid_h - shape_h - distance + 1)]
    raise ValueError(f"unknown edge {edge}")


def _shape_cells(shape: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for r, c in np.argwhere(shape):
        cells.append((top + int(r), left + int(c)))
    return cells


def _reserve_region(
    cells: list[tuple[int, int]],
    radius: int,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    region: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h - 1, r + radius) + 1):
            for cc in range(max(0, c - radius), min(w - 1, c + radius) + 1):
                region.add((rr, cc))
    return region


def _choose_object_colors(
    rng: np.random.Generator,
    count: int,
    bg: int,
    legend_colors: list[int],
) -> list[int]:
    palette = [c for c in range(10) if c != bg and c not in legend_colors]
    rng.shuffle(palette)
    colors = palette[:count]
    if count >= 2 and rng.random() < 0.35:
        src = int(rng.integers(count))
        dst = int(rng.integers(count))
        if src != dst:
            colors[dst] = colors[src]
    return colors


def _sample_specs(
    rng: np.random.Generator,
    bg: int,
    legend_colors: list[int],
) -> list[dict[str, object]]:
    count = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    distances = list(map(int, rng.choice(np.array(DISTANCES), size=count, replace=False)))
    edges = list(map(str, rng.permutation(np.array(EDGES))[:count]))
    colors = _choose_object_colors(rng, count, bg, legend_colors)

    specs = []
    for distance, edge, color in zip(distances, edges, colors):
        specs.append(
            {
                "distance": distance,
                "edge": edge,
                "color": color,
                "shape": _random_shape(rng),
            }
        )
    specs.sort(key=lambda spec: (-int(spec["distance"]), -int(np.sum(spec["shape"]))))
    return specs


def _place_specs(
    rng: np.random.Generator,
    h: int,
    w: int,
    legend_cells: list[tuple[int, int]],
    specs: list[dict[str, object]],
) -> list[tuple[dict[str, object], list[tuple[int, int]]]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in legend_cells:
        blocked[r, c] = True

    placements: list[tuple[dict[str, object], list[tuple[int, int]]]] = []
    for spec in specs:
        distance = int(spec["distance"])
        shape = np.asarray(spec["shape"], dtype=bool)
        sh, sw = shape.shape
        candidates = _candidate_positions(h, w, sh, sw, str(spec["edge"]), distance)
        if not candidates:
            return None

        order = rng.permutation(len(candidates))
        placed = False
        for idx in order:
            top, left = candidates[int(idx)]
            cells = _shape_cells(shape, top, left)
            reserve = _reserve_region(cells, distance, h, w)
            if any(blocked[r, c] for r, c in reserve):
                continue

            r0, c0, r1, c1 = _bbox(cells)
            if min(r0, c0, h - 1 - r1, w - 1 - c1) != distance:
                continue

            for r, c in reserve:
                blocked[r, c] = True
            placements.append((spec, cells))
            placed = True
            break

        if not placed:
            return None

    return placements


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    bg = int(rng.integers(10))

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    legend_colors = palette[:4]
    corner = str(rng.choice(CORNERS))
    legend_cells = _legend_positions(h, w, corner)

    specs = _sample_specs(rng, bg, legend_colors)
    placements = _place_specs(rng, h, w, legend_cells, specs)
    if placements is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(legend_cells, legend_colors):
        grid[r, c] = color
    for spec, cells in placements:
        for r, c in cells:
            grid[r, c] = int(spec["color"])

    try:
        mapping, legend_mask = _detect_legend(grid, bg)
        assert mapping == {i + 1: legend_colors[i] for i in range(4)}
        object_mask = (grid != bg) & ~legend_mask
        comps = _components4(object_mask)
        assert len(comps) == len(placements)
        for comp in comps:
            colors = {int(grid[r, c]) for r, c in comp}
            assert len(colors) == 1
            r0, c0, r1, c1 = _bbox(comp)
            d = min(r0, c0, h - 1 - r1, w - 1 - c1)
            assert d in DISTANCES
        out = main(grid)
        assert not np.array_equal(out, grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
