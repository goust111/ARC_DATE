from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE_CANDIDATES = (
    ("tl", "h"),
    ("tl", "v"),
    ("tr", "h"),
    ("tr", "v"),
    ("bl", "h"),
    ("bl", "v"),
    ("br", "h"),
    ("br", "v"),
)


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid, dtype=int).ravel())
    return int(counts.most_common(1)[0][0])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    next_id = 0

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or labels[r0, c0] != 0:
                continue
            next_id += 1
            labels[r0, c0] = next_id
            queue: deque[tuple[int, int]] = deque([(r0, c0)])
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and labels[rr, cc] == 0:
                        labels[rr, cc] = next_id
                        queue.append((rr, cc))
            components.append(np.asarray(coords, dtype=int))

    return labels, components


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _strip_spec(height: int, width: int, corner: str, orientation: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if corner == "tl":
        start = (0, 0)
        step = (0, 1) if orientation == "h" else (1, 0)
    elif corner == "tr":
        start = (0, width - 1)
        step = (0, -1) if orientation == "h" else (1, 0)
    elif corner == "bl":
        start = (height - 1, 0)
        step = (0, 1) if orientation == "h" else (-1, 0)
    elif corner == "br":
        start = (height - 1, width - 1)
        step = (0, -1) if orientation == "h" else (-1, 0)
    else:
        raise AssertionError("unknown corner")

    if orientation == "h":
        side = (1, 0) if corner in {"tl", "tr"} else (-1, 0)
    else:
        side = (0, 1) if corner in {"tl", "bl"} else (0, -1)
    return start, step, side


def _palette_candidate(grid: np.ndarray, bg: int, corner: str, orientation: str) -> tuple[list[int], list[tuple[int, int]]] | None:
    h, w = map(int, grid.shape)
    start, step, side = _strip_spec(h, w, corner, orientation)
    r, c = start
    colors: list[int] = []
    cells: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) != int(bg):
        colors.append(int(grid[r, c]))
        cells.append((int(r), int(c)))
        r += step[0]
        c += step[1]

    if len(colors) < 2 or len(set(colors)) != len(colors):
        return None

    for rr, cc in cells:
        sr, sc = rr + side[0], cc + side[1]
        if 0 <= sr < h and 0 <= sc < w and int(grid[sr, sc]) != int(bg):
            return None

    return colors, cells


def _find_palette(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    candidates: list[tuple[int, list[int], list[tuple[int, int]]]] = []
    for corner, orientation in PALETTE_CANDIDATES:
        cand = _palette_candidate(grid, bg, corner, orientation)
        if cand is None:
            continue
        colors, cells = cand
        candidates.append((len(colors), colors, cells))

    assert candidates, "expected a corner palette strip"
    best_len = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_len]
    assert len(best) == 1, "palette detection must be unambiguous"
    _length, colors, cells = best[0]
    return colors, set(cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested monochrome closed contours using the palette strip in a corner.

    Input structure:
    - The most common color is the background.
    - A corner-attached palette strip lists one distinct color per depth, read from
      the corner inward.
    - Away from the palette, several independent nested contour groups are drawn in
      a single foreground color; the deepest layer may also be a solid core.

    Rule:
    - Infer each contour/core layer by spatial containment, not by size or position.
    - Assign depth 1 to the outermost enclosed boundary, depth 2 to the next layer,
      and so on.
    - Recolor every layer with the palette color at the matching depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    palette, palette_cells = _find_palette(g, bg)

    work = g.copy()
    for r, c in palette_cells:
        work[r, c] = bg

    fg_mask = work != bg
    fg_labels, fg_components = _label_components(fg_mask)
    assert fg_components, "expected nested foreground contours"

    bg_labels, bg_components = _label_components(~fg_mask)
    assert bg_components, "expected background regions"

    h, w = map(int, g.shape)
    exterior_ids = set(map(int, bg_labels[0, :]))
    exterior_ids |= set(map(int, bg_labels[h - 1, :]))
    exterior_ids |= set(map(int, bg_labels[:, 0]))
    exterior_ids |= set(map(int, bg_labels[:, w - 1]))
    exterior_ids.discard(0)
    assert exterior_ids, "expected at least one exterior background region"

    region_graph: dict[int, list[int]] = {idx: [] for idx in range(1, len(bg_components) + 1)}
    adjacency_by_component: list[list[int]] = []
    for component_id, coords in enumerate(fg_components, start=1):
        adjacent = set()
        for r, c in coords:
            for dr, dc in NEIGHBORS4:
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    bg_id = int(bg_labels[rr, cc])
                    if bg_id != 0:
                        adjacent.add(bg_id)
        assert 1 <= len(adjacent) <= 2, "each layer must border one or two background regions"
        neighbors = sorted(adjacent)
        adjacency_by_component.append(neighbors)
        if len(neighbors) == 2:
            a, b = neighbors
            region_graph[a].append(b)
            region_graph[b].append(a)

    region_depth: dict[int, int] = {bg_id: 0 for bg_id in exterior_ids}
    queue: deque[int] = deque(sorted(exterior_ids))
    while queue:
        region = queue.popleft()
        for other in region_graph[region]:
            if other not in region_depth:
                region_depth[other] = region_depth[region] + 1
                queue.append(other)

    assert len(region_depth) == len(bg_components), "all background regions must be reachable from outside"

    out = g.copy()
    max_depth = 0
    for coords, neighbors in zip(fg_components, adjacency_by_component, strict=False):
        if len(neighbors) == 2:
            depth = max(region_depth[neighbors[0]], region_depth[neighbors[1]])
        else:
            depth = region_depth[neighbors[0]] + 1
        assert 1 <= depth <= len(palette), "layer depth must match the palette"
        max_depth = max(max_depth, depth)
        out[coords[:, 0], coords[:, 1]] = int(palette[depth - 1])

    assert max_depth == len(palette), "palette length must equal the nesting depth"
    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _shell_decomposition(mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    current = np.asarray(mask, dtype=bool).copy()
    shells: list[np.ndarray] = []
    erosions: list[np.ndarray] = [current.copy()]
    while current.any():
        nxt = _erode4(current)
        shells.append(current & ~nxt)
        current = nxt
        erosions.append(current.copy())
    return shells, erosions


def _component_count(mask: np.ndarray) -> int:
    _labels, comps = _label_components(mask)
    return len(comps)


def _solid_rectangle(k: int) -> np.ndarray:
    min_dim = 4 * k - 3
    h = random.randint(min_dim, min_dim + 6)
    w = random.randint(min_dim, min_dim + 8)
    return np.ones((h, w), dtype=bool)


def _solid_notched(k: int) -> np.ndarray:
    min_dim = 4 * k - 3
    h = random.randint(min_dim + 1, min_dim + 7)
    w = random.randint(min_dim + 1, min_dim + 9)
    solid = np.ones((h, w), dtype=bool)
    side_count = random.randint(1, 2)
    for side in random.sample(("top", "bottom", "left", "right"), side_count):
        if side in {"top", "bottom"}:
            depth = random.randint(1, max(1, min(3, h // 4)))
            width = random.randint(max(2, w // 4), max(2, w - 2))
            if w - width - 1 >= 1:
                start = random.randint(1, w - width - 1)
            else:
                start = 0
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            solid[rows, start : start + width] = False
        else:
            depth = random.randint(1, max(1, min(3, w // 4)))
            height = random.randint(max(2, h // 4), max(2, h - 2))
            if h - height - 1 >= 1:
                start = random.randint(1, h - height - 1)
            else:
                start = 0
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            solid[start : start + height, cols] = False
    return solid


def _solid_plus(k: int) -> np.ndarray:
    min_dim = 4 * k - 3
    h = random.randint(min_dim + 2, min_dim + 8)
    w = random.randint(min_dim + 2, min_dim + 8)
    vert_w = random.randint(min_dim, w)
    horiz_h = random.randint(min_dim, h)
    vert_c = random.randint(0, w - vert_w)
    horiz_r = random.randint(0, h - horiz_h)
    solid = np.zeros((h, w), dtype=bool)
    solid[:, vert_c : vert_c + vert_w] = True
    solid[horiz_r : horiz_r + horiz_h, :] = True
    return solid


def _base_solid(k: int) -> np.ndarray:
    family = random.choice(("rectangle", "notched", "plus"))
    if family == "rectangle":
        return _solid_rectangle(k)
    if family == "notched":
        return _solid_notched(k)
    return _solid_plus(k)


def _make_group_patch(k: int, with_core: bool) -> tuple[np.ndarray, list[tuple[np.ndarray, int]]]:
    for _ in range(200):
        solid = _base_solid(k)
        shells, erosions = _shell_decomposition(solid)
        depth_index = 2 * (k - 1)
        if len(shells) <= depth_index:
            continue

        raw_parts: list[tuple[np.ndarray, int]] = []
        valid = True
        if with_core:
            for depth in range(1, k):
                shell = shells[2 * (depth - 1)]
                if shell.sum() == 0 or _component_count(shell) != 1:
                    valid = False
                    break
                raw_parts.append((shell.copy(), depth))
            core = erosions[depth_index]
            if not valid or not core.any() or _component_count(core) != 1:
                continue
            raw_parts.append((core.copy(), k))
        else:
            for depth in range(1, k + 1):
                shell = shells[2 * (depth - 1)]
                if shell.sum() == 0 or _component_count(shell) != 1:
                    valid = False
                    break
                raw_parts.append((shell.copy(), depth))
            if not valid:
                continue

        union = np.zeros_like(solid, dtype=bool)
        for part, _depth in raw_parts:
            union |= part
        union, (r0, c0, r1, c1) = _crop_mask(union)

        placed_parts: list[tuple[np.ndarray, int]] = []
        for part, depth in raw_parts:
            cropped = part[r0 : r1 + 1, c0 : c1 + 1].copy()
            if cropped.sum() == 0:
                valid = False
                break
            placed_parts.append((cropped, depth))
        if not valid:
            continue

        if _component_count(union) != len(placed_parts):
            continue
        return union, placed_parts

    raise RuntimeError("failed to generate a valid nested group")


def _place_palette(grid: np.ndarray, colors: list[int], corner: str, orientation: str) -> tuple[set[tuple[int, int]], np.ndarray]:
    h, w = map(int, grid.shape)
    start, step, _side = _strip_spec(h, w, corner, orientation)
    cells: set[tuple[int, int]] = set()
    reserved = np.zeros_like(grid, dtype=bool)

    r, c = start
    for color in colors:
        grid[r, c] = int(color)
        cells.add((int(r), int(c)))
        r += step[0]
        c += step[1]

    rows = [rc[0] for rc in cells]
    cols = [rc[1] for rc in cells]
    r0, r1 = max(0, min(rows) - 2), min(h - 1, max(rows) + 2)
    c0, c1 = max(0, min(cols) - 2), min(w - 1, max(cols) + 2)
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    return cells, reserved


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = int(color)


def _sample_layout_params(k: int) -> tuple[int, int, int]:
    if k == 2:
        return random.randint(20, 28), random.randint(20, 30), random.randint(1, 3)
    if k == 3:
        return random.randint(24, 34), random.randint(24, 34), random.randint(1, 3)
    return random.randint(30, 36), random.randint(30, 36), random.randint(1, 2)


def _try_place_groups(
    shape: tuple[int, int],
    group_patches: list[tuple[np.ndarray, list[tuple[np.ndarray, int]]]],
    palette_reserved: np.ndarray,
    bg: int,
    outline_color: int,
    palette: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    grid = np.full((h, w), int(bg), dtype=int)
    expected = np.full((h, w), int(bg), dtype=int)
    blocked = palette_reserved.copy()

    for patch, parts in group_patches:
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(300):
            if h - ph - 2 < 2 or w - pw - 2 < 2:
                return None
            top = random.randint(2, h - ph - 2)
            left = random.randint(2, w - pw - 2)
            r0 = max(0, top - 1)
            r1 = min(h, top + ph + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + pw + 1)
            if blocked[r0:r1, c0:c1].any():
                continue

            _paint_mask(grid, patch, top, left, outline_color)
            for part, depth in parts:
                _paint_mask(expected, part, top, left, palette[depth - 1])
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    return grid, expected


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    outline = random.choice(colors)
    colors.remove(outline)

    k = random.choice((2, 3, 4))
    palette = random.sample(colors, k)
    height, width, group_count = _sample_layout_params(k)

    groups: list[tuple[np.ndarray, list[tuple[np.ndarray, int]]]] = []
    for _ in range(group_count):
        with_core = random.random() < 0.5
        groups.append(_make_group_patch(k, with_core))

    corner = random.choice(("tl", "tr", "bl", "br"))
    orientation = random.choice(("h", "v"))

    palette_grid = np.full((height, width), bg, dtype=int)
    _palette_cells, reserved = _place_palette(palette_grid, palette, corner, orientation)
    placed = _try_place_groups((height, width), groups, reserved, bg, outline, palette)
    if placed is None:
        raise RuntimeError("placement failed")

    group_grid, expected_layers = placed
    grid = np.full((height, width), bg, dtype=int)
    grid[group_grid != bg] = group_grid[group_grid != bg]
    grid[palette_grid != bg] = palette_grid[palette_grid != bg]

    expected = np.full((height, width), bg, dtype=int)
    expected[expected_layers != bg] = expected_layers[expected_layers != bg]
    expected[palette_grid != bg] = palette_grid[palette_grid != bg]
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, expected = _build_scene()
        except RuntimeError:
            continue
        if np.array_equal(grid, expected):
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
