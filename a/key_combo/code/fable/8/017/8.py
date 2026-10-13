from __future__ import annotations

import random

import numpy as np


ColorGrid = np.ndarray
Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: ColorGrid) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _filled_square_bbox(coords: np.ndarray) -> BBox | None:
    r0, c0, r1, c1 = _bbox(coords)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    if side_r != side_c:
        return None
    if int(coords.shape[0]) != side_r * side_c:
        return None
    return r0, c0, r1, c1


def _find_template(grid: ColorGrid, bg: int) -> tuple[BBox, np.ndarray]:
    candidates: list[tuple[BBox, np.ndarray]] = []
    for coords in _label_components(grid != bg):
        colors = set(map(int, grid[coords[:, 0], coords[:, 1]]))
        if bg in colors or len(colors) < 2:
            continue
        box = _filled_square_bbox(coords)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        candidates.append((box, grid[r0 : r1 + 1, c0 : c1 + 1].copy()))
    assert len(candidates) == 1, "expected exactly one filled multicolor square template"
    return candidates[0]


def _wall_color(grid: ColorGrid, bg: int, template_box: BBox) -> int:
    mask = grid != bg
    r0, c0, r1, c1 = template_box
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    colors = set(map(int, grid[mask]))
    assert len(colors) == 1, "expected one wall color outside the template"
    return int(next(iter(colors)))


def _is_wall_enclosed_square_hole(
    grid: ColorGrid, coords: np.ndarray, wall_color: int
) -> BBox | None:
    box = _filled_square_bbox(coords)
    if box is None:
        return None
    comp_mask = np.zeros(grid.shape, dtype=bool)
    comp_mask[coords[:, 0], coords[:, 1]] = True
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if comp_mask[nr, nc]:
                continue
            if int(grid[nr, nc]) != wall_color:
                return None
    return box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the square template into every closed square hole at the matching integer scale.

    Input structure:
    - A black background contains one filled multicolor m x m template square and several monochrome wall objects.
    - Some wall objects fully enclose square black holes; other wall objects leave gaps so their black recesses stay connected to the outer background.
    - Every stampable hole has side length equal to an integer multiple of the template side length.

    Rule:
    - Find the unique filled multicolor square and use it as the template.
    - Find every black 4-connected component that does not touch the grid edge, is itself a filled square, and is surrounded by the wall color.
    - For each such hole, let k = hole_side / template_side and fill the hole with the template scaled by k in both axes; leave walls, the template, and open recesses unchanged.

    Key insight:
    - Geometry alone is insufficient: a square recess is stamped only if it is topologically closed off from the background, and its side length sets the exact scaling factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    template_box, template = _find_template(g, bg)
    m = int(template.shape[0])
    assert template.shape == (m, m), "template must be square"
    wall_color = _wall_color(g, bg, template_box)

    out = g.copy()
    for coords in _label_components(g == bg):
        if _touches_edge(coords, g.shape):
            continue
        hole_box = _is_wall_enclosed_square_hole(g, coords, wall_color)
        if hole_box is None:
            continue
        r0, c0, r1, c1 = hole_box
        side = r1 - r0 + 1
        assert side % m == 0, "hole side must be an integer multiple of the template size"
        scale = side // m
        out[r0 : r1 + 1, c0 : c1 + 1] = np.kron(
            template, np.ones((scale, scale), dtype=int)
        )
    return out


# --------------- generator ---------------

_BG = 0
_COLORS = tuple(range(1, 10))
_GRID_MIN = 18
_GRID_MAX = 30
_M_RANGE = (2, 4)
_CLOSED_COUNT_RANGE = (1, 3)
_OPEN_COUNT_RANGE = (1, 3)
_THICKNESS_RANGE = (1, 2)
_PLACEMENT_GAP = 2
_GEN_RETRIES = 300


def _make_template(m: int, wall_color: int) -> np.ndarray:
    palette = [c for c in _COLORS if c != wall_color]
    palette_size = random.randint(2, min(4, len(palette), m * m))
    chosen = random.sample(palette, palette_size)
    cells = chosen[:]
    while len(cells) < m * m:
        cells.append(random.choice(chosen))
    random.shuffle(cells)
    template = np.array(cells[: m * m], dtype=int).reshape(m, m)
    assert len(set(map(int, template.ravel()))) >= 2
    return template


def _make_closed_patch(hole_side: int, wall_color: int, thickness: int) -> tuple[np.ndarray, BBox]:
    patch = np.zeros((hole_side + 2 * thickness, hole_side + 2 * thickness), dtype=int)
    patch[:thickness, :] = wall_color
    patch[-thickness:, :] = wall_color
    patch[:, :thickness] = wall_color
    patch[:, -thickness:] = wall_color
    hole_box = (
        thickness,
        thickness,
        thickness + hole_side - 1,
        thickness + hole_side - 1,
    )
    return patch, hole_box


def _make_open_patch(inner_h: int, inner_w: int, wall_color: int, thickness: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2 * thickness, inner_w + 2 * thickness), dtype=int)
    patch[:thickness, :] = wall_color
    patch[-thickness:, :] = wall_color
    patch[:, :thickness] = wall_color
    patch[:, -thickness:] = wall_color

    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        gap = random.randint(max(1, inner_w // 3), max(1, inner_w))
        start = random.randint(thickness, inner_w + thickness - gap)
        row_slice = slice(0, thickness) if side == "top" else slice(patch.shape[0] - thickness, patch.shape[0])
        patch[row_slice, start : start + gap] = _BG
    else:
        gap = random.randint(max(1, inner_h // 3), max(1, inner_h))
        start = random.randint(thickness, inner_h + thickness - gap)
        col_slice = slice(0, thickness) if side == "left" else slice(patch.shape[1] - thickness, patch.shape[1])
        patch[start : start + gap, col_slice] = _BG

    return patch


def _separated(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_patches(
    grid_h: int, grid_w: int, patches: list[np.ndarray]
) -> list[Coord] | None:
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    placed: list[Coord | None] = [None] * len(patches)
    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        candidates = [
            (r0, c0)
            for r0 in range(1, grid_h - h)
            for c0 in range(1, grid_w - w)
            if _separated((r0, c0, r0 + h - 1, c0 + w - 1), rects, _PLACEMENT_GAP)
        ]
        if not candidates:
            return None
        pos = random.choice(candidates)
        placed[idx] = pos
        rects.append((pos[0], pos[1], pos[0] + h - 1, pos[1] + w - 1))
    return [pos for pos in placed if pos is not None]


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _attempt() -> np.ndarray | None:
    m = random.randint(*_M_RANGE)
    wall_color = random.choice(_COLORS)
    template = _make_template(m, wall_color)

    closed_count = random.randint(*_CLOSED_COUNT_RANGE)
    open_count = random.randint(*_OPEN_COUNT_RANGE)

    patches: list[np.ndarray] = [template.copy()]
    placements_meta: list[tuple[BBox, int]] = []

    max_scale = max(1, 12 // m)
    matching_open_scales = [random.randint(1, max_scale)]
    for _ in range(1, open_count):
        matching_open_scales.append(random.randint(1, max_scale))

    for _ in range(closed_count):
        scale = random.randint(1, max_scale)
        thickness = random.randint(*_THICKNESS_RANGE)
        hole_side = m * scale
        patch, hole_box = _make_closed_patch(hole_side, wall_color, thickness)
        patches.append(patch)
        placements_meta.append((hole_box, scale))

    for i in range(open_count):
        thickness = random.randint(*_THICKNESS_RANGE)
        if i == 0 or random.random() < 0.6:
            side = m * matching_open_scales[i]
            inner_h = side
            inner_w = side
        else:
            inner_h = random.randint(3, 10)
            inner_w = random.randint(3, 10)
            if inner_h == inner_w and inner_h % m == 0:
                inner_w += 1 if inner_w < 10 else -1
        patches.append(_make_open_patch(inner_h, inner_w, wall_color, thickness))

    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    grid_h = random.randint(max(_GRID_MIN, max_h + 2), _GRID_MAX)
    grid_w = random.randint(max(_GRID_MIN, max_w + 2), _GRID_MAX)
    positions = _place_patches(grid_h, grid_w, patches)
    if positions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        _paste(grid, patch, r0, c0)

    expected = grid.copy()
    for patch_idx, (hole_box, scale) in enumerate(placements_meta, start=1):
        pr, pc = positions[patch_idx]
        hr0, hc0, hr1, hc1 = hole_box
        expected[
            pr + hr0 : pr + hr1 + 1,
            pc + hc0 : pc + hc1 + 1,
        ] = np.kron(template, np.ones((scale, scale), dtype=int))

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
