from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
KEY = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)

PATTERN_MASKS: tuple[np.ndarray, ...] = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
)

GRID_MIN = 22
GRID_MAX = 30
ACTIVE_ROW_CHOICES = (3, 4)
ROOM_COUNT_RANGE = (2, 4)
INNER_H_RANGE = (5, 8)
INNER_W_RANGE = (5, 8)
ROOM_PLACEMENT_RETRIES = 250
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _corner_origin(grid_h: int, grid_w: int, patch_h: int, patch_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - patch_w
    if corner == "bl":
        return grid_h - patch_h, 0
    if corner == "br":
        return grid_h - patch_h, grid_w - patch_w
    raise ValueError(corner)


def _extract_panel_mapping(g: np.ndarray) -> dict[tuple[tuple[int, int], ...], np.ndarray]:
    h, w = g.shape
    grey_comps = _components(g == GREY)
    panel = None
    for comp in grey_comps:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            panel = comp
            break
    assert panel is not None, "expected a grey dictionary panel touching the boundary"

    r0, c0, r1, c1 = _bbox(panel)
    ph = r1 - r0 + 1
    pw = c1 - c0 + 1
    assert pw == 9 and ph >= 13 and (ph - 1) % 4 == 0

    rows = (ph - 1) // 4
    mapping: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for i in range(rows):
        rr = r0 + 1 + 4 * i
        shape_patch = g[rr : rr + 3, c0 + 1 : c0 + 4]
        texture_patch = g[rr : rr + 3, c0 + 5 : c0 + 8]
        shape_cells = [(int(r), int(c)) for r, c in zip(*np.where(shape_patch == KEY), strict=False)]
        assert shape_cells, "each dictionary row must contain a maroon key shape"
        mapping[_normalize(shape_cells)] = texture_patch.copy()

    assert mapping, "dictionary panel must define at least one mapping"
    return mapping


def _seed_in_room(g: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (interior != BG) & (interior != GREY)
    comps = _components(seed_mask)
    assert len(comps) == 1, "each room must contain exactly one seed block"
    comp = comps[0]
    colors = {int(interior[r, c]) for r, c in comp}
    assert len(colors) == 1, "seed block must be monochrome"
    return next(iter(colors)), _normalize(comp)


def _fill_room(out: np.ndarray, bbox: tuple[int, int, int, int], texture: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            out[r, c] = int(texture[(r - (r0 + 1)) % 3, (c - (c0 + 1)) % 3])


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each closed grey rectangle with the dictionary texture keyed by its interior seed shape.

    Input structure:
    - The background is black.
    - Several grey rectangular frames enclose empty interior regions; each region contains one small monochrome seed block made of 2-4 adjacent cells.
    - A grey-framed dictionary panel sits in one grid corner, split into rows; each row shows a maroon seed shape on the left and a 3x3 two-color texture sample on the right.

    Rule:
    - Read the dictionary as a mapping from exact seed geometry to a 3x3 texture sample.
    - For each closed rectangular frame, extract the unique interior seed block, match its shape to the dictionary row, and replace maroon cells in that row's texture with the seed color.
    - Tile the resulting 3x3 texture over the full interior of the rectangle with top-left alignment, covering the original seed, while keeping all grey borders and the dictionary panel unchanged.

    Key insight:
    - The seed's geometry selects the texture template, while the seed's color only recolors the template's maroon foreground before tiling.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    mapping = _extract_panel_mapping(g)
    grey_comps = _components(g == GREY)
    for comp in grey_comps:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2
        assert np.all(g[r0, c0 : c1 + 1] == GREY)
        assert np.all(g[r1, c0 : c1 + 1] == GREY)
        assert np.all(g[r0 : r1 + 1, c0] == GREY)
        assert np.all(g[r0 : r1 + 1, c1] == GREY)

        seed_color, shape = _seed_in_room(g, bbox)
        assert shape in mapping, "room seed shape must exist in dictionary"
        texture = mapping[shape].copy()
        texture[texture == KEY] = seed_color
        _fill_room(out, bbox, texture)

    return out


def _draw_panel(
    grid: np.ndarray,
    rows: list[tuple[tuple[tuple[int, int], ...], np.ndarray]],
    corner: str,
) -> tuple[int, int, int, int]:
    ph = 4 * len(rows) + 1
    pw = 9
    r0, c0 = _corner_origin(grid.shape[0], grid.shape[1], ph, pw, corner)
    panel = np.full((ph, pw), BG, dtype=int)

    panel[0, :] = GREY
    panel[-1, :] = GREY
    panel[:, 0] = GREY
    panel[:, 4] = GREY
    panel[:, -1] = GREY
    for i in range(1, len(rows)):
        panel[4 * i, :] = GREY

    for i, (shape, texture) in enumerate(rows):
        rr = 1 + 4 * i
        for sr, sc in shape:
            panel[rr + sr, 1 + sc] = KEY
        panel[rr : rr + 3, 5:8] = texture

    grid[r0 : r0 + ph, c0 : c0 + pw] = panel
    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _expanded_bbox(bbox: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _reserve(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = _expanded_bbox(bbox, occupied.shape[0], occupied.shape[1], pad=pad)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _bbox_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = _expanded_bbox(bbox, occupied.shape[0], occupied.shape[1], pad=pad)
    return not bool(np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]))


def _render_room(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    shape: tuple[tuple[int, int], ...],
    seed_color: int,
    seed_offset: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    inner_r, inner_c = seed_offset
    for sr, sc in shape:
        grid[r0 + 1 + inner_r + sr, c0 + 1 + inner_c + sc] = seed_color


def _expected_output(
    inp: np.ndarray,
    room_specs: list[tuple[tuple[int, int, int, int], np.ndarray]],
) -> np.ndarray:
    out = inp.copy()
    for bbox, texture in room_specs:
        _fill_room(out, bbox, texture)
    return out


def _attempt() -> tuple[np.ndarray, np.ndarray] | None:
    row_count = random.choice(ACTIVE_ROW_CHOICES)
    active_shapes = random.sample(list(SHAPES), row_count)
    active_masks = random.sample(list(PATTERN_MASKS), row_count)
    accent_colors = random.sample(list(COLOR_POOL), row_count)
    row_specs = []
    shape_to_texture: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for shape, mask, accent in zip(active_shapes, active_masks, accent_colors, strict=False):
        texture = np.full((3, 3), accent, dtype=int)
        texture[mask] = KEY
        row_specs.append((shape, texture))
        shape_to_texture[shape] = texture
    random.shuffle(row_specs)

    panel_h = 4 * row_count + 1
    h = random.randint(max(GRID_MIN, panel_h + 2), GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    corner = random.choice(("tl", "tr", "bl", "br"))
    panel_bbox = _draw_panel(grid, row_specs, corner)

    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, panel_bbox, pad=1)

    room_count = random.randint(*ROOM_COUNT_RANGE)
    base_shapes = random.sample(active_shapes, min(room_count, len(active_shapes)))
    while len(base_shapes) < room_count:
        base_shapes.append(random.choice(active_shapes))
    random.shuffle(base_shapes)

    seed_choices = [c for c in COLOR_POOL if c not in accent_colors]
    room_outputs: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for shape in base_shapes:
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        sh, sw = _shape_dims(shape)
        if inner_h < sh + 2 or inner_w < sw + 2:
            return None
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        placed = False
        for _ in range(ROOM_PLACEMENT_RETRIES):
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _bbox_free(occupied, bbox, pad=1):
                continue

            seed_r = random.randint(1, inner_h - sh - 1)
            seed_c = random.randint(1, inner_w - sw - 1)
            seed_color = random.choice(seed_choices)
            _render_room(grid, bbox, shape, seed_color, (seed_r, seed_c))

            texture = shape_to_texture[shape].copy()
            texture[texture == KEY] = seed_color
            room_outputs.append((bbox, texture))
            _reserve(occupied, bbox, pad=1)
            placed = True
            break
        if not placed:
            return None

    out = _expected_output(grid, room_outputs)
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _attempt()
        if built is None:
            continue
        inp, out = built
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
