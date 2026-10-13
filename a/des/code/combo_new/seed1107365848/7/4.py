from __future__ import annotations

import numpy as np

BG = 0
SEED = 3
DICT_BORDER = 7
SEPARATOR = 9
TEMPLATE_PALETTE = np.array([1, 2, 4, 5, 6, 8], dtype=int)

GEN_ATTEMPTS = 200
ROOM_LAYOUTS = [(2, 2), (2, 3), (3, 2), (3, 3)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _dictionary_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == DICT_BORDER)
    assert len(coords) > 0, "dictionary border must exist"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 3 and c1 - c0 >= 3, "dictionary must have an interior"
    assert np.all(grid[r0, c0 : c1 + 1] == DICT_BORDER)
    assert np.all(grid[r1, c0 : c1 + 1] == DICT_BORDER)
    assert np.all(grid[r0 : r1 + 1, c0] == DICT_BORDER)
    assert np.all(grid[r0 : r1 + 1, c1] == DICT_BORDER)
    return r0, c0, r1, c1


def _templates_from_dictionary(grid: np.ndarray, dict_bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = dict_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components(inner != BG)
    templates: list[tuple[int, int, np.ndarray]] = []
    for comp in comps:
        tr0, tc0, tr1, tc1 = _bbox(comp)
        patch = inner[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
        templates.append((tc0, tr0, patch))
    templates.sort(key=lambda item: (item[0], item[1]))
    assert templates, "dictionary must contain at least one template"
    return [patch for _, _, patch in templates]


def _is_room_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
        return False
    return (
        np.all(grid[r0 - 1, c0 : c1 + 1] == SEPARATOR)
        and np.all(grid[r1 + 1, c0 : c1 + 1] == SEPARATOR)
        and np.all(grid[r0 : r1 + 1, c0 - 1] == SEPARATOR)
        and np.all(grid[r0 : r1 + 1, c1 + 1] == SEPARATOR)
    )


def _room_bboxes(grid: np.ndarray, dict_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    mask = grid != SEPARATOR
    r0, c0, r1, c1 = dict_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        bbox = _bbox(comp)
        if _is_room_bbox(grid, bbox):
            rooms.append(bbox)
    rooms.sort()
    assert rooms, "at least one room is required"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each maroon-walled room with the template indexed by its green seed count.

    Input structure:
    - The background is black.
    - Maroon horizontal and vertical separator lines form rectangular rooms.
    - A lower-right dictionary is enclosed by an orange border and contains several 2x2 or 3x3 color templates separated by black gaps.
    - Each room contains a small number of green seed pixels.

    Rule:
    - Count the green seeds inside each room; if a room has N seeds, choose the N-th template from the dictionary, reading templates from left to right.
    - Tile that template across the full room interior starting from the room's top-left inner cell, cropping the repeated pattern at the room boundary.
    - Preserve the original green seed pixels instead of overwriting them with the tiled template.
    - Clear the entire dictionary area, including its orange frame, back to black.

    Key insight:
    - The green pixels act both as a visible lookup index into the lower-right template dictionary and as fixed pins that remain on top of the final tiled fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    dict_bbox = _dictionary_bbox(g)
    templates = _templates_from_dictionary(g, dict_bbox)
    rooms = _room_bboxes(g, dict_bbox)

    out = g.copy()
    dr0, dc0, dr1, dc1 = dict_bbox
    out[dr0 : dr1 + 1, dc0 : dc1 + 1] = BG

    for r0, c0, r1, c1 in rooms:
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_count = int(np.count_nonzero(room == SEED))
        assert 1 <= seed_count <= len(templates), "room seed count must point to an existing template"
        template = templates[seed_count - 1]
        th, tw = template.shape
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if g[rr, cc] == SEED:
                    out[rr, cc] = SEED
                else:
                    out[rr, cc] = int(template[(rr - r0) % th, (cc - c0) % tw])

    return out


def _random_template(rng: np.random.Generator, size: int, used: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        color_count = int(rng.integers(2, 5))
        colors = rng.choice(TEMPLATE_PALETTE, size=color_count, replace=False)
        template = rng.choice(colors, size=(size, size), replace=True).astype(int)
        if len(np.unique(template)) < 2:
            continue
        if not np.any(template[:, 1:] != template[:, :-1]):
            continue
        if not np.any(template[1:, :] != template[:-1, :]):
            continue
        sig = (size,) + tuple(int(x) for x in template.ravel())
        if sig in used:
            continue
        used.add(sig)
        return template
    raise RuntimeError("failed to sample a distinct template")


def _choose_room_layout(rng: np.random.Generator, template_count: int) -> tuple[int, int]:
    candidates = [shape for shape in ROOM_LAYOUTS if shape[0] * shape[1] >= template_count]
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def _row_col_edges(sizes: np.ndarray) -> list[int]:
    edges = [0]
    cursor = 0
    for size in sizes:
        cursor += int(size) + 1
        edges.append(cursor)
    return edges


def _draw_main_rooms(
    grid: np.ndarray,
    row_heights: np.ndarray,
    col_widths: np.ndarray,
    seed_counts: list[int],
    rng: np.random.Generator,
) -> None:
    row_edges = _row_col_edges(row_heights)
    col_edges = _row_col_edges(col_widths)
    main_h = row_edges[-1] + 1
    main_w = col_edges[-1] + 1

    for edge in row_edges:
        grid[edge, :main_w] = SEPARATOR
    for edge in col_edges:
        grid[:main_h, edge] = SEPARATOR

    room_index = 0
    for ri in range(len(row_heights)):
        for ci in range(len(col_widths)):
            r0 = row_edges[ri] + 1
            r1 = row_edges[ri + 1] - 1
            c0 = col_edges[ci] + 1
            c1 = col_edges[ci + 1] - 1
            coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            count = seed_counts[room_index]
            picks = rng.choice(len(coords), size=count, replace=False)
            for idx in np.atleast_1d(picks):
                rr, cc = coords[int(idx)]
                grid[rr, cc] = SEED
            room_index += 1


def _draw_dictionary(grid: np.ndarray, templates: list[np.ndarray], top: int, left: int) -> None:
    inner_h = max(template.shape[0] for template in templates)
    inner_w = sum(template.shape[1] for template in templates) + (len(templates) - 1)
    dict_h = inner_h + 2
    dict_w = inner_w + 2

    grid[top, left : left + dict_w] = DICT_BORDER
    grid[top + dict_h - 1, left : left + dict_w] = DICT_BORDER
    grid[top : top + dict_h, left] = DICT_BORDER
    grid[top : top + dict_h, left + dict_w - 1] = DICT_BORDER

    cursor = left + 1
    for template in templates:
        th, tw = template.shape
        r0 = top + 1 + (inner_h - th) // 2
        grid[r0 : r0 + th, cursor : cursor + tw] = template
        cursor += tw + 1


def _build_input(rng: np.random.Generator) -> np.ndarray:
    template_count = int(rng.integers(2, 5))
    room_rows, room_cols = _choose_room_layout(rng, template_count)
    row_heights = rng.integers(3, 6, size=room_rows)
    col_widths = rng.integers(3, 6, size=room_cols)

    main_h = int(row_heights.sum()) + room_rows + 1
    main_w = int(col_widths.sum()) + room_cols + 1

    sizes = rng.integers(2, 4, size=template_count)
    if template_count >= 2 and rng.random() < 0.6 and np.all(sizes == sizes[0]):
        flip = int(rng.integers(0, template_count))
        sizes[flip] = 5 - int(sizes[flip])

    used: set[tuple[int, ...]] = set()
    templates = [_random_template(rng, int(size), used) for size in sizes]

    inner_h = max(template.shape[0] for template in templates)
    inner_w = sum(template.shape[1] for template in templates) + (template_count - 1)
    dict_h = inner_h + 2
    dict_w = inner_w + 2

    canvas_h = main_h + 1 + dict_h
    canvas_w = max(main_w, dict_w)
    assert canvas_h <= 30 and canvas_w <= 30

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)

    mandatory = list(range(1, template_count + 1))
    extras = rng.integers(1, template_count + 1, size=room_rows * room_cols - template_count).tolist()
    seed_counts = mandatory + [int(x) for x in extras]
    seed_counts = [int(x) for x in rng.permutation(np.asarray(seed_counts, dtype=int))]

    _draw_main_rooms(grid, row_heights, col_widths, seed_counts, rng)

    dict_top = canvas_h - dict_h
    dict_left = canvas_w - dict_w
    _draw_dictionary(grid, templates, dict_top, dict_left)

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input(rng)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if (
            isinstance(grid, np.ndarray)
            and isinstance(out, np.ndarray)
            and grid.ndim == 2
            and out.ndim == 2
            and max(grid.shape) <= 30
            and max(out.shape) <= 30
        ):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
