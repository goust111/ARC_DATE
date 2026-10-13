from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

BORDER_COLORS = (BLUE, RED, GREEN)
TRANSFORMS = ("hflip", "vflip", "rotcw")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

DICT_BOX_SIZES = (3, 4)
ROOM_LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
ROOM_OUTER_MIN = 6
ROOM_OUTER_MAX = 10
MAX_SIDE = 30
GEN_RETRIES = 400


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "hflip":
        return np.fliplr(mask)
    if name == "vflip":
        return np.flipud(mask)
    if name == "rotcw":
        return np.rot90(mask, -1)
    raise ValueError(f"unknown transform: {name}")


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(mask.shape[0]),
        int(mask.shape[1]),
        tuple(int(v) for v in mask.astype(int).ravel()),
    )


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(arr)
    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a nonempty interior"

    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    actual = {tuple(map(int, pt)) for pt in coords}
    assert actual == expected, "border component must be a one-pixel rectangular frame"
    return r0, c0, r1, c1


def _parse_dictionary(dict_band: np.ndarray) -> dict[int, str]:
    h, w = dict_band.shape
    assert h >= 4, "dictionary band must include a label row and example rows"

    sep_cols = [int(c) for c in range(w) if np.all(dict_band[:, c] == GRAY)]
    assert len(sep_cols) == 2, "expected exactly two gray separator columns"

    spans: list[tuple[int, int]] = []
    prev = -1
    for c in sep_cols + [w]:
        if c - prev > 1:
            spans.append((prev + 1, c - 1))
        prev = c
    assert len(spans) == 3, "expected exactly three dictionary groups"

    mapping: dict[int, str] = {}
    for c0, c1 in spans:
        group = dict_band[:, c0 : c1 + 1]
        label_colors = [int(v) for v in np.unique(group) if int(v) not in {BLACK, YELLOW}]
        assert len(label_colors) == 1, "each dictionary group must contain exactly one label color"
        label = label_colors[0]
        assert int(np.count_nonzero(group == label)) == 1, "dictionary label must be a single pixel"

        example = group[1:, :]
        assert example.shape[1] % 2 == 1, "dictionary group width must split into equal halves"
        box = example.shape[1] // 2
        assert np.all(example[:, box] == BLACK), "dictionary halves must be separated by a black gap"

        left = _crop_true(example[:, :box] == YELLOW)
        right = _crop_true(example[:, box + 1 :] == YELLOW)

        matches = [name for name in TRANSFORMS if np.array_equal(_apply_transform(left, name), right)]
        assert len(matches) == 1, "dictionary example must identify exactly one transform"
        mapping[label] = matches[0]

    assert set(mapping.keys()) == set(BORDER_COLORS), "dictionary must cover all three border colors"
    assert len(set(mapping.values())) == 3, "dictionary transforms must be one-to-one"
    return mapping


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= MAX_SIDE

    gray_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == GRAY)]
    assert len(gray_rows) == 1, "expected a single full-width gray separator row"
    sep_r = gray_rows[0]
    assert 1 < sep_r < g.shape[0] - 2, "dictionary and room area must both be present"

    mapping = _parse_dictionary(g[:sep_r, :])
    lower = g[sep_r + 1 :, :]
    out = g.copy()

    room_count = 0
    for color in BORDER_COLORS:
        for coords in _color_components(lower, color):
            room_count += 1
            r0, c0, r1, c1 = _frame_bbox(coords)
            inner = lower[r0 + 1 : r1, c0 + 1 : c1]
            pattern = inner == YELLOW
            assert pattern.any(), "each room must contain a yellow pattern"
            sprite = _crop_true(pattern)
            transformed = _apply_transform(sprite, mapping[color])

            ih, iw = inner.shape
            th, tw = transformed.shape
            assert th <= ih and tw <= iw, "transformed pattern must fit back inside its room"

            replacement = np.full((ih, iw), BLACK, dtype=int)
            top = (ih - th) // 2
            left = (iw - tw) // 2
            patch = replacement[top : top + th, left : left + tw]
            patch[transformed] = YELLOW

            abs_r0 = sep_r + 1 + r0
            out[abs_r0 + 1 : abs_r0 + 1 + ih, c0 + 1 : c1] = replacement

    assert room_count >= 1, "expected at least one framed room"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top color-to-transform dictionary to flip or rotate each framed yellow pattern and recenter it inside its room.

    Input structure:
    - The background is black.
    - A top dictionary band is split into three groups by full-height gray columns, and a full gray row separates that band from the room area below.
    - Each dictionary group contains one colored label pixel (blue, red, or green), a yellow source pattern on the left, and the corresponding transformed yellow pattern on the right.
    - Below the separator row, several one-pixel-thick colored rectangular frames enclose black rooms that each contain a small yellow asymmetric pattern.

    Rule:
    - Read the three dictionary examples and determine which transform each label color represents among horizontal flip, vertical flip, and clockwise 90-degree rotation.
    - For every framed room, look up its border color in that dictionary.
    - Crop the room's yellow pattern to its tight bounding box, apply the matched transform, then place the transformed pattern back into the same room centered within the room interior.
    - Keep the dictionary band, separator row, black background, and colored room borders unchanged.

    Key insight:
    - The top strip is an active legend of transformations rather than decoration: the label color names a geometric operation that must be generalized to new shapes in the rooms below.
    """
    return _solve(grid)


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _c in coords) + 1
    w = max(c for _r, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _is_discriminative(mask: np.ndarray) -> bool:
    variants = {_mask_signature(mask)}
    for name in TRANSFORMS:
        variants.add(_mask_signature(_apply_transform(mask, name)))
    return len(variants) == 4


def _random_polyomino(box_limit: int, min_cells: int = 3, max_cells: int = 5) -> np.ndarray:
    for _ in range(300):
        target = random.randint(min_cells, max_cells)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in NEIGHBORS4
                if (r + dr, c + dc) not in cells
            }
            cells.add(random.choice(tuple(frontier)))

        if random.random() < 0.5:
            cells = {(r, -c) for r, c in cells}
        for _ in range(random.randrange(4)):
            cells = {(-c, r) for r, c in cells}

        coords = _normalize_cells(cells)
        mask = _coords_to_mask(coords)
        if mask.shape[0] > box_limit or mask.shape[1] > box_limit:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not _is_discriminative(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a discriminative shape")


def _bounded_composition(total: int, parts: int, lower: int, upper: int) -> list[int] | None:
    if parts * lower > total or parts * upper < total:
        return None

    out: list[int] = []
    remaining = total
    for idx in range(parts):
        slots_left = parts - idx - 1
        lo = max(lower, remaining - upper * slots_left)
        hi = min(upper, remaining - lower * slots_left)
        if lo > hi:
            return None
        piece = random.randint(lo, hi)
        out.append(piece)
        remaining -= piece

    random.shuffle(out)
    return out


def _sample_axis_layout(total: int, count: int, size_min: int, size_max: int) -> list[tuple[int, int]]:
    gap_count = count + 1
    max_room_sum = min(count * size_max, total - gap_count)
    min_room_sum = count * size_min
    assert min_room_sum <= max_room_sum, "axis too small for requested room layout"

    room_sum = random.randint(min_room_sum, max_room_sum)
    sizes = _bounded_composition(room_sum, count, size_min, size_max)
    assert sizes is not None

    slack = total - gap_count - room_sum
    extras = _bounded_composition(slack, gap_count, 0, slack) if slack else [0] * gap_count
    assert extras is not None
    gaps = [1 + extra for extra in extras]

    spans: list[tuple[int, int]] = []
    cursor = gaps[0]
    for idx, size in enumerate(sizes):
        spans.append((cursor, cursor + size - 1))
        cursor += size + gaps[idx + 1]
    assert cursor == total
    return spans


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = canvas[top : top + h, left : left + w]
    patch[mask] = color


def _centered_mask(inner_h: int, inner_w: int, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((inner_h, inner_w), dtype=bool)
    h, w = mask.shape
    top = (inner_h - h) // 2
    left = (inner_w - w) // 2
    _paint_mask(out, mask, top, left, True)
    return out


def _sample_room_pattern(inner_h: int, inner_w: int, transform_name: str) -> np.ndarray:
    for _ in range(300):
        shape = _random_polyomino(box_limit=3, min_cells=3, max_cells=5)
        transformed = _apply_transform(shape, transform_name)
        sh, sw = shape.shape
        th, tw = transformed.shape
        if sh > inner_h or sw > inner_w or th > inner_h or tw > inner_w:
            continue

        top = random.randint(0, inner_h - sh)
        left = random.randint(0, inner_w - sw)
        inp = np.zeros((inner_h, inner_w), dtype=bool)
        _paint_mask(inp, shape, top, left, True)

        expected = _centered_mask(inner_h, inner_w, transformed)
        if np.array_equal(inp, expected):
            continue
        return inp
    raise RuntimeError("failed to place a valid room pattern")


def _build_dictionary(mapping: dict[int, str], box_size: int) -> np.ndarray:
    group_w = 2 * box_size + 1
    body_h = box_size + 1
    width = 3 * group_w + 2
    band = np.full((body_h, width), BLACK, dtype=int)

    for sep_c in (group_w, 2 * group_w + 1):
        band[:, sep_c] = GRAY

    group_order = random.sample(list(BORDER_COLORS), len(BORDER_COLORS))
    for idx, color in enumerate(group_order):
        c0 = idx * group_w + idx
        band[0, c0] = color

        source = _random_polyomino(box_limit=box_size, min_cells=3, max_cells=5)
        result = _apply_transform(source, mapping[color])

        left_box = band[1:, c0 : c0 + box_size]
        right_box = band[1:, c0 + box_size + 1 : c0 + 2 * box_size + 1]
        _paint_mask(left_box, source, (box_size - source.shape[0]) // 2, (box_size - source.shape[1]) // 2, YELLOW)
        _paint_mask(right_box, result, (box_size - result.shape[0]) // 2, (box_size - result.shape[1]) // 2, YELLOW)

    return band


def _gen_attempt() -> np.ndarray | None:
    box_size = random.choice(DICT_BOX_SIZES)
    dict_body = _build_dictionary(dict(zip(BORDER_COLORS, random.sample(list(TRANSFORMS), len(TRANSFORMS)))), box_size)
    width = int(dict_body.shape[1])
    dict_h = int(dict_body.shape[0] + 1)
    max_lower_h = MAX_SIDE - dict_h

    feasible_layouts = []
    for rows, cols in ROOM_LAYOUTS:
        min_lower_h = rows * ROOM_OUTER_MIN + rows + 1
        min_width = cols * ROOM_OUTER_MIN + cols + 1
        if min_lower_h <= max_lower_h and min_width <= width:
            feasible_layouts.append((rows, cols))
    if not feasible_layouts:
        return None

    rows, cols = random.choice(feasible_layouts)
    min_lower_h = rows * ROOM_OUTER_MIN + rows + 1
    lower_h = random.randint(min_lower_h, max_lower_h)

    try:
        row_spans = _sample_axis_layout(lower_h, rows, ROOM_OUTER_MIN, ROOM_OUTER_MAX)
        col_spans = _sample_axis_layout(width, cols, ROOM_OUTER_MIN, ROOM_OUTER_MAX)
    except AssertionError:
        return None

    mapping = dict(zip(BORDER_COLORS, random.sample(list(TRANSFORMS), len(TRANSFORMS))))
    dict_body = _build_dictionary(mapping, box_size)

    grid = np.full((dict_h + lower_h, width), BLACK, dtype=int)
    grid[: dict_body.shape[0], :] = dict_body
    grid[dict_body.shape[0], :] = GRAY

    room_total = rows * cols
    room_colors = list(BORDER_COLORS)
    while len(room_colors) < room_total:
        room_colors.append(random.choice(BORDER_COLORS))
    random.shuffle(room_colors)

    room_index = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            color = room_colors[room_index]
            room_index += 1

            abs_r0 = dict_h + r0
            abs_r1 = dict_h + r1
            grid[abs_r0, c0 : c1 + 1] = color
            grid[abs_r1, c0 : c1 + 1] = color
            grid[abs_r0 : abs_r1 + 1, c0] = color
            grid[abs_r0 : abs_r1 + 1, c1] = color

            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            try:
                pattern = _sample_room_pattern(inner_h, inner_w, mapping[color])
            except RuntimeError:
                return None
            inner = grid[abs_r0 + 1 : abs_r1, c0 + 1 : c1]
            inner[pattern] = YELLOW

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if any(not np.any(grid[dict_h:, :] == color) for color in BORDER_COLORS):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
