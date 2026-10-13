import numpy as np
from collections import deque


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

BACKGROUND_COLOR = 0
DIVIDER_COLOR = 5
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate(
        (grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]),
        axis=0,
    )
    return _mode(border)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, exclude: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in exclude:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _divider_row(grid: np.ndarray, bg: int) -> int:
    candidates = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    valid = [r for r in candidates if np.any(grid[:r, :] != bg) and np.any(grid[r + 1 :, :] != bg)]
    assert len(valid) == 1, "expected exactly one horizontal divider row"
    return int(valid[0])


def _template_mapping(top: np.ndarray, bg: int) -> dict[tuple[tuple[int, int], ...], int]:
    comps = _monochrome_components(top, exclude={bg})
    singletons = [comp for comp in comps if comp["area"] == 1]
    shapes = [comp for comp in comps if comp["area"] > 1]
    label_lookup = {(int(comp["coords"][0, 0]), int(comp["coords"][0, 1])): comp for comp in singletons}

    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    used_labels: set[tuple[int, int]] = set()
    for comp in shapes:
        r0, c0, r1, c1 = comp["bbox"]
        cells = {tuple(map(int, rc)) for rc in comp["coords"]}
        candidates = []
        for c in range(c0, c1 + 1):
            if (r1, c) in cells and (r1 + 1, c) in label_lookup:
                candidates.append(label_lookup[(r1 + 1, c)])
        assert len(candidates) == 1, "each template must have exactly one label directly below it"
        label = candidates[0]
        label_pos = tuple(map(int, label["coords"][0]))
        assert label_pos not in used_labels, "labels must be used by exactly one template"
        used_labels.add(label_pos)
        sig = _signature(comp["coords"])
        assert sig not in mapping, "template shapes must be unique"
        mapping[sig] = int(label["color"])

    assert mapping and len(mapping) == len(shapes) == len(singletons), "top half must be pure template+label pairs"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each board hole with a template shape from the upper half and fill it
    using the template's label color instead of the template's drawing color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _border_background(g)
    divider_r = _divider_row(g, bg)

    top = g[:divider_r, :]
    bottom = g[divider_r + 1 :, :]
    template_to_label = _template_mapping(top, bg)

    board_candidates = _monochrome_components(bottom, exclude={bg})
    assert board_candidates, "bottom half must contain a board"
    board = max(board_candidates, key=lambda comp: comp["area"])
    board_color = int(board["color"])
    br0, bc0, br1, bc1 = board["bbox"]
    patch = bottom[br0 : br1 + 1, bc0 : bc1 + 1]
    assert set(map(int, np.unique(patch))).issubset({bg, board_color}), "board patch must contain only board color and holes"

    holes = _mask_components(patch == bg)
    assert holes, "board must contain at least one hole"

    out = g.copy()
    for hole in holes:
        assert not _touches_boundary(hole, patch.shape), "holes must stay strictly inside the board"
        sig = _signature(hole)
        assert sig in template_to_label, "every hole must match a template"
        fill_color = template_to_label[sig]
        for r, c in hole:
            out[divider_r + 1 + br0 + int(r), bc0 + int(c)] = fill_color
    return out


def _has_interior_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    q = deque()
    h, w = bg.shape
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(bg & ~seen))


def _random_shape(rng: np.random.Generator) -> np.ndarray | None:
    raw_h = int(rng.choice([2, 3, 4, 4, 5]))
    raw_w = int(rng.choice([2, 3, 4, 4, 5]))
    max_area = min(raw_h * raw_w, 8)
    if max_area < 3:
        return None
    target_area = int(rng.integers(3, max_area + 1))
    cells = {(int(rng.integers(raw_h)), int(rng.integers(raw_w)))}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pick = frontier[int(rng.integers(len(frontier)))]
        cells.add(pick)

    mask = np.zeros((raw_h, raw_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    mask = mask[rows][:, cols]
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return None
    if _has_interior_hole(mask):
        return None
    return mask


def _build_shapes(rng: np.random.Generator, count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(200):
        mask = _random_shape(rng)
        if mask is None:
            continue
        sig = _signature(np.argwhere(mask))
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
        if len(shapes) == count:
            return shapes
    return None


def _place_holes(shapes: list[np.ndarray], board_h: int, board_w: int, rng: np.random.Generator):
    blocked = np.zeros((board_h, board_w), dtype=bool)
    placements = [None] * len(shapes)
    order = list(map(int, rng.permutation(len(shapes))))
    coords_cache = [np.argwhere(shape) for shape in shapes]

    for idx in order:
        shape = shapes[idx]
        coords = coords_cache[idx]
        sh, sw = shape.shape
        candidates = []
        for r in range(1, board_h - sh):
            for c in range(1, board_w - sw):
                ok = True
                for dr, dc in coords:
                    if blocked[r + int(dr), c + int(dc)]:
                        ok = False
                        break
                if ok:
                    candidates.append((r, c))
        if not candidates:
            return None

        top, left = candidates[int(rng.integers(len(candidates)))]
        placements[idx] = (top, left)
        for dr, dc in coords:
            rr = top + int(dr)
            cc = left + int(dc)
            for nr, nc in ((rr + a, cc + b) for a, b in NEIGHBORS8):
                if 0 <= nr < board_h and 0 <= nc < board_w:
                    blocked[nr, nc] = True
    return placements


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    coords = np.argwhere(mask)
    for dr, dc in coords:
        grid[top + int(dr), left + int(dc)] = color


def _gen_attempt(rng: np.random.Generator):
    template_count = int(rng.integers(2, 5))
    shapes = _build_shapes(rng, template_count)
    if shapes is None:
        return None

    max_h = max(shape.shape[0] for shape in shapes)
    max_w = max(shape.shape[1] for shape in shapes)
    total_area = int(sum(int(shape.sum()) for shape in shapes))

    placements = None
    board_h = board_w = 0
    for _ in range(60):
        board_h_min = max(max_h + 3, 8, total_area // 4)
        board_h = int(rng.integers(board_h_min, board_h_min + 5))
        board_w_min = max(max_w + 3, 10, (total_area + board_h - 3) // max(2, board_h - 2) + 5)
        board_w = int(rng.integers(board_w_min, board_w_min + 7))
        placements = _place_holes(shapes, board_h, board_w, rng)
        if placements is not None:
            break
    if placements is None:
        return None

    palette = list(map(int, rng.permutation(NON_SPECIAL_COLORS)))
    shape_palette_size = 1 if template_count == 4 or rng.random() < 0.6 else 2
    board_color = palette[0]
    shape_palette = palette[1 : 1 + shape_palette_size]
    label_colors = palette[1 + shape_palette_size : 1 + shape_palette_size + template_count]
    if len(label_colors) != template_count:
        return None

    if shape_palette_size == 1:
        shape_colors = [shape_palette[0]] * template_count
    else:
        shape_colors = [shape_palette[i % 2] for i in range(template_count)]
        rng.shuffle(shape_colors)

    order = list(map(int, rng.permutation(template_count)))
    widths = [int(shapes[idx].shape[1]) for idx in order]
    top_h = max_h + 3 + int(rng.integers(0, 3))
    gaps = [int(rng.integers(2, 5)) for _ in range(max(0, template_count - 1))]
    side_pad = int(rng.integers(1, 3))
    top_needed = side_pad * 2 + sum(widths) + sum(gaps)

    width_min = max(top_needed, board_w + 2)
    width = width_min + int(rng.integers(0, 5))
    if width - board_w < 2:
        return None

    bottom_h = board_h + int(rng.integers(3, 6))
    if bottom_h - board_h < 2:
        return None

    grid = np.full((top_h + 1 + bottom_h, width), BACKGROUND_COLOR, dtype=int)
    grid[top_h, :] = DIVIDER_COLOR

    x = side_pad + int(rng.integers(0, width - top_needed + 1))
    for i, idx in enumerate(order):
        shape = shapes[idx]
        sh, sw = shape.shape
        max_y = top_h - sh - 1
        if max_y < 1:
            return None
        y = int(rng.integers(1, max_y + 1))
        _paint_mask(grid, y, x, shape, shape_colors[idx])
        bottom_cols = np.flatnonzero(shape[-1])
        label_col = int(bottom_cols[int(rng.integers(len(bottom_cols)))])
        grid[y + sh, x + label_col] = label_colors[idx]
        if i < len(gaps):
            x += sw + gaps[i]

    board_top = int(rng.integers(1, bottom_h - board_h))
    board_left = int(rng.integers(1, width - board_w))
    board_r0 = top_h + 1 + board_top
    board_c0 = board_left
    grid[board_r0 : board_r0 + board_h, board_c0 : board_c0 + board_w] = board_color

    expected = grid.copy()
    for idx, (hole_r, hole_c) in enumerate(placements):
        shape = shapes[idx]
        coords = np.argwhere(shape)
        for dr, dc in coords:
            rr = board_r0 + hole_r + int(dr)
            cc = board_c0 + hole_c + int(dc)
            grid[rr, cc] = BACKGROUND_COLOR
            expected[rr, cc] = label_colors[idx]

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
