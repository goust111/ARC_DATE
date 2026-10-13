import random

import numpy as np


OCCLUDER_COLOR = 5
SEARCH_PAD = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 22
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 5
PLACEMENT_PAD = 1
BAR_EXTENSION_MIN = 1
BAR_EXTENSION_MAX = 2
MAX_OBJECT_BUILD_TRIES = 40
MAX_GENERATION_ATTEMPTS = 400

COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]
SHAPE_TYPES = ("rect", "L", "T")

L_MASKS = [
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
]

T_MASKS = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_cells(mask: np.ndarray, top: int = 0, left: int = 0) -> frozenset[tuple[int, int]]:
    coords = np.argwhere(mask)
    return frozenset((top + int(r), left + int(c)) for r, c in coords)


def _missing_is_straight_bar(missing: set[tuple[int, int]]) -> bool:
    if not missing:
        return True
    rows = {r for r, _ in missing}
    cols = {c for _, c in missing}
    if len(rows) == 1:
        ordered = sorted(c for _, c in missing)
    elif len(cols) == 1:
        ordered = sorted(r for r, _ in missing)
    else:
        return False
    return ordered == list(range(ordered[0], ordered[-1] + 1))


def _candidate_masks() -> list[np.ndarray]:
    return L_MASKS + T_MASKS


def _infer_shape_from_sets(
    visible: set[tuple[int, int]],
    grey_mask: np.ndarray,
    grid_shape: tuple[int, int],
) -> frozenset[tuple[int, int]]:
    assert visible, "expected at least one visible object pixel"
    h, w = grid_shape
    min_r = min(r for r, _ in visible)
    max_r = max(r for r, _ in visible)
    min_c = min(c for _, c in visible)
    max_c = max(c for _, c in visible)

    win_r0 = max(0, min_r - SEARCH_PAD)
    win_r1 = min(h - 1, max_r + SEARCH_PAD)
    win_c0 = max(0, min_c - SEARCH_PAD)
    win_c1 = min(w - 1, max_c + SEARCH_PAD)

    candidates: dict[frozenset[tuple[int, int]], int] = {}

    def consider(cells: frozenset[tuple[int, int]]) -> None:
        if not visible.issubset(cells):
            return
        missing = set(cells) - visible
        if any(not bool(grey_mask[r, c]) for r, c in missing):
            return
        if not _missing_is_straight_bar(missing):
            return
        candidates[cells] = len(missing)

    for r0 in range(win_r0, min_r + 1):
        for c0 in range(win_c0, min_c + 1):
            for r1 in range(max_r, win_r1 + 1):
                height = r1 - r0 + 1
                if height < 2:
                    continue
                for c1 in range(max_c, win_c1 + 1):
                    width = c1 - c0 + 1
                    if width < 2:
                        continue
                    cells = frozenset(
                        (r, c)
                        for r in range(r0, r1 + 1)
                        for c in range(c0, c1 + 1)
                    )
                    consider(cells)

    for mask in _candidate_masks():
        mh, mw = mask.shape
        for r0 in range(win_r0, win_r1 - mh + 2):
            for c0 in range(win_c0, win_c1 - mw + 2):
                consider(_mask_cells(mask, r0, c0))

    assert candidates, f"no valid completion for visible pixels {sorted(visible)}"
    best_missing = min(candidates.values())
    best = [cells for cells, missing in candidates.items() if missing == best_missing]
    assert len(best) == 1, f"ambiguous completion for visible pixels {sorted(visible)}"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove grey occluder bars and complete each colored object to its full
    standard geometry: rectangle, 2x2 L-triomino, or T-tetromino.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    assert OCCLUDER_COLOR != bg, "background cannot be the occluder color"
    assert np.any(g == OCCLUDER_COLOR), "expected at least one grey occluder"

    grey_mask = g == OCCLUDER_COLOR
    out = np.where(grey_mask, bg, g).copy()

    colors = [int(color) for color in np.unique(g) if color not in {bg, OCCLUDER_COLOR}]
    assert colors, "expected at least one foreground object"
    for color in colors:
        visible = {
            (int(r), int(c))
            for r, c in np.argwhere(g == color)
        }
        completed = _infer_shape_from_sets(visible, grey_mask, g.shape)
        for r, c in completed:
            out[r, c] = color

    return out


def _sample_shape_mask(shape_type: str) -> np.ndarray:
    if shape_type == "rect":
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        return np.ones((height, width), dtype=bool)
    if shape_type == "L":
        return random.choice(L_MASKS).copy()
    assert shape_type == "T"
    return random.choice(T_MASKS).copy()


def _trim_pair(inp_patch: np.ndarray, out_patch: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (inp_patch != bg) | (out_patch != bg)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (
        inp_patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
        out_patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
    )


def _patch_is_valid(inp_patch: np.ndarray, out_patch: np.ndarray, color: int) -> bool:
    visible = {(int(r), int(c)) for r, c in np.argwhere(inp_patch == color)}
    expected = {(int(r), int(c)) for r, c in np.argwhere(out_patch == color)}
    grey_mask = inp_patch == OCCLUDER_COLOR
    try:
        completed = _infer_shape_from_sets(visible, grey_mask, inp_patch.shape)
    except AssertionError:
        return False
    return completed == expected


def _occlusion_options(mask: np.ndarray) -> list[tuple[str, int, tuple[tuple[int, int], ...]]]:
    full = set(_mask_cells(mask))
    total = len(full)
    options: list[tuple[str, int, tuple[tuple[int, int], ...]]] = []

    for row in range(mask.shape[0]):
        cells = tuple((row, int(c)) for c in np.flatnonzero(mask[row]))
        if not cells or len(cells) >= total or total - len(cells) < 2:
            continue
        grey = np.zeros(mask.shape, dtype=bool)
        for r, c in cells:
            grey[r, c] = True
        visible = full - set(cells)
        try:
            if _infer_shape_from_sets(visible, grey, mask.shape) == full:
                options.append(("h", row, cells))
        except AssertionError:
            pass

    for col in range(mask.shape[1]):
        cells = tuple((int(r), col) for r in np.flatnonzero(mask[:, col]))
        if not cells or len(cells) >= total or total - len(cells) < 2:
            continue
        grey = np.zeros(mask.shape, dtype=bool)
        for r, c in cells:
            grey[r, c] = True
        visible = full - set(cells)
        try:
            if _infer_shape_from_sets(visible, grey, mask.shape) == full:
                options.append(("v", col, cells))
        except AssertionError:
            pass

    return options


def _make_object_patches(shape_type: str, color: int, bg: int, occluded: bool) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(MAX_OBJECT_BUILD_TRIES):
        mask = _sample_shape_mask(shape_type)
        out_patch = np.full(mask.shape, bg, dtype=int)
        out_patch[mask] = color
        if not occluded:
            if _patch_is_valid(out_patch, out_patch, color):
                return out_patch.copy(), out_patch
            continue

        options = _occlusion_options(mask)
        if not options:
            continue

        orientation, index, cells = random.choice(options)
        ext_a = random.randint(BAR_EXTENSION_MIN, BAR_EXTENSION_MAX)
        ext_b = random.randint(BAR_EXTENSION_MIN, BAR_EXTENSION_MAX)

        canvas_h = mask.shape[0] + 4
        canvas_w = mask.shape[1] + 4
        out_canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        out_canvas[2 : 2 + mask.shape[0], 2 : 2 + mask.shape[1]][mask] = color
        inp_canvas = out_canvas.copy()

        if orientation == "h":
            row = 2 + index
            cols = sorted(c for _, c in cells)
            c0 = 2 + cols[0] - ext_a
            c1 = 2 + cols[-1] + ext_b
            inp_canvas[row, c0 : c1 + 1] = OCCLUDER_COLOR
        else:
            col = 2 + index
            rows = sorted(r for r, _ in cells)
            r0 = 2 + rows[0] - ext_a
            r1 = 2 + rows[-1] + ext_b
            inp_canvas[r0 : r1 + 1, col] = OCCLUDER_COLOR

        inp_patch, out_patch = _trim_pair(inp_canvas, out_canvas, bg)
        if _patch_is_valid(inp_patch, out_patch, color):
            return inp_patch, out_patch

    return None


def _reserve_bbox(
    occupied: np.ndarray,
    row: int,
    col: int,
    height: int,
    width: int,
) -> None:
    h, w = occupied.shape
    r0 = max(0, row - PLACEMENT_PAD)
    c0 = max(0, col - PLACEMENT_PAD)
    r1 = min(h, row + height + PLACEMENT_PAD)
    c1 = min(w, col + width + PLACEMENT_PAD)
    occupied[r0:r1, c0:c1] = True


def _can_place(occupied: np.ndarray, row: int, col: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, row - PLACEMENT_PAD)
    c0 = max(0, col - PLACEMENT_PAD)
    r1 = min(h, row + height + PLACEMENT_PAD)
    c1 = min(w, col + width + PLACEMENT_PAD)
    return not bool(occupied[r0:r1, c0:c1].any())


def _stamp(grid: np.ndarray, patch: np.ndarray, row: int, col: int, bg: int) -> None:
    mask = patch != bg
    grid[row : row + patch.shape[0], col : col + patch.shape[1]][mask] = patch[mask]


def _place_all(
    inp: np.ndarray,
    out: np.ndarray,
    patches: list[tuple[np.ndarray, np.ndarray, int]],
    bg: int,
) -> bool:
    occupied = np.zeros(inp.shape, dtype=bool)
    order = sorted(
        range(len(patches)),
        key=lambda i: -np.count_nonzero((patches[i][0] != bg) | (patches[i][1] != bg)),
    )

    for idx in order:
        inp_patch, out_patch, _color = patches[idx]
        ph, pw = inp_patch.shape
        positions = [(r, c) for r in range(inp.shape[0] - ph + 1) for c in range(inp.shape[1] - pw + 1)]
        random.shuffle(positions)
        placed = False
        for row, col in positions:
            if not _can_place(occupied, row, col, ph, pw):
                continue
            _stamp(inp, inp_patch, row, col, bg)
            _stamp(out, out_patch, row, col, bg)
            _reserve_bbox(occupied, row, col, ph, pw)
            placed = True
            break
        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_POOL)
    colors = [color for color in COLOR_POOL if color != bg]
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    object_colors = random.sample(colors, n_objects)

    shape_types = [random.choice(SHAPE_TYPES) for _ in range(n_objects)]
    occluded = [random.random() < 0.5 for _ in range(n_objects)]
    if not any(occluded):
        occluded[random.randrange(n_objects)] = True

    patches: list[tuple[np.ndarray, np.ndarray, int]] = []
    for shape_type, color, is_occluded in zip(shape_types, object_colors, occluded):
        built = _make_object_patches(shape_type, color, bg, is_occluded)
        if built is None:
            return None
        inp_patch, out_patch = built
        patches.append((inp_patch, out_patch, color))

    height = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    if not _place_all(inp, out, patches, bg):
        return None

    if not np.any(inp == OCCLUDER_COLOR):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
