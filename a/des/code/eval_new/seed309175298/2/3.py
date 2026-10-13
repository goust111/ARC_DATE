import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 9)
TRANSFORM_NAMES = (
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)
GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    dst[top : top + h, left : left + w] = src


def _variants(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rot90": np.rot90(arr, 1),
        "rot180": np.rot90(arr, 2),
        "rot270": np.rot90(arr, 3),
        "flip_lr": np.fliplr(arr),
        "flip_ud": np.flipud(arr),
        "transpose": arr.T.copy(),
        "anti_transpose": np.fliplr(np.rot90(arr, 1)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the lower azure mask by matching its visible hole pixels to the unique rigid transform of the upper reference pattern.

    Input structure:
    - The grid has a black background and one full-width grey separator row that splits it into upper and lower panels.
    - The upper panel contains one complete non-black reference pattern in a rectangular bbox.
    - The lower panel contains one same-sized azure mask rectangle; some mask cells are holes that reveal hidden pattern pixels.

    Rule:
    - Extract the upper reference pattern and the lower masked bbox.
    - Enumerate the seven non-identity rigid symmetries of the reference pattern: 90/180/270 rotations and the four reflections.
    - Keep the unique transformed candidate whose values match every visible lower-panel hole pixel.
    - Return only the lower panel, with the full transformed pattern restored at the mask position on a black background.

    Key insight:
    - The lower image is not arbitrary: it is exactly one rigid transform of the upper pattern, and the sparse hole pixels are enough to eliminate the other seven-way symmetry candidates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_rows = [
        r for r in range(g.shape[0]) if int(g[r, 0]) == GREY and np.all(g[r] == GREY)
    ]
    assert len(divider_rows) == 1, "expected one full-width grey separator row"
    sep_r = divider_rows[0]

    top = g[:sep_r]
    bottom = g[sep_r + 1 :]
    ref_box = _bbox(top != BLACK)
    assert ref_box is not None, "upper panel must contain one reference pattern"
    reference = _crop(top, ref_box)

    non_black_bottom = bottom[bottom != BLACK]
    assert non_black_bottom.size > 0, "lower panel must contain a masked object"
    values, counts = np.unique(non_black_bottom, return_counts=True)
    mask_color = int(values[np.argmax(counts)])
    assert mask_color == AZURE, "lower mask must be azure"

    mask_box = _bbox(bottom != BLACK)
    assert mask_box is not None, "lower mask bbox must exist"
    masked = _crop(bottom, mask_box)
    visible = masked != AZURE
    assert visible.any(), "mask must expose at least one visible pixel"

    matches = []
    for transformed in _variants(reference).values():
        if transformed.shape != masked.shape:
            continue
        if np.array_equal(transformed[visible], masked[visible]):
            matches.append(transformed)
    assert len(matches) == 1, "visible holes must identify one transform"

    out = np.full_like(bottom, BLACK)
    _paste(out, matches[0], mask_box[0], mask_box[1])
    return out


def _paint_vertical_band(patch: np.ndarray, color: int) -> None:
    n = patch.shape[0]
    width = random.randint(1, 2 if n >= 6 else 1)
    col = random.randint(0, n - width)
    row0 = random.randint(0, n - 3)
    row1 = random.randint(row0 + 2, n)
    patch[row0:row1, col : col + width] = color


def _paint_horizontal_band(patch: np.ndarray, color: int) -> None:
    n = patch.shape[0]
    height = random.randint(1, 2 if n >= 6 else 1)
    row = random.randint(0, n - height)
    col0 = random.randint(0, n - 3)
    col1 = random.randint(col0 + 2, n)
    patch[row : row + height, col0:col1] = color


def _paint_block(patch: np.ndarray, color: int) -> None:
    n = patch.shape[0]
    height = random.randint(2, min(3, n))
    width = random.randint(2, min(3, n))
    row = random.randint(0, n - height)
    col = random.randint(0, n - width)
    patch[row : row + height, col : col + width] = color


def _paint_elbow(patch: np.ndarray, color: int) -> None:
    n = patch.shape[0]
    arm_a = random.randint(2, n - 1)
    arm_b = random.randint(2, n - 1)
    elbow = np.zeros((arm_a, arm_b), dtype=bool)
    elbow[:, 0] = True
    elbow[-1, :] = True
    for _ in range(random.randint(0, 3)):
        elbow = np.rot90(elbow)
    h, w = elbow.shape
    row = random.randint(0, n - h)
    col = random.randint(0, n - w)
    rr, cc = np.where(elbow)
    patch[row + rr, col + cc] = color


def _sample_reference(size: int) -> np.ndarray | None:
    palette = random.sample(PATTERN_COLORS, random.randint(4, 5))
    patch = np.full((size, size), palette[0], dtype=int)

    painters = [_paint_vertical_band, _paint_horizontal_band, _paint_block, _paint_elbow]
    random.shuffle(painters)
    for color, painter in zip(palette[1:], painters):
        painter(patch, color)
    for _ in range(random.randint(1, 3)):
        random.choice(painters)(patch, random.choice(palette))

    if set(map(int, np.unique(patch))) != set(palette):
        return None

    all_variants = {"identity": patch.copy(), **_variants(patch)}
    signatures = {variant.tobytes() for variant in all_variants.values()}
    if len(signatures) != len(all_variants):
        return None
    return patch


def _choose_holes(variants: dict[str, np.ndarray], target_name: str) -> list[tuple[int, int]]:
    target = variants[target_name]
    remaining = set(variants) - {target_name}
    available = [(r, c) for r in range(target.shape[0]) for c in range(target.shape[1])]
    chosen: list[tuple[int, int]] = []

    while remaining:
        best_cells: list[tuple[int, int]] = []
        best_remaining: set[str] | None = None
        for cell in available:
            value = int(target[cell])
            compatible = {name for name in remaining if int(variants[name][cell]) == value}
            if best_remaining is None or len(compatible) < len(best_remaining):
                best_cells = [cell]
                best_remaining = compatible
            elif len(compatible) == len(best_remaining):
                best_cells.append(cell)
        if best_remaining is None or len(best_remaining) == len(remaining):
            return []
        cell = random.choice(best_cells)
        chosen.append(cell)
        available.remove(cell)
        remaining = best_remaining

    informative = [
        cell
        for cell in available
        if any(int(variants[name][cell]) != int(target[cell]) for name in variants if name != target_name)
    ]
    random.shuffle(informative)
    target_count = random.randint(max(3, len(chosen)), min(8, len(chosen) + 3, target.size))
    while len(chosen) < target_count and informative:
        chosen.append(informative.pop())
    return chosen


def _sample_layout(size: int) -> tuple[int, int, int, int, int, int, int] | None:
    max_extra_sum = 29 - 2 * size
    if max_extra_sum < 4:
        return None
    top_extra = random.randint(2, min(7, max_extra_sum - 2))
    bottom_extra = random.randint(2, min(7, max_extra_sum - top_extra))
    top_h = size + top_extra
    bottom_h = size + bottom_extra
    width = random.randint(size + 2, min(30, size + 10))
    top_r = random.randint(1, top_h - size - 1)
    top_c = random.randint(1, width - size - 1)
    bottom_r = random.randint(1, bottom_h - size - 1)
    bottom_c = random.randint(1, width - size - 1)
    return top_h, bottom_h, width, top_r, top_c, bottom_r, bottom_c


def _expected_output(bottom_h: int, width: int, bottom_r: int, bottom_c: int, target: np.ndarray) -> np.ndarray:
    out = np.full((bottom_h, width), BLACK, dtype=int)
    _paste(out, target, bottom_r, bottom_c)
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(5, 8)
    reference = _sample_reference(size)
    if reference is None:
        return None

    variants = _variants(reference)
    target_name = random.choice(TRANSFORM_NAMES)
    target = variants[target_name]
    holes = _choose_holes(variants, target_name)
    if len(holes) < 3:
        return None

    layout = _sample_layout(size)
    if layout is None:
        return None
    top_h, bottom_h, width, top_r, top_c, bottom_r, bottom_c = layout

    grid = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
    grid[top_h, :] = GREY
    _paste(grid, reference, top_r, top_c)

    mask = np.full((size, size), AZURE, dtype=int)
    for r, c in holes:
        mask[r, c] = int(target[r, c])
    _paste(grid, mask, top_h + 1 + bottom_r, bottom_c)

    expected = _expected_output(bottom_h, width, bottom_r, bottom_c, target)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid masked-transform input")
