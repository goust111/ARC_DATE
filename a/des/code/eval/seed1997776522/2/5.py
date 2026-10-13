import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
PATTERN_SIZE = 5
BLOCK_H = 5
BLOCK_W = 9
GEN_RETRIES = 300

_RNG = np.random.default_rng()

_NEIGHBOR_KEYS = (
    ((-1, 0), (0, -1), (1, 0)),
    ((-1, -1), (-1, 1), (1, 0)),
    ((0, 1), (1, -1), (1, 1)),
    ((-1, 0), (0, 1), (1, 1)),
    ((-1, -1), (0, -1), (1, 0), (1, 1)),
    ((-1, 1), (0, -1), (0, 1), (1, -1)),
)


def _mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_SHAPES = (
    _mask(
        (
            "00100",
            "00100",
            "11111",
            "00100",
            "00100",
        )
    ),
    _mask(
        (
            "00100",
            "01110",
            "11111",
            "01110",
            "00100",
        )
    ),
    _mask(
        (
            "11111",
            "10001",
            "11111",
            "10001",
            "11111",
        )
    ),
    _mask(
        (
            "00100",
            "01110",
            "11111",
            "00100",
            "00100",
        )
    ),
    _mask(
        (
            "10001",
            "10001",
            "11111",
            "10001",
            "10001",
        )
    ),
    _mask(
        (
            "10001",
            "01010",
            "11111",
            "01010",
            "10001",
        )
    ),
)


def _neighbor_key(grid: np.ndarray, r: int, c: int) -> tuple[tuple[int, int], ...]:
    offsets = []
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == GREY:
                offsets.append((dr, dc))
    return tuple(offsets)


def _paint_pattern(out: np.ndarray, mask: np.ndarray, r: int, c: int, color: int) -> None:
    r0 = r - 2
    c0 = c - 2
    assert 0 <= r0 and r0 + PATTERN_SIZE <= out.shape[0]
    assert 0 <= c0 and c0 + PATTERN_SIZE <= out.shape[1]
    patch = out[r0 : r0 + PATTERN_SIZE, c0 : c0 + PATTERN_SIZE]
    patch[mask] = color


def _extract_library(grid: np.ndarray, divider: int) -> dict[tuple[tuple[int, int], ...], np.ndarray]:
    _, w = grid.shape
    mapping: dict[tuple[tuple[int, int], ...], np.ndarray] = {}

    for r in range(divider):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, GREY):
                continue
            if r - 2 < 0 or r + 2 >= divider or c - 1 < 0 or c + 7 >= w:
                continue

            key = _neighbor_key(grid, r, c)
            if not key:
                continue

            mask = grid[r - 2 : r + 3, c + 3 : c + 8] == color
            assert mask.shape == (PATTERN_SIZE, PATTERN_SIZE)
            assert mask[2, 2], "library growth pattern must include its center"
            previous = mapping.get(key)
            if previous is None:
                mapping[key] = mask.copy()
            else:
                assert np.array_equal(previous, mask), "duplicate library key with different patterns"

    assert 3 <= len(mapping) <= 4, "expected 3-4 library examples"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each workspace seed using the library example whose grey-neighbor pattern matches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(divider_rows) == 1, "expected a single full-width grey divider"
    divider = int(divider_rows[0])

    mapping = _extract_library(g, divider)
    out = g.copy()
    out[divider + 1 :][out[divider + 1 :] == GREY] = BG

    seeds: list[tuple[int, int]] = []
    for r in range(divider + 1, g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BG, GREY):
                continue
            key = _neighbor_key(g, r, c)
            if key:
                seeds.append((r, c))

    for r, c in seeds:
        color = int(g[r, c])
        key = _neighbor_key(g, r, c)
        assert key in mapping, "workspace seed pattern missing from library"
        _paint_pattern(out, mapping[key], r, c, color)

    return out


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _shape_overlap(center_a: tuple[int, int], mask_a: np.ndarray, center_b: tuple[int, int], mask_b: np.ndarray) -> bool:
    cells_a = set()
    for rr, cc in np.argwhere(mask_a):
        cells_a.add((int(center_a[0] + rr - 2), int(center_a[1] + cc - 2)))
    for rr, cc in np.argwhere(mask_b):
        cell = (int(center_b[0] + rr - 2), int(center_b[1] + cc - 2))
        if cell in cells_a:
            return True
    return False


def _place_centers(
    rng: np.random.Generator,
    divider: int,
    height: int,
    width: int,
    seed_count: int,
    want_overlap: bool,
) -> tuple[list[tuple[int, int]], bool] | None:
    row_min = divider + 3
    row_max = height - 3
    col_min = 2
    col_max = width - 3
    if row_min > row_max or col_min > col_max:
        return None

    centers: list[tuple[int, int]] = []
    occupied: list[tuple[int, int, int, int]] = []
    overlap_made = False

    if want_overlap:
        attempts = 0
        while attempts < 60 and not overlap_made:
            attempts += 1
            r = int(rng.integers(row_min, row_max + 1))
            c_left = int(rng.integers(col_min, col_max - 2))
            pair = [(r, c_left), (r, c_left + 3)]
            rects = [(r - 1, c - 1, r + 1, c + 1) for r, c in pair]
            if all(not _rectangles_overlap(rects[0], rect) for rect in occupied) and all(
                not _rectangles_overlap(rects[1], rect) for rect in occupied
            ):
                centers.extend(pair)
                occupied.extend(rects)
                overlap_made = True
        if not overlap_made:
            return None

    attempts = 0
    while len(centers) < seed_count and attempts < 500:
        attempts += 1
        r = int(rng.integers(row_min, row_max + 1))
        c = int(rng.integers(col_min, col_max + 1))
        rect = (r - 1, c - 1, r + 1, c + 1)
        if any(_rectangles_overlap(rect, prev) for prev in occupied):
            continue
        centers.append((r, c))
        occupied.append(rect)

    if len(centers) != seed_count:
        return None
    return centers, overlap_made


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    example_count = int(rng.integers(3, 5))
    slot_ids = rng.choice(4, size=example_count, replace=False)
    key_ids = rng.choice(len(_NEIGHBOR_KEYS), size=example_count, replace=False)
    shape_ids = rng.choice(len(_SHAPES), size=example_count, replace=False)
    pair_order = rng.permutation(example_count)

    left_pad = int(rng.integers(1, 4))
    right_pad = int(rng.integers(1, 4))
    top_pad = int(rng.integers(0, 3))
    bottom_pad = int(rng.integers(1, 3))
    hgap = int(rng.integers(1, 4))
    vgap = int(rng.integers(1, 3))
    workspace_h = int(rng.integers(10, 15))

    width = left_pad + BLOCK_W + hgap + BLOCK_W + right_pad
    library_h = top_pad + BLOCK_H + vgap + BLOCK_H + bottom_pad
    divider = library_h
    height = library_h + 1 + workspace_h

    grid = np.full((height, width), BG, dtype=int)
    grid[divider, :] = GREY

    library_colors = rng.choice(COLOR_POOL, size=example_count, replace=False)
    entries: list[dict[str, object]] = []
    for i in range(example_count):
        slot = int(slot_ids[i])
        slot_r = slot // 2
        slot_c = slot % 2
        r0 = top_pad + slot_r * (BLOCK_H + vgap)
        c0 = left_pad + slot_c * (BLOCK_W + hgap)
        center = (r0 + 2, c0 + 1)
        key = _NEIGHBOR_KEYS[int(key_ids[i])]
        shape = _SHAPES[int(shape_ids[pair_order[i]])]
        color = int(library_colors[i])
        entries.append({"center": center, "key": key, "shape": shape, "color": color})

        cr, cc = center
        grid[cr, cc] = color
        for dr, dc in key:
            grid[cr + dr, cc + dc] = GREY
        patch = grid[r0 : r0 + PATTERN_SIZE, c0 + 4 : c0 + 9]
        patch[shape] = color

    want_overlap = bool(rng.integers(0, 2))
    seed_count = example_count + int(rng.integers(1, 3))
    placed = _place_centers(rng, divider, height, width, seed_count, want_overlap)
    if placed is None:
        return None
    centers, overlap_made = placed
    if want_overlap and not overlap_made:
        return None

    example_ids = list(range(example_count))
    if seed_count > example_count:
        extra = rng.choice(example_count, size=seed_count - example_count, replace=True).tolist()
        example_ids.extend(int(x) for x in extra)
    if want_overlap and example_count >= 2:
        tail = example_ids[2:]
        rng.shuffle(tail)
        example_ids = [0, 1] + tail
    else:
        rng.shuffle(example_ids)

    seed_colors = rng.choice(COLOR_POOL, size=seed_count, replace=True).tolist()
    if len(set(seed_colors)) == 1 and seed_count > 1:
        alternatives = [c for c in COLOR_POOL if c != seed_colors[0]]
        seed_colors[0] = int(rng.choice(alternatives))
    if want_overlap and seed_count >= 2 and seed_colors[0] == seed_colors[1]:
        alternatives = [c for c in COLOR_POOL if c != seed_colors[0]]
        seed_colors[1] = int(rng.choice(alternatives))

    seed_specs = []
    for i, (r, c) in enumerate(centers):
        entry = entries[int(example_ids[i])]
        color = int(seed_colors[i])
        seed_specs.append((int(r), int(c), entry["key"], np.asarray(entry["shape"], dtype=bool), color))

    for r, c, key, _shape, color in seed_specs:
        grid[r, c] = color
        for dr, dc in key:
            grid[r + dr, c + dc] = GREY

    if want_overlap:
        assert _shape_overlap(seed_specs[0][:2], seed_specs[0][3], seed_specs[1][:2], seed_specs[1][3])

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert np.array_equal(out[:divider, :], grid[:divider, :])
    assert np.array_equal(out[divider, :], grid[divider, :])
    assert not np.any(out[divider + 1 :] == GREY)
    assert np.any(out != grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
