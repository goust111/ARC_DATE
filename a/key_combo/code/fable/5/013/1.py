from __future__ import annotations

from collections import Counter, deque

import numpy as np


MARKER_COLOR = 5
ARM_OFFSETS = (
    (0, -3),
    (0, -2),
    (0, 2),
    (0, 3),
    (-3, 0),
    (-2, 0),
    (2, 0),
    (3, 0),
)
GAP_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        queue = deque([(row, col)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_hidden_center(grid: np.ndarray, row: int, col: int, bg: int) -> bool:
    return (
        int(grid[row, col]) == bg
        and all(int(grid[row + dr, col + dc]) == MARKER_COLOR for dr, dc in ARM_OFFSETS)
        and all(int(grid[row + dr, col + dc]) == bg for dr, dc in GAP_OFFSETS)
    )


def _hidden_centers(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    height, width = grid.shape
    return [
        (r, c)
        for r in range(3, height - 3)
        for c in range(3, width - 3)
        if _is_hidden_center(grid, r, c, bg)
    ]


def _assert_marker_bars(grid: np.ndarray) -> None:
    bars = _components(grid == MARKER_COLOR)
    assert len(bars) == 8, "expected exactly eight separate grey bars"
    for bar in bars:
        assert len(bar) == 2, "each grey marker must contain two pixels"
        (r0, c0), (r1, c1) = map(tuple, bar)
        assert (r0 == r1 and abs(c0 - c1) == 1) or (
            c0 == c1 and abs(r0 - r1) == 1
        ), "grey marker components must be straight two-pixel bars"


def _solid_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(set(map(int, grid.flat)) - {bg, MARKER_COLOR}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            height, width = r1 - r0 + 1, c1 - c0 + 1
            assert len(coords) == height * width, "every sprite must be a solid rectangle"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "size": (height, width),
                }
            )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the uniquely size-matched solid sprite into a rectangle implied by two missing plus centers.

    Input structure:
    - The dominant color is background, and exactly eight separate grey two-pixel bars form two implied plus signs.
    - Each plus has four bars at offsets two and three from an unpainted center, with background one-cell gaps.
    - Several separated monochrome solid rectangular sprites have pairwise-distinct bounding-box dimensions.

    Rule:
    - Use the two missing plus centers as opposite inclusive corners of an axis-aligned invisible rectangle.
    - Select the unique sprite whose bounding-box height and width exactly equal that rectangle, without rotation.
    - Erase all grey bars and the selected sprite, then paint that sprite into the implied rectangle; preserve every distractor.

    Key insight:
    - Neither a visible cavity nor a marked corner exists: both centers must first be inferred from the bars, and only their rectangle makes the exact two-dimensional size match meaningful.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    height, width = g.shape
    assert height >= 7 and width >= 7, "grid is too small for implied plus markers"

    bg = _background(g)
    assert bg != MARKER_COLOR, "grey is reserved for the marker bars"
    _assert_marker_bars(g)
    centers = _hidden_centers(g, bg)
    assert len(centers) == 2, "the grey bars must imply exactly two missing centers"

    (ra, ca), (rb, cb) = centers
    r0, r1 = sorted((ra, rb))
    c0, c1 = sorted((ca, cb))
    assert ra != rb and ca != cb, "the centers must be opposite, not adjacent, rectangle corners"
    target_size = (r1 - r0 + 1, c1 - c0 + 1)

    sprites = _solid_sprites(g, bg)
    assert sprites, "expected at least one rectangular sprite"
    sizes = [sprite["size"] for sprite in sprites]
    assert len(sizes) == len(set(sizes)), "sprite bounding-box dimensions must be pairwise distinct"
    matches = [sprite for sprite in sprites if sprite["size"] == target_size]
    assert len(matches) == 1, "the implied rectangle must have one exact size match"
    source = matches[0]

    sr0, sc0, sr1, sc1 = source["bbox"]
    source_cells = {(int(r), int(c)) for r, c in source["coords"]}
    target_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    assert source_cells.isdisjoint(target_cells), "the source sprite must start outside its destination"

    out = g.copy()
    out[out == MARKER_COLOR] = bg
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
    assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == bg), (
        "the invisible destination may contain markers, but no other sprite"
    )
    out[r0 : r1 + 1, c0 : c1 + 1] = int(source["color"])
    return out


def _marker_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in ARM_OFFSETS]


def _free_with_padding(
    occupied: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int = 1
) -> bool:
    grid_h, grid_w = occupied.shape
    rr0, cc0 = max(0, r0 - padding), max(0, c0 - padding)
    rr1, cc1 = min(grid_h, r0 + height + padding), min(grid_w, c0 + width + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_rectangles(
    grid: np.ndarray,
    occupied: np.ndarray,
    specs: list[tuple[int, int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int, int]] | None:
    """Place (height, width, color) specs, returning records in original order."""
    height, width = grid.shape
    records: list[tuple[int, int, int, int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i][0] * specs[i][1], reverse=True)
    for index in order:
        rect_h, rect_w, color = specs[index]
        candidates = [
            (r, c)
            for r in range(height - rect_h + 1)
            for c in range(width - rect_w + 1)
            if _free_with_padding(occupied, r, c, rect_h, rect_w)
        ]
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        records[index] = (r0, c0, rect_h, rect_w, color)
    assert all(record is not None for record in records)
    return [record for record in records if record is not None]


def _semantic_validation(
    grid: np.ndarray,
    centers: tuple[tuple[int, int], tuple[int, int]],
    records: list[tuple[int, int, int, int, int]],
) -> bool:
    """Validate the stated semantics independently of generator/solver closure."""
    bg = _background(grid)
    found_centers = _hidden_centers(grid, bg)
    if set(found_centers) != set(centers) or len(found_centers) != 2:
        return False
    try:
        _assert_marker_bars(grid)
    except AssertionError:
        return False

    (ra, ca), (rb, cb) = centers
    r0, r1 = sorted((ra, rb))
    c0, c1 = sorted((ca, cb))
    target_size = (r1 - r0 + 1, c1 - c0 + 1)
    sizes = [(h, w) for _r, _c, h, w, _color in records]
    if len(sizes) < 4 or len(sizes) != len(set(sizes)) or sizes.count(target_size) != 1:
        return False

    # Required hard negatives rule out rotation, height-only, width-only, and area-only matching.
    th, tw = target_size
    if (tw, th) not in sizes:
        return False
    if not any(h == th and w != tw for h, w in sizes):
        return False
    if not any(w == tw and h != th for h, w in sizes):
        return False
    if sum(h * w == th * tw for h, w in sizes) < 2:
        return False

    target_index = sizes.index(target_size)
    sr, sc, sh, sw, source_color = records[target_index]
    if not (sr + sh <= r0 or r1 < sr or sc + sw <= c0 or c1 < sc):
        return False

    cleared = grid.copy()
    cleared[cleared == MARKER_COLOR] = bg
    if not np.all(cleared[r0 : r1 + 1, c0 : c1 + 1] == bg):
        return False

    # Each cross is necessary: deleting one of its uniquely owned arm pixels must destroy that center.
    for center in centers:
        mutated = grid.copy()
        mr, mc = _marker_cells(center)[0]
        mutated[mr, mc] = bg
        if center in _hidden_centers(mutated, bg):
            return False

    expected = grid.copy()
    expected[expected == MARKER_COLOR] = bg
    expected[sr : sr + sh, sc : sc + sw] = bg
    expected[r0 : r1 + 1, c0 : c1 + 1] = source_color
    if np.array_equal(expected, grid):
        return False
    try:
        return np.array_equal(main(grid), expected)
    except (AssertionError, ValueError, IndexError):
        return False


def _generation_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(27, 31))
    grid_w = int(rng.integers(27, 31))
    bg = int(rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9]))
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    # A non-square target makes rotation a visible, testable wrong answer.
    target_h = int(rng.integers(5, 9))
    target_w_choices = [size for size in range(5, 9) if size != target_h]
    target_w = int(rng.choice(target_w_choices))
    r0 = int(rng.integers(3, grid_h - target_h - 2))
    c0 = int(rng.integers(3, grid_w - target_w - 2))
    r1, c1 = r0 + target_h - 1, c0 + target_w - 1
    if bool(rng.integers(2)):
        centers = ((r0, c0), (r1, c1))
    else:
        centers = ((r0, c1), (r1, c0))

    marker_cells = _marker_cells(centers[0]) + _marker_cells(centers[1])
    if len(set(marker_cells)) != 16:
        return None
    for r, c in marker_cells:
        grid[r, c] = MARKER_COLOR
    if set(_hidden_centers(grid, bg)) != set(centers):
        return None
    try:
        _assert_marker_bars(grid)
    except AssertionError:
        return None

    # Reserve the invisible rectangle and every marker cell from sprite placement.
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    for r, c in marker_cells:
        occupied[r, c] = True

    used_sizes = {(target_h, target_w), (target_w, target_h)}
    same_height_w = int(rng.choice([w for w in range(2, 9) if w != target_w]))
    same_width_h = int(rng.choice([h for h in range(2, 9) if h != target_h]))
    mandatory_sizes = [
        (target_h, target_w),
        (target_w, target_h),
        (target_h, same_height_w),
        (same_width_h, target_w),
    ]
    if len(set(mandatory_sizes)) != 4:
        return None
    used_sizes.update(mandatory_sizes)

    sprite_count = int(rng.integers(4, 7))
    sizes = list(mandatory_sizes)
    while len(sizes) < sprite_count:
        size = (int(rng.integers(2, 8)), int(rng.integers(2, 8)))
        if size not in used_sizes:
            used_sizes.add(size)
            sizes.append(size)

    colors = [color for color in range(10) if color not in {bg, MARKER_COLOR}]
    chosen_colors = list(map(int, rng.choice(colors, size=sprite_count, replace=False)))
    rng.shuffle(chosen_colors)
    specs = [(h, w, color) for (h, w), color in zip(sizes, chosen_colors)]
    records = _place_rectangles(grid, occupied, specs, rng)
    if records is None:
        return None

    if not _semantic_validation(grid, centers, records):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        result = _generation_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid invisible-cavity puzzle")
