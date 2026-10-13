import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each fenced interior pattern out through the fence's unique gap.

    Input structure:
    - The background is the most common color.
    - Several monochrome 1-pixel-thick rectangular fences appear; each fence is
      missing exactly one non-corner border pixel, creating a single gap on one
      side.
    - Each fence contains one compact multicolor interior pattern made of
      2 to 6 non-background pixels, and the fence exteriors are otherwise blank.

    Rule:
    - Detect every fence and find the missing border pixel to determine the gap
      side and gap coordinate.
    - Extract the interior pattern's minimal bounding box content.
    - Clear the original interior pattern to background, then place the same
      bounded pattern immediately outside the gap side, aligned so that the
      pattern bbox center row or center column matches the gap row or column.
    - Keep the fence unchanged, including the missing gap pixel.

    Key insight:
    - The unique missing border pixel is an exit marker: it gives both the
      outward direction and the alignment axis for relocating the interior
      pattern from inside the fence to just outside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    fences = _find_fences(g, bg)
    assert fences, "expected at least one gapped fence"

    clear_cells: set[tuple[int, int]] = set()
    paint_cells: dict[tuple[int, int], int] = {}
    height, width = map(int, g.shape)

    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern_mask = inner != bg
        assert np.any(pattern_mask), "each fence must contain an interior pattern"
        assert not np.any(inner == fence["color"]), "fence color cannot reappear inside its own fence"

        pattern_coords = np.argwhere(pattern_mask)
        pr0, pc0 = pattern_coords.min(axis=0)
        pr1, pc1 = pattern_coords.max(axis=0)
        pattern = inner[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        ph, pw = map(int, pattern.shape)

        dst_r0, dst_c0 = _destination_origin(fence["bbox"], fence["gap"], fence["side"], ph, pw)
        assert 0 <= dst_r0 and 0 <= dst_c0, "moved pattern must stay in bounds"
        assert dst_r0 + ph <= height and dst_c0 + pw <= width, "moved pattern must stay in bounds"

        for rr, cc in pattern_coords:
            clear_cells.add((int(r0 + 1 + rr), int(c0 + 1 + cc)))

        for rr in range(ph):
            for cc in range(pw):
                color = int(pattern[rr, cc])
                if color == bg:
                    continue
                target = (int(dst_r0 + rr), int(dst_c0 + cc))
                assert int(g[target]) == bg, "destination area must be empty background in the input"
                assert target not in paint_cells, "moved patterns must not overlap"
                paint_cells[target] = color

    out = g.copy()
    for r, c in clear_cells:
        out[r, c] = bg
    for (r, c), color in paint_cells.items():
        out[r, c] = color
    return out


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_boundary(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            cells.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            cells.append((r, c0))
    return cells


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    fences: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            boundary = _rectangle_boundary(r0, c0, r1, c1)
            boundary_set = set(boundary)
            coord_set = {tuple(map(int, pt)) for pt in coords}
            if len(coord_set) != len(boundary) - 1:
                continue
            if any(cell not in boundary_set for cell in coord_set):
                continue

            missing = [cell for cell in boundary if cell not in coord_set]
            if len(missing) != 1:
                continue
            gap_r, gap_c = missing[0]
            if gap_r == r0 and gap_c in (c0, c1):
                continue
            if gap_r == r1 and gap_c in (c0, c1):
                continue

            if gap_r == r0:
                side = "top"
            elif gap_r == r1:
                side = "bottom"
            elif gap_c == c0:
                side = "left"
            elif gap_c == c1:
                side = "right"
            else:
                continue

            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not np.any(inner != bg):
                continue

            fences.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gap": (int(gap_r), int(gap_c)),
                    "side": side,
                }
            )

    fences.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return fences


def _destination_origin(
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    side: str,
    pattern_h: int,
    pattern_w: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    gap_r, gap_c = gap
    if side == "top":
        return r0 - pattern_h, gap_c - pattern_w // 2
    if side == "bottom":
        return r1 + 1, gap_c - pattern_w // 2
    if side == "left":
        return gap_r - pattern_h // 2, c0 - pattern_w
    if side == "right":
        return gap_r - pattern_h // 2, c1 + 1
    raise AssertionError(f"unknown side: {side}")


_RNG = np.random.default_rng()
_GEN_CALLS = 0
_GEN_RETRIES = 400
_PLACE_RETRIES = 120
_DIRECTIONS = ("top", "right", "bottom", "left")


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _build_shape_catalog() -> tuple[np.ndarray, ...]:
    bases = [
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ]
    unique: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for base in bases:
        for flipped in (base, np.fliplr(base)):
            for turns in range(4):
                mask = _normalize_mask(np.rot90(flipped, turns))
                if not (2 <= int(mask.sum()) <= 6):
                    continue
                key = (tuple(map(int, mask.shape)), mask.tobytes())
                unique[key] = mask.copy()
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda item: (
                int(unique[item].sum()),
                item[0][0],
                item[0][1],
                item[1],
            ),
        )
    )


_SHAPES = _build_shape_catalog()


def _shape_options(direction: str) -> list[np.ndarray]:
    if direction in ("top", "bottom"):
        return [shape for shape in _SHAPES if shape.shape[1] % 2 == 1]
    return [shape for shape in _SHAPES if shape.shape[0] % 2 == 1]


def _sample_pattern(mask: np.ndarray, bg: int, fence_color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    num_cells = int(coords.shape[0])
    color_pool = [color for color in range(10) if color not in {bg, fence_color}]
    num_colors = int(_RNG.integers(2, min(3, num_cells) + 1))
    chosen = [int(color) for color in _RNG.choice(color_pool, size=num_colors, replace=False)]
    assignments = chosen[:]
    while len(assignments) < num_cells:
        assignments.append(int(chosen[int(_RNG.integers(0, len(chosen)))]))
    _RNG.shuffle(assignments)

    pattern = np.full(mask.shape, bg, dtype=int)
    for (r, c), color in zip(coords, assignments):
        pattern[int(r), int(c)] = int(color)
    return pattern


def _expand_bbox(r0: int, c0: int, r1: int, c1: int, height: int, width: int, pad: int = 1) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(height - 1, r1 + pad),
        min(width - 1, c1 + pad),
    )


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    bg: int,
    fence_color: int,
    direction: str,
) -> bool:
    height, width = map(int, grid.shape)
    shapes = _shape_options(direction)

    for _ in range(_PLACE_RETRIES):
        mask = shapes[int(_RNG.integers(0, len(shapes)))]
        pattern = _sample_pattern(mask, bg, fence_color)
        ph, pw = map(int, pattern.shape)

        inner_h = int(ph + _RNG.integers(2, 6))
        inner_w = int(pw + _RNG.integers(2, 6))
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if direction == "top":
            row_low, row_high = ph, height - outer_h
            col_low, col_high = 0, width - outer_w
        elif direction == "bottom":
            row_low, row_high = 0, height - outer_h - ph
            col_low, col_high = 0, width - outer_w
        elif direction == "left":
            row_low, row_high = 0, height - outer_h
            col_low, col_high = pw, width - outer_w
        else:
            row_low, row_high = 0, height - outer_h
            col_low, col_high = 0, width - outer_w - pw

        if row_low > row_high or col_low > col_high:
            continue

        r0 = int(_RNG.integers(row_low, row_high + 1))
        c0 = int(_RNG.integers(col_low, col_high + 1))
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if direction in ("top", "bottom"):
            gap_c = int(_RNG.integers(c0 + 1, c1))
            gap = (r0 if direction == "top" else r1, gap_c)
        else:
            gap_r = int(_RNG.integers(r0 + 1, r1))
            gap = (gap_r, c0 if direction == "left" else c1)

        dst_r0, dst_c0 = _destination_origin((r0, c0, r1, c1), gap, direction, ph, pw)
        dst_r1 = dst_r0 + ph - 1
        dst_c1 = dst_c0 + pw - 1
        if not (0 <= dst_r0 <= dst_r1 < height and 0 <= dst_c0 <= dst_c1 < width):
            continue

        block_r0 = min(r0, dst_r0)
        block_c0 = min(c0, dst_c0)
        block_r1 = max(r1, dst_r1)
        block_c1 = max(c1, dst_c1)
        er0, ec0, er1, ec1 = _expand_bbox(block_r0, block_c0, block_r1, block_c1, height, width)
        if np.any(reserved[er0 : er1 + 1, ec0 : ec1 + 1]):
            continue

        pattern_r = int(_RNG.integers(1, inner_h - ph))
        pattern_c = int(_RNG.integers(1, inner_w - pw))

        boundary = _rectangle_boundary(r0, c0, r1, c1)
        for br, bc in boundary:
            if (br, bc) != gap:
                grid[br, bc] = fence_color

        for rr in range(ph):
            for cc in range(pw):
                color = int(pattern[rr, cc])
                if color != bg:
                    grid[r0 + 1 + pattern_r + rr, c0 + 1 + pattern_c + cc] = color

        reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True
        return True

    return False


def _gen_attempt(object_count: int, directions: list[str]) -> np.ndarray | None:
    min_side = 14 + 3 * object_count
    height = int(_RNG.integers(min_side, 31))
    width = int(_RNG.integers(min_side, 31))
    bg = int(_RNG.integers(0, 10))
    grid = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    color_pool = [color for color in range(10) if color != bg]
    fence_colors = [int(color) for color in _RNG.choice(color_pool, size=object_count, replace=False)]

    for fence_color, direction in zip(fence_colors, directions):
        if not _place_object(grid, reserved, bg, fence_color, direction):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    object_count = 1 + (_GEN_CALLS % 4)
    start = _GEN_CALLS % 4
    directions = [_DIRECTIONS[(start + idx) % 4] for idx in range(object_count)]
    _RNG.shuffle(directions)
    _GEN_CALLS += 1

    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(object_count, directions)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
