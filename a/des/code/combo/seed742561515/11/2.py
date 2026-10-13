import numpy as np


_RNG = np.random.default_rng()
_GEN_RETRIES = 400
_SAMPLE_SIDE_OPTIONS = ("bottom", "right")
_SAMPLE_DIM_OPTIONS = (
    (1, 2),
    (1, 3),
    (2, 2),
    (1, 5),
    (2, 3),
    (2, 4),
    (3, 3),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 4),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _tile_pattern(height: int, width: int, pattern: np.ndarray) -> np.ndarray:
    ph, pw = pattern.shape
    reps = ((height + ph - 1) // ph, (width + pw - 1) // pw)
    tiled = np.tile(pattern, reps)
    return tiled[:height, :width]


def _find_fences_for_color(grid: np.ndarray, bg: int, color: int) -> list[dict]:
    mask = grid == color
    if not np.any(mask):
        return []

    dilated = _dilate4(mask)
    fences: list[dict] = []
    covered = np.zeros_like(mask, dtype=bool)

    for comp in _components(dilated):
        orig = [(r, c) for r, c in comp if mask[r, c]]
        if not orig:
            continue
        r0, c0, r1, c1 = _bbox(orig)
        if r1 - r0 < 2 or c1 - c0 < 2:
            return []

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior != bg):
            return []

        perimeter = _perimeter_coords(r0, c0, r1, c1)
        hole_indices: list[int] = []
        perimeter_color_cells = set()
        for i, (r, c) in enumerate(perimeter):
            value = int(grid[r, c])
            if value == color:
                perimeter_color_cells.add((r, c))
            elif value == bg:
                hole_indices.append(i)
            else:
                return []

        if not 1 <= len(hole_indices) <= 4:
            return []

        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        for idx in hole_indices:
            if perimeter[idx] in corners:
                return []
            prev_cell = perimeter[(idx - 1) % len(perimeter)]
            next_cell = perimeter[(idx + 1) % len(perimeter)]
            if int(grid[prev_cell]) != color or int(grid[next_cell]) != color:
                return []

        if set(orig) != perimeter_color_cells:
            return []

        for r, c in perimeter_color_cells:
            covered[r, c] = True
        fences.append({"bbox": (r0, c0, r1, c1), "holes": len(hole_indices)})

    if not np.array_equal(covered, mask):
        return []
    return sorted(fences, key=lambda item: item["bbox"])


def _find_fences(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    colors = sorted(set(map(int, grid.flatten())) - {bg})
    candidates: list[tuple[int, list[dict]]] = []
    for color in colors:
        fences = _find_fences_for_color(grid, bg, color)
        if fences:
            candidates.append((color, fences))
    assert candidates, "expected at least one frame color"
    candidates.sort(key=lambda item: (-len(item[1]), -sum(len(_perimeter_coords(*f["bbox"])) for f in item[1])))
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_color, fences = _find_fences(g, bg)

    frame_mask = np.zeros_like(g, dtype=bool)
    for fence in fences:
        for r, c in _perimeter_coords(*fence["bbox"]):
            if int(g[r, c]) == frame_color:
                frame_mask[r, c] = True

    sample_mask = (g != bg) & (~frame_mask)
    sample_components = _components(sample_mask)
    samples: list[tuple[int, np.ndarray]] = []
    for comp in sample_components:
        r0, c0, r1, c1 = _bbox(comp)
        bbox_mask = sample_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(bbox_mask), "sample components must be filled rectangles"
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        samples.append((patch.shape[0] * patch.shape[1], patch))

    assert samples, "expected at least one sample pattern"
    areas = [area for area, _ in samples]
    assert len(areas) == len(set(areas)), "sample areas must be unique"
    samples.sort(key=lambda item: item[0])

    sample_count = len(samples)
    hole_kinds = sorted({int(fence["holes"]) for fence in fences})
    assert hole_kinds == list(range(1, sample_count + 1)), "hole counts must match sorted sample indices"

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        pattern = samples[int(fence["holes"]) - 1][1]
        inside_h = r1 - r0 - 1
        inside_w = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(inside_h, inside_w, pattern)
    return out


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _make_pattern(height: int, width: int, colors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    colors = np.asarray(colors, dtype=int)
    modes = (
        lambda r, c: r,
        lambda r, c: c,
        lambda r, c: r + c,
        lambda r, c: 2 * r + c,
        lambda r, c: r + 2 * c,
        lambda r, c: (r // max(1, height // 2)) + (c // max(1, width // 2)),
    )
    mode = modes[int(rng.integers(len(modes)))]
    patch = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            patch[r, c] = int(colors[mode(r, c) % len(colors)])

    used = list(map(int, np.unique(patch)))
    missing = [int(c) for c in colors if int(c) not in used]
    if missing:
        order = [(r, c) for r in range(height) for c in range(width)]
        rng.shuffle(order)
        for color, (r, c) in zip(missing, order):
            patch[r, c] = color

    if height == width:
        patch = np.rot90(patch, k=int(rng.integers(4)))
    elif rng.random() < 0.5:
        patch = np.rot90(patch, k=2)
    if rng.random() < 0.5:
        patch = patch[::-1, :]
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    return np.asarray(patch, dtype=int)


def _paint_fence(grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int, holes: int, rng: np.random.Generator) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    side_names = np.array(["top", "bottom", "left", "right"], dtype=object)
    chosen_sides = list(side_names[rng.choice(len(side_names), size=holes, replace=False)])
    for side in chosen_sides:
        if side == "top":
            c = int(rng.integers(c0 + 1, c1))
            grid[r0, c] = grid[r0 + 1, c]
        elif side == "bottom":
            c = int(rng.integers(c0 + 1, c1))
            grid[r1, c] = grid[r1 - 1, c]
        elif side == "left":
            r = int(rng.integers(r0 + 1, r1))
            grid[r, c0] = grid[r, c0 + 1]
        else:
            r = int(rng.integers(r0 + 1, r1))
            grid[r, c1] = grid[r, c1 - 1]


def _explicit_output(grid: np.ndarray, fences: list[dict], patterns: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        patch = patterns[int(fence["holes"])]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(r1 - r0 - 1, c1 - c0 - 1, patch)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    bg = int(rng.integers(10))
    frame_color = int(rng.choice([c for c in range(10) if c != bg]))
    sample_palette = np.array([c for c in range(10) if c not in (bg, frame_color)], dtype=int)

    sample_side = str(rng.choice(_SAMPLE_SIDE_OPTIONS))
    sample_count = int(rng.integers(1, 5))
    dim_ids = rng.choice(len(_SAMPLE_DIM_OPTIONS), size=sample_count, replace=False)
    sample_shapes = [tuple(_SAMPLE_DIM_OPTIONS[int(i)]) for i in dim_ids]
    sample_shapes.sort(key=lambda shape: shape[0] * shape[1])

    sample_gap = int(rng.integers(1, 3))
    if sample_side == "bottom":
        strip_h = max(h for h, _ in sample_shapes)
        strip_w = sum(w for _, w in sample_shapes) + sample_gap * (sample_count - 1)
    else:
        strip_h = sum(h for h, _ in sample_shapes) + sample_gap * (sample_count - 1)
        strip_w = max(w for _, w in sample_shapes)

    top_margin = int(rng.integers(1, 3))
    left_margin = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    right_margin = int(rng.integers(1, 3))
    separator = int(rng.integers(2, 4))

    fence_region_h = int(rng.integers(12, 18))
    fence_region_w = int(rng.integers(14, 22))

    if sample_side == "bottom":
        core_w = max(fence_region_w, strip_w + int(rng.integers(0, 5)))
        grid_h = top_margin + fence_region_h + separator + strip_h + bottom_margin
        grid_w = left_margin + core_w + right_margin
        fence_region = (top_margin, left_margin, top_margin + fence_region_h - 1, left_margin + core_w - 1)
        sample_row0 = top_margin + fence_region_h + separator
        sample_col0 = left_margin + int(rng.integers(0, core_w - strip_w + 1))
    else:
        core_h = max(fence_region_h, strip_h + int(rng.integers(0, 5)))
        grid_h = top_margin + core_h + bottom_margin
        grid_w = left_margin + fence_region_w + separator + strip_w + right_margin
        fence_region = (top_margin, left_margin, top_margin + core_h - 1, left_margin + fence_region_w - 1)
        sample_row0 = top_margin + int(rng.integers(0, core_h - strip_h + 1))
        sample_col0 = left_margin + fence_region_w + separator

    grid = np.full((grid_h, grid_w), bg, dtype=int)

    patterns: dict[int, np.ndarray] = {}
    if sample_side == "bottom":
        cursor = sample_col0
        for i, (ph, pw) in enumerate(sample_shapes, start=1):
            n_colors = 2 if ph * pw <= 4 else int(rng.integers(2, 4))
            colors = rng.choice(sample_palette, size=n_colors, replace=False)
            patch = _make_pattern(ph, pw, colors, rng)
            row = sample_row0 + int(rng.integers(0, strip_h - ph + 1))
            grid[row : row + ph, cursor : cursor + pw] = patch
            patterns[i] = patch
            cursor += pw + sample_gap
    else:
        cursor = sample_row0
        for i, (ph, pw) in enumerate(sample_shapes, start=1):
            n_colors = 2 if ph * pw <= 4 else int(rng.integers(2, 4))
            colors = rng.choice(sample_palette, size=n_colors, replace=False)
            patch = _make_pattern(ph, pw, colors, rng)
            col = sample_col0 + int(rng.integers(0, strip_w - pw + 1))
            grid[cursor : cursor + ph, col : col + pw] = patch
            patterns[i] = patch
            cursor += ph + sample_gap

    fence_count = int(rng.integers(max(2, sample_count), min(6, sample_count + 3)))
    hole_counts = list(range(1, sample_count + 1))
    while len(hole_counts) < fence_count:
        hole_counts.append(int(rng.integers(1, sample_count + 1)))
    rng.shuffle(hole_counts)

    fr0, fc0, fr1, fc1 = fence_region
    region_h = fr1 - fr0 + 1
    region_w = fc1 - fc0 + 1
    if region_h < 7 or region_w < 7:
        return None

    fences: list[dict] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for holes in hole_counts:
        placed = False
        for _ in range(150):
            max_h = min(9, region_h)
            max_w = min(10, region_w)
            outer_h = int(rng.integers(5, max_h + 1))
            outer_w = int(rng.integers(5, max_w + 1))
            if outer_h < 5 or outer_w < 5:
                continue
            r0 = int(rng.integers(fr0, fr1 - outer_h + 2))
            c0 = int(rng.integers(fc0, fc1 - outer_w + 2))
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_rectangles_overlap(bbox, other, pad=2) for other in placed_boxes):
                continue
            placed_boxes.append(bbox)
            fences.append({"bbox": bbox, "holes": holes})
            placed = True
            break
        if not placed:
            return None

    for fence in fences:
        _paint_fence(grid, fence["bbox"], frame_color, int(fence["holes"]), rng)

    expected = _explicit_output(grid, fences, patterns)

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
