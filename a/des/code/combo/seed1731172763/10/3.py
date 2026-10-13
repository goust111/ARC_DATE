import numpy as np
from collections import deque


BG = 8
FG_COLORS = tuple(color for color in range(10) if color != BG)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

RNG = np.random.default_rng()

# Generator bounds.
GRID_MAX = 30
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX_EXCLUSIVE = 5
OBJECT_SIZE_MIN = 3
OBJECT_SIZE_MAX = 6
INPUT_GAP_MIN = 2
INPUT_GAP_MAX = 4
BORDER_MIN = 1
BORDER_MAX = 3
GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _spiral_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top, bottom = 0, h - 1
    left, right = 0, w - 1

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        top += 1

        for r in range(top, bottom + 1):
            coords.append((r, right))
        right -= 1

        if top <= bottom:
            for c in range(right, left - 1, -1):
                coords.append((bottom, c))
            bottom -= 1

        if left <= right:
            for r in range(bottom, top - 1, -1):
                coords.append((r, left))
            left += 1

    return coords


def _spiral_repack(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    seq = [int(patch[r, c]) for r, c in _spiral_coords(h, w)]
    return np.array(seq, dtype=int).reshape(h, w)


def main(grid: np.ndarray) -> np.ndarray:
    """Spiral-read each foreground object's bbox and stack the repacked rectangles."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    if not components:
        return g.copy()

    components.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))

    patches: list[np.ndarray] = []
    total_h = len(components) - 1
    max_w = 0
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        repacked = _spiral_repack(patch)
        patches.append(repacked)
        total_h += repacked.shape[0]
        max_w = max(max_w, int(repacked.shape[1]))

    out = np.full((total_h, max_w), BG, dtype=int)
    row = 0
    for patch in patches:
        h, w = patch.shape
        out[row : row + h, :w] = patch
        row += h + 1
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start_r, start_c = map(int, coords[0])
    q = deque([(start_r, start_c)])
    seen = { (start_r, start_c) }

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _has_interior_bg(mask: np.ndarray) -> bool:
    if mask.shape[0] <= 2 or mask.shape[1] <= 2:
        return bool((~mask).any())
    return bool((~mask[1:-1, 1:-1]).any())


def _make_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    # Add one interior brace so holey objects are not all identical.
    if h >= 5 and w >= 4 and RNG.random() < 0.5:
        col = int(RNG.integers(1, w - 1))
        mask[1:-1, col] = True
    elif w >= 5 and h >= 4 and RNG.random() < 0.5:
        row = int(RNG.integers(1, h - 1))
        mask[row, 1:-1] = True

    return mask


def _make_carved_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    min_area = max(4, int(np.ceil(area * 0.55)))
    max_removed = max(1, area - min_area)

    for _ in range(120):
        mask = np.ones((h, w), dtype=bool)
        target = int(RNG.integers(1, max_removed + 1))
        removed = 0

        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        border = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        RNG.shuffle(interior)
        RNG.shuffle(border)
        candidates = interior + border

        for r, c in candidates:
            if removed >= target:
                break
            mask[r, c] = False
            if int(mask.sum()) < min_area or not _touches_all_sides(mask) or not _is_connected(mask):
                mask[r, c] = True
                continue
            removed += 1

        if removed >= 1 and _touches_all_sides(mask) and _is_connected(mask) and _has_interior_bg(mask):
            return mask

    return _make_frame_mask(h, w)


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h, w), BG, dtype=int)

    color_count = int(RNG.integers(2, min(5, int(mask.sum()) + 1)))
    palette = RNG.choice(np.array(FG_COLORS, dtype=int), size=color_count, replace=False)
    mode = str(RNG.choice(np.array(["rows", "cols", "checker", "quadrants"], dtype=object)))

    if mode == "rows":
        bands = RNG.choice(palette, size=h, replace=True)
        for r, c in np.argwhere(mask):
            patch[r, c] = int(bands[r])
    elif mode == "cols":
        bands = RNG.choice(palette, size=w, replace=True)
        for r, c in np.argwhere(mask):
            patch[r, c] = int(bands[c])
    elif mode == "checker":
        offset = int(RNG.integers(0, color_count))
        for r, c in np.argwhere(mask):
            patch[r, c] = int(palette[(r + c + offset) % color_count])
    else:
        split_r = int(RNG.integers(1, h))
        split_c = int(RNG.integers(1, w))
        block_colors = RNG.choice(palette, size=4, replace=True)
        for r, c in np.argwhere(mask):
            idx = (1 if r >= split_r else 0) * 2 + (1 if c >= split_c else 0)
            patch[r, c] = int(block_colors[idx])

    used = np.unique(patch[mask])
    if used.size < 2:
        coords = np.argwhere(mask)
        patch[tuple(coords[0])] = int(palette[0])
        patch[tuple(coords[-1])] = int(palette[1 % color_count])

    return patch


def _build_patch(h: int, w: int, mode: str) -> np.ndarray:
    if mode == "solid":
        mask = np.ones((h, w), dtype=bool)
    elif mode == "frame":
        mask = _make_frame_mask(h, w)
    else:
        mask = _make_carved_mask(h, w)
    return _colorize_mask(mask)


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _validate_grid(grid: np.ndarray, expected_count: int) -> bool:
    if grid.ndim != 2 or grid.shape[0] > GRID_MAX or grid.shape[1] > GRID_MAX:
        return False
    if not bool((grid == BG).any()):
        return False

    comps = _find_components(grid)
    if len(comps) != expected_count:
        return False

    comps.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    left_edges = [int(comp["bbox"][1]) for comp in comps]
    if len(set(left_edges)) != len(left_edges):
        return False

    changed = 0
    solid_count = 0
    holey_count = 0

    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = _bbox_shape(comp["bbox"])

        if h < OBJECT_SIZE_MIN or w < OBJECT_SIZE_MIN:
            return False
        if len(np.unique(grid[tuple(np.array(comp["coords"]).T)])) < 2:
            return False
        if int(comp["area"]) == h * w:
            solid_count += 1
        else:
            holey_count += 1
        if not np.array_equal(patch, _spiral_repack(patch)):
            changed += 1

    if solid_count < 1 or holey_count < 1 or changed < 1:
        return False

    out = main(grid)
    return out.shape[0] <= GRID_MAX and out.shape[1] <= GRID_MAX


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX_EXCLUSIVE))
    dims = [(int(RNG.integers(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX + 1)),
             int(RNG.integers(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX + 1)))
            for _ in range(count)]

    if sum(h for h, _ in dims) + count - 1 > GRID_MAX:
        return None

    modes = ["solid", str(RNG.choice(np.array(["frame", "carved"], dtype=object)))]
    while len(modes) < count:
        modes.append(str(RNG.choice(np.array(["solid", "frame", "carved"], dtype=object))))
    RNG.shuffle(modes)

    patches = [_build_patch(h, w, mode) for (h, w), mode in zip(dims, modes)]

    left_margin = int(RNG.integers(BORDER_MIN, BORDER_MAX + 1))
    right_margin = int(RNG.integers(BORDER_MIN, BORDER_MAX + 1))
    gaps = [int(RNG.integers(INPUT_GAP_MIN, INPUT_GAP_MAX + 1)) for _ in range(count - 1)]

    width = left_margin + right_margin + sum(w for _, w in dims) + sum(gaps)
    if width > GRID_MAX:
        return None

    max_h = max(h for h, _ in dims)
    min_height = max(max_h + 2, 10)
    max_height = min(GRID_MAX, max_h + 12)
    if min_height > max_height:
        return None
    height = int(RNG.integers(min_height, max_height + 1))

    cols: list[int] = []
    cursor = left_margin
    for idx, (_, w) in enumerate(dims):
        cols.append(cursor)
        cursor += w
        if idx < len(gaps):
            cursor += gaps[idx]

    rows = [int(RNG.integers(1, height - h)) for h, _ in dims]
    grid = np.full((height, width), BG, dtype=int)

    for patch, (r0, c0) in zip(patches, zip(rows, cols)):
        h, w = patch.shape
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        mask = patch != BG
        sub[mask] = patch[mask]

    if not _validate_grid(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
