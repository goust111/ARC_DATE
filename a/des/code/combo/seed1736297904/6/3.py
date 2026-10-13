import numpy as np


BG = 0
GROUND = 5
SHADOW = 9
OBJECT_PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_RNG = np.random.default_rng()

MIN_OBJECTS = 2
MAX_OBJECTS = 5
MIN_WIDTH = 1
MAX_WIDTH = 4
MIN_BOTTOM_SPACE = 3
MAX_BOTTOM_SPACE = 7
MAX_HEIGHT = 6
MAX_GRID_WIDTH = 30
MAX_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_ground_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GROUND, axis=1))
    assert rows.size == 1, "expected a single full-width ground line"
    return int(rows[0])


def _extract_objects(grid: np.ndarray, ground_row: int) -> list[dict[str, int]]:
    above = grid[:ground_row]
    mask = (above != BG) & (above != GROUND) & (above != SHADOW)
    comps = _components(mask)
    assert comps, "expected at least one object above the ground"

    objects: list[dict[str, int]] = []
    for comp in comps:
        r0 = int(comp[:, 0].min())
        r1 = int(comp[:, 0].max())
        c0 = int(comp[:, 1].min())
        c1 = int(comp[:, 1].max())
        color_values = np.unique(above[comp[:, 0], comp[:, 1]])
        assert color_values.size == 1, "each object must be monochrome"
        color = int(color_values[0])
        patch = above[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "each object must be a solid rectangle"
        assert r1 == ground_row - 1, "each object must rest on the ground line"
        objects.append(
            {
                "r0": r0,
                "c0": c0,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "color": color,
            }
        )

    objects.sort(key=lambda obj: (obj["c0"], obj["r0"]))
    return objects


def _shadow_mask(
    shape: tuple[int, int], ground_row: int, obj: dict[str, int], direction: int
) -> np.ndarray | None:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for dr in range(obj["h"]):
        rr = ground_row + 1 + dr
        cc0 = obj["c0"] + direction * dr
        cc1 = cc0 + obj["w"]
        if rr >= h or cc0 < 0 or cc1 > w:
            return None
        mask[rr, cc0:cc1] = True
    return mask


def _paint_shadow(grid: np.ndarray, ground_row: int, obj: dict[str, int], direction: int) -> None:
    for dr in range(obj["h"]):
        rr = ground_row + 1 + dr
        cc0 = obj["c0"] + direction * dr
        grid[rr, cc0 : cc0 + obj["w"]] = SHADOW


def _direction_match(
    grid: np.ndarray, ground_row: int, objects: list[dict[str, int]], direction: int
) -> tuple[int, bool]:
    maroon = grid == SHADOW
    union = np.zeros_like(maroon, dtype=bool)
    matched = 0
    for obj in objects:
        candidate = _shadow_mask(grid.shape, ground_row, obj, direction)
        if candidate is None:
            return -1, False
        if np.all(maroon[candidate]):
            union |= candidate
            matched += 1
    return matched, np.array_equal(union, maroon)


def _infer_direction(grid: np.ndarray, ground_row: int, objects: list[dict[str, int]]) -> int:
    maroon = grid == SHADOW
    assert np.any(maroon[ground_row + 1 :]), "expected at least one visible shadow"
    assert not np.any(maroon[: ground_row + 1]), "shadows must stay below the ground line"

    exact: list[tuple[int, int]] = []
    counts: list[tuple[int, int]] = []
    for direction in (-1, 1):
        matched, is_exact = _direction_match(grid, ground_row, objects, direction)
        if matched >= 0:
            counts.append((direction, matched))
            if is_exact and matched > 0:
                exact.append((direction, matched))

    if len(exact) == 1:
        return exact[0][0]
    if len(exact) == 2:
        exact.sort(key=lambda item: item[1], reverse=True)
        assert exact[0][1] > exact[1][1], "ambiguous shadow direction"
        return exact[0][0]

    counts.sort(key=lambda item: item[1], reverse=True)
    assert counts and counts[0][1] > 0, "could not match any visible shadow"
    if len(counts) > 1:
        assert counts[0][1] > counts[1][1], "ambiguous shadow direction"
    return counts[0][0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    ground_row = _find_ground_row(g)
    objects = _extract_objects(g, ground_row)
    direction = _infer_direction(g, ground_row, objects)

    out = g.copy()
    shadow_zone = out[ground_row + 1 :]
    shadow_zone[shadow_zone == SHADOW] = BG
    for obj in objects:
        _paint_shadow(out, ground_row, obj, direction)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete all sheared shadows below the grey ground line.

    Input structure
    - Black background.
    - One full-width grey row in the lower half is the ground line.
    - Above it sit several bottom-aligned solid colored rectangles.
    - Below it, some rectangles already have maroon shadows whose rows shift
      one column per step in a shared horizontal direction.

    Rule
    - Infer the common shadow direction from the visible maroon silhouettes.
    - Repaint the shadow of every object using the same one-pixel-per-row shear.
    - Keep the objects and ground line unchanged.
    """
    return _solve(grid)


def _layout_columns(widths: np.ndarray, heights: np.ndarray, direction: int) -> tuple[np.ndarray, int] | None:
    n = len(widths)
    allow_tight = bool(_RNG.integers(0, 2))
    if n == 1:
        gaps = np.zeros(0, dtype=int)
    elif allow_tight:
        gaps = _RNG.integers(1, 3, size=n - 1)
    else:
        gaps = _RNG.integers(2, 5, size=n - 1)

    base = np.zeros(n, dtype=int)
    for i in range(1, n):
        base[i] = base[i - 1] + widths[i - 1] + gaps[i - 1]

    if direction == -1:
        required_shift = int(max(0, np.max((heights - 1) - base)))
    else:
        required_shift = 0

    extra_left = int(_RNG.integers(0, 4))
    shift = required_shift + extra_left
    c0s = base + shift

    rightmost = 0
    for c0, w, h in zip(c0s, widths, heights):
        extent = c0 + w - 1 + (h - 1 if direction == 1 else 0)
        rightmost = max(rightmost, int(extent))

    extra_right = int(_RNG.integers(0, 4))
    width = rightmost + 1 + extra_right
    if width > MAX_GRID_WIDTH:
        return None
    return c0s, width


def _sample_objects() -> tuple[list[dict[str, int]], int, tuple[int, int], int] | None:
    n_objects = int(_RNG.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    direction = int(_RNG.choice((-1, 1)))

    bottom_space = int(_RNG.integers(MIN_BOTTOM_SPACE, MAX_BOTTOM_SPACE + 1))
    top_space = int(_RNG.integers(bottom_space + 1, bottom_space + 6))
    max_height = min(MAX_HEIGHT, top_space, bottom_space)
    if max_height < 2:
        return None

    heights = _RNG.integers(1, max_height + 1, size=n_objects)
    if int(heights.max()) < 2:
        heights[int(_RNG.integers(0, n_objects))] = 2
    widths = _RNG.integers(MIN_WIDTH, MAX_WIDTH + 1, size=n_objects)

    placed = _layout_columns(widths, heights, direction)
    if placed is None:
        return None
    c0s, width = placed

    ground_row = top_space
    grid_shape = (top_space + 1 + bottom_space, width)
    colors = _RNG.choice(np.array(OBJECT_PALETTE, dtype=int), size=n_objects, replace=False)

    objects = []
    for c0, w, h, color in zip(c0s, widths, heights, colors):
        objects.append(
            {
                "r0": ground_row - int(h),
                "c0": int(c0),
                "h": int(h),
                "w": int(w),
                "color": int(color),
            }
        )
    return objects, direction, grid_shape, ground_row


def _build_base_grid(shape: tuple[int, int], ground_row: int, objects: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[ground_row, :] = GROUND
    for obj in objects:
        r0 = obj["r0"]
        r1 = ground_row
        c0 = obj["c0"]
        c1 = c0 + obj["w"]
        grid[r0:r1, c0:c1] = obj["color"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_objects()
    if sampled is None:
        return None
    objects, direction, shape, ground_row = sampled

    base = _build_base_grid(shape, ground_row, objects)

    n_objects = len(objects)
    visible_count = int(_RNG.integers(1, n_objects))
    visible = set(map(int, _RNG.choice(n_objects, size=visible_count, replace=False)))

    tall = [i for i, obj in enumerate(objects) if obj["h"] >= 2]
    if tall and not any(i in visible for i in tall):
        missing_tall = int(_RNG.choice(tall))
        visible.remove(next(iter(visible)))
        visible.add(missing_tall)

    inp = base.copy()
    out = base.copy()
    for i, obj in enumerate(objects):
        _paint_shadow(out, ground_row, obj, direction)
        if i in visible:
            _paint_shadow(inp, ground_row, obj, direction)

    try:
        if _infer_direction(inp, ground_row, objects) != direction:
            return None
        if not np.array_equal(_solve(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-completion sample")
