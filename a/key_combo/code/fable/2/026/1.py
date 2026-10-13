import numpy as np


BLACK = 0
BLUE = 1
RED = 2
SOURCE_POOL = (3, 4, 5, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _swap_lookup(color_a: int, color_b: int) -> np.ndarray:
    lut = np.arange(10, dtype=int)
    lut[int(color_a)] = int(color_b)
    lut[int(color_b)] = int(color_a)
    return lut


def _find_mirror_lines(grid: np.ndarray) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    red_rows = [r for r in range(h) if np.all(g[r, :] == RED)]
    blue_rows = [r for r in range(h) if np.all(g[r, :] == BLUE)]
    red_cols = [c for c in range(w) if np.all(g[:, c] == RED)]
    blue_cols = [c for c in range(w) if np.all(g[:, c] == BLUE)]

    if len(red_rows) == 1 and len(blue_rows) == 1 and not red_cols and not blue_cols:
        return "h", int(red_rows[0]), int(blue_rows[0])
    if len(red_cols) == 1 and len(blue_cols) == 1 and not red_rows and not blue_rows:
        return "v", int(red_cols[0]), int(blue_cols[0])
    raise AssertionError("expected exactly one full red line and one full blue line with a shared orientation")


def _corridor_slice(grid: np.ndarray, orient: str, red_pos: int, blue_pos: int) -> np.ndarray:
    lo, hi = sorted((int(red_pos), int(blue_pos)))
    if orient == "v":
        return grid[:, lo + 1 : hi].copy()
    return grid[lo + 1 : hi, :].copy()


def _source_colors(grid: np.ndarray, orient: str, red_pos: int, blue_pos: int) -> tuple[int, int]:
    corridor = _corridor_slice(grid, orient, red_pos, blue_pos)
    colors = sorted(set(map(int, corridor.ravel())) - {BLACK})
    assert len(colors) == 2, "expected exactly two source colors between the mirror lines"
    return int(colors[0]), int(colors[1])


def _axis_projection(
    axis_len: int,
    red_pos: int,
    blue_pos: int,
    *,
    mirror: bool,
    swap_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gap = abs(int(blue_pos) - int(red_pos))
    assert gap >= 2, "mirror lines must have at least one interior cell between them"
    step = 1 if int(blue_pos) > int(red_pos) else -1

    src_index = np.full(axis_len, -1, dtype=int)
    line_mask = np.zeros(axis_len, dtype=bool)
    line_colors = np.zeros(axis_len, dtype=int)
    swap_mask = np.zeros(axis_len, dtype=bool)

    for axis_coord in range(axis_len):
        y = step * (int(axis_coord) - int(red_pos))
        strip_index, rem = divmod(int(y), int(gap))

        if rem == 0:
            line_mask[axis_coord] = True
            line_colors[axis_coord] = RED if (strip_index % 2 == 0) else BLUE
            continue

        if mirror and (strip_index % 2 != 0):
            y_src = gap - rem
        else:
            y_src = rem
        src_index[axis_coord] = int(red_pos) + step * int(y_src)

        if swap_style == "red_parity":
            swap_mask[axis_coord] = ((strip_index // 2) % 2) == 1
        elif swap_style == "alternate_strip":
            swap_mask[axis_coord] = (strip_index % 2) != 0
        elif swap_style == "none":
            swap_mask[axis_coord] = False
        else:
            raise ValueError(f"unknown swap_style: {swap_style}")

    return src_index, line_mask, line_colors, swap_mask


def _apply_tiling(
    grid: np.ndarray,
    orient: str,
    red_pos: int,
    blue_pos: int,
    color_a: int,
    color_b: int,
    *,
    mirror: bool = True,
    swap_style: str = "red_parity",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    swap_lut = _swap_lookup(color_a, color_b)

    axis_len = g.shape[1] if orient == "v" else g.shape[0]
    src_index, line_mask, line_colors, swap_mask = _axis_projection(
        axis_len, red_pos, blue_pos, mirror=mirror, swap_style=swap_style
    )

    out = np.empty_like(g)
    if orient == "v":
        for c in range(axis_len):
            if line_mask[c]:
                out[:, c] = int(line_colors[c])
            else:
                col = g[:, int(src_index[c])]
                out[:, c] = swap_lut[col] if swap_mask[c] else col
    else:
        for r in range(axis_len):
            if line_mask[r]:
                out[r, :] = int(line_colors[r])
            else:
                row = g[int(src_index[r]), :]
                out[r, :] = swap_lut[row] if swap_mask[r] else row
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the two-color corridor by repeated mirror reflections, swapping the two source colors only after red-mirror reflections.

    Input structure:
    - The grid has a black background, one full red mirror line, and one full blue mirror line.
    - The two mirror lines are parallel and enclose a corridor that contains a two-color source pattern.
    - Outside the two mirror lines, the input remains black.

    Rule:
    - Treat the strip between the red and blue lines as the fundamental corridor and reflect it repeatedly to both sides.
    - Adjacent strips alternate between preserved and mirrored orientation, and the periodic images of the mirror lines are also drawn.
    - For each reflected strip, swap the two source colors iff the reflection history crosses a red mirror an odd number of times; mirror-line colors never swap.

    Key insight:
    - Repeated reflections across two distinct mirrors create a strip wallpaper whose geometry is controlled by mirror parity, while the color state is controlled separately by the parity of red-mirror crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    orient, red_pos, blue_pos = _find_mirror_lines(g)
    color_a, color_b = _source_colors(g, orient, red_pos, blue_pos)
    return _apply_tiling(g, orient, red_pos, blue_pos, color_a, color_b)


def _grow_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    r = np.random.randint(height)
    c = np.random.randint(width)
    mask[r, c] = True
    frontier = {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    while int(mask.sum()) < int(target_area):
        frontier = {
            (rr, cc)
            for rr, cc in frontier
            if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]
        }
        if not frontier:
            break
        idx = np.random.randint(len(frontier))
        rr, cc = list(frontier)[idx]
        mask[rr, cc] = True
        frontier.update(((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)))
    return mask


def _random_corridor(height: int, inner_width: int, color_a: int, color_b: int) -> np.ndarray | None:
    swap_lut = _swap_lookup(color_a, color_b)
    for _ in range(200):
        corridor = np.zeros((height, inner_width), dtype=int)

        sprite_h = np.random.randint(4, min(height, 10) + 1)
        sprite_w = np.random.randint(2, inner_width + 1)
        top = np.random.randint(0, height - sprite_h + 1)
        left = np.random.randint(0, inner_width - sprite_w + 1)

        area_min = max(6, (sprite_h * sprite_w) // 3)
        area_max = max(area_min, sprite_h * sprite_w - 1)
        target_area = np.random.randint(area_min, area_max + 1)
        mask = _grow_mask(sprite_h, sprite_w, target_area)
        if int(mask.sum()) < 6 or bool(mask.all()):
            continue

        coords = np.argwhere(mask)
        if len(np.unique(coords[:, 0])) < 2 or len(np.unique(coords[:, 1])) < 2:
            continue

        seed_ids = np.random.choice(len(coords), size=2, replace=False)
        seed_a = coords[int(seed_ids[0])]
        seed_b = coords[int(seed_ids[1])]
        local = np.zeros((sprite_h, sprite_w), dtype=int)
        for rr, cc in coords:
            da = abs(int(rr) - int(seed_a[0])) + abs(int(cc) - int(seed_a[1]))
            db = abs(int(rr) - int(seed_b[0])) + abs(int(cc) - int(seed_b[1]))
            if da == db:
                local[int(rr), int(cc)] = int(np.random.choice((color_a, color_b)))
            else:
                local[int(rr), int(cc)] = int(color_a if da < db else color_b)

        flip_mask = (local != 0) & (np.random.rand(sprite_h, sprite_w) < 0.18)
        local[flip_mask] = swap_lut[local[flip_mask]]

        if color_a not in local or color_b not in local:
            continue

        corridor[top : top + sprite_h, left : left + sprite_w] = local

        bbox = _bbox(corridor != 0)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        if r1 == r0 or c1 == c0:
            continue
        if np.array_equal(corridor, np.fliplr(corridor)):
            continue
        if np.array_equal(corridor, swap_lut[corridor]):
            continue
        if np.array_equal(corridor, swap_lut[np.fliplr(corridor)]):
            continue
        return corridor
    return None


def _build_input(height: int, left_strips: int, right_strips: int, gap: int, corridor: np.ndarray) -> np.ndarray:
    width = (int(left_strips) + int(right_strips) + 1) * int(gap) + 1
    red_col = int(left_strips) * int(gap)
    blue_col = red_col + int(gap)

    grid = np.zeros((height, width), dtype=int)
    grid[:, red_col] = RED
    grid[:, blue_col] = BLUE
    grid[:, red_col + 1 : blue_col] = corridor
    return grid


def _supports_semantics(grid: np.ndarray) -> bool:
    try:
        orient, red_pos, blue_pos = _find_mirror_lines(grid)
        color_a, color_b = _source_colors(grid, orient, red_pos, blue_pos)
    except AssertionError:
        return False

    actual = _apply_tiling(grid, orient, red_pos, blue_pos, color_a, color_b, mirror=True, swap_style="red_parity")
    no_swap = _apply_tiling(grid, orient, red_pos, blue_pos, color_a, color_b, mirror=True, swap_style="none")
    no_mirror = _apply_tiling(grid, orient, red_pos, blue_pos, color_a, color_b, mirror=False, swap_style="red_parity")
    alt_swap = _apply_tiling(grid, orient, red_pos, blue_pos, color_a, color_b, mirror=True, swap_style="alternate_strip")

    if np.array_equal(actual, no_swap):
        return False
    if np.array_equal(actual, no_mirror):
        return False
    if np.array_equal(actual, alt_swap):
        return False

    axis_len = grid.shape[1] if orient == "v" else grid.shape[0]
    gap = abs(int(blue_pos) - int(red_pos))
    if min(int(red_pos), int(blue_pos)) // gap < 2:
        return False
    if (axis_len - 1 - max(int(red_pos), int(blue_pos))) // gap < 2:
        return False

    lo = min(int(red_pos), int(blue_pos))
    hi = max(int(red_pos), int(blue_pos))
    if orient == "v":
        outer_mask = np.zeros_like(actual, dtype=bool)
        outer_mask[:, : lo - gap] = True
        outer_mask[:, hi + gap + 1 :] = True
    else:
        outer_mask = np.zeros_like(actual, dtype=bool)
        outer_mask[: lo - gap, :] = True
        outer_mask[hi + gap + 1 :, :] = True
    if not np.any(actual[outer_mask] != BLACK):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    left_strips = int(np.random.randint(2, 4))
    right_strips = int(np.random.randint(2, 4))
    total_blocks = left_strips + right_strips + 1
    max_gap = (30 - 1) // total_blocks
    if max_gap < 4:
        return None
    gap = int(np.random.randint(4, max_gap + 1))

    width = total_blocks * gap + 1
    height = int(np.random.randint(11, 31))
    if width > 30 or height > 30:
        return None

    color_a, color_b = np.random.choice(SOURCE_POOL, size=2, replace=False)
    corridor = _random_corridor(height, gap - 1, int(color_a), int(color_b))
    if corridor is None:
        return None

    grid = _build_input(height, left_strips, right_strips, gap, corridor)

    if np.random.rand() < 0.5:
        grid = np.fliplr(grid)
    if np.random.rand() < 0.5:
        grid = grid.T
    if np.random.rand() < 0.5:
        grid = np.flipud(grid)

    grid = np.asarray(grid, dtype=int)
    if not _supports_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-mirror corridor input")
