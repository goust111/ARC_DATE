from __future__ import annotations

import random
import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _max_count(line: np.ndarray) -> int:
    counts = np.bincount(line.astype(int), minlength=10)
    return int(counts.max())


def _dominant_color(line: np.ndarray, *, require_unique: bool) -> int:
    counts = np.bincount(line.astype(int), minlength=10)
    dominant = int(np.argmax(counts))
    if require_unique:
        top = int(counts[dominant])
        second = int(np.partition(counts, -2)[-2]) if counts.size >= 2 else 0
        assert top > second, "expected a unique dominant color on each stripe line"
    return dominant


def _orientation_score(grid: np.ndarray, horizontal: bool) -> int:
    if horizontal:
        return sum(_max_count(grid[r, :]) for r in range(grid.shape[0]))
    return sum(_max_count(grid[:, c]) for c in range(grid.shape[1]))


def _extract_noise(
    grid: np.ndarray, horizontal: bool, *, require_unique: bool
) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    if horizontal:
        dominant = np.array(
            [_dominant_color(g[r, :], require_unique=require_unique) for r in range(g.shape[0])],
            dtype=int,
        )[:, None]
    else:
        dominant = np.array(
            [_dominant_color(g[:, c], require_unique=require_unique) for c in range(g.shape[1])],
            dtype=int,
        )[None, :]
    noise_mask = g != dominant
    box = _bbox(noise_mask)
    assert box is not None, "expected at least one noise pixel"
    crop = _crop_bbox(g, box)
    crop_mask = _crop_bbox(noise_mask.astype(int), box).astype(bool)
    out = np.zeros_like(crop)
    out[crop_mask] = crop[crop_mask]
    return out, noise_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the stripe direction by dominant coverage, then crop out only the off-stripe noise pixels.

    Input structure:
    - The whole canvas is filled by horizontal stripes or vertical stripes with no separate background color.
    - Every stripe line has one unique dominant color, while sparse noise pixels break those linewise majorities.
    - Noise colors are non-black and may match the dominant colors of other stripe lines.

    Rule:
    - Sum the dominant-color coverage over all rows and over all columns, and pick the direction with the larger total as the stripe direction.
    - Within that stripe direction, mark every pixel whose color differs from its own stripe line's dominant color as noise.
    - Output the noise pixels inside their bounding box, preserving each noise pixel's original color and relative position and filling every other output cell with black.

    Key insight:
    - The stripes are only the measuring stick for deciding which pixels are wrong; the required output is the removed noise itself, not the cleaned stripe field.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_score = _orientation_score(g, horizontal=True)
    col_score = _orientation_score(g, horizontal=False)
    assert row_score != col_score, "stripe direction must be uniquely determined"

    out, _noise_mask = _extract_noise(g, horizontal=row_score > col_score, require_unique=True)
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _balanced_line_colors(line_count: int) -> list[int]:
    palette_size = random.randint(4, min(7, line_count, len(COLORS)))
    palette = random.sample(COLORS, palette_size)
    colors = [palette[i % palette_size] for i in range(line_count)]
    random.shuffle(colors)
    for _ in range(60):
        bad = [i for i in range(1, line_count) if colors[i] == colors[i - 1]]
        if not bad:
            break
        i = random.choice(bad)
        swap_choices = [
            j
            for j in range(line_count)
            if j != i
            and colors[j] != colors[i - 1]
            and (i + 1 == line_count or colors[j] != colors[i + 1])
            and (j == 0 or colors[i] != colors[j - 1])
            and (j + 1 == line_count or colors[i] != colors[j + 1])
        ]
        if swap_choices:
            j = random.choice(swap_choices)
            colors[i], colors[j] = colors[j], colors[i]
    return colors


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    min_cells = max(4, (area * 2) // 5)
    max_cells = area - 1
    target = random.randint(min_cells, max_cells)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = {pos for pos in _neighbors4(start[0], start[1], height, width)}

    while len(cells) < target and frontier:
        if random.random() < 0.7:
            edge_frontier = [
                pos
                for pos in frontier
                if pos[0] in (0, height - 1) or pos[1] in (0, width - 1)
            ]
            choice_pool = edge_frontier or list(frontier)
        else:
            choice_pool = list(frontier)
        cell = random.choice(choice_pool)
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors4(cell[0], cell[1], height, width):
            if nbr not in cells:
                frontier.add(nbr)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    box = _bbox(mask)
    if box != (0, 0, height - 1, width - 1):
        raise ValueError("shape does not span its chosen bounding box")
    if _component_count(mask) != 1:
        raise ValueError("shape must stay connected")
    if len(set(np.argwhere(mask)[:, 0])) < 2 or len(set(np.argwhere(mask)[:, 1])) < 2:
        raise ValueError("shape must span multiple rows and columns")
    return mask


def _paint_stripes(horizontal: bool, rows: int, cols: int, line_colors: list[int]) -> np.ndarray:
    if horizontal:
        return np.repeat(np.array(line_colors, dtype=int)[:, None], cols, axis=1)
    return np.repeat(np.array(line_colors, dtype=int)[None, :], rows, axis=0)


def _noise_color_choices(
    horizontal: bool,
    line_colors: list[int],
    stripe_palette: tuple[int, ...],
    r: int,
    c: int,
) -> list[int]:
    base = line_colors[r] if horizontal else line_colors[c]
    return [color for color in stripe_palette if color != base]


def _expected_output_from_noise(
    bbox_top: int,
    bbox_left: int,
    bbox_height: int,
    bbox_width: int,
    coords_to_color: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.zeros((bbox_height, bbox_width), dtype=int)
    for (r, c), color in coords_to_color.items():
        out[r - bbox_top, c - bbox_left] = color
    return out


def _validate_attempt(
    grid: np.ndarray,
    expected: np.ndarray,
    horizontal: bool,
    noise_mask: np.ndarray,
    used_noise_colors: set[int],
) -> bool:
    row_score = _orientation_score(grid, horizontal=True)
    col_score = _orientation_score(grid, horizontal=False)
    chosen_score = row_score if horizontal else col_score
    other_score = col_score if horizontal else row_score
    if chosen_score <= other_score:
        return False
    if chosen_score - other_score < max(4, min(grid.shape) // 3):
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    opposite_out, opposite_noise = _extract_noise(grid, horizontal=not horizontal, require_unique=False)
    if np.array_equal(opposite_out, expected) and np.array_equal(opposite_noise, noise_mask):
        return False

    if _component_count(noise_mask) != 1:
        return False

    for color in used_noise_colors:
        inside = int(np.count_nonzero((grid == color) & noise_mask))
        outside = int(np.count_nonzero((grid == color) & ~noise_mask))
        if outside <= inside:
            return False

    stripe_line_colors = {int(v) for v in np.unique(grid)}
    if BLACK in stripe_line_colors or not used_noise_colors.issubset(stripe_line_colors):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.random() < 0.5
    rows = random.randint(10, 24)
    cols = random.randint(10, 24)
    line_count = rows if horizontal else cols
    line_colors = _balanced_line_colors(line_count)
    stripe_palette = tuple(sorted(set(line_colors)))
    grid = _paint_stripes(horizontal, rows, cols, line_colors)

    bbox_height = random.randint(3, min(9, rows))
    bbox_width = random.randint(3, min(9, cols))
    bbox_top = random.randint(0, rows - bbox_height)
    bbox_left = random.randint(0, cols - bbox_width)

    try:
        shape_mask = _random_connected_mask(bbox_height, bbox_width)
    except ValueError:
        return None

    coords_to_color: dict[tuple[int, int], int] = {}
    used_noise_colors: set[int] = set()
    for local_r, local_c in np.argwhere(shape_mask):
        r = bbox_top + int(local_r)
        c = bbox_left + int(local_c)
        choices = _noise_color_choices(horizontal, line_colors, stripe_palette, r, c)
        if not choices:
            return None
        if random.random() < 0.7:
            preferred = [color for color in choices if color != grid[r, c]]
            choices = preferred or choices
        color = random.choice(choices)
        grid[r, c] = color
        coords_to_color[(r, c)] = color
        used_noise_colors.add(color)

    if len(used_noise_colors) < 2:
        return None

    noise_mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords_to_color:
        noise_mask[r, c] = True

    expected = _expected_output_from_noise(
        bbox_top=bbox_top,
        bbox_left=bbox_left,
        bbox_height=bbox_height,
        bbox_width=bbox_width,
        coords_to_color=coords_to_color,
    )

    if not _validate_attempt(grid, expected, horizontal, noise_mask, used_noise_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid stripe-noise ARC sample")
