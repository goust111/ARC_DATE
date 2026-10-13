from __future__ import annotations

import numpy as np


MARKER = 5
GRID_SIZES = (3, 4, 5)
FOREGROUND_COLORS = tuple(color for color in range(1, 10) if color != MARKER)
GEN_RETRY_LIMIT = 600


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape[0] == 1, "expected exactly one grey marker"
    return tuple(map(int, coords[0]))


def _suppressed_color(grid: np.ndarray) -> int:
    marker_r, marker_c = _find_marker(grid)
    counts: dict[int, int] = {}
    for nr, nc in _neighbors4(grid.shape, marker_r, marker_c):
        color = int(grid[nr, nc])
        assert color != MARKER, "marker cannot touch another marker"
        counts[color] = counts.get(color, 0) + 1

    assert len(counts) >= 2, "marker neighborhood must contain at least two colors"
    ordered = sorted(counts.items(), key=lambda item: item[1])
    assert ordered[0][1] < ordered[1][1], "suppressed color must be the unique least frequent neighbor color"
    return int(ordered[0][0])


def _mask_block(grid: np.ndarray, source_color: int, paint_color: int) -> np.ndarray:
    return np.where(grid == int(source_color), int(paint_color), 0).astype(int)


def _transform_with_suppressed(grid: np.ndarray, suppressed: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros((h * h, w * w), dtype=int)

    suppressed_block = _mask_block(g, suppressed, suppressed)
    marker_block = _mask_block(g, suppressed, MARKER)
    blocks = {suppressed: suppressed_block, MARKER: marker_block}

    for color in np.unique(g):
        color = int(color)
        if color not in blocks:
            blocks[color] = _mask_block(g, color, color)

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = blocks[color]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the grid into self-color mask blocks, while the grey marker adopts the suppressed color mask.

    Input structure:
    - The input is a small 2D grid with no black cells and exactly one grey marker pixel.
    - All non-grey pixels use foreground colors, and the marker's 4-neighborhood contains at least two colors.
    - In that 4-neighborhood, exactly one color is the unique least frequent color.

    Rule:
    - Read the grey marker's 4-neighbors and identify the unique least frequent color S.
    - Replace every input pixel by a block the same shape as the input grid, forming an output of shape (H*H) by (W*W).
    - For a non-grey pixel of color c, paint color c only at input positions whose color is also c; all other positions are black.
    - For the grey marker pixel, paint grey only at input positions whose color is S; all other positions are black.

    Key insight:
    - The marker is both a selector and a special block: its neighborhood chooses the suppressed color mask, and the marker block reuses that chosen mask in grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all(g != 0), "input must not contain black"
    suppressed = _suppressed_color(g)
    return _transform_with_suppressed(g, suppressed)


def _choose_marker_position(h: int, w: int) -> tuple[int, int]:
    edge_positions = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1))
        and not (r in (0, h - 1) and c in (0, w - 1))
    ]
    interior_positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]

    use_edge = bool(edge_positions) and (not interior_positions or bool(np.random.randint(0, 2)))
    pool = edge_positions if use_edge else interior_positions
    idx = int(np.random.randint(0, len(pool)))
    return tuple(map(int, pool[idx]))


def _sample_color_counts(
    total_non_marker: int,
    degree: int,
    suppressed: int,
    heavy: int,
    tertiary: int,
    counterexample_branch: bool,
) -> dict[int, int]:
    counts = {suppressed: 2, heavy: degree - 1, tertiary: 2}
    remaining = total_non_marker - sum(counts.values())
    assert remaining >= 0

    if counterexample_branch:
        counts[suppressed] += 1
        remaining -= 1

    if remaining > 0:
        if counterexample_branch:
            probs = np.random.dirichlet(np.ones(3))
            extra = np.random.multinomial(remaining, probs)
            counts[suppressed] += int(extra[0])
            counts[heavy] += int(extra[1])
            counts[tertiary] += int(extra[2])
        else:
            probs = np.random.dirichlet(np.ones(2))
            extra = np.random.multinomial(remaining, probs)
            counts[heavy] += int(extra[0])
            counts[tertiary] += int(extra[1])

    return counts


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.choice(GRID_SIZES))
    w = int(np.random.choice(GRID_SIZES))
    colors = list(map(int, np.random.choice(FOREGROUND_COLORS, size=3, replace=False)))

    marker = _choose_marker_position(h, w)
    neighbors = _neighbors4((h, w), marker[0], marker[1])
    degree = len(neighbors)

    suppressed = int(colors[int(np.random.randint(0, len(colors)))])
    other_colors = [color for color in colors if color != suppressed]
    heavy = int(other_colors[int(np.random.randint(0, len(other_colors)))])
    tertiary = int([color for color in other_colors if color != heavy][0])

    counterexample_branch = bool(np.random.randint(0, 100) < 65)
    counts = _sample_color_counts(h * w - 1, degree, suppressed, heavy, tertiary, counterexample_branch)

    max_allowed = int(np.ceil((h * w - 1) * 0.65))
    if max(counts.values()) > max_allowed:
        return None

    grid = np.full((h, w), tertiary, dtype=int)
    grid[marker] = MARKER

    shuffled_neighbors = list(neighbors)
    np.random.shuffle(shuffled_neighbors)
    grid[shuffled_neighbors[0]] = suppressed
    for cell in shuffled_neighbors[1:]:
        grid[cell] = heavy

    unlocked = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) != marker and (r, c) not in neighbors
    ]
    pool = (
        [suppressed] * (counts[suppressed] - 1)
        + [heavy] * (counts[heavy] - (degree - 1))
        + [tertiary] * (counts[tertiary] - 0)
    )
    assert len(pool) == len(unlocked)
    np.random.shuffle(pool)
    for (r, c), color in zip(unlocked, pool):
        grid[r, c] = int(color)

    return grid


def _validate_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) < 3 or max(g.shape) > 5:
        return False
    if np.any(g == 0):
        return False
    if np.count_nonzero(g == MARKER) != 1:
        return False

    colors = [int(color) for color in np.unique(g) if int(color) != MARKER]
    if len(colors) != 3:
        return False

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    if min(counts.values()) < 2:
        return False
    if max(counts.values()) > int(np.ceil((g.size - 1) * 0.65)):
        return False

    try:
        marker = _find_marker(g)
        suppressed = _suppressed_color(g)
        out = main(g)
    except Exception:
        return False

    if out.shape != (g.shape[0] * g.shape[0], g.shape[1] * g.shape[1]):
        return False
    if max(out.shape) > 30:
        return False

    h, w = g.shape
    marker_block = out[marker[0] * h : (marker[0] + 1) * h, marker[1] * w : (marker[1] + 1) * w]
    if not np.array_equal(marker_block, _mask_block(g, suppressed, MARKER)):
        return False

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            if color == MARKER:
                expected = _mask_block(g, suppressed, MARKER)
            elif color == suppressed:
                expected = _mask_block(g, suppressed, suppressed)
            else:
                expected = _mask_block(g, color, color)
            if not np.array_equal(block, expected):
                return False

    for alt in colors:
        if alt == suppressed:
            continue
        if np.array_equal(_transform_with_suppressed(g, alt), out):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is None:
            continue
        if _validate_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
