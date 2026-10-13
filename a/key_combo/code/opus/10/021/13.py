import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
GEN_MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray) -> int:
    h, w = grid.shape
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(sep_cols) == 1, f"expected exactly one full grey separator column, got {sep_cols}"
    sep = sep_cols[0]
    assert 0 < sep < w - 1, "separator cannot lie on the boundary"
    assert np.count_nonzero(grid == GREY) == h, "grey pixels must belong only to the separator"
    assert sep == w - sep - 1, "separator must split the grid into equal-width halves"
    return sep


def _swap_colors(sprite: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = np.zeros_like(sprite)
    out[sprite == color_a] = color_b
    out[sprite == color_b] = color_a
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    sep = _find_separator(g)
    half_w = sep
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :].copy()
    assert left.shape == (h, half_w) and right.shape == (h, half_w)
    assert np.count_nonzero(left == GREY) == 0 and np.count_nonzero(right == GREY) == 0

    left_has = bool(np.any(left != BLACK))
    right_has = bool(np.any(right != BLACK))
    assert left_has ^ right_has, "exactly one side must contain the source pattern"

    if left_has:
        source_side = "left"
        source_half = left
        empty_half = right
    else:
        source_side = "right"
        source_half = right
        empty_half = left

    assert np.count_nonzero(empty_half) == 0, "the non-source side must be pure background"
    source_mask = source_half != BLACK
    comps = _components(source_mask)
    assert len(comps) == 1, f"expected one 4-connected source object, got {len(comps)}"

    colors = sorted(int(v) for v in np.unique(source_half) if v != BLACK)
    assert len(colors) == 2, f"expected exactly two source colors, got {colors}"
    count_by_color = {color: int(np.count_nonzero(source_half == color)) for color in colors}
    major_color, minor_color = sorted(colors, key=lambda color: (-count_by_color[color], color))
    major_count = count_by_color[major_color]
    minor_count = count_by_color[minor_color]
    assert major_count > minor_count, f"source colors must have a strict majority: {count_by_color}"

    if source_side == "left":
        assert not np.any(source_mask[:, -1]), "source pattern must stay at least one column away from the separator"
    else:
        assert not np.any(source_mask[:, 0]), "source pattern must stay at least one column away from the separator"

    reflected = np.fliplr(source_half)
    mirror_half = _swap_colors(reflected, major_color, minor_color)

    source_score = int(np.count_nonzero(source_half == major_color))
    mirror_score = int(np.count_nonzero(mirror_half == major_color))
    assert source_score == major_count and mirror_score == minor_count
    assert source_score > mirror_score, "the mirrored side must lose after the color swap"

    winner_side = source_side if source_score > mirror_score else ("right" if source_side == "left" else "left")
    winner_half = source_half if winner_side == source_side else mirror_half
    return {
        "separator": sep,
        "source_side": source_side,
        "source_half": source_half,
        "mirror_half": mirror_half,
        "winner_side": winner_side,
        "winner_half": winner_half,
        "major_color": major_color,
        "minor_color": minor_color,
        "major_count": major_count,
        "minor_count": minor_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror a two-color source pattern across a central separator, swap the mirrored colors, and keep the side with the larger source-main-color area.

    Input structure:
    - The grid has black background and one full grey vertical separator column at the center.
    - Exactly one side of the separator contains a single 4-connected source pattern made of exactly two non-black colors.
    - One of the two source colors occurs strictly more often, and the source pattern stays at least one column away from the separator.

    Rule:
    - Identify the source side and its two colors, treating the more frequent source color as the main color.
    - Reflect the source pattern across the separator onto the empty side and swap the two source colors in that mirrored copy.
    - Compare how many source-main-color pixels each candidate side contains after this construction, erase the side with fewer, and remove the separator column.
    - Return only the surviving half-grid crop.

    Key insight:
    - Reflection alone preserves the shape, but swapping the mirrored colors reduces the mirrored side's source-main-color area to the original minority count, so the original source side wins.
    """
    return _analyze(grid)["winner_half"].copy()


def _random_connected_shape() -> np.ndarray:
    for _ in range(200):
        area = random.randint(12, 20)
        canvas_h = random.randint(5, 7)
        canvas_w = random.randint(5, 7)
        cells = {(random.randrange(canvas_h), random.randrange(canvas_w))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if not (5 <= h <= 7 and 5 <= w <= 7):
            continue
        if area == h * w:
            continue

        shape = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            shape[r - r0, c - c0] = True
        if len(_components(shape)) != 1:
            continue
        return shape
    raise RuntimeError("failed to build a connected source shape")


def _random_connected_subset(shape: np.ndarray, size: int) -> set[tuple[int, int]]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
    cell_set = set(cells)
    subset = {random.choice(cells)}
    while len(subset) < size:
        frontier = set()
        for r, c in subset:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cell_set and (nr, nc) not in subset:
                    frontier.add((nr, nc))
        if not frontier:
            break
        subset.add(random.choice(tuple(frontier)))
    if len(subset) != size:
        raise RuntimeError("failed to grow a connected color subset")
    return subset


def _make_sprite() -> np.ndarray:
    shape = _random_connected_shape()
    area = int(np.count_nonzero(shape))
    minor_count = random.randint(3, (area - 1) // 2)
    major_count = area - minor_count
    assert major_count > minor_count

    major_color, minor_color = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    minor_cells = _random_connected_subset(shape, minor_count)

    sprite = np.zeros(shape.shape, dtype=int)
    for r, c in np.argwhere(shape):
        sprite[r, c] = major_color
    for r, c in minor_cells:
        sprite[r, c] = minor_color
    return sprite


def _build_candidate() -> np.ndarray:
    sprite = _make_sprite()
    sh, sw = sprite.shape

    h = random.randint(max(9, sh), 12)
    half_w = random.randint(max(8, sw + 1), 11)
    side = random.choice(["left", "right"])

    top = random.randint(0, h - sh)
    if side == "left":
        left_col = random.randint(0, half_w - sw - 1)
    else:
        left_col = random.randint(1, half_w - sw)

    source_half = np.zeros((h, half_w), dtype=int)
    source_half[top : top + sh, left_col : left_col + sw] = sprite

    grid = np.zeros((h, 2 * half_w + 1), dtype=int)
    grid[:, half_w] = GREY
    if side == "left":
        grid[:, :half_w] = source_half
    else:
        grid[:, half_w + 1 :] = source_half
    return grid


def _validate_example(grid: np.ndarray) -> None:
    info = _analyze(grid)
    out = main(grid)
    assert out.shape == info["winner_half"].shape
    assert np.array_equal(out, info["winner_half"])

    source_half = info["source_half"]
    mirror_half = info["mirror_half"]
    assert np.count_nonzero(source_half) >= 12
    assert len(np.unique(source_half[source_half != BLACK])) == 2
    assert np.count_nonzero(mirror_half == info["major_color"]) == info["minor_count"]
    assert np.count_nonzero(source_half == info["major_color"]) == info["major_count"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate()
        try:
            _validate_example(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 13")
