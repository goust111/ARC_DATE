from __future__ import annotations

import random

import numpy as np


SEPARATOR_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
COLOR_POOL = tuple(color for color in range(10) if color != SEPARATOR_COLOR)

BASE_SHAPES = (
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
)


def _normalize(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _rotate(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return _normalize({(int(c), int(-r)) for r, c in cells})


def _reflect(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return _normalize({(int(r), int(-c)) for r, c in cells})


def _shape_size(cells: frozenset[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _random_shape(min_area: int, max_area: int) -> frozenset[tuple[int, int]]:
    pool = [shape for shape in BASE_SHAPES if min_area <= len(shape) <= max_area]
    shape = random.choice(pool)
    if random.random() < 0.5:
        shape = _reflect(shape)
    for _ in range(random.randint(0, 3)):
        shape = _rotate(shape)
    return _normalize(shape)


def _shift_cells(
    cells: set[tuple[int, int]],
    dr: int,
    dc: int,
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    shifted: set[tuple[int, int]] = set()
    for r, c in cells:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < height and 0 <= cc < width:
            shifted.add((rr, cc))
    return shifted


def _find_vector(
    src: set[tuple[int, int]],
    dst: set[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for dr in range(-(height - 1), height):
        for dc in range(-(width - 1), width):
            if _shift_cells(src, dr, dc, height, width) == dst:
                matches.append((int(dr), int(dc)))
    assert len(matches) == 1, f"expected a unique translation, got {matches}"
    return matches[0]


def _candidate_layouts(grid: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    layouts: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        row_hits = np.where(np.all(grid == int(color), axis=1))[0]
        col_hits = np.where(np.all(grid == int(color), axis=0))[0]
        for sep_r in row_hits:
            for sep_c in col_hits:
                panel_h = int(sep_r)
                panel_w = int(sep_c)
                if panel_h <= 0 or panel_w <= 0:
                    continue
                if height - sep_r - 1 != panel_h or width - sep_c - 1 != panel_w:
                    continue
                if not np.all(grid[sep_r + 1 :, sep_c + 1 :] == int(color)):
                    continue
                layouts.append((int(color), int(sep_r), int(sep_c)))
    return sorted(set(layouts))


def _split_panels(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    layouts = _candidate_layouts(grid)
    assert len(layouts) == 1, f"expected one separator layout, got {layouts}"
    sep_color, sep_r, sep_c = layouts[0]
    train_in = grid[:sep_r, :sep_c].copy()
    train_out = grid[:sep_r, sep_c + 1 :].copy()
    test_in = grid[sep_r + 1 :, :sep_c].copy()
    assert train_in.shape == train_out.shape == test_in.shape
    return sep_color, train_in, train_out, test_in


def _panel_background(*panels: np.ndarray) -> int:
    merged = np.concatenate([panel.ravel() for panel in panels])
    values, counts = np.unique(merged, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, f"background must be unique, got {winners.tolist()}"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a per-color translation from the training pair, then apply it to the test panel.

    Input structure:
    - A separator color forms a 2x2 analogy layout.
    - Top-left is the training input, top-right is the training output.
    - Bottom-left is the test input, and the bottom-right quadrant is separator filler.

    Rule:
    - For each non-background color, compare its cells in the training input and output.
    - Infer the unique translation vector (dr, dc) whose clipped shift reproduces the output.
    - Apply each color's vector to all same-colored test cells.
    - Cells shifted out of bounds are discarded; original positions are cleared to background.

    Output:
    - Only the transformed test panel.
    """
    g = np.asarray(grid, dtype=int)
    _sep_color, train_in, train_out, test_in = _split_panels(g)
    height, width = train_in.shape
    bg = _panel_background(train_in, train_out, test_in)

    colors_in = set(map(int, np.unique(train_in))) - {int(bg)}
    colors_out = set(map(int, np.unique(train_out))) - {int(bg)}
    assert colors_in == colors_out, "training pair must preserve the non-background color set"

    vectors: dict[int, tuple[int, int]] = {}
    for color in sorted(colors_in):
        src = {(int(r), int(c)) for r, c in np.argwhere(train_in == int(color))}
        dst = {(int(r), int(c)) for r, c in np.argwhere(train_out == int(color))}
        assert src, f"missing training pixels for color {color}"
        vectors[int(color)] = _find_vector(src, dst, height, width)

    out = np.full(test_in.shape, int(bg), dtype=int)
    test_colors = set(map(int, np.unique(test_in))) - {int(bg)}
    assert test_colors <= set(vectors), "test panel uses an unseen color"

    for color, (dr, dc) in vectors.items():
        for r, c in np.argwhere(test_in == int(color)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < height and 0 <= cc < width:
                assert out[rr, cc] in (int(bg), int(color)), "different colors must not collide"
                out[rr, cc] = int(color)

    return out


def _sign(x: int) -> int:
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def _vector_pool(height: int, width: int) -> list[tuple[int, int]]:
    max_dr = min(3, height - 2)
    max_dc = min(3, width - 2)
    return [
        (int(dr), int(dc))
        for dr in range(-max_dr, max_dr + 1)
        for dc in range(-max_dc, max_dc + 1)
        if (dr, dc) != (0, 0)
    ]


def _choose_vectors(height: int, width: int, n_colors: int) -> dict[int, tuple[int, int]] | None:
    bg, *colors = random.sample(COLOR_POOL, n_colors + 1)
    pool = _vector_pool(height, width)
    if len(pool) < n_colors:
        return None

    for _ in range(80):
        vectors = random.sample(pool, n_colors)
        directions = {(_sign(dr), _sign(dc)) for dr, dc in vectors}
        if n_colors > 1 and len(directions) == 1:
            continue
        mapping = {int(color): (int(dr), int(dc)) for color, (dr, dc) in zip(colors, vectors)}
        mapping[int(bg)] = (0, 0)
        return mapping
    return None


def _placement_candidates(
    shape: frozenset[tuple[int, int]],
    dr: int,
    dc: int,
    height: int,
    width: int,
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    require_clip: bool,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    shape_h, shape_w = _shape_size(shape)
    candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
    for r0 in range(height - shape_h + 1):
        for c0 in range(width - shape_w + 1):
            src = {(int(r0 + r), int(c0 + c)) for r, c in shape}
            if src & occupied_in:
                continue
            dst = _shift_cells(src, dr, dc, height, width)
            if require_clip:
                if not (0 < len(dst) < len(src)):
                    continue
            else:
                if len(dst) != len(src):
                    continue
            if dst & occupied_out:
                continue
            candidates.append((src, dst))
    return candidates


def _paint_panel(
    height: int,
    width: int,
    bg: int,
    placements: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    panel = np.full((height, width), int(bg), dtype=int)
    for color, cells in placements.items():
        for r, c in cells:
            assert panel[r, c] == int(bg)
            panel[r, c] = int(color)
    return panel


def _build_training_pair(
    height: int,
    width: int,
    bg: int,
    vectors: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    occupied_in: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()
    train_src: dict[int, set[tuple[int, int]]] = {}
    train_dst: dict[int, set[tuple[int, int]]] = {}

    colors = list(vectors)
    random.shuffle(colors)
    for color in colors:
        dr, dc = vectors[color]
        placed = False
        for _ in range(100):
            shape = _random_shape(2, 4)
            candidates = _placement_candidates(
                shape,
                dr,
                dc,
                height,
                width,
                occupied_in,
                occupied_out,
                require_clip=False,
            )
            if not candidates:
                continue
            src, dst = random.choice(candidates)
            train_src[int(color)] = src
            train_dst[int(color)] = dst
            occupied_in |= src
            occupied_out |= dst
            placed = True
            break
        if not placed:
            return None

    return _paint_panel(height, width, bg, train_src), _paint_panel(height, width, bg, train_dst)


def _build_test_pair(
    height: int,
    width: int,
    bg: int,
    vectors: dict[int, tuple[int, int]],
    clip_colors: set[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    occupied_in: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()
    test_src: dict[int, set[tuple[int, int]]] = {}
    test_dst: dict[int, set[tuple[int, int]]] = {}

    colors = list(vectors)
    random.shuffle(colors)
    for color in colors:
        dr, dc = vectors[color]
        require_clip = int(color) in clip_colors
        placed = False
        for _ in range(140):
            shape = _random_shape(2, 5)
            candidates = _placement_candidates(
                shape,
                dr,
                dc,
                height,
                width,
                occupied_in,
                occupied_out,
                require_clip=require_clip,
            )
            if not candidates:
                continue
            src, dst = random.choice(candidates)
            test_src[int(color)] = src
            test_dst[int(color)] = dst
            occupied_in |= src
            occupied_out |= dst
            placed = True
            break
        if not placed:
            return None

    return _paint_panel(height, width, bg, test_src), _paint_panel(height, width, bg, test_dst)


def _compose_grid(
    train_in: np.ndarray,
    train_out: np.ndarray,
    test_in: np.ndarray,
) -> np.ndarray:
    height, width = train_in.shape
    grid = np.full((2 * height + 1, 2 * width + 1), SEPARATOR_COLOR, dtype=int)
    grid[:height, :width] = train_in
    grid[:height, width + 1 :] = train_out
    grid[height + 1 :, :width] = test_in
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(6, 9)
    width = random.randint(6, 9)
    n_colors = random.randint(2, 4)

    chosen = _choose_vectors(height, width, n_colors)
    if chosen is None:
        return None

    bg = next(color for color, vec in chosen.items() if vec == (0, 0))
    vectors = {color: vec for color, vec in chosen.items() if color != bg}

    train_pair = _build_training_pair(height, width, bg, vectors)
    if train_pair is None:
        return None
    train_in, train_out = train_pair

    clip_colors: set[int] = set()
    if random.random() < 0.5:
        clip_count = random.randint(1, len(vectors))
        clip_colors = set(random.sample(list(vectors), clip_count))

    test_pair = _build_test_pair(height, width, bg, vectors, clip_colors)
    if test_pair is None:
        return None
    test_in, expected = test_pair

    grid = _compose_grid(train_in, train_out, test_in)

    learned = main(grid)
    if not np.array_equal(learned, expected):
        return None

    _, parsed_train_in, parsed_train_out, parsed_test_in = _split_panels(grid)
    parsed_bg = _panel_background(parsed_train_in, parsed_train_out, parsed_test_in)
    if int(parsed_bg) != int(bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
