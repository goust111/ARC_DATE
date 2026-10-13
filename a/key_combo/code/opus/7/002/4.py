from __future__ import annotations

import random

import numpy as np


BLOCK = 3
LOW_H = 5
LOW_W = 5
INPUT_SHAPE = (LOW_H * BLOCK, LOW_W * BLOCK)

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE = (BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON)
ALL_BLOCKS = [(r, c) for r in range(LOW_H) for c in range(LOW_W)]
BLOCK_CELLS = [(r, c) for r in range(BLOCK) for c in range(BLOCK)]
EDGE_CELLS = [(r, c) for r, c in BLOCK_CELLS if r in (0, BLOCK - 1) or c in (0, BLOCK - 1)]
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _scan_nonblack_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) != BLACK
    ]


def _compress_thumbnail(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % BLOCK == 0 and w % BLOCK == 0, "grid must split into 3x3 blocks"
    assert get_bg(g) == BLACK, "expected black background"

    out = np.full((h // BLOCK, w // BLOCK), BLACK, dtype=int)
    marked_blocks = 0

    for br in range(h // BLOCK):
        for bc in range(w // BLOCK):
            block = g[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            has_marker = bool(np.any(block == RED))
            marked_blocks += int(has_marker)

            real_colors = sorted(
                int(color)
                for color in np.unique(block)
                if int(color) not in (BLACK, GREY, RED)
            )
            assert len(real_colors) <= 1, "each block must have at most one real color"

            if real_colors:
                color = real_colors[0]
                assert np.count_nonzero(block == color) >= 7, "main color must dominate its block"
                assert np.all(
                    (block == BLACK) | (block == GREY) | (block == RED) | (block == color)
                ), "block contains an unexpected extra color"
                out[br, bc] = color
            else:
                assert np.all(
                    (block == BLACK) | (block == GREY) | (block == RED)
                ), "empty block may only contain black, grey, and red"

    return out, marked_blocks


def _repaint_prefix(thumbnail: np.ndarray, count: int) -> np.ndarray:
    out = np.array(thumbnail, dtype=int, copy=True)
    nonblack = _scan_nonblack_positions(out)
    assert 0 < count <= len(nonblack), "marker count must fit the compressed foreground"
    for r, c in nonblack[:count]:
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each 3x3 block while ignoring grey/red cells, then recolor a red prefix whose length equals the number of marked blocks.

    Input structure:
    - The background is black and the grid is partitioned into 3x3 blocks.
    - Each block contains either only black plus optional grey/red cells, or one
      dominant non-black color plus optional grey noise and at most one red marker.
    - Grey cells are noise, while red cells are block markers whose positions are
      not preserved directly in the output.

    Rule:
    - Split the input into 3x3 blocks and count how many blocks contain a red marker.
    - Compress every block to one output pixel by ignoring grey and red; use the
      unique remaining non-black color, or black if none remains.
    - In the compressed grid, scan left-to-right and top-to-bottom, then recolor
      the first N non-black pixels red, where N is the marked-block count.

    Key insight:
    - Red plays two coupled roles: it is ignored locally during block compression,
      yet its global count survives and reappears as the length of the red prefix.
    """
    thumbnail, marked_blocks = _compress_thumbnail(grid)
    return _repaint_prefix(thumbnail, marked_blocks)


def _sample_lowres_colors(nonblack_count: int) -> list[int]:
    palette_size = random.randint(2, min(len(PALETTE), 5, nonblack_count))
    active = random.sample(PALETTE, palette_size)
    colors = active.copy()
    while len(colors) < nonblack_count:
        colors.append(random.choice(active))
    random.shuffle(colors)
    return colors


def _sample_thumbnail(n_marked: int) -> np.ndarray:
    min_nonblack = max(n_marked + 1, 5)
    nonblack_count = random.randint(min_nonblack, 16)

    colored_blocks = random.sample(ALL_BLOCKS, nonblack_count)
    colors = _sample_lowres_colors(nonblack_count)

    thumbnail = np.zeros((LOW_H, LOW_W), dtype=int)
    for pos, color in zip(colored_blocks, colors):
        thumbnail[pos] = color
    return thumbnail


def _sample_marker_blocks(
    thumbnail: np.ndarray, n_marked: int
) -> tuple[set[tuple[int, int]], str]:
    colored = [pos for pos in ALL_BLOCKS if int(thumbnail[pos]) != BLACK]
    black = [pos for pos in ALL_BLOCKS if int(thumbnail[pos]) == BLACK]
    assert len(colored) >= n_marked + 1
    assert black

    prefix = _scan_nonblack_positions(thumbnail)[:n_marked]
    late_colored = [pos for pos in colored if pos not in prefix]
    assert late_colored

    if random.random() < 0.55:
        black_count = random.randint(1, min(n_marked, len(black)))
        markers = set(random.sample(black, black_count))
        remaining_pool = [pos for pos in ALL_BLOCKS if pos not in markers]
        markers.update(random.sample(remaining_pool, n_marked - black_count))
        branch = "black-marker"
    else:
        late_count = random.randint(1, min(n_marked, len(late_colored)))
        markers = set(random.sample(late_colored, late_count))
        remaining_colored = [pos for pos in colored if pos not in markers]
        markers.update(random.sample(remaining_colored, n_marked - late_count))
        branch = "late-colored-marker"

    assert len(markers) == n_marked
    assert markers != set(prefix), "markers must not coincide with the recolored prefix"
    return markers, branch


def _sample_grey_counts(thumbnail: np.ndarray, markers: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    grey_counts: dict[tuple[int, int], int] = {}
    capacities: dict[tuple[int, int], int] = {}

    for pos in ALL_BLOCKS:
        is_colored = int(thumbnail[pos]) != BLACK
        cap = 1 if is_colored and pos in markers else 2
        capacities[pos] = cap
        if is_colored and pos in markers:
            choices = (0, 1)
            weights = (0.45, 0.55)
        elif is_colored:
            choices = (0, 1, 2)
            weights = (0.20, 0.45, 0.35)
        else:
            choices = (0, 1, 2)
            weights = (0.35, 0.40, 0.25)
        grey_counts[pos] = random.choices(choices, weights=weights, k=1)[0]

    while sum(grey_counts.values()) < 4 or sum(v > 0 for v in grey_counts.values()) < 3:
        pos = random.choice(ALL_BLOCKS)
        if grey_counts[pos] < capacities[pos]:
            grey_counts[pos] += 1

    return grey_counts


def _sample_cell(preferred: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    if random.random() < 0.7:
        return random.choice(list(preferred))
    return random.choice(BLOCK_CELLS)


def _render_input(
    thumbnail: np.ndarray,
    markers: set[tuple[int, int]],
    grey_counts: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)

    for br, bc in ALL_BLOCKS:
        color = int(thumbnail[br, bc])
        r0 = br * BLOCK
        c0 = bc * BLOCK
        block = np.full((BLOCK, BLOCK), color if color != BLACK else BLACK, dtype=int)

        used: set[tuple[int, int]] = set()
        if (br, bc) in markers:
            rr, cc = _sample_cell(EDGE_CELLS)
            block[rr, cc] = RED
            used.add((rr, cc))

        for _ in range(grey_counts[(br, bc)]):
            available = [cell for cell in BLOCK_CELLS if cell not in used]
            rr, cc = _sample_cell(available)
            while (rr, cc) in used:
                rr, cc = random.choice(available)
            block[rr, cc] = GREY
            used.add((rr, cc))

        grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = block

    return grid


def _validate_example(
    grid: np.ndarray,
    thumbnail: np.ndarray,
    markers: set[tuple[int, int]],
    branch: str,
) -> bool:
    try:
        compressed, count = _compress_thumbnail(grid)
        if not np.array_equal(compressed, thumbnail):
            return False
        if count != len(markers):
            return False

        nonblack = _scan_nonblack_positions(thumbnail)
        if not (len(nonblack) > count >= 1):
            return False

        prefix = set(nonblack[:count])
        if markers == prefix:
            return False

        if branch == "black-marker":
            if not any(int(thumbnail[pos]) == BLACK for pos in markers):
                return False
        else:
            if any(int(thumbnail[pos]) == BLACK for pos in markers):
                return False
            if not any(pos not in prefix for pos in markers):
                return False

        if np.count_nonzero(grid == GREY) < 4:
            return False
        if not np.array_equal(main(grid), _repaint_prefix(thumbnail, count)):
            return False
        return True
    except AssertionError:
        return False


def _generate_example() -> np.ndarray | None:
    n_marked = random.randint(1, 5)
    thumbnail = _sample_thumbnail(n_marked)
    markers, branch = _sample_marker_blocks(thumbnail, n_marked)
    grey_counts = _sample_grey_counts(thumbnail, markers)
    grid = _render_input(thumbnail, markers, grey_counts)
    if _validate_example(grid, thumbnail, markers, branch):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
