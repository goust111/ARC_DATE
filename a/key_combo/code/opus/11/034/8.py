from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

COLOR_BY_FREE_COUNT = {
    0: RED,
    1: BLUE,
    2: GREEN,
    3: YELLOW,
    4: PINK,
}

# Template A contains F=0 and template B contains F=4.
TEMPLATE_A = [
    (0, 3),
    (2, 9),
    (3, 4),
    (4, 1),
    (6, 4),
    (6, 8),
    (7, 0),
]
TEMPLATE_B = [
    (0, 9),
    (1, 6),
    (2, 2),
    (4, 8),
    (5, 5),
    (7, 0),
]
TEMPLATES = (TEMPLATE_A, TEMPLATE_B)

_GEN_COUNTER = 0


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _extract_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "input must contain only black and grey"

    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    blocks: list[tuple[int, int]] = []

    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != GREY or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            assert len(cells) == 4 and r1 == r0 + 1 and c1 == c0 + 1 and set(cells) == expected, (
                "every grey component must be a solid 2x2 block"
            )
            blocks.append((r0, c0))

    blocks.sort()
    assert 4 <= len(blocks) <= 7, "expected 4 to 7 blocks"

    for i, (r0, c0) in enumerate(blocks):
        for r1, c1 in blocks[i + 1 :]:
            assert not (abs(r0 - r1) <= 2 and abs(c0 - c1) <= 2), (
                "blocks must have at least one blank pixel between them in every direction"
            )

    return blocks


def _is_blocked_exact(src: tuple[int, int], other: tuple[int, int], direction: str) -> bool:
    sr, sc = src
    orow, ocol = other
    cx = sc + 1
    cy = sr + 1
    row_hit = orow <= cy <= orow + 2
    col_hit = ocol <= cx <= ocol + 2

    if direction == "up":
        return col_hit and orow + 2 <= cy
    if direction == "down":
        return col_hit and orow >= sr
    if direction == "left":
        return row_hit and ocol + 2 <= cx
    if direction == "right":
        return row_hit and ocol >= sc
    raise ValueError(f"unknown direction: {direction}")


def _free_count_exact(blocks: list[tuple[int, int]], idx: int) -> int:
    src = blocks[idx]
    total = 0
    for direction in ("up", "down", "left", "right"):
        if not any(_is_blocked_exact(src, other, direction) for j, other in enumerate(blocks) if j != idx):
            total += 1
    return total


def _free_count_band(blocks: list[tuple[int, int]], idx: int) -> int:
    r, c = blocks[idx]
    up_free = not any(rr < r and abs(cc - c) <= 1 for j, (rr, cc) in enumerate(blocks) if j != idx)
    down_free = not any(rr > r and abs(cc - c) <= 1 for j, (rr, cc) in enumerate(blocks) if j != idx)
    left_free = not any(cc < c and abs(rr - r) <= 1 for j, (rr, cc) in enumerate(blocks) if j != idx)
    right_free = not any(cc > c and abs(rr - r) <= 1 for j, (rr, cc) in enumerate(blocks) if j != idx)
    return int(up_free) + int(down_free) + int(left_free) + int(right_free)


def _render_output_like_main(grid: np.ndarray, blocks: list[tuple[int, int]], free_counter) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = BLACK
    for i, (r, c) in enumerate(blocks):
        color = COLOR_BY_FREE_COUNT[free_counter(blocks, i)]
        out[r : r + 2, c : c + 2] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey 2x2 block by counting how many cardinal rays from its center reach the boundary unblocked.

    Input structure:
    - The background is black.
    - There are 4 to 7 disconnected solid 2x2 grey blocks.
    - Blocks never touch and always leave at least one blank pixel between them.

    Rule:
    - For each 2x2 block, treat the intersection point at its center as the ray origin.
    - Shoot virtual rays upward, downward, leftward, and rightward.
    - A direction is blocked if the ray would hit any other 2x2 block before the grid edge, including corner-only contact.
    - Count the number of free directions F and recolor the whole block by F: 0->red, 1->blue, 2->green, 3->yellow, 4->pink.

    Key insight:
    - A block's output color is determined by the full arrangement of the other blocks, because each one can block
      rays in a 3-column or 3-row corridor centered on the block's midpoint.
    """
    g = np.asarray(grid, dtype=int)
    blocks = _extract_blocks(g)
    return _render_output_like_main(g, blocks, _free_count_exact)


def _normalize_positions(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in blocks)
    min_c = min(c for _, c in blocks)
    return sorted((r - min_r, c - min_c) for r, c in blocks)


def _transform_positions(blocks: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    pts = _normalize_positions(blocks)
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1

    if rng.randrange(2):
        pts = [(c, r) for r, c in pts]
        h, w = w, h
    if rng.randrange(2):
        pts = [(h - 1 - r, c) for r, c in pts]
    if rng.randrange(2):
        pts = [(r, w - 1 - c) for r, c in pts]
    return _normalize_positions(pts)


def _make_input_from_template(template: list[tuple[int, int]], rng: random.Random) -> np.ndarray:
    pts = _transform_positions(template, rng)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)

    top = rng.randint(0, 4)
    bottom = rng.randint(0, 4)
    left = rng.randint(0, 4)
    right = rng.randint(0, 4)

    H = top + (max_r + 2) + bottom
    W = left + (max_c + 2) + right
    assert max(H, W) <= 30

    grid = np.zeros((H, W), dtype=int)
    for r, c in pts:
        rr = r + top
        cc = c + left
        grid[rr : rr + 2, cc : cc + 2] = GREY
    return grid


def _validate_generated_instance(grid: np.ndarray, expected_special_free_count: int) -> None:
    g = np.asarray(grid, dtype=int)
    assert isinstance(g, np.ndarray)
    assert g.ndim == 2 and max(g.shape) <= 30

    blocks = _extract_blocks(g)
    counts_exact = [_free_count_exact(blocks, i) for i in range(len(blocks))]
    counts_band = [_free_count_band(blocks, i) for i in range(len(blocks))]
    assert counts_exact == counts_band, "independent free-count oracles disagree"

    distinct = set(counts_exact)
    assert len(distinct) >= 3, "need at least three distinct free-direction counts"
    assert expected_special_free_count in distinct, "template family branch coverage was lost"
    assert min(counts_exact) <= 1 and max(counts_exact) >= 3, "instance is too weakly differentiated"

    out_exact = _render_output_like_main(g, blocks, _free_count_exact)
    out_band = _render_output_like_main(g, blocks, _free_count_band)
    out_main = main(g)
    assert np.array_equal(out_exact, out_band), "independent output renderers disagree"
    assert np.array_equal(out_main, out_exact), "main() does not match the oracle"
    assert not np.array_equal(g, out_main), "output must differ from input"
    assert len(set(map(int, np.unique(out_main))) - {BLACK}) >= 3, "output should expose multiple mapped colors"


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = random.Random()
    family_index = _GEN_COUNTER % len(TEMPLATES)
    _GEN_COUNTER += 1

    template = TEMPLATES[family_index]
    expected_special_free_count = 0 if family_index == 0 else 4

    for _ in range(100):
        grid = _make_input_from_template(template, rng)
        try:
            _validate_generated_instance(grid, expected_special_free_count)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ray-collision recoloring input")

