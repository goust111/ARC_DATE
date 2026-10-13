from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

INSTRUCTION_COLORS = (BLUE, RED, GREEN)
_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _snake_positions(h: int, w: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for r in range(h):
        cols = range(w) if r % 2 == 0 else range(w - 1, -1, -1)
        order.extend((r, c) for c in cols)
    return order


def _decode_sequence_snake(grid: np.ndarray, split_row: int) -> list[int]:
    seq: list[int] = []
    top = grid[:split_row]
    for r, c in _snake_positions(*top.shape):
        color = int(top[r, c])
        if color not in (BLACK, GREY, MAROON):
            seq.append(color)
    return seq


def _decode_sequence_row_major(grid: np.ndarray, split_row: int) -> list[int]:
    seq: list[int] = []
    top = grid[:split_row]
    for r in range(top.shape[0]):
        for c in range(top.shape[1]):
            color = int(top[r, c])
            if color not in (BLACK, GREY, MAROON):
                seq.append(color)
    return seq


def _extract_template_mask(grid: np.ndarray, split_row: int) -> np.ndarray:
    lower = grid[split_row + 1 :]
    mask = lower == YELLOW
    box = _bbox(mask)
    assert box is not None, "missing yellow template"
    template = _crop_bbox(mask, box)
    assert template.shape[0] >= 2 and template.shape[1] >= 2
    assert _is_connected4(template)
    return template


def _apply_instruction(template: np.ndarray, color: int) -> np.ndarray:
    if color == RED:
        return np.flipud(template)
    if color == BLUE:
        return np.fliplr(template)
    if color == GREEN:
        return template.copy()
    raise AssertionError(f"unexpected instruction color {color}")


def _render_output(template: np.ndarray, seq: list[int]) -> np.ndarray:
    h, w = template.shape
    out = np.zeros((h, w * len(seq)), dtype=int)
    for i, color in enumerate(seq):
        piece = _apply_instruction(template, color)
        out[:, i * w : (i + 1) * w][piece] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a serpentine color instruction strip and concatenate mirrored template copies.

    Input structure:
    - A single full gray row splits the grid into an upper instruction area and a lower template area.
    - The instruction area has black background, one maroon start marker at the top-left, and scattered red/blue/green pixels.
    - The lower area contains one yellow 4-connected template on black background.

    Rule:
    - Scan the upper area in snake order: row 0 left-to-right, row 1 right-to-left, row 2 left-to-right, and so on.
    - Record every non-black, non-gray, non-maroon pixel encountered; red means flip the template top-to-bottom, blue means flip it left-to-right, and green means keep it unchanged.
    - Apply each instruction independently to the original template bbox and concatenate the resulting copies from left to right with no gap.

    Key insight:
    - The colored instruction pixels are meaningful only after converting their 2D placement into a 1D time sequence via the serpentine scan.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(divider_rows) == 1, "expected exactly one gray divider row"
    split_row = divider_rows[0]
    assert g[0, 0] == MAROON

    seq = _decode_sequence_snake(g, split_row)
    assert 3 <= len(seq) <= 4
    assert set(seq) == set(INSTRUCTION_COLORS)

    template = _extract_template_mask(g, split_row)
    return _render_output(template, seq)


def _random_connected_template() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        h = int(_RNG.integers(3, 5))
        w = int(_RNG.integers(3, 5))
        area = int(_RNG.integers(5, min(7, h * w) + 1))

        mask = np.zeros((h, w), dtype=bool)
        start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for rr, cc in _neighbors4(int(r), int(c)):
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.add((rr, cc))
            rr, cc = list(frontier)[int(_RNG.integers(len(frontier)))]
            mask[rr, cc] = True

        box = _bbox(mask)
        assert box is not None
        mask = _crop_bbox(mask, box)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if not _is_connected4(mask):
            continue
        flip_ud = np.flipud(mask)
        flip_lr = np.fliplr(mask)
        if np.array_equal(mask, flip_ud) or np.array_equal(mask, flip_lr):
            continue
        if np.array_equal(flip_ud, flip_lr):
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric template")


def _sample_instruction_sequence() -> list[int]:
    seq = [BLUE, RED, GREEN]
    if bool(_RNG.integers(0, 2)):
        seq.append(int(_RNG.choice(INSTRUCTION_COLORS)))
    _RNG.shuffle(seq)
    return [int(c) for c in seq]


def _sample_instruction_positions(top_h: int, width: int, n: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(top_h) for c in range(width) if (r, c) != (0, 0)]
    picks = _RNG.choice(len(cells), size=n, replace=False)
    return [cells[int(i)] for i in picks]


def _assemble_input(
    top_h: int,
    width: int,
    lower_h: int,
    template: np.ndarray,
    seq: list[int],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((top_h + 1 + lower_h, width), dtype=int)
    grid[top_h, :] = GREY
    grid[0, 0] = MAROON

    snake_rank = {pos: i for i, pos in enumerate(_snake_positions(top_h, width))}
    ordered_positions = sorted(positions, key=lambda pos: snake_rank[pos])
    for color, (r, c) in zip(seq, ordered_positions):
        grid[r, c] = color

    th, tw = template.shape
    max_top_margin = lower_h - th - 1
    top_margin = int(_RNG.integers(1, max_top_margin + 1))
    left_margin = int(_RNG.integers(1, width - tw))
    r0 = top_h + 1 + top_margin
    c0 = left_margin
    grid[r0 : r0 + th, c0 : c0 + tw][template] = YELLOW
    return grid


def _validate_generated_case(grid: np.ndarray, seq: list[int], template: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(grid[0, 0]) != MAROON:
        return False

    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    if len(divider_rows) != 1:
        return False
    split_row = divider_rows[0]

    expected = _render_output(template, seq)
    if not np.array_equal(main(grid), expected):
        return False

    decoded = _decode_sequence_snake(grid, split_row)
    if decoded != seq:
        return False

    alt_seq = _decode_sequence_row_major(grid, split_row)
    if alt_seq == seq:
        return False
    if np.array_equal(_render_output(template, alt_seq), expected):
        return False

    extracted = _extract_template_mask(grid, split_row)
    if not np.array_equal(extracted, template):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        template = _random_connected_template()
        th, tw = template.shape

        width = int(max(_RNG.integers(7, 11), tw + 2))
        top_h = int(_RNG.integers(3, 5))
        lower_h = int(max(_RNG.integers(5, 8), th + 2))
        seq = _sample_instruction_sequence()
        positions = _sample_instruction_positions(top_h, width, len(seq))
        grid = _assemble_input(top_h, width, lower_h, template, seq, positions)
        if _validate_generated_case(grid, seq, template):
            return grid
    raise RuntimeError("failed to generate a valid serpentine-mirror puzzle")
