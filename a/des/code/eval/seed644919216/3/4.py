from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400

DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_TO_ROT_K = {
    "up": 0,
    "right": 3,  # 90 degrees clockwise
    "down": 2,
    "left": 1,   # 270 degrees clockwise
}

# The pointer is a 3-cell L in a 2x2 box; its missing corner encodes the opening.
MISSING_TO_DIR = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 1): "down",
    (1, 0): "left",
}
DIR_TO_MISSING = {direction: corner for corner, direction in MISSING_TO_DIR.items()}


def _most_common(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.ravel())
    assert counts
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _axis_spans(lines: np.ndarray, n: int) -> tuple[list[tuple[int, int]], int] | None:
    lines = np.asarray(lines, dtype=int)
    if lines.size == 0 or np.unique(lines).size != lines.size:
        return None
    lines = np.sort(lines)

    if lines.size > 1:
        diffs = np.diff(lines)
        if np.any(diffs <= 1) or not np.all(diffs == diffs[0]):
            return None
        block_size = int(diffs[0] - 1)
    else:
        block_size = int(lines[0])
    if block_size < 2:
        return None

    if lines[0] == 0 and lines[-1] == n - 1:
        spans = [(int(lines[i] + 1), int(lines[i + 1])) for i in range(lines.size - 1)]
        if spans and all(end - start == block_size for start, end in spans):
            return spans, block_size

    if lines[0] == block_size and n - int(lines[-1]) - 1 == block_size:
        spans: list[tuple[int, int]] = []
        prev = -1
        for line in lines:
            spans.append((int(prev + 1), int(line)))
            prev = int(line)
        spans.append((int(prev + 1), int(n)))
        if spans and all(end - start == block_size for start, end in spans):
            return spans, block_size

    return None


def _parse_grid(g: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    best: tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None = None

    for color in np.unique(g):
        row_lines = np.flatnonzero(np.all(g == int(color), axis=1))
        col_lines = np.flatnonzero(np.all(g == int(color), axis=0))
        row_info = _axis_spans(row_lines, h)
        col_info = _axis_spans(col_lines, w)
        if row_info is None or col_info is None:
            continue

        row_spans, row_block = row_info
        col_spans, col_block = col_info
        if row_block != col_block or len(row_spans) != len(col_spans):
            continue

        score = len(row_lines) + len(col_lines)
        candidate = (score, int(color), row_spans, col_spans)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, "failed to detect the maroon grid lines"
    _score, line_color, row_spans, col_spans = best
    return line_color, row_spans[0][1] - row_spans[0][0], row_spans, col_spans


def _infer_bg(g: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> int:
    samples = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            samples.append(g[r0:r1, c0:c1].ravel())
    return _most_common(np.concatenate(samples))


def _pointer_direction(block: np.ndarray, bg: int) -> str | None:
    coords = np.argwhere(block != int(bg))
    if coords.shape[0] != 3:
        return None

    colors = {int(block[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if int(r1 - r0) != 1 or int(c1 - c0) != 1:
        return None

    occupied = {(int(r - r0), int(c - c0)) for r, c in coords}
    if occupied == {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return None

    missing = ({(0, 0), (0, 1), (1, 0), (1, 1)} - occupied)
    if len(missing) != 1:
        return None
    return MISSING_TO_DIR[next(iter(missing))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each neighboring pattern into the pointer block and rotate it according
    to the pointer's L-shaped opening.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_color, block_size, row_spans, col_spans = _parse_grid(g)
    bg = _infer_bg(g, row_spans, col_spans)

    blocks: list[list[dict[str, object]]] = []
    for r0, r1 in row_spans:
        row: list[dict[str, object]] = []
        for c0, c1 in col_spans:
            block = g[r0:r1, c0:c1].copy()
            assert block.shape == (block_size, block_size)
            assert not np.any(block == int(line_color))

            direction = _pointer_direction(block, bg)
            if direction is None:
                assert np.any(block != int(bg)), "every block should contain a pattern or a pointer"
                row.append({"kind": "pattern", "patch": block})
            else:
                row.append({"kind": "pointer", "patch": block, "direction": direction})
        blocks.append(row)

    out = g.copy()
    k = len(blocks)
    for br in range(k):
        for bc in range(k):
            cell = blocks[br][bc]
            if cell["kind"] != "pointer":
                continue

            direction = cell["direction"]
            dr, dc = DIR_TO_DELTA[direction]
            sr = br + dr
            sc = bc + dc
            assert 0 <= sr < k and 0 <= sc < k, "pointer source must stay inside the block board"
            source = blocks[sr][sc]
            assert source["kind"] == "pattern", "the indicated neighbor must be a pattern block"

            rotated = np.rot90(source["patch"], DIR_TO_ROT_K[direction])
            r0, r1 = row_spans[br]
            c0, c1 = col_spans[bc]
            out[r0:r1, c0:c1] = rotated

    return out


def _neighbors4(r: int, c: int, n: int) -> list[tuple[int, int]]:
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < n and 0 <= cc < n:
            result.append((rr, cc))
    return result


def _make_pattern(block_size: int) -> np.ndarray | None:
    inner = [(r, c) for r in range(1, block_size - 1) for c in range(1, block_size - 1)]
    max_area = min(8, len(inner))
    if max_area < 4:
        return None
    target_area = random.randint(4, max_area)

    mask = np.zeros((block_size, block_size), dtype=bool)
    start = random.choice(inner)
    mask[start] = True
    cells = [start]
    frontier = set(_neighbors4(start[0], start[1], block_size))
    frontier = {cell for cell in frontier if 1 <= cell[0] < block_size - 1 and 1 <= cell[1] < block_size - 1}

    while len(cells) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        cells.append(cell)
        for nxt in _neighbors4(cell[0], cell[1], block_size):
            if 1 <= nxt[0] < block_size - 1 and 1 <= nxt[1] < block_size - 1 and not mask[nxt]:
                frontier.add(nxt)

    if len(cells) != target_area:
        return None

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if int(r1 - r0) < 1 or int(c1 - c0) < 1:
        return None

    patch = np.full((block_size, block_size), BLACK, dtype=int)
    ordered = sorted((int(r), int(c)) for r, c in coords)
    random.shuffle(ordered)
    color_count = random.randint(2, min(4, len(ordered)))
    colors = random.sample(PALETTE, color_count)
    for idx, (r, c) in enumerate(ordered):
        patch[r, c] = colors[idx % color_count]

    if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
        return None
    if _pointer_direction(patch, BLACK) is not None:
        return None
    return patch


def _make_pointer(block_size: int, direction: str) -> np.ndarray:
    patch = np.full((block_size, block_size), BLACK, dtype=int)
    color = random.choice(PALETTE)
    anchor_r = random.randint(1, block_size - 3)
    anchor_c = random.randint(1, block_size - 3)
    missing = DIR_TO_MISSING[direction]
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                patch[anchor_r + dr, anchor_c + dc] = color
    return patch


def _choose_pointer_layout(k: int, pointer_count: int) -> tuple[np.ndarray, dict[tuple[int, int], str]] | None:
    kinds = np.full((k, k), "pattern", dtype=object)
    locked = np.zeros((k, k), dtype=bool)
    directions: dict[tuple[int, int], str] = {}

    pointer_dirs: list[str] = []
    while len(pointer_dirs) < pointer_count:
        cycle = list(DIR_TO_DELTA)
        random.shuffle(cycle)
        pointer_dirs.extend(cycle)
    pointer_dirs = pointer_dirs[:pointer_count]

    for direction in pointer_dirs:
        dr, dc = DIR_TO_DELTA[direction]
        candidates = []
        for r in range(k):
            for c in range(k):
                sr = r + dr
                sc = c + dc
                if not (0 <= sr < k and 0 <= sc < k):
                    continue
                if kinds[r, c] != "pattern" or locked[r, c]:
                    continue
                if kinds[sr, sc] != "pattern":
                    continue
                candidates.append((r, c, sr, sc))
        if not candidates:
            return None

        r, c, sr, sc = random.choice(candidates)
        kinds[r, c] = "pointer"
        locked[sr, sc] = True
        directions[(r, c)] = direction

    return kinds, directions


def _assemble(blocks: list[list[np.ndarray]], line_color: int) -> np.ndarray:
    k = len(blocks)
    block_size = blocks[0][0].shape[0]
    side = k * block_size + k + 1
    grid = np.full((side, side), int(line_color), dtype=int)
    for br in range(k):
        for bc in range(k):
            r0 = 1 + br * (block_size + 1)
            c0 = 1 + bc * (block_size + 1)
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = blocks[br][bc]
    return grid


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(3, 5)
    block_size = random.randint(5, 7)
    pointer_count = random.randint(4, min(6, k * k - 1))

    layout = _choose_pointer_layout(k, pointer_count)
    if layout is None:
        return None
    kinds, directions = layout

    input_blocks: list[list[np.ndarray]] = []
    for r in range(k):
        row = []
        for c in range(k):
            if kinds[r, c] == "pointer":
                row.append(_make_pointer(block_size, directions[(r, c)]))
            else:
                pattern = _make_pattern(block_size)
                if pattern is None:
                    return None
                row.append(pattern)
        input_blocks.append(row)

    output_blocks = [[block.copy() for block in row] for row in input_blocks]
    for (r, c), direction in directions.items():
        dr, dc = DIR_TO_DELTA[direction]
        source = input_blocks[r + dr][c + dc]
        output_blocks[r][c] = np.rot90(source, DIR_TO_ROT_K[direction])

    inp = _assemble(input_blocks, MAROON)
    expected = _assemble(output_blocks, MAROON)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
