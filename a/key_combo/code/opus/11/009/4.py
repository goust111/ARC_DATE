from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
PATTERN_SLOTS = [(0, 0), (0, 2), (1, 2), (2, 0), (2, 1), (1, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            block = grid[r0 : r0 + 3, c0 : c0 + 3]
            if int(block[1, 1]) in (BG, FRAME):
                continue
            border_ok = bool(
                np.all(block[0, :] == FRAME)
                and np.all(block[2, :] == FRAME)
                and np.all(block[:, 0] == FRAME)
                and np.all(block[:, 2] == FRAME)
            )
            if not border_ok:
                continue
            if bool(used[r0 : r0 + 3, c0 : c0 + 3].any()):
                continue
            used[r0 : r0 + 3, c0 : c0 + 3] = True
            frames.append(
                {
                    "top_left": (r0, c0),
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "center": (r0 + 1, c0 + 1),
                    "color": int(block[1, 1]),
                }
            )
    frames.sort(key=lambda f: f["top_left"])
    return frames


def _find_control_row(grid: np.ndarray, max_frame_row: int) -> int:
    for r in range(grid.shape[0] - 1, max_frame_row, -1):
        if int(np.count_nonzero((grid[r] != BG) & (grid[r] != FRAME))) >= 2:
            return r
    raise AssertionError("control row not found")


def _complete_rot180_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=int).copy()
    w = row.shape[0]
    assert w % 2 == 0, "control width must be even"
    for c in range(w):
        mc = w - 1 - c
        a = int(row[c])
        b = int(row[mc])
        if a == BG and b != BG:
            row[c] = b
        elif b == BG and a != BG:
            row[mc] = a
        elif a != BG and b != BG:
            assert a == b, "visible control colors must already satisfy 180-degree symmetry"
    for c in range(w):
        assert int(row[c]) == int(row[w - 1 - c]), "control row must be fully recoverable"
    return row


def _read_left_half_sequence(row: np.ndarray) -> list[int]:
    half = row.shape[0] // 2
    return [int(v) for v in row[:half] if int(v) != BG]


def _bridge_cells(a: dict, b: dict) -> tuple[list[tuple[int, int]], str]:
    ra, ca = map(int, a["center"])
    rb, cb = map(int, b["center"])
    r0a, c0a = map(int, a["top_left"])
    r0b, c0b = map(int, b["top_left"])

    if ra == rb:
        if ca < cb:
            cells = [(ra, c) for c in range(c0a + 3, c0b)]
        else:
            cells = [(ra, c) for c in range(c0b + 3, c0a)]
        return cells, "row"

    if ca == cb:
        if ra < rb:
            cells = [(r, ca) for r in range(r0a + 3, r0b)]
        else:
            cells = [(r, ca) for r in range(r0b + 3, r0a)]
        return cells, "col"

    step_h = 1 if cb > ca else -1
    start_c = ca + 2 * step_h
    horiz = [(ra, c) for c in range(start_c, cb + step_h, step_h)]

    step_v = 1 if rb > ra else -1
    end_r = rb - 2 * step_v
    vert = [(r, cb) for r in range(ra, end_r + step_v, step_v)]

    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in horiz + vert:
        if cell not in seen:
            cells.append(cell)
            seen.add(cell)
    return cells, "L"


def _apply_bridges(grid: np.ndarray, frames_by_color: dict[int, dict], sequence: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for color_a, color_b in zip(sequence, sequence[1:]):
        a = frames_by_color[int(color_a)]
        b = frames_by_color[int(color_b)]
        cells, _kind = _bridge_cells(a, b)
        for r, c in cells:
            out[r, c] = int(color_a)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a rotationally symmetric control strip, then use its recovered visit order to bridge framed color blocks.

    Input structure:
    - The background is black.
    - The upper area contains 4-6 disjoint 3x3 grey framed squares, each with a unique 1x1 colored center.
    - One bottom-row control strip contains mirrored colored pixels plus black occlusions; its right half is a 180-degree copy of the left half.

    Rule:
    - Detect the grey framed squares and the bottom control row.
    - Restore every black control cell from its 180-degree partner so the whole row becomes symmetric.
    - Read the recovered non-black colors from the left half, left-to-right, as the visit sequence.
    - For each adjacent color pair, connect the corresponding squares with a 1-pixel bridge colored by the first square's color: draw a straight segment when centers share a row or column, otherwise draw an L-shaped path that goes horizontal first and then vertical.

    Key insight:
    - The bottom strip is not the path itself but a corrupted, mirrored instruction tape; only after rotational completion does its left half reveal the unique bridge order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert 4 <= len(frames) <= 6, "expected 4-6 framed squares"
    colors = [int(f["color"]) for f in frames]
    assert len(set(colors)) == len(colors), "square center colors must be unique"
    frames_by_color = {int(f["color"]): f for f in frames}

    max_frame_row = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = _find_control_row(g, max_frame_row)

    out = np.array(g, dtype=int, copy=True)
    completed = _complete_rot180_row(out[ctrl_r])
    out[ctrl_r] = completed
    sequence = _read_left_half_sequence(completed)

    assert len(sequence) == len(frames), "the informative half must mention every square exactly once"
    assert set(sequence) == set(colors), "control sequence must match square colors"

    out = _apply_bridges(out, frames_by_color, sequence)
    return out


def _make_positions(rng: random.Random) -> list[tuple[int, int]]:
    row_margin = rng.randint(1, 2)
    col_margin = rng.randint(1, 2)
    row_steps = [rng.randint(4, 5), rng.randint(4, 5)]
    col_steps = [rng.randint(5, 6), rng.randint(5, 6)]

    rows = [row_margin]
    rows.append(rows[-1] + row_steps[0])
    rows.append(rows[-1] + row_steps[1])
    cols = [col_margin]
    cols.append(cols[-1] + col_steps[0])
    cols.append(cols[-1] + col_steps[1])

    return [(rows[r], cols[c]) for r, c in PATTERN_SLOTS]


def _make_control_row(
    rng: random.Random, width: int, sequence: list[int]
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    assert width % 2 == 0
    half = width // 2
    needed = len(sequence)
    left_cols: list[int] = []
    pos = rng.randint(1, 2)
    for idx in range(needed):
        left_cols.append(pos)
        remaining = needed - idx - 1
        min_room = remaining * 2 + 1
        if idx != needed - 1:
            max_next = half - min_room
            pos += rng.randint(1, min(3, max_next - pos))
    assert left_cols[-1] < half - 1

    full = np.zeros(width, dtype=int)
    for c, color in zip(left_cols, sequence):
        full[c] = int(color)
        full[width - 1 - c] = int(color)

    occluded = full.copy()
    pair_modes = [0] * needed
    left_hidden_idx = rng.randrange(needed)
    right_hidden_idx = rng.randrange(needed)
    pair_modes[left_hidden_idx] = 1
    pair_modes[right_hidden_idx] = 2
    for i in range(needed):
        if i in (left_hidden_idx, right_hidden_idx):
            continue
        pair_modes[i] = rng.choices([0, 1, 2], weights=[2, 2, 2], k=1)[0]

    for i, c in enumerate(left_cols):
        mc = width - 1 - c
        if pair_modes[i] == 1:
            occluded[c] = BG
        elif pair_modes[i] == 2:
            occluded[mc] = BG

    assert np.any(occluded != full)
    return occluded, full, left_cols, pair_modes


def _sequence_branch_counts(positions: list[tuple[int, int]]) -> tuple[int, int]:
    rowcol = 0
    elbow = 0
    for (r0, c0), (r1, c1) in zip(positions, positions[1:]):
        if r0 == r1 or c0 == c1:
            rowcol += 1
        else:
            elbow += 1
    return rowcol, elbow


def _validate_scene(inp: np.ndarray, expected: np.ndarray, sequence: list[int]) -> bool:
    frames = _find_frames(inp)
    if len(frames) != len(sequence):
        return False
    colors = {int(f["color"]) for f in frames}
    if colors != set(sequence):
        return False

    max_frame_row = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = _find_control_row(inp, max_frame_row)
    completed = _complete_rot180_row(inp[ctrl_r])
    if _read_left_half_sequence(completed) != list(map(int, sequence)):
        return False

    if not np.array_equal(main(inp), expected):
        return False

    left_hidden = False
    total_hidden = 0
    w = inp.shape[1]
    for c in range(w // 2):
        mc = w - 1 - c
        if int(completed[c]) == BG:
            continue
        if int(inp[ctrl_r, c]) == BG:
            left_hidden = True
            total_hidden += 1
        if int(inp[ctrl_r, mc]) == BG:
            total_hidden += 1
    return left_hidden and total_hidden >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(4, 6)
    positions = _make_positions(rng)[:count]
    rowcol, elbow = _sequence_branch_counts(positions)
    if rowcol == 0 or elbow == 0:
        return None

    colors = rng.sample(PALETTE, count)
    frames = [
        {
            "top_left": pos,
            "bbox": (pos[0], pos[1], pos[0] + 2, pos[1] + 2),
            "center": (pos[0] + 1, pos[1] + 1),
            "color": color,
        }
        for pos, color in zip(positions, colors)
    ]
    frames_by_color = {int(f["color"]): f for f in frames}
    sequence = [int(f["color"]) for f in frames]

    rightmost = max(c for _r, c in positions) + 3
    bottom_square = max(r for r, _c in positions) + 3
    min_width = max(rightmost + rng.randint(2, 5), 2 * (count * 2 + 2))
    width = min(30, max(16, min_width + (min_width % 2)))
    if width % 2 == 1:
        width += 1
    if width > 30:
        return None

    ctrl_gap = rng.randint(2, 4)
    ctrl_r = bottom_square + ctrl_gap
    height = ctrl_r + 1 + rng.randint(1, 2)
    if height > 30:
        return None

    inp = np.zeros((height, width), dtype=int)
    for frame in frames:
        r0, c0 = map(int, frame["top_left"])
        inp[r0 : r0 + 3, c0 : c0 + 3] = FRAME
        inp[r0 + 1, c0 + 1] = int(frame["color"])

    control_inp, control_full, _left_cols, _pair_modes = _make_control_row(rng, width, sequence)
    inp[ctrl_r] = control_inp

    expected = inp.copy()
    expected[ctrl_r] = control_full
    expected = _apply_bridges(expected, frames_by_color, sequence)

    occupied = {(r, c) for frame in frames for r in range(frame["bbox"][0], frame["bbox"][2] + 1) for c in range(frame["bbox"][1], frame["bbox"][3] + 1)}
    bridge_seen: set[tuple[int, int]] = set()
    kinds: set[str] = set()
    for color_a, color_b in zip(sequence, sequence[1:]):
        cells, kind = _bridge_cells(frames_by_color[int(color_a)], frames_by_color[int(color_b)])
        kinds.add(kind)
        if not cells:
            return None
        for cell in cells:
            if cell in occupied:
                return None
            if cell in bridge_seen:
                return None
            r, c = cell
            if not (0 <= r < height and 0 <= c < width):
                return None
            bridge_seen.add(cell)
    if not {"row", "col", "L"} & kinds:
        return None
    if "L" not in kinds or not ({"row", "col"} & kinds):
        return None

    if not _validate_scene(inp, expected, sequence):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid sample")
