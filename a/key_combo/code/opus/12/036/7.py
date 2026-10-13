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
ORANGE = 7

PATH_COLORS = (BLUE, GREEN, YELLOW, PINK, ORANGE)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # right, down, left, up

GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 400
HINT_ATTEMPTS = 200


def _scan_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for c, active in enumerate(mask.tolist()):
        if active and start is None:
            start = c
        elif not active and start is not None:
            segments.append((start, c - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _parse_glyphs(strip: np.ndarray) -> list[int]:
    red_mask = strip == RED
    active_cols = red_mask.any(axis=0)
    segments = _scan_segments(active_cols)
    assert segments, "expected at least one glyph"

    glyphs: list[int] = []
    for c0, c1 in segments:
        patch = strip[:, c0 : c1 + 1]
        h, w = patch.shape
        assert h == 2
        if w == 1:
            assert np.array_equal(patch, np.array([[RED], [RED]]))
            glyphs.append(0)
            continue

        assert w == 2
        assert np.count_nonzero(patch == RED) == 3
        assert np.array_equal(patch[1], np.array([RED, RED]))
        if patch[0, 0] == BLACK and patch[0, 1] == RED:
            glyphs.append(-1)
        elif patch[0, 0] == RED and patch[0, 1] == BLACK:
            glyphs.append(1)
        else:
            raise AssertionError("invalid elbow glyph")
    return glyphs


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected a grey lattice frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_lattice(lower: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    r0, c0, r1, c1 = _frame_bbox(lower)
    sub = np.array(lower[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    row_lines = [r for r in range(sub.shape[0]) if bool(np.all(sub[r] == GREY))]
    col_lines = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == GREY))]

    assert 4 <= len(row_lines) <= 6
    assert 4 <= len(col_lines) <= 6
    assert row_lines[0] == 0 and row_lines[-1] == sub.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == sub.shape[1] - 1
    assert all(a + 1 < b for a, b in zip(row_lines, row_lines[1:]))
    assert all(a + 1 < b for a, b in zip(col_lines, col_lines[1:]))

    expected = np.zeros_like(sub, dtype=bool)
    expected[row_lines, :] = True
    expected[:, col_lines] = True
    assert np.array_equal(sub == GREY, expected), "grey pixels must form a complete lattice"
    return sub, row_lines, col_lines


def _slot_slice(lines: list[int], idx: int) -> slice:
    return slice(lines[idx] + 1, lines[idx + 1])


def _read_slot_colors(sub: np.ndarray, row_lines: list[int], col_lines: list[int]) -> np.ndarray:
    rows = len(row_lines) - 1
    cols = len(col_lines) - 1
    slot_colors = np.zeros((rows, cols), dtype=int)

    for sr in range(rows):
        rs = _slot_slice(row_lines, sr)
        for sc in range(cols):
            cs = _slot_slice(col_lines, sc)
            patch = sub[rs, cs]
            vals = [int(v) for v in np.unique(patch) if int(v) not in (BLACK, GREY)]
            assert len(vals) <= 1, "a slot interior must be monochrome"
            if vals:
                color = vals[0]
                assert np.all(patch == color)
                slot_colors[sr, sc] = color
            else:
                assert np.all(patch == BLACK)
    return slot_colors


def _infer_slot_color(slot_colors: np.ndarray, sr: int, sc: int) -> int:
    row_hints = [
        (abs(int(c) - int(sc)), int(c), int(slot_colors[sr, c]))
        for c in range(slot_colors.shape[1])
        if int(slot_colors[sr, c]) != BLACK
    ]
    col_hints = [int(r) for r in range(slot_colors.shape[0]) if int(slot_colors[r, sc]) != BLACK]

    assert row_hints, "every logical row must expose at least one colored hint"
    assert col_hints, "every logical column must expose at least one colored hint"

    best_dist = min(dist for dist, _c, _color in row_hints)
    best_colors = {color for dist, _c, color in row_hints if dist == best_dist}
    assert len(best_colors) == 1, "nearest same-row hints must agree in color"
    return next(iter(best_colors))


def _trace_path(glyphs: list[int], rows: int, cols: int) -> tuple[list[tuple[int, int]], int]:
    r = 0
    c = 0
    direction = 0
    visited = [(0, 0)]
    wraps = 0

    for turn in glyphs:
        direction = (direction + int(turn)) % 4
        dr, dc = DIRS[direction]
        nr = r + dr
        nc = c + dc
        if not 0 <= nr < rows:
            wraps += 1
        if not 0 <= nc < cols:
            wraps += 1
        r = nr % rows
        c = nc % cols
        visited.append((r, c))
    return visited, wraps


def main(grid: np.ndarray) -> np.ndarray:
    """Decode a top red glyph strip into a wrapping lattice path and fill only the visited slots.

    Input structure:
    - The top two rows are a black strip containing red glyphs separated by black gaps; each glyph is either a 2x1 vertical pair or a 2x2 elbow missing one top corner.
    - Below the strip is one grey one-pixel lattice frame whose full horizontal and vertical bars partition the area into a logical R x C grid of rectangular slots.
    - Some slot interiors already contain one solid color from the non-grey palette, while the remaining slot interiors are black.

    Rule:
    - Read the red glyphs left to right. A vertical pair means keep moving in the current direction; an elbow with the missing top-left corner means turn left, and an elbow with the missing top-right corner means turn right.
    - Start at logical slot (0,0) facing right, execute the instructions one step per glyph, and wrap around the lattice when a move exits a side.
    - For every visited slot, keep its existing color if it is already filled; otherwise infer its color from the nearest colored slot in the same logical row, requiring that the slot's logical column also has at least one colored hint somewhere. Unvisited empty slots stay black.
    - Output only the lattice region with its grey frame preserved; the top glyph strip is removed.

    Key insight:
    - The red strip is a navigation program rather than a shape to copy, and the final colors come from a separate row-based hint system already embedded in the lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 6

    glyphs = _parse_glyphs(g[:2])
    sub, row_lines, col_lines = _parse_lattice(g[2:])
    slot_colors = _read_slot_colors(sub, row_lines, col_lines)

    rows, cols = slot_colors.shape
    visited, _wraps = _trace_path(glyphs, rows, cols)

    out = sub.copy()
    for sr, sc in visited:
        color = int(slot_colors[sr, sc])
        if color == BLACK:
            color = _infer_slot_color(slot_colors, sr, sc)
        rs = _slot_slice(row_lines, sr)
        cs = _slot_slice(col_lines, sc)
        out[rs, cs] = color
    return out


def _line_positions(lengths: list[int]) -> list[int]:
    pos = [0]
    cur = 0
    for length in lengths:
        cur += int(length) + 1
        pos.append(cur)
    return pos


def _render_lattice(
    row_heights: list[int],
    col_widths: list[int],
    row_colors: list[int],
    hint_mask: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    row_lines = _line_positions(row_heights)
    col_lines = _line_positions(col_widths)
    height = row_lines[-1] + 1
    width = col_lines[-1] + 1
    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_lines, :] = GREY
    grid[:, col_lines] = GREY

    for sr, color in enumerate(row_colors):
        rs = _slot_slice(row_lines, sr)
        for sc in range(len(col_widths)):
            if hint_mask[sr, sc]:
                cs = _slot_slice(col_lines, sc)
                grid[rs, cs] = color
    return grid, row_lines, col_lines


def _render_expected(
    lower: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    row_colors: list[int],
    visited: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(lower, copy=True)
    for sr, sc in visited:
        rs = _slot_slice(row_lines, sr)
        cs = _slot_slice(col_lines, sc)
        out[rs, cs] = row_colors[sr]
    return out


def _glyph_width(glyphs: list[int]) -> int:
    return sum(1 if turn == 0 else 2 for turn in glyphs) + max(0, len(glyphs) - 1)


def _encode_strip(glyphs: list[int], total_width: int) -> np.ndarray:
    width = _glyph_width(glyphs)
    assert width <= total_width
    offset = random.randint(0, total_width - width)
    strip = np.full((2, total_width), BLACK, dtype=int)

    c = offset
    for idx, turn in enumerate(glyphs):
        if turn == 0:
            strip[:, c] = RED
            c += 1
        elif turn == -1:
            strip[0, c + 1] = RED
            strip[1, c : c + 2] = RED
            c += 2
        else:
            strip[0, c] = RED
            strip[1, c : c + 2] = RED
            c += 2
        if idx + 1 < len(glyphs):
            c += 1
    return strip


def _sample_glyphs(rows: int, cols: int, frame_width: int) -> tuple[list[int], list[tuple[int, int]], bool] | None:
    max_steps = min(8, max(4, (frame_width + 1) // 2))
    if max_steps < 4:
        return None

    wrap_required = random.random() < 0.45
    prefer_both_turns = random.random() < 0.7

    for _ in range(PATH_ATTEMPTS):
        steps = random.randint(4, max_steps)
        straight_count = random.randint(1, steps - 1)
        turn_count = steps - straight_count
        if turn_count == 0:
            continue

        turns: list[int] = []
        if prefer_both_turns and turn_count >= 2:
            turns.extend([-1, 1])
            turns.extend(random.choice((-1, 1)) for _ in range(turn_count - 2))
        else:
            lead = random.choice((-1, 1))
            turns.extend([lead] * turn_count)
            if turn_count >= 2 and random.random() < 0.35:
                turns[random.randrange(turn_count)] *= -1

        glyphs = [0] * straight_count + turns
        random.shuffle(glyphs)
        if _glyph_width(glyphs) > frame_width:
            continue
        if 0 not in glyphs or not any(turn != 0 for turn in glyphs):
            continue

        visited, wraps = _trace_path(glyphs, rows, cols)
        wrapped = wraps > 0
        if wrapped != wrap_required:
            continue
        unique_slots = len(set(visited))
        if unique_slots < min(4, rows * cols):
            continue
        if unique_slots == rows * cols:
            continue
        return glyphs, visited, wrapped
    return None


def _sample_hints(rows: int, cols: int, visited: list[tuple[int, int]]) -> np.ndarray | None:
    visited_set = set(visited)
    all_slots = [(r, c) for r in range(rows) for c in range(cols)]

    for _ in range(HINT_ATTEMPTS):
        hint = np.zeros((rows, cols), dtype=bool)

        for r in range(rows):
            choices = random.sample(range(cols), random.randint(1, min(cols, 2)))
            hint[r, choices] = True
        for c in range(cols):
            if not bool(hint[:, c].any()):
                hint[random.randrange(rows), c] = True
        for _extra in range(random.randint(0, rows)):
            hint[random.randrange(rows), random.randrange(cols)] = True

        if not any(int(hint[r].sum()) >= 2 for r in range(rows)) and cols >= 2:
            rr = random.randrange(rows)
            cc = random.sample(range(cols), 2)
            hint[rr, cc] = True

        if not any(hint[pos] for pos in visited_set):
            rr, cc = random.choice(tuple(visited_set))
            hint[rr, cc] = True

        visited_empty = [pos for pos in visited_set if not hint[pos]]
        if not visited_empty:
            removable = [
                (r, c)
                for r, c in visited_set
                if hint[r, c] and int(hint[r].sum()) > 1 and int(hint[:, c].sum()) > 1
            ]
            if not removable:
                continue
            rr, cc = random.choice(removable)
            hint[rr, cc] = False
            visited_empty = [pos for pos in visited_set if not hint[pos]]
            if not visited_empty:
                continue

        unvisited_empty = [pos for pos in all_slots if pos not in visited_set and not hint[pos]]
        if not unvisited_empty:
            continue
        if not bool(hint.all(axis=1).any()) and random.random() < 0.2:
            continue
        if not bool(hint.any(axis=1).all()) or not bool(hint.any(axis=0).all()):
            continue
        return hint
    return None


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    glyphs: list[int],
    visited: list[tuple[int, int]],
    hint_mask: np.ndarray,
    row_colors: list[int],
) -> bool:
    rows, cols = hint_mask.shape
    if _parse_glyphs(inp[:2]) != glyphs:
        return False
    if not hint_mask.any(axis=1).all() or not hint_mask.any(axis=0).all():
        return False
    if not any(turn == 0 for turn in glyphs) or not any(turn != 0 for turn in glyphs):
        return False

    visited_set = set(visited)
    if not any(hint_mask[pos] for pos in visited_set):
        return False
    if not any(not hint_mask[pos] for pos in visited_set):
        return False
    if not any((r, c) not in visited_set and not hint_mask[r, c] for r in range(rows) for c in range(cols)):
        return False

    slot_colors = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if hint_mask[r, c]:
                slot_colors[r, c] = row_colors[r]

    for r, c in visited_set:
        if slot_colors[r, c] == BLACK and _infer_slot_color(slot_colors, r, c) != row_colors[r]:
            return False

    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rows = random.randint(3, 5)
        cols = random.randint(3, 5)
        row_heights = [random.randint(2, 4) for _ in range(rows)]
        col_widths = [random.randint(2, 4) for _ in range(cols)]
        frame_width = sum(col_widths) + cols + 1
        if frame_width > 30:
            continue

        path_sample = _sample_glyphs(rows, cols, frame_width)
        if path_sample is None:
            continue
        glyphs, visited, _wrapped = path_sample

        hint_mask = _sample_hints(rows, cols, visited)
        if hint_mask is None:
            continue

        row_colors = random.sample(PATH_COLORS, rows)
        lower, row_lines, col_lines = _render_lattice(row_heights, col_widths, row_colors, hint_mask)
        expected = _render_expected(lower, row_lines, col_lines, row_colors, visited)
        strip = _encode_strip(glyphs, lower.shape[1])
        inp = np.vstack([strip, lower])

        if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if not _validate_semantics(inp, expected, glyphs, visited, hint_mask, row_colors):
            continue
        return inp
    raise RuntimeError("failed to generate valid input")
