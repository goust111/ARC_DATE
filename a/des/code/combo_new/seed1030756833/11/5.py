from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _find_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected: set[tuple[int, int]] = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        if set(pts) != expected:
            continue
        if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG):
            continue
        frames.append((r0, c0, r1, c1))
    return sorted(frames)


def _frame_line_start(g: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[int, int]:
    h, w = g.shape
    r0, c0, r1, c1 = frame
    starts: set[tuple[int, int]] = set()
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for r, c in border:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                continue
            val = int(g[nr, nc])
            if val not in (BG, GREY):
                starts.add((nr, nc))
    assert len(starts) == 1, f"expected one guide-line endpoint at frame, got {starts}"
    return next(iter(starts))


def _is_stamp_patch(patch: np.ndarray, body: int) -> bool:
    if patch.size == 0 or np.any((patch == BG) | (patch == GREY)):
        return False
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    perimeter = np.zeros(patch.shape, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    if not np.all(patch[perimeter] == body):
        return False
    return bool(np.any(patch[~perimeter] != body))


def _adjacent_to_rect(cell: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = rect
    for nr, nc in _neighbors4(r, c):
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            return True
    return False


def _same_color_path_to_rect(
    g: np.ndarray,
    start: tuple[int, int],
    rect: tuple[int, int, int, int],
    color: int,
) -> list[tuple[int, int]] | None:
    h, w = g.shape
    r0, c0, r1, c1 = rect
    blocked_rect = np.zeros(g.shape, dtype=bool)
    blocked_rect[r0 : r1 + 1, c0 : c1 + 1] = True
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cell = q.popleft()
        if _adjacent_to_rect(cell, rect):
            path: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = cell
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in prev or blocked_rect[nr, nc] or int(g[nr, nc]) != color:
                continue
            prev[nxt] = cell
            q.append(nxt)
    return None


def _count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    steps = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    return sum(1 for a, b in zip(steps, steps[1:]) if a != b)


def _find_stamp_and_path(
    g: np.ndarray,
    frame: tuple[int, int, int, int],
    start: tuple[int, int],
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], int]:
    body = int(g[start])
    r0, c0, r1, c1 = frame
    sh, sw = r1 - r0 - 1, c1 - c0 - 1
    h, w = g.shape
    candidates: list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]] = []

    for sr in range(0, h - sh + 1):
        for sc in range(0, w - sw + 1):
            rect = (sr, sc, sr + sh - 1, sc + sw - 1)
            # A stamp cannot be a target interior or overlap a grey frame.
            if _is_stamp_patch(g[sr : sr + sh, sc : sc + sw], body):
                path = _same_color_path_to_rect(g, start, rect, body)
                if path is not None:
                    candidates.append((len(path), rect, path))

    assert candidates, f"no stamp found for guide color {body}"
    candidates.sort(key=lambda item: item[0])
    _, rect, path = candidates[0]
    return rect, path, _count_turns(path)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each stamp according to the turn count of its matching guide line, then stamp it into the matching grey frame.

    Input structure:
    - The background is black and each target is an empty one-cell-thick grey rectangular frame.
    - Outside the targets are small multicolor rectangular stamps: the stamp perimeter is its body color and its interior contains colored detail pixels.
    - A one-cell-wide 4-connected guide line of the same body color touches one side of each stamp and one side of one target frame.

    Rule:
    - For each grey target frame, follow the same-colored guide line from the frame to its stamp and count the number of 90-degree turns.
    - Rotate that whole stamp pattern clockwise by turn_count * 90 degrees.
    - Clear the original stamp and guide line to black, then place the rotated stamp in the target interior with top-left alignment while keeping the grey frame.

    Key insight: The guide line gives both the stamp-to-frame pairing and, through its number of bends, the rotation to apply before stamping.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one grey target frame"

    for frame in frames:
        start = _frame_line_start(g, frame)
        rect, path, turns = _find_stamp_and_path(g, frame, start)
        sr0, sc0, sr1, sc1 = rect
        stamp = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        rotated = np.rot90(stamp, (-turns) % 4)

        fr0, fc0, fr1, fc1 = frame
        ih, iw = fr1 - fr0 - 1, fc1 - fc0 - 1
        assert rotated.shape == (ih, iw), f"rotated stamp shape {rotated.shape} does not fit target {(ih, iw)}"

        for r, c in path:
            out[r, c] = BG
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = rotated

    return out


def _pad_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _rect_cells(r0: int, c0: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)}


def _rect_in_bounds(r0: int, c0: int, height: int, width: int, h: int, w: int) -> bool:
    return 0 <= r0 and 0 <= c0 and r0 + height <= h and c0 + width <= w


def _turn_dir(direction: tuple[int, int], right: bool) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr) if right else (-dc, dr)


def _random_direction_sequence(turns: int, rng: random.Random) -> list[tuple[int, int]]:
    seq = [rng.choice(DIRS)]
    for _ in range(turns):
        seq.append(_turn_dir(seq[-1], rng.choice([False, True])))
    return seq


def _build_path_from_segments(
    start: tuple[int, int],
    dirs: list[tuple[int, int]],
    lengths: list[int],
) -> list[tuple[int, int]]:
    r, c = start
    path = [(r, c)]
    for (dr, dc), length in zip(dirs, lengths):
        for _ in range(length):
            r += dr
            c += dc
            path.append((r, c))
    return path


def _frame_rect_from_path_start(
    start: tuple[int, int],
    direction: tuple[int, int],
    stamp_size: int,
    offset: int,
) -> tuple[int, int, int, int]:
    frame_size = stamp_size + 2
    br, bc = start[0] - direction[0], start[1] - direction[1]
    if direction == (0, 1):
        return br - offset, bc - frame_size + 1, frame_size, frame_size
    if direction == (0, -1):
        return br - offset, bc, frame_size, frame_size
    if direction == (1, 0):
        return br - frame_size + 1, bc - offset, frame_size, frame_size
    return br, bc - offset, frame_size, frame_size


def _stamp_rect_from_path_end(
    end: tuple[int, int],
    direction: tuple[int, int],
    stamp_size: int,
    offset: int,
) -> tuple[int, int, int, int]:
    sr, sc = end[0] + direction[0], end[1] + direction[1]
    if direction == (0, 1):
        return sr - offset, sc, stamp_size, stamp_size
    if direction == (0, -1):
        return sr - offset, sc - stamp_size + 1, stamp_size, stamp_size
    if direction == (1, 0):
        return sr, sc - offset, stamp_size, stamp_size
    return sr - stamp_size + 1, sc - offset, stamp_size, stamp_size


def _path_touches_rect(
    path: list[tuple[int, int]],
    rect: tuple[int, int, int, int],
    allowed: set[tuple[int, int]],
) -> bool:
    r0, c0, height, width = rect
    r1, c1 = r0 + height - 1, c0 + width - 1
    for cell in path:
        if cell in allowed:
            continue
        for nr, nc in _neighbors4(*cell):
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                return True
    return False


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int) -> None:
    grid[r0, c0 : c0 + size] = GREY
    grid[r0 + size - 1, c0 : c0 + size] = GREY
    grid[r0 : r0 + size, c0] = GREY
    grid[r0 : r0 + size, c0 + size - 1] = GREY


def _make_stamp(size: int, body: int, rng: random.Random) -> np.ndarray:
    detail_pool = [c for c in COLOR_POOL if c != body]
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(100):
        stamp = np.full((size, size), body, dtype=int)
        detail_count = rng.randint(2, min(4, len(interior)))
        for r, c in rng.sample(interior, detail_count):
            stamp[r, c] = rng.choice(detail_pool)
        if all(not np.array_equal(stamp, np.rot90(stamp, k)) for k in (1, 2, 3)):
            return stamp
    raise RuntimeError("failed to make asymmetric stamp")


def _try_place_pair(
    grid: np.ndarray,
    expected: np.ndarray,
    actual: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    body: int,
    turns: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    stamp_size = rng.choice([4, 5])
    frame_size = stamp_size + 2
    stamp = _make_stamp(stamp_size, body, rng)

    for _ in range(500):
        dirs = _random_direction_sequence(turns, rng)
        lengths = [rng.randint(3, 8) for _ in dirs]
        start = (rng.randrange(2, h - 2), rng.randrange(2, w - 2))
        path = _build_path_from_segments(start, dirs, lengths)
        if len(set(path)) != len(path):
            continue
        if any(not (0 <= r < h and 0 <= c < w) for r, c in path):
            continue
        path_set = set(path)
        if path_set & blocked:
            continue

        first_dir, last_dir = dirs[0], dirs[-1]
        frame_offsets = list(range(1, frame_size - 1))
        stamp_offsets = list(range(1, stamp_size - 1))
        rng.shuffle(frame_offsets)
        rng.shuffle(stamp_offsets)

        for foff in frame_offsets:
            fr0, fc0, fh, fw = _frame_rect_from_path_start(start, first_dir, stamp_size, foff)
            if not _rect_in_bounds(fr0, fc0, fh, fw, h, w):
                continue
            frame_cells = _rect_cells(fr0, fc0, fh, fw)
            if frame_cells & (actual | path_set | blocked):
                continue
            if _path_touches_rect(path[1:], (fr0, fc0, fh, fw), {start}):
                continue

            for soff in stamp_offsets:
                sr0, sc0, sh, sw = _stamp_rect_from_path_end(path[-1], last_dir, stamp_size, soff)
                if not _rect_in_bounds(sr0, sc0, sh, sw, h, w):
                    continue
                stamp_cells = _rect_cells(sr0, sc0, sh, sw)
                if stamp_cells & (actual | path_set | blocked | frame_cells):
                    continue
                if _pad_cells(frame_cells, h, w) & stamp_cells:
                    continue
                if _path_touches_rect(path[:-1], (sr0, sc0, sh, sw), {path[-1]}):
                    continue

                _draw_frame(grid, fr0, fc0, frame_size)
                _draw_frame(expected, fr0, fc0, frame_size)
                grid[sr0 : sr0 + stamp_size, sc0 : sc0 + stamp_size] = stamp
                for r, c in path:
                    grid[r, c] = body

                rotated = np.rot90(stamp, (-turns) % 4)
                expected[fr0 + 1 : fr0 + frame_size - 1, fc0 + 1 : fc0 + frame_size - 1] = rotated

                new_actual = frame_cells | stamp_cells | path_set
                actual |= new_actual
                blocked |= _pad_cells(new_actual, h, w)
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    count = rng.choices([2, 3, 4], weights=[2, 3, 3])[0]
    body_colors = rng.sample(COLOR_POOL, count)
    if count == 4:
        turn_counts = [0, 1, 2, 3]
    else:
        turn_counts = rng.sample([0, 1, 2, 3], count)
    rng.shuffle(turn_counts)

    actual: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    for body, turns in zip(body_colors, turn_counts):
        if not _try_place_pair(grid, expected, actual, blocked, body, turns, rng):
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed to generate a valid stamp-guide input")
