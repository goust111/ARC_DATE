from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}
MAZE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_COMBOS = (
    ("left", "right"),
    ("left", "bottom"),
    ("top", "right"),
    ("top", "bottom"),
)
MIN_TURNS = 1
MAX_TURNS = 7
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_side(r: int, c: int, h: int, w: int) -> str:
    sides = []
    if c == 0:
        sides.append("left")
    if r == 0:
        sides.append("top")
    if c == w - 1:
        sides.append("right")
    if r == h - 1:
        sides.append("bottom")
    assert len(sides) == 1, "openings must not sit on corners"
    return sides[0]


def _trace_unique_path(sub: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = sub.shape
    path_mask = sub == BG
    coords = [(int(r), int(c)) for r, c in zip(*np.where(path_mask), strict=False)]
    degs: dict[tuple[int, int], int] = {}
    for r, c in coords:
        degs[(r, c)] = sum(
            1 for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc]
        )

    assert start in degs and end in degs
    for cell, degree in degs.items():
        if cell in (start, end):
            assert degree == 1, "path endpoints must have degree 1"
        else:
            assert degree == 2, "maze path must be a single simple corridor"

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        nxts = []
        for nr, nc in _neighbors4(*cur):
            if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc] and (nr, nc) != prev:
                nxts.append((nr, nc))
        assert len(nxts) == 1, "path must have exactly one continuation"
        prev, cur = cur, nxts[0]
        path.append(cur)

    assert len(path) == len(coords), "all black cells inside the maze must belong to the corridor"
    return path


def _turn_colors(path: list[tuple[int, int]], maze_color: int) -> list[int]:
    seq: list[int] = []
    for i in range(1, len(path) - 1):
        prev_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_dir == next_dir:
            continue
        prev_idx = DIR_TO_INDEX[prev_dir]
        next_idx = DIR_TO_INDEX[next_dir]
        if next_idx == (prev_idx + 1) % 4:
            seq.append(GREY)
        elif next_idx == (prev_idx - 1) % 4:
            seq.append(maze_color)
        else:
            raise AssertionError("corridor may only turn left or right")
    return seq


def _maze_sequences(g: np.ndarray) -> tuple[int, list[list[int]]]:
    _h, w = g.shape
    split = w // 2
    left = g[:, :split]
    mazes = []
    colors = [int(color) for color in np.unique(left) if int(color) not in (BG, GREY)]
    for maze_color in colors:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(left == maze_color), strict=False)]
        assert pts, "maze color must appear in the left half"
        r0, c0, r1, c1 = _bbox(pts)
        sub = left[r0 : r1 + 1, c0 : c1 + 1]
        openings: list[tuple[int, int, str]] = []
        sh, sw = sub.shape
        for rr in range(sh):
            for cc in range(sw):
                if rr not in (0, sh - 1) and cc not in (0, sw - 1):
                    continue
                if int(sub[rr, cc]) == BG:
                    openings.append((rr, cc, _border_side(rr, cc, sh, sw)))
        assert len(openings) == 2, "every maze must have exactly two openings"
        starts = [(r, c) for r, c, side in openings if side in ("left", "top")]
        ends = [(r, c) for r, c, side in openings if side in ("right", "bottom")]
        assert len(starts) == 1 and len(ends) == 1
        path = _trace_unique_path(sub, starts[0], ends[0])
        mazes.append((r0, c0, _turn_colors(path, maze_color)))

    mazes.sort(key=lambda item: (item[0], item[1]))
    return split, [seq for _, _, seq in mazes]


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each left-half maze's unique corridor and write its turn code into the blank right half.

    Input structure:
    - The background is black.
    - The left half contains 2-4 small monochrome maze components, each with a colored wall and a 1-cell-wide black corridor.
    - Each maze has exactly two border openings; the entrance is on the left or top edge, and the exit is on the right or bottom edge.
    - The right half is blank black space where the answer is written one row per maze, in top-to-bottom maze order.

    Rule:
    - Find every maze in the left half and sort them by their vertical position.
    - Inside each maze bounding box, follow the unique black corridor from the left/top opening to the right/bottom opening.
    - Record only 90-degree turns: a left turn becomes one pixel of the maze's own wall color, and a right turn becomes one grey pixel.
    - Keep the mazes unchanged and draw each maze's turn sequence from left to right in the corresponding top row of the right half.

    Key insight:
    - The colored walls identify separate maze objects, but the information to encode lives in the ordered turn directions of the black corridor once the entrance/exit orientation is fixed.
    """
    g = np.asarray(grid, dtype=int)
    split, rows = _maze_sequences(g)
    out = g.copy()
    out[:, split:] = BG
    for row_idx, seq in enumerate(rows):
        assert len(seq) <= out.shape[1] - split
        out[row_idx, split : split + len(seq)] = seq
    return out


def _on_border(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _side_cells(h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    raise AssertionError("unknown side")


def _turn_letters(path: list[tuple[int, int]]) -> list[str]:
    letters: list[str] = []
    for i in range(1, len(path) - 1):
        prev_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_dir == next_dir:
            continue
        prev_idx = DIR_TO_INDEX[prev_dir]
        next_idx = DIR_TO_INDEX[next_dir]
        if next_idx == (prev_idx + 1) % 4:
            letters.append("R")
        elif next_idx == (prev_idx - 1) % 4:
            letters.append("L")
        else:
            raise AssertionError("generator built a non-orthogonal turn")
    return letters


def _reachable(h: int, w: int, cur: tuple[int, int], end: tuple[int, int], visited: set[tuple[int, int]]) -> bool:
    q = deque([cur])
    seen = {cur}
    blocked = visited - {cur}
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            return True
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in blocked or nxt in seen:
                continue
            if _on_border(nxt, h, w) and nxt != end:
                continue
            seen.add(nxt)
            q.append(nxt)
    return False


def _valid_next(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    end: tuple[int, int],
    visited: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    nr, nc = nxt
    if not (0 <= nr < h and 0 <= nc < w):
        return False
    if nxt in visited:
        return False
    if _on_border(nxt, h, w) and nxt != end:
        return False
    for ar, ac in _neighbors4(nr, nc):
        if (ar, ac) in visited and (ar, ac) != cur:
            return False
    return True


def _search_path(
    h: int,
    w: int,
    start: tuple[int, int],
    end: tuple[int, int],
    min_turns: int,
    max_turns: int,
) -> list[tuple[int, int]] | None:
    path = [start]
    visited = {start}
    max_cells = min(h * w, 18)

    def dfs(cur: tuple[int, int], last_dir: tuple[int, int] | None, turn_count: int) -> list[tuple[int, int]] | None:
        if turn_count > max_turns or len(path) > max_cells:
            return None
        if cur == end:
            if turn_count >= min_turns:
                return list(path)
            return None
        if not _reachable(h, w, cur, end, visited):
            return None

        candidates = []
        for nr, nc in _neighbors4(*cur):
            nxt = (nr, nc)
            if not _valid_next(nxt, cur, end, visited, h, w):
                continue
            step = (nr - cur[0], nc - cur[1])
            next_turns = turn_count + int(last_dir is not None and step != last_dir)
            if next_turns > max_turns:
                continue
            dist = abs(end[0] - nr) + abs(end[1] - nc)
            bias = random.random()
            candidates.append((dist, bias, nxt, step, next_turns))

        candidates.sort(key=lambda item: (item[0] + item[1]))
        if random.random() < 0.6:
            random.shuffle(candidates)

        for _dist, _bias, nxt, step, next_turns in candidates:
            visited.add(nxt)
            path.append(nxt)
            result = dfs(nxt, step, next_turns)
            if result is not None:
                return result
            path.pop()
            visited.remove(nxt)
        return None

    return dfs(start, None, 0)


def _build_maze(
    h: int,
    w: int,
    color: int,
    start_side: str,
    end_side: str,
) -> tuple[np.ndarray, list[int]] | None:
    starts = _side_cells(h, w, start_side)
    ends = _side_cells(h, w, end_side)
    choices = [(s, e) for s in starts for e in ends if s != e]
    random.shuffle(choices)
    for start, end in choices:
        path = _search_path(h, w, start, end, MIN_TURNS, MAX_TURNS)
        if path is None:
            continue
        letters = _turn_letters(path)
        if not letters:
            continue
        maze = np.full((h, w), color, dtype=int)
        for r, c in path:
            maze[r, c] = BG
        seq = [color if ch == "L" else GREY for ch in letters]
        return maze, seq
    return None


def _sample_heights(count: int) -> tuple[list[int], list[int], int] | None:
    for _ in range(120):
        heights = [random.randint(5, 7) for _ in range(count)]
        gaps = [random.randint(1, 2) for _ in range(count - 1)]
        top_margin = random.randint(0, 1)
        bottom_margin = random.randint(0, 1)
        total_h = top_margin + sum(heights) + sum(gaps) + bottom_margin
        if total_h <= 30:
            return heights, gaps, top_margin
    return None


def _render_expected(inp: np.ndarray, split: int, rows: list[list[int]]) -> np.ndarray:
    out = inp.copy()
    out[:, split:] = BG
    for row_idx, seq in enumerate(rows):
        out[row_idx, split : split + len(seq)] = seq
    return out


def _gen_attempt() -> np.ndarray | None:
    maze_count = random.randint(2, 4)
    height_sample = _sample_heights(maze_count)
    if height_sample is None:
        return None
    heights, gaps, top_margin = height_sample
    widths = [random.randint(5, 7) for _ in range(maze_count)]
    colors = random.sample(MAZE_COLORS, maze_count)
    side_combos = random.sample(list(SIDE_COMBOS), maze_count)

    mazes: list[np.ndarray] = []
    row_codes: list[list[int]] = []
    for (h, w, color, (start_side, end_side)) in zip(heights, widths, colors, side_combos, strict=False):
        built = _build_maze(h, w, color, start_side, end_side)
        if built is None:
            return None
        maze, seq = built
        mazes.append(maze)
        row_codes.append(seq)

    all_codes = [code for seq in row_codes for code in seq]
    if GREY not in all_codes:
        return None
    if not any(code in colors for code in all_codes):
        return None

    half_w_min = max(max(widths) + 3, max(len(seq) for seq in row_codes) + 1, 10)
    if half_w_min > 15:
        return None
    half_w = random.randint(half_w_min, 15)
    grid_h = top_margin + sum(heights) + sum(gaps)
    grid = np.full((grid_h, 2 * half_w), BG, dtype=int)

    current_r = top_margin
    placements = []
    for maze, gap in zip(mazes, gaps + [0], strict=False):
        mh, mw = maze.shape
        c0 = random.randint(1, half_w - mw - 1)
        grid[current_r : current_r + mh, c0 : c0 + mw] = maze
        placements.append((current_r, c0))
        current_r += mh + gap

    expected = _render_expected(grid, half_w, row_codes)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-turn ARC sample")
