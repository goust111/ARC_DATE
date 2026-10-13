from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
BORDER = 5
FRAME_SIZE = 4
INNER_COLORS = (1, 2, 3, 4, 6)
MAX_GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Frame:
    idx: int
    r0: int
    c0: int
    color: int

    @property
    def r1(self) -> int:
        return self.r0 + FRAME_SIZE - 1

    @property
    def c1(self) -> int:
        return self.c0 + FRAME_SIZE - 1


def _find_frames(grid: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    seen: set[tuple[int, int]] = set()
    h, w = grid.shape
    for r in range(h - FRAME_SIZE + 1):
        for c in range(w - FRAME_SIZE + 1):
            block = grid[r : r + FRAME_SIZE, c : c + FRAME_SIZE]
            if not np.all(block[0, :] == BORDER):
                continue
            if not np.all(block[-1, :] == BORDER):
                continue
            if not np.all(block[:, 0] == BORDER):
                continue
            if not np.all(block[:, -1] == BORDER):
                continue
            inner = block[1:3, 1:3]
            color = int(inner[0, 0])
            if color in (BG, BORDER):
                continue
            if not np.all(inner == color):
                continue
            if (r, c) in seen:
                continue
            seen.add((r, c))
            frames.append(Frame(len(frames), int(r), int(c), color))
    return frames


def _control_sequence(grid: np.ndarray, frames: list[Frame]) -> list[int]:
    lowest_frame_row = max(frame.r1 for frame in frames)
    ctrl_rows = [r for r in range(lowest_frame_row + 1, grid.shape[0]) if np.any(grid[r] != BG)]
    assert ctrl_rows, "expected a bottom control row"
    ctrl_r = max(ctrl_rows)
    seq = [int(grid[ctrl_r, c]) for c in range(grid.shape[1]) if int(grid[ctrl_r, c]) != BG]
    assert len(seq) >= 2, "control row must contain at least two colors"
    return seq


def _orientation(a: Frame, b: Frame) -> str | None:
    if a.idx == b.idx:
        return None
    if a.r0 == b.r0:
        return "h"
    if a.c0 == b.c0:
        return "v"
    return None


def _clear_corridor(a: Frame, b: Frame, frames: list[Frame]) -> bool:
    orient = _orientation(a, b)
    if orient == "h":
        left_c = min(a.c0, b.c0)
        right_c = max(a.c0, b.c0)
        for other in frames:
            if other.idx in (a.idx, b.idx):
                continue
            if other.r0 == a.r0 and left_c < other.c0 < right_c:
                return False
        return True
    if orient == "v":
        top_r = min(a.r0, b.r0)
        bot_r = max(a.r0, b.r0)
        for other in frames:
            if other.idx in (a.idx, b.idx):
                continue
            if other.c0 == a.c0 and top_r < other.r0 < bot_r:
                return False
        return True
    return False


def _axis_distance(a: Frame, b: Frame) -> int:
    orient = _orientation(a, b)
    assert orient is not None
    return abs(a.c0 - b.c0) if orient == "h" else abs(a.r0 - b.r0)


def _candidate_targets(current: Frame, target_color: int, frames: list[Frame]) -> list[Frame]:
    candidates = []
    for frame in frames:
        if frame.color != target_color or frame.idx == current.idx:
            continue
        if _orientation(current, frame) is None:
            continue
        if not _clear_corridor(current, frame, frames):
            continue
        candidates.append(frame)
    return candidates


def _select_target(current: Frame, target_color: int, frames: list[Frame]) -> Frame:
    candidates = _candidate_targets(current, target_color, frames)
    assert candidates, f"no valid target for color {target_color}"
    best_dist = min(_axis_distance(current, cand) for cand in candidates)
    best = [cand for cand in candidates if _axis_distance(current, cand) == best_dist]
    assert len(best) == 1, "nearest aligned target must be unique"
    return best[0]


def _resolve_start(frames: list[Frame], seq: list[int]) -> Frame:
    starts = [frame for frame in frames if frame.color == seq[0]]
    assert starts, "first control color must match a frame"
    if len(starts) == 1:
        return starts[0]
    assert len(seq) >= 2
    viable = []
    for frame in starts:
        try:
            nxt = _select_target(frame, seq[1], frames)
        except AssertionError:
            continue
        viable.append((int(_axis_distance(frame, nxt)), frame))
    assert viable, "could not resolve start frame"
    best_dist = min(dist for dist, _frame in viable)
    best = [frame for dist, frame in viable if dist == best_dist]
    assert len(best) == 1, "start frame must be uniquely determined"
    return best[0]


def _paint_bridge_region(out: np.ndarray, a: Frame, b: Frame, palette: tuple[int, int]) -> None:
    orient = _orientation(a, b)
    assert orient is not None
    if orient == "h":
        left, right = (a, b) if a.c0 < b.c0 else (b, a)
        r0, r1 = left.r0, left.r1
        c0, c1 = left.c1 + 1, right.c0 - 1
    else:
        top, bot = (a, b) if a.r0 < b.r0 else (b, a)
        r0, r1 = top.r1 + 1, bot.r0 - 1
        c0, c1 = top.c0, top.c1
    assert r0 <= r1 and c0 <= c1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            d = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = palette[d % 2]


def _solve(grid: np.ndarray) -> np.ndarray:
    frames = _find_frames(grid)
    assert 4 <= len(frames) <= 5, "expected 4-5 framed squares"
    seq = _control_sequence(grid, frames)
    cur = _resolve_start(frames, seq)
    out = np.array(grid, dtype=int, copy=True)
    for i in range(len(seq) - 1):
        nxt = _select_target(cur, seq[i + 1], frames)
        _paint_bridge_region(out, cur, nxt, (seq[i], seq[i + 1]))
        cur = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bridge aligned framed squares and fill each bridge cavity with two-color concentric bands.

    Input structure:
    - The background is black.
    - The main area contains 4x4 grey framed squares whose 2x2 interiors are solid colors.
    - The bottom row contains a left-to-right control sequence made of those interior colors.

    Rule:
    - Start from the square matching the first control color.
    - For each consecutive pair (ci, ci+1), among ci+1-colored squares choose the uniquely nearest one that is aligned with the current square in the same row or column and has a clear corridor.
    - Between the two squares, create the rectangular bridge region spanning the gap; its boundary band therefore uses ci.
    - Fill every bridge region cell by Chebyshev distance to that region's boundary, using the alternating palette [ci, ci+1].
    - Leave the framed squares and the control row unchanged.

    Key insight:
    The control row determines both the visitation path and, for each visited gap, the two-color palette that drives the concentric bridge fill.
    """
    return _solve(np.asarray(grid, dtype=int))


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    grid[frame.r0 : frame.r1 + 1, frame.c0 : frame.c1 + 1] = BORDER
    grid[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] = frame.color


def _render_input(frames: list[Frame], control: list[int], height: int, width: int, ctrl_left: int, ctrl_step: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)
    row = height - 1
    for i, color in enumerate(control):
        grid[row, ctrl_left + i * ctrl_step] = color
    return grid


def _transform_dihedral(coords: list[tuple[int, int]], height: int, width: int, allow_rotate90: bool) -> tuple[list[tuple[int, int]], int, int]:
    rot_choices = (0, 1, 2, 3) if allow_rotate90 else (0, 2)
    rot = random.choice(rot_choices)
    cur = [(int(r), int(c)) for r, c in coords]
    cur_h, cur_w = int(height), int(width)
    for _ in range(rot):
        cur = [(c, cur_h - 1 - r) for r, c in cur]
        cur_h, cur_w = cur_w, cur_h
    if random.choice((False, True)):
        cur = [(r, cur_w - 1 - c) for r, c in cur]
    return cur, cur_h, cur_w


def _layout_branch() -> tuple[list[tuple[int, int]], list[int], int, int, bool]:
    if random.random() < 0.65:
        colors = random.sample(list(INNER_COLORS), 4)
        control = [colors[0], colors[1], colors[2], colors[3], colors[1]]
        path = [(0, 1), (0, 0), (1, 0), (1, 3), (0, 3)]
        return path, control, 2, 4, True
    colors = random.sample(list(INNER_COLORS), 4)
    control = colors[:]
    path = [(0, 0), (0, 1), (1, 1), (1, 2)]
    return path, control, 2, 3, False


def _make_frames(path: list[tuple[int, int]], control: list[int], box_h: int, box_w: int, duplicate_branch: bool) -> tuple[list[Frame], list[int], int, int]:
    allow_rotate90 = not duplicate_branch
    t_path, rows_count, cols_count = _transform_dihedral(path, box_h, box_w, allow_rotate90)
    gap_r = random.randint(4, 7)
    gap_c = 4 if duplicate_branch else random.randint(4, 7)
    box_pix_h = rows_count * FRAME_SIZE + (rows_count - 1) * gap_r
    box_pix_w = cols_count * FRAME_SIZE + (cols_count - 1) * gap_c

    ctrl_gap = random.randint(2, 4)
    top_min = 1
    if top_min + box_pix_h + ctrl_gap + 1 > 30:
        raise ValueError("layout too tall")
    top = random.randint(top_min, min(3, 30 - (box_pix_h + ctrl_gap + 1)))
    height = top + box_pix_h + ctrl_gap + 1

    ctrl_step = random.randint(2, 4)
    ctrl_width = 1 + (len(control) - 1) * ctrl_step
    min_width = max(box_pix_w + 2, ctrl_width + 2)
    if min_width > 30:
        raise ValueError("layout too wide")
    width = random.randint(min_width, min(30, min_width + 4))
    frame_left = random.randint(1, width - box_pix_w - 1)
    ctrl_left = random.randint(1, width - ctrl_width - 1)

    frames = []
    for idx, ((lr, lc), color) in enumerate(zip(t_path, control)):
        if duplicate_branch and idx == len(control) - 1:
            color = control[1]
        r0 = top + lr * (FRAME_SIZE + gap_r)
        c0 = frame_left + lc * (FRAME_SIZE + gap_c)
        frames.append(Frame(idx, int(r0), int(c0), int(color)))
    return frames, control, height, width, ctrl_left, ctrl_step


def _expected_output(inp: np.ndarray, frames: list[Frame], control: list[int]) -> np.ndarray:
    out = inp.copy()
    for a, b in zip(frames, frames[1:]):
        _paint_bridge_region(out, a, b, (a.color, b.color))
    return out


def _validate_layout(inp: np.ndarray, out: np.ndarray, frames: list[Frame], control: list[int], duplicate_branch: bool) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(inp, out):
        return False
    if len(_find_frames(inp)) != len(frames):
        return False
    parsed_frames = sorted(_find_frames(inp), key=lambda f: f.idx)
    seq = _control_sequence(inp, parsed_frames)
    if seq != control:
        return False
    current = _resolve_start(parsed_frames, seq)
    if current.color != control[0]:
        return False

    used_second_band = 0
    multi_candidate_steps = 0
    for step, intended_next in enumerate(frames[1:], start=1):
        candidates = _candidate_targets(current, control[step], parsed_frames)
        if len(candidates) > 1:
            multi_candidate_steps += 1
        chosen = _select_target(current, control[step], parsed_frames)
        if chosen.r0 != intended_next.r0 or chosen.c0 != intended_next.c0 or chosen.color != intended_next.color:
            return False
        probe = np.zeros_like(inp)
        _paint_bridge_region(probe, current, chosen, (current.color, chosen.color))
        region_vals = probe[probe != 0]
        if len(set(map(int, region_vals.tolist()))) < 2:
            return False
        used_second_band += int(np.count_nonzero(probe == chosen.color))
        current = chosen
    if used_second_band == 0:
        return False
    if duplicate_branch and multi_candidate_steps == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        path, control, box_h, box_w, duplicate_branch = _layout_branch()
        frames, control, height, width, ctrl_left, ctrl_step = _make_frames(path, control, box_h, box_w, duplicate_branch)
    except ValueError:
        return None

    inp = _render_input(frames, control, height, width, ctrl_left, ctrl_step)
    out = _expected_output(inp, frames, control)
    if _validate_layout(inp, out, frames, control, duplicate_branch):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
