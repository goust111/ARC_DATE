from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
SCALE = 2
SEQ_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 500


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _parse_sequence(strip: np.ndarray, bg: int) -> tuple[tuple[int, ...], int] | None:
    non_bg = [i for i, value in enumerate(strip) if int(value) != bg]
    if len(non_bg) not in (2, 3):
        return None
    start = non_bg[0]
    if non_bg != list(range(start, start + len(non_bg))):
        return None
    seq = tuple(int(v) for v in strip[start : start + len(non_bg)])
    if any(v in (bg, GREY) for v in seq):
        return None
    if len(set(seq)) != len(seq):
        return None
    if np.any(strip[:start] != bg) or np.any(strip[start + len(non_bg) :] != bg):
        return None
    return seq, start


def _parse_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    h, w = g.shape
    frames = []

    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        ph, pw = patch.shape
        if ph < 6 or pw < 5:
            raise AssertionError("grey component too small to be a capped frame")

        top = patch[0]
        left_cap = np.all(top[:2] == GREY) and np.all(top[2:] == bg)
        right_cap = np.all(top[:-2] == bg) and np.all(top[-2:] == GREY)
        if left_cap == right_cap:
            raise AssertionError("frame cap must appear on exactly one side")

        expected = np.full_like(patch, bg)
        expected[1, :] = GREY
        expected[-1, :] = GREY
        expected[2:-1, 0] = GREY
        expected[2:-1, -1] = GREY
        if left_cap:
            expected[0, :2] = GREY
            direction = "right"
        else:
            expected[0, -2:] = GREY
            direction = "left"
        if not np.array_equal(patch, expected):
            raise AssertionError("grey component is not a valid capped hollow frame")

        if r1 + 1 >= h:
            raise AssertionError("frame sequence row leaves the grid")
        seq_info = _parse_sequence(g[r1 + 1, c0 + 1 : c1], bg)
        if seq_info is None:
            raise AssertionError("frame is missing a valid bottom color sequence")
        seq, seq_offset = seq_info

        frames.append(
            {
                "top": r0 + 1,
                "left": c0,
                "bottom": r1,
                "right": c1,
                "direction": direction,
                "seq": seq,
                "seq_offset": seq_offset,
            }
        )

    frames.sort(key=lambda frame: (frame["top"], frame["left"]))
    if not 1 <= len(frames) <= 2:
        raise AssertionError("expected 1 or 2 frames")
    return frames


def _filled_grid(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        top = frame["top"]
        left = frame["left"]
        bottom = frame["bottom"]
        right = frame["right"]
        seq = frame["seq"]
        inner_w = right - left - 1
        for offset in range(inner_w):
            if frame["direction"] == "right":
                color = seq[offset % len(seq)]
            else:
                color = seq[(inner_w - 1 - offset) % len(seq)]
            out[top + 1 : bottom, left + 1 + offset] = color
    return out


def _scale2(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, SCALE, axis=0), SCALE, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each capped grey frame by repeating its bottom color sequence from the cap-indicated side, then scale the whole picture by 2.

    Input structure:
    - The background is black.
    - There are 1 or 2 grey hollow rectangular frames whose interiors are entirely background.
    - Each frame has a 2-pixel grey cap just above the left or right side of its top edge.
    - Each frame also has a contiguous row of 2 or 3 distinct colored pixels directly below the frame bottom, giving a color sequence.

    Rule:
    - Read each frame independently.
    - A left cap means fill the interior by vertical stripes progressing to the right; a right cap means start from the interior's rightmost column and progress left.
    - The stripe colors repeat the bottom sequence periodically, one full column per sequence element.
    - After all frame interiors are filled, enlarge the entire grid by replacing every pixel with a 2x2 block of the same color.

    Key insight:
    - The cap does not add new content by itself; it specifies which interior edge receives the first color of the repeated bottom sequence, so the same sequence mirrors when the cap moves to the opposite side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _parse_frames(g)
    return _scale2(_filled_grid(g, frames))


def _frame_cells(frame: dict) -> set[tuple[int, int]]:
    top = frame["top"]
    left = frame["left"]
    bottom = frame["bottom"]
    right = frame["right"]
    cells: set[tuple[int, int]] = set()

    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top + 1, bottom):
        cells.add((r, left))
        cells.add((r, right))

    if frame["direction"] == "right":
        cells.add((top - 1, left))
        cells.add((top - 1, left + 1))
    else:
        cells.add((top - 1, right - 1))
        cells.add((top - 1, right))

    seq_row = bottom + 1
    seq_offset = frame["seq_offset"]
    for idx in range(len(frame["seq"])):
        cells.add((seq_row, left + 1 + seq_offset + idx))
    return cells


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    top = frame["top"]
    left = frame["left"]
    bottom = frame["bottom"]
    right = frame["right"]

    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top + 1 : bottom, left] = GREY
    grid[top + 1 : bottom, right] = GREY

    if frame["direction"] == "right":
        grid[top - 1, left : left + 2] = GREY
    else:
        grid[top - 1, right - 1 : right + 1] = GREY

    seq_row = bottom + 1
    seq_offset = frame["seq_offset"]
    for idx, color in enumerate(frame["seq"]):
        grid[seq_row, left + 1 + seq_offset + idx] = int(color)


def _candidate_positions(frame: dict, shape: tuple[int, int], occupied: set[tuple[int, int]]) -> list[dict]:
    h, w = shape
    outer_h = frame["outer_h"]
    outer_w = frame["outer_w"]
    candidates = []
    for top in range(1, h - outer_h):
        bottom = top + outer_h - 1
        if bottom + 1 >= h:
            continue
        for left in range(0, w - outer_w + 1):
            placed = dict(frame)
            placed["top"] = top
            placed["left"] = left
            placed["bottom"] = bottom
            placed["right"] = left + outer_w - 1
            cells = _frame_cells(placed)
            ok = True
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (r + dr, c + dc) in occupied:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                candidates.append(placed)
    return candidates


def _planned_specs(frame_count: int) -> list[dict]:
    if frame_count == 2:
        directions = ["right", "left"]
        random.shuffle(directions)
        seq_lengths = [2, 3]
        random.shuffle(seq_lengths)
    else:
        directions = [random.choice(["right", "left"])]
        seq_lengths = [random.choice([2, 3])]

    specs = []
    for i in range(frame_count):
        seq_len = seq_lengths[i]
        if frame_count == 2:
            inner_h = random.randint(3, 4)
            inner_w = random.randint(max(3, seq_len), 5)
        else:
            inner_h = random.randint(3, 5)
            inner_w = random.randint(max(3, seq_len), 6)
        specs.append(
            {
                "direction": directions[i],
                "seq": tuple(random.sample(SEQ_COLORS, seq_len)),
                "seq_offset": random.randint(0, inner_w - seq_len),
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
    return specs


def _frame_signature(frame: dict) -> tuple:
    return (
        int(frame["top"]),
        int(frame["left"]),
        int(frame["bottom"]),
        int(frame["right"]),
        frame["direction"],
        tuple(int(v) for v in frame["seq"]),
        int(frame["seq_offset"]),
    )


def _non_degenerate(inp: np.ndarray, frames: list[dict]) -> bool:
    filled = _filled_grid(inp, frames)
    if np.array_equal(filled, inp):
        return False
    for i, frame in enumerate(frames):
        alt_frames = [dict(f) for f in frames]
        alt_frames[i]["direction"] = "left" if frame["direction"] == "right" else "right"
        if np.array_equal(_filled_grid(inp, alt_frames), filled):
            return False

        reversed_seq = tuple(reversed(frame["seq"]))
        if reversed_seq == frame["seq"]:
            return False
        alt_frames = [dict(f) for f in frames]
        alt_frames[i]["seq"] = reversed_seq
        if np.array_equal(_filled_grid(inp, alt_frames), filled):
            return False
    return True


def _attempt() -> np.ndarray | None:
    frame_count = random.choices([1, 2], weights=[2, 3])[0]
    if frame_count == 1:
        height = random.randint(9, 15)
        width = random.randint(9, 15)
    else:
        height = random.randint(13, 15)
        width = random.randint(13, 15)

    specs = _planned_specs(frame_count)
    order = sorted(range(frame_count), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)

    occupied: set[tuple[int, int]] = set()
    placed: list[dict | None] = [None] * frame_count
    for i in order:
        candidates = _candidate_positions(specs[i], (height, width), occupied)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placed[i] = chosen
        occupied.update(_frame_cells(chosen))

    frames = [frame for frame in placed if frame is not None]
    frames.sort(key=lambda frame: (frame["top"], frame["left"]))

    grid = np.full((height, width), BG, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    try:
        parsed = _parse_frames(grid)
    except AssertionError:
        return None
    if [_frame_signature(frame) for frame in parsed] != [_frame_signature(frame) for frame in frames]:
        return None

    expected = _scale2(_filled_grid(grid, frames))
    if not np.array_equal(main(grid), expected):
        return None
    if not _non_degenerate(grid, frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid capped periodic-fill frame puzzle")
