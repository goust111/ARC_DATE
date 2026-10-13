import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
FRAME_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if np.any(values == BG):
        return BG
    return int(values[np.argmax(counts)])


def _candidate_frame(
    grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int, int, int, int, int, int] | None:
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    border_color = int(grid[r0, c0])
    if border_color == bg:
        return None

    if not np.all(grid[r0, c0:c1] == border_color):
        return None
    if not np.all(grid[r1 - 1, c0:c1] == border_color):
        return None
    if not np.all(grid[r0:r1, c0] == border_color):
        return None
    if not np.all(grid[r0:r1, c1 - 1] == border_color):
        return None

    interior = grid[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1]
    non_bg = np.argwhere(interior != bg)
    if len(non_bg) != 1:
        return None

    mr, mc = map(int, non_bg[0])
    marker_color = int(interior[mr, mc])
    return (r0, c0, r1, c1, border_color, marker_color, (r0 + 1 + mr) * grid.shape[1] + (c0 + 1 + mc))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    frames: list[dict[str, int]] = []
    occupied = np.zeros((h, w), dtype=bool)

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) == bg or occupied[r0, c0]:
                continue

            color = int(grid[r0, c0])
            right_cols = [c for c in range(c0 + 3, w + 1) if int(grid[r0, c - 1]) == color]
            bottom_rows = [r for r in range(r0 + 3, h + 1) if int(grid[r - 1, c0]) == color]
            found = None

            for r1 in reversed(bottom_rows):
                for c1 in reversed(right_cols):
                    candidate = _candidate_frame(grid, bg, r0, c0, r1, c1)
                    if candidate is None:
                        continue

                    rr0, cc0, rr1, cc1, border_color, marker_color, marker_key = candidate
                    border_mask = np.zeros((h, w), dtype=bool)
                    border_mask[rr0, cc0:cc1] = True
                    border_mask[rr1 - 1, cc0:cc1] = True
                    border_mask[rr0:rr1, cc0] = True
                    border_mask[rr0:rr1, cc1 - 1] = True
                    if np.any(occupied & border_mask):
                        continue

                    found = {
                        "r0": rr0,
                        "c0": cc0,
                        "r1": rr1,
                        "c1": cc1,
                        "border_color": border_color,
                        "marker_color": marker_color,
                        "marker_key": marker_key,
                    }
                    occupied |= border_mask
                    break
                if found is not None:
                    break

            if found is not None:
                frames.append(found)

    frames.sort(key=lambda item: (item["r0"], item["c0"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region based on whether its interior marker matches the frame color.

    - Matching marker/frame colors produce a solid interior fill in the frame color.
    - Non-matching colors produce alternating horizontal interior stripes starting
      with the marker color on the first interior row.
    - The one-pixel frame border remains unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        border_color = frame["border_color"]
        marker_color = frame["marker_color"]
        inner = out[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1]

        if marker_color == border_color:
            inner[:] = border_color
            continue

        for offset, row in enumerate(range(r0 + 1, r1 - 1)):
            fill = marker_color if offset % 2 == 0 else border_color
            out[row, c0 + 1 : c1 - 1] = fill

    return out


def _make_modes(count: int, rng: random.Random) -> list[bool]:
    same_count = count // 2
    if count % 2 == 1:
        same_count += rng.randint(0, 1)
    diff_count = count - same_count
    modes = [True] * same_count + [False] * diff_count
    rng.shuffle(modes)
    return modes


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 1)
    cc1 = min(w, c1 + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    border_color: int,
    marker_color: int,
    marker_pos: tuple[int, int],
) -> None:
    r1 = r0 + outer_h
    c1 = c0 + outer_w
    grid[r0, c0:c1] = border_color
    grid[r1 - 1, c0:c1] = border_color
    grid[r0:r1, c0] = border_color
    grid[r0:r1, c1 - 1] = border_color
    mr, mc = marker_pos
    grid[mr, mc] = marker_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    max_frames = min(5, max(2, (h * w) // 80))
    count = rng.randint(2, max_frames)
    modes = _make_modes(count, rng)

    border_colors = rng.sample(PALETTE, count)
    specs: list[tuple[int, int, int, int, int, int, tuple[int, int], bool]] = []

    for index in range(count):
        same = modes[index]
        border_color = border_colors[index]
        marker_color = border_color if same else rng.choice([c for c in PALETTE if c != border_color])
        placed = False

        for _ in range(FRAME_ATTEMPTS):
            inner_h = rng.randint(2, 6)
            inner_w = rng.randint(2, 6)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 > h or outer_w + 2 > w:
                continue

            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h
            c1 = c0 + outer_w

            if np.any(reserved[r0:r1, c0:c1]):
                continue

            mr = rng.randint(r0 + 1, r1 - 2)
            mc = rng.randint(c0 + 1, c1 - 2)
            specs.append((r0, c0, outer_h, outer_w, border_color, marker_color, (mr, mc), same))
            _reserve(reserved, r0, c0, r1, c1)
            placed = True
            break

        if not placed:
            return None

    rng.shuffle(specs)
    for r0, c0, outer_h, outer_w, border_color, marker_color, marker_pos, _same in specs:
        _draw_frame(grid, r0, c0, outer_h, outer_w, border_color, marker_color, marker_pos)

    detected = _find_frames(grid, BG)
    if len(detected) != count:
        return None

    detected_modes = Counter(frame["marker_color"] == frame["border_color"] for frame in detected)
    if detected_modes[True] == 0 or detected_modes[False] == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid input")
