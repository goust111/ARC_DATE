import random

import numpy as np


BG = 0
GEN_RETRIES = 400
GEN_RNG = random.Random()
DIRS = ("top", "bottom", "left", "right")


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _gap_run(line: np.ndarray, frame_color: int, bg: int) -> tuple[int, int] | None:
    if line.size < 2:
        return None
    if np.any((line != frame_color) & (line != bg)):
        return None
    gap = np.flatnonzero(line == bg)
    if gap.size < 2:
        return None
    if int(gap[-1] - gap[0] + 1) != int(gap.size):
        return None
    if np.any(line[: gap[0]] != frame_color):
        return None
    if np.any(line[gap[-1] + 1 :] != frame_color):
        return None
    return int(gap[0]), int(gap[-1])


def _find_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, int, int, int, int, str, int, int]:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue

        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        on_border = (
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        )
        if not bool(np.all(on_border)):
            continue

        border_mask = np.zeros_like(grid, dtype=bool)
        border_mask[r0, c0 : c1 + 1] = True
        border_mask[r1, c0 : c1 + 1] = True
        border_mask[r0 : r1 + 1, c0] = True
        border_mask[r0 : r1 + 1, c1] = True
        border_values = grid[border_mask]
        if np.any((border_values != color) & (border_values != bg)):
            continue
        if any(int(grid[r, c]) != color for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
            continue

        side_lines = {
            "top": grid[r0, c0 + 1 : c1],
            "bottom": grid[r1, c0 + 1 : c1],
            "left": grid[r0 + 1 : r1, c0],
            "right": grid[r0 + 1 : r1, c1],
        }
        side_gaps: list[tuple[str, int, int]] = []
        valid = True
        for side, line in side_lines.items():
            gap = _gap_run(line, color, bg)
            if gap is None:
                if np.any(line != color):
                    valid = False
                    break
            else:
                side_gaps.append((side, gap[0], gap[1]))
        if not valid or len(side_gaps) != 1:
            continue

        if not bool(np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg)):
            continue

        side, gap_start, gap_end = side_gaps[0]
        score = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidate = (score, color, r0, c0, r1, c1, side, gap_start, gap_end)
        if best is None or candidate > best:
            best = candidate

    assert best is not None, "expected one gapped monochrome frame"
    _, color, r0, c0, r1, c1, side, gap_start, gap_end = best
    return color, r0, c0, r1, c1, side, gap_start, gap_end


def _seed_sequence(interior: np.ndarray, bg: int, side: str) -> list[int]:
    pts = np.argwhere(interior != bg)
    assert pts.size > 0, "seed must contain non-background pixels"
    sr0, sc0 = map(int, pts.min(axis=0))
    sr1, sc1 = map(int, pts.max(axis=0))

    seq: list[int] = []
    if side == "right":
        for c in range(sc0, sc1 + 1):
            for r in range(sr0, sr1 + 1):
                value = int(interior[r, c])
                if value != bg:
                    seq.append(value)
    elif side == "left":
        for c in range(sc1, sc0 - 1, -1):
            for r in range(sr0, sr1 + 1):
                value = int(interior[r, c])
                if value != bg:
                    seq.append(value)
    elif side == "bottom":
        for r in range(sr0, sr1 + 1):
            for c in range(sc0, sc1 + 1):
                value = int(interior[r, c])
                if value != bg:
                    seq.append(value)
    else:
        for r in range(sr1, sr0 - 1, -1):
            for c in range(sc0, sc1 + 1):
                value = int(interior[r, c])
                if value != bg:
                    seq.append(value)
    assert seq, "seed sequence must be non-empty"
    return seq


def _paint_projection(
    out: np.ndarray,
    sequence: list[int],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    gap_start: int,
    gap_end: int,
) -> None:
    height, width = map(int, out.shape)
    if side in ("left", "right"):
        gap_r0 = r0 + 1 + gap_start
        gap_r1 = r0 + 1 + gap_end
        columns = range(c1, width) if side == "right" else range(c0, -1, -1)
        for idx, col in enumerate(columns):
            color = sequence[idx % len(sequence)]
            out[gap_r0 : gap_r1 + 1, col] = color
    else:
        gap_c0 = c0 + 1 + gap_start
        gap_c1 = c0 + 1 + gap_end
        rows = range(r1, height) if side == "bottom" else range(r0, -1, -1)
        for idx, row in enumerate(rows):
            color = sequence[idx % len(sequence)]
            out[row, gap_c0 : gap_c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project a seed-derived color cycle outward through a missing side of a rectangular frame.

    Input structure:
    - Black is the dominant background color.
    - One non-black color forms a one-pixel-thick rectangular frame whose top,
      bottom, left, or right side contains a single contiguous gap.
    - Inside the frame sits a smaller multi-color seed pattern on the black
      interior; the rest of the canvas is background.

    Rule:
    - Detect which frame side contains the gap; that side gives the outward
      projection direction.
    - Read the seed into a 1D color sequence by scanning along the projection
      axis: right uses columns left-to-right, left uses columns right-to-left,
      bottom uses rows top-to-bottom, and top uses rows bottom-to-top. Within
      each scanned column or row, keep only non-background pixels in natural
      top-to-bottom or left-to-right order.
    - Starting at the gap itself, paint successive projected columns or rows up
      to the grid boundary. Each whole projected stripe spanning the gap range
      takes the next color from the sequence, cycling when needed.
    - Keep the surviving frame pixels, the seed, and the frame interior
      unchanged.

    Key insight:
    - The gap fixes the projection direction, while the seed's row or column
      structure defines the repeating color order used on the outward stripes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frame_color, r0, c0, r1, c1, side, gap_start, gap_end = _find_frame(g, bg)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    sequence = _seed_sequence(interior, bg, side)

    out = g.copy()
    _paint_projection(out, sequence, r0, c0, r1, c1, side, gap_start, gap_end)
    return out


def _build_seed(
    rng: random.Random, seed_h: int, seed_w: int, palette: list[int], by_columns: bool
) -> np.ndarray:
    seed = np.zeros((seed_h, seed_w), dtype=int)
    if by_columns:
        special = rng.randrange(seed_w)
        center = rng.randrange(seed_h)
        for c in range(seed_w):
            min_len = 2 if c == special and seed_h >= 2 else 1
            max_len = min(seed_h, 3)
            length = rng.randint(min_len, max_len)
            center = min(max(center + rng.choice((-1, 0, 1)), 0), seed_h - 1)
            start = min(max(center - length // 2, 0), seed_h - length)
            for r in range(start, start + length):
                seed[r, c] = rng.choice(palette)

        removable = [
            (r, c)
            for r in range(seed_h)
            for c in range(seed_w)
            if seed[r, c] != 0 and np.count_nonzero(seed[:, c]) > 1
        ]
    else:
        special = rng.randrange(seed_h)
        center = rng.randrange(seed_w)
        for r in range(seed_h):
            min_len = 2 if r == special and seed_w >= 2 else 1
            max_len = min(seed_w, 3)
            length = rng.randint(min_len, max_len)
            center = min(max(center + rng.choice((-1, 0, 1)), 0), seed_w - 1)
            start = min(max(center - length // 2, 0), seed_w - length)
            for c in range(start, start + length):
                seed[r, c] = rng.choice(palette)

        removable = [
            (r, c)
            for r in range(seed_h)
            for c in range(seed_w)
            if seed[r, c] != 0 and np.count_nonzero(seed[r, :]) > 1
        ]

    if removable and rng.random() < 0.45:
        rr, cc = removable[rng.randrange(len(removable))]
        seed[rr, cc] = 0

    used = sorted({int(v) for v in seed[seed != 0]})
    if len(used) < 2:
        filled = np.argwhere(seed != 0)
        rr, cc = map(int, filled[rng.randrange(len(filled))])
        choices = [color for color in palette if color != int(seed[rr, cc])]
        seed[rr, cc] = rng.choice(choices)
    return seed


def _place_frame(
    grid: np.ndarray,
    frame_color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    gap_start: int,
    gap_end: int,
) -> None:
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    if side == "top":
        grid[r0, c0 + 1 + gap_start : c0 + 2 + gap_end] = BG
    elif side == "bottom":
        grid[r1, c0 + 1 + gap_start : c0 + 2 + gap_end] = BG
    elif side == "left":
        grid[r0 + 1 + gap_start : r0 + 2 + gap_end, c0] = BG
    else:
        grid[r0 + 1 + gap_start : r0 + 2 + gap_end, c1] = BG


def _projection_depth(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int, side: str) -> int:
    height, width = map(int, shape)
    if side == "right":
        return width - c1
    if side == "left":
        return c0 + 1
    if side == "bottom":
        return height - r1
    return r0 + 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(DIRS)
    frame_color = rng.randint(1, 9)
    palette = [color for color in range(1, 10) if color != frame_color]

    inner_h = rng.randint(4, 9)
    inner_w = rng.randint(4, 9)
    box_h = inner_h + 2
    box_w = inner_w + 2

    if side in ("left", "right"):
        side_len = inner_h
        gap_len = rng.randint(2, side_len - 1)
        gap_start = rng.randint(0, side_len - gap_len)
        top_margin = rng.randint(1, 5)
        bottom_margin = rng.randint(1, 5)
        opposite_margin = rng.randint(1, 5)
        outside = rng.randint(2, 8)
        height = top_margin + box_h + bottom_margin
        width = opposite_margin + box_w + outside
        if side == "right":
            r0 = top_margin
            c0 = opposite_margin
        else:
            r0 = top_margin
            c0 = outside
    else:
        side_len = inner_w
        gap_len = rng.randint(2, side_len - 1)
        gap_start = rng.randint(0, side_len - gap_len)
        left_margin = rng.randint(1, 5)
        right_margin = rng.randint(1, 5)
        opposite_margin = rng.randint(1, 5)
        outside = rng.randint(2, 8)
        height = opposite_margin + box_h + outside
        width = left_margin + box_w + right_margin
        if side == "bottom":
            r0 = opposite_margin
            c0 = left_margin
        else:
            r0 = outside
            c0 = left_margin

    if max(height, width) > 30:
        return None

    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    grid = np.zeros((height, width), dtype=int)
    _place_frame(grid, frame_color, r0, c0, r1, c1, side, gap_start, gap_end=gap_start + gap_len - 1)

    max_seed_h = min(5, inner_h)
    max_seed_w = min(5, inner_w)
    seed_h = rng.randint(2, max_seed_h)
    seed_w = rng.randint(2, max_seed_w)
    if seed_h == inner_h and seed_w == inner_w:
        if seed_h > 2:
            seed_h -= 1
        else:
            seed_w -= 1

    seed_palette = rng.sample(palette, rng.randint(2, min(4, len(palette))))
    seed = _build_seed(rng, seed_h, seed_w, seed_palette, by_columns=side in ("left", "right"))
    seed_r = rng.randint(0, inner_h - seed_h)
    seed_c = rng.randint(0, inner_w - seed_w)
    window = grid[r0 + 1 + seed_r : r0 + 1 + seed_r + seed_h, c0 + 1 + seed_c : c0 + 1 + seed_c + seed_w]
    window[seed != 0] = seed[seed != 0]

    if len({int(v) for v in seed[seed != 0]}) < 2:
        return None

    if _projection_depth(grid.shape, r0, c0, r1, c1, side) < 3:
        return None

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != BG:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    changed = out != grid
    if int(changed.sum()) < gap_len * 2:
        return None
    if len(set(map(int, out[changed].tolist()))) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
