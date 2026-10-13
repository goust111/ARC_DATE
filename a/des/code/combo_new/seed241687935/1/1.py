from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEED = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 26)
GRID_W_RANGE = (14, 28)
MAX_SEGMENTS = 6
MAX_FRAMES = 5
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _top_palette_mapping(grid: np.ndarray) -> dict[int, int]:
    row = [int(x) for x in grid[0]]
    segments: list[int] = []
    prev = None
    for value in row:
        if value != prev:
            segments.append(value)
            prev = value
    assert len(segments) >= 2, "top color strip must contain at least two segments"
    assert len(set(segments)) == len(segments), "top strip segments must use distinct colors"
    return {color: segments[i - 1] for i, color in enumerate(segments)}


def _frame_border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = {(r0, c) for c in range(c0, c1 + 1)}
    pts |= {(r1, c) for c in range(c0, c1 + 1)}
    pts |= {(r, c0) for r in range(r0, r1 + 1)}
    pts |= {(r, c1) for r in range(r0, r1 + 1)}
    return pts


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, int]] = []
    for color in sorted(set(map(int, g[1:].reshape(-1))) - {BG, SEED}):
        mask = g == int(color)
        mask[0, :] = False
        for comp in _components(mask):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if set(comp) != _frame_border_set(r0, c0, r1, c1):
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            seeds = np.argwhere(interior == SEED)
            assert len(seeds) == 1, "each frame interior must contain exactly one grey seed"
            assert np.all((interior == BG) | (interior == SEED)), "frame interior must otherwise be empty"
            sr, sc = map(int, seeds[0])
            frames.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "seed_r": int(r0 + 1 + sr),
                    "seed_c": int(c0 + 1 + sc),
                }
            )
    assert frames, "expected at least one hollow frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame with the color dictated by the top strip and the seed's opposite half.

    Input structure:
    - The background is black.
    - The top row is a continuous horizontal strip of differently colored segments with no gaps.
    - The remaining area contains several hollow rectangular frames whose border colors appear in the top strip.
    - Each frame interior contains exactly one grey seed pixel positioned in the upper or lower half.

    Rule:
    - Read the top strip as a cyclic legend: for each segment color, the fill color is the color of the segment immediately to its left, wrapping from the first segment to the last.
    - For every hollow frame, look up its border color in that legend.
    - If the grey seed lies in the upper half of the frame interior, fill the lower half of the interior with the mapped color; if the seed lies in the lower half, fill the upper half instead.
    - When the interior height is odd, leave the middle row empty, and remove every grey seed pixel after filling.

    Key insight:
    - The top strip is not a direct color label but a cyclic left-neighbor lookup table, and the seed marks the blank half while the opposite half gets filled.
    """
    g = np.asarray(grid, dtype=int)
    palette = _top_palette_mapping(g)
    frames = _find_frames(g)

    out = g.copy()
    out[out == SEED] = BG

    for frame in frames:
        fill_color = int(palette[int(frame["color"])])
        r0, c0, r1, c1 = map(int, (frame["r0"], frame["c0"], frame["r1"], frame["c1"]))
        inner_h = int(r1 - r0 - 1)
        split = inner_h // 2
        seed_local_r = int(frame["seed_r"] - (r0 + 1))

        in_upper = seed_local_r < split
        in_lower = seed_local_r >= inner_h - split
        assert in_upper ^ in_lower, "seed must lie strictly in the upper or lower half"

        if in_upper:
            fill_r0 = r0 + 1 + (inner_h - split)
            fill_r1 = r1
        else:
            fill_r0 = r0 + 1
            fill_r1 = r0 + 1 + split

        out[fill_r0:fill_r1, c0 + 1 : c1] = fill_color

    return out


def _random_composition(rng: random.Random, total: int, parts: int, min_part: int) -> list[int]:
    remaining = total
    lengths: list[int] = []
    for i in range(parts):
        slots_left = parts - i - 1
        lo = min_part
        hi = remaining - slots_left * min_part
        lengths.append(rng.randint(lo, hi))
        remaining -= lengths[-1]
    rng.shuffle(lengths)
    return lengths


def _segment_lengths(rng: random.Random, width: int, n_segments: int) -> list[int]:
    for _ in range(50):
        lengths = _random_composition(rng, width, n_segments, min_part=2)
        if len(set(lengths)) > 1:
            return lengths
    lengths = [2] * n_segments
    lengths[-1] += width - sum(lengths)
    return lengths


def _paint_strip(grid: np.ndarray, colors: list[int], lengths: list[int]) -> dict[int, int]:
    c = 0
    for color, length in zip(colors, lengths):
        grid[0, c : c + length] = int(color)
        c += int(length)
    return {color: colors[i - 1] for i, color in enumerate(colors)}


def _frame_size_candidates(rng: random.Random, height: int, width: int) -> list[tuple[int, int]]:
    max_inner_h = min(7, height - 5)
    max_inner_w = min(8, width - 4)
    sizes = [(ih, iw) for ih in range(2, max_inner_h + 1) for iw in range(2, max_inner_w + 1)]
    rng.shuffle(sizes)
    return sizes


def _valid_positions(occ: np.ndarray, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    valid: list[tuple[int, int]] = []
    for r0 in range(2, h - outer_h):
        for c0 in range(1, w - outer_w):
            if occ[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1].any():
                continue
            valid.append((int(r0), int(c0)))
    return valid


def _seed_row(rng: random.Random, inner_h: int, upper_half: bool) -> int:
    split = inner_h // 2
    if upper_half:
        return rng.choice(list(range(split)))
    return rng.choice(list(range(inner_h - split, inner_h)))


def _apply_expected(input_grid: np.ndarray, frames: list[dict[str, int]], palette: dict[int, int]) -> np.ndarray:
    out = input_grid.copy()
    out[out == SEED] = BG
    for frame in frames:
        r0, c0 = int(frame["r0"]), int(frame["c0"])
        inner_h = int(frame["inner_h"])
        inner_w = int(frame["inner_w"])
        split = inner_h // 2
        fill_color = int(palette[int(frame["color"])])

        if int(frame["seed_upper"]) == 1:
            fill_rows = slice(r0 + 1 + (inner_h - split), r0 + 1 + inner_h)
        else:
            fill_rows = slice(r0 + 1, r0 + 1 + split)
        out[fill_rows, c0 + 1 : c0 + 1 + inner_w] = fill_color
    return out


def _choose_frame_colors(
    rng: random.Random, strip_colors: list[int], n_frames: int
) -> list[int]:
    chosen: list[int] = []
    if rng.random() < 0.5:
        chosen.append(int(strip_colors[0]))
    if n_frames > len(chosen) and len(strip_colors) > 1 and rng.random() < 0.7:
        chosen.append(int(rng.choice(strip_colors[1:])))
    while len(chosen) < n_frames:
        chosen.append(int(rng.choice(strip_colors)))
    rng.shuffle(chosen)
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)

    max_segments = min(MAX_SEGMENTS, width // 2, len(PALETTE_COLORS))
    n_segments = rng.randint(3, max_segments)
    strip_colors = rng.sample(list(PALETTE_COLORS), n_segments)
    strip_lengths = _segment_lengths(rng, width, n_segments)

    grid = np.full((height, width), BG, dtype=int)
    palette = _paint_strip(grid, strip_colors, strip_lengths)

    occ = np.zeros((height, width), dtype=bool)
    occ[0:2, :] = True

    frame_cap = min(MAX_FRAMES, max(1, ((height - 2) * (width - 2)) // 70 + 1))
    n_frames = rng.randint(1, frame_cap)
    frame_colors = _choose_frame_colors(rng, strip_colors, n_frames)

    frames: list[dict[str, int]] = []
    for frame_color in frame_colors:
        placed = False
        for inner_h, inner_w in _frame_size_candidates(rng, height, width):
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            positions = _valid_positions(occ, outer_h, outer_w)
            if not positions:
                continue

            r0, c0 = rng.choice(positions)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            border = _frame_border_set(r0, c0, r1, c1)
            for rr, cc in border:
                grid[rr, cc] = int(frame_color)

            seed_upper = 1 if rng.random() < 0.5 else 0
            local_seed_r = _seed_row(rng, inner_h, upper_half=bool(seed_upper))
            local_seed_c = rng.randrange(inner_w)
            grid[r0 + 1 + local_seed_r, c0 + 1 + local_seed_c] = SEED

            occ[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1] = True
            frames.append(
                {
                    "color": int(frame_color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "inner_h": int(inner_h),
                    "inner_w": int(inner_w),
                    "seed_upper": int(seed_upper),
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _apply_expected(grid, frames, palette)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
