from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 300
PLACE_TRIES = 80
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
FRAME_COUNT_CHOICES = (1, 2, 3)
FRAME_MARGIN = 2
FRAME_INNER_H_RANGE = (2, 5)
FRAME_INNER_W_RANGE = (2, 6)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _border_nonzero_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    border_cells: list[tuple[int, int]] = []
    border_cells.extend((0, c) for c in range(w))
    border_cells.extend((h - 1, c) for c in range(w))
    border_cells.extend((r, 0) for r in range(1, h - 1))
    border_cells.extend((r, w - 1) for r in range(1, h - 1))

    for sr, sc in border_cells:
        if seen[sr, sc] or int(grid[sr, sc]) == BG:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _extract_palette_mapping(grid: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for comp in _border_nonzero_components(grid):
        if len(comp) != 2:
            raise AssertionError("border objects must be isolated 1x2 palette pairs")
        (r1, c1), (r2, c2) = comp
        if r1 != r2 or abs(c1 - c2) != 1:
            raise AssertionError("palette pairs must be horizontal")
        left, right = sorted(comp, key=lambda p: p[1])
        seed_color = int(grid[left])
        fill_color = int(grid[right])
        if seed_color in mapping and mapping[seed_color] != fill_color:
            raise AssertionError("seed color maps to multiple fill colors")
        mapping[seed_color] = fill_color
    if not mapping:
        raise AssertionError("palette mapping not found")
    return mapping


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    frames: list[dict[str, int]] = []

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        if np.any((pts[:, 0] == 0) | (pts[:, 0] == h - 1) | (pts[:, 1] == 0) | (pts[:, 1] == w - 1)):
            continue

        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not np.all((pts[:, 0] == r0) | (pts[:, 0] == r1) | (pts[:, 1] == c0) | (pts[:, 1] == c1)):
            continue

        perimeter = set()
        for c in range(c0, c1 + 1):
            perimeter.add((r0, c))
            perimeter.add((r1, c))
        for r in range(r0 + 1, r1):
            perimeter.add((r, c0))
            perimeter.add((r, c1))

        frame_pts = {(int(r), int(c)) for r, c in pts}
        if len(frame_pts) != len(perimeter) - 1:
            continue
        missing = list(perimeter - frame_pts)
        if len(missing) != 1:
            continue
        gap_r, gap_c = missing[0]
        if int(grid[gap_r, gap_c]) != BG:
            continue
        if gap_r in (r0, r1) and gap_c in (c0, c1):
            continue

        inside = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] != BG)
        if inside.shape[0] != 1:
            continue
        seed_r, seed_c = map(int, inside[0])
        seed_r += r0 + 1
        seed_c += c0 + 1
        seed_color = int(grid[seed_r, seed_c])
        if seed_color == color:
            continue

        if gap_r == r0:
            side = "top"
        elif gap_r == r1:
            side = "bottom"
        elif gap_c == c0:
            side = "left"
        elif gap_c == c1:
            side = "right"
        else:
            continue

        frames.append(
            {
                "frame_color": color,
                "seed_color": seed_color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "seed_r": seed_r,
                "seed_c": seed_c,
                "side": side,
            }
        )

    if not frames:
        raise AssertionError("no valid frames found")
    frames.sort(key=lambda item: (item["r0"], item["c0"], item["frame_color"]))
    return frames


def _paint_strip(out: np.ndarray, frame: dict[str, int], fill_color: int) -> None:
    r0 = frame["r0"]
    c0 = frame["c0"]
    r1 = frame["r1"]
    c1 = frame["c1"]
    seed_r = frame["seed_r"]
    seed_c = frame["seed_c"]
    side = frame["side"]

    if side == "top":
        out[r0 + 1 : seed_r + 1, seed_c] = fill_color
    elif side == "bottom":
        out[seed_r:r1, seed_c] = fill_color
    elif side == "left":
        out[seed_r, c0 + 1 : seed_c + 1] = fill_color
    else:
        out[seed_r, seed_c:c1] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    mapping = _extract_palette_mapping(g)
    frames = _find_frames(g)

    out = g.copy()
    for frame in frames:
        seed_color = frame["seed_color"]
        if seed_color not in mapping:
            raise AssertionError("seed color missing from palette")
        _paint_strip(out, frame, mapping[seed_color])
    return out


def _pick_spaced_values(rng: random.Random, values: list[int], count: int, min_delta: int) -> list[int] | None:
    values = values[:]
    rng.shuffle(values)
    chosen: list[int] = []
    for value in values:
        if all(abs(value - prev) >= min_delta for prev in chosen):
            chosen.append(value)
            if len(chosen) == count:
                chosen.sort()
                return chosen
    return None


def _place_palette(
    grid: np.ndarray,
    seed_colors: list[int],
    fill_colors: list[int],
    side: str,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    pairs = list(zip(seed_colors, fill_colors))
    rng.shuffle(pairs)

    if side in ("top", "bottom"):
        starts = _pick_spaced_values(rng, list(range(w - 1)), len(pairs), 3)
        if starts is None:
            return False
        row = 0 if side == "top" else h - 1
        for start, (seed_color, fill_color) in zip(starts, pairs):
            grid[row, start] = seed_color
            grid[row, start + 1] = fill_color
    else:
        rows = _pick_spaced_values(rng, list(range(h)), len(pairs), 2)
        if rows is None:
            return False
        left_col = 0 if side == "left" else w - 2
        for row, (seed_color, fill_color) in zip(rows, pairs):
            grid[row, left_col] = seed_color
            grid[row, left_col + 1] = fill_color
    return True


def _place_frame(
    grid: np.ndarray,
    occupied: np.ndarray,
    frame_color: int,
    seed_color: int,
    fill_color: int,
    side: str,
    rng: random.Random,
) -> dict[str, int] | None:
    h, w = grid.shape
    max_inner_h = min(FRAME_INNER_H_RANGE[1], h - 2 * FRAME_MARGIN - 2)
    max_inner_w = min(FRAME_INNER_W_RANGE[1], w - 2 * FRAME_MARGIN - 2)
    if max_inner_h < FRAME_INNER_H_RANGE[0] or max_inner_w < FRAME_INNER_W_RANGE[0]:
        return None

    for _ in range(PLACE_TRIES):
        inner_h = rng.randint(FRAME_INNER_H_RANGE[0], max_inner_h)
        inner_w = rng.randint(FRAME_INNER_W_RANGE[0], max_inner_w)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        r0_max = h - outer_h - FRAME_MARGIN
        c0_max = w - outer_w - FRAME_MARGIN
        if r0_max < FRAME_MARGIN or c0_max < FRAME_MARGIN:
            continue

        r0 = rng.randint(FRAME_MARGIN, r0_max)
        c0 = rng.randint(FRAME_MARGIN, c0_max)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        rr0 = max(0, r0 - 1)
        rr1 = min(h, r1 + 2)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c1 + 2)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        if side == "top":
            gap_r = r0
            gap_c = rng.randint(c0 + 1, c1 - 1)
        elif side == "bottom":
            gap_r = r1
            gap_c = rng.randint(c0 + 1, c1 - 1)
        elif side == "left":
            gap_r = rng.randint(r0 + 1, r1 - 1)
            gap_c = c0
        else:
            gap_r = rng.randint(r0 + 1, r1 - 1)
            gap_c = c1
        grid[gap_r, gap_c] = BG

        seed_r = rng.randint(r0 + 1, r1 - 1)
        seed_c = rng.randint(c0 + 1, c1 - 1)
        grid[seed_r, seed_c] = seed_color

        occupied[rr0:rr1, cc0:cc1] = True
        return {
            "frame_color": frame_color,
            "seed_color": seed_color,
            "fill_color": fill_color,
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "seed_r": seed_r,
            "seed_c": seed_c,
            "side": side,
        }
    return None


def _expected_output(grid: np.ndarray, frames: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        _paint_strip(out, frame, frame["fill_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    min_h = max(GRID_H_RANGE[0], 10 + 2 * (frame_count - 1))
    min_w = max(GRID_W_RANGE[0], 10 + 2 * (frame_count - 1))
    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])

    grid = np.zeros((h, w), dtype=int)
    colors = list(range(1, 10))
    rng.shuffle(colors)
    frame_colors = colors[:frame_count]
    seed_colors = colors[frame_count : 2 * frame_count]
    fill_colors = colors[2 * frame_count : 3 * frame_count]
    palette_side = rng.choice(SIDES)

    if not _place_palette(grid, seed_colors, fill_colors, palette_side, rng):
        return None

    occupied = grid != BG
    directions = list(rng.sample(SIDES, frame_count))
    rng.shuffle(directions)

    frames: list[dict[str, int]] = []
    for frame_color, seed_color, fill_color, side in zip(frame_colors, seed_colors, fill_colors, directions):
        frame = _place_frame(grid, occupied, frame_color, seed_color, fill_color, side, rng)
        if frame is None:
            return None
        frames.append(frame)

    expected = _expected_output(grid, frames)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(produced, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
