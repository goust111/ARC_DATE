from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


YELLOW = 4
PALETTE = [0, 1, 2, 3, 5, 6, 7, 8, 9]

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_CHOICES = [1, 2, 2, 3, 3]
SHORT_SIDE_CHOICES = [2, 3, 4]
LONG_DELTA_CHOICES = [1, 2]
BAND_GAP = 1
MAX_SHAPE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue

            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return comps


def _projection_counts(comp: dict) -> tuple[list[int], int, int, int, int, int]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    area = int(comp["area"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height != width, "each foreground object must have a non-square bbox"

    if width > height:
        counts = [0] * width
        for rr, cc in comp["coords"]:
            counts[int(cc) - c0] += 1
    else:
        counts = [0] * height
        for rr, cc in comp["coords"]:
            counts[int(rr) - r0] += 1

    assert sum(counts) == area
    return counts, area, r0, c0, height, width


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll each monochrome object into a horizontal strip encoding its projection on the longest bbox axis.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)
    h, w = map(int, g.shape)

    for comp in _find_components(g, bg):
        color = int(comp["color"])
        counts, area, r0, c0, _bbox_h, _bbox_w = _projection_counts(comp)
        assert 0 <= r0 < h
        assert 0 <= c0 <= w - area, "strip must fit within the canvas"
        assert np.all(out[r0, c0 : c0 + area] == bg), "output strips must not overlap"

        cursor = c0
        for count in counts:
            if count > 0:
                out[r0, cursor : cursor + count] = color
            cursor += count
        assert cursor == c0 + area

        if area % 2 == 1:
            out[r0, c0 + area // 2] = YELLOW

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _build_intervals(long_len: int, short_len: int, target_parity: int) -> tuple[list[int], list[int]] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        anchor = random.randrange(long_len)
        counts = [random.randint(1, short_len) for _ in range(long_len)]
        counts[anchor] = short_len
        if sum(counts) % 2 != target_parity:
            continue

        starts = [0] * long_len
        starts[anchor] = 0
        ok = True

        for idx in range(anchor - 1, -1, -1):
            prev_start = starts[idx + 1]
            prev_end = prev_start + counts[idx + 1] - 1
            lo = max(0, prev_start - counts[idx] + 1)
            hi = min(short_len - counts[idx], prev_end)
            if lo > hi:
                ok = False
                break
            starts[idx] = random.randint(lo, hi)

        if not ok:
            continue

        for idx in range(anchor + 1, long_len):
            prev_start = starts[idx - 1]
            prev_end = prev_start + counts[idx - 1] - 1
            lo = max(0, prev_start - counts[idx] + 1)
            hi = min(short_len - counts[idx], prev_end)
            if lo > hi:
                ok = False
                break
            starts[idx] = random.randint(lo, hi)

        if ok:
            return counts, starts

    return None


def _generate_shape(axis: str, target_parity: int) -> tuple[np.ndarray, list[int]]:
    short_len = random.choice(SHORT_SIDE_CHOICES)
    long_options = [short_len + delta for delta in LONG_DELTA_CHOICES if short_len + delta <= 6]
    long_len = random.choice(long_options)
    built = _build_intervals(long_len, short_len, target_parity)
    assert built is not None
    counts, starts = built

    if axis == "h":
        patch = np.zeros((short_len, long_len), dtype=bool)
        for col, (count, start) in enumerate(zip(counts, starts)):
            patch[start : start + count, col] = True
    else:
        patch = np.zeros((long_len, short_len), dtype=bool)
        for row, (count, start) in enumerate(zip(counts, starts)):
            patch[row, start : start + count] = True

    return patch, counts


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.nonzero(patch)
    for rr, cc in zip(rows, cols):
        grid[int(r0 + rr), int(c0 + cc)] = int(color)


def _paint_strip(grid: np.ndarray, r0: int, c0: int, color: int, counts: list[int]) -> None:
    cursor = c0
    for count in counts:
        if count > 0:
            grid[r0, cursor : cursor + count] = int(color)
        cursor += count
    total = sum(counts)
    if total % 2 == 1:
        grid[r0, c0 + total // 2] = YELLOW


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    specs: list[dict] = []

    for _ in range(n_objects):
        axis = random.choice(["h", "v"])
        parity = random.randint(0, 1)
        patch, counts = _generate_shape(axis, parity)
        specs.append(
            {
                "patch": patch,
                "counts": counts,
                "area": int(sum(counts)),
                "bbox_h": int(patch.shape[0]),
                "bbox_w": int(patch.shape[1]),
            }
        )

    min_h = sum(spec["bbox_h"] for spec in specs) + BAND_GAP * (n_objects - 1)
    min_w = max(spec["area"] for spec in specs)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h_lo = max(GRID_H_RANGE[0], min_h)
    w_lo = max(GRID_W_RANGE[0], min_w)
    if h_lo > GRID_H_RANGE[1] or w_lo > GRID_W_RANGE[1]:
        return None

    h = random.randint(h_lo, GRID_H_RANGE[1])
    w = random.randint(w_lo, GRID_W_RANGE[1])

    bg_choices = [color for color in PALETTE if color != YELLOW]
    bg = random.choice(bg_choices)
    fg_choices = [color for color in PALETTE if color not in (bg, YELLOW)]
    if len(fg_choices) < n_objects:
        return None
    colors = random.sample(fg_choices, n_objects)

    slack = h - min_h
    extras = _random_partition(slack, n_objects + 1)

    r = extras[0]
    input_grid = np.full((h, w), bg, dtype=int)
    output_grid = np.full((h, w), bg, dtype=int)

    for idx, spec in enumerate(specs):
        area = int(spec["area"])
        bbox_h = int(spec["bbox_h"])
        bbox_w = int(spec["bbox_w"])
        c0 = random.randint(0, w - area)
        color = int(colors[idx])

        _paint_patch(input_grid, spec["patch"], r, c0, color)
        _paint_strip(output_grid, r, c0, color, spec["counts"])

        r += bbox_h
        if idx < n_objects - 1:
            r += BAND_GAP + extras[idx + 1]

    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
