from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
SIDE_ORDER = {"top": 0, "bottom": 1, "left": 2, "right": 3}
ALL_COLORS = tuple(range(10))


def _rect_prefix(mask: np.ndarray) -> np.ndarray:
    prefix = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    prefix[1:, 1:] = np.cumsum(np.cumsum(mask.astype(int), axis=0), axis=1)
    return prefix


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1] - prefix[r1 + 1, c0] + prefix[r0, c0])


def _find_pattern_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = g != BG
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected non-background cells"

    prefix = _rect_prefix(mask)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int, int]] = []

    for r0 in range(h):
        for r1 in range(r0 + 1, h):
            for c0 in range(w):
                for c1 in range(c0 + 1, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if _rect_sum(prefix, r0, c0, r1, c1) != area:
                        continue

                    inside_colors = set(map(int, g[r0 : r1 + 1, c0 : c1 + 1].ravel()))
                    inside_colors.discard(BG)
                    if len(inside_colors) < 2:
                        continue

                    side_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
                    valid = True
                    for r, c in coords:
                        r = int(r)
                        c = int(c)
                        if r0 <= r <= r1 and c0 <= c <= c1:
                            continue

                        in_row = r0 <= r <= r1
                        in_col = c0 <= c <= c1
                        if in_row == in_col:
                            valid = False
                            break

                        if int(g[r, c]) in inside_colors:
                            valid = False
                            break

                        if in_col:
                            side = "top" if r < r0 else "bottom"
                        else:
                            side = "left" if c < c0 else "right"
                        side_counts[side] += 1

                    if valid and all(side_counts[side] > 0 for side in side_counts):
                        candidates.append((area, r0, c0, r1, c1))

    assert candidates, "failed to locate the filled center rectangle"
    best_area = max(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_area]
    assert len(best) == 1, "center rectangle must be uniquely identifiable"
    _area, r0, c0, r1, c1 = best[0]
    return r0, c0, r1, c1


def _collect_arrows(
    g: np.ndarray, bbox: tuple[int, int, int, int]
) -> list[dict[str, int | str]]:
    r0, c0, r1, c1 = bbox
    arrows: list[dict[str, int | str]] = []

    coords = np.argwhere(g != BG)
    for r, c in coords:
        r = int(r)
        c = int(c)
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue

        if c0 <= c <= c1:
            if r < r0:
                side = "top"
                line = c - c0
                distance = r0 - r
            else:
                side = "bottom"
                line = c - c0
                distance = r - r1
        else:
            if c < c0:
                side = "left"
                line = r - r0
                distance = c0 - c
            else:
                side = "right"
                line = r - r0
                distance = c - c1

        arrows.append(
            {
                "side": side,
                "line": int(line),
                "distance": int(distance),
                "color": int(g[r, c]),
            }
        )

    assert arrows, "expected at least one arrow"
    return arrows


def _apply_arrows(core: np.ndarray, arrows: list[dict[str, int | str]], counts: Counter[int]) -> np.ndarray:
    out = core.copy()
    h, w = out.shape

    ordered = sorted(
        arrows,
        key=lambda arrow: (
            -int(arrow["distance"]),
            SIDE_ORDER[str(arrow["side"])],
            int(arrow["line"]),
        ),
    )
    for arrow in ordered:
        side = str(arrow["side"])
        line = int(arrow["line"])
        step = int(counts[int(arrow["color"])])
        if side == "top":
            out[:, line] = np.roll(out[:, line], -step % h)
        elif side == "bottom":
            out[:, line] = np.roll(out[:, line], step % h)
        elif side == "left":
            out[line, :] = np.roll(out[line, :], -step % w)
        else:
            out[line, :] = np.roll(out[line, :], step % w)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the center rectangle along rows and columns according to surrounding arrows.

    The center pattern is the unique largest filled non-black rectangle. Every
    non-black cell outside that rectangle is treated as a one-pixel arrow aligned
    with one rectangle row or column. Arrow colors are counted globally, and arrows
    are executed from farther to nearer relative to the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BG), "expected black background"

    bbox = _find_pattern_bbox(g)
    arrows = _collect_arrows(g, bbox)
    counts = Counter(map(int, g[g != BG].ravel()))

    r0, c0, r1, c1 = bbox
    core = g[r0 : r1 + 1, c0 : c1 + 1]
    shifted = _apply_arrows(core, arrows, counts)

    out = np.full_like(g, BG)
    out[r0 : r1 + 1, c0 : c1 + 1] = shifted
    return out


def _sample_pattern(h: int, w: int, colors: list[int]) -> np.ndarray:
    mode = random.choice(["cells", "row_bands", "col_bands", "blocks"])
    pattern = np.empty((h, w), dtype=int)

    if mode == "cells":
        pattern[:, :] = random.choice(colors)
        for r in range(h):
            for c in range(w):
                pattern[r, c] = random.choice(colors)
    elif mode == "row_bands":
        row_palette = colors[:]
        random.shuffle(row_palette)
        for r in range(h):
            base = row_palette[r % len(row_palette)]
            pattern[r, :] = base
            swaps = random.randint(1, max(1, w // 2))
            for _ in range(swaps):
                pattern[r, random.randrange(w)] = random.choice(colors)
    elif mode == "col_bands":
        col_palette = colors[:]
        random.shuffle(col_palette)
        for c in range(w):
            base = col_palette[c % len(col_palette)]
            pattern[:, c] = base
            swaps = random.randint(1, max(1, h // 2))
            for _ in range(swaps):
                pattern[random.randrange(h), c] = random.choice(colors)
    else:
        split_r = random.randint(1, h - 1)
        split_c = random.randint(1, w - 1)
        block_colors = random.choices(colors, k=4)
        pattern[:split_r, :split_c] = block_colors[0]
        pattern[:split_r, split_c:] = block_colors[1]
        pattern[split_r:, :split_c] = block_colors[2]
        pattern[split_r:, split_c:] = block_colors[3]
        noise = random.randint(1, max(1, (h * w) // 4))
        for _ in range(noise):
            pattern[random.randrange(h), random.randrange(w)] = random.choice(colors)

    if len(set(map(int, pattern.ravel()))) < 2:
        pattern[0, 0] = colors[0]
        pattern[-1, -1] = colors[-1]
    return pattern


def _split_indices(size: int) -> tuple[list[int], list[int]]:
    indices = list(range(size))
    random.shuffle(indices)
    left_count = random.randint(1, size - 1)
    right_count = random.randint(1, size - left_count)
    first = sorted(indices[:left_count])
    second = sorted(indices[left_count : left_count + right_count])
    return first, second


def _sample_line_distances(options: list[int]) -> list[int]:
    count = 1
    if len(options) > 1 and random.random() < 0.4:
        count = 2
    return sorted(random.sample(options, k=count), reverse=True)


def _assign_arrow_colors(total_arrows: int, available_colors: list[int]) -> list[int]:
    max_colors = min(len(available_colors), max(2, min(4, total_arrows - 1)))
    color_count = random.randint(2, max_colors)
    colors = random.sample(available_colors, color_count)

    remaining = total_arrows
    counts: list[int] = []
    for i in range(color_count):
        left = color_count - i - 1
        if i == color_count - 1:
            value = remaining
        else:
            lo = 1
            hi = remaining - left
            value = random.randint(lo, hi)
        counts.append(value)
        remaining -= value

    if max(counts) == 1:
        counts[0] += 1
        counts[-1] -= 1
    random.shuffle(counts)

    assigned: list[int] = []
    for color, count in zip(colors, counts):
        assigned.extend([color] * count)
    random.shuffle(assigned)
    return assigned


def _build_arrow_specs(h: int, w: int) -> tuple[dict[str, list[int]], dict[str, list[tuple[int, list[int]]]]]:
    mode = random.choice(["rows_outer", "cols_outer"])
    if mode == "rows_outer":
        distance_options = {
            "top": [1, 2],
            "bottom": [1, 2],
            "left": [3, 4],
            "right": [3, 4],
        }
    else:
        distance_options = {
            "top": [3, 4],
            "bottom": [3, 4],
            "left": [1, 2],
            "right": [1, 2],
        }

    left_rows, right_rows = _split_indices(h)
    top_cols, bottom_cols = _split_indices(w)
    sides = {
        "left": [(row, _sample_line_distances(distance_options["left"])) for row in left_rows],
        "right": [(row, _sample_line_distances(distance_options["right"])) for row in right_rows],
        "top": [(col, _sample_line_distances(distance_options["top"])) for col in top_cols],
        "bottom": [(col, _sample_line_distances(distance_options["bottom"])) for col in bottom_cols],
    }
    return distance_options, sides


def _place_arrows(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    sides: dict[str, list[tuple[int, list[int]]]],
    colors: list[int],
) -> None:
    r0, c0, r1, c1 = bbox
    idx = 0
    for side in ("left", "right", "top", "bottom"):
        for line, distances in sides[side]:
            for distance in distances:
                color = colors[idx]
                idx += 1
                if side == "left":
                    grid[r0 + line, c0 - distance] = color
                elif side == "right":
                    grid[r0 + line, c1 + distance] = color
                elif side == "top":
                    grid[r0 - distance, c0 + line] = color
                else:
                    grid[r1 + distance, c0 + line] = color


def _count_total_arrows(sides: dict[str, list[tuple[int, list[int]]]]) -> int:
    return sum(len(distances) for items in sides.values() for _line, distances in items)


def _gen_attempt() -> np.ndarray | None:
    pattern_h = random.randint(3, 6)
    pattern_w = random.randint(3, 6)

    pattern_color_count = random.randint(2, min(4, pattern_h * pattern_w))
    pattern_colors = random.sample([color for color in ALL_COLORS if color != BG], pattern_color_count)
    arrow_palette = [color for color in ALL_COLORS if color not in pattern_colors and color != BG]
    if len(arrow_palette) < 2:
        return None

    _distance_options, sides = _build_arrow_specs(pattern_h, pattern_w)
    total_arrows = _count_total_arrows(sides)
    if total_arrows < 4:
        return None

    max_top = max(distance for _line, distances in sides["top"] for distance in distances)
    max_bottom = max(distance for _line, distances in sides["bottom"] for distance in distances)
    max_left = max(distance for _line, distances in sides["left"] for distance in distances)
    max_right = max(distance for _line, distances in sides["right"] for distance in distances)

    grid_h = max_top + pattern_h + max_bottom
    grid_w = max_left + pattern_w + max_right
    r0 = max_top
    c0 = max_left
    r1 = r0 + pattern_h - 1
    c1 = c0 + pattern_w - 1

    pattern = _sample_pattern(pattern_h, pattern_w, pattern_colors)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = pattern

    arrow_colors = _assign_arrow_colors(total_arrows, arrow_palette)
    _place_arrows(grid, (r0, c0, r1, c1), sides, arrow_colors)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], pattern):
        return None
    if np.count_nonzero(out != BG) != pattern_h * pattern_w:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
