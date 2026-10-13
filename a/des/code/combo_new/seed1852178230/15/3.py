from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIRECTIONS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            components.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _legend_patterns() -> dict[str, np.ndarray]:
    patterns: dict[str, np.ndarray] = {}
    for direction, missing in {
        "up": (0, 1),
        "down": (2, 1),
        "left": (1, 0),
        "right": (1, 2),
    }.items():
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        mask[missing] = False
        patterns[direction] = mask
    return patterns


PATTERNS = _legend_patterns()


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, tuple[int, int, int, int], int] | None = None

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h != 5 or w < 9 or (w - 1) % 4 != 0:
            continue
        if not np.all(sub[0, :] == color) or not np.all(sub[-1, :] == color):
            continue
        divider_cols = [c for c in range(w) if bool(np.all(sub[:, c] == color))]
        if divider_cols[0] != 0 or divider_cols[-1] != w - 1:
            continue
        if any(b - a != 4 for a, b in zip(divider_cols, divider_cols[1:], strict=False)):
            continue

        cell_count = len(divider_cols) - 1
        candidate = (cell_count, (r0, c0, r1, c1), color)
        if best is None or candidate > best:
            best = candidate

    assert best is not None, "expected one legend separator component"
    _cell_count, bbox, sep_color = best
    return sep_color, bbox


def _read_legend(grid: np.ndarray, bg: int) -> tuple[int, dict[int, str], tuple[int, int, int, int]]:
    sep_color, (r0, c0, r1, c1) = _find_legend(grid, bg)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    divider_cols = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == sep_color))]

    mapping: dict[int, str] = {}
    for left, right in zip(divider_cols, divider_cols[1:], strict=False):
        patch = sub[1:4, left + 1 : right]
        colors = [int(v) for v in np.unique(patch) if int(v) not in {bg, sep_color}]
        assert len(colors) == 1, "each legend cell must contain one symbol color"
        symbol_color = colors[0]
        fg_mask = patch != bg
        matches = [direction for direction, pattern in PATTERNS.items() if np.array_equal(fg_mask, pattern)]
        assert len(matches) == 1, "legend symbol must encode one opening direction"
        mapping[symbol_color] = matches[0]

    return sep_color, mapping, (r0, c0, r1, c1)


def _find_target_rectangle(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components_of_color(grid, color):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if comp["area"] == h * w and h % 2 == 0 and w % 2 == 0:
            candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "each legend color must match one solid even rectangle"
    return candidates[0]


def _apply_crop(out: np.ndarray, bbox: tuple[int, int, int, int], bg: int, direction: str) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 0 and w % 2 == 0

    if direction == "up":
        out[r0 + h // 2 : r1 + 1, c0 : c1 + 1] = bg
    elif direction == "down":
        out[r0 : r0 + h // 2, c0 : c1 + 1] = bg
    elif direction == "left":
        out[r0 : r1 + 1, c0 + w // 2 : c1 + 1] = bg
    elif direction == "right":
        out[r0 : r1 + 1, c0 : c0 + w // 2] = bg
    else:
        raise AssertionError("unknown direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the opening direction of each top legend symbol to crop the matching lower color block to one half.

    Input structure:
    - The background is Azure.
    - Near the top there is one Grey separator frame made of two full horizontal lines and repeated vertical dividers.
    - Between adjacent Grey dividers, each 3x3 legend cell contains one monochrome open-square symbol whose missing side indicates up, down, left, or right.
    - Elsewhere below the legend, each legend color appears again as exactly one solid even-sized filled rectangle.

    Rule:
    - Keep the Grey legend frame and every legend symbol unchanged.
    - Match each lower solid rectangle to the legend symbol of the same color.
    - Read the symbol's opening direction.
    - Split the rectangle into two equal halves across the corresponding middle line, keep the half toward the opening direction, and recolor the opposite half to the Azure background.

    Key insight:
    - The top symbols are directional instructions rather than decorative shapes: the missing edge of each 3x3 open square tells which half of the same-colored rectangle survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(_most_common_color(g))
    _sep_color, color_to_direction, _legend_bbox = _read_legend(g, bg)

    out = g.copy()
    for color, direction in color_to_direction.items():
        bbox = _find_target_rectangle(g, color)
        _apply_crop(out, bbox, bg, direction)
    return out


def _symbol_patch(color: int, direction: str, bg: int) -> np.ndarray:
    patch = np.full((3, 3), bg, dtype=int)
    patch[PATTERNS[direction]] = int(color)
    return patch


def _build_legend(colors: list[int], directions: list[str], bg: int, sep: int) -> np.ndarray:
    width = 4 * len(colors) + 1
    legend = np.full((5, width), bg, dtype=int)
    legend[0, :] = sep
    legend[-1, :] = sep
    for i, (color, direction) in enumerate(zip(colors, directions, strict=False)):
        c0 = 4 * i
        legend[:, c0] = sep
        legend[:, c0 + 4] = sep
        legend[1:4, c0 + 1 : c0 + 4] = _symbol_patch(color, direction, bg)
    return legend


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _mark(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _direction_list(rng: random.Random, count: int) -> list[str]:
    base = list(DIRECTIONS)
    rng.shuffle(base)
    if count <= 4:
        return base[:count]
    return base + [rng.choice(DIRECTIONS)]


def _expected_output(
    grid: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
    directions: dict[int, str],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for color, bbox in rects.items():
        _apply_crop(out, bbox, bg, directions[color])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    count = rng.randint(2, 5)
    colors = rng.sample(list(PALETTE), count)
    directions = _direction_list(rng, count)
    legend = _build_legend(colors, directions, AZURE, GREY)

    min_h = max(14, 10 + 2 * count)
    min_w = max(16, legend.shape[1] + rng.randint(0, 5))
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    if legend.shape[1] > grid_w or legend.shape[0] > grid_h:
        return None

    legend_r0 = rng.randint(0, min(2, grid_h - legend.shape[0]))
    legend_c0 = rng.randint(0, grid_w - legend.shape[1])
    bottom_start = legend_r0 + legend.shape[0] + rng.randint(1, 4)
    if bottom_start >= grid_h - 1:
        return None

    grid = np.full((grid_h, grid_w), AZURE, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    grid[legend_r0 : legend_r0 + legend.shape[0], legend_c0 : legend_c0 + legend.shape[1]] = legend
    _mark(occupied, legend_r0, legend_c0, legend.shape[0], legend.shape[1])

    rects: dict[int, tuple[int, int, int, int]] = {}
    size_choices_h = [2, 4, 6]
    size_choices_w = [2, 4, 6, 8]
    for color in rng.sample(colors, len(colors)):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            h = rng.choice(size_choices_h)
            w = rng.choice(size_choices_w)
            if bottom_start + h > grid_h:
                continue
            r0 = rng.randint(bottom_start, grid_h - h)
            c0 = rng.randint(0, grid_w - w)
            if not _fits(occupied, r0, c0, h, w):
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = color
            _mark(occupied, r0, c0, h, w)
            rects[color] = (r0, c0, r0 + h - 1, c0 + w - 1)
            placed = True
            break
        if not placed:
            return None

    direction_map = dict(zip(colors, directions, strict=False))
    expected = _expected_output(grid, rects, direction_map, AZURE)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-cropping ARC input")
