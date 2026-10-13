from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
GEN_TOP_HEIGHT_RANGE = (3, 8)
GEN_BOTTOM_MARGIN = 1
GEN_BOTTOM_EXTRA_RANGE = (1, 8)
GEN_WIDTH_RANGE = (8, 16)
GEN_EXTRA_COLUMN_RANGE = (0, 4)

_GEN_COUNTER = 0


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one non-axis cell"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant background"
    return int(ranked[0][0])


def _axis_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one full gray fold axis"
    return int(rows[0])


def _gravity(region: np.ndarray, bg: int) -> np.ndarray:
    out = np.full_like(region, bg)
    height, width = map(int, region.shape)
    for c in range(width):
        values = [int(v) for v in region[:, c] if int(v) != bg]
        if values:
            out[height - len(values) :, c] = np.asarray(values, dtype=region.dtype)
    return out


def _fold_into_bottom(top: np.ndarray, bottom: np.ndarray, bg: int) -> tuple[np.ndarray, dict[str, int]]:
    assert top.shape[0] <= bottom.shape[0], "top half may not be taller than the bottom half"

    out = np.array(bottom, copy=True)
    overlay = np.flipud(top)
    stats = {"top_pixels": 0, "inserted": 0, "cancellations": 0}

    for r in range(overlay.shape[0]):
        for c in range(overlay.shape[1]):
            color = int(overlay[r, c])
            if color == bg:
                continue

            stats["top_pixels"] += 1
            existing = int(out[r, c])
            assert existing in (bg, color), "folded pixels may only hit background or same-color pixels"
            if existing == bg:
                out[r, c] = color
                stats["inserted"] += 1
            else:
                out[r, c] = bg
                stats["cancellations"] += 1

    return out, stats


def _split(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    axis = _axis_row(g)
    body = np.concatenate((g[:axis, :].reshape(-1), g[axis + 1 :, :].reshape(-1)))
    bg = _dominant_color(body)
    assert bg != GREY, "background cannot be the fold-axis color"
    top = g[:axis, :]
    bottom = g[axis + 1 :, :]
    assert top.shape[0] < bottom.shape[0], "the gray axis must not be centered"
    return top, bottom, bg


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    top, bottom, bg = _split(grid)
    folded, stats = _fold_into_bottom(top, bottom, bg)
    dropped = _gravity(folded, bg)
    return dropped, stats, folded


def _gravity_first_then_fold(grid: np.ndarray) -> np.ndarray:
    top, bottom, bg = _split(grid)
    top_g = _gravity(top, bg)
    bottom_g = _gravity(bottom, bg)
    folded, _stats = _fold_into_bottom(top_g, bottom_g, bg)
    return _gravity(folded, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the upper fragments across the gray horizontal axis, cancel same-color overlaps, then drop the survivors to the bottom.

    Input structure:
    - A full gray horizontal line divides the canvas into an upper region and a taller lower region.
    - Non-gray colored pixels are sparse single-cell fragments placed above and below that line.
    - The upper half is no taller than the lower half, so every upper pixel has a mirrored landing cell below the axis.

    Rule:
    - Remove the gray line from consideration and mirror every upper-region pixel downward across it.
    - At each mirrored landing cell, a same-color lower pixel annihilates with the folded pixel and both disappear; an empty landing cell keeps the folded pixel.
    - After all fold collisions are resolved, apply column-wise gravity inside the lower region so every surviving pixel falls straight down and stacks at the bottom.

    Key insight:
    - The puzzle is not two independent gravity problems: collisions must be decided at the original mirrored coordinates before any global falling happens, and only then are the surviving lower pixels compacted downward.
    """
    out, _stats, _folded = _solve(grid)
    return out


def _pick_color(palette: list[int], *, exclude: set[int] | None = None) -> int:
    blocked = exclude or set()
    options = [color for color in palette if color not in blocked]
    return int(random.choice(options or palette))


def _set_top(top: np.ndarray, row: int, col: int, color: int) -> None:
    assert int(top[row, col]) == BACKGROUND
    top[row, col] = int(color)


def _set_bottom(bottom: np.ndarray, row: int, col: int, color: int) -> None:
    assert int(bottom[row, col]) == BACKGROUND
    bottom[row, col] = int(color)


def _mapped_row(top_h: int, top_row: int) -> int:
    return int(top_h - 1 - top_row)


def _place_order_witness(top: np.ndarray, bottom: np.ndarray, col: int, palette: list[int]) -> None:
    color = _pick_color(palette)
    top_row = random.randint(0, top.shape[0] - 2)
    _set_top(top, top_row, col, color)
    _set_bottom(bottom, 0, col, color)


def _place_cancel_pair(top: np.ndarray, bottom: np.ndarray, col: int, palette: list[int]) -> None:
    color = _pick_color(palette)
    mapped = random.randint(0, top.shape[0] - 1)
    top_row = top.shape[0] - 1 - mapped
    _set_top(top, top_row, col, color)
    _set_bottom(bottom, mapped, col, color)

    extra_rows = [r for r in range(mapped + 1, bottom.shape[0] - 1) if int(bottom[r, col]) == BACKGROUND]
    if extra_rows and random.random() < 0.7:
        _set_bottom(bottom, random.choice(extra_rows), col, _pick_color(palette, exclude={color}))


def _place_top_drop(top: np.ndarray, bottom: np.ndarray, col: int, palette: list[int], *, allow_double: bool) -> None:
    count = 2 if allow_double and top.shape[0] >= 4 and random.random() < 0.6 else 1
    rows = random.sample(range(top.shape[0]), count)
    for row in rows:
        _set_top(top, row, col, _pick_color(palette))


def _place_mixed_stack(top: np.ndarray, bottom: np.ndarray, col: int, palette: list[int]) -> None:
    top_color = _pick_color(palette)
    bottom_color = _pick_color(palette, exclude={top_color})
    top_row = random.randint(0, top.shape[0] - 1)
    mapped = _mapped_row(top.shape[0], top_row)
    bottom_rows = [r for r in range(1, bottom.shape[0] - 1) if r != mapped]
    assert bottom_rows, "expected space for a non-colliding bottom fragment"
    bottom_row = random.choice(bottom_rows)
    _set_top(top, top_row, col, top_color)
    _set_bottom(bottom, bottom_row, col, bottom_color)


def _place_bottom_only(top: np.ndarray, bottom: np.ndarray, col: int, palette: list[int], *, allow_double: bool) -> None:
    del top
    candidates = list(range(bottom.shape[0] - 1))
    count = 2 if allow_double and bottom.shape[0] >= 4 and random.random() < 0.5 else 1
    rows = random.sample(candidates, count)
    used: set[int] = set()
    for row in rows:
        color = _pick_color(palette, exclude=used if len(palette) > len(used) else None)
        used.add(color)
        _set_bottom(bottom, row, col, color)


def _extra_patterns(mode: int) -> list[str]:
    if mode == 0:
        return ["top_drop", "mixed_stack"]
    if mode == 1:
        return ["cancel_pair", "bottom_only"]
    if mode == 2:
        return ["top_drop", "bottom_only", "mixed_stack"]
    return ["cancel_pair"]


def _render_grid(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    axis = np.full((1, top.shape[1]), GREY, dtype=int)
    return np.vstack((top, axis, bottom))


def _semantic_checks(inp: np.ndarray) -> bool:
    out, stats, folded = _solve(inp)
    wrong = _gravity_first_then_fold(inp)

    if stats["cancellations"] < 1:
        return False
    if stats["inserted"] < 2:
        return False
    if np.array_equal(out, folded):
        return False
    if np.array_equal(out, wrong):
        return False
    if not np.any(out != BACKGROUND):
        return False

    axis = _axis_row(inp)
    bottom = inp[axis + 1 :, :]
    if np.count_nonzero(bottom != BACKGROUND) < 2:
        return False
    return True


def _build_attempt(mode: int) -> np.ndarray:
    top_h = random.randint(*GEN_TOP_HEIGHT_RANGE)
    bottom_h = top_h + random.randint(*GEN_BOTTOM_EXTRA_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    assert top_h + 1 + bottom_h <= 30

    top = np.full((top_h, width), BACKGROUND, dtype=int)
    bottom = np.full((bottom_h, width), BACKGROUND, dtype=int)
    palette_size = random.randint(3, 5)
    palette = random.sample(COLOR_POOL, palette_size)

    mandatory = ["order_witness", "cancel_pair", "top_drop", "mixed_stack", "bottom_only"]
    extra_count = min(width - len(mandatory), random.randint(*GEN_EXTRA_COLUMN_RANGE))
    columns = random.sample(range(width), len(mandatory) + extra_count)

    for col, pattern in zip(columns[: len(mandatory)], mandatory):
        if pattern == "order_witness":
            _place_order_witness(top, bottom, col, palette)
        elif pattern == "cancel_pair":
            _place_cancel_pair(top, bottom, col, palette)
        elif pattern == "top_drop":
            _place_top_drop(top, bottom, col, palette, allow_double=True)
        elif pattern == "mixed_stack":
            _place_mixed_stack(top, bottom, col, palette)
        else:
            _place_bottom_only(top, bottom, col, palette, allow_double=True)

    for col in columns[len(mandatory) :]:
        pattern = random.choice(_extra_patterns(mode))
        if pattern == "cancel_pair":
            _place_cancel_pair(top, bottom, col, palette)
        elif pattern == "mixed_stack":
            _place_mixed_stack(top, bottom, col, palette)
        elif pattern == "bottom_only":
            _place_bottom_only(top, bottom, col, palette, allow_double=bool(mode % 2))
        else:
            _place_top_drop(top, bottom, col, palette, allow_double=True)

    return _render_grid(top, bottom)


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(GEN_RETRIES):
        mode = _GEN_COUNTER % 4
        _GEN_COUNTER += 1
        inp = _build_attempt(mode)
        if _semantic_checks(inp):
            return inp
    raise RuntimeError("failed to generate a valid fold-annihilate-gravity input")
