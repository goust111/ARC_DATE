from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, bbox, components_mask, crop_bbox, get_bg, in_bounds, paste


def _detect_mirrors(grid: np.ndarray) -> tuple[str, tuple[int, int], int, int]:
    h, w = map(int, grid.shape)
    grey = int(Color.GREY)
    grey_rows = [r for r in range(h) if np.all(grid[r, :] == grey)]
    grey_cols = [c for c in range(w) if np.all(grid[:, c] == grey)]

    if len(grey_cols) == 2 and not grey_rows:
        c1, c2 = sorted(grey_cols)
        assert c2 - c1 >= 2, "vertical mirrors must leave interior space"
        return "vertical", (c1, c2), h, w
    if len(grey_rows) == 2 and not grey_cols:
        r1, r2 = sorted(grey_rows)
        assert r2 - r1 >= 2, "horizontal mirrors must leave interior space"
        return "horizontal", (r1, r2), h, w
    raise AssertionError("expected exactly two full grey parallel mirror lines")


def _extract_sprite(grid: np.ndarray, orientation: str, mirrors: tuple[int, int], bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    grey = int(Color.GREY)
    if orientation == "vertical":
        c1, c2 = mirrors
        assert np.all(grid[:, :c1] == bg), "left exterior must be background"
        assert np.all(grid[:, c2 + 1 :] == bg), "right exterior must be background"
        inner = grid[:, c1 + 1 : c2]
        offset = (0, c1 + 1)
    else:
        r1, r2 = mirrors
        assert np.all(grid[:r1, :] == bg), "upper exterior must be background"
        assert np.all(grid[r2 + 1 :, :] == bg), "lower exterior must be background"
        inner = grid[r1 + 1 : r2, :]
        offset = (r1 + 1, 0)

    mask = (inner != bg) & (inner != grey)
    comps = components_mask(mask, connectivity=8)
    assert len(comps) == 1, "expected exactly one sprite between the mirrors"
    inner_box = bbox(mask)
    assert inner_box is not None
    r0, c0, r1, c1 = inner_box
    abs_box = (r0 + offset[0], c0 + offset[1], r1 + offset[0], c1 + offset[1])
    return abs_box, crop_bbox(grid, abs_box)


def _reflect_fill_vertical(out: np.ndarray, sprite: np.ndarray, sprite_box: tuple[int, int, int, int], mirrors: tuple[int, int], bg: int) -> np.ndarray:
    h, w = map(int, out.shape)
    r0, c0, _r1, c1 = sprite_box
    left_mirror, right_mirror = mirrors
    period = right_mirror - left_mirror
    rel_left = c0 - left_mirror
    rel_right = right_mirror - c1
    max_k = w // period + 3

    for k in range(-max_k, max_k + 1):
        if k == 0:
            continue
        strip_left = left_mirror + k * period
        if k % 2 == 0:
            patch = sprite
            dst_c = c0 + k * period
        else:
            patch = np.fliplr(sprite)
            dst_c = strip_left + rel_right
        if dst_c >= w or dst_c + patch.shape[1] <= 0:
            continue
        paste(out, patch, r0, dst_c, transparent=bg, clip=True)
    return out


def _reflect_fill_horizontal(out: np.ndarray, sprite: np.ndarray, sprite_box: tuple[int, int, int, int], mirrors: tuple[int, int], bg: int) -> np.ndarray:
    h, w = map(int, out.shape)
    r0, c0, r1, _c1 = sprite_box
    top_mirror, bottom_mirror = mirrors
    period = bottom_mirror - top_mirror
    rel_top = r0 - top_mirror
    rel_bottom = bottom_mirror - r1
    max_k = h // period + 3

    for k in range(-max_k, max_k + 1):
        if k == 0:
            continue
        strip_top = top_mirror + k * period
        if k % 2 == 0:
            patch = sprite
            dst_r = r0 + k * period
        else:
            patch = np.flipud(sprite)
            dst_r = strip_top + rel_bottom
        if dst_r >= h or dst_r + patch.shape[0] <= 0:
            continue
        paste(out, patch, dst_r, c0, transparent=bg, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the sprite by alternating mirror reflections between the two grey parallel lines.

    Input structure:
    - The grid has a black background, plus exactly two complete grey parallel lines.
    - The lines are both vertical or both horizontal, and a single multicolor sprite lies fully between them.
    - Outside the two mirror lines, the input contains only background.

    Rule:
    - Keep the original sprite and both grey mirror lines unchanged.
    - Reflect the sprite across each grey line to create the first outer images, then keep reflecting each new image across the opposite line.
    - Continue this alternating process as an infinite strip pattern with period twice the mirror spacing, clipping any parts that leave the canvas.

    Key insight:
    - The far images are not made by one direct mirror of the original; they are images of images, so orientation alternates while colors stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    orientation, mirrors, _h, _w = _detect_mirrors(g)
    sprite_box, sprite = _extract_sprite(g, orientation, mirrors, bg)

    out = g.copy()
    if orientation == "vertical":
        return _reflect_fill_vertical(out, sprite, sprite_box, mirrors, bg)
    return _reflect_fill_horizontal(out, sprite, sprite_box, mirrors, bg)


_PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_MAX_ATTEMPTS = 500


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in zip(*np.where(mask)):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.append((rr, cc))
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    return mask


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask.astype(int), box).astype(bool)


def _make_sprite(max_h: int, max_w: int, active_axis: str) -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, max_h)
        w = random.randint(2, max_w) if active_axis == "vertical" else random.randint(3, max_w)
        if active_axis == "horizontal":
            h = random.randint(2, max_h)
            w = random.randint(3, max_w)
        area_lo = max(4, (h * w + 2) // 3)
        area_hi = max(area_lo, h * w - random.randint(0, max(1, (h * w) // 4)))
        mask = _tight_crop_mask(_grow_connected_mask(h, w, random.randint(area_lo, area_hi)))
        h2, w2 = map(int, mask.shape)
        if active_axis == "vertical" and w2 > max_w:
            continue
        if active_axis == "horizontal" and h2 > max_h:
            continue

        n_colors = random.randint(2, min(3, int(mask.sum())))
        colors = random.sample(_PALETTE, n_colors)
        coords = [tuple(x) for x in np.argwhere(mask)]
        random.shuffle(coords)
        sprite = np.zeros((h2, w2), dtype=int)
        for idx, (r, c) in enumerate(coords):
            if idx < n_colors:
                sprite[r, c] = colors[idx]
            else:
                sprite[r, c] = random.choice(colors)

        if len({int(x) for x in sprite[mask]}) < 2:
            continue
        if active_axis == "vertical":
            if np.array_equal(sprite, np.fliplr(sprite)):
                continue
        else:
            if np.array_equal(sprite, np.flipud(sprite)):
                continue
        return sprite
    raise RuntimeError("failed to build an asymmetric sprite")


def _independent_expected(grid: np.ndarray) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    orientation, mirrors, h, w = _detect_mirrors(g)
    sprite_box, sprite = _extract_sprite(g, orientation, mirrors, bg)
    out = g.copy()
    placed: dict[int, set[tuple[int, int]]] = {}

    r0, c0, r1, c1 = sprite_box
    if orientation == "vertical":
        left_mirror, right_mirror = mirrors
        period = right_mirror - left_mirror
        max_k = w // period + 3
        for rr in range(sprite.shape[0]):
            for cc in range(sprite.shape[1]):
                color = int(sprite[rr, cc])
                if color == bg:
                    continue
                src_r = r0 + rr
                src_c = c0 + cc
                offset = src_c - left_mirror
                for k in range(-max_k, max_k + 1):
                    if k == 0:
                        continue
                    if k % 2 == 0:
                        dst_c = src_c + k * period
                    else:
                        dst_c = left_mirror + (k + 1) * period - offset
                    if in_bounds((h, w), src_r, dst_c):
                        out[src_r, dst_c] = color
                        placed.setdefault(k, set()).add((src_r, dst_c))
    else:
        top_mirror, bottom_mirror = mirrors
        period = bottom_mirror - top_mirror
        max_k = h // period + 3
        for rr in range(sprite.shape[0]):
            for cc in range(sprite.shape[1]):
                color = int(sprite[rr, cc])
                if color == bg:
                    continue
                src_r = r0 + rr
                src_c = c0 + cc
                offset = src_r - top_mirror
                for k in range(-max_k, max_k + 1):
                    if k == 0:
                        continue
                    if k % 2 == 0:
                        dst_r = src_r + k * period
                    else:
                        dst_r = top_mirror + (k + 1) * period - offset
                    if in_bounds((h, w), dst_r, src_c):
                        out[dst_r, src_c] = color
                        placed.setdefault(k, set()).add((dst_r, src_c))
    return out, placed


def _validate_instance(inp: np.ndarray, expected: np.ndarray, placed: dict[int, set[tuple[int, int]]]) -> None:
    actual = main(inp)
    assert np.array_equal(actual, expected), "main() does not match the independent construction"
    assert placed.get(-1), "left/up first-generation image must be visible"
    assert placed.get(1), "right/down first-generation image must be visible"
    assert any(placed.get(k) for k in placed if abs(k) >= 2), "at least one second-generation image must be visible"

    bg = int(Color.BLACK)
    orientation, mirrors, _h, _w = _detect_mirrors(inp)
    sprite_box, sprite = _extract_sprite(inp, orientation, mirrors, bg)
    if orientation == "vertical":
        assert not np.array_equal(sprite, np.fliplr(sprite)), "sprite must visibly change when mirrored"
    else:
        assert not np.array_equal(sprite, np.flipud(sprite)), "sprite must visibly change when mirrored"
    assert len(np.unique(sprite[sprite != bg])) >= 2, "sprite should be visibly multicolor"


def _build_vertical_input() -> np.ndarray:
    period = random.randint(4, 7)
    tall = random.randint(9, 18)
    long_side = random.choice(["left", "right"])
    left_margin = 2 * period if long_side == "left" else period
    right_margin = 2 * period if long_side == "right" else period
    width = left_margin + period + 1 + right_margin
    assert width <= 30

    sprite = _make_sprite(tall - 2, period - 1, "vertical")
    sh, sw = map(int, sprite.shape)

    grid = np.zeros((tall, width), dtype=int)
    c1 = left_margin
    c2 = c1 + period
    grid[:, c1] = Color.GREY
    grid[:, c2] = Color.GREY
    r0 = random.randint(0, tall - sh)
    c0 = random.randint(c1 + 1, c2 - sw)
    paste(grid, sprite, r0, c0, transparent=Color.BLACK)
    return grid


def _build_horizontal_input() -> np.ndarray:
    period = random.randint(4, 7)
    wide = random.randint(9, 18)
    long_side = random.choice(["up", "down"])
    top_margin = 2 * period if long_side == "up" else period
    bottom_margin = 2 * period if long_side == "down" else period
    height = top_margin + period + 1 + bottom_margin
    assert height <= 30

    sprite = _make_sprite(period - 1, wide - 2, "horizontal")
    sh, sw = map(int, sprite.shape)

    grid = np.zeros((height, wide), dtype=int)
    r1 = top_margin
    r2 = r1 + period
    grid[r1, :] = Color.GREY
    grid[r2, :] = Color.GREY
    r0 = random.randint(r1 + 1, r2 - sh)
    c0 = random.randint(0, wide - sw)
    paste(grid, sprite, r0, c0, transparent=Color.BLACK)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        inp = _build_vertical_input() if random.choice([False, True]) else _build_horizontal_input()
        expected, placed = _independent_expected(inp)
        try:
            _validate_instance(inp, expected, placed)
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid mirror-of-mirror ARC instance")
