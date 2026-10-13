from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


SPRITE_COLORS = (1, 2, 3, 4, 6)
NOISE_COLORS = (5, 7, 8, 9)
PATTERN_LIBRARY: dict[int, list[list[tuple[int, int]]]] = {
    3: [
        [(5, 8), (6, 7), (7, 6)],
        [(5, 7), (6, 8), (7, 6)],
        [(6, 8), (7, 7), (8, 6)],
    ],
    4: [
        [(5, 8), (6, 7), (7, 6), (8, 5)],
        [(5, 7), (6, 8), (7, 6), (8, 5)],
        [(5, 8), (6, 6), (7, 7), (8, 5)],
    ],
    5: [
        [(5, 8), (6, 7), (7, 6), (8, 5), (8, 6)],
        [(5, 7), (6, 8), (7, 6), (8, 5), (8, 6)],
        [(5, 8), (6, 7), (7, 5), (8, 6), (8, 5)],
    ],
}
GEN_RETRIES = 500
ATTEMPT_RETRIES = 300


def _bbox_of(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return int(2 * h + 2 * w - 4)


def _ring_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: set[tuple[int, int]] = set()
    for c in range(int(c0), int(c1) + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(int(r0) + 1, int(r1)):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _fit_ring_template(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None

    r0, c0, r1, c1 = _bbox_of(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if min(h, w) < 5 or max(h, w) > 8:
        return None

    top = bottom = left = right = False
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr not in (r0, r1) and cc not in (c0, c1):
            return None
        if rr == r0:
            top = True
        if rr == r1:
            bottom = True
        if cc == c0:
            left = True
        if cc == c1:
            right = True

    rect = (int(r0), int(c0), int(r1), int(c1))
    area = _ring_area(rect)
    if not (top and bottom and left and right):
        return None
    if int(coords.shape[0]) < int(math.ceil(0.4 * area)):
        return None

    return {
        "color": int(color),
        "rect": rect,
        "cells": _ring_cells(rect),
        "visible": int(coords.shape[0]),
        "area": int(area),
    }


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {0}):
        fitted = _fit_ring_template(grid, int(color))
        if fitted is not None:
            sprites.append(fitted)
    assert 3 <= len(sprites) <= 5, "expected 3 to 5 ring sprites"
    return sprites


def _infer_order(grid: np.ndarray, sprites: list[dict]) -> list[dict]:
    by_color = {sprite["color"]: idx for idx, sprite in enumerate(sprites)}
    coverers: dict[tuple[int, int], list[int]] = {}
    for idx, sprite in enumerate(sprites):
        for cell in sprite["cells"]:
            coverers.setdefault(cell, []).append(int(idx))

    n = len(sprites)
    above: list[set[int]] = [set() for _ in range(n)]
    indeg = [0] * n

    for (r, c), ids in coverers.items():
        if len(ids) < 2:
            continue
        visible_color = int(grid[int(r), int(c)])
        if visible_color not in by_color:
            continue
        top_idx = int(by_color[visible_color])
        if top_idx not in ids:
            continue
        for other in ids:
            if other == top_idx or other in above[top_idx]:
                continue
            above[top_idx].add(int(other))
            indeg[int(other)] += 1

    ordered: list[dict] = []
    remaining = set(range(n))
    while remaining:
        candidates = sorted(idx for idx in remaining if indeg[idx] == 0)
        assert len(candidates) == 1, "occlusion order must be uniquely determined"
        idx = int(candidates[0])
        ordered.append(sprites[idx])
        remaining.remove(idx)
        for nxt in above[idx]:
            indeg[int(nxt)] -= 1

    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapping hollow rectangles, infer their front-to-back order, and invert each row by that row's top sprite color.

    Input structure:
    - The grid background is black (0).
    - Three to five distinct non-black colors belong to one-pixel-thick rectangular ring sprites whose visible pixels still determine each sprite bbox.
    - Additional isolated noise pixels may appear off every sprite template.

    Rule:
    - For each true sprite color, fit the unique rectangular ring whose border contains all visible pixels and ignore colors that do not form such a ring.
    - Compare overlapping ring-template cells: whenever one template cell is visibly occupied by another sprite color, that visible sprite is above the others covering that cell; this yields a unique top-to-bottom order.
    - For each row independently, take the highest-order sprite whose template covers that row as the row's main color.
    - In rows with a main color, change every pixel equal to that color to black and every other pixel, including black, lower-layer colors, and ignored noise, to the main color; rows uncovered by every sprite become all black.

    Key insight:
    - The controlling color is not the most frequent visible color in a row but the color of the frontmost reconstructed sprite whose full template spans that row.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    sprites = _parse_sprites(g)
    ordered = _infer_order(g, sprites)

    out = np.zeros((H, W), dtype=int)
    for r in range(H):
        main_color = None
        for sprite in ordered:
            r0, _c0, r1, _c1 = sprite["rect"]
            if int(r0) <= r <= int(r1):
                main_color = int(sprite["color"])
                break
        if main_color is None:
            continue
        out[r, :] = np.where(g[r, :] == int(main_color), 0, int(main_color))
    return out


def _render_input(
    shape: tuple[int, int], sprites: list[dict], noise: list[tuple[int, int, int]]
) -> np.ndarray:
    H, W = shape
    grid = np.zeros((H, W), dtype=int)
    for sprite in reversed(sprites):
        for r, c in sprite["cells"]:
            grid[int(r), int(c)] = int(sprite["color"])
    for r, c, color in noise:
        grid[int(r), int(c)] = int(color)
    return grid


def _expected_output(shape: tuple[int, int], sprites: list[dict], inp: np.ndarray) -> np.ndarray:
    H, W = shape
    out = np.zeros((H, W), dtype=int)
    for r in range(H):
        row_color = None
        for sprite in sprites:
            r0, _c0, r1, _c1 = sprite["rect"]
            if int(r0) <= r <= int(r1):
                row_color = int(sprite["color"])
                break
        if row_color is None:
            continue
        out[r, :] = np.where(inp[r, :] == int(row_color), 0, int(row_color))
    return out


def _row_main_colors(shape: tuple[int, int], sprites: list[dict]) -> list[int]:
    H, _W = shape
    row_colors: list[int] = []
    for r in range(H):
        color = 0
        for sprite in sprites:
            r0, _c0, r1, _c1 = sprite["rect"]
            if int(r0) <= r <= int(r1):
                color = int(sprite["color"])
                break
        row_colors.append(int(color))
    return row_colors


def _intersections_ok(sprites: list[dict]) -> bool:
    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            if len(sprites[i]["cells"] & sprites[j]["cells"]) < 2:
                return False
    return True


def _build_rect(center_r: int, center_c: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0 = int(center_r - h // 2)
    c0 = int(center_c - w // 2)
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _sample_sprites() -> tuple[tuple[int, int], list[dict]] | None:
    n = random.choices([3, 4, 5], weights=[4, 4, 3], k=1)[0]
    H = random.randint(18, 24)
    W = random.randint(18, 24)
    dims = [tuple(pair) for pair in random.choice(PATTERN_LIBRARY[n])]
    colors = random.sample(SPRITE_COLORS, n)

    base_r = random.randint(7, H - 8)
    base_c = random.randint(7, W - 8)

    sprites: list[dict] = []
    for idx, (h, w) in enumerate(dims):
        placed = None
        for _ in range(60):
            dr = random.randint(-2, 2)
            dc = random.randint(-2, 2)
            if idx > 0 and random.random() < 0.5:
                dr += 1 if idx % 2 else -1
            rect = _build_rect(int(base_r + dr), int(base_c + dc), int(h), int(w))
            r0, c0, r1, c1 = rect
            if r0 < 1 or c0 < 1 or r1 >= H - 1 or c1 >= W - 1:
                continue
            placed = {
                "color": int(colors[idx]),
                "rect": rect,
                "cells": _ring_cells(rect),
                "area": _ring_area(rect),
            }
            break
        if placed is None:
            return None
        sprites.append(placed)

    if not _intersections_ok(sprites):
        return None
    return (int(H), int(W)), sprites


def _pick_noise(shape: tuple[int, int], sprites: list[dict]) -> list[tuple[int, int, int]]:
    H, W = shape
    covered_rows = {r for sprite in sprites for r in range(sprite["rect"][0], sprite["rect"][2] + 1)}
    template_cells = set().union(*(sprite["cells"] for sprite in sprites))
    candidates: list[tuple[int, int]] = []
    for r in covered_rows:
        for c in range(W):
            if (int(r), int(c)) in template_cells:
                continue
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)

    noise: list[tuple[int, int, int]] = []
    used_cells: set[tuple[int, int]] = set()
    count = random.choices([0, 1, 2], weights=[1, 4, 4], k=1)[0]
    palette = list(NOISE_COLORS)
    random.shuffle(palette)
    for r, c in candidates:
        if len(noise) >= count:
            break
        if (r, c) in used_cells:
            continue
        color = palette[len(noise) % len(palette)]
        noise.append((int(r), int(c), int(color)))
        used_cells.add((int(r), int(c)))
    return noise


def _semantic_checks(inp: np.ndarray, sprites: list[dict], noise: list[tuple[int, int, int]]) -> bool:
    try:
        parsed = _parse_sprites(inp)
    except AssertionError:
        return False
    parsed_by_color = {sprite["color"]: sprite for sprite in parsed}

    if set(parsed_by_color) != {sprite["color"] for sprite in sprites}:
        return False
    for sprite in sprites:
        parsed_sprite = parsed_by_color.get(sprite["color"])
        if parsed_sprite is None or parsed_sprite["rect"] != sprite["rect"]:
            return False
        if parsed_sprite["visible"] < int(math.ceil(0.4 * sprite["area"])):
            return False

    try:
        ordered = _infer_order(inp, parsed)
    except AssertionError:
        return False
    if [sprite["color"] for sprite in ordered] != [sprite["color"] for sprite in sprites]:
        return False

    row_colors = _row_main_colors(inp.shape, sprites)
    if len({color for color in row_colors if color != 0}) < 2:
        return False

    freq_counterexample = False
    for r, main_color in enumerate(row_colors):
        if main_color == 0:
            continue
        counts = Counter(int(v) for v in inp[r, :] if int(v) != 0)
        if not counts:
            continue
        if counts.most_common(1)[0][0] != int(main_color):
            freq_counterexample = True
            break
    if not freq_counterexample:
        return False

    for r, c, _color in noise:
        if any((int(r), int(c)) in sprite["cells"] for sprite in sprites):
            return False
        if row_colors[int(r)] == 0:
            return False

    expected = _expected_output(inp.shape, sprites, inp)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, actual):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    sampled = _sample_sprites()
    if sampled is None:
        return None
    shape, sprites = sampled
    noise = _pick_noise(shape, sprites)
    inp = _render_input(shape, sprites, noise)
    if not _semantic_checks(inp, sprites, noise):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        for _ in range(ATTEMPT_RETRIES):
            grid = _generate_attempt()
            if grid is not None:
                return grid
    raise RuntimeError("generate_input failed to construct a valid occlusion-order sample")
