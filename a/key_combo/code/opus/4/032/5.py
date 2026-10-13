from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
TEAL = 8
MAROON = 9

COLOR_TO_HEIGHT = {
    RED: 2,
    YELLOW: 4,
    GREEN: 6,
    TEAL: 3,
}

GLYPHS = {
    RED: np.array(
        [
            [GRAY, GRAY, GRAY, GRAY],
            [GRAY, GRAY, GRAY, GRAY],
            [GRAY, GRAY, GRAY, GRAY],
            [GRAY, GRAY, GRAY, GRAY],
        ],
        dtype=int,
    ),
    TEAL: np.array(
        [
            [GRAY, GRAY, GRAY, GRAY],
            [GRAY, BLACK, BLACK, GRAY],
            [GRAY, BLACK, BLACK, GRAY],
            [GRAY, GRAY, GRAY, GRAY],
        ],
        dtype=int,
    ),
    YELLOW: np.array(
        [
            [GRAY, GRAY, GRAY, GRAY],
            [GRAY, GRAY, GRAY, GRAY],
            [GRAY, BLACK, BLACK, GRAY],
            [GRAY, BLACK, BLACK, GRAY],
        ],
        dtype=int,
    ),
    GREEN: np.array(
        [
            [GRAY, GRAY, GRAY, GRAY],
            [BLACK, GRAY, GRAY, BLACK],
            [BLACK, GRAY, GRAY, BLACK],
            [GRAY, GRAY, GRAY, GRAY],
        ],
        dtype=int,
    ),
}

PATTERN_TO_COLOR = {tuple(pattern.flatten().tolist()): color for color, pattern in GLYPHS.items()}


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in dirs:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _extract_glyphs(grid: np.ndarray) -> list[dict]:
    comps = _components_mask(grid == GRAY, connectivity=8)
    glyphs: list[dict] = []
    for coords in comps:
        box = _bbox_coords(coords)
        patch = _crop_bbox(grid, box)
        if patch.shape != (4, 4):
            raise ValueError("expected every glyph bbox to be 4x4")
        key = tuple(patch.flatten().tolist())
        if key not in PATTERN_TO_COLOR:
            raise ValueError("unrecognized glyph pattern")
        glyphs.append({"bbox": box, "patch": patch, "color": PATTERN_TO_COLOR[key]})
    glyphs.sort(key=lambda rec: rec["bbox"][1])
    return glyphs


def _terrain_output(grid: np.ndarray, glyphs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    heights = [COLOR_TO_HEIGHT[int(rec["color"])] for rec in glyphs]

    for rec, height in zip(glyphs, heights):
        _r0, c0, _r1, c1 = rec["bbox"]
        top = out.shape[0] - height
        out[top:, c0 : c1 + 1] = int(rec["color"])

    tallest = max(range(len(glyphs)), key=lambda idx: (heights[idx], idx))
    shortest = min(range(len(glyphs)), key=lambda idx: (heights[idx], -idx))

    tall_box = glyphs[tallest]["bbox"]
    tall_top = out.shape[0] - heights[tallest]
    out[tall_top, tall_box[1] : tall_box[3] + 1] = MAROON

    short_box = glyphs[shortest]["bbox"]
    short_top = out.shape[0] - heights[shortest]
    out[short_top, short_box[1] : short_box[3] + 1] = MAGENTA

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode top-row glyphs into terrain columns and mark the highest and lowest band tops.

    Input structure:
    - A black grid contains 3 or 4 gray 4x4 glyphs near the top, separated left-to-right by exactly one black column.
    - Each glyph is one of four fixed gray/black masks, and the rest of the grid is black.
    - The lower portion is empty input space where terrain must be constructed.

    Rule:
    - Find the gray glyph components, crop each 4x4 block, and decode its mask to one of four colors.
    - Convert each decoded color to a terrain height: red->2, yellow->4, green->6, teal->3.
    - For each glyph, fill the aligned 4-column band at the bottom of the grid up to that height using the glyph's decoded color, preserving the 1-column gaps.
    - Recolor the top row of the tallest band maroon and the top row of the shortest band magenta; when an extreme height is tied, choose the rightmost tied band.
    - Leave the original glyphs unchanged.

    Key insight:
    - The glyphs do not encode final output colors directly; they first decode to colors, those colors then act as keys for hidden terrain heights, and only the finished terrain reveals which bands receive the extremum markers.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    glyphs = _extract_glyphs(g)
    if len(glyphs) < 1:
        raise ValueError("expected at least one glyph")
    return _terrain_output(g, glyphs)


def _is_monotone(values: list[int]) -> bool:
    nondecreasing = all(a <= b for a, b in zip(values, values[1:]))
    nonincreasing = all(a >= b for a, b in zip(values, values[1:]))
    return nondecreasing or nonincreasing


def _choose_sequence(rng: random.Random) -> list[int]:
    mode = rng.choice(["three_distinct", "four_distinct", "tie_tallest", "tie_shortest"])
    if mode == "three_distinct":
        seq = rng.sample([RED, YELLOW, GREEN, TEAL], 3)
    elif mode == "four_distinct":
        seq = rng.sample([RED, YELLOW, GREEN, TEAL], 4)
    elif mode == "tie_tallest":
        seq = [GREEN, GREEN] + rng.sample([RED, YELLOW, TEAL], 2)
        rng.shuffle(seq)
    else:
        seq = [RED, RED] + rng.sample([GREEN, YELLOW, TEAL], 2)
        rng.shuffle(seq)

    heights = [COLOR_TO_HEIGHT[color] for color in seq]
    if _is_monotone(heights):
        return _choose_sequence(rng)
    return seq


def _build_input_from_sequence(seq: list[int], rng: random.Random) -> np.ndarray:
    n = len(seq)
    top_row = rng.randint(0, 1)
    left_margin = rng.randint(0, 4)
    right_margin = rng.randint(0, 4)
    height = rng.randint(15, 20)
    width = left_margin + 4 * n + (n - 1) + right_margin
    grid = np.zeros((height, width), dtype=int)

    for idx, color in enumerate(seq):
        c0 = left_margin + idx * 5
        grid[top_row : top_row + 4, c0 : c0 + 4] = GLYPHS[color]
    return grid


def _validate_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    glyphs = _extract_glyphs(g)
    assert 3 <= len(glyphs) <= 4
    assert len({int(rec["color"]) for rec in glyphs}) >= 3
    assert all(rec["bbox"][2] <= 4 for rec in glyphs), "glyphs must stay in the top 5 rows"
    assert np.all(g[5:, :] == BLACK), "bottom region must be empty in the input"

    for left, right in zip(glyphs, glyphs[1:]):
        assert right["bbox"][1] - left["bbox"][3] == 2, "glyphs must have a 1-column gap"

    heights = [COLOR_TO_HEIGHT[int(rec["color"])] for rec in glyphs]
    assert len(set(heights)) >= 3
    assert any(color != height for color, height in zip([int(rec["color"]) for rec in glyphs], heights))
    assert not _is_monotone(heights), "sample should rule out simple sorting shortcuts"

    out = main(g)
    expected = _terrain_output(g, glyphs)
    assert np.array_equal(out, expected)
    assert np.array_equal(out[:5, :], g[:5, :]), "top glyph area must remain unchanged"

    tallest = max(range(len(glyphs)), key=lambda idx: (heights[idx], idx))
    shortest = min(range(len(glyphs)), key=lambda idx: (heights[idx], -idx))
    for idx, (rec, height) in enumerate(zip(glyphs, heights)):
        _r0, c0, _r1, c1 = rec["bbox"]
        top = g.shape[0] - height
        band = out[top:, c0 : c1 + 1]
        body_color = int(rec["color"])
        if idx == tallest:
            assert np.all(out[top, c0 : c1 + 1] == MAROON)
            assert np.all(band[1:, :] == body_color)
        elif idx == shortest:
            assert np.all(out[top, c0 : c1 + 1] == MAGENTA)
            assert np.all(band[1:, :] == body_color)
        else:
            assert np.all(band == body_color)

    gaps = [glyphs[i]["bbox"][3] + 1 for i in range(len(glyphs) - 1)]
    for col in gaps:
        assert np.all(out[:, col] == BLACK), "separator columns must stay black"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        seq = _choose_sequence(rng)
        grid = _build_input_from_sequence(seq, rng)
        try:
            _validate_case(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid non-degenerate puzzle instance")
