from __future__ import annotations

import itertools
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
MAROON = 9

SEED_FILL = {
    BLUE: RED,
    YELLOW: PINK,
    GREY: MAROON,
}

SEED_COLORS = tuple(SEED_FILL)
VERTICAL = None

MAX_ATTEMPTS = 500
LOWER_TOP = 4  # 2 strip rows + 2 separator rows


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _parse_glyphs(strip: np.ndarray) -> list[int | None]:
    strip = np.asarray(strip, dtype=int)
    assert strip.shape[0] == 2, "glyph strip must occupy exactly two rows"
    assert strip.shape[1] % 2 == 0, "glyph strip width must be even"
    allowed = {BLACK, RED}
    assert set(map(int, np.unique(strip))).issubset(allowed)

    glyphs: list[int | None] = []
    c = 0
    while c < strip.shape[1]:
        if np.all(strip[:, c] == BLACK):
            c += 1
            continue

        start = c
        while c < strip.shape[1] and not np.all(strip[:, c] == BLACK):
            c += 1
        block = strip[:, start:c]

        if block.shape == (2, 1):
            assert np.array_equal(block, np.array([[RED], [RED]])), "bad vertical glyph"
            glyphs.append(VERTICAL)
            continue

        assert block.shape == (2, 2), "bad elbow width"
        if np.array_equal(block, np.array([[BLACK, RED], [RED, RED]])):
            glyphs.append(1)
        elif np.array_equal(block, np.array([[RED, BLACK], [RED, RED]])):
            glyphs.append(-1)
        else:
            raise AssertionError("bad elbow glyph")

    assert 4 <= len(glyphs) <= 7, "glyph count must be in [4, 7]"
    return glyphs


def _encode_strip(glyphs: list[int | None]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for step in glyphs:
        if step is VERTICAL:
            pieces.append(np.array([[RED], [RED]], dtype=int))
        elif step == 1:
            pieces.append(np.array([[BLACK, RED], [RED, RED]], dtype=int))
        elif step == -1:
            pieces.append(np.array([[RED, BLACK], [RED, RED]], dtype=int))
        else:
            raise AssertionError("unexpected glyph step")

    width = sum(piece.shape[1] for piece in pieces) + len(pieces) - 1
    strip = np.full((2, width), BLACK, dtype=int)
    c = 0
    for piece in pieces:
        w = piece.shape[1]
        strip[:, c : c + w] = piece
        c += w + 1
    return strip


def _decode_base_output(glyphs: list[int | None], out_w: int | None = None) -> np.ndarray:
    n = len(glyphs)
    strip = _encode_strip(glyphs)
    if out_w is None:
        out_w = strip.shape[1] // 2 + 1
    out = np.full((n + 2, out_w), BLACK, dtype=int)

    c = out_w // 2
    out[0, c] = GREEN
    for idx, step in enumerate(glyphs, start=1):
        out[idx, c] = RED
        if step is VERTICAL:
            out[idx + 1, c] = RED
        else:
            next_c = c + int(step)
            assert 0 <= next_c < out_w, "decoded path left the output canvas"
            out[idx, next_c] = RED
            c = next_c
    return out


def _label_black_regions(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if g[r, c] != BLACK or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = len(regions)
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == BLACK and labels[nr, nc] == -1:
                        labels[nr, nc] = len(regions)
                        stack.append((nr, nc))
            regions.append(np.array(coords, dtype=int))

    return labels, regions


def _seed_positions_from_input(grid: np.ndarray, out_shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    green_cells = np.argwhere(g == GREEN)
    assert green_cells.shape[0] == 1, "expected exactly one lower start marker"
    green_r, green_c = map(int, green_cells[0])

    out_h, out_w = map(int, out_shape)
    out_center = out_w // 2
    seed_positions: dict[tuple[int, int], int] = {}

    for r_raw, c_raw in np.argwhere(np.isin(g, list(SEED_COLORS))):
        r, c = int(r_raw), int(c_raw)
        rr = r - green_r
        cc = out_center + (c - green_c)
        assert 0 <= rr < out_h and 0 <= cc < out_w, "seed maps outside the decoded canvas"
        key = (rr, cc)
        assert key not in seed_positions, "multiple seeds mapped onto one output cell"
        seed_positions[key] = int(g[r, c])

    assert 2 <= len(seed_positions) <= 3, "expected 2-3 seed pixels"
    return seed_positions


def _paint_seeded_regions(base: np.ndarray, seed_positions: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    labels, regions = _label_black_regions(base)
    region_to_seed: dict[int, tuple[tuple[int, int], int]] = {}

    for (r, c), color in seed_positions.items():
        assert out[r, c] == BLACK, "seed must land on a background cell"
        region_id = int(labels[r, c])
        assert region_id >= 0, "seed must lie inside a black region"
        assert region_id not in region_to_seed, "a region may contain only one seed"
        region_to_seed[region_id] = ((r, c), int(color))

    for region_id, (_pos, seed_color) in region_to_seed.items():
        fill_color = SEED_FILL[seed_color]
        coords = regions[region_id]
        out[coords[:, 0], coords[:, 1]] = fill_color

    for (r, c), seed_color in seed_positions.items():
        out[r, c] = int(seed_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the two-row red glyph strip into a partitioning path, then recolor the seeded regions with paired colors.

    Input structure:
    - The top two rows are a red-on-black glyph strip read left to right, where each glyph is either a vertical column or a 2x2 elbow with one missing top cell.
    - Two fully black separator rows sit below the strip, and the lower area contains one green start marker plus 2-3 blue/yellow/grey seed pixels on black.
    - The lower marker/seed area is aligned to the decoded canvas by relative offset from the green marker rather than by a visible border.

    Rule:
    - Decode the strip into a black output of width `strip_width // 2 + 1` and height `glyph_count + 2`, place a green marker at the top-center, and trace the red path downward one glyph per row.
    - A vertical glyph paints the current cell and the cell directly below; an elbow paints the current cell and the horizontal neighbor indicated by its missing top corner (`missing upper-left -> right`, `missing upper-right -> left`).
    - Find all 4-connected black regions cut out by that path, map each input seed into the decoded canvas by its offset from the input green marker, and fill that seed's whole region with the seed's paired color (`1->2`, `4->6`, `5->9`) while leaving the seed pixel itself unchanged.

    Key insight:
    - The red glyphs encode a wall, not a picture to copy, and the lower colored pixels only label which decoded region gets which paired fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= LOWER_TOP + 6, "grid is too short for the described layout"
    assert np.all(g[2:4] == BLACK), "rows 3-4 must be the black separator"

    strip = g[:2]
    glyphs = _parse_glyphs(strip)
    base = _decode_base_output(glyphs, out_w=strip.shape[1] // 2 + 1)
    seed_positions = _seed_positions_from_input(g, base.shape)

    return _paint_seeded_regions(base, seed_positions)


def _candidate_glyphs() -> list[tuple[int | None, ...]]:
    candidates: list[tuple[int | None, ...]] = []
    for n in range(4, 8):
        for v in range(1, n):
            if v % 2 != (n - 1) % 2 or n - v < 2:
                continue
            for verticals in itertools.combinations(range(n - 1), v - 1):
                vertical_set = set(verticals) | {n - 1}
                for dirs in itertools.product([-1, 1], repeat=n - v):
                    glyphs: list[int | None] = []
                    dir_idx = 0
                    for idx in range(n):
                        if idx in vertical_set:
                            glyphs.append(VERTICAL)
                        else:
                            glyphs.append(int(dirs[dir_idx]))
                            dir_idx += 1
                    try:
                        base = _decode_base_output(glyphs)
                    except AssertionError:
                        continue
                    labels, regions = _label_black_regions(base)
                    if labels.shape[0] >= 0 and len(regions) == 3:
                        candidates.append(tuple(glyphs))

    mirrored = [tuple(VERTICAL if step is VERTICAL else -int(step) for step in glyphs) for glyphs in candidates]
    out: list[tuple[int | None, ...]] = []
    seen: set[tuple[int | None, ...]] = set()
    for glyphs in candidates + mirrored:
        if glyphs not in seen:
            seen.add(glyphs)
            out.append(glyphs)
    return out


GEN_CANDIDATES = _candidate_glyphs()


def _pick_seed_cell(rng: random.Random, region: np.ndarray) -> tuple[int, int]:
    preferred = [tuple(map(int, cell)) for cell in region if int(cell[0]) > 0]
    pool = preferred or [tuple(map(int, cell)) for cell in region]
    return rng.choice(pool)


def _build_input_from_scene(
    glyphs: tuple[int | None, ...],
    seed_positions: dict[tuple[int, int], int],
) -> np.ndarray:
    strip = _encode_strip(list(glyphs))
    out_w = strip.shape[1] // 2 + 1
    out_h = len(glyphs) + 2

    grid = np.full((LOWER_TOP + out_h, strip.shape[1]), BLACK, dtype=int)
    grid[:2] = strip

    lower_green_r = LOWER_TOP
    lower_green_c = strip.shape[1] // 2
    out_center = out_w // 2
    grid[lower_green_r, lower_green_c] = GREEN

    for (r, c), color in seed_positions.items():
        rr = lower_green_r + int(r)
        cc = lower_green_c + (int(c) - out_center)
        assert 0 <= rr < grid.shape[0]
        assert 0 <= cc < grid.shape[1]
        assert grid[rr, cc] == BLACK
        grid[rr, cc] = int(color)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    glyphs = rng.choice(GEN_CANDIDATES)
    base = _decode_base_output(list(glyphs))
    labels, regions = _label_black_regions(base)
    if len(regions) != 3:
        return None

    unseeded = rng.randrange(len(regions))
    seeded_regions = [idx for idx in range(len(regions)) if idx != unseeded]
    seed_colors = rng.sample(list(SEED_COLORS), k=len(seeded_regions))

    seed_positions: dict[tuple[int, int], int] = {}
    for region_id, color in zip(seeded_regions, seed_colors):
        pos = _pick_seed_cell(rng, regions[region_id])
        if pos in seed_positions:
            return None
        seed_positions[pos] = int(color)

    expected = _paint_seeded_regions(base, seed_positions)
    if len(_label_black_regions(base)[1]) != len(seed_positions) + 1:
        return None

    used_seed_colors = {color for color in seed_positions.values()}
    if not used_seed_colors.issubset(SEED_FILL):
        return None

    grid = _build_input_from_scene(glyphs, seed_positions)

    # Semantic validation: parser, offset mapping, region count, and oracle output must all agree.
    if _parse_glyphs(grid[:2]) != list(glyphs):
        return None
    mapped = _seed_positions_from_input(grid, base.shape)
    if mapped != seed_positions:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
