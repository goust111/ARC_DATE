from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, in_bounds


BG = Color.BLACK
PALETTE_LEN = 3
SEPARATOR = Color.GREY
SEED = Color.YELLOW
PALETTE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
QUADRANTS = ("tl", "tr", "bl", "br")
GRID_MIN = 15
GRID_MAX = 23
MIN_CROSS_MARGIN = 5
MIN_SEED_TO_SEPARATOR = 2
MAX_GENERATE_ATTEMPTS = 400


def _read_palette(grid: np.ndarray) -> list[int]:
    row = np.asarray(grid[0], dtype=int)
    non_bg = np.flatnonzero(row != BG)
    assert non_bg.size > 0, "palette row must contain foreground colors"

    start = int(non_bg[0])
    end = start
    while end < row.size and int(row[end]) != BG:
        end += 1

    palette = [int(v) for v in row[start:end]]
    assert len(palette) == PALETTE_LEN, "expected exactly three adjacent palette colors"
    assert np.count_nonzero(row != BG) == PALETTE_LEN, "palette row must contain a single contiguous run"
    assert all(color not in {BG, SEPARATOR, SEED} for color in palette), "palette colors must differ from background, separator, and seed"
    return palette


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    rows = [r for r in range(1, h) if np.all(g[r, :] == SEPARATOR)]
    cols = [c for c in range(w) if np.all(g[1:, c] == SEPARATOR) and int(g[0, c]) != SEPARATOR]

    assert len(rows) == 1, "expected exactly one horizontal separator row"
    assert len(cols) == 1, "expected exactly one vertical separator column"
    return int(rows[0]), int(cols[0])


def _quadrant_bounds(shape: tuple[int, int], sep_r: int, sep_c: int, quadrant: str) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    if quadrant == "tl":
        bounds = (1, sep_r - 1, 0, sep_c - 1)
    elif quadrant == "tr":
        bounds = (1, sep_r - 1, sep_c + 1, w - 1)
    elif quadrant == "bl":
        bounds = (sep_r + 1, h - 1, 0, sep_c - 1)
    elif quadrant == "br":
        bounds = (sep_r + 1, h - 1, sep_c + 1, w - 1)
    else:
        raise ValueError(f"unknown quadrant {quadrant}")

    r0, r1, c0, c1 = bounds
    assert r0 <= r1 and c0 <= c1, "separator cross must leave all quadrants non-empty"
    return bounds


def _quadrant_of(seed: tuple[int, int], sep_r: int, sep_c: int) -> str:
    r, c = map(int, seed)
    assert r != sep_r and c != sep_c, "seed cannot lie on a separator"
    if r < sep_r and c < sep_c:
        return "tl"
    if r < sep_r and c > sep_c:
        return "tr"
    if r > sep_r and c < sep_c:
        return "bl"
    return "br"


def _distance_to_separator(seed: tuple[int, int], sep_r: int, sep_c: int) -> int:
    r, c = map(int, seed)
    return int(min(abs(r - sep_r), abs(c - sep_c)))


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, list[int], int, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 5 <= h <= 30 and 5 <= w <= 30, "grid must fit ARC size limits"

    palette = _read_palette(g)
    sep_r, sep_c = _find_separator_cross(g)

    work_allowed = {BG, SEPARATOR, SEED}
    assert set(map(int, np.unique(g[1:, :]))) <= work_allowed, "working area may contain only background, separator, and yellow seeds"

    grey_coords = np.argwhere(g == SEPARATOR)
    for r_raw, c_raw in grey_coords:
        r = int(r_raw)
        c = int(c_raw)
        assert r >= 1, "separator may not occupy the palette row"
        assert r == sep_r or c == sep_c, "grey pixels must belong to the separator cross"

    seed_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == SEED)]
    assert 2 <= len(seed_coords) <= 4, "expected two to four seeds"

    seen_quadrants: set[str] = set()
    for seed in seed_coords:
        r, c = seed
        assert in_bounds(g.shape, r, c)
        quadrant = _quadrant_of(seed, sep_r, sep_c)
        assert quadrant not in seen_quadrants, "each quadrant may contain at most one seed"
        seen_quadrants.add(quadrant)
        assert _distance_to_separator(seed, sep_r, sep_c) >= MIN_SEED_TO_SEPARATOR, "seed too close to separator"

    return g, palette, sep_r, sep_c, seed_coords


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g, palette, sep_r, sep_c, seeds = _parse_grid(grid)
    out = g.copy()
    period = len(palette)

    for seed in seeds:
        sr, sc = seed
        phase = _distance_to_separator(seed, sep_r, sep_c)
        r0, r1, c0, c1 = _quadrant_bounds(g.shape, sep_r, sep_c, _quadrant_of(seed, sep_r, sep_c))

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        rr, cc = np.indices(sub.shape)
        rr += r0
        cc += c0
        dist = np.abs(rr - sr) + np.abs(cc - sc)
        fill_mask = (sub == BG) & (dist > 0)

        color_index = (dist - 1 + phase) % period
        for idx, color in enumerate(palette):
            out[r0 : r1 + 1, c0 : c1 + 1][fill_mask & (color_index == idx)] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seed's quadrant with Manhattan diamond rings whose palette phase is offset by the seed's distance to the separator cross.

    Input structure:
    - The top row is a black header containing one contiguous run of three non-black palette colors.
    - Below it, a grey horizontal line and a grey vertical line form a cross that splits the working area into four quadrants.
    - Two to four yellow seed pixels appear in distinct quadrants on an otherwise black background.

    Rule:
    - Read the three-color palette from the first contiguous non-black run in the top row.
    - For each yellow seed, compute D as its Manhattan distance to the nearest grey separator line.
    - Inside that seed's own quadrant only, color every black cell at Manhattan distance r >= 1 from the seed with palette[(r - 1 + D) mod 3].
    - Do not recolor the palette row, the grey separator cross, or the yellow seed pixels.

    Key insight:
    - The separator cross is both a hard expansion boundary and the source of the palette phase, so equally distant ring cells around different seeds can take different colors.
    """
    g, palette, sep_r, sep_c, seeds = _parse_grid(grid)
    out = g.copy()
    period = len(palette)

    for seed in seeds:
        sr, sc = seed
        phase = _distance_to_separator(seed, sep_r, sep_c)
        r0, r1, c0, c1 = _quadrant_bounds(g.shape, sep_r, sep_c, _quadrant_of(seed, sep_r, sep_c))

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(g[r, c]) != BG:
                    continue
                dist = abs(r - sr) + abs(c - sc)
                if dist == 0:
                    continue
                out[r, c] = int(palette[(dist - 1 + phase) % period])

    return out


def _candidate_seed_positions(shape: tuple[int, int], sep_r: int, sep_c: int, quadrant: str) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = _quadrant_bounds(shape, sep_r, sep_c, quadrant)
    candidates: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _distance_to_separator((r, c), sep_r, sep_c) >= MIN_SEED_TO_SEPARATOR:
                candidates.append((r, c))
    return candidates


def _max_visible_distance(shape: tuple[int, int], sep_r: int, sep_c: int, seed: tuple[int, int]) -> int:
    r0, r1, c0, c1 = _quadrant_bounds(shape, sep_r, sep_c, _quadrant_of(seed, sep_r, sep_c))
    sr, sc = seed
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    return max(abs(sr - r) + abs(sc - c) for r, c in corners)


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    g, palette, sep_r, sep_c, seeds = _parse_grid(inp)
    expected = _oracle_output(inp)
    assert np.array_equal(out, expected), "solver output does not match the puzzle semantics"
    assert not np.array_equal(inp, out), "output must differ from input"
    assert np.array_equal(out[0], g[0]), "palette row must stay unchanged"
    assert np.all(out[sep_r, :] == SEPARATOR), "horizontal separator must stay grey"
    assert np.all(out[1:, sep_c] == SEPARATOR), "vertical separator must stay grey"

    phases = [_distance_to_separator(seed, sep_r, sep_c) % len(palette) for seed in seeds]
    assert len(set(phases)) >= 2, "generator must expose different palette phases"
    assert any(_max_visible_distance(g.shape, sep_r, sep_c, seed) >= 4 for seed in seeds), "at least one seed should have multiple visible rings"

    for seed in seeds:
        sr, sc = seed
        assert int(out[sr, sc]) == SEED, "seed color must be preserved"
        quadrant = _quadrant_of(seed, sep_r, sep_c)
        r0, r1, c0, c1 = _quadrant_bounds(g.shape, sep_r, sep_c, quadrant)
        phase = _distance_to_separator(seed, sep_r, sep_c)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) == seed or int(g[r, c]) != BG:
                    continue
                dist = abs(r - sr) + abs(c - sc)
                assert int(out[r, c]) == int(palette[(dist - 1 + phase) % len(palette)])


def _build_input(
    height: int,
    width: int,
    palette: list[int],
    palette_start: int,
    sep_r: int,
    sep_c: int,
    seeds_by_quadrant: dict[str, tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[0, palette_start : palette_start + len(palette)] = np.array(palette, dtype=int)
    grid[sep_r, :] = SEPARATOR
    grid[1:, sep_c] = SEPARATOR
    for seed in seeds_by_quadrant.values():
        grid[seed] = SEED
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    if height < 2 * MIN_CROSS_MARGIN + 1 or width < 2 * MIN_CROSS_MARGIN + 1:
        return None

    sep_r = random.randint(MIN_CROSS_MARGIN, height - MIN_CROSS_MARGIN - 1)
    sep_c = random.randint(MIN_CROSS_MARGIN, width - MIN_CROSS_MARGIN - 1)
    if sep_r <= 1 or sep_r >= height - 1:
        return None

    palette = random.sample(PALETTE_COLORS, PALETTE_LEN)
    palette_start = random.randint(0, width - PALETTE_LEN)

    seed_count = random.choice((2, 3, 4))
    chosen_quadrants = random.sample(QUADRANTS, seed_count)

    seeds_by_quadrant: dict[str, tuple[int, int]] = {}
    phases: list[int] = []
    for quadrant in chosen_quadrants:
        candidates = _candidate_seed_positions((height, width), sep_r, sep_c, quadrant)
        if not candidates:
            return None
        seed = random.choice(candidates)
        seeds_by_quadrant[quadrant] = seed
        phases.append(_distance_to_separator(seed, sep_r, sep_c) % PALETTE_LEN)

    if len(set(phases)) < 2:
        return None

    if max(_max_visible_distance((height, width), sep_r, sep_c, seed) for seed in seeds_by_quadrant.values()) < 4:
        return None

    inp = _build_input(height, width, palette, palette_start, sep_r, sep_c, seeds_by_quadrant)
    out = main(inp)
    _semantic_checks(inp, out)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-phase diamond puzzle input")
