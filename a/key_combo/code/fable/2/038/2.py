from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, get_bg, in_bounds


GEN_ATTEMPTS = 500
GRID_W_RANGE = (12, 24)
TOP_H_RANGE = (4, 9)
BOTTOM_EXTRA_RANGE = (0, 4)
POINT_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
MAX_TOTAL_POINTS = 11
KEY_MARGIN = 2
KEY_GAP = 2
_SEEN_INPUT_SIGNATURES: set[bytes] = set()


def _grid_signature(grid: np.ndarray) -> bytes:
    shape_bytes = np.asarray(grid.shape, dtype=np.int16).tobytes()
    return shape_bytes + np.asarray(grid, dtype=np.int8).tobytes()


def _find_wall_row(grid: np.ndarray, bg: int) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[int] = []
    for r in range(h):
        row = g[r]
        if not np.all(row != bg):
            continue
        values, counts = np.unique(row, return_counts=True)
        if len(values) < 2:
            continue
        if int(counts.max()) <= w // 2:
            continue
        candidates.append(r)
    assert len(candidates) == 1, "expected exactly one full-span wall row"
    return int(candidates[0])


def _extract_keyholes(grid: np.ndarray, wall_r: int) -> tuple[int, dict[int, int]]:
    row = np.asarray(grid[wall_r], dtype=int)
    values, counts = np.unique(row, return_counts=True)
    order = np.argsort(-counts)
    wall_color = int(values[order[0]])
    assert len(order) >= 2
    assert int(counts[order[0]]) > int(counts[order[1]]), "wall color must be uniquely most frequent on the wall"

    key_cols = np.flatnonzero(row != wall_color)
    key_colors = [int(row[c]) for c in key_cols]
    assert len(key_cols) >= 2, "expected multiple keyholes"
    assert len(key_colors) == len(set(key_colors)), "keyhole colors must be distinct"
    return wall_color, {int(row[c]): int(c) for c in key_cols}


def _project_pixel(wall_r: int, key_col: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * wall_r - r), int(2 * key_col - c)


def _semantic_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    wall_r = _find_wall_row(g, bg)
    wall_color, keyholes = _extract_keyholes(g, wall_r)

    upper = g[:wall_r]
    lower = g[wall_r + 1 :]
    assert lower.size > 0 and np.all(lower == bg), "lower chamber must be empty in the input"

    upper_colors = set(colors_present(upper, exclude={bg}))
    assert upper_colors, "upper chamber must contain a pattern"
    assert wall_color not in upper_colors, "wall color must not appear in the upper pattern"
    assert upper_colors == set(keyholes), "upper pattern colors must match the wall keyholes exactly"

    out = g.copy()
    occupied_lower: set[tuple[int, int]] = set()
    for color, key_col in keyholes.items():
        coords = np.argwhere(upper == color)
        assert len(coords) > 0, "every keyhole color must appear in the upper chamber"
        for r0, c0 in coords:
            r = int(r0)
            c = int(c0)
            pr, pc = _project_pixel(wall_r, key_col, r, c)
            assert pr > wall_r, "projected pixel must land in the lower chamber"
            assert in_bounds(g.shape, pr, pc), "projected pixel must stay in bounds"
            assert int(g[pr, pc]) == bg, "projection must land on empty space"
            assert (pr, pc) not in occupied_lower, "projected pixels must not collide"
            occupied_lower.add((pr, pc))
            out[pr, pc] = color

    return out


def _sample_spaced_columns(width: int, n_cols: int, *, margin: int, min_gap: int) -> list[int] | None:
    candidates = list(range(margin, width - margin))
    if len(candidates) < n_cols:
        return None
    for _ in range(200):
        cols = sorted(random.sample(candidates, n_cols))
        if all(b - a >= min_gap for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _choose_counts(n_colors: int) -> list[int]:
    counts = [int(random.choice(POINT_COUNT_CHOICES)) for _ in range(n_colors)]
    if max(counts) == 1:
        counts[random.randrange(n_colors)] = 2
    while sum(counts) > MAX_TOTAL_POINTS:
        shrinkable = [i for i, count in enumerate(counts) if count > 1]
        if not shrinkable:
            break
        counts[random.choice(shrinkable)] -= 1
    if max(counts) == 1:
        counts[random.randrange(n_colors)] = 2
    return counts


def _candidate_sources(height: int, width: int, wall_r: int, key_col: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r in range(wall_r):
        for c in range(width):
            pr, pc = _project_pixel(wall_r, key_col, r, c)
            if pr > wall_r and in_bounds((height, width), pr, pc):
                coords.append((r, c))
    return coords


def _pick_points_for_color(
    *,
    height: int,
    width: int,
    wall_r: int,
    key_col: int,
    count: int,
    used_upper: set[tuple[int, int]],
    used_lower: set[tuple[int, int]],
    prefer_off_center: bool,
) -> list[tuple[int, int]] | None:
    allowed = _candidate_sources(height, width, wall_r, key_col)
    allowed_set = set(allowed)
    if not allowed:
        return None

    def can_place(rc: tuple[int, int], local_upper: set[tuple[int, int]], local_lower: set[tuple[int, int]]) -> bool:
        r, c = rc
        if rc in local_upper or rc in used_upper:
            return False
        pr, pc = _project_pixel(wall_r, key_col, r, c)
        return (pr, pc) not in local_lower and (pr, pc) not in used_lower

    for _ in range(200):
        points: list[tuple[int, int]] = []
        local_upper: set[tuple[int, int]] = set()
        local_lower: set[tuple[int, int]] = set()

        base_pool = [rc for rc in allowed if can_place(rc, local_upper, local_lower)]
        if prefer_off_center:
            off_center = [rc for rc in base_pool if rc[1] != key_col]
            if off_center:
                base_pool = off_center
        if not base_pool:
            continue

        first = random.choice(base_pool)
        points.append(first)
        local_upper.add(first)
        local_lower.add(_project_pixel(wall_r, key_col, first[0], first[1]))

        while len(points) < count:
            grow_candidates: set[tuple[int, int]] = set()
            for r, c in points:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    cand = (r + dr, c + dc)
                    if cand not in allowed_set:
                        continue
                    if can_place(cand, local_upper, local_lower):
                        grow_candidates.add(cand)

            scatter_pool = [rc for rc in allowed if can_place(rc, local_upper, local_lower)]
            if not scatter_pool:
                break

            if grow_candidates and random.random() < 0.65:
                cand = random.choice(list(grow_candidates))
            else:
                cand = random.choice(scatter_pool)

            points.append(cand)
            local_upper.add(cand)
            local_lower.add(_project_pixel(wall_r, key_col, cand[0], cand[1]))

        if len(points) == count:
            return points

    return None


def _manual_projection(grid: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]], dict[int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall_r = _find_wall_row(g, bg)
    wall_color, keyholes = _extract_keyholes(g, wall_r)
    out = g.copy()
    lower_hits: set[tuple[int, int]] = set()
    grouped: dict[int, list[tuple[int, int]]] = {}
    for color, key_col in keyholes.items():
        grouped[color] = []
        for r0, c0 in np.argwhere(g[:wall_r] == color):
            r = int(r0)
            c = int(c0)
            pr, pc = _project_pixel(wall_r, key_col, r, c)
            assert in_bounds(g.shape, pr, pc)
            assert pr > wall_r
            assert (pr, pc) not in lower_hits
            lower_hits.add((pr, pc))
            grouped[color].append((pr, pc))
            out[pr, pc] = color
    return out, grouped, keyholes, wall_r, wall_color


def _validate_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out_manual, grouped, keyholes, wall_r, wall_color = _manual_projection(g)
    out_solver = main(g)

    assert np.array_equal(out_solver, out_manual), "main() must match the semantic projection"
    assert np.array_equal(out_solver[: wall_r + 1], g[: wall_r + 1]), "upper chamber and wall must stay unchanged"
    assert np.all(g[wall_r + 1 :] == bg), "input lower chamber must be empty"

    upper_mask = g[:wall_r] != bg
    upper_coords = np.argwhere(upper_mask)
    assert len(upper_coords) > len(keyholes), "need repeated color usage to expose same-color grouping"
    r0, c0, r1, c1 = bbox_coords(upper_coords)
    assert (r1 > r0) or (c1 > c0), "upper pattern must not collapse to a singleton"

    upper_colors = set(colors_present(g[:wall_r], exclude={bg}))
    assert upper_colors == set(keyholes)
    assert wall_color not in upper_colors
    assert len(keyholes) >= 2
    assert any(len(np.argwhere(g[:wall_r] == color)) >= 2 for color in keyholes), "at least one color must appear multiple times"

    col_shifts: list[int] = []
    mirror_only = np.full_like(g, bg)
    for color, key_col in keyholes.items():
        for r0, c0 in np.argwhere(g[:wall_r] == color):
            r = int(r0)
            c = int(c0)
            pr, pc = _project_pixel(wall_r, key_col, r, c)
            col_shifts.append(pc - c)
            mr = 2 * wall_r - r
            if in_bounds(g.shape, mr, c):
                mirror_only[mr, c] = color

    assert any(shift != 0 for shift in col_shifts), "projection must differ from a pure vertical mirror"
    assert len(set(col_shifts)) >= 2, "projection must not reduce to a single global horizontal shift"
    assert not np.array_equal(out_manual[wall_r + 1 :], mirror_only[wall_r + 1 :]), "need a counterexample to wall-only mirroring"

    for color, hits in grouped.items():
        assert hits, f"missing projected pixels for color {color}"
        assert len(hits) == len(set(hits))


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*GRID_W_RANGE)
    top_h = random.randint(*TOP_H_RANGE)
    bottom_h = top_h + random.randint(*BOTTOM_EXTRA_RANGE)
    height = top_h + 1 + bottom_h
    if height > 30 or width > 30:
        return None

    max_colors = min(5, max(2, len(range(KEY_MARGIN, width - KEY_MARGIN, KEY_GAP))))
    if max_colors < 2:
        return None
    n_colors = random.randint(2, max_colors)

    cols = _sample_spaced_columns(width, n_colors, margin=KEY_MARGIN, min_gap=KEY_GAP)
    if cols is None:
        return None

    palette = random.sample(list(Color.ALL_COLORS), n_colors + 2)
    bg = int(palette[0])
    wall_color = int(palette[1])
    colors = [int(c) for c in palette[2:]]
    random.shuffle(colors)

    counts = _choose_counts(n_colors)

    grid = np.full((height, width), bg, dtype=int)
    grid[top_h, :] = wall_color
    for color, col in zip(colors, cols):
        grid[top_h, col] = color

    used_upper: set[tuple[int, int]] = set()
    used_lower: set[tuple[int, int]] = set()
    force_off_center = random.sample(range(n_colors), k=min(2, n_colors))

    for idx, (color, key_col, count) in enumerate(zip(colors, cols, counts)):
        points = _pick_points_for_color(
            height=height,
            width=width,
            wall_r=top_h,
            key_col=key_col,
            count=count,
            used_upper=used_upper,
            used_lower=used_lower,
            prefer_off_center=idx in force_off_center or random.random() < 0.7,
        )
        if points is None:
            return None

        for r, c in points:
            grid[r, c] = color
            used_upper.add((r, c))
            used_lower.add(_project_pixel(top_h, key_col, r, c))

    try:
        _validate_case(grid)
    except AssertionError:
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each upper color subset through its same-color wall keyhole into the lower chamber.

    Input structure:
    - A single full-width solid horizontal wall splits the grid into an upper chamber and an empty lower chamber.
    - The upper chamber contains a sparse multicolor pixel pattern.
    - The wall contains one uniquely colored keyhole cell for each color used in the upper pattern, with no background gaps.

    Rule:
    - Identify the wall row, its majority wall color, and the one-cell keyhole of each other wall color.
    - For every upper pixel of color c, find the wall keyhole with the same color and map the pixel by 180-degree point reflection around that keyhole.
    - Keep the upper chamber and the wall unchanged, and paint every reflected pixel in the lower chamber with its original color.

    Key insight:
    - The wall colors are per-color symmetry centers, so the lower image is not a whole-pattern mirror across the wall but a color-wise recombination of separate point reflections.
    """
    return _semantic_output(grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        sig = _grid_signature(grid)
        if sig in _SEEN_INPUT_SIGNATURES:
            continue
        _SEEN_INPUT_SIGNATURES.add(sig)
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
