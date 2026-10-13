from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
CODE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
CORNERS = (0, 1, 2, 3)  # TL, TR, BL, BR
MAX_GENERATION_ATTEMPTS = 500


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _grid_coord(lattice_index: int) -> int:
    return 1 + 2 * int(lattice_index)


def _corner_to_global_lattice(i: int, j: int, lattice_n: int, corner: int) -> tuple[int, int]:
    if corner in (1, 3):
        j = lattice_n - 1 - j
    if corner in (2, 3):
        i = lattice_n - 1 - i
    return int(i), int(j)


def _corner_to_global_grid(i: int, j: int, lattice_n: int, corner: int) -> tuple[int, int]:
    gi, gj = _corner_to_global_lattice(i, j, lattice_n, corner)
    return _grid_coord(gi), _grid_coord(gj)


def _quadrant_from_coord(r: int, c: int, center: int) -> int:
    top = int(r) < center
    left = int(c) < center
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3


def _quadrant_slice(side: int, quadrant: int) -> tuple[slice, slice]:
    center = side // 2
    row_slice = slice(0, center) if quadrant in (0, 1) else slice(center + 1, side)
    col_slice = slice(0, center) if quadrant in (0, 2) else slice(center + 1, side)
    return row_slice, col_slice


def _crop_quadrant(grid: np.ndarray, quadrant: int) -> np.ndarray:
    rs, cs = _quadrant_slice(grid.shape[0], quadrant)
    return grid[rs, cs].copy()


def _extract_local_code_pixels(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    side = g.shape[0]
    assert g.ndim == 2 and side == g.shape[1] and side % 2 == 1

    bg = _bg_color(g)
    code_coords = np.argwhere((g != bg) & (g != RED))
    assert len(code_coords) > 0
    assert all(int(r) % 2 == 1 and int(c) % 2 == 1 for r, c in code_coords)

    lattice_n = (side - 1) // 2
    lattice_pixels = [
        ((int(r) - 1) // 2, (int(c) - 1) // 2, int(g[int(r), int(c)]))
        for r, c in code_coords
    ]

    orientations: list[tuple[tuple[int, int], int, list[tuple[int, int, int]]]] = []
    for corner in CORNERS:
        local_pixels = []
        for lr, lc, color in lattice_pixels:
            i, j = _corner_to_global_lattice(lr, lc, lattice_n, corner)
            local_pixels.append((i, j, color))
        score = (
            max(max(i, j) for i, j, _color in local_pixels),
            sum(i + j for i, j, _color in local_pixels),
        )
        orientations.append((score, corner, local_pixels))

    orientations.sort(key=lambda item: item[0])
    assert orientations[0][0] < orientations[1][0], "corner code must be uniquely closest to one corner"
    _score, corner, local_pixels = orientations[0]
    return lattice_n, corner, local_pixels


def _decode_rings(
    local_pixels: list[tuple[int, int, int]],
    lattice_n: int,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    corner_colors: dict[int, int] = {}
    horizontal_colors: dict[int, int] = {}
    vertical_colors: dict[int, int] = {}

    for i, j, color in local_pixels:
        assert 0 <= i < lattice_n and 0 <= j < lattice_n
        if i == j:
            assert i not in corner_colors or corner_colors[i] == color
            corner_colors[i] = color
            continue
        if i < j:
            depth = i
            assert depth < j < lattice_n - depth
            assert depth not in horizontal_colors or horizontal_colors[depth] == color
            horizontal_colors[depth] = color
        else:
            depth = j
            assert depth < i < lattice_n - depth
            assert depth not in vertical_colors or vertical_colors[depth] == color
            vertical_colors[depth] = color

    assert corner_colors, "expected at least one coded ring"
    for depth in set(horizontal_colors) | set(vertical_colors):
        assert depth in corner_colors
    return corner_colors, horizontal_colors, vertical_colors


def _render_pattern(
    side: int,
    corner_colors: dict[int, int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
) -> np.ndarray:
    lattice_n = (side - 1) // 2
    out = np.full((side, side), BG, dtype=int)

    for depth, color in corner_colors.items():
        lo = int(depth)
        hi = lattice_n - 1 - lo
        for ri in (lo, hi):
            for ci in (lo, hi):
                out[_grid_coord(ri), _grid_coord(ci)] = int(color)

    for depth, color in horizontal_colors.items():
        lo = int(depth)
        hi = lattice_n - 1 - lo
        if lo >= hi:
            continue
        row_top = _grid_coord(lo)
        row_bottom = _grid_coord(hi)
        for ci in range(lo + 1, hi):
            gc = _grid_coord(ci)
            out[row_top, gc] = int(color)
            out[row_bottom, gc] = int(color)

    for depth, color in vertical_colors.items():
        lo = int(depth)
        hi = lattice_n - 1 - lo
        if lo >= hi:
            continue
        col_left = _grid_coord(lo)
        col_right = _grid_coord(hi)
        for ri in range(lo + 1, hi):
            gr = _grid_coord(ri)
            out[gr, col_left] = int(color)
            out[gr, col_right] = int(color)

    return out


def _selected_quadrant_output(grid: np.ndarray, full_pattern: np.ndarray) -> np.ndarray:
    red_coords = np.argwhere(grid == RED)
    assert red_coords.shape[0] == 1, "expected exactly one red control pixel"
    rr, cc = map(int, red_coords[0])
    center = grid.shape[0] // 2
    assert rr != center and cc != center, "red control pixel cannot lie on the center row or column"
    quadrant = _quadrant_from_coord(rr, cc, center)
    return _crop_quadrant(full_pattern, quadrant)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold the corner code into full square rings, then output only the quadrant containing the red control pixel.

    Input structure:
    - The grid is an odd-sized square on a black background.
    - All non-red foreground pixels lie on odd row/column coordinates and form one compact corner code near a uniquely nearest corner.
    - Exactly one red control pixel appears away from the center row and center column, in a different quadrant from the code corner.

    Rule:
    - Ignore the red pixel while reading the corner code.
    - A diagonal code pixel sets the four corners of its ring, while a code pixel on a local horizontal or vertical side sets the two opposite sides of that ring.
    - Expand the full virtual concentric-square pattern on the original grid, then crop the quadrant selected by the red pixel; the red pixel itself is not copied into the output.

    Key insight:
    - The visible input only shows one corner-sized generator of a symmetric ring pattern; the red pixel is a camera selector for one quadrant of the unseen completed picture.
    """
    g = np.asarray(grid, dtype=int)
    side = g.shape[0]
    assert g.ndim == 2 and side == g.shape[1] and side % 2 == 1

    lattice_n, _corner, local_pixels = _extract_local_code_pixels(g)
    corner_colors, horizontal_colors, vertical_colors = _decode_rings(local_pixels, lattice_n)
    full_pattern = _render_pattern(side, corner_colors, horizontal_colors, vertical_colors)
    return _selected_quadrant_output(g, full_pattern)


def _sample_side_slots(num_rings: int, lattice_n: int) -> tuple[dict[int, int], dict[int, int]] | None:
    available_h: dict[int, list[int]] = {}
    available_v: dict[int, list[int]] = {}
    for depth in range(num_rings):
        # Keep side markers near the corner so the closest-corner inference stays unambiguous.
        max_local = min(lattice_n - 3 - depth, depth + 2)
        if max_local < depth + 1:
            continue
        positions = list(range(depth + 1, max_local + 1))
        available_h[depth] = positions[:]
        available_v[depth] = positions[:]

    all_slots = [("h", depth) for depth in available_h] + [("v", depth) for depth in available_v]
    if len(all_slots) < 2:
        return None

    slot_cap = min(4, len(all_slots), len(CODE_COLORS) - num_rings)
    if slot_cap < 2:
        return None

    for _ in range(40):
        slot_count = random.randint(2, slot_cap)
        chosen: list[tuple[str, int]] = []

        if random.random() < 0.5:
            pair_depths = sorted(set(available_h) & set(available_v))
            if pair_depths and slot_count >= 2:
                depth = random.choice(pair_depths)
                chosen.extend([("h", depth), ("v", depth)])

        remaining_pool = [slot for slot in all_slots if slot not in chosen]
        random.shuffle(remaining_pool)
        chosen.extend(remaining_pool[: slot_count - len(chosen)])

        if slot_count >= 2 and {kind for kind, _depth in chosen} != {"h", "v"}:
            continue

        horizontal_positions: dict[int, int] = {}
        vertical_positions: dict[int, int] = {}
        for kind, depth in chosen:
            if kind == "h":
                horizontal_positions[depth] = random.choice(available_h[depth])
            else:
                vertical_positions[depth] = random.choice(available_v[depth])
        return horizontal_positions, vertical_positions

    return None


def _sample_code_spec(
    side: int,
) -> tuple[int, dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[int, int]] | None:
    lattice_n = (side - 1) // 2
    max_rings = min(4, lattice_n // 2)
    if max_rings < 2:
        return None

    num_rings = random.randint(2, max_rings)
    side_positions = _sample_side_slots(num_rings, lattice_n)
    if side_positions is None:
        return None
    horizontal_positions, vertical_positions = side_positions

    slot_count = num_rings + len(horizontal_positions) + len(vertical_positions)
    if slot_count > len(CODE_COLORS):
        return None

    sampled_colors = random.sample(CODE_COLORS, slot_count)
    corner_colors = {depth: sampled_colors[depth] for depth in range(num_rings)}
    cursor = num_rings
    horizontal_colors: dict[int, int] = {}
    vertical_colors: dict[int, int] = {}
    for depth in sorted(horizontal_positions):
        horizontal_colors[depth] = sampled_colors[cursor]
        cursor += 1
    for depth in sorted(vertical_positions):
        vertical_colors[depth] = sampled_colors[cursor]
        cursor += 1

    return num_rings, corner_colors, horizontal_colors, vertical_colors, horizontal_positions, vertical_positions


def _build_input_grid(
    side: int,
    code_corner: int,
    corner_colors: dict[int, int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
    horizontal_positions: dict[int, int],
    vertical_positions: dict[int, int],
    red_position: tuple[int, int],
) -> np.ndarray:
    grid = np.full((side, side), BG, dtype=int)
    lattice_n = (side - 1) // 2

    for depth, color in corner_colors.items():
        r, c = _corner_to_global_grid(depth, depth, lattice_n, code_corner)
        grid[r, c] = int(color)
    for depth, color in horizontal_colors.items():
        r, c = _corner_to_global_grid(depth, horizontal_positions[depth], lattice_n, code_corner)
        grid[r, c] = int(color)
    for depth, color in vertical_colors.items():
        r, c = _corner_to_global_grid(vertical_positions[depth], depth, lattice_n, code_corner)
        grid[r, c] = int(color)

    rr, cc = red_position
    assert grid[rr, cc] == BG
    grid[rr, cc] = RED
    return grid


def _quadrant_candidates(side: int, quadrant: int) -> list[tuple[int, int]]:
    rs, cs = _quadrant_slice(side, quadrant)
    return [(r, c) for r in range(rs.start, rs.stop) for c in range(cs.start, cs.stop)]


def _all_quadrant_crops(grid: np.ndarray) -> list[np.ndarray]:
    return [_crop_quadrant(grid, quadrant) for quadrant in CORNERS]


def _pairwise_distinct(arrays: list[np.ndarray]) -> bool:
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            if np.array_equal(arrays[i], arrays[j]):
                return False
    return True


def _shortcut_variant(
    side: int,
    corner_colors: dict[int, int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
) -> np.ndarray:
    merged: dict[int, int] = {}
    for depth in corner_colors:
        if depth in horizontal_colors:
            merged[depth] = horizontal_colors[depth]
        elif depth in vertical_colors:
            merged[depth] = vertical_colors[depth]
    return _render_pattern(side, corner_colors, merged, merged)


def _choose_red_position(full_pattern: np.ndarray, red_quadrant: int, force_on_pattern: bool) -> tuple[int, int] | None:
    candidates = _quadrant_candidates(full_pattern.shape[0], red_quadrant)
    if force_on_pattern:
        candidates = [(r, c) for r, c in candidates if int(full_pattern[r, c]) != BG]
    else:
        candidates = [(r, c) for r, c in candidates if int(full_pattern[r, c]) == BG]
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(list(range(15, 30, 2)))
    code_spec = _sample_code_spec(side)
    if code_spec is None:
        return None

    _num_rings, corner_colors, horizontal_colors, vertical_colors, horizontal_positions, vertical_positions = code_spec
    lattice_n = (side - 1) // 2
    code_corner = random.choice(CORNERS)

    full_pattern = _render_pattern(side, corner_colors, horizontal_colors, vertical_colors)
    crops = _all_quadrant_crops(full_pattern)
    if not _pairwise_distinct(crops):
        return None

    red_quadrant = random.choice([q for q in CORNERS if q != code_corner])
    red_position = _choose_red_position(full_pattern, red_quadrant, force_on_pattern=random.random() < 0.5)
    if red_position is None:
        return None

    grid = _build_input_grid(
        side,
        code_corner,
        corner_colors,
        horizontal_colors,
        vertical_colors,
        horizontal_positions,
        vertical_positions,
        red_position,
    )

    try:
        decoded_lattice_n, decoded_corner, local_pixels = _extract_local_code_pixels(grid)
        decoded_corners, decoded_h, decoded_v = _decode_rings(local_pixels, decoded_lattice_n)
        oracle_full = _render_pattern(side, decoded_corners, decoded_h, decoded_v)
        expected = _selected_quadrant_output(grid, oracle_full)
        actual = main(grid)
    except Exception:
        return None

    if decoded_corner != code_corner:
        return None
    if decoded_corners != corner_colors or decoded_h != horizontal_colors or decoded_v != vertical_colors:
        return None
    if not np.array_equal(oracle_full, full_pattern):
        return None
    if not np.array_equal(actual, expected):
        return None

    selected_crop = _crop_quadrant(full_pattern, red_quadrant)
    if not np.array_equal(expected, selected_crop):
        return None

    corner_only = _crop_quadrant(_render_pattern(side, corner_colors, {}, {}), red_quadrant)
    if np.array_equal(selected_crop, corner_only):
        return None

    merged_sides = _crop_quadrant(_shortcut_variant(side, corner_colors, horizontal_colors, vertical_colors), red_quadrant)
    if np.array_equal(selected_crop, merged_sides):
        return None

    input_quadrant = _crop_quadrant(grid, red_quadrant)
    input_without_red = input_quadrant.copy()
    input_without_red[input_without_red == RED] = BG
    if np.array_equal(selected_crop, input_without_red):
        return None

    center = side // 2
    if _quadrant_from_coord(red_position[0], red_position[1], center) != red_quadrant:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
