from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, components_mask, paste


BG = Color.GREY
LINE = Color.BLACK
MARK = Color.TEAL
DIRECTIONS = ((1, 1), (1, -1))
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _line_key(r: int, c: int, direction: tuple[int, int]) -> int:
    if direction == (1, 1):
        return int(r - c)
    return int(r + c)


def _axis_side(r: int, c: int, segment: dict) -> int:
    key = int(segment["line"])
    if segment["direction"] == (1, 1):
        return int(r - c - key)
    return int(r + c - key)


def _reflect_coord(r: int, c: int, segment: dict) -> tuple[int, int]:
    key = int(segment["line"])
    if segment["direction"] == (1, 1):
        return int(c + key), int(r - key)
    return int(key - c), int(key - r)


def _diagonal_segments(g: np.ndarray) -> list[dict]:
    mask = g == LINE
    h, w = g.shape
    segments: list[dict] = []
    for direction in DIRECTIONS:
        dr, dc = direction
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if _in_bounds(g.shape, pr, pc) and mask[pr, pc]:
                    continue

                coords: list[tuple[int, int]] = []
                rr, cc = r, c
                while _in_bounds(g.shape, rr, cc) and mask[rr, cc]:
                    coords.append((int(rr), int(cc)))
                    rr += dr
                    cc += dc

                if len(coords) >= 2:
                    segments.append(
                        {
                            "direction": direction,
                            "line": _line_key(r, c, direction),
                            "coords": tuple(coords),
                            "length": len(coords),
                        }
                    )
    return segments


def _select_segment(g: np.ndarray) -> dict:
    segments = _diagonal_segments(g)
    assert segments, "expected at least one black diagonal segment"
    max_len = max(int(seg["length"]) for seg in segments)
    candidates = [seg for seg in segments if int(seg["length"]) == max_len]
    if len(candidates) == 1:
        return candidates[0]

    directions = {tuple(seg["direction"]) for seg in candidates}
    assert len(directions) == 1, "longest candidates must be one parallel band"
    lines = sorted(int(seg["line"]) for seg in candidates)
    assert len(set(lines)) == len(lines), "band lines must be distinct"
    assert len(lines) % 2 == 1 and len(lines) >= 3, "parallel band must have odd width"
    assert lines == list(range(lines[0], lines[-1] + 1)), "band lines must be adjacent"

    middle = lines[len(lines) // 2]
    chosen = [seg for seg in candidates if int(seg["line"]) == middle]
    assert len(chosen) == 1
    return chosen[0]


def _sprite_component(g: np.ndarray) -> np.ndarray:
    sprite_mask = (g != BG) & (g != LINE)
    comps = components_mask(sprite_mask, connectivity=8)
    assert len(comps) == 1, "expected exactly one colored sprite"
    assert int(comps[0].shape[0]) >= 2, "sprite must have multiple pixels"
    return comps[0]


def _solve_with_segment(g: np.ndarray, selected: dict) -> np.ndarray:
    out = g.copy()
    sprite = _sprite_component(g)
    original = {(int(r), int(c)) for r, c in sprite}
    sides = [_axis_side(int(r), int(c), selected) for r, c in sprite]
    side_signs = {1 if side > 0 else -1 for side in sides if side != 0}
    assert 0 not in sides and len(side_signs) == 1, "sprite must lie on one side of the hidden axis"

    reflected_any = False
    for r, c in sprite:
        rr, cc = _reflect_coord(int(r), int(c), selected)
        assert _in_bounds(g.shape, rr, cc), "reflected sprite must stay in bounds"
        assert int(g[rr, cc]) != LINE, "reflected sprite must not collide with black segments"
        out[rr, cc] = int(g[r, c])
        reflected_any |= (rr, cc) not in original

    assert reflected_any, "reflection must add a visible copy"
    for r, c in selected["coords"]:
        out[int(r), int(c)] = MARK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the colored sprite across the hidden diagonal selected by black segments.

    Input structure:
    - A grey field contains several black diagonal segments in the two 45-degree
      directions; some longest parallel segments may form an odd-width adjacent band.
    - A single multi-pixel colored sprite is separated from the black segments and
      lies completely on one side of the selected diagonal line.

    Rule:
    - Find every maximal continuous black segment along both diagonal directions.
    - Select the unique longest segment; if the longest candidates are an adjacent
      odd-width parallel band, select the middle segment of that band.
    - Treat the selected segment's whole diagonal line as a mirror, copy the sprite
      to the reflected positions with its original colors, and recolor only the
      selected black segment teal.

    Key insight:
    - The black stroke is not merely highlighted; its containing diagonal line is
      the reflection axis that determines where the sprite copy must appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert np.count_nonzero(g == BG) > g.size // 2, "expected a grey field"
    selected = _select_segment(g)
    return _solve_with_segment(g, selected)


# ----------------------------- generator -----------------------------

GEN_RETRIES = 500
GEN_SIDE_RANGE = (18, 30)
GEN_LINE_LEN_RANGE = (7, 11)
GEN_DISTRACTOR_COUNT = (2, 5)
GEN_SPRITE_HW = (3, 5)
GEN_SPRITE_AREA = (5, 9)


def _segment_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    offset: int = 0,
) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = direction
    return [(int(r0 + offset + i * dr), int(c0 + i * dc)) for i in range(int(length))]


def _random_segment_start(
    shape: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    half_width: int,
) -> tuple[int, int] | None:
    h, w = shape
    border = 1
    r_min = border + half_width
    r_max = h - border - half_width - length
    if direction == (1, 1):
        c_min = border
        c_max = w - border - length
    else:
        c_min = border + length - 1
        c_max = w - border - 1
    if r_max < r_min or c_max < c_min:
        return None
    return random.randint(r_min, r_max), random.randint(c_min, c_max)


def _band_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    half_width: int,
) -> list[list[tuple[int, int]]]:
    return [_segment_cells(start, direction, length, off) for off in range(-half_width, half_width + 1)]


def _can_place_black_group(g: np.ndarray, groups: list[list[tuple[int, int]]]) -> bool:
    new_cells = {cell for group in groups for cell in group}
    if len(new_cells) != sum(len(group) for group in groups):
        return False
    for r, c in new_cells:
        if not _in_bounds(g.shape, r, c) or int(g[r, c]) != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in new_cells:
                    continue
                if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == LINE:
                    return False
    return True


def _paint_black_group(g: np.ndarray, groups: list[list[tuple[int, int]]]) -> None:
    for group in groups:
        for r, c in group:
            g[r, c] = LINE


def _place_selected(g: np.ndarray) -> tuple[dict, bool] | None:
    direction = random.choice(DIRECTIONS)
    is_band = random.random() < 0.58
    half_width = random.choice([1, 1, 2]) if is_band else 0
    length = random.randint(*GEN_LINE_LEN_RANGE)
    h, w = g.shape
    border = 1
    r_min = border + half_width
    r_max = h - border - half_width - length
    if direction == (1, 1):
        c_min = border
        c_max = w - border - length
        center_key = 0
    else:
        c_min = border + length - 1
        c_max = w - border - 1
        center_key = (h - 1 + w - 1) // 2
    if r_max < r_min or c_max < c_min:
        return None

    max_axis_offset = max(3, min(h, w) // 5)
    starts = []
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            if abs(_line_key(r, c, direction) - center_key) <= max_axis_offset:
                starts.append((r, c))
    if not starts:
        return None
    random.shuffle(starts)

    start = None
    groups = None
    for candidate in starts:
        cand_groups = _band_cells(candidate, direction, length, half_width)
        if _can_place_black_group(g, cand_groups):
            start = candidate
            groups = cand_groups
            break
    if start is None or groups is None:
        return None

    if not _can_place_black_group(g, groups):
        return None
    _paint_black_group(g, groups)

    selected_cells = tuple(groups[half_width])
    selected = {
        "direction": direction,
        "line": _line_key(selected_cells[0][0], selected_cells[0][1], direction),
        "coords": selected_cells,
        "length": length,
    }
    return selected, is_band


def _place_distractors(g: np.ndarray, max_len: int) -> bool:
    count = random.randint(*GEN_DISTRACTOR_COUNT)
    placed = 0
    for _ in range(250):
        if placed >= count:
            return True
        direction = random.choice(DIRECTIONS)
        length = random.randint(2, max(2, max_len - 2))
        half_width = 1 if length >= 4 and random.random() < 0.25 else 0
        start = _random_segment_start(g.shape, direction, length, half_width)
        if start is None:
            continue
        groups = _band_cells(start, direction, length, half_width)
        if not _can_place_black_group(g, groups):
            continue
        _paint_black_group(g, groups)
        placed += 1
    return placed >= max(1, count - 1)


def _connected_sprite_mask(h: int, w: int, area: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite_patch() -> np.ndarray:
    h = random.randint(*GEN_SPRITE_HW)
    w = random.randint(*GEN_SPRITE_HW)
    area = random.randint(GEN_SPRITE_AREA[0], min(GEN_SPRITE_AREA[1], h * w))
    mask = _connected_sprite_mask(h, w, area)
    colors = random.sample(SPRITE_COLORS, random.randint(2, 3))
    patch = np.full((h, w), BG, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    for i, (r, c) in enumerate(coords):
        patch[r, c] = colors[i % len(colors)]
    return patch


def _near_black(g: np.ndarray, coords: set[tuple[int, int]], radius: int = 1) -> bool:
    for r, c in coords:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == LINE:
                    return True
    return False


def _sprite_position_options(
    g: np.ndarray,
    patch: np.ndarray,
    selected: dict,
) -> list[tuple[int, int]]:
    h, w = patch.shape
    opaque = [(int(r), int(c)) for r, c in np.argwhere(patch != BG)]
    options: list[tuple[int, int]] = []
    for r0 in range(g.shape[0] - h + 1):
        for c0 in range(g.shape[1] - w + 1):
            cells = {(r0 + r, c0 + c) for r, c in opaque}
            if any(int(g[r, c]) != BG for r, c in cells):
                continue
            sides = [_axis_side(r, c, selected) for r, c in cells]
            side_signs = {1 if side > 0 else -1 for side in sides if side != 0}
            if 0 in sides or len(side_signs) != 1 or min(abs(s) for s in sides) < 3:
                continue
            reflected = {_reflect_coord(r, c, selected) for r, c in cells}
            if any(not _in_bounds(g.shape, r, c) for r, c in reflected):
                continue
            if cells & reflected:
                continue
            if any(int(g[r, c]) != BG for r, c in reflected):
                continue
            if _near_black(g, cells, radius=1) or _near_black(g, reflected, radius=1):
                continue
            options.append((r0, c0))
    return options


def _valid_sprite_position(
    g: np.ndarray,
    patch: np.ndarray,
    selected: dict,
    r0: int,
    c0: int,
) -> bool:
    opaque = [(int(r), int(c)) for r, c in np.argwhere(patch != BG)]
    cells = {(int(r0 + r), int(c0 + c)) for r, c in opaque}
    if any(int(g[r, c]) != BG for r, c in cells):
        return False
    sides = [_axis_side(r, c, selected) for r, c in cells]
    side_signs = {1 if side > 0 else -1 for side in sides if side != 0}
    if 0 in sides or len(side_signs) != 1 or min(abs(s) for s in sides) < 3:
        return False
    reflected = {_reflect_coord(r, c, selected) for r, c in cells}
    if any(not _in_bounds(g.shape, r, c) for r, c in reflected):
        return False
    if cells & reflected:
        return False
    if any(int(g[r, c]) != BG for r, c in reflected):
        return False
    if _near_black(g, cells, radius=1) or _near_black(g, reflected, radius=1):
        return False
    return True


def _place_sprite(g: np.ndarray, selected: dict) -> bool:
    for _ in range(60):
        patch = _make_sprite_patch()
        h, w = patch.shape
        if h > g.shape[0] or w > g.shape[1]:
            continue
        for _ in range(500):
            r0 = random.randrange(g.shape[0] - h + 1)
            c0 = random.randrange(g.shape[1] - w + 1)
            if _valid_sprite_position(g, patch, selected, r0, c0):
                paste(g, patch, r0, c0, transparent=BG)
                return True
    return False


def _same_coords(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    return {tuple(map(int, x)) for x in a} == {tuple(map(int, x)) for x in b}


def _semantic_ok(inp: np.ndarray, intended: dict, expected_band: bool) -> bool:
    try:
        selected = _select_segment(inp)
        if tuple(selected["direction"]) != tuple(intended["direction"]):
            return False
        if not _same_coords(selected["coords"], intended["coords"]):
            return False

        segments = _diagonal_segments(inp)
        max_len = max(int(seg["length"]) for seg in segments)
        candidates = [seg for seg in segments if int(seg["length"]) == max_len]
        if expected_band:
            if len(candidates) not in (3, 5):
                return False
            lines = sorted(int(seg["line"]) for seg in candidates)
            if lines != list(range(lines[0], lines[-1] + 1)):
                return False
        elif len(candidates) != 1:
            return False

        out = main(inp)
        manual = _solve_with_segment(inp, intended)
        if not np.array_equal(out, manual):
            return False
        if np.array_equal(out, inp):
            return False
        if np.count_nonzero(out == MARK) != len(intended["coords"]):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_SIDE_RANGE)
    w = random.randint(*GEN_SIDE_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    placed = _place_selected(grid)
    if placed is None:
        return None
    intended, was_band = placed
    if not _place_distractors(grid, int(intended["length"])):
        return None
    if not _place_sprite(grid, intended):
        return None
    if not _semantic_ok(grid, intended, was_band):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid hidden-axis diagonal reflection input")
