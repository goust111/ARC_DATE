from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

MARKER_COLORS = (RED, GREEN, YELLOW, GREY)

MAX_GENERATION_ATTEMPTS = 400
MAX_CUP_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _blue_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[set[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLUE or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == BLUE:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _geometry_from_component(shape: tuple[int, int], component: set[tuple[int, int]]) -> dict:
    h, w = shape
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in component:
        comp_mask[r, c] = True

    rows = sorted({r for r, _c in component})
    if not rows:
        raise ValueError("empty component")
    top = rows[0]
    bottom = rows[-1]
    if bottom - top + 1 < 3:
        raise ValueError("cup height too small")

    row_spans: dict[int, tuple[int, int]] = {}
    for r in range(top, bottom + 1):
        cols = np.flatnonzero(comp_mask[r])
        if cols.size == 0:
            raise ValueError("cup must occupy every row in its bbox")
        left = int(cols[0])
        right = int(cols[-1])
        row_spans[r] = (left, right)

    bottom_left, bottom_right = row_spans[bottom]
    if bottom_right - bottom_left < 2:
        raise ValueError("bottom edge too narrow")
    if not bool(np.all(comp_mask[bottom, bottom_left : bottom_right + 1])):
        raise ValueError("bottom edge must be contiguous")

    interior_mask = np.zeros((h, w), dtype=bool)
    interior_coords: list[tuple[int, int]] = []
    for r in range(top, bottom):
        left, right = row_spans[r]
        if right - left < 2:
            raise ValueError("open row too narrow")
        row_mask = np.zeros(w, dtype=bool)
        row_mask[left + 1 : right] = True
        row_mask &= ~comp_mask[r]
        row_coords = [(r, c) for c in np.flatnonzero(row_mask)]
        if not row_coords:
            raise ValueError("every open row must expose interior cells")
        interior_mask[r, row_mask] = True
        interior_coords.extend(row_coords)

    if not interior_coords:
        raise ValueError("cup interior is empty")

    top_left, top_right = row_spans[top]
    return {
        "component_mask": comp_mask,
        "interior_mask": interior_mask,
        "interior_coords": interior_coords,
        "top_row": top,
        "top_left": top_left,
        "top_right": top_right,
        "bottom_row": bottom,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
        "filled_count": len(interior_coords),
        "bbox": (top, min(c for _r, c in component), bottom, max(c for _r, c in component)),
    }


def _collect_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for component in _blue_components(grid):
        geom = _geometry_from_component(grid.shape, component)
        interior_values = grid[geom["interior_mask"]]
        marker_colors = set(int(v) for v in interior_values if int(v) not in (BLACK, BLUE))
        if len(marker_colors) != 1:
            raise ValueError("cup must contain exactly one marker color")
        marker_color = next(iter(marker_colors))
        if np.count_nonzero(interior_values == marker_color) != 1:
            raise ValueError("cup must contain exactly one marker pixel")
        if np.any((interior_values != BLACK) & (interior_values != marker_color)):
            raise ValueError("interior may only contain black cells and one marker")
        geom["marker_color"] = marker_color
        cups.append(geom)
    cups.sort(key=lambda cup: (cup["bbox"][1], cup["bbox"][0]))
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue cup with its internal marker color, add a cap, and let only odd-area cups drip downward.

    Input structure:
    - A black background contains 2-4 disjoint blue U-shaped cup outlines.
    - Each cup spans several rows, has a contiguous blue bottom edge, and has one colored marker pixel inside.
    - The interior is the visible cavity between the cup's left and right blue walls on each non-bottom row.

    Rule:
    - Detect each blue cup independently and read its unique internal marker color C.
    - Fill every interior black cell of that cup with C and paint one cap row above the cup mouth in the same color.
    - Count the total number of filled interior cells for that cup, excluding the cap and excluding blue outline pixels.
    - If the count is odd, grow rows below the blue bottom edge: first pink, then orange, alternating while each next row shrinks by one cell on both sides.

    Key insight:
    - The marker color always controls the fill and cap, but only the parity of the cup's filled interior area decides whether the pink/orange drip appears.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for cup in _collect_cups(g):
        color = cup["marker_color"]
        out[cup["interior_mask"]] = color

        cap_row = cup["top_row"] - 1
        if cap_row >= 0:
            out[cap_row, cup["top_left"] : cup["top_right"] + 1] = color

        if cup["filled_count"] % 2 == 0:
            continue

        left = cup["bottom_left"]
        right = cup["bottom_right"]
        drip_row = cup["bottom_row"] + 1
        drip_color = PINK
        while drip_row < g.shape[0] and left <= right:
            out[drip_row, left : right + 1] = drip_color
            drip_row += 1
            left += 1
            right -= 1
            drip_color = ORANGE if drip_color == PINK else PINK

    return out


def _connect_down(coords: set[tuple[int, int]], p0: tuple[int, int], p1: tuple[int, int]) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r1 != r0 + 1:
        raise ValueError("wall steps must move down one row at a time")
    for c in range(min(c0, c1), max(c0, c1) + 1):
        coords.add((r0, c))
    coords.add((r1, c1))


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _render_cup_profile(
    lefts: list[int],
    rights: list[int],
    bottom_left: int,
    bottom_right: int,
    width: int,
) -> np.ndarray:
    height = len(lefts) + 1
    coords: set[tuple[int, int]] = {(0, lefts[0]), (0, rights[0])}

    for r in range(len(lefts) - 1):
        _connect_down(coords, (r, lefts[r]), (r + 1, lefts[r + 1]))
        _connect_down(coords, (r, rights[r]), (r + 1, rights[r + 1]))
    _connect_down(coords, (height - 2, lefts[-1]), (height - 1, bottom_left))
    _connect_down(coords, (height - 2, rights[-1]), (height - 1, bottom_right))
    for c in range(bottom_left, bottom_right + 1):
        coords.add((height - 1, c))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return _crop_to_bbox(mask)


def _sample_cup_sprite(marker_color: int) -> tuple[np.ndarray, dict]:
    for _ in range(MAX_CUP_ATTEMPTS):
        width = random.randint(4, 6)
        height = random.randint(3, 5)
        open_rows = height - 1

        interior_width = random.randint(1, width - 2)
        left = random.randint(0, width - interior_width - 2)
        right = left + interior_width + 1
        lefts = [left]
        rights = [right]

        for _row in range(1, open_rows):
            candidates = []
            for nl in (left - 1, left, left + 1):
                for nr in (right - 1, right, right + 1):
                    if 0 <= nl < nr < width and nr - nl >= 2:
                        candidates.append((nl, nr))
            if not candidates:
                break
            left, right = random.choice(candidates)
            lefts.append(left)
            rights.append(right)
        if len(lefts) != open_rows:
            continue

        bottom_candidates = []
        for bl in (lefts[-1] - 1, lefts[-1], lefts[-1] + 1):
            for br in (rights[-1] - 1, rights[-1], rights[-1] + 1):
                if 0 <= bl < br < width and br - bl >= 2:
                    bottom_candidates.append((bl, br))
        if not bottom_candidates:
            continue
        bottom_left, bottom_right = random.choice(bottom_candidates)

        sprite_mask = _render_cup_profile(lefts, rights, bottom_left, bottom_right, width)
        if not (3 <= sprite_mask.shape[0] <= 5 and 4 <= sprite_mask.shape[1] <= 6):
            continue

        blue_only = np.where(sprite_mask, BLUE, BLACK)
        components = _blue_components(blue_only)
        if len(components) != 1:
            continue

        try:
            geom = _geometry_from_component(blue_only.shape, components[0])
        except ValueError:
            continue

        rect_count = (blue_only.shape[0] - 1) * (blue_only.shape[1] - 2)
        if geom["filled_count"] == rect_count:
            continue

        marker_r, marker_c = random.choice(geom["interior_coords"])
        sprite = blue_only.copy()
        sprite[marker_r, marker_c] = marker_color
        geom["marker_color"] = marker_color
        geom["marker_coord"] = (marker_r, marker_c)
        geom["irregular"] = True
        return sprite, geom

    raise RuntimeError("failed to sample irregular cup sprite")


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    grid[top : top + h, left : left + w] = np.where(sprite != BLACK, sprite, grid[top : top + h, left : left + w])


def _expected_from_layout(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        cup = item["geom"]
        color = item["color"]
        row_off = item["top"]
        col_off = item["left"]

        for r, c in cup["interior_coords"]:
            out[row_off + r, col_off + c] = color

        cap_row = row_off + cup["top_row"] - 1
        out[cap_row, col_off + cup["top_left"] : col_off + cup["top_right"] + 1] = color

        if cup["filled_count"] % 2 == 0:
            continue

        left = col_off + cup["bottom_left"]
        right = col_off + cup["bottom_right"]
        row = row_off + cup["bottom_row"] + 1
        paint = PINK
        while row < out.shape[0] and left <= right:
            out[row, left : right + 1] = paint
            row += 1
            left += 1
            right -= 1
            paint = ORANGE if paint == PINK else PINK
    return out


def _select_cups(count: int) -> list[dict] | None:
    marker_colors = random.sample(list(MARKER_COLORS), count)
    cups: list[dict] = []

    for color in marker_colors:
        for _ in range(MAX_CUP_ATTEMPTS):
            sprite, geom = _sample_cup_sprite(color)
            shape = tuple(int(x) for x in sprite.shape)
            filled_count = geom["filled_count"]
            if any(existing["geom"]["filled_count"] == filled_count for existing in cups):
                continue
            if any(tuple(int(x) for x in existing["sprite"].shape) == shape for existing in cups):
                continue
            cups.append({"sprite": sprite, "geom": geom, "color": color})
            break
        else:
            return None

    parities = {cup["geom"]["filled_count"] % 2 for cup in cups}
    if parities != {0, 1}:
        return None
    return cups


def _layout_attempt() -> np.ndarray | None:
    count = random.choices([2, 3, 4], weights=[2, 3, 2], k=1)[0]
    cups = _select_cups(count)
    if cups is None:
        return None

    gaps = [random.randint(2, 4) for _ in range(count - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(5, 8)

    grid_width = left_margin + right_margin + sum(cup["sprite"].shape[1] for cup in cups) + sum(gaps)
    grid_height = top_margin + max(cup["sprite"].shape[0] for cup in cups) + bottom_margin
    if grid_width > 30 or grid_height > 30:
        return None

    grid = np.full((grid_height, grid_width), BLACK, dtype=int)
    placements: list[dict] = []
    col = left_margin
    for idx, cup in enumerate(cups):
        sprite = cup["sprite"]
        geom = cup["geom"]
        top = top_margin
        _paste_sprite(grid, sprite, top, col)
        placements.append({"sprite": sprite, "geom": geom, "color": cup["color"], "top": top, "left": col})
        col += sprite.shape[1]
        if idx < len(gaps):
            col += gaps[idx]

    try:
        parsed = _collect_cups(grid)
    except ValueError:
        return None
    if len(parsed) != count:
        return None

    counts = sorted(cup["filled_count"] for cup in parsed)
    if len(set(counts)) != count:
        return None
    if {count % 2 for count in counts} != {0, 1}:
        return None

    expected = _expected_from_layout(grid, placements)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if expected.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _layout_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate puzzle input")
