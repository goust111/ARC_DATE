from __future__ import annotations

import random
import sys
from typing import Iterable

import numpy as np

_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste


Coord = tuple[int, int]


def _non_bg_cells(patch: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(patch != int(bg)):
        cells.append((int(r), int(c), int(patch[r, c])))
    return cells


def _stamp_cells(
    sprite: np.ndarray,
    bg: int,
    anchor: Coord,
    marker: Coord,
    shape: tuple[int, int],
) -> list[tuple[int, int, int]]:
    h, w = map(int, sprite.shape)
    ar, ac = map(int, anchor)
    mr, mc = map(int, marker)
    start_r = int(mr - ar)
    start_c = int(mc - ac)
    H, W = map(int, shape)

    cells: list[tuple[int, int, int]] = []
    for sr, sc, color in _non_bg_cells(sprite, bg):
        rr = start_r + sr
        cc = start_c + sc
        if 0 <= rr < H and 0 <= cc < W:
            cells.append((int(rr), int(cc), int(color)))
    return cells


def _build_anchor_map(sprite: np.ndarray, bg: int) -> dict[int, Coord]:
    anchors: dict[int, Coord] = {}
    for r, c in np.argwhere(sprite != int(bg)):
        color = int(sprite[r, c])
        assert color not in anchors, "each sprite color must occur at most once"
        anchors[color] = (int(r), int(c))
    assert len(anchors) >= 2, "expected a multicolor sprite"
    return anchors


def _parse_scene(grid: np.ndarray) -> tuple[int, np.ndarray, dict[int, Coord], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert components, "expected foreground components"

    sprite_components = [comp for comp in components if int(comp["area"]) > 1]
    marker_components = [comp for comp in components if int(comp["area"]) == 1]
    assert len(sprite_components) == 1, "expected exactly one multi-pixel palette sprite"
    assert marker_components, "expected isolated marker pixels"

    sprite_comp = sprite_components[0]
    sprite = crop_bbox(g, tuple(map(int, sprite_comp["bbox"])))
    anchors = _build_anchor_map(sprite, bg)

    markers: list[tuple[int, int, int]] = []
    for comp in marker_components:
        (r, c) = map(int, comp["coords"][0])
        markers.append((int(r), int(c), int(g[r, c])))
    markers.sort()
    return bg, sprite, anchors, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the palette sprite onto each matching isolated marker so the sprite cell of the same color lands on the marker.

    Input structure:
    - The background is dominant.
    - Exactly one 4-connected multi-pixel multicolor sprite appears; each of its colors occurs in at most one sprite cell.
    - Every other foreground component is an isolated single-pixel marker, and some marker colors do not occur in the sprite.

    Rule:
    - Keep the original sprite in place.
    - For each marker whose color occurs in the sprite, copy the whole sprite and translate it so the sprite cell of that color coincides with the marker pixel.
    - Clip copied cells that would fall outside the grid, and leave absent-color markers unchanged.

    Key insight:
    - The marker color identifies both which sprite to copy and the anchor position inside that sprite; the anchor is the color's location, not the sprite center.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprite, anchors, markers = _parse_scene(g)
    out = g.copy()

    H, W = map(int, g.shape)
    for r, c, color in markers:
        if color not in anchors:
            continue
        cells = _stamp_cells(sprite, bg, anchors[color], (r, c), (H, W))
        for rr, cc, value in cells:
            existing = int(out[rr, cc])
            assert existing in (int(bg), int(value)), "stamps must not create color conflicts"
        paste(out, sprite, int(r - anchors[color][0]), int(c - anchors[color][1]), transparent=bg, clip=True)

    return out


GRID_SIDE_RANGE = (14, 24)
SPRITE_SIDE_CHOICES = (3, 5)
SPRITE_AREA_RANGE = (4, 8)
MARKER_COUNT_RANGE = (2, 5)
DECOY_COUNT_RANGE = (1, 2)
GENERATION_ATTEMPTS = 400


def _is_connected(cells: set[Coord]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _make_sprite(bg: int) -> tuple[np.ndarray, dict[int, Coord], int]:
    h = random.choice(SPRITE_SIDE_CHOICES)
    w = random.choice(SPRITE_SIDE_CHOICES)
    center = (h // 2, w // 2)
    max_area = min(SPRITE_AREA_RANGE[1], h * w)
    area = random.randint(SPRITE_AREA_RANGE[0], max_area)

    cells = {center}
    while len(cells) < area:
        frontier = []
        for r, c in list(cells):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    assert _is_connected(cells)
    palette = random.sample([c for c in range(10) if c != int(bg)], len(cells))
    random.shuffle(palette)

    sprite = np.full((h, w), int(bg), dtype=int)
    anchors: dict[int, Coord] = {}
    for color, (r, c) in zip(palette, random.sample(list(cells), len(cells))):
        sprite[r, c] = int(color)
        anchors[int(color)] = (int(r), int(c))

    center_color = int(sprite[center])
    assert center_color != int(bg), "sprite center must be colored"
    assert any(pos != center for pos in anchors.values()), "need a non-center anchor"
    return sprite, anchors, center_color


def _cells_from_dict(mapping: dict[Coord, int]) -> set[Coord]:
    return set(mapping.keys())


def _touches_input(pos: Coord, occupied_input: set[Coord]) -> bool:
    r, c = map(int, pos)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (r + dr, c + dc) in occupied_input:
                return True
    return False


def _find_marker_spot(
    shape: tuple[int, int],
    sprite: np.ndarray,
    bg: int,
    anchor: Coord,
    occupied_input: set[Coord],
    occupied_output: dict[Coord, int],
    *,
    clipped: bool,
) -> tuple[Coord, list[tuple[int, int, int]]] | None:
    H, W = map(int, shape)
    full_area = int(np.count_nonzero(sprite != int(bg)))
    candidates: list[tuple[Coord, list[tuple[int, int, int]]]] = []

    for r in range(H):
        for c in range(W):
            pos = (int(r), int(c))
            if pos in occupied_input or _touches_input(pos, occupied_input):
                continue

            cells = _stamp_cells(sprite, bg, anchor, pos, shape)
            if len(cells) != full_area and not clipped:
                continue
            if len(cells) == full_area and clipped:
                continue
            if len(cells) <= 1:
                continue

            conflict = False
            for rr, cc, _value in cells:
                cell = (int(rr), int(cc))
                if cell in occupied_output:
                    conflict = True
                    break
                if cell in occupied_input and cell != pos:
                    conflict = True
                    break
            if conflict:
                continue

            candidates.append((pos, cells))

    if not candidates:
        return None
    return random.choice(candidates)


def _find_decoy_spot(
    shape: tuple[int, int],
    occupied_input: set[Coord],
    occupied_output: dict[Coord, int],
) -> Coord | None:
    H, W = map(int, shape)
    candidates: list[Coord] = []
    blocked = _cells_from_dict(occupied_output)
    for r in range(H):
        for c in range(W):
            pos = (int(r), int(c))
            if pos in occupied_input or pos in blocked or _touches_input(pos, occupied_input):
                continue
            candidates.append(pos)
    if not candidates:
        return None
    return random.choice(candidates)


def _render_expected(
    grid: np.ndarray,
    planned_stamps: Iterable[list[tuple[int, int, int]]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for cells in planned_stamps:
        for r, c, color in cells:
            out[int(r), int(c)] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    bg = random.randrange(10)

    sprite, anchors, center_color = _make_sprite(bg)
    h, w = map(int, sprite.shape)
    if H <= h + 2 or W <= w + 2:
        return None

    master_r = random.randint(1, H - h - 1)
    master_c = random.randint(1, W - w - 1)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, sprite, master_r, master_c, transparent=bg)

    occupied_input: set[Coord] = set()
    occupied_output: dict[Coord, int] = {}
    for sr, sc, color in _non_bg_cells(sprite, bg):
        cell = (int(master_r + sr), int(master_c + sc))
        occupied_input.add(cell)
        occupied_output[cell] = int(color)

    sprite_colors = list(anchors.keys())
    non_center_colors = [color for color, pos in anchors.items() if pos != (h // 2, w // 2)]
    if not non_center_colors:
        return None

    present_target = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(sprite_colors)))
    forced_colors = {random.choice(non_center_colors)}
    if random.random() < 0.7:
        forced_colors.add(center_color)
    while len(forced_colors) < present_target:
        forced_colors.add(random.choice(sprite_colors))
    present_colors = list(forced_colors)
    random.shuffle(present_colors)

    planned_stamps: list[list[tuple[int, int, int]]] = []

    first_color = present_colors[0]
    clipped_pick = _find_marker_spot(
        grid.shape,
        sprite,
        bg,
        anchors[first_color],
        occupied_input,
        occupied_output,
        clipped=True,
    )
    if clipped_pick is None:
        return None
    (mr, mc), cells = clipped_pick
    grid[mr, mc] = int(first_color)
    occupied_input.add((mr, mc))
    for rr, cc, value in cells:
        occupied_output[(rr, cc)] = int(value)
    planned_stamps.append(cells)

    for color in present_colors[1:]:
        want_clipped = random.random() < 0.25
        placed = _find_marker_spot(
            grid.shape,
            sprite,
            bg,
            anchors[color],
            occupied_input,
            occupied_output,
            clipped=want_clipped,
        )
        if placed is None and want_clipped:
            placed = _find_marker_spot(
                grid.shape,
                sprite,
                bg,
                anchors[color],
                occupied_input,
                occupied_output,
                clipped=False,
            )
        if placed is None:
            return None
        (mr, mc), cells = placed
        grid[mr, mc] = int(color)
        occupied_input.add((mr, mc))
        for rr, cc, value in cells:
            occupied_output[(rr, cc)] = int(value)
        planned_stamps.append(cells)

    absent_palette = [color for color in range(10) if color not in anchors and color != int(bg)]
    if not absent_palette:
        return None
    decoy_count = random.randint(*DECOY_COUNT_RANGE)
    decoy_colors = random.sample(absent_palette, min(decoy_count, len(absent_palette)))
    for color in decoy_colors:
        pos = _find_decoy_spot(grid.shape, occupied_input, occupied_output)
        if pos is None:
            return None
        grid[pos] = int(color)
        occupied_input.add(pos)

    expected = _render_expected(grid, planned_stamps)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if len(decoy_colors) == 0:
        return None

    clipped_count = sum(
        1
        for cells in planned_stamps
        if len(cells) < int(np.count_nonzero(sprite != int(bg)))
    )
    if clipped_count == 0:
        return None
    if not any(color in anchors and anchors[color] != (h // 2, w // 2) for color in present_colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-anchor stamping input")
