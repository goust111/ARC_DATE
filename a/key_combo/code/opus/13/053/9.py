from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BG = 0
MARKER_COLOR = 2
SIDES = ("top", "right", "bottom", "left")
DIRECTIONS = ("up", "right", "down", "left")
SIDE_TO_DEFAULT_DIRECTION = {
    "top": "up",
    "right": "right",
    "bottom": "down",
    "left": "left",
}
DIR_TO_CORNER = {
    "up": (0, 0),
    "right": (0, 3),
    "down": (3, 3),
    "left": (3, 0),
}
DIR_TO_T_COORDS = {
    "up": frozenset(((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))),
    "right": frozenset(((0, 2), (1, 0), (1, 1), (1, 2), (2, 2))),
    "down": frozenset(((0, 1), (1, 1), (2, 0), (2, 1), (2, 2))),
    "left": frozenset(((0, 0), (1, 0), (1, 1), (1, 2), (2, 0))),
}
SPRITE_COLOR_POOL = (1, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 500
EXTRA_MARGIN_CHOICES = (0, 1, 2)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _union_bbox(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _identify_t_direction(mask: np.ndarray) -> str:
    assert mask.shape == (3, 3)
    coords = frozenset((int(r), int(c)) for r, c in np.argwhere(mask))
    for direction, expected in DIR_TO_T_COORDS.items():
        if coords == expected:
            return direction
    raise AssertionError("invalid T marker")


def _marker_side(
    marker_bbox: tuple[int, int, int, int],
    compact_bbox: tuple[int, int, int, int],
) -> str:
    mr0, mc0, mr1, mc1 = marker_bbox
    sr0, sc0, sr1, sc1 = compact_bbox
    if mr1 < sr0:
        assert sr0 - mr1 == 2
        return "top"
    if mr0 > sr1:
        assert mr0 - sr1 == 2
        return "bottom"
    if mc1 < sc0:
        assert sc0 - mc1 == 2
        return "left"
    if mc0 > sc1:
        assert mc0 - sc1 == 2
        return "right"
    raise AssertionError("marker must be outside the compact sprite region")


def _sprite_center2_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    assert (r1 - r0, c1 - c0) == (1, 1)
    return 2 * int(r0) + 1, 2 * int(c0) + 1


def _marker_center2_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    assert (r1 - r0, c1 - c0) == (2, 2)
    return 2 * (int(r0) + 1), 2 * (int(c0) + 1)


def _render_output(corner_to_color: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.zeros((5, 5), dtype=int)
    for (r0, c0), color in corner_to_color.items():
        out[r0 : r0 + 2, c0 : c0 + 2] = int(color)
    return out


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    sprite_colors = sorted(set(map(int, np.unique(g))) - {bg, MARKER_COLOR})
    assert len(sprite_colors) == 4

    sprites: list[dict] = []
    for color in sprite_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (4, 2)
        bbox = _bbox_from_coords(coords)
        assert (_bbox_h(bbox), _bbox_w(bbox)) == (2, 2)
        r0, c0, r1, c1 = bbox
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)
        sprites.append(
            {
                "color": int(color),
                "bbox": bbox,
                "center2": _sprite_center2_from_bbox(bbox),
            }
        )

    compact_bbox = _union_bbox([sprite["bbox"] for sprite in sprites])

    marker_components = _components_of_color(g, MARKER_COLOR)
    assert len(marker_components) == 4
    markers: list[dict] = []
    for coords in marker_components:
        assert coords.shape[0] == 5
        bbox = _bbox_from_coords(coords)
        assert (_bbox_h(bbox), _bbox_w(bbox)) == (3, 3)
        patch = _crop(g, bbox) == MARKER_COLOR
        direction = _identify_t_direction(patch)
        markers.append(
            {
                "bbox": bbox,
                "direction": direction,
                "side": _marker_side(bbox, compact_bbox),
                "center2": _marker_center2_from_bbox(bbox),
            }
        )

    assert len({marker["direction"] for marker in markers}) == 4
    assert len({marker["side"] for marker in markers}) == 4

    claimed_sprites: list[int] = []
    for marker in markers:
        distances = [
            abs(marker["center2"][0] - sprite["center2"][0])
            + abs(marker["center2"][1] - sprite["center2"][1])
            for sprite in sprites
        ]
        best = min(distances)
        assert distances.count(best) == 1
        marker["sprite_index"] = int(distances.index(best))
        claimed_sprites.append(marker["sprite_index"])

    assert len(set(claimed_sprites)) == 4
    return sprites, markers


def main(grid: np.ndarray) -> np.ndarray:
    """Separate four compact 2x2 sprites and place each one into the corner named by its nearby T marker.

    Input structure:
    - A black background contains four differently colored 2x2 sprites packed into one compact central region.
    - Four red 3x3 T markers sit outside that compact region, one on each of the top, right, bottom, and left sides, with a one-cell gap.
    - Each red T has one of the four distinct orientations up/right/down/left, and each marker has a unique nearest sprite by Manhattan distance between their centers.

    Rule:
    - Identify the four colored 2x2 sprites and the four red T markers.
    - Pair each T marker with its unique nearest sprite.
    - Read the T orientation as a corner instruction: up -> upper-left, right -> upper-right, down -> lower-right, left -> lower-left.
    - Output a 5x5 black grid and stamp each paired 2x2 sprite into the instructed corner, leaving the middle row and middle column black.

    Key insight:
    - The central cluster only tells you which four sprites exist; the nearby red T orientations decide where each separated sprite must go in the final four-corner layout.
    """
    sprites, markers = _parse_scene(np.asarray(grid, dtype=int))

    corner_to_color: dict[tuple[int, int], int] = {}
    for marker in markers:
        sprite = sprites[marker["sprite_index"]]
        corner = DIR_TO_CORNER[marker["direction"]]
        assert corner not in corner_to_color
        corner_to_color[corner] = int(sprite["color"])

    assert len(corner_to_color) == 4
    return _render_output(corner_to_color)


def _square_cells(anchor: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = anchor
    return {(r + dr, c + dc) for dr in range(2) for dc in range(2)}


def _squares_touch(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return any(
        abs(r1 - r2) + abs(c1 - c2) == 1
        for r1, c1 in _square_cells(a)
        for r2, c2 in _square_cells(b)
    )


def _normalize_layout(layout: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in layout)
    min_c = min(c for _, c in layout)
    return tuple(sorted((r - min_r, c - min_c) for r, c in layout))


def _layout_shape(layout: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    cells = [cell for anchor in layout for cell in _square_cells(anchor)]
    coords = np.array(cells, dtype=int)
    bbox = _bbox_from_coords(coords)
    return _bbox_h(bbox), _bbox_w(bbox)


def _layout_is_connected(layout: tuple[tuple[int, int], ...]) -> bool:
    pending = [0]
    seen = {0}
    while pending:
        i = pending.pop()
        for j in range(len(layout)):
            if j in seen:
                continue
            if _squares_touch(layout[i], layout[j]):
                seen.add(j)
                pending.append(j)
    return len(seen) == len(layout)


def _enumerate_layouts() -> dict[tuple[int, int], list[tuple[tuple[int, int], ...]]]:
    grouped: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {}
    anchors = [(r, c) for r in range(4) for c in range(4)]
    for combo in itertools.combinations(anchors, 4):
        occupied: set[tuple[int, int]] = set()
        ok = True
        for anchor in combo:
            cells = _square_cells(anchor)
            if occupied & cells:
                ok = False
                break
            occupied |= cells
        if not ok:
            continue
        layout = _normalize_layout(combo)
        if not _layout_is_connected(layout):
            continue
        shape = _layout_shape(layout)
        if shape[0] not in (4, 5) or shape[1] not in (4, 5):
            continue
        grouped.setdefault(shape, [])
        if layout not in grouped[shape]:
            grouped[shape].append(layout)
    assert grouped
    return grouped


LAYOUTS_BY_SHAPE = _enumerate_layouts()


def _marker_position_candidates(
    side: str,
    region_h: int,
    region_w: int,
) -> list[tuple[int, int]]:
    if side == "top":
        return [(-4, c) for c in range(-1, region_w - 1)]
    if side == "bottom":
        return [(region_h + 1, c) for c in range(-1, region_w - 1)]
    if side == "left":
        return [(r, -4) for r in range(-1, region_h - 1)]
    if side == "right":
        return [(r, region_w + 1) for r in range(-1, region_h - 1)]
    raise ValueError(side)


def _layout_candidate_map(
    layout: tuple[tuple[int, int], ...]
) -> dict[str, dict[int, list[tuple[int, int]]]]:
    region_h, region_w = _layout_shape(layout)
    sprite_centers = [(2 * r + 1, 2 * c + 1) for r, c in layout]
    out: dict[str, dict[int, list[tuple[int, int]]]] = {side: {} for side in SIDES}

    for side in SIDES:
        for marker_pos in _marker_position_candidates(side, region_h, region_w):
            mr, mc = marker_pos
            center2 = (2 * (mr + 1), 2 * (mc + 1))
            distances = [
                abs(center2[0] - sprite_center[0]) + abs(center2[1] - sprite_center[1])
                for sprite_center in sprite_centers
            ]
            best = min(distances)
            if distances.count(best) != 1:
                continue
            sprite_index = int(distances.index(best))
            out[side].setdefault(sprite_index, []).append(marker_pos)
    return out


def _perfect_matchings(
    candidate_map: dict[str, dict[int, list[tuple[int, int]]]]
) -> list[dict[str, int]]:
    matchings: list[dict[str, int]] = []
    for perm in itertools.permutations(range(4)):
        side_to_sprite = dict(zip(SIDES, perm, strict=True))
        if all(side_to_sprite[side] in candidate_map[side] for side in SIDES):
            matchings.append(side_to_sprite)
    return matchings


def _render_t(direction: str) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for r, c in DIR_TO_T_COORDS[direction]:
        patch[r, c] = MARKER_COLOR
    return patch


def _build_input_grid(
    layout: tuple[tuple[int, int], ...],
    sprite_colors: list[int],
    side_to_direction: dict[str, str],
    side_to_sprite: dict[str, int],
    side_to_marker_pos: dict[str, tuple[int, int]],
) -> np.ndarray:
    region_h, region_w = _layout_shape(layout)
    margin_top = 4 + random.choice(EXTRA_MARGIN_CHOICES)
    margin_left = 4 + random.choice(EXTRA_MARGIN_CHOICES)
    margin_bottom = 4 + random.choice(EXTRA_MARGIN_CHOICES)
    margin_right = 4 + random.choice(EXTRA_MARGIN_CHOICES)
    grid = np.zeros(
        (region_h + margin_top + margin_bottom, region_w + margin_left + margin_right),
        dtype=int,
    )

    for sprite_index, (r, c) in enumerate(layout):
        color = int(sprite_colors[sprite_index])
        rr = margin_top + r
        cc = margin_left + c
        grid[rr : rr + 2, cc : cc + 2] = color

    for side in SIDES:
        direction = side_to_direction[side]
        marker_r, marker_c = side_to_marker_pos[side]
        rr = margin_top + marker_r
        cc = margin_left + marker_c
        patch = _render_t(direction)
        for r, c in np.argwhere(patch == MARKER_COLOR):
            grid[rr + int(r), cc + int(c)] = MARKER_COLOR

    return grid


def _expected_output(
    sprite_colors: list[int],
    side_to_direction: dict[str, str],
    side_to_sprite: dict[str, int],
) -> np.ndarray:
    corner_to_color = {
        DIR_TO_CORNER[side_to_direction[side]]: int(sprite_colors[side_to_sprite[side]])
        for side in SIDES
    }
    return _render_output(corner_to_color)


def _naive_side_output(sprite_colors: list[int], side_to_sprite: dict[str, int]) -> np.ndarray:
    corner_to_color = {
        DIR_TO_CORNER[SIDE_TO_DEFAULT_DIRECTION[side]]: int(sprite_colors[side_to_sprite[side]])
        for side in SIDES
    }
    return _render_output(corner_to_color)


def _quadrant_output(
    layout: tuple[tuple[int, int], ...],
    sprite_colors: list[int],
) -> np.ndarray:
    centers = [(2 * r + 1, 2 * c + 1) for r, c in layout]
    row_order = sorted(range(4), key=lambda i: (centers[i][0], centers[i][1]))
    col_order = sorted(range(4), key=lambda i: (centers[i][1], centers[i][0]))
    top_two = set(row_order[:2])
    left_two = set(col_order[:2])

    corner_to_color: dict[tuple[int, int], int] = {}
    for i in range(4):
        if i in top_two and i in left_two:
            corner = (0, 0)
        elif i in top_two and i not in left_two:
            corner = (0, 3)
        elif i not in top_two and i in left_two:
            corner = (3, 0)
        else:
            corner = (3, 3)
        assert corner not in corner_to_color
        corner_to_color[corner] = int(sprite_colors[i])
    return _render_output(corner_to_color)


def _same_matching(
    parsed_markers: list[dict],
    layout: tuple[tuple[int, int], ...],
    side_to_sprite: dict[str, int],
    sprite_colors: list[int],
) -> bool:
    local_sprite_by_color = {
        int(sprite_colors[i]): int(i)
        for i in range(4)
    }
    parsed_by_side = {marker["side"]: marker for marker in parsed_markers}
    for side in SIDES:
        marker = parsed_by_side[side]
        parsed_color = marker["paired_color"]
        if local_sprite_by_color[parsed_color] != side_to_sprite[side]:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    shape = random.choice(sorted(LAYOUTS_BY_SHAPE))
    layout = random.choice(LAYOUTS_BY_SHAPE[shape])
    candidate_map = _layout_candidate_map(layout)
    matchings = _perfect_matchings(candidate_map)
    if not matchings:
        return None

    side_to_sprite = random.choice(matchings)
    side_to_marker_pos = {
        side: random.choice(candidate_map[side][side_to_sprite[side]])
        for side in SIDES
    }
    direction_perm = random.sample(DIRECTIONS, k=4)
    side_to_direction = dict(zip(SIDES, direction_perm, strict=True))

    sprite_colors = random.sample(SPRITE_COLOR_POOL, k=4)
    expected = _expected_output(sprite_colors, side_to_direction, side_to_sprite)
    if np.array_equal(expected, _naive_side_output(sprite_colors, side_to_sprite)):
        return None
    if np.array_equal(expected, _quadrant_output(layout, sprite_colors)):
        return None

    grid = _build_input_grid(layout, sprite_colors, side_to_direction, side_to_sprite, side_to_marker_pos)
    if max(grid.shape) > 30:
        return None

    try:
        sprites, markers = _parse_scene(grid)
        parsed_by_side = {marker["side"]: marker for marker in markers}
        for marker in markers:
            marker["paired_color"] = int(sprites[marker["sprite_index"]]["color"])
        if not _same_matching(markers, layout, side_to_sprite, sprite_colors):
            return None
        if {parsed_by_side[side]["direction"] for side in SIDES} != set(DIRECTIONS):
            return None
        solved = main(grid)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
