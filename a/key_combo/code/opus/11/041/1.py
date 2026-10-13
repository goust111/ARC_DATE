from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
GREY = 5
RECT_COLORS = (1, 2, 3)
COLOR_DRAW_ORDER = (1, 2, 3)
CARDINALS = ("up", "down", "left", "right")
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

# Each template is a grey-body shape around a colored marker at (0, 0).
# The body centroid is biased toward the key direction.
SPRITE_TEMPLATES = {
    "left": [
        [(-1, -2), (0, -2), (1, -2), (1, -1), (0, -1)],
        [(-1, -2), (0, -2), (1, -2), (1, -1), (0, -1), (-1, -1)],
    ],
    "right": [
        [(-1, 2), (0, 2), (1, 2), (1, 1), (0, 1)],
        [(-1, 2), (0, 2), (1, 2), (1, 1), (0, 1), (-1, 1)],
    ],
    "up": [
        [(-2, -1), (-2, 0), (-2, 1), (-1, 1), (-1, 0)],
        [(-2, -1), (-2, 0), (-2, 1), (-1, 1), (-1, 0), (-1, -1)],
    ],
    "down": [
        [(2, -1), (2, 0), (2, 1), (1, 1), (1, 0)],
        [(2, -1), (2, 0), (2, 1), (1, 1), (1, 0), (1, -1)],
    ],
}

_GEN_COUNTER = 0


@dataclass(frozen=True)
class SpriteInfo:
    color: int
    marker: tuple[int, int]
    grey_coords: tuple[tuple[int, int], ...]
    body_direction: str
    stretch_direction: str
    cells: frozenset[tuple[int, int]]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _inflate_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _box_margin(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return r0, h - 1 - r1, c0, w - 1 - c1


def _draw_hollow_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_hollow_rectangle_component(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == BG))


def _expand_box(box: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    if direction == "up":
        return 0, c0, r1, c1
    if direction == "down":
        return r0, c0, h - 1, c1
    if direction == "left":
        return r0, 0, r1, c1
    if direction == "right":
        return r0, c0, r1, w - 1
    raise ValueError(direction)


def _decode_body_direction(grey_coords: np.ndarray, marker: tuple[int, int]) -> str:
    mr, mc = marker
    centroid_r = float(grey_coords[:, 0].mean())
    centroid_c = float(grey_coords[:, 1].mean())
    dr = centroid_r - mr
    dc = centroid_c - mc
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _find_rectangles(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    rectangles: dict[int, tuple[int, int, int, int]] = {}
    for color in RECT_COLORS:
        comps = _components(grid == color)
        hits = [coords for coords in comps if _is_hollow_rectangle_component(grid, coords, color)]
        assert len(hits) == 1, f"expected one hollow rectangle of color {color}"
        rectangles[color] = _bbox_coords(hits[0])
    return rectangles


def _find_sprites(grid: np.ndarray) -> dict[int, SpriteInfo]:
    infos: dict[int, SpriteInfo] = {}
    grey_components = _components(grid == GREY)
    assert len(grey_components) == 3, "expected three grey directional sprites"
    used_markers: set[tuple[int, int]] = set()
    for coords in grey_components:
        grey_set = {tuple(map(int, rc)) for rc in coords}
        marker_hits: set[tuple[int, int]] = set()
        marker_colors: set[int] = set()
        for r, c in grey_set:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                color = int(grid[nr, nc])
                if color in RECT_COLORS:
                    marker_hits.add((nr, nc))
                    marker_colors.add(color)
        assert len(marker_hits) == 1, "sprite must have exactly one adjacent colored marker"
        assert len(marker_colors) == 1
        marker = next(iter(marker_hits))
        assert marker not in used_markers
        used_markers.add(marker)
        color = next(iter(marker_colors))
        body_direction = _decode_body_direction(coords, marker)
        info = SpriteInfo(
            color=color,
            marker=marker,
            grey_coords=tuple(sorted(grey_set)),
            body_direction=body_direction,
            stretch_direction=OPPOSITE[body_direction],
            cells=frozenset(grey_set | {marker}),
        )
        infos[color] = info
    assert set(infos) == set(RECT_COLORS)
    return infos


def _parse_scene(grid: np.ndarray) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, SpriteInfo]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    rectangles = _find_rectangles(g)
    sprites = _find_sprites(g)
    assert set(rectangles) == set(sprites)
    return rectangles, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each grey directional sprite and stretch the same-colored hollow rectangle toward the opposite side until it reaches the grid boundary.

    Input structure:
    - The grid has a black background with exactly three hollow rectangular frames in blue, red, and green.
    - Each frame is paired by color with one nearby directional sprite made of grey body pixels plus one embedded colored marker pixel.
    - Every sprite body is an irregular 4-connected grey cluster whose centroid is clearly biased to one side of its colored marker.

    Rule:
    - For each sprite, compute the centroid of its grey body pixels and compare it to the colored marker position.
    - The centroid offset gives the body direction, and the rectangle stretches in the opposite direction.
    - Match the sprite marker color to the frame border color, move only that one side of the matching hollow rectangle to the corresponding grid edge, keep the other three sides fixed, and erase all sprite pixels to black.
    - Apply the final redraw order blue, then red, then green so later colors overwrite any border-line overlaps.

    Key insight:
    - The grey body sits behind the motion cue: the centroid points opposite the rectangle growth direction, so color matching and direction reversal are both necessary.
    """
    g = np.asarray(grid, dtype=int)
    rectangles, sprites = _parse_scene(g)

    out = g.copy()
    for info in sprites.values():
        for r, c in info.cells:
            out[r, c] = BG

    for color in COLOR_DRAW_ORDER:
        box = rectangles[color]
        stretch_direction = sprites[color].stretch_direction
        expanded = _expand_box(box, stretch_direction, g.shape)
        _draw_hollow_rectangle(out, expanded, color)
    return out


def _sprite_box(marker: tuple[int, int], template: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    all_cells = [(0, 0), *template]
    rs = [marker[0] + dr for dr, _ in all_cells]
    cs = [marker[1] + dc for _, dc in all_cells]
    return min(rs), min(cs), max(rs), max(cs)


def _stamp_sprite(grid: np.ndarray, marker: tuple[int, int], color: int, template: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    cells = {marker}
    for dr, dc in template:
        rr = marker[0] + dr
        cc = marker[1] + dc
        grid[rr, cc] = GREY
        cells.add((rr, cc))
    grid[marker] = color
    return frozenset(cells)


def _all_cells_of_box(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _choose_mode_directions(counter: int) -> dict[int, str]:
    target = CARDINALS[counter % 4]
    dirs = [target]
    while len(dirs) < 3:
        dirs.append(random.choice(CARDINALS))
    random.shuffle(dirs)
    return {color: dirs[i] for i, color in enumerate(RECT_COLORS)}


def _pick_original_box(shape: tuple[int, int], direction: str) -> tuple[int, int, int, int] | None:
    h, w = shape
    rect_h = random.randint(3, 5)
    rect_w = random.randint(3, 5)
    margin = 3
    r0_lo = margin
    r0_hi = h - rect_h - margin
    c0_lo = margin
    c0_hi = w - rect_w - margin
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    if direction == "up" and r0 < margin:
        return None
    if direction == "down" and h - 1 - r1 < margin:
        return None
    if direction == "left" and c0 < margin:
        return None
    if direction == "right" and w - 1 - c1 < margin:
        return None
    return r0, c0, r1, c1


def _can_place_box(
    box: tuple[int, int, int, int],
    final_box: tuple[int, int, int, int],
    placed_boxes: list[tuple[int, int, int, int]],
    placed_final_boxes: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> bool:
    padded_box = _inflate_box(box, 1, shape)
    for other in placed_boxes:
        if _box_intersects(padded_box, other):
            return False
    for other_final in placed_final_boxes:
        if _box_intersects(final_box, other_final):
            return False
    return True


def _place_rectangles(shape: tuple[int, int], directions: dict[int, str]) -> dict[int, tuple[int, int, int, int]] | None:
    rectangles: dict[int, tuple[int, int, int, int]] = {}
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_final_boxes: list[tuple[int, int, int, int]] = []
    colors = list(RECT_COLORS)
    random.shuffle(colors)
    for color in colors:
        direction = directions[color]
        placed = None
        for _ in range(300):
            box = _pick_original_box(shape, direction)
            if box is None:
                continue
            final_box = _expand_box(box, direction, shape)
            if not _can_place_box(box, final_box, placed_boxes, placed_final_boxes, shape):
                continue
            placed = box
            placed_boxes.append(box)
            placed_final_boxes.append(final_box)
            rectangles[color] = box
            break
        if placed is None:
            return None
    return rectangles


def _candidate_marker_positions(
    shape: tuple[int, int],
    template: list[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    rel_cells = [(0, 0), *template]
    for mr in range(h):
        for mc in range(w):
            ok = True
            for dr, dc in rel_cells:
                rr = mr + dr
                cc = mc + dc
                if not _in_bounds(shape, rr, cc) or blocked[rr, cc]:
                    ok = False
                    break
            if ok:
                candidates.append((mr, mc))
    random.shuffle(candidates)
    return candidates


def _place_sprites(
    grid: np.ndarray,
    rectangles: dict[int, tuple[int, int, int, int]],
    directions: dict[int, str],
) -> dict[int, frozenset[tuple[int, int]]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for color in RECT_COLORS:
        final_box = _expand_box(rectangles[color], directions[color], grid.shape)
        r0, c0, r1, c1 = _inflate_box(final_box, 1, grid.shape)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    sprites: dict[int, frozenset[tuple[int, int]]] = {}
    colors = list(RECT_COLORS)
    random.shuffle(colors)
    for color in colors:
        body_direction = OPPOSITE[directions[color]]
        templates = SPRITE_TEMPLATES[body_direction][:]
        random.shuffle(templates)
        placed = None
        for template in templates:
            for marker in _candidate_marker_positions(grid.shape, template, blocked):
                sprite_box = _sprite_box(marker, template)
                box_pad = _inflate_box(sprite_box, 1, grid.shape)
                if any(blocked[r, c] for r in range(box_pad[0], box_pad[2] + 1) for c in range(box_pad[1], box_pad[3] + 1)):
                    continue
                cells = _stamp_sprite(grid, marker, color, template)
                sprites[color] = cells
                for r, c in cells:
                    blocked[r, c] = True
                for r in range(box_pad[0], box_pad[2] + 1):
                    for c in range(box_pad[1], box_pad[3] + 1):
                        blocked[r, c] = True
                placed = cells
                break
            if placed is not None:
                break
        if placed is None:
            return None
    return sprites


def _all_direction_expansions(box: tuple[int, int, int, int], shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    return {direction: _expand_box(box, direction, shape) for direction in CARDINALS}


def _audit_case(grid: np.ndarray) -> None:
    rectangles, sprites = _parse_scene(grid)
    out = main(grid)

    assert grid.shape == out.shape
    assert set(rectangles) == set(RECT_COLORS) == set(sprites)

    # All three rectangles are away from every edge, so every direction would yield a distinct expansion.
    for color, box in rectangles.items():
        top, bottom, left, right = _box_margin(box, grid.shape)
        assert min(top, bottom, left, right) >= 3
        expansions = _all_direction_expansions(box, grid.shape)
        assert len(set(expansions.values())) == 4
        chosen = expansions[sprites[color].stretch_direction]
        assert chosen != box
        rr0, cc0, rr1, cc1 = chosen
        if sprites[color].stretch_direction == "up":
            assert rr0 == 0 and rr1 == box[2] and cc0 == box[1] and cc1 == box[3]
        elif sprites[color].stretch_direction == "down":
            assert rr1 == grid.shape[0] - 1 and rr0 == box[0] and cc0 == box[1] and cc1 == box[3]
        elif sprites[color].stretch_direction == "left":
            assert cc0 == 0 and rr0 == box[0] and rr1 == box[2] and cc1 == box[3]
        else:
            assert cc1 == grid.shape[1] - 1 and rr0 == box[0] and rr1 == box[2] and cc0 == box[1]

    # Sprites must be fully erased in the output.
    assert int((out == GREY).sum()) == 0
    for color, info in sprites.items():
        assert out[info.marker] == BG
        grey_coords = np.array(info.grey_coords, dtype=int)
        body = _decode_body_direction(grey_coords, info.marker)
        assert body == info.body_direction
        assert OPPOSITE[body] == info.stretch_direction

    # Output must contain exactly three hollow rectangles of the matched colors.
    out_rectangles = _find_rectangles(out)
    assert set(out_rectangles) == set(RECT_COLORS)
    for color in RECT_COLORS:
        assert out_rectangles[color] == _expand_box(rectangles[color], sprites[color].stretch_direction, grid.shape)

    # Generated inputs intentionally keep growth regions disjoint for clarity.
    final_boxes = [_expand_box(rectangles[color], sprites[color].stretch_direction, grid.shape) for color in RECT_COLORS]
    for i in range(len(final_boxes)):
        for j in range(i + 1, len(final_boxes)):
            assert not _box_intersects(final_boxes[i], final_boxes[j])


def _build_case() -> np.ndarray | None:
    global _GEN_COUNTER
    _GEN_COUNTER += 1

    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    directions = _choose_mode_directions(_GEN_COUNTER)
    rectangles = _place_rectangles(grid.shape, directions)
    if rectangles is None:
        return None

    for color in RECT_COLORS:
        _draw_hollow_rectangle(grid, rectangles[color], color)

    sprites = _place_sprites(grid, rectangles, directions)
    if sprites is None:
        return None

    try:
        _audit_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional-stretch ARC input")
