from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BODY = 5
STAMP_ORANGE = 7
STAMP_PINK = 6
MARKER_COLORS = (1, 2, 3, 4)

DIR_TO_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}
OPPOSITE = {"up": "down", "right": "left", "down": "up", "left": "right"}

# Same directional body geometry as the centroid-bias reference primitive:
# the grey mass sits behind the marker, so its centroid points opposite
# the sprite's pointing direction.
BODY_REL = {
    "up": [(0, -2), (0, -1), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1)],
    "down": [(-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 1), (0, 2)],
    "left": [(-2, 0), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1), (2, 0)],
    "right": [(-2, 0), (-1, -1), (-1, 0), (0, -1), (1, -1), (1, 0), (2, 0)],
}

BASE_STAMP = np.array(
    [
        [STAMP_ORANGE, STAMP_ORANGE, BG],
        [BG, BG, BG],
        [BG, BG, STAMP_PINK],
    ],
    dtype=int,
)

_SPRITE_EXTENT: dict[str, tuple[int, int, int, int]] = {}
for _name, _rels in BODY_REL.items():
    _all = _rels + [(0, 0)]
    _SPRITE_EXTENT[_name] = (
        min(dr for dr, _dc in _all),
        max(dr for dr, _dc in _all),
        min(dc for _dr, dc in _all),
        max(dc for _dr, dc in _all),
    )


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return out
    for r, c in coords:
        r0 = max(0, int(r) - radius)
        r1 = min(H, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(W, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    H, W = dst.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _sprite_cells(direction: str, marker_r: int, marker_c: int) -> list[tuple[int, int]]:
    return [(marker_r + dr, marker_c + dc) for dr, dc in BODY_REL[direction]]


def _marker_ranges(H: int, W: int, direction: str) -> tuple[range, range]:
    min_r, max_r, min_c, max_c = _SPRITE_EXTENT[direction]
    return (
        range(-min_r, H - max_r),
        range(-min_c, W - max_c),
    )


def _bias_to_direction(row_bias: int, col_bias: int) -> str:
    if abs(row_bias) == abs(col_bias):
        raise AssertionError("centroid bias must resolve to a unique cardinal direction")
    if abs(row_bias) > abs(col_bias):
        return "down" if row_bias > 0 else "up"
    return "right" if col_bias > 0 else "left"


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components8(g != BG)

    sprites: list[dict] = []
    anchors: dict[int, tuple[int, int]] = {}
    for comp in comps:
        values = [int(g[r, c]) for r, c in comp]
        if len(comp) == 1:
            color = values[0]
            assert color in MARKER_COLORS, "anchor color must be one of the marker colors"
            assert color not in anchors, "anchor colors must be unique"
            anchors[color] = comp[0]
            continue

        gray = [(r, c) for r, c in comp if int(g[r, c]) == BODY]
        markers = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) not in (BG, BODY)]
        assert len(markers) == 1, "each sprite must contain exactly one colored marker"
        assert 5 <= len(gray) <= 7, "sprite must contain 5-7 grey body pixels"
        mr, mc, color = markers[0]
        assert color in MARKER_COLORS, "marker color must be in the allowed palette"

        row_bias = sum(r - mr for r, _c in gray)
        col_bias = sum(c - mc for _r, c in gray)
        assert row_bias != 0 or col_bias != 0, "grey centroid must be offset from the marker"
        tail = _bias_to_direction(int(row_bias), int(col_bias))
        pointing = OPPOSITE[tail]

        sprites.append(
            {
                "color": color,
                "marker": (mr, mc),
                "coords": tuple(comp),
                "pointing": pointing,
                "turns": DIR_TO_TURNS[pointing],
            }
        )

    assert 2 <= len(sprites) <= 4, "expected 2-4 directional sprites"
    assert len(anchors) == len(sprites), "anchor count must match sprite count"
    sprite_colors = {sprite["color"] for sprite in sprites}
    assert sprite_colors == set(anchors), "sprite markers and anchors must match by color"
    return sprites, anchors


def _scene_output(shape: tuple[int, int], anchors: dict[int, tuple[int, int]], turns_by_color: dict[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, (r, c) in anchors.items():
        patch = np.rot90(BASE_STAMP, -int(turns_by_color[color]))
        _paste(out, patch, int(r) - 1, int(c) - 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each directional sprite as a rotation instruction and stamp the rotated fixed pattern onto its same-color anchor.

    Input structure:
    - The grid is black except for 2-4 multicolor directional sprites and the same number of isolated anchor pixels.
    - Each sprite contains 5-7 grey body pixels plus one unique colored marker pixel, and all sprites share the same body template up to rotation.
    - Each anchor is a singleton pixel whose color matches exactly one sprite marker color.

    Rule:
    - For each sprite, compute the centroid offset of its grey body relative to its colored marker; that offset is the tail direction, and the opposite cardinal direction is the sprite's pointing direction.
    - Map pointing direction to a clockwise rotation of the fixed 3x3 stamp: up->0, right->90, down->180, left->270.
    - For each anchor, find the sprite with the same color, rotate the stamp accordingly, and paste it with the anchor at the stamp center.
    - Erase every sprite and the original anchor pixels; only the stamped patterns remain on the black background.

    Key insight:
    - The sprite's asymmetric grey mass does not indicate where to paint directly; it only encodes an orientation, which must then be transferred by same-color matching to a separate anchor location.
    """
    g = np.asarray(grid, dtype=int)
    sprites, anchors = _parse_scene(g)
    turns_by_color = {int(sprite["color"]): int(sprite["turns"]) for sprite in sprites}
    return _scene_output(g.shape, anchors, turns_by_color)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _has_matching_counterexample(
    anchors: dict[int, tuple[int, int]],
    sprite_markers: dict[int, tuple[int, int]],
) -> bool:
    for color, anchor in anchors.items():
        own = _manhattan(anchor, sprite_markers[color])
        if any(
            other_color != color and _manhattan(anchor, marker) < own
            for other_color, marker in sprite_markers.items()
        ):
            return True
    return False


def _place_anchors(H: int, W: int, count: int) -> list[tuple[int, int]] | None:
    blocked = np.zeros((H, W), dtype=bool)
    anchors: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(1, H - 1)
            for c in range(1, W - 1)
            if not blocked[r, c]
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        anchors.append((r, c))
        mask = np.zeros((H, W), dtype=bool)
        mask[r, c] = True
        blocked |= _dilate_chebyshev(mask, 3)
    return anchors


def _place_sprites(
    grid: np.ndarray,
    colors: list[int],
    directions: list[str],
    anchors: dict[int, tuple[int, int]],
) -> dict[int, tuple[int, int]] | None:
    H, W = grid.shape
    anchor_mask = np.zeros((H, W), dtype=bool)
    for r, c in anchors.values():
        anchor_mask[r, c] = True
    anchor_blocked = _dilate_chebyshev(anchor_mask, 3)

    sprite_mask = np.zeros((H, W), dtype=bool)
    sprite_markers: dict[int, tuple[int, int]] = {}
    items = list(zip(colors, directions))
    random.shuffle(items)
    for color, direction in items:
        rows, cols = _marker_ranges(H, W, direction)
        blocked = anchor_blocked | _dilate_chebyshev(sprite_mask, 1)
        candidates: list[tuple[int, int]] = []
        for r in rows:
            for c in cols:
                cells = _sprite_cells(direction, r, c)
                if any(blocked[rr, cc] for rr, cc in cells):
                    continue
                if grid[r, c] != BG:
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        mr, mc = random.choice(candidates)
        for rr, cc in _sprite_cells(direction, mr, mc):
            grid[rr, cc] = BODY
            sprite_mask[rr, cc] = True
        grid[mr, mc] = int(color)
        sprite_mask[mr, mc] = True
        sprite_markers[int(color)] = (int(mr), int(mc))
    return sprite_markers


def _validate_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    anchors: dict[int, tuple[int, int]],
    turns_by_color: dict[int, int],
    sprite_markers: dict[int, tuple[int, int]],
) -> bool:
    try:
        parsed_sprites, parsed_anchors = _parse_scene(grid)
    except AssertionError:
        return False

    parsed_turns = {int(sprite["color"]): int(sprite["turns"]) for sprite in parsed_sprites}
    parsed_markers = {int(sprite["color"]): tuple(map(int, sprite["marker"])) for sprite in parsed_sprites}

    if parsed_anchors != anchors:
        return False
    if parsed_turns != turns_by_color:
        return False
    if parsed_markers != sprite_markers:
        return False
    if len(set(parsed_turns.values())) != len(parsed_turns):
        return False
    if not _has_matching_counterexample(anchors, sprite_markers):
        return False

    recomputed = _scene_output(grid.shape, parsed_anchors, parsed_turns)
    if not np.array_equal(recomputed, expected):
        return False
    if np.count_nonzero(expected) != 3 * len(anchors):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 26)
    W = random.randint(18, 26)
    count = random.choices([2, 3, 4], weights=[2, 3, 5], k=1)[0]
    colors = random.sample(list(MARKER_COLORS), count)
    directions = random.sample(list(DIR_TO_TURNS), count)

    anchor_positions = _place_anchors(H, W, count)
    if anchor_positions is None:
        return None
    anchors = {color: pos for color, pos in zip(colors, anchor_positions)}

    grid = np.zeros((H, W), dtype=int)
    for color, (r, c) in anchors.items():
        grid[r, c] = int(color)

    sprite_markers = _place_sprites(grid, colors, directions, anchors)
    if sprite_markers is None:
        return None

    turns_by_color = {int(color): DIR_TO_TURNS[direction] for color, direction in zip(colors, directions)}
    expected = _scene_output(grid.shape, anchors, turns_by_color)
    if not _validate_scene(grid, expected, anchors, turns_by_color, sprite_markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional-stamp input")
