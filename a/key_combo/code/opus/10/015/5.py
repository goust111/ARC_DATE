from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
MARKER_COLORS = (2, 3, 4, 6)
SELECTOR_COLOR = 9

SELECTOR_SIZE = 7
SPRITE_SIZE = 3
INNER_SIZE = 7
OUTPUT_SIZE = 9

MARKER_POSITIONS = ((1, 1), (1, 5), (5, 1), (5, 5))
QUADRANT_ANCHORS = {
    (1, 1): (0, 0),
    (1, 5): (0, 4),
    (5, 1): (4, 0),
    (5, 5): (4, 4),
}
QUADRANT_CW_TURNS = {
    (1, 1): 0,
    (1, 5): 1,
    (5, 5): 2,
    (5, 1): 3,
}

INPUT_ROWS_RANGE = (16, 19)
INPUT_COLS_RANGE = (16, 19)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
SPRITE_AREA_RANGE = (4, 7)


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _rotate_cw(tile: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(tile, -int(quarter_turns) % 4)


def _selector_marker_map(selector: np.ndarray) -> dict[tuple[int, int], int]:
    markers = {(r, c): int(selector[r, c]) for r, c in MARKER_POSITIONS}
    if set(markers.values()) != set(MARKER_COLORS):
        raise AssertionError("selector markers must use exactly the four designated colors")

    for r in range(SELECTOR_SIZE):
        for c in range(SELECTOR_SIZE):
            if (r, c) in markers:
                continue
            if int(selector[r, c]) != SELECTOR_COLOR:
                raise AssertionError("selector body must be uniform outside marker positions")
    return markers


def _find_selector(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[tuple[int, int], int]]:
    candidates: list[tuple[tuple[int, int, int, int], dict[tuple[int, int], int]]] = []
    for coords in _components(grid == SELECTOR_COLOR, connectivity=4):
        if len(coords) != 45:
            continue
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1, c1 - c0 + 1) != (SELECTOR_SIZE, SELECTOR_SIZE):
            continue
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block == BG):
            continue
        try:
            markers = _selector_marker_map(block)
        except AssertionError:
            continue
        candidates.append((box, markers))

    if len(candidates) != 1:
        raise AssertionError("expected exactly one valid 7x7 selector block")
    return candidates[0]


def _outside_mask(shape: tuple[int, int], selector_bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    r0, c0, r1, c1 = selector_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _extract_sprite(grid: np.ndarray, color: int, selector_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, int]:
    mask = (grid == int(color)) & _outside_mask(grid.shape, selector_bbox)
    comps = _components(mask, connectivity=8)
    if len(comps) != 1:
        raise AssertionError("each marker color must correspond to exactly one external sprite")
    coords = comps[0]
    area = int(len(coords))
    if area < SPRITE_AREA_RANGE[0] or area > SPRITE_AREA_RANGE[1]:
        raise AssertionError("sprite area must stay within the described bounds")
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (SPRITE_SIZE, SPRITE_SIZE):
        raise AssertionError("each external sprite must have a 3x3 bbox")
    sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), BG, dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return sprite, area


def _render_with_turns(
    markers: dict[tuple[int, int], int],
    sprites: dict[int, np.ndarray],
    turns: dict[tuple[int, int], int],
) -> np.ndarray:
    inner = np.full((INNER_SIZE, INNER_SIZE), BG, dtype=int)
    for pos in MARKER_POSITIONS:
        color = markers[pos]
        rotated = _rotate_cw(sprites[color], turns[pos])
        anchor_r, anchor_c = QUADRANT_ANCHORS[pos]
        _paste(inner, rotated, anchor_r, anchor_c, transparent=BG)

    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), FRAME, dtype=int)
    out[1:-1, 1:-1] = inner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each external sprite according to the selector quadrant that names its destination, then frame the four-way summary.

    Input structure:
    - The grid has a black background with one 7x7 maroon selector block.
    - The selector block has four marker pixels at the centers of its 3x3 quadrants, using exactly red, green, yellow, and magenta.
    - Outside the selector there is exactly one same-colored 8-connected sprite for each marker color, and each sprite has a 3x3 bbox.

    Rule:
    - Read the four marker positions in the selector block.
    - Match each marker color to the unique external sprite of the same color.
    - Place the matched sprites into a black 7x7 quadrant canvas, rotating by quadrant: upper-left 0 degrees, upper-right 90 degrees clockwise, lower-right 180 degrees, lower-left 270 degrees clockwise.
    - Add a one-cell blue frame around that 7x7 canvas to produce the 9x9 output.

    Key insight:
    - The selector quadrant does two jobs at once: it chooses the output quadrant and it encodes the rotation angle for the same-colored sprite.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    if int(np.max(g.shape)) > 30:
        raise AssertionError("ARC grids must stay within 30x30")
    if BG not in set(map(int, np.unique(g))):
        raise AssertionError("expected black background to be present")

    selector_bbox, markers = _find_selector(g)
    sprites: dict[int, np.ndarray] = {}
    outside_non_bg = int(np.count_nonzero(_outside_mask(g.shape, selector_bbox) & (g != BG)))
    total_area = 0
    for color in MARKER_COLORS:
        sprite, area = _extract_sprite(g, color, selector_bbox)
        sprites[int(color)] = sprite
        total_area += area
    if total_area != outside_non_bg:
        raise AssertionError("all non-background cells outside the selector must belong to the four sprites")

    return _render_with_turns(markers, sprites, QUADRANT_CW_TURNS)


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _mask_rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k).astype(bool) for k in range(4)]


def _build_sprite_classes() -> list[list[np.ndarray]]:
    classes: dict[bytes, list[np.ndarray]] = {}
    for bits in range(1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.array(
            [(bits >> k) & 1 for k in range(SPRITE_SIZE * SPRITE_SIZE)],
            dtype=bool,
        ).reshape(SPRITE_SIZE, SPRITE_SIZE)
        area = int(mask.sum())
        if area < SPRITE_AREA_RANGE[0] or area > SPRITE_AREA_RANGE[1]:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        comps = _components(mask, connectivity=8)
        if len(comps) != 1:
            continue
        rotations = _mask_rotations(mask)
        if len({_mask_key(rot) for rot in rotations}) != 4:
            continue
        canon = min(_mask_key(rot) for rot in rotations)
        classes.setdefault(canon, [])
        as_key = _mask_key(mask)
        if as_key not in {_mask_key(existing) for existing in classes[canon]}:
            classes[canon].append(mask.copy())
    return list(classes.values())


SPRITE_CLASSES = _build_sprite_classes()


def _sample_sprite_masks() -> list[np.ndarray]:
    if len(SPRITE_CLASSES) < 4:
        raise RuntimeError("not enough distinct sprite classes")
    chosen_classes = random.sample(SPRITE_CLASSES, 4)
    return [random.choice(sprite_class).copy() for sprite_class in chosen_classes]


def _sprite_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    g = int(gap)
    return ar1 + g < br0 or br1 + g < ar0 or ac1 + g < bc0 or bc1 + g < ac0


def _selector_bbox(rows: int, cols: int) -> tuple[int, int, int, int]:
    r0 = random.choice((0, rows - SELECTOR_SIZE))
    c0 = random.choice((0, cols - SELECTOR_SIZE))
    return r0, c0, r0 + SELECTOR_SIZE - 1, c0 + SELECTOR_SIZE - 1


def _sprite_bbox(top: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = top
    return r, c, r + SPRITE_SIZE - 1, c + SPRITE_SIZE - 1


def _candidate_sprite_positions(rows: int, cols: int, selector_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(rows - SPRITE_SIZE + 1):
        for c in range(cols - SPRITE_SIZE + 1):
            box = _sprite_bbox((r, c))
            if _bbox_gap_ok(box, selector_bbox):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _choose_positions(rows: int, cols: int, selector_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    candidates = _candidate_sprite_positions(rows, cols, selector_bbox)

    def search(
        start: int,
        chosen: list[tuple[int, int]],
        chosen_boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int]] | None:
        if len(chosen) == 4:
            return chosen.copy()
        remaining_need = 4 - len(chosen)
        if len(candidates) - start < remaining_need:
            return None

        for idx in range(start, len(candidates)):
            pos = candidates[idx]
            box = _sprite_bbox(pos)
            if not all(_bbox_gap_ok(box, other) for other in chosen_boxes):
                continue
            chosen.append(pos)
            chosen_boxes.append(box)
            found = search(idx + 1, chosen, chosen_boxes)
            if found is not None:
                return found
            chosen.pop()
            chosen_boxes.pop()
        return None

    return search(0, [], [])


def _selector_from_markers(markers: dict[tuple[int, int], int]) -> np.ndarray:
    selector = np.full((SELECTOR_SIZE, SELECTOR_SIZE), SELECTOR_COLOR, dtype=int)
    for pos, color in markers.items():
        selector[pos] = int(color)
    return selector


def _wrong_turn_maps() -> list[dict[tuple[int, int], int]]:
    return [
        {pos: 0 for pos in MARKER_POSITIONS},
        {(1, 1): 0, (1, 5): 3, (5, 5): 2, (5, 1): 1},
        {(1, 1): 0, (1, 5): 2, (5, 5): 1, (5, 1): 3},
        {(1, 1): 0, (1, 5): 1, (5, 5): 3, (5, 1): 2},
    ]


ALT_TURN_MAPS = _wrong_turn_maps()


def _valid_candidate(
    grid: np.ndarray,
    markers: dict[tuple[int, int], int],
    sprites: dict[int, np.ndarray],
) -> bool:
    expected = _render_with_turns(markers, sprites, QUADRANT_CW_TURNS)
    if expected.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
        return False
    if any(np.array_equal(expected, _render_with_turns(markers, sprites, turns)) for turns in ALT_TURN_MAPS):
        return False
    return np.array_equal(main(grid), expected)


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(*INPUT_ROWS_RANGE)
    cols = random.randint(*INPUT_COLS_RANGE)
    selector_bbox = _selector_bbox(rows, cols)
    sprite_positions = _choose_positions(rows, cols, selector_bbox)
    if sprite_positions is None:
        return None

    marker_assignment = random.sample(MARKER_COLORS, len(MARKER_COLORS))
    markers = {pos: color for pos, color in zip(MARKER_POSITIONS, marker_assignment)}

    masks = _sample_sprite_masks()
    sprites = {
        color: _sprite_from_mask(mask, color)
        for color, mask in zip(MARKER_COLORS, masks)
    }

    selector = _selector_from_markers(markers)
    grid = np.full((rows, cols), BG, dtype=int)
    sr0, sc0, _sr1, _sc1 = selector_bbox
    _paste(grid, selector, sr0, sc0)

    sprite_items = list(sprites.items())
    random.shuffle(sprite_items)
    for (color, sprite), (r0, c0) in zip(sprite_items, sprite_positions):
        _paste(grid, sprite, r0, c0, transparent=BG)

    if not _valid_candidate(grid, markers, sprites):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid quadrant-rotation selector input")
