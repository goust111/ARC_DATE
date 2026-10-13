import random
from collections import deque

import numpy as np


BG = 0
CROSS_COLOR = 5
FRAME_COLOR = 9
SPRITE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
QUADRANTS = ("ul", "ur", "br", "bl")
ROTATE_CW_BY_QUADRANT = {"ul": 0, "ur": 1, "br": 2, "bl": 3}
CORNER_KEYS = ("ul", "ur", "br", "bl")
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append({"coords": coords, "bbox": _bbox(coords)})
    return comps


def _crop_component(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _is_target_frame(mask: np.ndarray, cross_r: int, cross_c: int) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if np.array_equal(mask, expected):
        return True
    gapped = expected.copy()
    gapped[cross_r, 0] = False
    gapped[cross_r, -1] = False
    gapped[0, cross_c] = False
    gapped[-1, cross_c] = False
    return bool(np.array_equal(mask, gapped))


def _quadrant_for_bbox(bbox: tuple[int, int, int, int], cross_row: int, cross_col: int) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    if center_r < cross_row and center_c < cross_col:
        return "ul"
    if center_r < cross_row and center_c > cross_col:
        return "ur"
    if center_r > cross_row and center_c > cross_col:
        return "br"
    if center_r > cross_row and center_c < cross_col:
        return "bl"
    raise AssertionError("sprite must lie strictly inside one quadrant")


def _hole_corner(patch: np.ndarray, bg: int) -> str:
    h, w = patch.shape
    holes = np.argwhere(patch == bg)
    assert len(holes) == 1, "each sprite must have exactly one missing pixel"
    hole = tuple(int(x) for x in holes[0])
    mapping = {
        (0, 0): "ul",
        (0, w - 1): "ur",
        (h - 1, w - 1): "br",
        (h - 1, 0): "bl",
    }
    assert hole in mapping, "missing pixel must sit at a bbox corner after rotation normalization"
    return mapping[hole]


def _slot_boxes(
    frame_bbox: tuple[int, int, int, int], cross_row: int, cross_col: int
) -> dict[str, tuple[int, int, int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    boxes = {
        "ul": (fr0 + 1, fc0 + 1, cross_row - 1, cross_col - 1),
        "ur": (fr0 + 1, cross_col + 1, cross_row - 1, fc1 - 1),
        "br": (cross_row + 1, cross_col + 1, fr1 - 1, fc1 - 1),
        "bl": (cross_row + 1, fc0 + 1, fr1 - 1, cross_col - 1),
    }
    for r0, c0, r1, c1 in boxes.values():
        assert r0 <= r1 and c0 <= c1, "frame must contain four non-empty interior quadrants"
    return boxes


def _make_sprite(height: int, width: int, color: int, hole_corner: str) -> np.ndarray:
    patch = np.full((height, width), int(color), dtype=int)
    if hole_corner == "ul":
        patch[0, 0] = BG
    elif hole_corner == "ur":
        patch[0, -1] = BG
    elif hole_corner == "br":
        patch[-1, -1] = BG
    elif hole_corner == "bl":
        patch[-1, 0] = BG
    else:
        raise ValueError(hole_corner)
    return patch


def _draw_structure(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], cross_row: int, cross_col: int) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[cross_row, :] = CROSS_COLOR
    grid[:, cross_col] = CROSS_COLOR


def _slot_origins(
    frame_bbox: tuple[int, int, int, int], cross_row: int, cross_col: int
) -> dict[str, tuple[int, int]]:
    fr0, fc0, _, _ = frame_bbox
    return {
        "ul": (fr0 + 1, fc0 + 1),
        "ur": (fr0 + 1, cross_col + 1),
        "br": (cross_row + 1, cross_col + 1),
        "bl": (cross_row + 1, fc0 + 1),
    }


def _sample_offset(space: int, size: int) -> int:
    slack = space - size
    assert slack >= 0
    if slack == 0:
        return 0
    if slack == 1:
        return random.randint(0, 1)
    return random.randint(1, slack - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant sprite to a common facing, then move it into the matching quadrant of the central target frame.

    Input structure:
    - The background is black.
    - One complete grey horizontal line and one complete grey vertical line divide the grid into four quadrants.
    - A small hollow rectangular target frame of another color surrounds the central crossing while leaving the grey cross visible inside it.
    - Each quadrant contains one monochrome sprite of a unique color; all sprites share the same rectangular template and each is missing exactly one corner pixel.

    Rule:
    - Determine a sprite's rotation correction from its quadrant: upper-left keeps its orientation, upper-right rotates clockwise 90 degrees, lower-right rotates 180 degrees, and lower-left rotates clockwise 270 degrees.
    - After this normalization, read the missing corner of the sprite: missing upper-left goes to the frame's upper-left slot, missing upper-right to the upper-right slot, missing lower-right to the lower-right slot, and missing lower-left to the lower-left slot.
    - Clear the four original sprite locations and paste the corrected sprites into the four frame slots. Keep the grey cross and the target-frame outline unchanged.

    Key insight:
    - The puzzle uses two coupled cues: quadrant position tells you how to undo the sprite rotation, and only after that can the missing corner be read as the destination slot inside the center frame.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    cross_candidates = []
    for color in colors:
        row_hits = [r for r in range(h) if int(np.count_nonzero(g[r, :] == color)) >= w - 2]
        col_hits = [c for c in range(w) if int(np.count_nonzero(g[:, c] == color)) >= h - 2]
        if len(row_hits) == 1 and len(col_hits) == 1:
            cross_candidates.append((int(color), int(row_hits[0]), int(col_hits[0])))
    assert len(cross_candidates) == 1, "expected a unique full-row/full-column cross"
    cross_color, cross_row, cross_col = cross_candidates[0]

    frame_match = None
    for color in colors:
        if color == cross_color:
            continue
        mask = g == color
        if not np.any(mask):
            continue
        coords = [tuple(int(x) for x in rc) for rc in np.argwhere(mask)]
        r0, c0, r1, c1 = _bbox(coords)
        if not (r0 < cross_row < r1 and c0 < cross_col < c1):
            continue
        submask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if not _is_target_frame(submask, cross_row - r0, cross_col - c0):
            continue
        inside_colors = set(int(x) for x in np.unique(g[r0 + 1 : r1, c0 + 1 : c1]))
        if not inside_colors.issubset({bg, cross_color}):
            continue
        assert frame_match is None, "expected a unique target frame"
        frame_match = (int(color), (r0, c0, r1, c1))
    assert frame_match is not None, "target frame not found"
    frame_color, frame_bbox = frame_match

    slots = _slot_boxes(frame_bbox, cross_row, cross_col)
    slot_shapes = {
        key: (r1 - r0 + 1, c1 - c0 + 1)
        for key, (r0, c0, r1, c1) in slots.items()
    }
    assert len(set(slot_shapes.values())) == 1, "all four frame slots must share a shape"

    sprite_mask = (g != bg) & (g != cross_color) & (g != frame_color)
    sprites = _components(sprite_mask)
    assert len(sprites) == 4, "expected one sprite per quadrant"

    out = g.copy()
    placed = {}
    seen_quadrants = set()
    for comp in sprites:
        quadrant = _quadrant_for_bbox(comp["bbox"], cross_row, cross_col)
        assert quadrant not in seen_quadrants, "quadrant assignment must be unique"
        seen_quadrants.add(quadrant)

        patch = _crop_component(g, comp, bg)
        corrected = np.rot90(patch, -ROTATE_CW_BY_QUADRANT[quadrant])
        hole_corner = _hole_corner(corrected, bg)
        assert hole_corner not in placed, "missing-corner destinations must be unique"

        slot_r0, slot_c0, slot_r1, slot_c1 = slots[hole_corner]
        assert corrected.shape == (slot_r1 - slot_r0 + 1, slot_c1 - slot_c0 + 1)
        assert np.all(out[slot_r0 : slot_r1 + 1, slot_c0 : slot_c1 + 1] == bg)
        placed[hole_corner] = (corrected, slot_r0, slot_c0)

        for r, c in comp["coords"]:
            out[r, c] = bg

    assert seen_quadrants == set(QUADRANTS)
    assert set(placed) == set(CORNER_KEYS)
    for patch, r0, c0 in placed.values():
        ph, pw = patch.shape
        out[r0 : r0 + ph, c0 : c0 + pw] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    sprite_h = random.randint(3, 5)
    sprite_w = random.randint(3, 5)
    max_dim = max(sprite_h, sprite_w)

    top = max_dim + random.randint(0, 2)
    bottom = max_dim + random.randint(0, 2)
    left = max_dim + random.randint(0, 2)
    right = max_dim + random.randint(0, 2)

    frame_h = 2 * sprite_h + 3
    frame_w = 2 * sprite_w + 3
    height = top + frame_h + bottom
    width = left + frame_w + right
    if max(height, width) > 30:
        return None

    fr0, fc0 = top, left
    fr1, fc1 = fr0 + frame_h - 1, fc0 + frame_w - 1
    cross_row = fr0 + 1 + sprite_h
    cross_col = fc0 + 1 + sprite_w
    frame_bbox = (fr0, fc0, fr1, fc1)

    colors = random.sample(SPRITE_COLOR_POOL, 4)
    hole_by_quadrant = dict(zip(QUADRANTS, random.sample(list(CORNER_KEYS), 4)))

    structure = np.full((height, width), BG, dtype=int)
    _draw_structure(structure, frame_bbox, cross_row, cross_col)

    expected = structure.copy()
    for quadrant, color in zip(QUADRANTS, colors):
        hole_corner = hole_by_quadrant[quadrant]
        corrected = _make_sprite(sprite_h, sprite_w, color, hole_corner)
        slot_r0, slot_c0 = _slot_origins(frame_bbox, cross_row, cross_col)[hole_corner]
        expected[slot_r0 : slot_r0 + sprite_h, slot_c0 : slot_c0 + sprite_w] = corrected

    inp = structure.copy()
    regions = {
        "ul": (0, 0, top, left),
        "ur": (0, fc1 + 1, top, right),
        "br": (fr1 + 1, fc1 + 1, bottom, right),
        "bl": (fr1 + 1, 0, bottom, left),
    }
    for quadrant, color in zip(QUADRANTS, colors):
        corrected = _make_sprite(sprite_h, sprite_w, color, hole_by_quadrant[quadrant])
        rotated = np.rot90(corrected, ROTATE_CW_BY_QUADRANT[quadrant])
        ph, pw = rotated.shape
        base_r, base_c, space_h, space_w = regions[quadrant]
        if ph > space_h or pw > space_w:
            return None
        off_r = _sample_offset(space_h, ph)
        off_c = _sample_offset(space_w, pw)
        rr = base_r + off_r
        cc = base_c + off_c
        inp[rr : rr + ph, cc : cc + pw] = rotated

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid ARC instance")
