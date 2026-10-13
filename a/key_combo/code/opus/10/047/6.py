from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SPRITE_COLORS = (BLUE, RED, GREEN, YELLOW)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
QUADRANTS = ("ul", "ur", "ll", "lr")
QUADRANT_OFFSETS = {
    "ul": (0, 0),
    "ur": (0, 2),
    "ll": (2, 0),
    "lr": (2, 2),
}


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)


_NO_HOLE_BASES = (
    _mask(["###", "#..", "#.."]),
    _mask(["###", ".##", ".#."]),
    _mask(["#..", "###", "##."]),
    _mask([".##", "###", "##."]),
    _mask([".##", "##.", "#.."]),
    _mask(["##..", ".###", "..#."]),
    _mask(["###", "###", "##."]),
)
_HOLE_TEMPLATE = _mask(["###", "#.#", "###"])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            transformed = np.rot90(base, k)
            cropped = _crop_mask(transformed)
            key = tuple(tuple(int(v) for v in row) for row in cropped)
            variants.setdefault(key, cropped.astype(int))
    return list(variants.values())


NO_HOLE_LIBRARY = [variant for base in _NO_HOLE_BASES for variant in _all_transforms(base)]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _find_anchor(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == GREY
    anchors = []
    for coords in _components_of_mask(mask):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if coords.shape[0] != 16 or (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            anchors.append((r0, c0, r1, c1))
    assert len(anchors) == 1, "expected exactly one solid 4x4 grey anchor"
    return anchors[0]


def _extract_sprite(grid: np.ndarray, color: int, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    local = coords - np.array([[r0, c0]], dtype=int)
    crop[local[:, 0], local[:, 1]] = int(color)
    return {
        "color": int(color),
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "crop": crop,
    }


def _find_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BLACK, GREY)):
        for coords in _components_of_mask(grid == color):
            sprite = _extract_sprite(grid, color, coords)
            r0, c0, r1, c1 = sprite["bbox"]
            assert 4 <= coords.shape[0] <= 8, "sprite size must be 4-8 pixels"
            assert r1 - r0 + 1 <= 4 and c1 - c0 + 1 <= 4, "sprite bbox must fit within 4x4"
            sprites.append(sprite)
    assert len(sprites) == 4, "expected four colored sprite components"
    return sprites


def _quadrant_of_sprite(sprite: dict, anchor_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = anchor_bbox
    anchor_center_r = (r0 + r1) / 2.0
    anchor_center_c = (c0 + c1) / 2.0
    coords = sprite["coords"]
    center_r = float(coords[:, 0].mean())
    center_c = float(coords[:, 1].mean())
    vertical = "u" if center_r < anchor_center_r else "l"
    horizontal = "l" if center_c < anchor_center_c else "r"
    return vertical + horizontal


def _has_hole(crop: np.ndarray, color: int) -> bool:
    h, w = crop.shape
    bg_mask = crop != color
    seen = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(bg_mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        touches_boundary = False
        while queue:
            rr, cc = queue.popleft()
            if rr in (0, h - 1) or cc in (0, w - 1):
                touches_boundary = True
            for nr, nc in _neighbors4(rr, cc, h, w):
                if bg_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        if not touches_boundary:
            return True
    return False


def _shrink_sprite(crop: np.ndarray, color: int) -> np.ndarray:
    h, w = crop.shape
    out = np.zeros((2, 2), dtype=int)
    for tr in range(2):
        rs = (tr * h) // 2
        re = ((tr + 1) * h) // 2
        for tc in range(2):
            cs = (tc * w) // 2
            ce = ((tc + 1) * w) // 2
            region = crop[rs:re, cs:ce]
            if region.size == 0:
                continue
            colored = int(np.count_nonzero(region == color))
            background = int(region.size - colored)
            if colored >= background:
                out[tr, tc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each quadrant sprite into the central anchor, using hole topology to choose between a 2x2 miniature and a single-pixel marker.

    Input structure:
    - The grid is mostly black and contains one solid 4x4 grey anchor block.
    - Four differently colored 4-connected sprites lie in the four diagonal quadrants around that anchor.
    - Each sprite fits inside a 4x4 bounding box, and some sprites contain an enclosed black hole.

    Rule:
    - Find the unique grey 4x4 anchor and assign each sprite to the matching anchor quadrant by comparing sprite center vs. anchor center.
    - If a sprite has an enclosed 4-connected black hole inside its bbox, place only one pixel of its color at the top-left cell of that quadrant's 2x2 anchor subregion.
    - Otherwise, clear the sprite from the canvas, shrink its bbox content to 2x2 by quadrant-majority, and write that 2x2 miniature into the matching anchor subregion.
    - Everything outside the modified anchor becomes black.

    Key insight:
    - Topology decides what representation to draw, while spatial quadrant decides where to draw it; both signals are required and independent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    anchor_bbox = _find_anchor(g)
    sprites = _find_sprites(g)

    sprite_quadrants = {_quadrant_of_sprite(sprite, anchor_bbox): sprite for sprite in sprites}
    assert set(sprite_quadrants) == set(QUADRANTS), "expected exactly one sprite per quadrant"

    out = np.zeros_like(g)
    ar0, ac0, _, _ = anchor_bbox
    out[ar0 : ar0 + 4, ac0 : ac0 + 4] = GREY

    hole_count = 0
    for quadrant, sprite in sprite_quadrants.items():
        qr, qc = QUADRANT_OFFSETS[quadrant]
        rr = ar0 + qr
        cc = ac0 + qc
        color = int(sprite["color"])
        crop = sprite["crop"]
        if _has_hole(crop, color):
            hole_count += 1
            out[rr, cc] = color
        else:
            out[rr : rr + 2, cc : cc + 2] = _shrink_sprite(crop, color)

    assert hole_count == 2, "expected exactly two holed sprites"
    return out


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask.astype(bool)] = int(color)


def _quadrant_ranges(
    grid_shape: tuple[int, int],
    anchor_top: int,
    anchor_left: int,
    sprite_shape: tuple[int, int],
    quadrant: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid_shape
    sh, sw = sprite_shape
    if quadrant == "ul":
        rows = (1, anchor_top - sh)
        cols = (1, anchor_left - sw)
    elif quadrant == "ur":
        rows = (1, anchor_top - sh)
        cols = (anchor_left + 5, w - sw - 1)
    elif quadrant == "ll":
        rows = (anchor_top + 5, h - sh - 1)
        cols = (1, anchor_left - sw)
    else:
        rows = (anchor_top + 5, h - sh - 1)
        cols = (anchor_left + 5, w - sw - 1)
    return rows, cols


def _expected_output_from_metadata(
    shape: tuple[int, int],
    anchor_top: int,
    anchor_left: int,
    placements: list[dict],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[anchor_top : anchor_top + 4, anchor_left : anchor_left + 4] = GREY
    for item in placements:
        qr, qc = QUADRANT_OFFSETS[item["quadrant"]]
        rr = anchor_top + qr
        cc = anchor_left + qc
        color = int(item["color"])
        if item["has_hole"]:
            out[rr, cc] = color
        else:
            crop = np.where(item["mask"], color, BLACK)
            out[rr : rr + 2, cc : cc + 2] = _shrink_sprite(crop, color)
    return out


def _validate_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    placements: list[dict],
    anchor_top: int,
    anchor_left: int,
) -> bool:
    try:
        parsed_anchor = _find_anchor(grid)
        if parsed_anchor != (anchor_top, anchor_left, anchor_top + 3, anchor_left + 3):
            return False
        sprites = _find_sprites(grid)
    except AssertionError:
        return False

    by_quadrant = {}
    hole_count = 0
    for sprite in sprites:
        quadrant = _quadrant_of_sprite(sprite, parsed_anchor)
        if quadrant in by_quadrant:
            return False
        has_hole = _has_hole(sprite["crop"], int(sprite["color"]))
        by_quadrant[quadrant] = (int(sprite["color"]), has_hole, _shrink_sprite(sprite["crop"], int(sprite["color"])))
        hole_count += int(has_hole)
    if set(by_quadrant) != set(QUADRANTS) or hole_count != 2:
        return False

    metadata = {
        item["quadrant"]: (
            int(item["color"]),
            bool(item["has_hole"]),
            _shrink_sprite(np.where(item["mask"], int(item["color"]), BLACK), int(item["color"])),
        )
        for item in placements
    }
    if metadata.keys() != by_quadrant.keys():
        return False
    for quadrant in metadata:
        m_color, m_hole, m_shrink = metadata[quadrant]
        p_color, p_hole, p_shrink = by_quadrant[quadrant]
        if (m_color, m_hole) != (p_color, p_hole):
            return False
        if not m_hole and not np.array_equal(m_shrink, p_shrink):
            return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _sample_no_hole_mask() -> np.ndarray:
    return random.choice(NO_HOLE_LIBRARY).copy()


def _sample_hole_mask() -> np.ndarray:
    return _HOLE_TEMPLATE.copy()


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    anchor_top = random.randint(5, h - 9)
    anchor_left = random.randint(5, w - 9)

    grid = np.zeros((h, w), dtype=int)
    grid[anchor_top : anchor_top + 4, anchor_left : anchor_left + 4] = GREY

    hole_quadrants = set(random.sample(QUADRANTS, 2))
    colors = list(SPRITE_COLORS)
    random.shuffle(colors)

    placements: list[dict] = []
    for quadrant, color in zip(QUADRANTS, colors):
        has_hole = quadrant in hole_quadrants
        mask = _sample_hole_mask() if has_hole else _sample_no_hole_mask()
        (r_lo, r_hi), (c_lo, c_hi) = _quadrant_ranges(grid.shape, anchor_top, anchor_left, mask.shape, quadrant)
        if r_lo > r_hi or c_lo > c_hi:
            return None
        top = random.randint(r_lo, r_hi)
        left = random.randint(c_lo, c_hi)
        _paint_mask(grid, top, left, mask, color)
        placements.append(
            {
                "quadrant": quadrant,
                "color": int(color),
                "has_hole": bool(has_hole),
                "mask": mask.astype(bool),
                "top": int(top),
                "left": int(left),
            }
        )

    expected = _expected_output_from_metadata(grid.shape, anchor_top, anchor_left, placements)
    if not _validate_scene(grid, expected, placements, anchor_top, anchor_left):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
