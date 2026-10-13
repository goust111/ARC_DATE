from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
MARKER = 7
STAMP_PALETTE = [2, 3, 4, 5, 6, 8, 9]

GEN_ATTEMPTS = 400
GRID_MIN = 14
GRID_MAX = 30
RECT_COUNT_RANGE = (1, 4)
INTERIOR_MIN = 3
INTERIOR_MAX = 8
PLACEMENT_PAD = 1
PLACEMENT_TRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.array_equal(sub == FRAME, border):
            frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one rectangular frame"
    frames.sort()
    return frames


def _marker_belongs(comp: np.ndarray, frame: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = frame
    rows = np.unique(comp[:, 0])
    cols = np.unique(comp[:, 1])
    if len(rows) == 1:
        row = int(rows[0])
        if row not in (r0 - 1, r1 + 1):
            return False
        cs = np.sort(comp[:, 1])
        if not np.array_equal(cs, np.arange(int(cs[0]), int(cs[-1]) + 1)):
            return False
        return c0 < int(cs[0]) and int(cs[-1]) < c1
    if len(cols) == 1:
        col = int(cols[0])
        if col not in (c0 - 1, c1 + 1):
            return False
        rs = np.sort(comp[:, 0])
        if not np.array_equal(rs, np.arange(int(rs[0]), int(rs[-1]) + 1)):
            return False
        return r0 < int(rs[0]) and int(rs[-1]) < r1
    return False


def _count_markers(grid: np.ndarray, frame: tuple[int, int, int, int], orange_parts: list[np.ndarray]) -> int:
    matches = [comp for comp in orange_parts if _marker_belongs(comp, frame)]
    assert matches, "each frame must have at least one marker segment"
    return len(matches)


def _corner_name(mask_bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = mask_bbox
    h, w = shape
    touch_top = r0 == 0
    touch_bottom = r1 == h - 1
    touch_left = c0 == 0
    touch_right = c1 == w - 1
    assert touch_top or touch_bottom
    assert touch_left or touch_right
    vertical = "t" if touch_top else "b"
    horizontal = "l" if touch_left else "r"
    return vertical + horizontal


def _flip_to_top_left(arr: np.ndarray, corner: str) -> np.ndarray:
    out = arr
    if corner[0] == "b":
        out = out[::-1, :]
    if corner[1] == "r":
        out = out[:, ::-1]
    return out


def _flip_from_top_left(arr: np.ndarray, corner: str) -> np.ndarray:
    out = arr
    if corner[1] == "r":
        out = out[:, ::-1]
    if corner[0] == "b":
        out = out[::-1, :]
    return out


def _extract_stamp(oriented: np.ndarray) -> np.ndarray:
    stamp_mask = (oriented != BG) & (oriented != FRAME) & (oriented != MARKER)
    coords = np.argwhere(stamp_mask)
    assert coords.size > 0, "frame interior must contain one stamp"
    r0, c0, r1, c1 = _bbox(coords)
    assert r0 == 0 and c0 == 0, "normalized stamp must start from the top-left corner"
    block = stamp_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(block.all()), "stamp must be a dense rectangle"
    return oriented[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tile_horizontal(stamp: np.ndarray, height: int, width: int) -> np.ndarray:
    sh, sw = stamp.shape
    band = np.tile(stamp, (1, math.ceil(width / sw)))[:, :width]
    return np.tile(band, (math.ceil(height / sh), 1))[:height, :]


def _tile_vertical(stamp: np.ndarray, height: int, width: int) -> np.ndarray:
    sh, sw = stamp.shape
    strip = np.tile(stamp, (math.ceil(height / sh), 1))[:height, :]
    return np.tile(strip, (1, math.ceil(width / sw)))[:, :width]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region by repeating its corner stamp in the direction selected by the parity of the outside markers.

    Input structure:
    - The background is black.
    - One or more blue one-cell-thick rectangular frames enclose black interiors.
    - Short orange line segments sit immediately outside some frame sides, parallel to those sides.
    - Each frame interior contains exactly one dense non-blue, non-orange rectangular stamp in a corner.

    Rule:
    - For each blue frame, count the attached orange line segments around that frame.
    - Extract the corner stamp and keep its orientation relative to that corner.
    - If the segment count is even, repeat the stamp horizontally across the frame interior and continue that row pattern until the whole interior is covered.
    - If the segment count is odd, repeat the stamp vertically across the frame interior and continue that column pattern until the whole interior is covered.
    - Keep the blue frames and orange marker segments unchanged.

    Key insight:
    The orange segments do not encode a literal repeat count; only their parity matters, because it selects whether the stamp propagates by row-wise or column-wise repetition.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    orange_parts = _components(g == MARKER)

    for frame in frames:
        r0, c0, r1, c1 = frame
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        mask = (interior != BG) & (interior != FRAME) & (interior != MARKER)
        coords = np.argwhere(mask)
        assert coords.size > 0, "each frame needs a stamp"
        corner = _corner_name(_bbox(coords), interior.shape)
        oriented = _flip_to_top_left(interior, corner)
        stamp = _extract_stamp(oriented)

        marker_count = _count_markers(g, frame, orange_parts)
        ih, iw = interior.shape
        if marker_count % 2 == 0:
            filled = _tile_horizontal(stamp, ih, iw)
        else:
            filled = _tile_vertical(stamp, ih, iw)

        out[r0 + 1 : r1, c0 + 1 : c1] = _flip_from_top_left(filled, corner)

    return out


def _random_stamp(rng: random.Random, horizontal: bool) -> np.ndarray:
    if horizontal:
        shape = (1, rng.randint(1, 3))
    else:
        shape = (rng.randint(1, 3), 1)
    stamp = np.array([rng.choice(STAMP_PALETTE) for _ in range(shape[0] * shape[1])], dtype=int).reshape(shape)
    if stamp.size > 1 and rng.random() < 0.65 and len(set(map(int, stamp.flat))) == 1:
        alt = rng.choice([color for color in STAMP_PALETTE if color != int(stamp.flat[0])])
        stamp.flat[-1] = alt
    return stamp


def _corner_offset(corner: str, inner_h: int, inner_w: int, stamp_h: int, stamp_w: int) -> tuple[int, int]:
    sr = 0 if corner[0] == "t" else inner_h - stamp_h
    sc = 0 if corner[1] == "l" else inner_w - stamp_w
    return sr, sc


def _expected_interior(stamp: np.ndarray, corner: str, inner_h: int, inner_w: int, horizontal: bool) -> np.ndarray:
    top_left_stamp = _flip_to_top_left(stamp, corner)
    if horizontal:
        filled = _tile_horizontal(top_left_stamp, inner_h, inner_w)
    else:
        filled = _tile_vertical(top_left_stamp, inner_h, inner_w)
    return _flip_from_top_left(filled, corner)


def _draw_markers(
    patch: np.ndarray,
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    horizontal: bool,
) -> None:
    count = rng.choice((2, 4)) if horizontal else rng.choice((1, 3))
    sides = rng.sample(["top", "bottom", "left", "right"], count)
    for side in sides:
        if side in ("top", "bottom"):
            length = rng.randint(1, min(3, inner_w))
            start = rng.randint(0, inner_w - length)
            row = 0 if side == "top" else patch.shape[0] - 1
            col0 = 2 + start
            patch[row, col0 : col0 + length] = MARKER
        else:
            length = rng.randint(1, min(3, inner_h))
            start = rng.randint(0, inner_h - length)
            col = 0 if side == "left" else patch.shape[1] - 1
            row0 = 2 + start
            patch[row0 : row0 + length, col] = MARKER


def _build_patch(rng: random.Random, horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    stamp = _random_stamp(rng, horizontal)
    sh, sw = stamp.shape
    inner_h = rng.randint(max(INTERIOR_MIN, sh + 2), INTERIOR_MAX)
    inner_w = rng.randint(max(INTERIOR_MIN, sw + 2), INTERIOR_MAX)
    corner = rng.choice(["tl", "tr", "bl", "br"])

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    patch = np.full((outer_h + 2, outer_w + 2), BG, dtype=int)
    patch[1, 1 : outer_w + 1] = FRAME
    patch[outer_h, 1 : outer_w + 1] = FRAME
    patch[1 : outer_h + 1, 1] = FRAME
    patch[1 : outer_h + 1, outer_w] = FRAME

    sr, sc = _corner_offset(corner, inner_h, inner_w, sh, sw)
    patch[2 + sr : 2 + sr + sh, 2 + sc : 2 + sc + sw] = stamp
    _draw_markers(patch, rng, inner_h, inner_w, horizontal)

    expected = patch.copy()
    expected[2 : 2 + inner_h, 2 : 2 + inner_w] = _expected_interior(stamp, corner, inner_h, inner_w, horizontal)
    return patch, expected


def _can_place(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int) -> bool:
    h, w = occupied.shape
    ph, pw = shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + ph + pad)
    c1 = min(w, left + pw + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    sub[mask] = patch[mask]


def _place_patches(
    rng: random.Random,
    patches: list[tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    items = list(patches)
    rng.shuffle(items)
    for in_patch, out_patch in items:
        ph, pw = in_patch.shape
        if ph > h or pw > w:
            return None
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = rng.randint(0, h - ph)
            left = rng.randint(0, w - pw)
            if not _can_place(occupied, top, left, in_patch.shape, PLACEMENT_PAD):
                continue
            _paste(inp, in_patch, top, left)
            _paste(out, out_patch, top, left)
            occupied[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    rect_count = rng.randint(*RECT_COUNT_RANGE)
    orientations = [bool(rng.getrandbits(1)) for _ in range(rect_count)]
    patches = [_build_patch(rng, horizontal) for horizontal in orientations]

    max_h = max(patch.shape[0] for patch, _expected in patches)
    max_w = max(patch.shape[1] for patch, _expected in patches)
    grid_h = rng.randint(max(GRID_MIN, max_h + 1), GRID_MAX)
    grid_w = rng.randint(max(GRID_MIN, max_w + 1), GRID_MAX)

    placed = _place_patches(rng, patches, (grid_h, grid_w))
    if placed is None:
        return None
    inp, expected = placed

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
