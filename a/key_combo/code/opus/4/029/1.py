from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 1
MARKER = 2
SPRITE = 3


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


OPEN_SPRITES: list[np.ndarray] = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]
FILLED_SPRITES: list[np.ndarray] = [
    np.array([[1, 1], [1, 1]], dtype=bool),
]


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _rect_frame_bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    if dst_len == 1:
        return 0
    if src_len == 1:
        return (dst_len - 1) // 2
    return int(round(i * (dst_len - 1) / (src_len - 1)))


def _draw_axis_line(mask: np.ndarray, a: Coord, b: Coord) -> None:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        mask[r0, c_lo : c_hi + 1] = True
        return
    assert c0 == c1
    r_lo, r_hi = sorted((r0, r1))
    mask[r_lo : r_hi + 1, c0] = True


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[Coord] = deque()

    for r in range(padded.shape[0]):
        for c in (0, padded.shape[1] - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(padded.shape[1]):
        for r in (0, padded.shape[0] - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke | holes


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = target_shape
    stroke = np.zeros((dst_h, dst_w), dtype=bool)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite)]
    mapped: dict[Coord, Coord] = {}
    for sr, sc in coords:
        tr = _scaled_index(sr, src_h, dst_h)
        tc = _scaled_index(sc, src_w, dst_w)
        mapped[(sr, sc)] = (tr, tc)
        stroke[tr, tc] = True

    coord_set = set(coords)
    for sr, sc in coords:
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) not in coord_set:
                continue
            _draw_axis_line(stroke, mapped[(sr, sc)], mapped[(nr, nc)])

    return _fill_enclosed(stroke)


def _split_from_markers(grid: np.ndarray) -> tuple[BBox, BBox]:
    markers = np.argwhere(grid == MARKER)
    assert markers.shape == (2, 2), "expected exactly two red markers"
    h, w = grid.shape
    (r0, c0), (r1, c1) = markers
    if int(r0) == int(r1):
        mid = w // 2
        in_left = int(c0) < mid and int(c1) < mid
        in_right = int(c0) >= mid and int(c1) >= mid
        assert in_left ^ in_right, "row-aligned markers must lie in one column half"
        if in_left:
            return (0, 0, h - 1, mid - 1), (0, mid, h - 1, w - 1)
        return (0, mid, h - 1, w - 1), (0, 0, h - 1, mid - 1)
    assert int(c0) == int(c1), "markers must share one row or one column"
    mid = h // 2
    in_top = int(r0) < mid and int(r1) < mid
    in_bottom = int(r0) >= mid and int(r1) >= mid
    assert in_top ^ in_bottom, "column-aligned markers must lie in one row half"
    if in_top:
        return (0, 0, mid - 1, w - 1), (mid, 0, h - 1, w - 1)
    return (mid, 0, h - 1, w - 1), (0, 0, mid - 1, w - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop to the marker-selected half, then stretch the discarded-half green sprite into the kept-half blue frame.

    Input structure:
    - The background is black, with exactly two red marker pixels in one half of the grid.
    - The marker half also contains one blue one-pixel-thick hollow rectangle, while the opposite half contains one 4-connected green sprite.
    - The markers share one row or one column, and no foreground object crosses the chosen half boundary.

    Rule:
    - If the markers share a row, split the grid into left and right halves; if they share a column, split it into top and bottom halves.
    - Keep the half containing the markers, but before discarding the other half record the green sprite mask and bbox from that discarded half.
    - Erase the red markers in the kept crop, map the sprite bbox to the blue frame interior bbox, draw straight segments between mapped adjacent sprite pixels, and flood-fill enclosed interior holes green.
    - Leave the blue frame unchanged and return only the kept half.

    Key insight:
    - The crop target and the stretch source come from opposite halves: the kept half supplies the frame, while the discarded half secretly supplies the shape that must be transplanted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    keep_bbox, discard_bbox = _split_from_markers(g)
    keep = crop_bbox(g, keep_bbox)
    discard = crop_bbox(g, discard_bbox)

    sprite_components = components_mask(discard == SPRITE)
    assert len(sprite_components) == 1, "expected one green sprite in the discarded half"
    sprite_coords = sprite_components[0]
    sr0, sc0, sr1, sc1 = bbox_coords(sprite_coords)
    sprite = discard[sr0 : sr1 + 1, sc0 : sc1 + 1] == SPRITE

    assert len(components_mask(keep == FRAME)) == 1, "expected one blue frame in the kept half"
    frame_bbox = _rect_frame_bbox(keep == FRAME)
    assert frame_bbox is not None, "blue pixels must form a hollow rectangle"
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    assert ir0 <= ir1 and ic0 <= ic1
    assert np.all(keep[ir0 : ir1 + 1, ic0 : ic1 + 1] == BG), "frame interior must start empty"

    out = keep.copy()
    out[out == MARKER] = BG
    stretched = _stretch_sprite(sprite, (ir1 - ir0 + 1, ic1 - ic0 + 1))
    out[ir0 : ir1 + 1, ic0 : ic1 + 1][stretched] = SPRITE
    return out


def _quadrant_bbox(h: int, w: int, name: str) -> BBox:
    mid_r, mid_c = h // 2, w // 2
    if name == "TL":
        return 0, 0, mid_r - 1, mid_c - 1
    if name == "TR":
        return 0, mid_c, mid_r - 1, w - 1
    if name == "BL":
        return mid_r, 0, h - 1, mid_c - 1
    if name == "BR":
        return mid_r, mid_c, h - 1, w - 1
    raise ValueError(name)


def _half_bbox(h: int, w: int, orientation: str, keep_side: str) -> BBox:
    if orientation == "vertical":
        mid_c = w // 2
        if keep_side == "left":
            return 0, 0, h - 1, mid_c - 1
        return 0, mid_c, h - 1, w - 1
    mid_r = h // 2
    if keep_side == "top":
        return 0, 0, mid_r - 1, w - 1
    return mid_r, 0, h - 1, w - 1


def _place_patch(grid: np.ndarray, patch: np.ndarray, bbox: BBox, color: int) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    ph, pw = patch.shape
    choices = [
        (r, c)
        for r in range(r0, r1 - ph + 2)
        for c in range(c0, c1 - pw + 2)
        if np.all(grid[r : r + ph, c : c + pw][patch] == BG)
    ]
    if not choices:
        return None
    rr, cc = random.choice(choices)
    grid[rr : rr + ph, cc : cc + pw][patch] = color
    return rr, cc


def _frame_positions(h: int, w: int, bbox: BBox, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    fh, fw = inner_h + 2, inner_w + 2
    return [
        (r, c)
        for r in range(r0, r1 - fh + 2)
        for c in range(c0, c1 - fw + 2)
    ]


def _paint_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = FRAME
    grid[top : bottom + 1, right] = FRAME
    grid[top, left : right + 1] = FRAME
    grid[bottom, left : right + 1] = FRAME


def _marker_pairs(grid: np.ndarray, bbox: BBox, orientation: str) -> list[tuple[Coord, Coord]]:
    r0, c0, r1, c1 = bbox
    pairs: list[tuple[Coord, Coord]] = []
    if orientation == "vertical":
        for r in range(r0, r1 + 1):
            cols = [c for c in range(c0, c1 + 1) if grid[r, c] == BG]
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if cols[j] - cols[i] >= 2:
                        pairs.append(((r, cols[i]), (r, cols[j])))
    else:
        for c in range(c0, c1 + 1):
            rows = [r for r in range(r0, r1 + 1) if grid[r, c] == BG]
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if rows[j] - rows[i] >= 2:
                        pairs.append(((rows[i], c), (rows[j], c)))
    return pairs


def _diagonal_quadrants(orientation: str, keep_side: str, diag: int) -> tuple[str, str]:
    if orientation == "vertical":
        if keep_side == "left":
            return ("TL", "BR") if diag == 0 else ("BL", "TR")
        return ("TR", "BL") if diag == 0 else ("BR", "TL")
    if keep_side == "top":
        return ("TL", "BR") if diag == 0 else ("TR", "BL")
    return ("BL", "TR") if diag == 0 else ("BR", "TL")


def _sprite_choice(inner_h: int, inner_w: int) -> tuple[np.ndarray, bool]:
    want_fill = random.random() < 0.5
    pool = FILLED_SPRITES if want_fill else OPEN_SPRITES
    choices = []
    for patch in pool:
        ph, pw = patch.shape
        if ph <= inner_h and pw <= inner_w and (ph < inner_h or pw < inner_w):
            choices.append((patch.copy(), want_fill))
    assert choices
    return random.choice(choices)


def _semantic_checks(grid: np.ndarray) -> None:
    out = main(grid)
    keep_bbox, discard_bbox = _split_from_markers(grid)
    keep = crop_bbox(grid, keep_bbox)
    erased = keep.copy()
    erased[erased == MARKER] = BG
    assert not np.array_equal(out, erased), "stretch step must change the kept half"
    assert np.count_nonzero(keep == SPRITE) == 0
    assert np.count_nonzero(crop_bbox(grid, discard_bbox) == FRAME) == 0
    assert np.count_nonzero(out == MARKER) == 0

    frame_global = _rect_frame_bbox(grid == FRAME)
    sprite_bbox = bbox_coords(np.argwhere(grid == SPRITE))
    assert frame_global is not None
    fr0, fc0, fr1, fc1 = frame_global
    sr0, sc0, sr1, sc1 = sprite_bbox
    mid_r, mid_c = grid.shape[0] // 2, grid.shape[1] // 2
    frame_row_half = 0 if fr1 < mid_r else 1
    sprite_row_half = 0 if sr1 < mid_r else 1
    frame_col_half = 0 if fc1 < mid_c else 1
    sprite_col_half = 0 if sc1 < mid_c else 1
    assert frame_row_half != sprite_row_half, "frame and sprite should also differ by row half"
    assert frame_col_half != sprite_col_half, "frame and sprite should also differ by column half"


def _build_case() -> np.ndarray | None:
    h = random.choice((16, 18, 20))
    w = random.choice((16, 18, 20))
    orientation = random.choice(("vertical", "horizontal"))
    keep_side = random.choice(("left", "right")) if orientation == "vertical" else random.choice(("top", "bottom"))
    diag = random.randint(0, 1)
    frame_quad, sprite_quad = _diagonal_quadrants(orientation, keep_side, diag)

    frame_qbox = _quadrant_bbox(h, w, frame_quad)
    fq_h = frame_qbox[2] - frame_qbox[0] + 1
    fq_w = frame_qbox[3] - frame_qbox[1] + 1
    size_choices = [
        (inner_h, inner_w)
        for inner_h in range(4, 7)
        for inner_w in range(4, 7)
        if inner_h + 2 <= fq_h and inner_w + 2 <= fq_w
    ]
    if not size_choices:
        return None
    inner_h, inner_w = random.choice(size_choices)

    sprite_patch, expect_fill = _sprite_choice(inner_h, inner_w)
    grid = np.zeros((h, w), dtype=int)

    frame_positions = _frame_positions(h, w, frame_qbox, inner_h, inner_w)
    if not frame_positions:
        return None
    frame_top, frame_left = random.choice(frame_positions)
    _paint_frame(grid, frame_top, frame_left, inner_h, inner_w)

    if _place_patch(grid, sprite_patch, _quadrant_bbox(h, w, sprite_quad), SPRITE) is None:
        return None

    marker_options = _marker_pairs(grid, _half_bbox(h, w, orientation, keep_side), orientation)
    if not marker_options:
        return None
    (r0, c0), (r1, c1) = random.choice(marker_options)
    grid[r0, c0] = MARKER
    grid[r1, c1] = MARKER

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != crop_bbox(grid, _split_from_markers(grid)[0]).shape:
        return None
    if expect_fill:
        green_area = int(np.count_nonzero(out == SPRITE))
        if green_area <= int(np.count_nonzero(sprite_patch)):
            return None

    try:
        _semantic_checks(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        case = _build_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid cross-half transplant input")
