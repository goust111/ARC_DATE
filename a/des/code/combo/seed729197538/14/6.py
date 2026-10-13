from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 9
STAMP_COLORS = (1, 2, 3, 4, 5, 6, 8)
OUTER_MARGIN = 1
STAMP_GAP = 2
GEN_ATTEMPTS = 400

CORNER_TO_TRANSFORM = {
    frozenset(("tl", "tr")): "flip_lr",
    frozenset(("bl", "br")): "identity",
    frozenset(("tl", "bl")): "flip_ud",
    frozenset(("tr", "br")): "rot180",
    frozenset(("tl", "br")): "rot90_cw",
    frozenset(("tr", "bl")): "rot90_ccw",
}
RANDOM_TRANSFORMS = (
    "identity",
    "rot90_cw",
    "rot180",
    "rot90_ccw",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue
        seen[start_r, start_c] = True
        q = deque([(start_r, start_c)])
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox(coords)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        out = np.array(mask, dtype=bool, copy=True)
    elif name == "flip_lr":
        out = np.fliplr(mask)
    elif name == "flip_ud":
        out = np.flipud(mask)
    elif name == "rot180":
        out = np.rot90(mask, 2)
    elif name == "rot90_cw":
        out = np.rot90(mask, -1)
    elif name == "rot90_ccw":
        out = np.rot90(mask, 1)
    elif name == "transpose":
        out = np.array(mask.T, dtype=bool, copy=True)
    elif name == "anti_transpose":
        out = np.rot90(mask.T, 2)
    else:
        raise AssertionError(name)
    return _crop_mask(out)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_mask(mask)
    return cropped.shape, cropped.astype(np.uint8).tobytes()


def _find_canvas(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, frozenset[str]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[tuple[int, int, int, int], int, frozenset[str]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue

            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(_mask_from_coords(comp), border):
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.shape[0] < 2 or interior.shape[1] < 2:
                continue

            allowed = np.zeros_like(interior, dtype=bool)
            allowed[0, 0] = True
            allowed[0, -1] = True
            allowed[-1, 0] = True
            allowed[-1, -1] = True
            if np.any((interior != BG) & ~allowed):
                continue

            corner_values = {
                "tl": int(interior[0, 0]),
                "tr": int(interior[0, -1]),
                "bl": int(interior[-1, 0]),
                "br": int(interior[-1, -1]),
            }
            marker_values = [v for v in corner_values.values() if v not in (BG, color)]
            if len(marker_values) != 2 or len(set(marker_values)) != 1:
                continue

            marker_color = int(marker_values[0])
            marked = frozenset(name for name, value in corner_values.items() if value == marker_color)
            if marked not in CORNER_TO_TRANSFORM:
                continue
            if any(value not in (BG, marker_color) for value in corner_values.values()):
                continue
            candidates.append(((r0, c0, r1, c1), marker_color, marked))

    assert len(candidates) == 1, "expected a unique canvas"
    return candidates[0]


def _find_stamp(grid: np.ndarray, color: int, frame_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    outside_frame = np.ones(g.shape, dtype=bool)
    outside_frame[r0 : r1 + 1, c0 : c1 + 1] = False

    candidates: list[np.ndarray] = []
    for comp in _components(g == int(color)):
        if np.all(outside_frame[comp[:, 0], comp[:, 1]]):
            candidates.append(comp)
    assert len(candidates) == 1, "expected a unique outside stamp"
    comp = candidates[0]
    return comp, _mask_from_coords(comp)


def _place_centered(canvas_shape: tuple[int, int], stamp: np.ndarray) -> tuple[int, int]:
    inner_h, inner_w = canvas_shape
    stamp_h, stamp_w = stamp.shape
    assert stamp_h <= inner_h and stamp_w <= inner_w
    return (inner_h - stamp_h) // 2, (inner_w - stamp_w) // 2


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_bbox, stamp_color, marked_corners = _find_canvas(g)
    stamp_coords, stamp_mask = _find_stamp(g, stamp_color, frame_bbox)
    transform = CORNER_TO_TRANSFORM[marked_corners]
    transformed = _apply_transform(stamp_mask, transform)

    fr0, fc0, fr1, fc1 = frame_bbox
    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    inner_h, inner_w = inner.shape
    off_r, off_c = _place_centered((inner_h, inner_w), transformed)
    dst_r0 = fr0 + 1 + off_r
    dst_c0 = fc0 + 1 + off_c
    dst_r1 = dst_r0 + transformed.shape[0]
    dst_c1 = dst_c0 + transformed.shape[1]

    out = g.copy()
    out[stamp_coords[:, 0], stamp_coords[:, 1]] = BG
    patch = out[dst_r0:dst_r1, dst_c0:dst_c1]
    assert patch.shape == transformed.shape
    assert np.all(patch[transformed] == BG), "center stamp placement must not overwrite markers"
    patch[transformed] = stamp_color
    return out


def _is_strongly_asymmetric(mask: np.ndarray) -> bool:
    signatures = {_mask_signature(_apply_transform(mask, name)) for name in RANDOM_TRANSFORMS}
    return len(signatures) == len(RANDOM_TRANSFORMS)


def _sample_stamp() -> np.ndarray:
    for _ in range(500):
        box_h = random.randint(2, 5)
        box_w = random.randint(2, 5)
        max_cells = min(8, box_h * box_w)
        if max_cells < 4:
            continue
        target_cells = random.randint(4, max_cells)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        frontier = set()
        for nb in _neighbors4(*start):
            rr, cc = nb
            if 0 <= rr < box_h and 0 <= cc < box_w:
                frontier.add(nb)

        while len(cells) < target_cells and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(*cell):
                rr, cc = nb
                if 0 <= rr < box_h and 0 <= cc < box_w and nb not in cells:
                    frontier.add(nb)

        if len(cells) != target_cells:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if min(mask.shape) < 2:
            continue
        if _is_strongly_asymmetric(mask):
            return mask
    raise RuntimeError("failed to sample an asymmetric stamp")


def _choose_centered_size(min_size: int, parity: int) -> int:
    candidates = [size for size in range(max(min_size, 6), max(min_size, 6) + 9) if size % 2 == parity]
    assert candidates
    return random.choice(candidates)


def _stamp_positions_for_side(
    grid_shape: tuple[int, int],
    stamp_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    sh, sw = stamp_shape
    fr0, fc0, fr1, fc1 = frame_bbox

    row_min = OUTER_MARGIN
    col_min = OUTER_MARGIN
    row_max = h - sh - OUTER_MARGIN
    col_max = w - sw - OUTER_MARGIN
    if row_min > row_max or col_min > col_max:
        return []

    positions: list[tuple[int, int]] = []
    if side == "top":
        max_row = fr0 - sh - STAMP_GAP
        if row_min > max_row:
            return []
        for r0 in range(row_min, max_row + 1):
            for c0 in range(col_min, col_max + 1):
                positions.append((r0, c0))
    elif side == "bottom":
        min_row = fr1 + STAMP_GAP + 1
        if min_row > row_max:
            return []
        for r0 in range(min_row, row_max + 1):
            for c0 in range(col_min, col_max + 1):
                positions.append((r0, c0))
    elif side == "left":
        max_col = fc0 - sw - STAMP_GAP
        if col_min > max_col:
            return []
        for r0 in range(row_min, row_max + 1):
            for c0 in range(col_min, max_col + 1):
                positions.append((r0, c0))
    elif side == "right":
        min_col = fc1 + STAMP_GAP + 1
        if min_col > col_max:
            return []
        for r0 in range(row_min, row_max + 1):
            for c0 in range(min_col, col_max + 1):
                positions.append((r0, c0))
    else:
        raise AssertionError(side)
    return positions


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _expected_output(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    stamp_top_left: tuple[int, int],
    stamp_mask: np.ndarray,
    transformed: np.ndarray,
    stamp_color: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    sr0, sc0 = stamp_top_left
    out[sr0 : sr0 + stamp_mask.shape[0], sc0 : sc0 + stamp_mask.shape[1]][stamp_mask] = BG

    fr0, fc0, fr1, fc1 = frame_bbox
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    off_r, off_c = _place_centered((inner_h, inner_w), transformed)
    dr0 = fr0 + 1 + off_r
    dc0 = fc0 + 1 + off_c
    patch = out[dr0 : dr0 + transformed.shape[0], dc0 : dc0 + transformed.shape[1]]
    assert np.all(patch[transformed] == BG)
    patch[transformed] = stamp_color
    return out


def _gen_attempt() -> np.ndarray | None:
    marked_corners = random.choice(tuple(CORNER_TO_TRANSFORM.keys()))
    transform_name = CORNER_TO_TRANSFORM[marked_corners]
    stamp_color = random.choice(STAMP_COLORS)

    stamp = _sample_stamp()
    stamp = _apply_transform(stamp, random.choice(RANDOM_TRANSFORMS))
    transformed = _apply_transform(stamp, transform_name)

    inner_h = _choose_centered_size(transformed.shape[0] + 4, transformed.shape[0] % 2)
    inner_w = _choose_centered_size(transformed.shape[1] + 4, transformed.shape[1] % 2)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    stamp_h, stamp_w = stamp.shape
    min_h = max(frame_h + stamp_h + STAMP_GAP + 2 * OUTER_MARGIN, 12)
    min_w = max(frame_w + stamp_w + STAMP_GAP + 2 * OUTER_MARGIN, 12)
    max_h = min(30, min_h + 8)
    max_w = min(30, min_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)
    side = random.choice(("top", "bottom", "left", "right"))

    frame_row_lo = OUTER_MARGIN
    frame_col_lo = OUTER_MARGIN
    frame_row_hi = grid_h - frame_h - OUTER_MARGIN
    frame_col_hi = grid_w - frame_w - OUTER_MARGIN
    if frame_row_lo > frame_row_hi or frame_col_lo > frame_col_hi:
        return None

    for _ in range(200):
        fr0 = random.randint(frame_row_lo, frame_row_hi)
        fc0 = random.randint(frame_col_lo, frame_col_hi)
        frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)
        stamp_positions = _stamp_positions_for_side((grid_h, grid_w), stamp.shape, frame_bbox, side)
        if not stamp_positions:
            continue

        sr0, sc0 = random.choice(stamp_positions)
        grid = np.zeros((grid_h, grid_w), dtype=int)
        _draw_frame(grid, frame_bbox, FRAME_COLOR)

        inner_corners = {
            "tl": (fr0 + 1, fc0 + 1),
            "tr": (fr0 + 1, fc0 + frame_w - 2),
            "bl": (fr0 + frame_h - 2, fc0 + 1),
            "br": (fr0 + frame_h - 2, fc0 + frame_w - 2),
        }
        for name in marked_corners:
            rr, cc = inner_corners[name]
            grid[rr, cc] = stamp_color

        _paint_mask(grid, sr0, sc0, stamp, stamp_color)

        expected = _expected_output(grid, frame_bbox, (sr0, sc0), stamp, transformed, stamp_color)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
