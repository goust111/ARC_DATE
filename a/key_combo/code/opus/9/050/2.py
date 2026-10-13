from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (BLUE, GREEN, YELLOW)
CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRS8 = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]

GRID_HEIGHT_RANGE = (20, 29)
GRID_WIDTH_RANGE = (20, 29)
SQUARE_SIDES = (6, 8, 10)
FRAME_SIDE_RANGE = (3, 7)
GEN_RETRY_LIMIT = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _marker_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(np.isin(grid, MARKER_COLORS))
    assert coords.shape[0] == 3, "expected exactly three colored corner markers"
    colors = [int(grid[r, c]) for r, c in coords]
    assert len(set(colors)) == 3, "marker colors must be distinct"
    return [tuple(map(int, rc)) for rc in coords]


def _infer_square(marker_cells: list[tuple[int, int]]):
    rows = sorted({r for r, _ in marker_cells})
    cols = sorted({c for _, c in marker_cells})
    assert len(rows) == 2 and len(cols) == 2, "markers must occupy two rows and two columns"

    r0, r1 = rows
    c0, c1 = cols
    side = r1 - r0
    assert side >= 6 and side == c1 - c0 and side % 2 == 0, "markers must define an even square"

    corners = {(r, c) for r in rows for c in cols}
    marker_set = set(marker_cells)
    assert len(marker_set) == 3 and marker_set < corners, "markers must be three square corners"

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    missing_corner = next(iter(corners - marker_set))
    return center, (r0, c0, r1, c1), missing_corner


def _trace_rays(grid: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    ray_mask = np.zeros_like(grid, dtype=bool)
    occupied = grid != BG
    cr, cc = center

    for dr, dc in DIRS8:
        r, c = cr, cc
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc):
                break
            if occupied[nr, nc]:
                break
            ray_mask[nr, nc] = True
            r, c = nr, nc
    return ray_mask


def _outline_mask(shape: tuple[int, int], frame: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = frame
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _frame_touched(ray_mask: np.ndarray, frame: tuple[int, int, int, int]) -> bool:
    adjacent = _dilate4(ray_mask)
    r0, c0, r1, c1 = frame
    return (
        adjacent[r0, c0 : c1 + 1].any()
        or adjacent[r1, c0 : c1 + 1].any()
        or adjacent[r0 : r1 + 1, c0].any()
        or adjacent[r0 : r1 + 1, c1].any()
    )


def _find_frame_components(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = grid == GREY
    visited = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    frames: list[tuple[int, int, int, int]] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if visited[sr, sc]:
            continue
        stack = [(sr, sc)]
        coords = []
        visited[sr, sc] = True
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frame must be at least 3x3"

        expected = _outline_mask(grid.shape, (r0, c0, r1, c1))
        actual = np.zeros_like(mask, dtype=bool)
        rr, cc = zip(*coords)
        actual[rr, cc] = True
        assert np.array_equal(actual, expected), "grey component must be a 1-pixel hollow rectangle"
        assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG), "frame interior must be black"
        frames.append((r0, c0, r1, c1))

    frames.sort()
    assert len(frames) == 3, "expected exactly three frames"
    return frames


def _apply_rule_from_frames(grid: np.ndarray, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    center, _square_bbox, _missing = _infer_square(_marker_cells(out))
    ray_mask = _trace_rays(out, center)
    cr, cc = center
    out[cr, cc] = RED
    out[ray_mask] = RED

    for frame in frames:
        if _frame_touched(ray_mask, frame):
            r0, c0, r1, c1 = frame
            out[r0 + 1 : r1, c0 + 1 : c1] = YELLOW

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a square center from three corner markers, emit truncated red rays, and fill only touched gray frame interiors.

    Input structure:
    - The background is black.
    - Exactly three single-pixel markers in blue, green, and yellow occupy three corners of one axis-aligned even-sided square.
    - Three separate gray hollow rectangular frames appear elsewhere, each with a black interior and no overlap with the implied square region.

    Rule:
    - Infer the missing square corner and the square center from the three colored corner markers.
    - Paint the center red, then send red rays from that center in all eight compass directions.
    - Each ray advances cell by cell until its next step would leave the grid or enter a non-black obstacle cell, so it stops before markers and frame outlines.
    - Any gray frame whose outline has at least one 4-neighbor contact with a ray has its enclosed interior filled yellow; untouched frames stay hollow.

    Key insight:
    The three corner markers do not merely identify a square: they determine the unique ray source, and only frames that end up adjacent to the obstacle-limited rays should be filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _find_frame_components(g)
    return _apply_rule_from_frames(g, frames)


def _paint_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _frame_clear(
    frame: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
    square_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = frame
    if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
        return False
    if _boxes_intersect(frame, square_bbox, pad=0):
        return False
    for other in existing:
        if _boxes_intersect(frame, other, pad=1):
            return False
    return True


def _frame_from_cardinal_hit(
    center: tuple[int, int],
    square_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
    anchor_offset: int,
    distance: int,
) -> tuple[int, int, int, int]:
    cr, cc = center
    sr0, sc0, sr1, sc1 = square_bbox
    dr, dc = direction

    if (dr, dc) == (0, 1):
        r0 = cr - anchor_offset
        c0 = sc1 + distance
    elif (dr, dc) == (0, -1):
        r0 = cr - anchor_offset
        c0 = sc0 - distance - width + 1
    elif (dr, dc) == (-1, 0):
        r0 = sr0 - distance - height + 1
        c0 = cc - anchor_offset
    elif (dr, dc) == (1, 0):
        r0 = sr1 + distance
        c0 = cc - anchor_offset
    else:
        raise AssertionError("not a cardinal direction")
    return (r0, c0, r0 + height - 1, c0 + width - 1)


def _frame_from_diag_hit(
    center: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
    distance: int,
) -> tuple[int, int, int, int]:
    cr, cc = center
    dr, dc = direction

    if (dr, dc) == (-1, -1):
        r0 = cr - distance
        c0 = cc - distance
    elif (dr, dc) == (-1, 1):
        r0 = cr - distance
        c0 = cc + distance - width + 1
    elif (dr, dc) == (1, -1):
        r0 = cr + distance - height + 1
        c0 = cc - distance
    elif (dr, dc) == (1, 1):
        r0 = cr + distance - height + 1
        c0 = cc + distance - width + 1
    else:
        raise AssertionError("not a diagonal direction")
    return (r0, c0, r0 + height - 1, c0 + width - 1)


def _current_grid(base_grid: np.ndarray, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.array(base_grid, copy=True)
    for frame in frames:
        _paint_frame(grid, frame)
    return grid


def _touch_vector(
    base_grid: np.ndarray, frames: list[tuple[int, int, int, int]]
) -> list[bool]:
    grid = _current_grid(base_grid, frames)
    center, _square, _missing = _infer_square(_marker_cells(grid))
    ray_mask = _trace_rays(grid, center)
    return [_frame_touched(ray_mask, frame) for frame in frames]


def _try_add_frame(
    base_grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    square_bbox: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
    want_touched: bool,
) -> bool:
    if not _frame_clear(frame, frames, square_bbox, base_grid.shape):
        return False
    touched = _touch_vector(base_grid, frames + [frame])[-1]
    return touched == want_touched


def _place_cardinal_touch(
    rng: np.random.Generator,
    base_grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    square_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    reserve_after: int = 0,
) -> tuple[int, int, int, int] | None:
    h, w = base_grid.shape
    center, _bbox, _missing = _infer_square(_marker_cells(base_grid))

    for _ in range(120):
        fh = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        fw = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        if direction[0] == 0:
            anchor = int(rng.integers(1, fh - 1))
        else:
            anchor = int(rng.integers(1, fw - 1))
        dist = int(rng.integers(1, 6))
        frame = _frame_from_cardinal_hit(center, square_bbox, direction, fh, fw, anchor, dist)
        r0, c0, r1, c1 = frame

        if direction == (0, 1) and w - 1 - c1 < reserve_after:
            continue
        if direction == (0, -1) and c0 < reserve_after:
            continue
        if direction == (1, 0) and h - 1 - r1 < reserve_after:
            continue
        if direction == (-1, 0) and r0 < reserve_after:
            continue

        if _try_add_frame(base_grid, frames, square_bbox, frame, want_touched=True):
            return frame
    return None


def _place_diag_touch(
    rng: np.random.Generator,
    base_grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    square_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    center, sq, _missing = _infer_square(_marker_cells(base_grid))
    radius = (sq[2] - sq[0]) // 2

    for _ in range(120):
        fh = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        fw = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        min_dist = radius + max(fh, fw) + 1
        max_dist = min(base_grid.shape) - 2
        if min_dist > max_dist:
            return None
        dist = int(rng.integers(min_dist, max_dist + 1))
        frame = _frame_from_diag_hit(center, direction, fh, fw, dist)
        if _try_add_frame(base_grid, frames, square_bbox, frame, want_touched=True):
            return frame
    return None


def _place_occluded_untouched(
    rng: np.random.Generator,
    base_grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    square_bbox: tuple[int, int, int, int],
    blocker: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    center, _bbox, _missing = _infer_square(_marker_cells(base_grid))
    cr, cc = center
    br0, bc0, br1, bc1 = blocker
    h, w = base_grid.shape

    for _ in range(160):
        fh = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        fw = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        if direction[0] == 0:
            anchor = int(rng.integers(1, fh - 1))
        else:
            anchor = int(rng.integers(1, fw - 1))
        gap = int(rng.integers(2, 6))

        if direction == (0, 1):
            r0 = cr - anchor
            c0 = bc1 + gap
        elif direction == (0, -1):
            r0 = cr - anchor
            c0 = bc0 - gap - fw + 1
        elif direction == (1, 0):
            r0 = br1 + gap
            c0 = cc - anchor
        elif direction == (-1, 0):
            r0 = br0 - gap - fh + 1
            c0 = cc - anchor
        else:
            raise AssertionError("occlusion branch expects a cardinal direction")

        frame = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if _try_add_frame(base_grid, frames, square_bbox, frame, want_touched=False):
            return frame
    return None


def _place_far_untouched(
    rng: np.random.Generator,
    base_grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    square_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    h, w = base_grid.shape
    for _ in range(300):
        fh = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        fw = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
        r0 = int(rng.integers(0, h - fh + 1))
        c0 = int(rng.integers(0, w - fw + 1))
        frame = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if _try_add_frame(base_grid, frames, square_bbox, frame, want_touched=False):
            return frame
    return None


def _make_square_case(rng: np.random.Generator):
    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))
    side_options = [s for s in SQUARE_SIDES if s <= min(h, w) - 4]
    if not side_options:
        return None
    side = int(rng.choice(side_options))
    r0 = int(rng.integers(2, h - side - 1))
    c0 = int(rng.integers(2, w - side - 1))
    r1 = r0 + side
    c1 = c0 + side

    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing_index = int(rng.integers(0, 4))
    marker_positions = [pos for i, pos in enumerate(corners) if i != missing_index]
    colors = rng.permutation(np.array(MARKER_COLORS, dtype=int))

    grid = np.zeros((h, w), dtype=int)
    for color, (r, c) in zip(colors, marker_positions):
        grid[r, c] = int(color)

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    missing_corner = corners[missing_index]
    missing_dir = (int(np.sign(missing_corner[0] - center[0])), int(np.sign(missing_corner[1] - center[1])))
    return grid, (r0, c0, r1, c1), center, missing_dir


def _validate_semantics(
    base_grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    branch: str,
    blocker_index: int | None = None,
    occluded_index: int | None = None,
    diag_index: int | None = None,
) -> np.ndarray | None:
    inp = _current_grid(base_grid, frames)
    try:
        detected = _find_frame_components(inp)
        assert detected == sorted(frames)
        out = _apply_rule_from_frames(inp, detected)
        assert np.array_equal(main(inp), out)
    except Exception:
        return None

    touched = _touch_vector(base_grid, frames)
    if not any(touched) or all(touched):
        return None

    if branch == "occluded":
        assert blocker_index is not None and occluded_index is not None
        if not touched[blocker_index] or touched[occluded_index]:
            return None
        reduced_frames = [frame for i, frame in enumerate(frames) if i != blocker_index]
        reduced_touch = _touch_vector(base_grid, reduced_frames)
        occluded_reduced_index = occluded_index - (1 if blocker_index < occluded_index else 0)
        if not reduced_touch[occluded_reduced_index]:
            return None

    if branch == "missing_diag":
        assert diag_index is not None
        if not touched[diag_index]:
            return None

    if np.array_equal(inp, out):
        return None
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    square_case = _make_square_case(rng)
    if square_case is None:
        return None
    base_grid, square_bbox, _center, missing_dir = square_case
    frames: list[tuple[int, int, int, int]] = []

    branch = rng.choice(np.array(["occluded", "missing_diag", "mixed"]))

    if branch == "occluded":
        cardinal = CARDINAL_DIRS[int(rng.integers(len(CARDINAL_DIRS)))]
        blocker = _place_cardinal_touch(rng, base_grid, frames, square_bbox, cardinal, reserve_after=6)
        if blocker is None:
            return None
        frames.append(blocker)

        occluded = _place_occluded_untouched(rng, base_grid, frames, square_bbox, blocker, cardinal)
        if occluded is None:
            return None
        frames.append(occluded)

        extra = None
        if rng.random() < 0.5:
            if missing_dir in DIAGONAL_DIRS:
                extra = _place_diag_touch(rng, base_grid, frames, square_bbox, missing_dir)
            if extra is None:
                other_cards = [d for d in CARDINAL_DIRS if d != cardinal]
                rng.shuffle(other_cards)
                for other in other_cards:
                    extra = _place_cardinal_touch(rng, base_grid, frames, square_bbox, other)
                    if extra is not None:
                        break
        if extra is None:
            extra = _place_far_untouched(rng, base_grid, frames, square_bbox)
        if extra is None:
            return None
        frames.append(extra)
        return _validate_semantics(base_grid, frames, branch, blocker_index=0, occluded_index=1)

    if branch == "missing_diag":
        diag_frame = _place_diag_touch(rng, base_grid, frames, square_bbox, missing_dir)
        if diag_frame is None:
            return None
        frames.append(diag_frame)

        second = None
        if rng.random() < 0.5:
            cards = CARDINAL_DIRS.copy()
            rng.shuffle(cards)
            for cardinal in cards:
                second = _place_cardinal_touch(rng, base_grid, frames, square_bbox, cardinal)
                if second is not None:
                    break
        if second is None:
            second = _place_far_untouched(rng, base_grid, frames, square_bbox)
        if second is None:
            return None
        frames.append(second)

        third = _place_far_untouched(rng, base_grid, frames, square_bbox)
        if third is None:
            return None
        frames.append(third)
        return _validate_semantics(base_grid, frames, branch, diag_index=0)

    cards = CARDINAL_DIRS.copy()
    rng.shuffle(cards)
    first = _place_cardinal_touch(rng, base_grid, frames, square_bbox, cards[0])
    if first is None:
        return None
    frames.append(first)

    second = None
    if rng.random() < 0.5:
        second = _place_diag_touch(rng, base_grid, frames, square_bbox, missing_dir)
    if second is None:
        for cardinal in cards[1:]:
            second = _place_cardinal_touch(rng, base_grid, frames, square_bbox, cardinal)
            if second is not None:
                break
    if second is None:
        return None
    frames.append(second)

    third = _place_far_untouched(rng, base_grid, frames, square_bbox)
    if third is None:
        return None
    frames.append(third)
    return _validate_semantics(base_grid, frames, branch)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
