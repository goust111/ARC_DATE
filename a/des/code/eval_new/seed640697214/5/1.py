import numpy as np
from collections import deque


BG = 0
STEP = 3
MAX_RETRIES = 300
RNG = np.random.default_rng()

TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

COLOR_POOL = np.arange(1, 10, dtype=int)

SHAPE_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    vals = np.bincount(grid.ravel())
    return int(np.argmax(vals))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _rotate_for_side(arr: np.ndarray, side: int) -> np.ndarray:
    if side == TOP:
        return arr.copy()
    if side == RIGHT:
        return np.rot90(arr, 3)
    if side == BOTTOM:
        return np.rot90(arr, 2)
    if side == LEFT:
        return np.rot90(arr, 1)
    raise ValueError(side)


def _is_contiguous(indices: np.ndarray) -> bool:
    return bool(len(indices) > 0 and np.all(np.diff(indices) == 1))


def _find_gap(frame_mask: np.ndarray) -> tuple[int, int, int]:
    sides = (
        (TOP, np.flatnonzero(~frame_mask[0, :])),
        (RIGHT, np.flatnonzero(~frame_mask[:, -1])),
        (BOTTOM, np.flatnonzero(~frame_mask[-1, :])),
        (LEFT, np.flatnonzero(~frame_mask[:, 0])),
    )
    found: list[tuple[int, int, int]] = []
    for side, idx in sides:
        if len(idx) == 0:
            continue
        assert _is_contiguous(idx), "gap segment must be contiguous"
        found.append((side, int(idx[0]), int(len(idx))))
    assert len(found) == 1, "each frame must have exactly one gap side"
    return found[0]


def _attached_edge_filled(mask: np.ndarray, side: int) -> bool:
    if side == TOP:
        return bool(mask[-1, :].all())
    if side == RIGHT:
        return bool(mask[:, 0].all())
    if side == BOTTOM:
        return bool(mask[0, :].all())
    if side == LEFT:
        return bool(mask[:, -1].all())
    return False


def _slot_top_left(
    frame_bbox: tuple[int, int, int, int],
    side: int,
    segment_start: int,
    rotated_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    rh, rw = rotated_shape
    if side == TOP:
        return r0 - rh + 1, c0 + segment_start
    if side == RIGHT:
        return r0 + segment_start, c1
    if side == BOTTOM:
        return r1, c0 + segment_start
    if side == LEFT:
        return r0 + segment_start, c0 - rw + 1
    raise ValueError(side)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for color in sorted(set(map(int, g.ravel())) - {bg}):
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form one frame and one piece"
        comps.sort(key=len, reverse=True)
        frame_coords, piece_coords = comps

        frame_mask, frame_bbox = _mask_from_coords(frame_coords)
        side, segment_start, segment_len = _find_gap(frame_mask)

        piece_mask, piece_bbox = _mask_from_coords(piece_coords)
        rotated = _rotate_for_side(piece_mask, side)
        rh, rw = rotated.shape
        assert _attached_edge_filled(rotated, side), "piece edge must cover the whole gap segment"
        if side in (TOP, BOTTOM):
            assert rw == segment_len
        else:
            assert rh == segment_len

        pr0, pc0, pr1, pc1 = piece_bbox
        out[pr0 : pr1 + 1, pc0 : pc1 + 1][piece_mask] = bg

        top, left = _slot_top_left(frame_bbox, side, segment_start, rotated.shape)
        assert 0 <= top and 0 <= left
        assert top + rh <= out.shape[0] and left + rw <= out.shape[1]
        _paint_mask(out, rotated, top, left, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each broken nested frame side by rotating the matching same-color fragment into its gap.

    Input structure:
    - Black background surrounds a central set of 3-4 concentric rectangular frames, each in a different color.
    - Every frame is one pixel thick and has exactly one continuous missing segment on one side.
    - Outside the outer frame, each frame color appears once more as a small connected fragment.

    Rule:
    - For each color, treat the large component as the broken frame and the small component as its loose fragment.
    - Detect which side of the frame contains the missing segment.
    - Rotate the matching fragment according to the side mapping: top -> 0 degrees, right -> 90 degrees clockwise, bottom -> 180 degrees, left -> 90 degrees counterclockwise.
    - Remove the loose fragment from its original position and paste the rotated fragment into the frame gap so the fragment's bounding box is flush with that side.

    Key insight:
    - Color chooses which fragment belongs to which frame, while the side of the gap encodes the fragment rotation needed before insertion.
    """
    return _solve(grid)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _erase_gap(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: int,
    segment_start: int,
    segment_len: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if side == TOP:
        grid[r0, c0 + segment_start : c0 + segment_start + segment_len] = BG
    elif side == RIGHT:
        grid[r0 + segment_start : r0 + segment_start + segment_len, c1] = BG
    elif side == BOTTOM:
        grid[r1, c0 + segment_start : c0 + segment_start + segment_len] = BG
    else:
        grid[r0 + segment_start : r0 + segment_start + segment_len, c0] = BG


def _frame_spec(
    outer_bbox: tuple[int, int, int, int],
    layer: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = outer_bbox
    d = layer * STEP
    return r0 + d, c0 + d, r1 - d, c1 - d


def _random_gap_start(side: int, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    rh, rw = shape
    if side in (TOP, BOTTOM):
        side_len = c1 - c0 + 1
        return int(RNG.integers(1, side_len - rw))
    side_len = r1 - r0 + 1
    return int(RNG.integers(1, side_len - rh))


def _can_place_piece(
    grid: np.ndarray,
    piece_mask: np.ndarray,
    top: int,
    left: int,
    blocked_boxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = grid.shape
    ph, pw = piece_mask.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    piece_box = (top, left, top + ph - 1, left + pw - 1)
    for blocked in blocked_boxes:
        if _intersects(piece_box, blocked):
            return False
    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(h, top + ph + 1)
    cc1 = min(w, left + pw + 1)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _expected_output(
    grid: np.ndarray,
    piece_specs: list[dict],
    slot_specs: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for spec in piece_specs:
        mask = spec["mask"]
        top = spec["top"]
        left = spec["left"]
        out[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = BG
    for spec in slot_specs:
        _paint_mask(out, spec["mask"], spec["top"], spec["left"], spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_layers = int(RNG.integers(3, 5))
    inner_h = int(RNG.integers(5, 8))
    inner_w = int(RNG.integers(5, 8))
    outer_h = inner_h + 2 * STEP * (n_layers - 1)
    outer_w = inner_w + 2 * STEP * (n_layers - 1)

    extra_h = int(RNG.integers(4, 31 - outer_h))
    extra_w = int(RNG.integers(4, 31 - outer_w))
    grid_h = outer_h + extra_h
    grid_w = outer_w + extra_w

    top_margin = int(RNG.integers(2, extra_h - 1))
    left_margin = int(RNG.integers(2, extra_w - 1))
    outer_bbox = (
        top_margin,
        left_margin,
        top_margin + outer_h - 1,
        left_margin + outer_w - 1,
    )

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = RNG.choice(COLOR_POOL, size=n_layers, replace=False)
    sides = RNG.integers(0, 4, size=n_layers)
    shape_ids = RNG.choice(len(SHAPE_LIBRARY), size=n_layers, replace=True)

    slot_specs: list[dict] = []
    for i in range(n_layers):
        bbox = _frame_spec(outer_bbox, i)
        color = int(colors[i])
        side = int(sides[i])
        base_piece = SHAPE_LIBRARY[int(shape_ids[i])]
        rotated = _rotate_for_side(base_piece, side)
        segment_start = _random_gap_start(side, bbox, rotated.shape)

        _draw_frame(grid, bbox, color)
        segment_len = rotated.shape[1] if side in (TOP, BOTTOM) else rotated.shape[0]
        _erase_gap(grid, bbox, side, segment_start, segment_len)

        slot_top, slot_left = _slot_top_left(bbox, side, segment_start, rotated.shape)
        slot_specs.append(
            {
                "color": color,
                "mask": rotated,
                "top": slot_top,
                "left": slot_left,
            }
        )

    piece_specs: list[dict] = []
    blocked_boxes = [(
        outer_bbox[0] - 1,
        outer_bbox[1] - 1,
        outer_bbox[2] + 1,
        outer_bbox[3] + 1,
    )]
    for spec in slot_specs:
        sh, sw = spec["mask"].shape
        blocked_boxes.append((spec["top"], spec["left"], spec["top"] + sh - 1, spec["left"] + sw - 1))
    for i in range(n_layers):
        color = int(colors[i])
        piece_mask = SHAPE_LIBRARY[int(shape_ids[i])]
        placed = False
        for _ in range(200):
            top = int(RNG.integers(0, grid_h - piece_mask.shape[0] + 1))
            left = int(RNG.integers(0, grid_w - piece_mask.shape[1] + 1))
            if not _can_place_piece(grid, piece_mask, top, left, blocked_boxes):
                continue
            _paint_mask(grid, piece_mask, top, left, color)
            piece_specs.append({"mask": piece_mask, "top": top, "left": left})
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, piece_specs, slot_specs)
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
