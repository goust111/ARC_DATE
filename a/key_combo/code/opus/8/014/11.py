from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

FRAME_COLORS = {
    "up": RED,
    "down": BLUE,
    "left": GREEN,
    "right": YELLOW,
}
MARKER_PATTERNS = [
    ("up", "down", "left"),
    ("right", "up"),
    ("down", "left", "right"),
    ("up", "down"),
    ("left", "right", "up"),
    ("down", "left"),
]
_GEN_CALL_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_outline(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    mask[r0, c0:c1 + 1] = True
    mask[r1, c0:c1 + 1] = True
    mask[r0:r1 + 1, c0] = True
    mask[r0:r1 + 1, c1] = True
    return mask


def _infer_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    frame_bbox = _bbox(grid != BLACK)
    assert frame_bbox is not None, "expected a non-empty frame"
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 >= 6 and c1 - c0 >= 6, "frame interior must be large enough"

    outline = _frame_outline(grid.shape, frame_bbox)
    assert np.all(grid[outline] != BLACK), "frame outline must be colored"

    side_colors = {
        "up": int(grid[r0, c0]),
        "right": int(grid[r0, c1]),
        "left": int(grid[r1, c0]),
        "down": int(grid[r1, c1]),
    }
    assert len(set(side_colors.values())) == 4, "frame sides must have distinct colors"
    return frame_bbox, side_colors


def _marker_records(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    inner_mask = np.zeros_like(grid, dtype=bool)
    inner_mask[r0 + 1:r1, c0 + 1:c1] = True
    coords = np.argwhere((grid != BLACK) & inner_mask)
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert markers, "expected at least one marker inside the frame"
    return markers


def _paint_mask(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    side: str,
    *,
    use_parity: bool,
    directional_half: bool,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    inner_shape = (r1 - r0 - 1, c1 - c0 - 1)
    rr = np.arange(r0 + 1, r1)[:, None]
    cc = np.arange(c0 + 1, c1)[None, :]
    rm, cm = marker

    if directional_half:
        if side == "up":
            half = rr <= rm
        elif side == "down":
            half = rr >= rm
        elif side == "left":
            half = cc <= cm
        elif side == "right":
            half = cc >= cm
        else:
            raise AssertionError(f"unknown side {side}")
    else:
        half = np.ones(inner_shape, dtype=bool)
    half = np.broadcast_to(half, inner_shape)

    if use_parity:
        dist = np.maximum(np.abs(rr - rm), np.abs(cc - cm))
        return half & (dist % 2 == 0)
    return half


def _transform(
    grid: np.ndarray,
    *,
    use_parity: bool = True,
    directional_half: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_bbox, side_colors = _infer_frame(g)
    color_to_side = {color: side for side, color in side_colors.items()}

    out = g.copy()
    markers = _marker_records(g, frame_bbox)
    for rm, cm, _color in markers:
        out[rm, cm] = BLACK

    inner = out[frame_bbox[0] + 1:frame_bbox[2], frame_bbox[1] + 1:frame_bbox[3]]
    for rm, cm, color in markers:
        assert color in color_to_side, "marker color must match exactly one frame side"
        mask = _paint_mask(
            g.shape,
            frame_bbox,
            (rm, cm),
            color_to_side[color],
            use_parity=use_parity,
            directional_half=directional_half,
        )
        inner[mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each interior marker into a direction-coded alternating half-ring toward its matching frame side.

    Input structure:
    - A black background contains a one-pixel-thick colored rectangular frame.
    - The four frame sides have four distinct colors, with corners assigned to one side each.
    - Inside the frame there are 2-3 isolated single-pixel markers whose colors match frame-side colors.

    Rule:
    - Identify the frame bbox and read which color belongs to the top, bottom, left, and right side.
    - For every interior marker, use its color to choose the matching frame side and therefore the expansion direction.
    - Repaint the frame unchanged, then fill only the frame interior cells that lie in that directional half-plane and have even Chebyshev distance from the marker; later markers in row-major order overwrite earlier ones.

    Key insight:
    - The marker color is not the answer color by itself; it is a lookup key into the frame, and the frame encodes the spatial direction of the alternating half-ring expansion.
    """
    return _transform(grid, use_parity=True, directional_half=True)


def _build_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0:c1] = RED
    grid[r1, c0 + 1:c1 + 1] = BLUE
    grid[r0 + 1:r1 + 1, c0] = GREEN
    grid[r0:r1, c1] = YELLOW


def _safe_ranges(frame_bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = frame_bbox
    rows = list(range(r0 + 3, r1 - 2))
    cols = list(range(c0 + 3, c1 - 2))
    return rows, cols


def _direction_candidates(
    frame_bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    rows, cols = _safe_ranges(frame_bbox)
    assert rows and cols
    mid_r = (rows[0] + rows[-1]) // 2
    mid_c = (cols[0] + cols[-1]) // 2

    if side == "up":
        pref_rows = [r for r in rows if r <= mid_r - 1] or [r for r in rows if r <= mid_r] or rows
        pref_cols = cols
    elif side == "down":
        pref_rows = [r for r in rows if r >= mid_r + 1] or [r for r in rows if r >= mid_r] or rows
        pref_cols = cols
    elif side == "left":
        pref_rows = rows
        pref_cols = [c for c in cols if c <= mid_c - 1] or [c for c in cols if c <= mid_c] or cols
    elif side == "right":
        pref_rows = rows
        pref_cols = [c for c in cols if c >= mid_c + 1] or [c for c in cols if c >= mid_c] or cols
    else:
        raise AssertionError(f"unknown side {side}")

    return [(r, c) for r in pref_rows for c in pref_cols]


def _pairwise_well_spaced(
    placed: list[tuple[str, int, int]], candidate: tuple[int, int]
) -> bool:
    r, c = candidate
    for _side, pr, pc in placed:
        if abs(r - pr) < 3 or abs(c - pc) < 3:
            return False
    return True


def _select_markers(
    frame_bbox: tuple[int, int, int, int], pattern: tuple[str, ...]
) -> list[tuple[str, int, int]] | None:
    candidate_map = {side: _direction_candidates(frame_bbox, side) for side in pattern}
    if any(not candidates for candidates in candidate_map.values()):
        return None

    for _ in range(300):
        placed: list[tuple[str, int, int]] = []
        side_order = list(pattern)
        random.shuffle(side_order)
        for side in side_order:
            candidates = candidate_map[side][:]
            random.shuffle(candidates)
            for candidate in candidates:
                if _pairwise_well_spaced(placed, candidate):
                    placed.append((side, candidate[0], candidate[1]))
                    break
            else:
                break
        if len(placed) == len(pattern):
            return placed
    return None


def _markers_sorted(markers: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    return sorted(markers, key=lambda item: (item[1], item[2]))


def _painted_owner_counts(
    frame_bbox: tuple[int, int, int, int], markers: list[tuple[str, int, int]]
) -> dict[tuple[str, int, int], int]:
    inner_shape = (frame_bbox[2] - frame_bbox[0] - 1, frame_bbox[3] - frame_bbox[1] - 1)
    owners = np.full(inner_shape, -1, dtype=int)
    ordered = _markers_sorted(markers)
    for idx, (side, r, c) in enumerate(ordered):
        mask = _paint_mask(
            (frame_bbox[2] + 1, frame_bbox[3] + 1),
            frame_bbox,
            (r, c),
            side,
            use_parity=True,
            directional_half=True,
        )
        owners[mask] = idx
    counts: dict[tuple[str, int, int], int] = {}
    for idx, marker in enumerate(ordered):
        counts[marker] = int(np.sum(owners == idx))
    return counts


def _validate_layout(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    markers: list[tuple[str, int, int]],
) -> bool:
    ordered = _markers_sorted(markers)
    masks = [
        _paint_mask(grid.shape, frame_bbox, (r, c), side, use_parity=True, directional_half=True)
        for side, r, c in ordered
    ]
    areas = [int(mask.sum()) for mask in masks]
    if min(areas) < 8:
        return False

    overlap = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            pair_overlap = int(np.sum(masks[i] & masks[j]))
            if pair_overlap > min(areas[i], areas[j]) // 4 + 2:
                return False
            overlap += pair_overlap
    if overlap > 10:
        return False

    owner_counts = _painted_owner_counts(frame_bbox, ordered)
    if min(owner_counts.values()) < 6:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _transform(grid, use_parity=False, directional_half=True)):
        return False
    if np.array_equal(out, _transform(grid, use_parity=True, directional_half=False)):
        return False

    inner = out[frame_bbox[0] + 1:frame_bbox[2], frame_bbox[1] + 1:frame_bbox[3]]
    colored_fraction = float(np.count_nonzero(inner)) / float(inner.size)
    return 0.18 <= colored_fraction <= 0.72


def _gen_attempt(pattern: tuple[str, ...]) -> np.ndarray | None:
    height = random.randint(15, 22)
    width = random.randint(15, 22)
    inset_top = random.randint(0, 2)
    inset_left = random.randint(0, 2)
    inset_bottom = random.randint(0, 2)
    inset_right = random.randint(0, 2)
    frame_bbox = (
        inset_top,
        inset_left,
        height - 1 - inset_bottom,
        width - 1 - inset_right,
    )
    if frame_bbox[2] - frame_bbox[0] < 10 or frame_bbox[3] - frame_bbox[1] < 10:
        return None

    rows, cols = _safe_ranges(frame_bbox)
    if len(rows) < 3 or len(cols) < 3:
        return None

    markers = _select_markers(frame_bbox, pattern)
    if markers is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    _build_frame(grid, frame_bbox)
    for side, r, c in markers:
        grid[r, c] = FRAME_COLORS[side]

    if not _validate_layout(grid, frame_bbox, markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    pattern = MARKER_PATTERNS[_GEN_CALL_COUNT % len(MARKER_PATTERNS)]
    _GEN_CALL_COUNT += 1
    for _ in range(400):
        grid = _gen_attempt(pattern)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
