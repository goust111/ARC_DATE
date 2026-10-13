from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each loose right-side pattern to the same-colored broken frame and patch the missing edge segment.

    Input structure:
    - The grid has a dominant background color and one full-height vertical separator column.
    - The separator splits the canvas into equal left and right halves.
    - The left half contains several single-pixel rectangular frames in distinct colors, each with one contiguous gap on one edge.
    - The right half contains one small 4-connected monochrome pattern for each frame, with matching color.

    Rule:
    - Find the separator and background, then inspect the left half color by color.
    - For each colored frame, recover its bounding rectangle and identify the missing contiguous border segment.
    - Match the frame to the unique right-half component of the same color.
    - Fill the frame's missing cells with that color and erase the matched right-half component back to background.

    Key insight:
    - Color determines which right-side piece belongs to which left-side frame, and the frame's bounding box reveals the exact border cells that must be restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    sep_col, sep_color = _find_separator(g, bg)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    frames = _find_frames(left, bg)
    patterns = _find_patterns(right, bg, sep_color, sep_col + 1)

    assert frames, "expected at least one broken frame"
    assert set(frames) == set(patterns), "frame colors and pattern colors must match exactly"

    out = g.copy()
    for color, frame in frames.items():
        pattern = patterns[color]
        gap_shape = _normalize(frame["gap_coords"])
        pattern_shape = _normalize(pattern["local_coords"])
        assert gap_shape in _shape_rotations(pattern_shape), "pattern shape must match the frame gap"

        for r, c in frame["gap_coords"]:
            out[r, c] = color
        for r, c in pattern["global_coords"]:
            out[r, c] = bg

    return out


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        left_w = c
        right_w = w - c - 1
        if left_w != right_w:
            continue
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected exactly one separator column"
    return candidates[0]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(left: np.ndarray, bg: int) -> dict[int, dict[str, object]]:
    frames: dict[int, dict[str, object]] = {}
    for color in np.unique(left):
        color = int(color)
        if color == bg:
            continue
        comps = _connected_components(left == color)
        assert len(comps) == 1, "each frame color should form exactly one component"
        coords = comps[0]
        gap_coords, bbox = _frame_gap(coords)
        frames[color] = {"gap_coords": gap_coords, "bbox": bbox}
    return frames


def _frame_gap(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    bbox = (r0, c0, r1, c1)

    actual = set(coords)
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in {r0, r1} or c in {c0, c1}
    }
    assert actual <= expected, "frame pixels must stay on the rectangle perimeter"

    missing = sorted(expected - actual)
    assert 2 <= len(missing) <= 4, "gap must be a short border segment"

    side_candidates = {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }
    missing_set = set(missing)
    winners: list[list[tuple[int, int]]] = []
    for side_coords in side_candidates.values():
        side_set = set(side_coords)
        if not missing_set <= side_set:
            continue
        if _is_contiguous_on_line(missing):
            winners.append(missing)
    assert len(winners) == 1, "gap must lie on exactly one rectangle side"
    return winners[0], bbox


def _is_contiguous_on_line(coords: list[tuple[int, int]]) -> bool:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    same_row = len(rows) == 1 and cols == list(range(cols[0], cols[0] + len(cols)))
    same_col = len(cols) == 1 and rows == list(range(rows[0], rows[0] + len(rows)))
    return same_row or same_col


def _find_patterns(
    right: np.ndarray, bg: int, sep_color: int, col_offset: int
) -> dict[int, dict[str, object]]:
    patterns: dict[int, dict[str, object]] = {}
    for color in np.unique(right):
        color = int(color)
        if color in {bg, sep_color}:
            continue
        comps = _connected_components(right == color)
        assert len(comps) == 1, "each pattern color should form exactly one component"
        local_coords = comps[0]
        patterns[color] = {
            "local_coords": local_coords,
            "global_coords": [(r, c + col_offset) for r, c in local_coords],
        }
    return patterns


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_rotations(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    cur = shape
    for _ in range(4):
        variants.add(cur)
        cur = _rotate_shape(cur)
    return variants


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in shape) + 1
    rotated = [(c, h - 1 - r) for r, c in shape]
    return _normalize(rotated)


# --------------- generator ---------------

GEN_RETRIES = 300
FRAME_COUNT_CHOICES = (2, 3, 4)
HALF_WIDTH_RANGE = (10, 14)
GRID_H_BASE = 10
GRID_H_PER_FRAME = 4
GRID_H_MAX = 28
FRAME_H_RANGE = (5, 8)
FRAME_W_RANGE = (5, 8)
GAP_LENGTH_CHOICES = (2, 3, 4)
SIDES = ("top", "bottom", "left", "right")
COLOR_POOL = list(range(10))
PLACEMENT_RETRIES = 80


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    half_w = random.randint(HALF_WIDTH_RANGE[0], HALF_WIDTH_RANGE[1])
    h_min = GRID_H_BASE + GRID_H_PER_FRAME * n_frames
    h = random.randint(h_min, GRID_H_MAX)
    w = half_w * 2 + 1

    if max(h, w) > 30:
        return None

    palette_size = n_frames + 2
    if palette_size > len(COLOR_POOL):
        return None
    bg, sep_color, *frame_colors = random.sample(COLOR_POOL, palette_size)

    grid = np.full((h, w), bg, dtype=int)
    grid[:, half_w] = sep_color

    left_occ = np.zeros((h, half_w), dtype=bool)
    frame_specs: list[dict[str, object]] = []

    for color in frame_colors:
        spec = _place_frame(grid, left_occ, half_w, int(color))
        if spec is None:
            return None
        frame_specs.append(spec)

    right_occ = np.zeros((h, half_w), dtype=bool)
    order = frame_specs[:]
    random.shuffle(order)
    for spec in order:
        if not _place_pattern(grid, right_occ, half_w + 1, spec):
            return None

    if _most_common_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def _place_frame(
    grid: np.ndarray, occupied: np.ndarray, half_w: int, color: int
) -> dict[str, object] | None:
    h = grid.shape[0]
    for _ in range(PLACEMENT_RETRIES):
        side = random.choice(SIDES)
        gap_len = random.choice(GAP_LENGTH_CHOICES)

        min_h = max(FRAME_H_RANGE[0], gap_len + 2 if side in {"left", "right"} else FRAME_H_RANGE[0])
        min_w = max(FRAME_W_RANGE[0], gap_len + 2 if side in {"top", "bottom"} else FRAME_W_RANGE[0])
        max_h = min(FRAME_H_RANGE[1], h - 2)
        max_w = min(FRAME_W_RANGE[1], half_w - 2)
        if min_h > max_h or min_w > max_w:
            continue

        frame_h = random.randint(min_h, max_h)
        frame_w = random.randint(min_w, max_w)
        if frame_h - 2 < (gap_len if side in {"left", "right"} else 1):
            continue
        if frame_w - 2 < (gap_len if side in {"top", "bottom"} else 1):
            continue

        r0 = random.randint(1, h - frame_h - 1)
        c0 = random.randint(1, half_w - frame_w - 1)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        if not _bbox_is_clear(occupied, r0, c0, r1, c1, pad=1):
            continue

        coords = _rectangle_border(r0, c0, r1, c1)
        gap_coords = _pick_gap(r0, c0, r1, c1, side, gap_len)
        gap_set = set(gap_coords)
        for r, c in coords:
            if (r, c) not in gap_set:
                grid[r, c] = color

        _mark_bbox(occupied, r0, c0, r1, c1, pad=1)
        return {"color": color, "gap_coords": gap_coords}
    return None


def _rectangle_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _pick_gap(
    r0: int, c0: int, r1: int, c1: int, side: str, gap_len: int
) -> list[tuple[int, int]]:
    if side == "top":
        start = random.randint(c0 + 1, c1 - gap_len)
        return [(r0, c) for c in range(start, start + gap_len)]
    if side == "bottom":
        start = random.randint(c0 + 1, c1 - gap_len)
        return [(r1, c) for c in range(start, start + gap_len)]
    if side == "left":
        start = random.randint(r0 + 1, r1 - gap_len)
        return [(r, c0) for r in range(start, start + gap_len)]
    start = random.randint(r0 + 1, r1 - gap_len)
    return [(r, c1) for r in range(start, start + gap_len)]


def _bbox_is_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not np.any(mask[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _mark_bbox(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_pattern(
    grid: np.ndarray, occupied: np.ndarray, col_offset: int, spec: dict[str, object]
) -> bool:
    gap_coords = spec["gap_coords"]
    assert isinstance(gap_coords, list)
    shape = _normalize(gap_coords)
    ph = max(r for r, _ in shape) + 1
    pw = max(c for _, c in shape) + 1
    color = int(spec["color"])
    h, half_w = occupied.shape

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, half_w - pw - 1)
        if not _bbox_is_clear(occupied, r0, c0, r0 + ph - 1, c0 + pw - 1, pad=1):
            continue

        for dr, dc in shape:
            grid[r0 + dr, col_offset + c0 + dc] = color
        _mark_bbox(occupied, r0, c0, r0 + ph - 1, c0 + pw - 1, pad=1)
        return True
    return False
