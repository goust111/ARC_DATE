from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

SIDE_ORDER = ("top", "left", "bottom", "right")
SIDE_COLORS = (GREEN, YELLOW, BLUE, ORANGE)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    n_rows, n_cols = mask.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < n_rows and 0 <= cc < n_cols and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _distances_to_sides(r: int, c: int, n: int) -> list[int]:
    return [int(r), int(c), int(n - 1 - r), int(n - 1 - c)]


def _recolor_interior(interior: np.ndarray) -> np.ndarray:
    n = int(interior.shape[0])
    out = interior.copy()
    for r, c in np.argwhere(interior == GREY):
        rr = int(r)
        cc = int(c)
        distances = _distances_to_sides(rr, cc, n)
        nearest = min(distances)
        if distances.count(nearest) == 1:
            out[rr, cc] = SIDE_COLORS[distances.index(nearest)]
        else:
            out[rr, cc] = GREY
    return out


def _render_output_from_mask(mask: np.ndarray) -> np.ndarray:
    n = int(mask.shape[0])
    interior = np.where(mask, GREY, BLACK)
    recolored = _recolor_interior(interior)
    out = np.full((n + 2, n + 2), BLACK, dtype=int)
    out[0, 1:-1] = GREEN
    out[-1, 1:-1] = BLUE
    out[1:-1, 0] = YELLOW
    out[1:-1, -1] = ORANGE
    out[1:-1, 1:-1] = recolored
    return out


def _classification_counts(mask: np.ndarray) -> dict[str, int]:
    n = int(mask.shape[0])
    counts = {side: 0 for side in SIDE_ORDER}
    counts["tie"] = 0
    for r, c in np.argwhere(mask):
        distances = _distances_to_sides(int(r), int(c), n)
        nearest = min(distances)
        if distances.count(nearest) == 1:
            counts[SIDE_ORDER[distances.index(nearest)]] += 1
        else:
            counts["tie"] += 1
    return counts


def _parse_control(grid: np.ndarray) -> tuple[int, np.ndarray]:
    assert grid.ndim == 2 and grid.shape[1] >= 2, "expected a control column plus main region"
    control = grid[:, 0]
    red_rows = np.flatnonzero(control == RED)
    assert len(red_rows) == 1, "expected exactly one red control marker"
    assert np.all((control == PINK) | (control == RED)), "control column must contain only pink and one red marker"
    return int(red_rows[0]), np.asarray(grid[:, 1:], dtype=int).copy()


def _is_consecutive(values: np.ndarray) -> bool:
    vals = np.sort(np.asarray(values, dtype=int))
    return bool(len(vals) > 0 and np.array_equal(vals, np.arange(int(vals[0]), int(vals[-1]) + 1)))


def _detect_frame(core: np.ndarray) -> tuple[int, int, int]:
    top_coords = np.argwhere(core == GREEN)
    bottom_coords = np.argwhere(core == BLUE)
    left_coords = np.argwhere(core == YELLOW)
    right_coords = np.argwhere(core == ORANGE)
    assert all(arr.size > 0 for arr in (top_coords, bottom_coords, left_coords, right_coords)), "expected all four frame sides"

    top_row = np.unique(top_coords[:, 0])
    bottom_row = np.unique(bottom_coords[:, 0])
    left_col = np.unique(left_coords[:, 1])
    right_col = np.unique(right_coords[:, 1])
    assert len(top_row) == len(bottom_row) == len(left_col) == len(right_col) == 1, "each frame side must be a straight bar"

    tr = int(top_row[0])
    br = int(bottom_row[0])
    lc = int(left_col[0])
    rc = int(right_col[0])
    assert tr < br and lc < rc, "invalid frame geometry"

    inner_r0 = tr + 1
    inner_r1 = br - 1
    inner_c0 = lc + 1
    inner_c1 = rc - 1
    n = inner_r1 - inner_r0 + 1
    assert n >= 2 and inner_c1 - inner_c0 + 1 == n, "expected an n x n frame interior"

    assert _is_consecutive(top_coords[:, 1]) and _is_consecutive(bottom_coords[:, 1]), "top/bottom sides must be contiguous"
    assert _is_consecutive(left_coords[:, 0]) and _is_consecutive(right_coords[:, 0]), "left/right sides must be contiguous"
    assert int(top_coords[:, 1].min()) == inner_c0 and int(top_coords[:, 1].max()) == inner_c1
    assert int(bottom_coords[:, 1].min()) == inner_c0 and int(bottom_coords[:, 1].max()) == inner_c1
    assert int(left_coords[:, 0].min()) == inner_r0 and int(left_coords[:, 0].max()) == inner_r1
    assert int(right_coords[:, 0].min()) == inner_r0 and int(right_coords[:, 0].max()) == inner_r1

    assert core[tr, lc] == BLACK and core[tr, rc] == BLACK and core[br, lc] == BLACK and core[br, rc] == BLACK
    return inner_r0, inner_c0, n


def _mask_is_irregular(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _candidate_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8, copy=False).tobytes()


def _all_shift_outputs_differ(mask: np.ndarray, target_shift: int) -> bool:
    target = _render_output_from_mask(np.roll(mask, int(target_shift), axis=0))
    for shift in range(1, int(mask.shape[0])):
        if shift == int(target_shift):
            continue
        other = _render_output_from_mask(np.roll(mask, shift, axis=0))
        if np.array_equal(other, target):
            return False
    return True


def _mask_uses_required_branches(mask: np.ndarray, shift: int) -> bool:
    rolled = np.roll(mask, int(shift), axis=0)
    counts = _classification_counts(rolled)
    if not all(counts[side] > 0 for side in SIDE_ORDER):
        return False
    if counts["tie"] <= 0:
        return False
    if np.array_equal(mask, rolled):
        return False
    if np.array_equal(_render_output_from_mask(mask), _render_output_from_mask(rolled)):
        return False
    return _all_shift_outputs_differ(mask, shift)


@lru_cache(maxsize=None)
def _candidate_masks(n: int, shift: int) -> tuple[np.ndarray, ...]:
    candidates: list[np.ndarray] = []
    seen: set[bytes] = set()

    for center_r in range(n):
        for center_c in range(n):
            for up in range(1, center_r + 1):
                for down in range(1, n - center_r):
                    for left in range(1, center_c + 1):
                        for right in range(1, n - center_c):
                            base = np.zeros((n, n), dtype=bool)
                            base[center_r - up : center_r + down + 1, center_c] = True
                            base[center_r, center_c - left : center_c + right + 1] = True
                            if not (6 <= int(base.sum()) <= 12):
                                continue

                            extra_cells: list[tuple[int, int] | None] = [None]
                            adjacent: set[tuple[int, int]] = set()
                            for r, c in np.argwhere(base):
                                rr = int(r)
                                cc = int(c)
                                for nr, nc in _neighbors4(rr, cc):
                                    if 0 <= nr < n and 0 <= nc < n and not bool(base[nr, nc]):
                                        adjacent.add((nr, nc))
                            extra_cells.extend(sorted(adjacent))

                            for extra in extra_cells:
                                mask = base.copy()
                                if extra is not None:
                                    if int(mask.sum()) >= 12:
                                        continue
                                    mask[extra] = True
                                if not _is_connected(mask):
                                    continue
                                if not _mask_is_irregular(mask):
                                    continue
                                if not _mask_uses_required_branches(mask, shift):
                                    continue

                                signature = _candidate_signature(mask)
                                if signature in seen:
                                    continue
                                seen.add(signature)
                                candidates.append(mask)

    assert candidates, f"no valid templates found for n={n}, shift={shift}"
    return tuple(candidates)


def _build_input_grid(mask: np.ndarray, shift: int, height: int, width: int, frame_top: int, frame_left: int) -> np.ndarray:
    n = int(mask.shape[0])
    frame_size = n + 2
    assert width >= frame_size + 2 and height >= frame_size + 2
    assert 1 <= frame_top <= height - frame_size - 1
    assert 1 <= frame_left <= width - frame_size - 1

    grid = np.full((height, width + 1), BLACK, dtype=int)
    grid[:, 0] = PINK
    grid[int(shift), 0] = RED
    core = grid[:, 1:]

    top = frame_top
    left = frame_left
    bottom = top + frame_size - 1
    right = left + frame_size - 1
    inner_r0 = top + 1
    inner_c0 = left + 1

    core[top, left + 1 : right] = GREEN
    core[bottom, left + 1 : right] = BLUE
    core[top + 1 : bottom, left] = YELLOW
    core[top + 1 : bottom, right] = ORANGE
    core[inner_r0 : inner_r0 + n, inner_c0 : inner_c0 + n] = np.where(mask, GREY, BLACK)
    return grid


def _instance_matches_spec(grid: np.ndarray) -> bool:
    shift, core = _parse_control(grid)
    inner_r0, inner_c0, n = _detect_frame(core)
    interior = _crop_bbox(core, (inner_r0, inner_c0, inner_r0 + n - 1, inner_c0 + n - 1))
    if not np.all(np.isin(interior, [BLACK, GREY])):
        return False
    mask = interior == GREY
    if not (6 <= int(mask.sum()) <= 12):
        return False
    if not _is_connected(mask):
        return False
    return _mask_uses_required_branches(mask, shift)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll the frame interior downward by the control-column offset, then recolor each rolled gray pixel by its unique nearest frame side.

    Input structure:
    - The leftmost column is a control column: all pink except for one red marker pixel.
    - To the right sits one hollow rectangular frame on a black background, with green top, blue bottom, yellow left, and orange right sides; the four corners stay black.
    - The frame encloses an n x n interior containing only black cells and a 4-connected gray foreground pattern.

    Rule:
    - Read the red marker row index r from the control column, then remove that column from the output.
    - Detect the frame interior and cyclically roll the entire n x n interior downward by r rows, wrapping bottom rows to the top.
    - Keep the frame fixed, and for each rolled gray pixel compare its distances to the interior's top, left, bottom, and right sides.
    - Recolor a gray pixel with the color of its uniquely nearest side; if the nearest distance is tied, keep that pixel gray.

    Key insight:
    - The control column does not mark a location inside the frame; it encodes the roll amount that changes every gray pixel's nearest-side ownership before recoloring.
    """
    shift, core = _parse_control(np.asarray(grid, dtype=int))
    inner_r0, inner_c0, n = _detect_frame(core)

    out = core.copy()
    interior = _crop_bbox(core, (inner_r0, inner_c0, inner_r0 + n - 1, inner_c0 + n - 1))
    rolled = np.roll(interior, int(shift) % n, axis=0)
    recolored = _recolor_interior(rolled)
    out[inner_r0 : inner_r0 + n, inner_c0 : inner_c0 + n] = recolored
    return out


def generate_input() -> np.ndarray:
    keys = [(n, shift) for n in range(5, 8) for shift in range(1, n)]
    n, shift = random.choice(keys)
    mask = random.choice(_candidate_masks(n, shift)).copy()

    frame_size = n + 2
    height = random.randint(max(10, frame_size + 2), min(13, frame_size + 5))
    width = random.randint(max(11, frame_size + 2), min(14, frame_size + 6))
    frame_top = random.randint(1, height - frame_size - 1)
    frame_left = random.randint(1, width - frame_size - 1)

    grid = _build_input_grid(mask, shift, height, width, frame_top, frame_left)
    assert _instance_matches_spec(grid), "generator produced a degenerate instance"
    expected = main(grid)

    core = grid[:, 1:]
    inner_r0, inner_c0, _ = _detect_frame(core)
    rolled_mask = np.roll(mask, shift, axis=0)
    target_patch = _render_output_from_mask(rolled_mask)
    observed_patch = core[frame_top : frame_top + frame_size, frame_left : frame_left + frame_size].copy()
    observed_patch[1:-1, 1:-1] = expected[inner_r0 : inner_r0 + n, inner_c0 : inner_c0 + n]
    assert np.array_equal(observed_patch, target_patch), "main() disagrees with generator semantics"
    return grid
