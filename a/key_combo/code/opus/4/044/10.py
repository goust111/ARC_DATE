from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4)
MAX_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _is_domino(coords: np.ndarray) -> bool:
    if len(coords) != 2:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    return (r1 - r0 + 1, c1 - c0 + 1) in {(1, 2), (2, 1)}


def _is_hollow_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    if len(coords) < 12:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return False
    mask = _frame_mask(grid.shape, (r0, c0, r1, c1))
    return np.array_equal(np.argwhere(mask & (grid == color)), coords[np.lexsort((coords[:, 1], coords[:, 0]))])


def _parse_objects(grid: np.ndarray) -> tuple[int, dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    wall_row = grid.shape[0] - 1
    wall_color = int(grid[wall_row, 0])
    assert wall_color == WALL_COLOR
    assert np.all(grid[wall_row, :] == wall_color), "bottom wall must occupy the full last row"
    assert not np.any(grid[:wall_row, :] == wall_color), "wall color must be reserved for the bottom wall"

    frames: dict[int, tuple[int, int, int, int]] = {}
    dominoes: dict[int, np.ndarray] = {}
    for color in FRAME_COLORS:
        comps = _components_of_color(grid, color)
        for coords in comps:
            if _is_domino(coords):
                assert color not in dominoes, "each color must have exactly one domino"
                dominoes[color] = coords
            else:
                assert _is_hollow_frame(grid, color, coords), "non-domino foreground object must be a hollow frame"
                assert color not in frames, "each color must have exactly one frame"
                frames[color] = _bbox_coords(coords)

    assert 3 <= len(frames) <= 4, "expected 3 or 4 frames"
    assert set(frames) == set(dominoes), "frame colors and domino colors must match one-to-one"
    return wall_row, frames, dominoes


def _expected_from_specs(
    grid: np.ndarray,
    frames: dict[int, tuple[int, int, int, int]],
    dominoes: dict[int, np.ndarray],
    wall_row: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for coords in dominoes.values():
        out[coords[:, 0], coords[:, 1]] = BG
    for color, coords in dominoes.items():
        bottom = int(coords[:, 0].max())
        depth = int(wall_row - bottom)
        r0, c0, r1, c1 = frames[color]
        inner_h = r1 - r0 - 1
        assert 1 <= depth <= inner_h, "domino distance must fit inside its frame interior"
        out[r1 - depth : r1, c0 + 1 : c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each same-color domino's distance from the bottom wall to set the fill depth of its matching hollow frame.

    Input structure:
    - Black background with a full gray wall on the last row.
    - The upper part contains 3 or 4 one-cell-thick hollow rectangular frames in distinct colors.
    - The lower part contains 3 or 4 same-color dominoes, one per frame color.

    Rule:
    - Keep the bottom wall unchanged.
    - Pair each domino with the hollow frame of the same color.
    - Let D be the wall row index minus the domino's lowest occupied row index.
    - Fill the matching frame's interior from its bottom interior row upward for exactly D rows, using the frame color.
    - Erase every domino to black and leave frame borders unchanged.

    Key insight:
    - A domino is only a distance-encoding token: its color selects the target frame, and its vertical offset from the wall becomes that frame's fill height.
    """
    g = np.asarray(grid, dtype=int)
    wall_row, frames, dominoes = _parse_objects(g)
    return _expected_from_specs(g, frames, dominoes, wall_row)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = np.where(patch != BG, patch, dst[r0 : r0 + h, c0 : c0 + w])


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = occupied.shape
    rows, cols = np.nonzero(patch != BG)
    for pr, pc in zip(rows.tolist(), cols.tolist(), strict=False):
        rr = r0 + pr
        cc = c0 + pc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r_lo = max(0, rr - padding)
        r_hi = min(h, rr + padding + 1)
        c_lo = max(0, cc - padding)
        c_hi = min(w, cc + padding + 1)
        if occupied[r_lo:r_hi, c_lo:c_hi].any():
            return False
    return True


def _mark_occupied(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    rows, cols = np.nonzero(patch != BG)
    occupied[r0 + rows, c0 + cols] = True


def _frame_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _domino_patch(color: int, vertical: bool) -> np.ndarray:
    return np.full((2, 1) if vertical else (1, 2), color, dtype=int)


def _assign_depths(inner_heights: list[int]) -> list[int] | None:
    order = list(range(len(inner_heights)))
    random.shuffle(order)
    chosen = [0] * len(inner_heights)

    def backtrack(pos: int, used: set[int]) -> bool:
        if pos == len(order):
            has_full = any(chosen[i] == inner_heights[i] for i in range(len(chosen)))
            has_partial = any(chosen[i] < inner_heights[i] for i in range(len(chosen)))
            return has_full and has_partial
        idx = order[pos]
        options = list(range(1, inner_heights[idx] + 1))
        random.shuffle(options)
        for depth in options:
            if depth in used:
                continue
            chosen[idx] = depth
            used.add(depth)
            if backtrack(pos + 1, used):
                return True
            used.remove(depth)
        chosen[idx] = 0
        return False

    return chosen if backtrack(0, set()) else None


def _pair_by_column_output(
    grid: np.ndarray,
    frame_meta: list[dict[str, int]],
    domino_meta: list[dict[str, int | bool]],
    wall_row: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for item in domino_meta:
        patch = _domino_patch(int(item["color"]), bool(item["vertical"]))
        r0 = int(item["row"]) - (1 if bool(item["vertical"]) else 0)
        out[r0 : r0 + patch.shape[0], int(item["col"]) : int(item["col"]) + patch.shape[1]] = BG
    frames_sorted = sorted(frame_meta, key=lambda item: int(item["col"]) + int(item["width"]) / 2.0)
    dominoes_sorted = sorted(domino_meta, key=lambda item: int(item["col"]) + (0.0 if bool(item["vertical"]) else 0.5))
    for frame, dom in zip(frames_sorted, dominoes_sorted, strict=True):
        depth = int(wall_row - int(dom["row"]))
        r0 = int(frame["row"])
        c0 = int(frame["col"])
        h = int(frame["height"])
        w = int(frame["width"])
        out[r0 + h - 1 - depth : r0 + h - 1, c0 + 1 : c0 + w - 1] = int(frame["color"])
    return out


def _top_cell_distance_output(
    grid: np.ndarray,
    frame_meta: list[dict[str, int]],
    domino_meta: list[dict[str, int | bool]],
    wall_row: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for item in domino_meta:
        color = int(item["color"])
        patch = _domino_patch(color, bool(item["vertical"]))
        r0 = int(item["row"]) - (1 if bool(item["vertical"]) else 0)
        c0 = int(item["col"])
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = BG
        frame = next(frame for frame in frame_meta if int(frame["color"]) == color)
        top_row = r0
        depth = int(wall_row - top_row)
        r_start = int(frame["row"]) + int(frame["height"]) - 1 - depth
        r_end = int(frame["row"]) + int(frame["height"]) - 1
        out[max(r_start, int(frame["row"]) + 1) : r_end, int(frame["col"]) + 1 : int(frame["col"]) + int(frame["width"]) - 1] = color
    return out


def _build_expected_from_meta(
    grid: np.ndarray,
    frame_meta: list[dict[str, int]],
    domino_meta: list[dict[str, int | bool]],
    wall_row: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    by_color = {int(item["color"]): item for item in domino_meta}
    for item in domino_meta:
        patch = _domino_patch(int(item["color"]), bool(item["vertical"]))
        r0 = int(item["row"]) - (1 if bool(item["vertical"]) else 0)
        c0 = int(item["col"])
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = BG
    for frame in frame_meta:
        dom = by_color[int(frame["color"])]
        depth = int(wall_row - int(dom["row"]))
        r0 = int(frame["row"])
        c0 = int(frame["col"])
        h = int(frame["height"])
        w = int(frame["width"])
        out[r0 + h - 1 - depth : r0 + h - 1, c0 + 1 : c0 + w - 1] = int(frame["color"])
    return out


def _make_instance() -> tuple[np.ndarray, np.ndarray, list[dict[str, int]], list[dict[str, int | bool]]]:
    height = random.randint(20, 24)
    width = random.randint(20, 26)
    wall_row = height - 1
    grid = np.zeros((height, width), dtype=int)
    grid[wall_row, :] = WALL_COLOR
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[wall_row, :] = True

    frame_count = random.choice((3, 4))
    colors = random.sample(FRAME_COLORS, frame_count)

    if frame_count == 3:
        inner_heights = random.choice(([2, 3, 3], [2, 4, 4], [3, 3, 4], [3, 4, 5]))
    else:
        inner_heights = random.choice(([2, 3, 4, 4], [2, 3, 4, 5], [3, 3, 4, 5], [2, 4, 4, 5]))
    inner_heights = list(inner_heights)
    random.shuffle(inner_heights)

    frame_widths = [random.choice((4, 5, 6, 7)) for _ in range(frame_count)]
    frame_dims = list(zip([inner_h + 2 for inner_h in inner_heights], frame_widths, strict=True))

    frame_meta: list[dict[str, int]] = []
    top_row = random.randint(1, 2)
    bottom_row = random.randint(7, 8)
    mid = width // 2
    slots = [
        (top_row, 1, mid - 2),
        (top_row, mid + 1, width - 2),
        (bottom_row, 1, mid - 2),
        (bottom_row, mid + 1, width - 2),
    ]
    chosen_slots = random.sample(slots, frame_count)

    for color, (fh, fw), (r0, c_min, c_max) in zip(colors, frame_dims, chosen_slots, strict=True):
        if c_max - c_min + 1 < fw:
            raise ValueError("frame slot too narrow")
        patch = _frame_patch(fh, fw, color)
        candidates = [c0 for c0 in range(c_min, c_max - fw + 2) if _can_place(occupied, patch, r0, c0, padding=1)]
        if not candidates or r0 + fh >= wall_row - 5:
            raise ValueError("failed to place frame")
        c0 = random.choice(candidates)
        _paste(grid, patch, r0, c0)
        _mark_occupied(occupied, patch, r0, c0)
        frame_meta.append({"color": int(color), "row": r0, "col": c0, "height": fh, "width": fw})

    depths = _assign_depths(inner_heights)
    if depths is None:
        raise ValueError("failed to assign unique depths")

    vertical_partial_idx = [i for i, depth in enumerate(depths) if depth < inner_heights[i]]
    if not vertical_partial_idx:
        raise ValueError("need at least one partial fill")
    vertical_partial_pick = random.choice(vertical_partial_idx)

    orientations: list[bool | None] = [None] * frame_count
    orientations[vertical_partial_pick] = True
    others = [i for i in range(frame_count) if i != vertical_partial_pick]
    horizontal_pick = random.choice(others)
    orientations[horizontal_pick] = False
    for i in range(frame_count):
        if orientations[i] is None:
            orientations[i] = random.choice((False, True))

    domino_meta: list[dict[str, int | bool]] = []
    color_to_frame = {int(item["color"]): item for item in frame_meta}
    frame_order = [int(item["color"]) for item in sorted(frame_meta, key=lambda item: int(item["col"]) + int(item["width"]) / 2.0)]
    domino_order = frame_order.copy()
    while domino_order == frame_order:
        random.shuffle(domino_order)
    slot_cols = np.linspace(1, width - 3, frame_count, dtype=int).tolist()
    for slot_col, color in zip(slot_cols, domino_order, strict=True):
        idx = colors.index(color)
        depth = depths[idx]
        vertical = bool(orientations[idx])
        bottom_row = wall_row - depth
        patch = _domino_patch(color, vertical)
        r0 = bottom_row - (1 if vertical else 0)
        c0 = min(int(slot_col), width - patch.shape[1] - 1)
        if not _can_place(occupied, patch, r0, c0, padding=1):
            raise ValueError("failed to place domino")
        _paste(grid, patch, r0, c0)
        _mark_occupied(occupied, patch, r0, c0)
        frame = color_to_frame[int(color)]
        domino_meta.append(
            {
                "color": int(color),
                "row": int(bottom_row),
                "col": int(c0),
                "vertical": bool(vertical),
                "depth": int(depth),
                "frame_inner_h": int(frame["height"] - 2),
            }
        )

    expected = _build_expected_from_meta(grid, frame_meta, domino_meta, wall_row)
    return grid, expected, frame_meta, domino_meta


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    frame_meta: list[dict[str, int]],
    domino_meta: list[dict[str, int | bool]],
) -> bool:
    h, w = grid.shape
    wall_row = h - 1
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.all(grid[wall_row, :] == WALL_COLOR):
        return False
    if len(frame_meta) != len(domino_meta) or len(frame_meta) not in (3, 4):
        return False

    frame_colors = {int(item["color"]) for item in frame_meta}
    domino_colors = {int(item["color"]) for item in domino_meta}
    if frame_colors != domino_colors:
        return False

    depths = [int(item["depth"]) for item in domino_meta]
    if len(set(depths)) != len(depths):
        return False
    if not any(int(item["depth"]) == int(item["frame_inner_h"]) for item in domino_meta):
        return False
    if not any(int(item["depth"]) < int(item["frame_inner_h"]) for item in domino_meta):
        return False
    if not any(bool(item["vertical"]) and int(item["depth"]) < int(item["frame_inner_h"]) for item in domino_meta):
        return False
    if not any(bool(item["vertical"]) for item in domino_meta):
        return False
    if not any(not bool(item["vertical"]) for item in domino_meta):
        return False

    frame_order = [int(item["color"]) for item in sorted(frame_meta, key=lambda item: int(item["col"]) + int(item["width"]) / 2.0)]
    domino_order = [int(item["color"]) for item in sorted(domino_meta, key=lambda item: int(item["col"]) + (0.0 if bool(item["vertical"]) else 0.5))]
    if frame_order == domino_order:
        return False

    if np.array_equal(_pair_by_column_output(grid, frame_meta, domino_meta, wall_row), expected):
        return False
    if np.array_equal(_top_cell_distance_output(grid, frame_meta, domino_meta, wall_row), expected):
        return False

    parsed_expected = main(grid)
    if not np.array_equal(parsed_expected, expected):
        return False

    if np.array_equal(grid, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected, frame_meta, domino_meta = _make_instance()
        except ValueError:
            continue
        if _validate_instance(grid, expected, frame_meta, domino_meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
