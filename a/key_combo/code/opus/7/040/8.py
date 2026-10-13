from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3, 4, 6)
MARKER_COLOR = 7
FILL_COLOR = 8
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in FOUR_NEIGHBORS:
        yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(color) for color in np.unique(grid) if int(color) != BG):
        for coords in _component_coords(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            assert height >= 3 and width >= 3, "expected rectangular frames with non-empty interiors"

            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((height, width), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True

            assert int(coords.shape[0]) == int(border.sum()), "expected a one-pixel-thick frame"
            assert np.all(sub[border] == color), "expected a complete monochrome border"
            assert np.all(sub[~border] == BG), "expected frame interior to be background"

            frames.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "interior_area": int((height - 2) * (width - 2)),
                }
            )
    return frames


def _paint_largest_frame(half: np.ndarray) -> np.ndarray:
    frames = _extract_frames(half)
    assert frames, "expected at least one frame in the retained half"
    max_area = max(frame["interior_area"] for frame in frames)
    winners = [frame for frame in frames if frame["interior_area"] == max_area]
    assert len(winners) == 1, "expected a unique largest frame interior"

    out = half.copy()
    r0, c0, r1, c1 = winners[0]["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = FILL_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop to the marker-selected half and repaint only the largest hollow frame interior.

    Input structure
    - The background is black.
    - Exactly two orange marker pixels appear in one half of the grid, aligned either in one row or one column.
    - Each half contains separated monochrome hollow rectangular frames whose interiors are black.

    Rule
    - If the orange markers share a row, keep the left or right half containing them; if they share a column, keep the top or bottom half containing them.
    - Erase the orange markers to black inside the retained half.
    - Among the hollow rectangular frames in that retained half, find the unique one with the largest enclosed interior area.
    - Repaint only that winning frame's strict interior to teal while preserving every frame border.

    Key insight
    - The markers first restrict the search to one half of the canvas, and only inside that cropped half does the largest enclosed frame determine where the new fill appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    markers = np.argwhere(g == MARKER_COLOR)
    assert markers.shape == (2, 2), "expected exactly two orange markers"

    same_row = int(markers[0, 0]) == int(markers[1, 0])
    same_col = int(markers[0, 1]) == int(markers[1, 1])
    assert same_row != same_col, "expected markers aligned in exactly one axis"

    height, width = g.shape
    if same_row:
        assert width % 2 == 0, "left/right selection requires an even width"
        half_w = width // 2
        cols = markers[:, 1]
        if np.all(cols < half_w):
            kept = g[:, :half_w].copy()
        else:
            assert np.all(cols >= half_w), "row-aligned markers must lie in one horizontal half"
            kept = g[:, half_w:].copy()
    else:
        assert height % 2 == 0, "top/bottom selection requires an even height"
        half_h = height // 2
        rows = markers[:, 0]
        if np.all(rows < half_h):
            kept = g[:half_h, :].copy()
        else:
            assert np.all(rows >= half_h), "column-aligned markers must lie in one vertical half"
            kept = g[half_h:, :].copy()

    kept[kept == MARKER_COLOR] = BG
    return _paint_largest_frame(kept)


def _frame_candidates(height: int, width: int) -> dict[int, list[tuple[int, int]]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    max_h = min(height, 8)
    max_w = min(width, 8)
    for frame_h in range(4, max_h + 1):
        for frame_w in range(4, max_w + 1):
            area = (frame_h - 2) * (frame_w - 2)
            candidates.setdefault(area, []).append((frame_h, frame_w))
    return candidates


def _feasible_locations(occupied: np.ndarray, frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    height, width = occupied.shape
    locations: list[tuple[int, int]] = []
    for r in range(height - frame_h + 1):
        for c in range(width - frame_w + 1):
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(height, r + frame_h + 1)
            c1 = min(width, c + frame_w + 1)
            if not occupied[r0:r1, c0:c1].any():
                locations.append((r, c))
    return locations


def _draw_frame(grid: np.ndarray, r: int, c: int, frame_h: int, frame_w: int, color: int) -> None:
    grid[r, c : c + frame_w] = color
    grid[r + frame_h - 1, c : c + frame_w] = color
    grid[r : r + frame_h, c] = color
    grid[r : r + frame_h, c + frame_w - 1] = color


def _generate_half(height: int, width: int, frame_count: int) -> tuple[np.ndarray, list[dict]]:
    area_to_dims = _frame_candidates(height, width)
    viable_areas = [area for area in area_to_dims if area >= 4]
    if len(viable_areas) < frame_count:
        raise ValueError("not enough frame size variety for this half")

    for _ in range(200):
        chosen_areas = random.sample(viable_areas, frame_count)
        colors = random.sample(FRAME_COLORS, frame_count)
        frames = []
        for area, color in zip(chosen_areas, colors):
            frame_h, frame_w = random.choice(area_to_dims[area])
            frames.append({"h": frame_h, "w": frame_w, "color": color, "interior_area": area})
        frames.sort(key=lambda frame: frame["h"] * frame["w"], reverse=True)

        grid = np.zeros((height, width), dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        placed: list[dict] = []
        ok = True
        for frame in frames:
            locations = _feasible_locations(occupied, frame["h"], frame["w"])
            if not locations:
                ok = False
                break
            r, c = random.choice(locations)
            _draw_frame(grid, r, c, frame["h"], frame["w"], frame["color"])
            occupied[r : r + frame["h"], c : c + frame["w"]] = True
            placed.append(
                {
                    "color": frame["color"],
                    "bbox": (r, c, r + frame["h"] - 1, c + frame["w"] - 1),
                    "interior_area": frame["interior_area"],
                }
            )
        if ok:
            return grid, placed
    raise RuntimeError("failed to place hollow frames in half-grid")


def _isolated_bg_cells(grid: np.ndarray, row_range: range, col_range: range) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    height, width = grid.shape
    for r in row_range:
        for c in col_range:
            if grid[r, c] != BG:
                continue
            ok = True
            for nr in range(max(0, r - 1), min(height, r + 2)):
                for nc in range(max(0, c - 1), min(width, c + 2)):
                    if grid[nr, nc] != BG:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cells.append((r, c))
    return cells


def _place_markers(
    grid: np.ndarray,
    axis: str,
    chosen_side: int,
    half_h: int,
    half_w: int,
) -> bool:
    if axis == "lr":
        row_range = range(0, half_h)
        col_range = range(0, half_w) if chosen_side == 0 else range(half_w, 2 * half_w)
        cells = _isolated_bg_cells(grid, row_range, col_range)
        by_row: dict[int, list[int]] = {}
        for r, c in cells:
            by_row.setdefault(r, []).append(c)
        pairs = [
            ((r, c1), (r, c2))
            for r, cols in by_row.items()
            for i, c1 in enumerate(cols)
            for c2 in cols[i + 1 :]
            if c2 - c1 >= 2
        ]
    else:
        row_range = range(0, half_h) if chosen_side == 0 else range(half_h, 2 * half_h)
        col_range = range(0, half_w)
        cells = _isolated_bg_cells(grid, row_range, col_range)
        by_col: dict[int, list[int]] = {}
        for r, c in cells:
            by_col.setdefault(c, []).append(r)
        pairs = [
            ((r1, c), (r2, c))
            for c, rows in by_col.items()
            for i, r1 in enumerate(rows)
            for r2 in rows[i + 1 :]
            if r2 - r1 >= 2
        ]

    if not pairs:
        return False
    a, b = random.choice(pairs)
    grid[a] = MARKER_COLOR
    grid[b] = MARKER_COLOR
    return True


def _expected_output(half: np.ndarray, frames: list[dict]) -> np.ndarray:
    winner = max(frames, key=lambda frame: frame["interior_area"])
    out = half.copy()
    r0, c0, r1, c1 = winner["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = FILL_COLOR
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    axis = random.choice(("lr", "tb"))
    chosen_side = random.randint(0, 1)
    frame_count = random.choice((2, 3))

    if axis == "lr":
        half_h = random.randint(10, 14)
        half_w = random.randint(8, 12)
    else:
        half_h = random.randint(8, 11)
        half_w = random.randint(12, 18)

    try:
        selected_half, selected_frames = _generate_half(half_h, half_w, frame_count)
        other_half, _ = _generate_half(half_h, half_w, frame_count)
    except RuntimeError:
        return None

    expected = _expected_output(selected_half, selected_frames)
    if axis == "lr":
        halves = (selected_half, other_half) if chosen_side == 0 else (other_half, selected_half)
        grid = np.concatenate(halves, axis=1)
    else:
        halves = (selected_half, other_half) if chosen_side == 0 else (other_half, selected_half)
        grid = np.concatenate(halves, axis=0)

    if not _place_markers(grid, axis, chosen_side, half_h, half_w):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 8")
