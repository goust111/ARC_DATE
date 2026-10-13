from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))

GRID_ROWS_RANGE = (14, 22)
GRID_COLS_RANGE = (14, 22)
FRAME_COUNT_OPTIONS = (3, 4)
GENERATION_RETRIES = 800
PLACEMENT_RETRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert 3 <= len(components) <= 4, "expected 3 or 4 hollow frames"

    frames: list[dict] = []
    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "frame must have a non-empty interior"

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert comp["area"] == int(border.sum()), "component must be a 1-cell-thick rectangular border"
        assert np.all(sub[border] == color), "frame border must be complete and monochrome"
        assert np.all(sub[~border] == bg), "frame interior must be background"

        frames.append(
            {
                "color": color,
                "bbox": comp["bbox"],
                "outer_height": height,
                "outer_width": width,
                "inner_area": (height - 2) * (width - 2),
                "perimeter": int(border.sum()),
            }
        )

    areas = [frame["inner_area"] for frame in frames]
    assert len(set(areas)) == len(areas), "expected pairwise distinct interior areas"
    return bg, frames


def _build_wrapped_sequence(width: int, bg: int, ordered_frames: list[dict]) -> np.ndarray:
    seq: list[int] = []
    for i, frame in enumerate(ordered_frames):
        seq.extend([int(frame["color"])] * int(frame["inner_area"]))
        if i != len(ordered_frames) - 1:
            seq.append(bg)

    out_h = math.ceil(len(seq) / width)
    assert out_h <= 30, "output exceeds ARC height limit"
    out = np.full((out_h, width), bg, dtype=int)
    out.ravel()[: len(seq)] = np.asarray(seq, dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile sorted hollow-frame interior areas into wrapped colored runs.

    Input structure:
    - The grid has a black background and 3 or 4 separate monochrome hollow rectangular frames.
    - Every frame is a complete 1-cell-thick rectangle with a background-colored interior.
    - Different frames have different interior areas, and their spatial positions are irrelevant.

    Rule:
    - Detect each hollow frame and measure its interior area as (width - 2) x (height - 2).
    - Sort the frames by interior area from smallest to largest.
    - For each sorted frame, emit one run whose length equals that interior area and whose color equals the frame color, inserting one background cell between adjacent runs.
    - Wrap the resulting 1D sequence row-major into rows whose width equals the input width, padding the last row with background.

    Key insight:
    - The output order is driven by a hidden geometric quantity, the empty area inside each frame, while the output colors come from the frame borders themselves.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _extract_frames(g)
    ordered = sorted(frames, key=lambda frame: frame["inner_area"])
    return _build_wrapped_sequence(g.shape[1], bg, ordered)


def _boxes_separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0 = int(frame["r"])
    c0 = int(frame["c"])
    h = int(frame["outer_height"])
    w = int(frame["outer_width"])
    color = int(frame["color"])
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _sample_frame_specs(rows: int, cols: int, n_frames: int, require_perimeter_conflict: bool) -> list[dict] | None:
    max_dim = min(8 if n_frames == 3 else 7, rows - 2, cols - 2)
    if max_dim < 4:
        return None

    pool: list[dict] = []
    for outer_h in range(3, max_dim + 1):
        for outer_w in range(3, max_dim + 1):
            inner_area = (outer_h - 2) * (outer_w - 2)
            if inner_area < 2:
                continue
            pool.append(
                {
                    "outer_height": outer_h,
                    "outer_width": outer_w,
                    "inner_area": inner_area,
                    "perimeter": 2 * (outer_h + outer_w) - 4,
                }
            )

    target_height_low, target_height_high = random.choice(((2, 3), (3, 5)))

    for _ in range(400):
        specs = [dict(item) for item in random.sample(pool, n_frames)]
        areas = [spec["inner_area"] for spec in specs]
        if len(set(areas)) != n_frames:
            continue
        if not any(spec["outer_height"] != spec["outer_width"] for spec in specs):
            continue
        if require_perimeter_conflict:
            area_order = [idx for idx, _ in sorted(enumerate(specs), key=lambda item: item[1]["inner_area"])]
            per_order = [idx for idx, _ in sorted(enumerate(specs), key=lambda item: item[1]["perimeter"])]
            if area_order == per_order:
                continue

        seq_len = sum(areas) + (n_frames - 1)
        out_h = math.ceil(seq_len / cols)
        if not (target_height_low <= out_h <= target_height_high):
            continue
        if out_h > min(8, rows):
            continue

        total_border_cells = sum(spec["perimeter"] for spec in specs)
        if total_border_cells >= rows * cols // 2:
            continue
        return specs
    return None


def _place_frames(rows: int, cols: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    order = sorted(specs, key=lambda spec: spec["outer_height"] * spec["outer_width"], reverse=True)

    for spec in order:
        h = int(spec["outer_height"])
        w = int(spec["outer_width"])
        positions = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
        random.shuffle(positions)
        chosen = None
        for r, c in positions:
            box = (r, c, r + h - 1, c + w - 1)
            if all(_boxes_separated(box, existing["bbox"]) for existing in placed):
                chosen = (r, c, box)
                break
        if chosen is None:
            return None
        r, c, box = chosen
        new_spec = dict(spec)
        new_spec["r"] = r
        new_spec["c"] = c
        new_spec["bbox"] = box
        placed.append(new_spec)

    random.shuffle(placed)
    return placed


def _frame_output_from_specs(width: int, specs: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(specs, key=lambda frame: frame["inner_area"])
    return _build_wrapped_sequence(width, bg, ordered)


def _valid_semantics(grid: np.ndarray, expected: np.ndarray, chosen_specs: list[dict]) -> bool:
    try:
        bg, detected = _extract_frames(grid)
    except AssertionError:
        return False

    if bg != BACKGROUND:
        return False
    if len(detected) != len(chosen_specs):
        return False

    detected_sig = sorted((f["color"], f["outer_height"], f["outer_width"], f["inner_area"]) for f in detected)
    chosen_sig = sorted(
        (int(f["color"]), int(f["outer_height"]), int(f["outer_width"]), int(f["inner_area"])) for f in chosen_specs
    )
    if detected_sig != chosen_sig:
        return False

    area_order = [frame["color"] for frame in sorted(detected, key=lambda frame: frame["inner_area"])]
    spatial_order = [frame["color"] for frame in sorted(detected, key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))]
    if area_order == spatial_order:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    seq_len = sum(frame["inner_area"] for frame in detected) + len(detected) - 1
    out_h = math.ceil(seq_len / grid.shape[1])
    return 2 <= out_h <= 8 and out_h == expected.shape[0]


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    n_frames = random.choice(FRAME_COUNT_OPTIONS)
    require_perimeter_conflict = random.random() < 0.6

    specs = _sample_frame_specs(rows, cols, n_frames, require_perimeter_conflict)
    if specs is None:
        return None

    colors = random.sample(COLOR_POOL, n_frames)
    for spec, color in zip(specs, colors):
        spec["color"] = color

    for _ in range(PLACEMENT_RETRIES):
        placed = _place_frames(rows, cols, specs)
        if placed is None:
            continue

        grid = np.full((rows, cols), BACKGROUND, dtype=int)
        for frame in placed:
            _draw_frame(grid, frame)

        expected = _frame_output_from_specs(cols, placed, BACKGROUND)
        if _valid_semantics(grid, expected, placed):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
