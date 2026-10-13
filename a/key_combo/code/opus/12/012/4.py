from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
CONTENT = 5
FRAME_COUNT = 3
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
VERTICAL_DIRS = {"up", "down"}

GRID_H_RANGE = (20, 26)
GRID_W_RANGE = (20, 26)
EXPANDED_H_RANGE = (7, 12)
EXPANDED_W_RANGE = (7, 12)
GEN_RETRIES = 500
PLACEMENT_RETRIES = 300
MIN_BOX_GAP = 1

BBox = tuple[int, int, int, int]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _border_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    grid[_border_mask(grid.shape, box)] = int(color)


def _full_frame(grid: np.ndarray, color: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return (
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _frame_candidates_for_color(grid: np.ndarray, color: int) -> list[BBox]:
    rows = np.flatnonzero(np.any(grid == color, axis=1))
    cols = np.flatnonzero(np.any(grid == color, axis=0))
    candidates: list[BBox] = []
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if int(r1) - int(r0) < 3:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if int(c1) - int(c0) < 3:
                        continue
                    box = (int(r0), int(c0), int(r1), int(c1))
                    if not _full_frame(grid, int(color), box):
                        continue
                    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                    border_cells = int(2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4)
                    if int(np.count_nonzero(patch == color)) != border_cells:
                        continue
                    candidates.append(box)
    return candidates


def _marker_direction(box: BBox, marker: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = box
    mr, mc = marker
    matches: list[str] = []
    if mr < r0 and c0 <= mc <= c1:
        matches.append("up")
    if mr > r1 and c0 <= mc <= c1:
        matches.append("down")
    if mc < c0 and r0 <= mr <= r1:
        matches.append("left")
    if mc > c1 and r0 <= mr <= r1:
        matches.append("right")
    if len(matches) != 1:
        return None
    return matches[0]


def _expand_box(box: BBox, marker: tuple[int, int]) -> BBox:
    r0, c0, r1, c1 = box
    mr, mc = marker
    direction = _marker_direction(box, marker)
    assert direction is not None
    if direction == "up":
        return mr, c0, r1, c1
    if direction == "down":
        return r0, c0, mr, c1
    if direction == "left":
        return r0, mc, r1, c1
    return r0, c0, r1, mc


def _interior_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(max(0, r1 - r0 - 1) * max(0, c1 - c0 - 1))


def _parse_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    records: list[dict[str, object]] = []
    seen_colors: set[int] = set()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        candidates = _frame_candidates_for_color(g, color)
        if not candidates:
            continue

        valid: list[dict[str, object]] = []
        coords = np.argwhere(g == color)
        cells = {(int(r), int(c)) for r, c in coords}
        for box in candidates:
            border = _border_mask(g.shape, box)
            extras = [cell for cell in cells if not border[cell]]
            if len(extras) != 1:
                continue
            marker = extras[0]
            direction = _marker_direction(box, marker)
            if direction is None:
                continue
            expanded = _expand_box(box, marker)
            valid.append(
                {
                    "color": int(color),
                    "box": box,
                    "marker": marker,
                    "direction": direction,
                    "expanded_box": expanded,
                    "inner": g[box[0] + 1 : box[2], box[1] + 1 : box[3]].copy(),
                }
            )

        assert len(valid) == 1, "each frame color must map to one frame and one marker"
        record = valid[0]
        assert int(record["color"]) not in seen_colors
        seen_colors.add(int(record["color"]))
        records.append(record)

    assert len(records) >= 1, "expected at least one stretchable frame"
    return records


def _winner(records: list[dict[str, object]]) -> dict[str, object]:
    return max(
        records,
        key=lambda record: (
            _interior_area(record["expanded_box"]),  # type: ignore[arg-type]
            -record["expanded_box"][0],  # type: ignore[index]
            -record["expanded_box"][1],  # type: ignore[index]
        ),
    )


def _crop_expanded_inner(grid: np.ndarray, record: dict[str, object], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = record["box"]  # type: ignore[assignment]
    nr0, nc0, nr1, nc1 = record["expanded_box"]  # type: ignore[assignment]
    out = np.full((nr1 - nr0 - 1, nc1 - nc0 - 1), bg, dtype=int)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
    off_r = r0 - nr0
    off_c = c0 - nc0
    out[off_r : off_r + inner.shape[0], off_c : off_c + inner.shape[1]] = inner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch every marked hollow frame, then crop the interior of the one whose stretched area becomes largest.

    Input structure:
    - The background is black.
    - Several different colors each form a one-pixel-thick hollow rectangular frame.
    - Each frame has exactly one same-colored singleton marker outside one side, aligned to that side's row or column span.
    - The strict interior of each frame may contain fixed content pixels on the background.

    Rule:
    - Pair every marker with the frame of the same color.
    - Stretch that frame outward toward the marker so the marker row or column becomes the new outer border.
    - Keep the original interior contents at their absolute positions; any newly created interior cells become background.
    - Compute every stretched frame's strict interior area and select the unique maximum, breaking ties by the top-left expanded bbox.
    - Output only that winning frame's strict interior, with the border removed.

    Key insight:
    - The target frame is not chosen from the input as drawn; it is chosen only after the marker-driven stretch changes each frame's area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = _most_common_color(g)
    records = _parse_frames(g)
    chosen = _winner(records)
    return _crop_expanded_inner(g, chosen, bg)


def _boxes_too_close(a: BBox, b: BBox, gap: int = MIN_BOX_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    return not separated


def _random_box(h: int, w: int, box_h: int, box_w: int) -> BBox:
    r0 = random.randint(1, h - box_h - 1)
    c0 = random.randint(1, w - box_w - 1)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _feasible_distance(expanded_box: BBox, direction: str, lo: int, hi: int) -> int | None:
    span = (expanded_box[2] - expanded_box[0] + 1) if direction in VERTICAL_DIRS else (expanded_box[3] - expanded_box[1] + 1)
    hi = min(int(hi), int(span - 4))
    if lo > hi:
        return None
    return random.randint(int(lo), int(hi))


def _shrink_box(expanded_box: BBox, direction: str, distance: int) -> tuple[BBox, tuple[int, int]]:
    r0, c0, r1, c1 = expanded_box
    if direction == "up":
        box = (r0 + distance, c0, r1, c1)
        marker = (r0, random.randint(c0, c1))
    elif direction == "down":
        box = (r0, c0, r1 - distance, c1)
        marker = (r1, random.randint(c0, c1))
    elif direction == "left":
        box = (r0, c0 + distance, r1, c1)
        marker = (random.randint(r0, r1), c0)
    else:
        box = (r0, c0, r1, c1 - distance)
        marker = (random.randint(r0, r1), c1)
    return box, marker


def _grow_blob(h: int, w: int, count: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < count:
        frontier: list[tuple[int, int]] = []
        for r, c in sorted(cells):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _inner_pattern(h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    area = h * w
    max_count = max(2, area - 1)
    count = random.randint(2, min(max_count, max(2, (2 * area) // 3)))
    for r, c in _grow_blob(h, w, count):
        patch[r, c] = CONTENT
    if area >= 6 and random.random() < 0.35:
        extra_r = random.randrange(h)
        extra_c = random.randrange(w)
        patch[extra_r, extra_c] = CONTENT
    if np.all(patch == CONTENT):
        patch[random.randrange(h), random.randrange(w)] = BG
    return patch


def _render_input(shape: tuple[int, int], records: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for record in records:
        box = record["box"]  # type: ignore[assignment]
        marker = record["marker"]  # type: ignore[assignment]
        color = int(record["color"])
        inner = record["inner"]  # type: ignore[assignment]
        _draw_frame(grid, box, color)
        r0, c0, r1, c1 = box
        grid[r0 + 1 : r1, c0 + 1 : c1] = inner
        grid[marker] = color
    return grid


def _expected_output(shape: tuple[int, int], records: list[dict[str, object]]) -> np.ndarray:
    dummy = np.zeros(shape, dtype=int)
    for record in records:
        box = record["box"]  # type: ignore[assignment]
        inner = record["inner"]  # type: ignore[assignment]
        r0, c0, r1, c1 = box
        dummy[r0 + 1 : r1, c0 + 1 : c1] = inner
    return _crop_expanded_inner(dummy, _winner(records), BG)


def _semantic_checks(grid: np.ndarray, design: list[dict[str, object]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        parsed = _parse_frames(grid)
    except AssertionError:
        return False

    if len(parsed) != FRAME_COUNT:
        return False

    designed_by_color = {int(record["color"]): record for record in design}
    parsed_by_color = {int(record["color"]): record for record in parsed}
    if set(designed_by_color) != set(parsed_by_color):
        return False

    transformed_areas = set()
    original_areas = set()
    for color, record in designed_by_color.items():
        parsed_record = parsed_by_color[color]
        if parsed_record["box"] != record["box"]:
            return False
        if parsed_record["marker"] != record["marker"]:
            return False
        if parsed_record["direction"] != record["direction"]:
            return False
        if parsed_record["expanded_box"] != record["expanded_box"]:
            return False
        if not np.array_equal(parsed_record["inner"], record["inner"]):
            return False
        transformed_areas.add(_interior_area(record["expanded_box"]))  # type: ignore[arg-type]
        original_areas.add(_interior_area(record["box"]))  # type: ignore[arg-type]

    if len(transformed_areas) != FRAME_COUNT or len(original_areas) != FRAME_COUNT:
        return False

    original_winner = max(
        design,
        key=lambda record: (
            _interior_area(record["box"]),  # type: ignore[arg-type]
            -record["box"][0],  # type: ignore[index]
            -record["box"][1],  # type: ignore[index]
        ),
    )
    transformed_winner = _winner(design)
    if int(original_winner["color"]) == int(transformed_winner["color"]):
        return False

    if len({str(record["direction"]) for record in design}) != FRAME_COUNT:
        return False

    for i, record in enumerate(design):
        for other in design[i + 1 :]:
            if _boxes_too_close(record["expanded_box"], other["expanded_box"], gap=0):  # type: ignore[arg-type]
                return False

    expected = _expected_output(grid.shape, design)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    return True


def _sample_specs(h: int, w: int) -> list[dict[str, object]] | None:
    colors = random.sample(PALETTE, FRAME_COUNT)
    directions = random.sample(DIRS, FRAME_COUNT)
    roles = list(range(FRAME_COUNT))
    target_idx = random.randrange(FRAME_COUNT)
    competitor_idx = random.choice([idx for idx in roles if idx != target_idx])

    specs: list[dict[str, object]] = []
    for idx, (color, direction) in enumerate(zip(colors, directions)):
        for _ in range(80):
            if idx == target_idx:
                box_h = random.randint(9, EXPANDED_H_RANGE[1])
                box_w = random.randint(8, EXPANDED_W_RANGE[1])
                dist_band = (4, 7)
            elif idx == competitor_idx:
                box_h = random.randint(7, 10)
                box_w = random.randint(7, 10)
                dist_band = (2, 3)
            else:
                box_h = random.randint(*EXPANDED_H_RANGE)
                box_w = random.randint(*EXPANDED_W_RANGE)
                dist_band = (2, 5)

            if h - box_h < 2 or w - box_w < 2:
                continue
            expanded_box = _random_box(h, w, box_h, box_w)
            distance = _feasible_distance(expanded_box, direction, *dist_band)
            if distance is None:
                continue
            box, marker = _shrink_box(expanded_box, direction, distance)
            inner_h = box[2] - box[0] - 1
            inner_w = box[3] - box[1] - 1
            if inner_h < 2 or inner_w < 2:
                continue
            specs.append(
                {
                    "color": color,
                    "direction": direction,
                    "expanded_box": expanded_box,
                    "distance": distance,
                    "box": box,
                    "marker": marker,
                    "inner": _inner_pattern(inner_h, inner_w),
                }
            )
            break
        else:
            return None

    specs.sort(key=lambda record: _interior_area(record["expanded_box"]), reverse=True)  # type: ignore[arg-type]
    for i, record in enumerate(specs):
        for other in specs[i + 1 :]:
            if _boxes_too_close(record["expanded_box"], other["expanded_box"]):  # type: ignore[arg-type]
                return None
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    for _ in range(PLACEMENT_RETRIES):
        design = _sample_specs(h, w)
        if design is None:
            continue

        transformed_areas = [_interior_area(record["expanded_box"]) for record in design]  # type: ignore[arg-type]
        original_areas = [_interior_area(record["box"]) for record in design]  # type: ignore[arg-type]
        if len(set(transformed_areas)) != FRAME_COUNT or len(set(original_areas)) != FRAME_COUNT:
            continue

        transformed_winner = max(range(FRAME_COUNT), key=lambda idx: transformed_areas[idx])
        original_winner = max(range(FRAME_COUNT), key=lambda idx: original_areas[idx])
        if transformed_winner == original_winner:
            continue

        grid = _render_input((h, w), design)
        if _semantic_checks(grid, design):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
