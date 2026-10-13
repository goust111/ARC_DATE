from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
RED = 2
WALL = 5
SLIDER_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mirror_column(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]
    assert len(candidates) == 1, "expected exactly one full-height red mirror line"
    return int(candidates[0])


def _find_slider_components(grid: np.ndarray, bg: int, mirror_c: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    blocked = {int(bg), RED, WALL}
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(mirror_c):
            color = int(grid[r0, c0])
            if color in blocked or seen[r0, c0]:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < h and 0 <= nc < mirror_c):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            bbox = (
                int(coords[:, 0].min()),
                int(coords[:, 1].min()),
                int(coords[:, 0].max()),
                int(coords[:, 1].max()),
            )
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "distance": int(mirror_c - 1 - coords[:, 1].max()),
                }
            )

    components.sort(key=lambda comp: (comp["distance"], comp["bbox"][0], comp["bbox"][1]))
    return components


def _mirror_coords(coords: np.ndarray, mirror_c: int) -> np.ndarray:
    twin = coords.copy()
    twin[:, 1] = 2 * mirror_c - twin[:, 1]
    return twin


def _left_limit(coords: np.ndarray, mirror_c: int, occupied_left: np.ndarray, block_by_settled: bool) -> int:
    shift = 0
    while True:
        next_cols = coords[:, 1] + shift + 1
        if np.any(next_cols == mirror_c):
            return shift
        if block_by_settled and np.any(occupied_left[coords[:, 0], next_cols] != -1):
            return shift
        shift += 1


def _simulate(
    grid: np.ndarray,
    *,
    block_by_walls: bool = True,
    block_by_settled: bool = True,
    with_trace: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mirror_c = _mirror_column(g)
    components = _find_slider_components(g, bg, mirror_c)
    assert components, "expected at least one left-side slider"

    out = g.copy()
    for comp in components:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = bg

    wall_mask = g == WALL
    occupied_left = np.full(g.shape, -1, dtype=int)
    occupied_right = np.full(g.shape, -1, dtype=int)
    trace: list[dict] = []

    for idx, comp in enumerate(components):
        coords = comp["coords"]
        twin = _mirror_coords(coords, mirror_c)
        color = int(comp["color"])
        shift = 0
        left_limit = _left_limit(coords, mirror_c, occupied_left, block_by_settled)

        while True:
            step = shift + 1
            next_left = coords.copy()
            next_right = twin.copy()
            next_left[:, 1] += step
            next_right[:, 1] -= step

            left_kind = None
            left_blocker = None
            if np.any(next_left[:, 1] == mirror_c):
                left_kind = "mirror"
            elif block_by_settled:
                left_hits = occupied_left[next_left[:, 0], next_left[:, 1]]
                if np.any(left_hits != -1):
                    left_kind = "slider"
                    left_blocker = int(left_hits[left_hits != -1][0])

            right_kind = None
            right_blocker = None
            if block_by_walls and np.any(wall_mask[next_right[:, 0], next_right[:, 1]]):
                right_kind = "wall"
            if block_by_settled:
                right_hits = occupied_right[next_right[:, 0], next_right[:, 1]]
                if np.any(right_hits != -1):
                    if right_kind is None:
                        right_kind = "twin"
                        right_blocker = int(right_hits[right_hits != -1][0])
                    elif right_blocker is None:
                        right_blocker = int(right_hits[right_hits != -1][0])

            if left_kind is not None or right_kind is not None:
                break
            shift += 1

        final_left = coords.copy()
        final_right = twin.copy()
        final_left[:, 1] += shift
        final_right[:, 1] -= shift
        out[final_left[:, 0], final_left[:, 1]] = color
        out[final_right[:, 0], final_right[:, 1]] = color

        if block_by_settled:
            occupied_left[final_left[:, 0], final_left[:, 1]] = idx
            occupied_right[final_right[:, 0], final_right[:, 1]] = idx

        if with_trace:
            trace.append(
                {
                    "index": idx,
                    "color": color,
                    "distance": int(comp["distance"]),
                    "shift": int(shift),
                    "left_limit": int(left_limit),
                    "left_kind": left_kind,
                    "left_blocker": left_blocker,
                    "right_kind": right_kind,
                    "right_blocker": right_blocker,
                }
            )

    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each left-side block toward a red mirror line while its hidden mirror twin slides inward on the right.

    Input structure:
    - A full-height red vertical line marks the mirror axis.
    - On the left of the line, colored 4-connected slider blocks sit on a black background.
    - On the right of the line, there are only gray wall segments; the mirrored twins are absent from the input.

    Rule:
    - For every left slider, imagine an unseen same-colored twin obtained by left-right reflection across the red line.
    - Process sliders from nearest to farthest from the mirror line.
    - Each slider moves right while its twin moves left in perfect mirror synchrony.
    - A pair stops exactly when the slider's next step would hit the red line or an already settled slider, or the twin's next step would hit a gray wall or an already settled twin.
    - Draw every slider and every twin at their final resting positions, keeping the red line and gray walls unchanged.

    Key insight:
    - The visible left block may stop in empty space because an invisible mirrored twin on the right is what actually gets blocked first.
    """
    out, _trace = _simulate(grid, with_trace=False)
    return out


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = int(color)


def _paint_wall_segment(grid: np.ndarray, rows: range, col: int) -> None:
    for r in rows:
        grid[int(r), int(col)] = WALL


def _make_distractor_wall(grid: np.ndarray, mirror_c: int, forbidden_rows: set[int]) -> bool:
    free_rows = [r for r in range(grid.shape[0]) if r not in forbidden_rows]
    if not free_rows:
        return False
    row = random.choice(free_rows)
    length = random.randint(1, min(4, grid.shape[1] - mirror_c - 1))
    start = random.randint(mirror_c + 1, grid.shape[1] - length)
    if np.any(grid[row, start : start + length] != BG):
        return False
    grid[row, start : start + length] = WALL
    return True


def _build_candidate() -> np.ndarray | None:
    n_sliders = 4 if random.random() < 0.45 else 3
    widths = [random.randint(1, 3) for _ in range(n_sliders)]
    gap_cols = [random.randint(1, 2) for _ in range(n_sliders - 1)]
    band1_h = random.randint(1, 2)
    band2_h = random.randint(1, 2)
    row_gap = random.randint(2, 4)
    near_gap = random.randint(2, 4)

    min_side = near_gap + sum(widths) + sum(gap_cols) - 1
    if min_side > 13:
        return None
    side = random.randint(max(8, min_side), 13)
    width = 2 * side + 1

    min_height = band1_h + band2_h + row_gap + 3
    if min_height > 16:
        return None
    height = random.randint(max(9, min_height), 16)
    band1_top = random.randint(1, height - (band1_h + band2_h + row_gap) - 1)
    band2_top = random.randint(band1_top + band1_h + row_gap, height - band2_h - 1)
    mirror_c = side

    colors = random.sample(SLIDER_COLORS, n_sliders)
    input_right_edges = [mirror_c - near_gap]
    for i in range(1, n_sliders):
        input_right_edges.append(input_right_edges[-1] - widths[i - 1] - gap_cols[i - 1])
    if any(input_right_edges[i] - widths[i] + 1 < 0 for i in range(n_sliders)):
        return None

    stop_a_with_wall = random.random() < 0.45 and near_gap >= 3
    shift_a = random.randint(1, near_gap - 2) if stop_a_with_wall else near_gap - 1
    if shift_a <= 0:
        return None

    left_a = input_right_edges[0] - widths[0] + 1 + shift_a
    shift_b = shift_a + gap_cols[0]

    left_limit_c = mirror_c - 1 - input_right_edges[2]
    if left_limit_c < 2:
        return None
    shift_c = random.randint(1, left_limit_c - 1)

    shifts = [shift_a, shift_b, shift_c]
    if n_sliders == 4:
        shifts.append(shift_c + gap_cols[2])

    tops = [band1_top, band1_top, band2_top]
    heights = [band1_h, band1_h, band2_h]
    if n_sliders == 4:
        tops.append(band2_top)
        heights.append(band2_h)

    grid = np.zeros((height, width), dtype=int)
    grid[:, mirror_c] = RED

    for idx in range(n_sliders):
        left = input_right_edges[idx] - widths[idx] + 1
        _paint_rect(grid, tops[idx], left, heights[idx], widths[idx], colors[idx])

    if stop_a_with_wall:
        wall_col_a = 2 * mirror_c - input_right_edges[0] - shift_a - 1
        if wall_col_a <= mirror_c:
            return None
        _paint_wall_segment(grid, range(band1_top, band1_top + band1_h), wall_col_a)

    wall_col_c = 2 * mirror_c - input_right_edges[2] - shift_c - 1
    if wall_col_c <= mirror_c:
        return None
    _paint_wall_segment(grid, range(band2_top, band2_top + band2_h), wall_col_c)

    used_rows = set(range(band1_top, band1_top + band1_h)) | set(range(band2_top, band2_top + band2_h))
    functional_wall_count = 2 if stop_a_with_wall else 1
    extra_walls_needed = 0 if functional_wall_count >= 2 else 1
    extra_walls_needed += random.randint(0, 1)
    for _ in range(extra_walls_needed):
        _make_distractor_wall(grid, mirror_c, used_rows)

    return grid


def _valid_generated_case(grid: np.ndarray) -> bool:
    try:
        out, trace = _simulate(grid, with_trace=True)
    except Exception:
        return False

    if len(trace) < 3:
        return False
    if len({item["distance"] for item in trace}) != len(trace):
        return False
    if any(item["shift"] <= 0 for item in trace):
        return False

    has_wall_forced_gap = any(
        item["right_kind"] == "wall" and item["shift"] < item["left_limit"]
        for item in trace
    )
    if not has_wall_forced_gap:
        return False

    has_settled_blocker = any(
        item["left_kind"] == "slider" or item["right_kind"] == "twin"
        for item in trace
    )
    if not has_settled_blocker:
        return False

    out_without_walls, _ = _simulate(grid, block_by_walls=False, block_by_settled=True, with_trace=False)
    if np.array_equal(out, out_without_walls):
        return False

    out_without_settled, _ = _simulate(grid, block_by_walls=True, block_by_settled=False, with_trace=False)
    if np.array_equal(out, out_without_settled):
        return False

    mirror_c = _mirror_column(grid)
    colors_right = set(map(int, grid[:, mirror_c + 1 :].ravel()))
    if any(color not in {BG, WALL} for color in colors_right):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _valid_generated_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirror-twin sliding input")
