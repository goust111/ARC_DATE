from __future__ import annotations

import random
from typing import Any

import numpy as np


GREEN = 3
RED = 2
PINK = 6
ORANGE = 7
MARKER_COLORS = (1, 4, 5, 6, 8, 9)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _marker_positions(grid: np.ndarray) -> set[tuple[int, int]]:
    coords = np.argwhere((grid != GREEN) & (grid != RED))
    return {tuple(map(int, p)) for p in coords}


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    starts = np.argwhere(grid == RED)
    assert len(starts) == 1, "expected exactly one red start pixel"
    r, c = map(int, starts[0])
    assert c == 0, "start pixel must be on the left edge"
    return r, c


def _trace_ray_network(grid: np.ndarray) -> dict[str, Any]:
    h, w = grid.shape
    start = _find_start(grid)
    markers = _marker_positions(grid)
    assert markers, "expected at least one marker"

    occupied = {start}
    pending: list[tuple[int, int, str]] = [(start[0], start[1], "h")]
    seen_segments: set[tuple[int, int, str]] = set()
    opened_markers: set[tuple[int, int]] = set()
    horizontal_segments: list[tuple[int, int, int]] = []

    while pending:
        r, c, axis = pending.pop()
        key = (r, c, axis)
        if key in seen_segments:
            continue
        seen_segments.add(key)

        if axis == "h":
            dirs = ((0, -1), (0, 1))
            next_axis = "v"
        else:
            dirs = ((-1, 0), (1, 0))
            next_axis = "h"

        segment_cells = {(r, c)}
        new_segments: list[tuple[int, int, str]] = []
        for dr, dc in dirs:
            prev_r, prev_c = r, c
            rr, cc = r + dr, c + dc
            while _in_bounds((h, w), rr, cc):
                if (rr, cc) in occupied:
                    break
                if (rr, cc) in markers:
                    marker = (rr, cc)
                    if marker not in opened_markers:
                        opened_markers.add(marker)
                        new_segments.append((prev_r, prev_c, next_axis))
                    break
                segment_cells.add((rr, cc))
                prev_r, prev_c = rr, cc
                rr += dr
                cc += dc

        occupied.update(segment_cells)
        if axis == "h":
            cols = [cell_c for cell_r, cell_c in segment_cells if cell_r == r]
            horizontal_segments.append((r, min(cols), max(cols)))

        # Left/up appended first so the stack pops right/down first.
        pending.extend(new_segments)

    return {
        "start": start,
        "markers": markers,
        "opened_markers": opened_markers,
        "horizontal_segments": horizontal_segments,
    }


def _grow_from_segments(
    grid: np.ndarray, trace: dict[str, Any], *, return_meta: bool = False
) -> tuple[np.ndarray, dict[str, Any]] | np.ndarray:
    h, w = grid.shape
    start = trace["start"]
    markers = trace["markers"]
    segments: list[tuple[int, int, int]] = list(trace["horizontal_segments"])

    out = np.asarray(grid, dtype=int).copy()
    seeds: list[dict[str, Any]] = []
    for seg_id, (row, c0, c1) in enumerate(segments):
        active = set(range(c0, c1 + 1))
        seeds.append({"id": seg_id, "row": row, "active": active})
        for c in range(c0, c1 + 1):
            if (row, c) == start or (row, c) in markers:
                continue
            out[row, c] = ORANGE

    generation = 0
    blocked_by_occupancy = 0
    blocked_midpoint = 0
    blocked_same_step = 0
    generated_pixels = 0

    while True:
        proposals: dict[tuple[int, int], list[int]] = {}
        generation += 1
        paint_color = PINK if generation % 2 == 1 else ORANGE

        for seed in seeds:
            row = int(seed["row"])
            active: set[int] = set(seed["active"])
            if row + 2 >= h or len(active) < 3:
                continue

            mask = np.zeros(w, dtype=bool)
            for c in active:
                mask[c] = True

            # The ray-defined row is a logical span: cells already active for this
            # branch do not block their own midpoint contraction, but any foreign
            # non-green cell still blocks further branching.
            midpoint_cols = np.flatnonzero(mask[:-2] & mask[2:]) + 1
            for c_raw in midpoint_cols.tolist():
                c = int(c_raw)
                if int(out[row, c]) != GREEN and c not in active:
                    blocked_midpoint += 1
                    continue
                target = (row + 2, c)
                if int(out[target]) != GREEN:
                    blocked_by_occupancy += 1
                    continue
                proposals.setdefault(target, []).append(int(seed["id"]))

        if not proposals:
            generation -= 1
            break

        accepted: dict[int, set[int]] = {int(seed["id"]): set() for seed in seeds}
        for target, seed_ids in proposals.items():
            unique_ids = set(seed_ids)
            if len(unique_ids) > 1:
                blocked_same_step += 1
                continue
            seed_id = next(iter(unique_ids))
            accepted[seed_id].add(int(target[1]))

        if not any(accepted.values()):
            generation -= 1
            break

        for seed in seeds:
            seed_id = int(seed["id"])
            new_cols = accepted[seed_id]
            if not new_cols:
                seed["active"] = set()
                continue
            seed["row"] = int(seed["row"]) + 2
            seed["active"] = new_cols
            for c in new_cols:
                out[int(seed["row"]), c] = paint_color
                generated_pixels += 1

    meta = {
        "horizontal_segments": segments,
        "rows": [int(r) for r, _c0, _c1 in segments],
        "opened_markers": set(trace["opened_markers"]),
        "generated_pixels": generated_pixels,
        "blocked_by_occupancy": blocked_by_occupancy,
        "blocked_midpoint": blocked_midpoint,
        "blocked_same_step": blocked_same_step,
        "generations": generation,
    }
    if return_meta:
        return out, meta
    return out


def _solve(grid: np.ndarray, *, return_meta: bool = False) -> tuple[np.ndarray, dict[str, Any]] | np.ndarray:
    trace = _trace_ray_network(grid)
    return _grow_from_segments(grid, trace, return_meta=return_meta)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace an invisible rectilinear ray network, convert each horizontal span into a competing seed row, and grow the spans downward.

    Input structure:
    - The grid uses green as background, with exactly one red start pixel on the left edge.
    - Several isolated non-green, non-red marker pixels are scattered in the interior.
    - No orange seed rows are pre-drawn; the hidden ray network is inferred only from the start and marker geometry.

    Rule:
    - Fire the initial ray horizontally from the red start. Whenever a ray first reaches a marker, stop one cell before it and emit the perpendicular axis from that stopping cell; later hits on the same marker only stop.
    - Rays are logical only: they are blocked by the grid edge or by previously traced rays, but their paths are not painted.
    - Collect every horizontal traced span. Paint each span as an orange seed row in the output (preserving the red start and every marker).
    - Then grow all seed rows synchronously. For one seed branch, any two active cells two columns apart contract to the midpoint two rows lower; the branch ignores its own current-row cells when testing that midpoint, but any foreign occupied cell blocks it. First-generation children are pink, second-generation children are orange, and the colors alternate.
    - If two different seed rows claim the same child cell in the same generation, that child is cancelled. A child is also cancelled if its destination is already occupied by an earlier seed row, marker, start pixel, or previously grown descendant.

    Key insight:
    - The colored output is driven by two coupled hidden structures: the invisible ray topology decides where the horizontal seed spans lie, and those spans then contract downward until they collide with one another or with earlier occupied cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit inside 30x30"
    assert int(np.min(g)) >= 0, "colors must be non-negative"
    return _solve(g)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _markers_are_well_spaced(markers: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _sample_rows(height: int) -> tuple[int, int, int] | None:
    candidates = []
    for start_row in range(5, min(height - 7, 10)):
        max_up = (start_row - 2) // 2
        max_down = (height - start_row - 4) // 2
        for up_steps in range(2, max_up + 1):
            for down_steps in range(2, max_down + 1):
                upper_row = start_row - 2 * up_steps
                lower_row = start_row + 2 * down_steps
                if upper_row < 2 or lower_row > height - 3:
                    continue
                if height - lower_row < 4:
                    continue
                candidates.append((start_row, upper_row, lower_row))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_marker_colors(count: int) -> list[int]:
    color_count = random.randint(2, min(4, count))
    palette = random.sample(MARKER_COLORS, color_count)
    return [random.choice(palette) for _ in range(count)]


def _build_candidate_grid() -> np.ndarray | None:
    height = random.randint(20, 28)
    width = random.randint(18, 24)
    rows = _sample_rows(height)
    if rows is None:
        return None
    start_row, upper_row, lower_row = rows

    kind = random.choices(
        ["upper_only", "lower_only", "both"],
        weights=(2, 3, 5),
        k=1,
    )[0]

    x1_max = width - 7
    if x1_max < 6:
        return None
    x1 = random.randint(6, min(11, x1_max))
    start_marker_col = x1 + 1

    upper_stop_col = None
    lower_stop_col = None
    if kind in {"upper_only", "both"}:
        upper_stop_col = random.randint(x1 + 3, min(width - 2, x1 + 7))
    if kind in {"lower_only", "both"}:
        lower_lo = x1 + 4
        if kind == "both":
            lower_lo = max(lower_lo, int(upper_stop_col) + 2)
        if lower_lo > width - 2:
            return None
        lower_stop_col = random.randint(lower_lo, width - 2)

    markers = [
        (start_row, start_marker_col),
        (upper_row - 1, x1),
        (lower_row + 1, x1),
    ]
    if upper_stop_col is not None:
        markers.append((upper_row, upper_stop_col))
    if lower_stop_col is not None:
        markers.append((lower_row, lower_stop_col))

    if not _markers_are_well_spaced(markers):
        return None

    colors = _choose_marker_colors(len(markers))
    grid = np.full((height, width), GREEN, dtype=int)
    grid[start_row, 0] = RED
    for (r, c), color in zip(markers, colors, strict=False):
        grid[r, c] = color
    return grid


def _validate_instance(grid: np.ndarray, out: np.ndarray, meta: dict[str, Any]) -> bool:
    markers = list(_marker_positions(grid))
    if not (4 <= len(markers) <= 6):
        return False
    if len(meta["opened_markers"]) != len(markers):
        return False

    segments: list[tuple[int, int, int]] = list(meta["horizontal_segments"])
    if len(segments) < 3:
        return False
    if len(set(meta["rows"])) < 3:
        return False

    overlap_found = False
    for i, (r1, a0, a1) in enumerate(segments):
        for r2, b0, b1 in segments[i + 1 :]:
            if r1 != r2 and _intervals_overlap(a0, a1, b0, b1):
                overlap_found = True
                break
        if overlap_found:
            break
    if not overlap_found:
        return False

    widths = [c1 - c0 + 1 for _r, c0, c1 in segments]
    if max(widths) < 6:
        return False
    if sum(width >= 5 for width in widths) < 2:
        return False

    if meta["generated_pixels"] < 8:
        return False
    if meta["blocked_by_occupancy"] < 1:
        return False
    if meta["generations"] < 2:
        return False

    used_colors = set(map(int, np.unique(out)))
    if PINK not in used_colors:
        return False

    density = float(np.count_nonzero(out != GREEN)) / float(out.size)
    if not (0.06 <= density <= 0.40):
        return False

    # Preserve the essential visible cues.
    if int(out[_find_start(grid)]) != RED:
        return False
    for r, c in markers:
        if int(out[r, c]) != int(grid[r, c]):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate_grid()
        if grid is None:
            continue
        try:
            out, meta = _solve(grid, return_meta=True)
        except Exception:
            continue
        if _validate_instance(grid, out, meta):
            return grid
    raise RuntimeError("failed to generate a valid ray-growth ARC input")
