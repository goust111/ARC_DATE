from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
SIDES = ("top", "bottom", "left", "right")
H_SIDES = {"top", "bottom"}
V_SIDES = {"left", "right"}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_markers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    markers: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if len(coords) != 2:
                raise AssertionError("each colored object must be a 2-pixel domino marker")

            coords = sorted(coords)
            (r0, c0), (r1, c1) = coords
            if r0 == r1 and c1 == c0 + 1:
                orientation = "h"
            elif c0 == c1 and r1 == r0 + 1:
                orientation = "v"
            else:
                raise AssertionError("marker must be a horizontal or vertical domino")

            markers.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "orientation": orientation,
                }
            )

    colors = [marker["color"] for marker in markers]
    if len(colors) != len(set(colors)):
        raise AssertionError("marker colors must be distinct")
    return sorted(markers, key=lambda marker: (marker["color"], marker["coords"]))


def _trace_ray(grid: np.ndarray, start_r: int, start_c: int, dr: int, dc: int) -> tuple[list[tuple[int, int]], bool]:
    cells: list[tuple[int, int]] = []
    r, c = start_r, start_c
    while _in_bounds(grid.shape, r, c):
        value = int(grid[r, c])
        if value == GREY:
            return cells, True
        if value != BLACK:
            raise AssertionError("ray path must pass only through black cells")
        grid[r, c] = GREY
        cells.append((r, c))
        r += dr
        c += dc
    return cells, False


def _apply_markers(grid: np.ndarray, active_indices: set[int] | None = None) -> tuple[np.ndarray, list[dict], list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    markers = _find_markers(out)
    if active_indices is None:
        active_indices = set(range(len(markers)))

    ray_records: list[dict] = []
    for idx, marker in enumerate(markers):
        if idx not in active_indices:
            continue
        for r, c in marker["coords"]:
            out[r, c] = GREY
        coords = marker["coords"]
        if marker["orientation"] == "h":
            row = coords[0][0]
            left_start = min(c for _, c in coords) - 1
            right_start = max(c for _, c in coords) + 1
            left_cells, left_hit_wall = _trace_ray(out, row, left_start, 0, -1)
            right_cells, right_hit_wall = _trace_ray(out, row, right_start, 0, 1)
            ray_records.append(
                {
                    "index": idx,
                    "orientation": "h",
                    "neg_cells": left_cells,
                    "pos_cells": right_cells,
                    "neg_hit_wall": left_hit_wall,
                    "pos_hit_wall": right_hit_wall,
                }
            )
        else:
            col = coords[0][1]
            up_start = min(r for r, _ in coords) - 1
            down_start = max(r for r, _ in coords) + 1
            up_cells, up_hit_wall = _trace_ray(out, up_start, col, -1, 0)
            down_cells, down_hit_wall = _trace_ray(out, down_start, col, 1, 0)
            ray_records.append(
                {
                    "index": idx,
                    "orientation": "v",
                    "neg_cells": up_cells,
                    "pos_cells": down_cells,
                    "neg_hit_wall": up_hit_wall,
                    "pos_hit_wall": down_hit_wall,
                }
            )

    for idx, marker in enumerate(markers):
        if idx in active_indices:
            continue
        for r, c in marker["coords"]:
            out[r, c] = BLACK
    return out, markers, ray_records


def _erase_markers_only(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for marker in _find_markers(out):
        for r, c in marker["coords"]:
            out[r, c] = BLACK
    return out


def _exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if int(grid[r, c]) == BLACK and not vis[r, c]:
            vis[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK and not vis[nr, nc]:
                vis[nr, nc] = True
                q.append((nr, nc))
    return vis


def _enclosed_black_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == BLACK) & ~_exterior_black_mask(grid)


def _semantic_partition(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    interior = _enclosed_black_mask(grid)
    return interior, _components_mask(interior)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker-guided wall gap with grey rays, then recolor outside green and enclosed regions red.

    Input structure:
    - The grid has black background and grey wall segments.
    - It also contains 2-4 colored 2-pixel domino markers, each of a distinct color.
    - A horizontal marker sits in a horizontal wall gap; a vertical marker sits in a vertical wall gap.

    Rule:
    - Read every marker's orientation from its two-pixel domino shape.
    - From the marker, emit grey rays in both directions along that orientation until hitting an existing grey wall or leaving the grid; do not overwrite a wall cell.
    - Erase all colored marker pixels.
    - Flood-fill from the border through black cells: border-reachable black becomes green, remaining enclosed black becomes red, and grey walls stay grey.

    Key insight:
    - The domino marker is not the object to preserve; it is an orientation instruction for structurally completing the wall network, and only after that completion does flood-fill reveal which cells are inside versus outside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    completed, _markers, _ray_records = _apply_markers(g)
    exterior = _exterior_black_mask(completed)

    out = completed.copy()
    out[exterior] = GREEN
    out[out == BLACK] = RED
    return out


GEN_MARKER_COUNT_RANGE = (2, 4)
GEN_REGION_COUNT_CHOICES = (1, 2)
GEN_RECT_SIZE_RANGE = (8, 10)
GEN_MARGIN_RANGE = (1, 2)
GEN_REGION_GAP_RANGE = (3, 4)
GEN_GAP_LEN_RANGE = (4, 5)
GEN_MAX_ATTEMPTS = 500


def _choose_region_counts(marker_count: int, region_count: int) -> list[int]:
    if region_count == 1:
        return [marker_count]
    split = random.randint(1, marker_count - 1)
    return [split, marker_count - split]


def _choose_region_sides(region_gap_counts: list[int]) -> list[list[str]]:
    for _ in range(200):
        picked = [random.sample(SIDES, k=count) for count in region_gap_counts]
        all_sides = [side for region_sides in picked for side in region_sides]
        has_h = any(side in H_SIDES for side in all_sides)
        has_v = any(side in V_SIDES for side in all_sides)
        if has_h and has_v:
            return picked
    raise RuntimeError("failed to sample both horizontal and vertical marker orientations")


def _choose_gap_spec(side: str, height: int, width: int) -> dict:
    if side in H_SIDES:
        gap_len = random.randint(GEN_GAP_LEN_RANGE[0], min(GEN_GAP_LEN_RANGE[1], width - 3))
        start = random.randint(1, width - gap_len - 1)
        marker_start = random.randint(start + 1, start + gap_len - 3)
        return {
            "side": side,
            "gap_start": start,
            "gap_len": gap_len,
            "marker_start": marker_start,
            "orientation": "h",
        }
    gap_len = random.randint(GEN_GAP_LEN_RANGE[0], min(GEN_GAP_LEN_RANGE[1], height - 3))
    start = random.randint(1, height - gap_len - 1)
    marker_start = random.randint(start + 1, start + gap_len - 3)
    return {
        "side": side,
        "gap_start": start,
        "gap_len": gap_len,
        "marker_start": marker_start,
        "orientation": "v",
    }


def _sample_layout(region_gap_counts: list[int], region_sides: list[list[str]]) -> tuple[tuple[int, int], list[dict]] | None:
    rect_sizes = []
    for sides in region_sides:
        min_h = 8 if any(side in V_SIDES for side in sides) else GEN_RECT_SIZE_RANGE[0]
        min_w = 8 if any(side in H_SIDES for side in sides) else GEN_RECT_SIZE_RANGE[0]
        h = random.randint(min_h, GEN_RECT_SIZE_RANGE[1])
        w = random.randint(min_w, GEN_RECT_SIZE_RANGE[1])
        rect_sizes.append((h, w))

    if len(rect_sizes) == 1:
        h, w = rect_sizes[0]
        top = random.randint(*GEN_MARGIN_RANGE)
        left = random.randint(*GEN_MARGIN_RANGE)
        bottom = random.randint(*GEN_MARGIN_RANGE)
        right = random.randint(*GEN_MARGIN_RANGE)
        grid_shape = (top + h + bottom, left + w + right)
        regions = [{"r0": top, "c0": left, "h": h, "w": w}]
    else:
        arrangement = random.choice(("row", "col"))
        gap = random.randint(*GEN_REGION_GAP_RANGE)
        top = random.randint(*GEN_MARGIN_RANGE)
        left = random.randint(*GEN_MARGIN_RANGE)
        bottom = random.randint(*GEN_MARGIN_RANGE)
        right = random.randint(*GEN_MARGIN_RANGE)
        (h0, w0), (h1, w1) = rect_sizes
        if arrangement == "row":
            total_h = top + max(h0, h1) + bottom
            total_w = left + w0 + gap + w1 + right
            r0 = top + random.randint(0, max(h0, h1) - h0)
            r1 = top + random.randint(0, max(h0, h1) - h1)
            c0 = left
            c1 = left + w0 + gap
            grid_shape = (total_h, total_w)
            regions = [
                {"r0": r0, "c0": c0, "h": h0, "w": w0},
                {"r0": r1, "c0": c1, "h": h1, "w": w1},
            ]
        else:
            total_h = top + h0 + gap + h1 + bottom
            total_w = left + max(w0, w1) + right
            r0 = top
            r1 = top + h0 + gap
            c0 = left + random.randint(0, max(w0, w1) - w0)
            c1 = left + random.randint(0, max(w0, w1) - w1)
            grid_shape = (total_h, total_w)
            regions = [
                {"r0": r0, "c0": c0, "h": h0, "w": w0},
                {"r0": r1, "c0": c1, "h": h1, "w": w1},
            ]

    if max(grid_shape) > 30:
        return None

    colors = random.sample(MARKER_COLORS, sum(region_gap_counts))
    color_idx = 0
    region_specs: list[dict] = []
    for region, sides in zip(regions, region_sides):
        gaps = []
        for side in sides:
            gap = _choose_gap_spec(side, region["h"], region["w"])
            gap["color"] = colors[color_idx]
            color_idx += 1
            gaps.append(gap)
        region_specs.append({**region, "gaps": gaps})
    return grid_shape, region_specs


def _draw_region(grid: np.ndarray, region: dict) -> None:
    r0 = region["r0"]
    c0 = region["c0"]
    h = region["h"]
    w = region["w"]
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for gap in region["gaps"]:
        side = gap["side"]
        start = gap["gap_start"]
        end = start + gap["gap_len"] - 1
        marker_start = gap["marker_start"]
        color = gap["color"]

        if side == "top":
            cols = range(c0 + start, c0 + end + 1)
            grid[r0, list(cols)] = BLACK
            grid[r0, c0 + marker_start : c0 + marker_start + 2] = color
        elif side == "bottom":
            cols = range(c0 + start, c0 + end + 1)
            grid[r1, list(cols)] = BLACK
            grid[r1, c0 + marker_start : c0 + marker_start + 2] = color
        elif side == "left":
            rows = range(r0 + start, r0 + end + 1)
            grid[list(rows), c0] = BLACK
            grid[r0 + marker_start : r0 + marker_start + 2, c0] = color
        else:
            rows = range(r0 + start, r0 + end + 1)
            grid[list(rows), c1] = BLACK
            grid[r0 + marker_start : r0 + marker_start + 2, c1] = color


def _build_candidate() -> tuple[np.ndarray, int] | None:
    marker_count = random.randint(*GEN_MARKER_COUNT_RANGE)
    region_count = random.choice(GEN_REGION_COUNT_CHOICES)
    if region_count > marker_count:
        region_count = 1

    region_gap_counts = _choose_region_counts(marker_count, region_count)
    region_sides = _choose_region_sides(region_gap_counts)
    layout = _sample_layout(region_gap_counts, region_sides)
    if layout is None:
        return None
    grid_shape, regions = layout

    grid = np.zeros(grid_shape, dtype=int)
    for region in regions:
        _draw_region(grid, region)
    return grid, region_count


def _semantics_ok(grid: np.ndarray, expected_region_count: int) -> bool:
    try:
        markers = _find_markers(grid)
        if not (2 <= len(markers) <= 4):
            return False
        if any(marker["color"] not in MARKER_COLORS for marker in markers):
            return False

        base = _erase_markers_only(grid)
        before_inside, before_components = _semantic_partition(base)
        if before_inside.any() or before_components:
            return False

        completed, markers, ray_records = _apply_markers(grid)
        if len(ray_records) != len(markers):
            return False

        ray_cells: list[tuple[int, int]] = []
        for record in ray_records:
            if not record["neg_cells"] or not record["pos_cells"]:
                return False
            if not record["neg_hit_wall"] or not record["pos_hit_wall"]:
                return False
            ray_cells.extend(record["neg_cells"])
            ray_cells.extend(record["pos_cells"])
        if len(ray_cells) != len(set(ray_cells)):
            return False

        inside, components = _semantic_partition(completed)
        if len(components) != expected_region_count or not (1 <= len(components) <= 2):
            return False
        if any(len(comp) < 4 for comp in components):
            return False

        full_inside = inside.copy()
        active = set(range(len(markers)))
        for idx in range(len(markers)):
            partial_completed, _partial_markers, _ = _apply_markers(grid, active - {idx})
            partial_inside, _ = _semantic_partition(partial_completed)
            if np.array_equal(partial_inside, full_inside):
                return False

        out = main(grid)
        if set(map(int, np.unique(out))) - {GREY, RED, GREEN}:
            return False
        if GREY not in out or RED not in out or GREEN not in out:
            return False
        if out.shape != grid.shape or max(out.shape) > 30:
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is None:
            continue
        grid, expected_region_count = candidate
        if _semantics_ok(grid, expected_region_count):
            return grid
    raise RuntimeError("failed to generate a valid ray-wall flood partition input")
