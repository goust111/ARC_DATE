from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
WALL = 3
MARKER_COLORS = (1, 2, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _wave_shift(distance_from_bottom: int) -> int:
    if distance_from_bottom % 2 == 0:
        return 0
    return -1 if (distance_from_bottom // 2) % 2 == 0 else 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert colors, "expected a wall object and markers"
    return max(colors, key=lambda c: int(np.count_nonzero(grid == c)))


def _marker_info(grid: np.ndarray, wall_color: int, bg: int) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, wall_color):
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), "each marker color must appear exactly once"
        r, c = map(int, coords[0])
        assert r == 0, "markers must be on the top row"
        markers.append((color, c))
    markers.sort(key=lambda item: item[1])
    assert markers, "expected at least one marker"
    return markers


def _apply_wave(grid: np.ndarray, wall_color: int, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    mask = out == wall_color
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a wall object"
    bottom = int(coords[:, 0].max())
    out[mask] = bg

    for r, c in coords:
        nr = int(r)
        nc = int(c) + _wave_shift(bottom - int(r))
        if 0 <= nc < out.shape[1]:
            assert out[nr, nc] == bg, "wave-shifted wall rows must not collide"
            out[nr, nc] = wall_color

    return out


def _contiguous_segments(mask_row: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    c = 0
    while c < mask_row.size:
        if not bool(mask_row[c]):
            c += 1
            continue
        start = c
        while c + 1 < mask_row.size and bool(mask_row[c + 1]):
            c += 1
        segments.append((start, c))
        c += 1
    return segments


def _nearest_wall_col(mask_row: np.ndarray, center: int) -> int | None:
    w = mask_row.size
    candidates = [c for c in range(max(0, center - 1), min(w, center + 2)) if bool(mask_row[c])]
    if not candidates:
        return None
    return min(candidates, key=lambda c: (abs(c - center), c))


def _detect_containers(grid: np.ndarray, wall_color: int, bg: int) -> list[dict[str, int]]:
    wall = grid == wall_color
    h, _ = grid.shape
    containers: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    for bottom in range(1, h):
        for left, right in _contiguous_segments(wall[bottom]):
            if right - left + 1 < 3:
                continue

            left_cols: list[int] = []
            right_cols: list[int] = []
            top = bottom

            for r in range(bottom - 1, -1, -1):
                lc = _nearest_wall_col(wall[r], left)
                rc = _nearest_wall_col(wall[r], right)
                if lc is None or rc is None or lc >= rc:
                    break
                if np.any(wall[r, lc + 1 : rc]):
                    break
                left_cols.append(int(lc))
                right_cols.append(int(rc))
                top = r

            if not left_cols:
                continue

            inner_left = max(left_cols) + 1
            inner_right = min(right_cols) - 1
            if inner_left > inner_right:
                continue
            if not np.all(grid[top:bottom, inner_left : inner_right + 1] == bg):
                continue

            key = (top, bottom, left, right, inner_left)
            if key in seen:
                continue
            seen.add(key)
            containers.append(
                {
                    "top": int(top),
                    "bottom": int(bottom),
                    "left": int(left),
                    "right": int(right),
                    "inner_left": int(inner_left),
                    "inner_right": int(inner_right),
                }
            )

    containers.sort(key=lambda box: (box["left"], box["top"]))
    return containers


def _column_drops_into_container(
    grid: np.ndarray, wall_color: int, container: dict[str, int], col: int
) -> bool:
    if not (container["inner_left"] <= col <= container["inner_right"]):
        return False
    return bool(np.all(grid[1 : container["bottom"], col] != wall_color))


def _transform_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    wall_color = _wall_color(g, bg)
    markers = _marker_info(g, wall_color, bg)

    shifted = _apply_wave(g, wall_color, bg)
    out = shifted.copy()
    for _, col in markers:
        out[0, col] = bg

    containers = _detect_containers(out, wall_color, bg)
    used_containers: set[int] = set()
    floor_color: int | None = None
    assignments: list[dict[str, Any]] = []

    for color, col in markers:
        hits = [
            idx
            for idx, container in enumerate(containers)
            if _column_drops_into_container(out, wall_color, container, col)
        ]
        assert len(hits) <= 1, "each marker column must select at most one shifted container"
        if hits:
            idx = hits[0]
            assert idx not in used_containers, "multiple markers cannot target the same container"
            used_containers.add(idx)
            container = containers[idx]
            out[
                container["top"] : container["bottom"],
                container["inner_left"] : container["inner_right"] + 1,
            ] = color
            assignments.append({"color": color, "col": col, "type": "container", "index": idx})
        else:
            assert floor_color is None, "at most one marker may miss all containers"
            floor_color = color
            assert np.all(out[-1] == bg), "bottom-row puddle requires an empty floor row"
            out[-1, :] = color
            assignments.append({"color": color, "col": col, "type": "floor"})

    meta = {
        "bg": bg,
        "wall_color": wall_color,
        "markers": markers,
        "shifted": shifted,
        "containers": containers,
        "assignments": assignments,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """Wave-shift the green U-container lattice, then let each top marker drop into the deformed container it can still reach.

    Input structure:
    - The grid is mostly black background with one green foreground lattice and 2-3 singleton marker pixels on the top row.
    - The green lattice contains 2-3 U-shaped containers connected into one 4-connected structure.
    - Each container has two side bars and a bottom segment, while marker colors are distinct single pixels in different columns.

    Rule:
    - Identify the green wall object and apply a one-cell alternating horizontal wave to it, anchored at the wall object's bottom row: distance 0 fixed, distance 1 left, distance 2 fixed, distance 3 right, and so on.
    - Re-identify the deformed containers after the wave by using the rows whose two side-wall pixels still leave a common empty rectangular interior.
    - Erase the original top-row markers, then drop each marker straight down its own column.
    - If a marker's column can pass unobstructed into exactly one deformed container interior, fill that container's interior rectangle with the marker color; otherwise fill the entire bottom row with that marker color.

    Key insight:
    - The wave acts on the container walls themselves, so the set of columns that remain truly enclosed is determined only after deformation; a marker that looked catchable before the shift can miss afterward.
    """
    return _transform_with_metadata(grid)[0]


def _container_columns_after_wave(left: int, right: int, depth: int) -> tuple[list[int], list[int], int, int]:
    left_cols: list[int] = []
    right_cols: list[int] = []
    for d in range(depth, 0, -1):
        shift = _wave_shift(d)
        left_cols.append(left + shift)
        right_cols.append(right + shift)
    inner_left = max(left_cols) + 1
    inner_right = min(right_cols) - 1
    return left_cols, right_cols, inner_left, inner_right


def _build_input(
    h: int,
    w: int,
    containers: list[dict[str, int]],
    marker_specs: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)

    for box in containers:
        top = box["top"]
        bottom = box["bottom"]
        left = box["left"]
        right = box["right"]
        grid[top : bottom + 1, left] = WALL
        grid[top : bottom + 1, right] = WALL
        grid[bottom, left : right + 1] = WALL

    for a, b in zip(containers, containers[1:]):
        grid[a["top"], a["right"] : b["left"] + 1] = WALL

    for color, col in marker_specs:
        grid[0, col] = color

    return grid


def _analyze_drop_targets(grid: np.ndarray) -> dict[str, Any]:
    bg = _most_common_color(grid)
    wall_color = _wall_color(grid, bg)
    markers = _marker_info(grid, wall_color, bg)

    original = np.array(grid, copy=True)
    for _, col in markers:
        original[0, col] = bg
    shifted = _apply_wave(grid, wall_color, bg)
    shifted_no_markers = shifted.copy()
    for _, col in markers:
        shifted_no_markers[0, col] = bg

    original_containers = _detect_containers(original, wall_color, bg)
    shifted_containers = _detect_containers(shifted_no_markers, wall_color, bg)

    before: list[tuple[str, int | None]] = []
    after: list[tuple[str, int | None]] = []
    for _, col in markers:
        before_hits = [
            idx
            for idx, box in enumerate(original_containers)
            if _column_drops_into_container(original, wall_color, box, col)
        ]
        after_hits = [
            idx
            for idx, box in enumerate(shifted_containers)
            if _column_drops_into_container(shifted_no_markers, wall_color, box, col)
        ]
        if before_hits:
            assert len(before_hits) == 1
            before.append(("container", before_hits[0]))
        else:
            before.append(("floor", None))
        if after_hits:
            assert len(after_hits) == 1
            after.append(("container", after_hits[0]))
        else:
            after.append(("floor", None))

    return {
        "markers": markers,
        "original_containers": original_containers,
        "shifted_containers": shifted_containers,
        "before": before,
        "after": after,
        "shifted_no_markers": shifted_no_markers,
    }


def _valid_generated_scene(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        analysis = _analyze_drop_targets(grid)
        out, meta = _transform_with_metadata(grid)
    except AssertionError:
        return False

    if len(analysis["markers"]) not in (2, 3):
        return False
    if len(analysis["original_containers"]) != len(analysis["shifted_containers"]):
        return False
    if len(analysis["shifted_containers"]) not in (2, 3):
        return False

    changed = [
        idx
        for idx, (before, after) in enumerate(zip(analysis["before"], analysis["after"]))
        if before != after
    ]
    if not changed:
        return False

    if not any(before[0] == "container" and after[0] == "floor" for before, after in zip(analysis["before"], analysis["after"])):
        return False

    after_types = [kind for kind, _ in analysis["after"]]
    if after_types.count("floor") != 1:
        return False
    if after_types.count("container") < 1:
        return False

    used_after = [idx for kind, idx in analysis["after"] if kind == "container"]
    if len(set(used_after)) != len(used_after):
        return False

    wall_color = meta["wall_color"]
    for (color, col), (kind, idx) in zip(analysis["markers"], analysis["after"]):
        if kind == "container":
            box = analysis["shifted_containers"][int(idx)]
            if not np.all(analysis["shifted_no_markers"][1 : box["bottom"], col] != wall_color):
                return False
            filled = out[box["top"] : box["bottom"], box["inner_left"] : box["inner_right"] + 1]
            if not np.all(filled == color):
                return False
        else:
            if not np.all(out[-1] == color):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    num_containers = random.choice((2, 3))
    h = random.randint(14, 18)
    interior_depth = random.randint(4, 6)
    bottom = random.randint(h - 5, h - 3)
    top = bottom - interior_depth
    if top < 3:
        return None

    if num_containers == 2:
        widths = [random.randint(3, 5) for _ in range(num_containers)]
        gaps = [random.randint(3, 5) for _ in range(num_containers - 1)]
        width_cap = 24
        extra_slack = 8
        min_canvas = 15
    else:
        widths = [random.randint(3, 4) for _ in range(num_containers)]
        gaps = [random.randint(3, 4) for _ in range(num_containers - 1)]
        width_cap = 28
        extra_slack = 7
        min_canvas = 18
    outer_widths = [width + 2 for width in widths]
    total_width = sum(outer_widths) + sum(gaps)
    w_min = max(min_canvas, total_width + 4)
    w_max = min(width_cap, total_width + extra_slack)
    if w_min > w_max:
        return None
    w = random.randint(w_min, w_max)
    left_margin = random.randint(2, w - total_width - 2)

    containers: list[dict[str, int]] = []
    cursor = left_margin
    for idx, interior_width in enumerate(widths):
        left = cursor
        right = left + interior_width + 1
        _, _, inner_left, inner_right = _container_columns_after_wave(left, right, interior_depth)
        if inner_left > inner_right:
            return None
        containers.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "inner_left": inner_left,
                "inner_right": inner_right,
                "orig_inner_left": left + 1,
                "orig_inner_right": right - 1,
            }
        )
        cursor = right + 1
        if idx < len(gaps):
            cursor += gaps[idx]

    changed_index = random.randrange(num_containers)
    changed_box = containers[changed_index]
    edge_choices = [
        c
        for c in range(changed_box["orig_inner_left"], changed_box["orig_inner_right"] + 1)
        if not (changed_box["inner_left"] <= c <= changed_box["inner_right"])
    ]
    if not edge_choices:
        return None

    marker_count = num_containers
    colors = random.sample(MARKER_COLORS, marker_count)
    marker_specs: list[tuple[int, int]] = []
    used_cols: set[int] = set()

    miss_col = random.choice(edge_choices)
    marker_specs.append((colors[0], miss_col))
    used_cols.add(miss_col)

    color_idx = 1
    for idx, box in enumerate(containers):
        if idx == changed_index:
            continue
        hit_choices = list(range(box["inner_left"], box["inner_right"] + 1))
        random.shuffle(hit_choices)
        hit_col = next((c for c in hit_choices if c not in used_cols), None)
        if hit_col is None:
            return None
        marker_specs.append((colors[color_idx], int(hit_col)))
        used_cols.add(int(hit_col))
        color_idx += 1

    marker_specs.sort(key=lambda item: item[1])
    grid = _build_input(h, w, containers, marker_specs)
    if not _valid_generated_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating-wave container puzzle")
