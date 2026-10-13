import numpy as np
from collections import Counter, defaultdict


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
CORNERS = ("TL", "TR", "BL", "BR")
MARKER_DIRECTIONS = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")

GENERATE_RETRIES = 400
MAX_GRID_DIM = 36


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(cells: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(cells[:, 0].min()),
        int(cells[:, 1].min()),
        int(cells[:, 0].max()),
        int(cells[:, 1].max()),
    )


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, max(b0 - a1 - 1, a0 - b1 - 1))


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = _interval_gap(a[0], a[2], b[0], b[2])
    col_gap = _interval_gap(a[1], a[3], b[1], b[3])
    return min(row_gap, col_gap)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(cells, dtype=int)
            comps.append({"color": color, "cells": arr, "bbox": _bbox(arr)})
    return comps


def _is_marker(comp: dict) -> bool:
    cells = comp["cells"]
    r0, c0, r1, c1 = comp["bbox"]
    return cells.shape[0] == 4 and (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2


def _missing_region_for_l_shape(comp: dict) -> np.ndarray | None:
    cells = comp["cells"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    local[cells[:, 0] - r0, cells[:, 1] - c0] = True
    missing = np.argwhere(~local)
    if missing.size == 0:
        return None

    mr0 = int(missing[:, 0].min())
    mc0 = int(missing[:, 1].min())
    mr1 = int(missing[:, 0].max())
    mc1 = int(missing[:, 1].max())
    if missing.shape[0] != (mr1 - mr0 + 1) * (mc1 - mc0 + 1):
        return None
    if (mr1 - mr0 + 1) == h or (mc1 - mc0 + 1) == w:
        return None

    anchored = (
        (mr0 == 0 and mc0 == 0)
        or (mr0 == 0 and mc1 == w - 1)
        or (mr1 == h - 1 and mc0 == 0)
        or (mr1 == h - 1 and mc1 == w - 1)
    )
    if not anchored:
        return None

    return missing + np.array([r0, c0], dtype=int)


def _classify_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    bg = _background_color(grid)
    markers: list[dict] = []
    l_shapes: list[dict] = []
    for comp in _components(grid, bg):
        if _is_marker(comp):
            markers.append(comp)
            continue
        missing = _missing_region_for_l_shape(comp)
        if missing is None:
            raise AssertionError("every non-marker component must be a valid L shape")
        enriched = dict(comp)
        enriched["missing"] = missing
        l_shapes.append(enriched)
    return markers, l_shapes


def _assign_markers(markers: list[dict], l_shapes: list[dict]) -> list[int]:
    if len(markers) != len(l_shapes) or not markers:
        raise AssertionError("expected the same non-zero number of markers and L shapes")

    grouped: dict[int, list[int]] = defaultdict(list)
    for marker_idx, marker in enumerate(markers):
        distances = [_bbox_distance(marker["bbox"], shape["bbox"]) for shape in l_shapes]
        best = min(distances)
        winners = [idx for idx, value in enumerate(distances) if value == best]
        if len(winners) != 1:
            raise AssertionError("each marker must have a unique nearest L shape")
        grouped[winners[0]].append(marker_idx)

    assignment = []
    for shape_idx in range(len(l_shapes)):
        owners = grouped.get(shape_idx, [])
        if len(owners) != 1:
            raise AssertionError("each L shape must be paired with exactly one marker")
        assignment.append(owners[0])
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each L shape into a rectangle using the color of its nearest 2x2 marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    markers, l_shapes = _classify_scene(g)
    assignment = _assign_markers(markers, l_shapes)

    out = g.copy()
    for marker in markers:
        out[marker["cells"][:, 0], marker["cells"][:, 1]] = bg
    for shape_idx, shape in enumerate(l_shapes):
        marker = markers[assignment[shape_idx]]
        fill_color = int(marker["color"])
        missing = shape["missing"]
        out[missing[:, 0], missing[:, 1]] = fill_color
    return out


def _make_l_mask(height: int, width: int, corner: str, missing_h: int, missing_w: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones((height, width), dtype=bool)
    if corner == "TL":
        mask[:missing_h, :missing_w] = False
    elif corner == "TR":
        mask[:missing_h, width - missing_w :] = False
    elif corner == "BL":
        mask[height - missing_h :, :missing_w] = False
    else:
        mask[height - missing_h :, width - missing_w :] = False
    return mask, np.argwhere(~mask)


def _sample_local_pair(rng: np.random.Generator, l_mask: np.ndarray, missing_local: np.ndarray, direction: str) -> dict:
    height, width = l_mask.shape
    if direction in ("N", "S"):
        gap = int(rng.integers(1, 3))
        marker_left = int(rng.integers(-1, width))
        marker_top = -(gap + 2) if direction == "N" else height + gap
    elif direction in ("W", "E"):
        gap = int(rng.integers(1, 3))
        marker_top = int(rng.integers(-1, height))
        marker_left = -(gap + 2) if direction == "W" else width + gap
    else:
        row_gap = int(rng.integers(1, 3))
        col_gap = int(rng.integers(1, 3))
        marker_top = -(row_gap + 2) if "N" in direction else height + row_gap
        marker_left = -(col_gap + 2) if "W" in direction else width + col_gap

    min_r = min(0, marker_top)
    min_c = min(0, marker_left)
    max_r = max(height - 1, marker_top + 1)
    max_c = max(width - 1, marker_left + 1)
    shift = np.array([-min_r, -min_c], dtype=int)

    l_cells = np.argwhere(l_mask) + shift
    marker_cells = np.array(
        [
            [marker_top + shift[0], marker_left + shift[1]],
            [marker_top + shift[0], marker_left + shift[1] + 1],
            [marker_top + shift[0] + 1, marker_left + shift[1]],
            [marker_top + shift[0] + 1, marker_left + shift[1] + 1],
        ],
        dtype=int,
    )
    return {
        "env_h": int(max_r - min_r + 1),
        "env_w": int(max_c - min_c + 1),
        "l_cells": l_cells,
        "missing": missing_local + shift,
        "marker_cells": marker_cells,
        "l_bbox": _bbox(l_cells),
        "marker_bbox": _bbox(marker_cells),
    }


def _render_expected(input_grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for spec in specs:
        marker_cells = spec["marker_cells"]
        out[marker_cells[:, 0], marker_cells[:, 1]] = BG
    for spec in specs:
        missing = spec["missing"]
        out[missing[:, 0], missing[:, 1]] = int(spec["marker_color"])
    return out


def _strictly_inside(shape: tuple[int, int], cells: np.ndarray) -> bool:
    h, w = shape
    return bool(
        np.all(cells[:, 0] > 0)
        and np.all(cells[:, 0] < h - 1)
        and np.all(cells[:, 1] > 0)
        and np.all(cells[:, 1] < w - 1)
    )


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice([2, 3]))
    colors = rng.permutation(PALETTE)[: 2 * pair_count]
    diagonal_trend = int(rng.choice([-1, 1]))

    local_specs = []
    for pair_idx in range(pair_count):
        if pair_count == 2:
            height = int(rng.integers(4, 7))
            width = int(rng.integers(4, 7))
        else:
            height = int(rng.integers(3, 6))
            width = int(rng.integers(3, 6))
        corner = str(rng.choice(CORNERS))
        missing_h = int(rng.integers(1, height))
        missing_w = int(rng.integers(1, width))
        l_mask, missing = _make_l_mask(height, width, corner, missing_h, missing_w)
        direction = str(rng.choice(MARKER_DIRECTIONS))
        spec = _sample_local_pair(rng, l_mask, missing, direction)
        spec["l_color"] = int(colors[pair_idx])
        spec["marker_color"] = int(colors[pair_idx + pair_count])
        local_specs.append(spec)

    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    row_gaps = [int(rng.integers(2, 5)) for _ in range(pair_count - 1)]
    col_gaps = [int(rng.integers(2, 5)) for _ in range(pair_count - 1)]

    row_starts = []
    cursor = top_margin
    for idx, spec in enumerate(local_specs):
        row_starts.append(cursor)
        cursor += int(spec["env_h"])
        if idx < pair_count - 1:
            cursor += row_gaps[idx]
    grid_h = cursor + bottom_margin

    col_starts = [0] * pair_count
    cursor = left_margin
    order = list(range(pair_count)) if diagonal_trend > 0 else list(range(pair_count - 1, -1, -1))
    for order_idx, pair_idx in enumerate(order):
        col_starts[pair_idx] = cursor
        cursor += int(local_specs[pair_idx]["env_w"])
        if order_idx < pair_count - 1:
            cursor += col_gaps[order_idx]
    grid_w = cursor + right_margin

    if grid_h > MAX_GRID_DIM or grid_w > MAX_GRID_DIM:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    specs: list[dict] = []
    for pair_idx, local in enumerate(local_specs):
        offset = np.array([row_starts[pair_idx], col_starts[pair_idx]], dtype=int)
        l_cells = local["l_cells"] + offset
        missing = local["missing"] + offset
        marker_cells = local["marker_cells"] + offset

        if not (_strictly_inside(grid.shape, l_cells) and _strictly_inside(grid.shape, marker_cells)):
            return None
        if np.any(occupied[l_cells[:, 0], l_cells[:, 1]]) or np.any(occupied[marker_cells[:, 0], marker_cells[:, 1]]):
            return None

        occupied[l_cells[:, 0], l_cells[:, 1]] = True
        occupied[marker_cells[:, 0], marker_cells[:, 1]] = True
        grid[l_cells[:, 0], l_cells[:, 1]] = int(local["l_color"])
        grid[marker_cells[:, 0], marker_cells[:, 1]] = int(local["marker_color"])

        specs.append(
            {
                "l_color": int(local["l_color"]),
                "marker_color": int(local["marker_color"]),
                "l_cells": l_cells,
                "missing": missing,
                "marker_cells": marker_cells,
                "l_bbox": _bbox(l_cells),
                "marker_bbox": _bbox(marker_cells),
            }
        )

    if _background_color(grid) != BG:
        return None

    expected = _render_expected(grid, specs)
    try:
        markers, l_shapes = _classify_scene(grid)
        _assign_markers(markers, l_shapes)
    except AssertionError:
        return None
    if len(markers) != pair_count or len(l_shapes) != pair_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate random scenes with multiple L shapes and unique nearest color markers.
    """
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
