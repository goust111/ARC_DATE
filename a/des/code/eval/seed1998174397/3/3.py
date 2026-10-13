from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
LEGEND_SIZE = 2
LEGEND_GAP = 1
MIN_PARALLEL_GAP = 3
COUNT_CHOICES = (
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
)
GRID_MAX = 28
GEN_ATTEMPTS = 400


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                }
            )

    return comps


def _is_legend_block(comp: dict) -> bool:
    return int(comp["h"]) == LEGEND_SIZE and int(comp["w"]) == LEGEND_SIZE and int(comp["area"]) == LEGEND_SIZE * LEGEND_SIZE


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[str, list[int], np.ndarray]:
    comps = _components(grid, bg)
    top_blocks = [comp for comp in comps if _is_legend_block(comp) and int(comp["r0"]) == 0]
    left_blocks = [comp for comp in comps if _is_legend_block(comp) and int(comp["c0"]) == 0]

    if len(top_blocks) >= 2 and len(top_blocks) > len(left_blocks):
        ordered = sorted(top_blocks, key=lambda comp: int(comp["c0"]))
        orientation = "top"
    elif len(left_blocks) >= 2 and len(left_blocks) > len(top_blocks):
        ordered = sorted(left_blocks, key=lambda comp: int(comp["r0"]))
        orientation = "left"
    elif len(top_blocks) >= 2:
        ordered = sorted(top_blocks, key=lambda comp: int(comp["c0"]))
        orientation = "top"
    elif len(left_blocks) >= 2:
        ordered = sorted(left_blocks, key=lambda comp: int(comp["r0"]))
        orientation = "left"
    else:
        raise AssertionError("legend not found")

    legend_colors = [int(comp["color"]) for comp in ordered]
    assert len(legend_colors) == len(set(legend_colors)), "legend colors must be unique"

    legend_mask = np.zeros(grid.shape, dtype=bool)
    for comp in ordered:
        coords = comp["coords"]
        legend_mask[coords[:, 0], coords[:, 1]] = True

    return orientation, legend_colors, legend_mask


def _extract_paths(grid: np.ndarray, legend_mask: np.ndarray, legend_colors: list[int]) -> list[dict]:
    paths: list[dict] = []
    for color in legend_colors:
        coords = np.argwhere((grid == int(color)) & (~legend_mask))
        assert coords.size > 0, "every legend color must have a path"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            paths.append(
                {
                    "color": int(color),
                    "orientation": "h",
                    "fixed": int(rows[0]),
                    "start": int(coords[:, 1].min()),
                    "end": int(coords[:, 1].max()),
                }
            )
        elif len(cols) == 1:
            paths.append(
                {
                    "color": int(color),
                    "orientation": "v",
                    "fixed": int(cols[0]),
                    "start": int(coords[:, 0].min()),
                    "end": int(coords[:, 0].max()),
                }
            )
        else:
            raise AssertionError("path must be a straight horizontal or vertical segment")

    return paths


def _crossing_actions(paths: list[dict], priority: dict[int, int]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    horizontal = [path for path in paths if path["orientation"] == "h"]
    vertical = [path for path in paths if path["orientation"] == "v"]
    actions: list[tuple[int, int, list[tuple[int, int]]]] = []

    for hp in horizontal:
        for vp in vertical:
            r = int(hp["fixed"])
            c = int(vp["fixed"])
            if not (int(hp["start"]) <= c <= int(hp["end"]) and int(vp["start"]) <= r <= int(vp["end"])):
                continue

            if priority[int(hp["color"])] < priority[int(vp["color"])]:
                high = hp
                low = vp
            else:
                high = vp
                low = hp

            cells = [(r, c)]
            if low["orientation"] == "h":
                for dc in (-1, 1):
                    cc = c + dc
                    if int(low["start"]) <= cc <= int(low["end"]):
                        cells.append((r, cc))
            else:
                for dr in (-1, 1):
                    rr = r + dr
                    if int(low["start"]) <= rr <= int(low["end"]):
                        cells.append((rr, c))

            actions.append((int(priority[int(high["color"])]), int(high["color"]), cells))

    actions.sort(key=lambda item: item[0], reverse=True)
    return actions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the 2x2 legend blocks to recover path priority, then repaint each crossing so
    the higher-priority path forms a 3-pixel bridge along the lower-priority path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    _orientation, legend_colors, legend_mask = _detect_legend(g, bg)
    paths = _extract_paths(g, legend_mask, legend_colors)
    priority = {color: idx for idx, color in enumerate(legend_colors)}

    out = g.copy()
    for _rank, color, cells in _crossing_actions(paths, priority):
        for r, c in cells:
            if not legend_mask[r, c]:
                out[r, c] = int(color)

    return out


def _sample_spaced_positions(count: int, low: int, high: int, gap: int) -> list[int] | None:
    span = high - low + 1
    effective = span - (gap - 1) * (count - 1)
    if effective < count:
        return None
    base = sorted(random.sample(range(effective), count))
    return [low + base[i] + i * (gap - 1) for i in range(count)]


def _interleaved_priority_order(colors: list[int], horiz_colors: list[int], vert_colors: list[int]) -> list[int] | None:
    for _ in range(80):
        order = random.sample(colors, len(colors))
        rank = {color: idx for idx, color in enumerate(order)}
        has_vertical_bridge = any(rank[h] < rank[v] for h in horiz_colors for v in vert_colors)
        has_horizontal_bridge = any(rank[v] < rank[h] for h in horiz_colors for v in vert_colors)
        if has_vertical_bridge and has_horizontal_bridge:
            return order
    return None


def _legend_extent(count: int) -> int:
    return count * LEGEND_SIZE + (count - 1) * LEGEND_GAP


def _build_paths(
    horiz_colors: list[int],
    vert_colors: list[int],
    draw_r0: int,
    draw_c0: int,
    active_h: int,
    active_w: int,
) -> list[dict] | None:
    draw_r1 = draw_r0 + active_h - 1
    draw_c1 = draw_c0 + active_w - 1

    rows = _sample_spaced_positions(len(horiz_colors), draw_r0 + 1, draw_r1 - 1, MIN_PARALLEL_GAP)
    cols = _sample_spaced_positions(len(vert_colors), draw_c0 + 1, draw_c1 - 1, MIN_PARALLEL_GAP)
    if rows is None or cols is None:
        return None

    row_min = min(rows)
    row_max = max(rows)
    col_min = min(cols)
    col_max = max(cols)

    paths: list[dict] = []
    for color, row in zip(horiz_colors, rows):
        start = random.randint(draw_c0, col_min - 1)
        end = random.randint(col_max + 1, draw_c1)
        paths.append({"color": int(color), "orientation": "h", "fixed": int(row), "start": int(start), "end": int(end)})

    for color, col in zip(vert_colors, cols):
        start = random.randint(draw_r0, row_min - 1)
        end = random.randint(row_max + 1, draw_r1)
        paths.append({"color": int(color), "orientation": "v", "fixed": int(col), "start": int(start), "end": int(end)})

    return paths


def _render_legend(grid: np.ndarray, orientation: str, legend_colors: list[int]) -> None:
    count = len(legend_colors)
    extent = _legend_extent(count)
    if orientation == "top":
        start = random.randint(0, grid.shape[1] - extent)
        for idx, color in enumerate(legend_colors):
            c0 = start + idx * (LEGEND_SIZE + LEGEND_GAP)
            grid[0:LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] = int(color)
        return

    start = random.randint(0, grid.shape[0] - extent)
    for idx, color in enumerate(legend_colors):
        r0 = start + idx * (LEGEND_SIZE + LEGEND_GAP)
        grid[r0 : r0 + LEGEND_SIZE, 0:LEGEND_SIZE] = int(color)


def _render_paths(grid: np.ndarray, paths: list[dict]) -> None:
    for path in paths:
        color = int(path["color"])
        if path["orientation"] == "h":
            row = int(path["fixed"])
            grid[row, int(path["start"]) : int(path["end"]) + 1] = color
        else:
            col = int(path["fixed"])
            grid[int(path["start"]) : int(path["end"]) + 1, col] = color

    horizontal = [path for path in paths if path["orientation"] == "h"]
    vertical = [path for path in paths if path["orientation"] == "v"]
    for hp in horizontal:
        for vp in vertical:
            r = int(hp["fixed"])
            c = int(vp["fixed"])
            if int(hp["start"]) <= c <= int(hp["end"]) and int(vp["start"]) <= r <= int(vp["end"]):
                grid[r, c] = random.choice((int(hp["color"]), int(vp["color"])))


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("top", "left"))
    nh, nv = random.choice(COUNT_CHOICES)
    total = nh + nv
    colors = random.sample(PALETTE, total)
    horiz_colors = colors[:nh]
    vert_colors = colors[nh:]
    legend_colors = _interleaved_priority_order(colors, horiz_colors, vert_colors)
    if legend_colors is None:
        return None

    min_active_h = 3 + (nh - 1) * MIN_PARALLEL_GAP
    min_active_w = 3 + (nv - 1) * MIN_PARALLEL_GAP
    active_h = min_active_h + random.randint(2, 7)
    active_w = min_active_w + random.randint(2, 7)

    if orientation == "top":
        legend_h = LEGEND_SIZE
        legend_w = _legend_extent(total)
        gap_above_draw = random.randint(2, 4)
        h_min = legend_h + gap_above_draw + active_h + 1
        w_min = max(legend_w, active_w + 2)
        if h_min > GRID_MAX or w_min > GRID_MAX:
            return None
        h = random.randint(h_min, GRID_MAX)
        w = random.randint(w_min, GRID_MAX)
        draw_r0 = random.randint(legend_h + gap_above_draw, h - active_h - 1)
        draw_c0 = random.randint(1, w - active_w - 1)
    else:
        legend_h = _legend_extent(total)
        legend_w = LEGEND_SIZE
        gap_left_of_draw = random.randint(2, 4)
        h_min = max(legend_h, active_h + 2)
        w_min = legend_w + gap_left_of_draw + active_w + 1
        if h_min > GRID_MAX or w_min > GRID_MAX:
            return None
        h = random.randint(h_min, GRID_MAX)
        w = random.randint(w_min, GRID_MAX)
        draw_r0 = random.randint(1, h - active_h - 1)
        draw_c0 = random.randint(legend_w + gap_left_of_draw, w - active_w - 1)

    paths = _build_paths(horiz_colors, vert_colors, draw_r0, draw_c0, active_h, active_w)
    if paths is None:
        return None

    rank = {color: idx for idx, color in enumerate(legend_colors)}
    bridge_directions = set()
    for path_h in [path for path in paths if path["orientation"] == "h"]:
        for path_v in [path for path in paths if path["orientation"] == "v"]:
            if rank[int(path_h["color"])] < rank[int(path_v["color"])]:
                bridge_directions.add("vertical")
            else:
                bridge_directions.add("horizontal")
    if bridge_directions != {"horizontal", "vertical"}:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _render_legend(grid, orientation, legend_colors)
    _render_paths(grid, paths)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
