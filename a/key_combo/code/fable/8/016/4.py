from __future__ import annotations

from collections import Counter, deque
import numpy as np


BLACK = 0
GREY = 5
_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 300
_PLACE_ATTEMPTS = 80

_BASE_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (-1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (-1, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _black_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLACK:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _component_rows_cols(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    best = None
    for comp in _black_components(grid):
        r0, c0, r1, c1 = _component_rows_cols(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(sub != BLACK):
            continue
        area = len(comp)
        cand = (area, r0, c0, r1, c1)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        raise AssertionError("expected one black legend box")
    return best[1], best[2], best[3], best[4]


def _extract_keys(box: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    key_col = None
    for c in range(box.shape[1]):
        if np.any(box[:, c] != BLACK):
            key_col = c
            break
    if key_col is None:
        raise AssertionError("legend box must contain key pixels")
    keys = [(r, int(box[r, key_col])) for r in range(box.shape[0]) if int(box[r, key_col]) != BLACK]
    if not keys:
        raise AssertionError("legend box must expose at least one keyed row")
    return int(key_col), keys


def _color_components(
    grid: np.ndarray, color: int, outside: np.ndarray
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not outside[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and outside[nr, nc]
                        and int(grid[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_clear(arr: np.ndarray, bg: int) -> bool:
    return bool(np.all(arr == int(bg)))


def _pixel_visible(
    grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int], r: int, c: int
) -> bool:
    r0, c0, r1, c1 = bbox

    if r0 <= r <= r1:
        if c < c0:
            return _segment_clear(grid[r, c + 1 : c0], bg)
        if c > c1:
            return _segment_clear(grid[r, c1 + 1 : c], bg)

    if c0 <= c <= c1:
        if r < r0:
            return _segment_clear(grid[r + 1 : r0, c], bg)
        if r > r1:
            return _segment_clear(grid[r1 + 1 : r, c], bg)

    return False


def _component_visible(
    grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int], comp: list[tuple[int, int]]
) -> bool:
    return any(_pixel_visible(grid, bg, bbox, r, c) for r, c in comp)


def _visible_count_map(
    grid: np.ndarray, bbox: tuple[int, int, int, int], key_colors: list[int]
) -> dict[int, int]:
    bg = _get_bg(grid)
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    outside = np.ones((h, w), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    counts: dict[int, int] = {int(color): 0 for color in key_colors}
    for color in key_colors:
        for comp in _color_components(grid, int(color), outside):
            if _component_visible(grid, bg, bbox, comp):
                counts[int(color)] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count only the objects that can see the legend box and redraw their keyed tallies.

    Input structure:
    - A black legend box sits in one corner, with one keyed color pixel on each
      relevant row inside its leftmost non-black column.
    - Outside the box are several monochrome foreground objects in the keyed colors,
      plus grey wall segments, on a uniform background.
    - Walls and objects are all opaque for horizontal or vertical line-of-sight.

    Rule:
    - Find the legend box and read its keyed rows from top to bottom.
    - For each keyed color, split the same-colored pixels outside the box into
      separate 4-connected objects.
    - Count only the objects that have at least one pixel with a clear horizontal
      or vertical background-only segment to the legend box boundary.
    - Output only the legend box crop, drawing one dot per visible object on the
      matching row, starting at the key pixel column and advancing every other cell.

    Key insight:
    - The legend box is both the output canvas and the observation target:
      same-colored objects behind any wall or object do not count unless some pixel
      still has an unobstructed straight view to the box.
    """
    g = np.asarray(grid, dtype=int)
    bbox = _find_legend_bbox(g)
    r0, c0, r1, c1 = bbox
    box = g[r0 : r1 + 1, c0 : c1 + 1]
    key_col, keys = _extract_keys(box)
    key_colors = [color for _row, color in keys]
    counts = _visible_count_map(g, bbox, key_colors)

    out = np.zeros_like(box)
    for row, color in keys:
        for i in range(counts[int(color)]):
            cc = key_col + 2 * i
            if cc < out.shape[1]:
                out[row, cc] = int(color)
    return out


def _legend_bbox_for(
    height: int, width: int, corner: str, box_h: int, box_w: int
) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, box_h - 1, box_w - 1
    if corner == "tr":
        return 0, width - box_w, box_h - 1, width - 1
    if corner == "bl":
        return height - box_h, 0, height - 1, box_w - 1
    if corner == "br":
        return height - box_h, width - box_w, height - 1, width - 1
    raise ValueError(corner)


def _row_side(corner: str) -> str:
    return "right" if corner in {"tl", "bl"} else "left"


def _col_side(corner: str) -> str:
    return "bottom" if corner in {"tl", "tr"} else "top"


def _orient_shape(shape: list[tuple[int, int]], side: str) -> list[tuple[int, int]]:
    if side == "right":
        return [(dr, dc) for dr, dc in shape]
    if side == "left":
        return [(dr, -dc) for dr, dc in shape]
    if side == "bottom":
        return [(dc, dr) for dr, dc in shape]
    if side == "top":
        return [(-dc, dr) for dr, dc in shape]
    raise ValueError(side)


def _anchor_for(side: str, lane: int, dist: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "right":
        return lane, c1 + dist
    if side == "left":
        return lane, c0 - dist
    if side == "bottom":
        return r1 + dist, lane
    if side == "top":
        return r0 - dist, lane
    raise ValueError(side)


def _translate(shape: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + dr, ac + dc) for dr, dc in shape]


def _cells_fit(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    occupied: np.ndarray,
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
    return True


def _wall_cells(
    side: str,
    obj_cells: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rows = [r for r, _ in obj_cells]
    cols = [c for _, c in obj_cells]
    obj_r0, obj_r1 = min(rows), max(rows)
    obj_c0, obj_c1 = min(cols), max(cols)

    if side == "right":
        wall_c = (c1 + obj_c0) // 2
        return [(r, wall_c) for r in range(obj_r0, obj_r1 + 1)]
    if side == "left":
        wall_c = (c0 + obj_c1) // 2
        return [(r, wall_c) for r in range(obj_r0, obj_r1 + 1)]
    if side == "bottom":
        wall_r = (r1 + obj_r0) // 2
        return [(wall_r, c) for c in range(obj_c0, obj_c1 + 1)]
    if side == "top":
        wall_r = (r0 + obj_r1) // 2
        return [(wall_r, c) for c in range(obj_c0, obj_c1 + 1)]
    raise ValueError(side)


def _draw_legend(
    grid: np.ndarray, corner: str, key_colors: list[int], slots: int
) -> tuple[int, int, int, int]:
    box_h = len(key_colors) + 2
    box_w = 2 * slots + 1
    bbox = _legend_bbox_for(grid.shape[0], grid.shape[1], corner, box_h, box_w)
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for i, color in enumerate(key_colors):
        grid[r0 + 1 + i, c0 + 1] = int(color)
    return bbox


def _build_records(
    key_colors: list[int], row_lanes: list[int], col_lanes: list[int], slots: int
) -> list[dict[str, int | bool]]:
    lanes = [("row", lane) for lane in row_lanes] + [("col", lane) for lane in col_lanes]
    _RNG.shuffle(lanes)
    records: list[dict[str, int | bool]] = []
    visible_counts = {int(color): 0 for color in key_colors}

    for idx, color in enumerate(key_colors):
        lane_kind, lane_value = lanes.pop()
        visible = True if idx == 0 else False if idx == 1 else bool(int(_RNG.integers(0, 2)))
        if visible:
            visible_counts[int(color)] += 1
        records.append(
            {
                "color": int(color),
                "visible": visible,
                "lane_kind": lane_kind,
                "lane_value": int(lane_value),
            }
        )

    if lanes:
        extra_count = int(_RNG.integers(1, min(len(lanes), len(key_colors) + 1) + 1))
        for _ in range(extra_count):
            lane_kind, lane_value = lanes.pop()
            color = int(_RNG.choice(key_colors))
            visible = bool(int(_RNG.integers(0, 2)))
            if visible and visible_counts[color] >= slots:
                visible = False
            if visible:
                visible_counts[color] += 1
            records.append(
                {
                    "color": color,
                    "visible": visible,
                    "lane_kind": lane_kind,
                    "lane_value": int(lane_value),
                }
            )
    return records


def _place_record(
    grid: np.ndarray,
    occupied: np.ndarray,
    bbox: tuple[int, int, int, int],
    record: dict[str, int | bool],
    corner: str,
) -> bool:
    lane_kind = str(record["lane_kind"])
    side = _row_side(corner) if lane_kind == "row" else _col_side(corner)
    lane = int(record["lane_value"])
    visible = bool(record["visible"])
    color = int(record["color"])

    for _ in range(_PLACE_ATTEMPTS):
        shape = _orient_shape(_BASE_SHAPES[int(_RNG.integers(len(_BASE_SHAPES)))], side)
        dist = int(_RNG.integers(2, 9))
        anchor = _anchor_for(side, lane, dist, bbox)
        obj_cells = _translate(shape, anchor)
        if not _cells_fit(grid.shape, obj_cells, occupied):
            continue

        wall_cells: list[tuple[int, int]] = []
        if not visible:
            wall_cells = _wall_cells(side, obj_cells, bbox)
            if not _cells_fit(grid.shape, wall_cells, occupied):
                continue
            if any(cell in obj_cells for cell in wall_cells):
                continue

        for r, c in obj_cells:
            grid[r, c] = color
            occupied[r, c] = True
        for r, c in wall_cells:
            grid[r, c] = GREY
            occupied[r, c] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(18, 29))
    width = int(_RNG.integers(18, 29))
    bg_choices = [1, 2, 3, 4, 6, 7, 8, 9]
    bg = int(_RNG.choice(bg_choices))
    key_count = int(_RNG.integers(2, 5))
    slots = int(_RNG.integers(3, 5))

    color_pool = [c for c in range(1, 10) if c not in {bg, GREY}]
    key_colors = [int(c) for c in _RNG.choice(color_pool, size=key_count, replace=False)]
    corner = str(_RNG.choice(["tl", "tr", "bl", "br"]))

    grid = np.full((height, width), bg, dtype=int)
    bbox = _draw_legend(grid, corner, key_colors, slots)
    occupied = grid != bg

    r0, c0, r1, c1 = bbox
    row_lanes = list(range(r0 + 1, r1))
    col_lanes = list(range(c0 + 1, c1))
    if len(row_lanes) + len(col_lanes) < key_count + 1:
        return None

    records = _build_records(key_colors, row_lanes, col_lanes, slots)
    for record in records:
        if not _place_record(grid, occupied, bbox, record, corner):
            return None

    counts = _visible_count_map(grid, bbox, key_colors)
    visible_total = sum(counts.values())
    if visible_total == 0:
        return None

    object_totals = {color: 0 for color in key_colors}
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    for color in key_colors:
        comps = _color_components(grid, int(color), outside)
        object_totals[int(color)] = len(comps)
        if counts[int(color)] > slots:
            return None
        if object_totals[int(color)] == 0:
            return None

    hidden_total = sum(object_totals[color] - counts[color] for color in key_colors)
    if hidden_total == 0:
        return None
    if not np.any(grid == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
