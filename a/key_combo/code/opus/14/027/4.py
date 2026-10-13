from __future__ import annotations

from dataclasses import dataclass
import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
MAROON = 9
OBSTACLE = 5
BODY_COLORS = (2, 3, 4, 6, 7, 8)

GRID_H_RANGE = (12, 16)
GRID_W_RANGE = (14, 18)
OBJECT_COUNT_WEIGHTS = ((2, 4), (3, 6))
AREA_RANGE = (3, 6)
WIDTH_OPTIONS = (2, 3)
HEIGHT_RANGE = (2, 4)
EDGE_MARGIN = 1
MIN_GAP = 1
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120


@dataclass(frozen=True)
class FallingObject:
    cells: tuple[tuple[int, int], ...]
    control: int
    body: int
    control_cell: tuple[int, int]
    left_col: int

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def columns(self) -> dict[int, list[int]]:
        cols: dict[int, list[int]] = {}
        for r, c in self.cells:
            cols.setdefault(int(c), []).append(int(r))
        for c in cols:
            cols[c].sort()
        return cols


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _extract_objects(grid: np.ndarray) -> list[FallingObject]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    active = (g != bg) & (g != OBSTACLE)
    comps = _component_cells(active)
    objects: list[FallingObject] = []
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        control_cells = [(r, c) for r, c in comp if int(g[r, c]) in (BLUE, MAROON)]
        assert len(control_cells) == 1, "each object must contain exactly one control cell"
        control = int(g[control_cells[0]])
        body_colors = colors - {BLUE, MAROON}
        assert len(body_colors) == 1, "each object must have exactly one body color"
        body = int(next(iter(body_colors)))
        assert body not in (BG, OBSTACLE, BLUE, MAROON)
        cols = {c for _, c in comp}
        objects.append(
            FallingObject(
                cells=tuple(sorted((int(r), int(c)) for r, c in comp)),
                control=control,
                body=body,
                control_cell=(int(control_cells[0][0]), int(control_cells[0][1])),
                left_col=min(cols),
            )
        )
    objects.sort(key=lambda obj: obj.left_col)
    return objects


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    assert bg == BG, "background must be black"

    objects = _extract_objects(g)
    assert 2 <= len(objects) <= 3, "expected 2-3 falling objects"

    used_cols: set[int] = set()
    for obj in objects:
        assert 3 <= obj.area <= 6, "object area must be 3-6"
        cols = set(obj.columns.keys())
        assert used_cols.isdisjoint(cols), "objects must not share columns"
        used_cols |= cols

    out = g.copy()
    for obj in objects:
        for r, c in obj.cells:
            out[r, c] = bg
    base = out.copy()

    traces: list[dict[str, object]] = []
    for obj in objects:
        trail_color = OBSTACLE if obj.control == BLUE else obj.body
        col_info: dict[int, dict[str, object]] = {}
        for c, rows in obj.columns.items():
            current_bottom = rows[-1]
            obstacle_rows = [r for r in range(current_bottom + 1, h) if int(base[r, c]) == OBSTACLE]
            if obj.control == BLUE and obstacle_rows:
                target_bottom = obstacle_rows[0] - 1
            else:
                target_bottom = h - 1
            shift = target_bottom - current_bottom
            assert shift >= 1, "objects must have room to fall"
            target_rows = [r + shift for r in rows]
            for src_r, dst_r in zip(rows, target_rows):
                for rr in range(src_r + 1, dst_r):
                    if int(base[rr, c]) != OBSTACLE:
                        out[rr, c] = trail_color
            for src_r, dst_r in zip(rows, target_rows):
                if int(base[dst_r, c]) != OBSTACLE:
                    color = obj.control if (src_r, c) == obj.control_cell else obj.body
                    out[dst_r, c] = color
            col_info[int(c)] = {
                "source_rows": list(rows),
                "target_rows": target_rows,
                "first_obstacle": obstacle_rows[0] if obstacle_rows else None,
                "trail_color": trail_color,
            }
        traces.append({"object": obj, "columns": col_info})
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each mixed-color top blob by column, using its control pixel to decide whether it stops on obstacles or falls through to the bottom.

    Input structure:
    - The grid has a black background, 2-3 small 4-connected mixed-color blobs near the top, and grey obstacle pixels in the lower region.
    - Each blob uses one body color plus exactly one internal control pixel, which is either blue or maroon.
    - Different blobs do not share any columns, and each relevant column contains at most one obstacle below the blobs.

    Rule:
    - Erase the original blobs, then process them from left to right.
    - For each blob column, shift that whole column of blob pixels straight downward until the column bottom stops just above the first grey obstacle if the blob control is blue, otherwise ignore obstacles and bottom-align the column.
    - Paint the fall trail between source and landing cells: blue-controlled blobs leave grey trails, while maroon-controlled blobs leave trails in their own body color; the landed body cells keep the body color and the landed control cell keeps its blue or maroon color.

    Key insight:
    - The single control pixel couples two decisions at once: it selects both the stopping behavior against obstacles and the color used to record the descent trail.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _random_shape(width: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(max(HEIGHT_RANGE[0], 2), HEIGHT_RANGE[1])
        max_area = min(AREA_RANGE[1], width * height)
        min_area = max(AREA_RANGE[0], width)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < area:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in _neighbors4(r, c, height, width)
                    if nbr not in cells
                }
            )
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        if {c for _, c in cells} != set(range(width)):
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _distribute_columns(widths: list[int], total_w: int) -> list[int] | None:
    required = sum(widths) + MIN_GAP * (len(widths) - 1) + 2 * EDGE_MARGIN
    if required > total_w:
        return None
    spare = total_w - required
    slots = [0] * (len(widths) + 1)
    for _ in range(spare):
        slots[random.randrange(len(slots))] += 1
    starts: list[int] = []
    c = EDGE_MARGIN + slots[0]
    for i, width in enumerate(widths):
        starts.append(c)
        c += width
        if i + 1 < len(widths):
            c += MIN_GAP + slots[i + 1]
    return starts


def _paint_object(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    body: int,
    control: int,
    control_local: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr, cc = top + int(r), left + int(c)
        grid[rr, cc] = body
        coords.append((rr, cc))
    control_global = (top + int(control_local[0]), left + int(control_local[1]))
    grid[control_global] = control
    return tuple(sorted(coords))


def _place_extra_obstacles(grid: np.ndarray, reserved_cols: set[int], reserved_cells: set[tuple[int, int]]) -> None:
    h, w = grid.shape
    top = max(h // 3 + 1, 4)
    target = random.randint(3, 7)
    for _ in range(target):
        for _ in range(30):
            r = random.randint(top, h - 1)
            c = random.randint(0, w - 1)
            if c in reserved_cols and random.random() < 0.85:
                continue
            if (r, c) in reserved_cells or int(grid[r, c]) != BG:
                continue
            grid[r, c] = OBSTACLE
            reserved_cells.add((r, c))
            if random.random() < 0.25:
                for nr, nc in random.sample(_neighbors4(r, c, h, w), k=len(_neighbors4(r, c, h, w))):
                    if nr < top or (nr, nc) in reserved_cells or int(grid[nr, nc]) != BG:
                        continue
                    if nc in reserved_cols:
                        continue
                    grid[nr, nc] = OBSTACLE
                    reserved_cells.add((nr, nc))
                    break
            break


def _validate_semantics(inp: np.ndarray) -> bool:
    out, traces = _simulate(inp)
    h, w = inp.shape
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert np.any(inp == OBSTACLE), "must contain obstacles"

    objects = [entry["object"] for entry in traces]
    controls = [obj.control for obj in objects]
    assert BLUE in controls and MAROON in controls, "must contain both control colors"
    assert len({obj.body for obj in objects}) == len(objects), "body colors should be distinct"

    used_cols: set[int] = set()
    has_blue_stop = False
    has_maroon_pass = False
    has_no_obstacle_branch = False

    base = inp.copy()
    for obj in objects:
        rows = [r for r, _ in obj.cells]
        assert max(rows) < (h + 2) // 3 + 1, "objects must start near the top"
        assert 3 <= obj.area <= 6
        cols = set(obj.columns.keys())
        assert used_cols.isdisjoint(cols)
        used_cols |= cols
        for r, c in obj.cells:
            base[r, c] = BG

    for entry in traces:
        obj: FallingObject = entry["object"]  # type: ignore[assignment]
        columns: dict[int, dict[str, object]] = entry["columns"]  # type: ignore[assignment]
        for c, info in columns.items():
            src_rows = info["source_rows"]  # type: ignore[assignment]
            dst_rows = info["target_rows"]  # type: ignore[assignment]
            first_obstacle = info["first_obstacle"]
            trail_color = int(info["trail_color"])

            if first_obstacle is None:
                has_no_obstacle_branch = True
            if obj.control == BLUE and first_obstacle is not None:
                has_blue_stop = True
                assert dst_rows[-1] == int(first_obstacle) - 1
                assert trail_color == OBSTACLE
            if obj.control == MAROON and first_obstacle is not None:
                has_maroon_pass = True
                assert dst_rows[-1] == h - 1
                assert trail_color == obj.body
            for src_r, dst_r in zip(src_rows, dst_rows):
                expected = obj.control if (src_r, c) == obj.control_cell else obj.body
                assert out[dst_r, c] == expected
                for rr in range(src_r + 1, dst_r):
                    if int(base[rr, c]) == OBSTACLE:
                        assert out[rr, c] == OBSTACLE
                    else:
                        assert out[rr, c] == trail_color

    assert has_blue_stop, "must include a blue-controlled stop-on-obstacle example"
    assert has_maroon_pass, "must include a maroon-controlled pass-through-obstacle example"
    assert np.any(out != inp), "transformation must change the grid"
    assert has_no_obstacle_branch or len(objects) == 2
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    object_count = random.choices(
        [pair[0] for pair in OBJECT_COUNT_WEIGHTS],
        weights=[pair[1] for pair in OBJECT_COUNT_WEIGHTS],
        k=1,
    )[0]

    specs = [{"control": BLUE, "mode": "stop"}, {"control": MAROON, "mode": "pass"}]
    if object_count == 3:
        specs.append(
            {
                "control": random.choice((BLUE, MAROON)),
                "mode": random.choice(("bottom_only", "bottom_only", "light_pass")),
            }
        )
    random.shuffle(specs)

    widths = [random.choice(WIDTH_OPTIONS) for _ in specs]
    starts = _distribute_columns(widths, w)
    if starts is None:
        return None

    body_colors = random.sample(BODY_COLORS, k=object_count)
    reserved_cells: set[tuple[int, int]] = set()
    reserved_cols: set[int] = set()
    column_plan: dict[int, int | None] = {}

    for spec, width, left, body in zip(specs, widths, starts, body_colors):
        mask = _random_shape(width)
        if mask is None:
            return None
        shape_h = mask.shape[0]
        top_limit = max(0, h // 3 - shape_h + 1)
        if top_limit < 0:
            return None
        top = random.randint(0, top_limit)
        coords = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
        if any(cell in reserved_cells for cell in coords):
            return None
        reserved_cells.update(coords)
        reserved_cols.update(left + c for c in range(width))

        control_local = random.choice([tuple(map(int, x)) for x in np.argwhere(mask)])
        _paint_object(grid, top, left, mask, body, spec["control"], control_local)

        cols_to_fill = list(range(left, left + width))
        per_col_bottoms: dict[int, int] = {}
        for c in cols_to_fill:
            col_rows = [r for r, cc in coords if cc == c]
            per_col_bottoms[c] = max(col_rows)

        if spec["mode"] == "stop":
            missing_allowed = object_count == 3 and random.random() < 0.35 and width >= 2
            if missing_allowed:
                missing_col = random.choice(cols_to_fill)
            else:
                missing_col = None
            for c in cols_to_fill:
                if c == missing_col:
                    column_plan[c] = None
                    continue
                low = max(h // 3 + 1, per_col_bottoms[c] + 2)
                high = h - 2
                if low > high:
                    return None
                column_plan[c] = random.randint(low, high)
        elif spec["mode"] == "pass":
            forced = random.sample(cols_to_fill, k=random.randint(1, len(cols_to_fill)))
            for c in cols_to_fill:
                if c not in forced and random.random() < 0.35:
                    column_plan[c] = None
                    continue
                low = max(h // 3 + 1, per_col_bottoms[c] + 2)
                high = h - 2
                if low > high:
                    return None
                column_plan[c] = random.randint(low, high)
        elif spec["mode"] == "light_pass":
            for c in cols_to_fill:
                if random.random() < 0.55:
                    low = max(h // 3 + 1, per_col_bottoms[c] + 2)
                    high = h - 2
                    if low > high:
                        return None
                    column_plan[c] = random.randint(low, high)
                else:
                    column_plan[c] = None
        else:
            for c in cols_to_fill:
                column_plan[c] = None

    for c, obstacle_row in column_plan.items():
        if obstacle_row is None:
            continue
        grid[int(obstacle_row), int(c)] = OBSTACLE
        reserved_cells.add((int(obstacle_row), int(c)))

    _place_extra_obstacles(grid, reserved_cols, reserved_cells)

    try:
        if _validate_semantics(grid):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
