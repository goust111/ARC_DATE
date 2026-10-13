from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == BG:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            rows = coords[:, 0]
            cols = coords[:, 1]
            colors = np.unique(grid[rows, cols])
            assert len(colors) == 1, "each object must be monochromatic"
            components.append(
                {
                    "coords": coords,
                    "color": int(colors[0]),
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return components


def _classify(comp: dict) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if width > height:
        return "H"
    if height > width:
        return "V"
    return "S"


def _trace_ray(
    grid: np.ndarray,
    plan: dict[tuple[int, int], int],
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    source_color: int,
) -> int:
    h, w = map(int, grid.shape)
    conflicts = 0
    r, c = start_r, start_c
    step = 0

    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BG:
            break
        color = GREY if step % 2 == 0 else source_color
        prev = plan.get((r, c))
        if prev is None:
            plan[(r, c)] = color
        elif prev != color:
            conflicts += 1
        r += dr
        c += dc
        step += 1

    return conflicts


def _build_paint_plan(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], int]:
    plan: dict[tuple[int, int], int] = {}
    conflicts = 0

    for comp in _find_components(grid):
        orientation = _classify(comp)
        coords = comp["coords"]
        color = int(comp["color"])

        if orientation == "H":
            row_extents: dict[int, list[int]] = {}
            for r, c in coords:
                r = int(r)
                c = int(c)
                if r not in row_extents:
                    row_extents[r] = [c, c]
                else:
                    row_extents[r][0] = min(row_extents[r][0], c)
                    row_extents[r][1] = max(row_extents[r][1], c)

            for r in sorted(row_extents):
                left, right = row_extents[r]
                conflicts += _trace_ray(grid, plan, r, left - 1, 0, -1, color)
                conflicts += _trace_ray(grid, plan, r, right + 1, 0, 1, color)

        elif orientation == "V":
            col_extents: dict[int, list[int]] = {}
            for r, c in coords:
                r = int(r)
                c = int(c)
                if c not in col_extents:
                    col_extents[c] = [r, r]
                else:
                    col_extents[c][0] = min(col_extents[c][0], r)
                    col_extents[c][1] = max(col_extents[c][1], r)

            for c in sorted(col_extents):
                top, bottom = col_extents[c]
                conflicts += _trace_ray(grid, plan, top - 1, c, -1, 0, color)
                conflicts += _trace_ray(grid, plan, bottom + 1, c, 1, 0, color)

    return plan, conflicts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each foreground object toward the grid edge along its dominant axis.

    Horizontal objects extend left and right row by row.
    Vertical objects extend up and down column by column.
    Square-bbox objects stay unchanged.
    New ray pixels alternate between grey and the object's own color, starting
    with grey in the cell adjacent to the object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    plan, _conflicts = _build_paint_plan(g)
    out = g.copy()
    for (r, c), color in plan.items():
        if int(out[r, c]) == BG:
            out[r, c] = color
    return out


def _can_place(grid: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = map(int, grid.shape)
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    return bool(np.all(grid[r0:r1, c0:c1] == BG))


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = color


def _sample_bar_dims(orientation: str, grid_h: int, grid_w: int) -> tuple[int, int] | None:
    if orientation == "H":
        height = random.randint(1, 3)
        max_width = min(7, grid_w - 4)
        min_width = max(3, height + 1)
        if min_width > max_width:
            return None
        width = random.randint(min_width, max_width)
        return height, width

    width = random.randint(1, 3)
    max_height = min(7, grid_h - 4)
    min_height = max(3, width + 1)
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)
    return height, width


def _place_emitters(grid: np.ndarray, orientation: str, count: int, colors: list[int]) -> bool:
    h, w = map(int, grid.shape)
    used_lines: set[int] = set()

    for idx in range(count):
        dims = _sample_bar_dims(orientation, h, w)
        if dims is None:
            return False
        obj_h, obj_w = dims

        placed = False
        for _ in range(150):
            if orientation == "H":
                if h - obj_h - 2 <= 1 or w - obj_w - 3 < 2:
                    return False
                top = random.randint(1, h - obj_h - 2)
                left = random.randint(2, w - obj_w - 3)
                rows = set(range(top, top + obj_h))
                if rows & used_lines:
                    continue
            else:
                if h - obj_h - 3 < 2 or w - obj_w - 2 <= 1:
                    return False
                top = random.randint(2, h - obj_h - 3)
                left = random.randint(1, w - obj_w - 2)
                rows = set(range(left, left + obj_w))
                if rows & used_lines:
                    continue

            if not _can_place(grid, top, left, obj_h, obj_w, pad=1):
                continue

            _paint_rect(grid, top, left, obj_h, obj_w, colors[idx])
            used_lines |= rows
            placed = True
            break

        if not placed:
            return False

    return True


def _place_squares(grid: np.ndarray, count: int, colors: list[int]) -> bool:
    h, w = map(int, grid.shape)
    for idx in range(count):
        placed = False
        for _ in range(150):
            size = random.randint(2, min(4, h - 2, w - 2))
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
            if not _can_place(grid, top, left, size, size, pad=1):
                continue
            _paint_rect(grid, top, left, size, size, colors[idx])
            placed = True
            break
        if not placed:
            return False
    return True


def _branch_counts() -> tuple[int, int]:
    return random.choice(((1, 1), (2, 0), (2, 1)))


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(11, 18)
    grid_w = random.randint(11, 18)
    orientation = random.choice(("H", "V"))
    emitter_count, square_count = _branch_counts()
    total_objects = emitter_count + square_count
    colors = random.sample(PALETTE, total_objects)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    if not _place_emitters(grid, orientation, emitter_count, colors[:emitter_count]):
        return None
    if not _place_squares(grid, square_count, colors[emitter_count:]):
        return None

    components = _find_components(grid)
    if len(components) != total_objects:
        return None
    if sum(_classify(comp) == "S" for comp in components) != square_count:
        return None
    if sum(_classify(comp) == orientation for comp in components) != emitter_count:
        return None

    plan, conflicts = _build_paint_plan(grid)
    if conflicts != 0 or len(plan) < 6:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if GREY not in set(map(int, np.unique(out))):
        return None

    changed = int(np.sum(out != grid))
    if changed < 6:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
