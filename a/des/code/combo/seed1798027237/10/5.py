from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


SEA = 8
ISLAND = 5
WALL = 9
LIGHTHOUSE_COLORS = (2, 3, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _ordered_lighthouses(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    h, w = grid.shape
    items: list[dict[str, int | tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()

    def add_if_lighthouse(r: int, c: int, dr: int, dc: int) -> None:
        if (r, c) in seen:
            return
        val = int(grid[r, c])
        if val in (SEA, ISLAND, WALL):
            return
        seen.add((r, c))
        items.append({"pos": (r, c), "dir": (dr, dc), "color": val})

    for c in range(w):
        add_if_lighthouse(0, c, 1, 0)
    for c in range(w):
        add_if_lighthouse(h - 1, c, -1, 0)
    for r in range(h):
        add_if_lighthouse(r, 0, 0, 1)
    for r in range(h):
        add_if_lighthouse(r, w - 1, 0, -1)
    return items


def _analyze_islands(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    comps = _components(grid == ISLAND)
    ids = np.full(grid.shape, -1, dtype=int)
    islands: list[dict[str, object]] = []
    for idx, comp in enumerate(comps):
        for r, c in comp:
            ids[r, c] = idx
        r0, c0, r1, c1 = _bbox(comp)
        islands.append(
            {
                "cells": comp,
                "bbox": (r0, c0, r1, c1),
                "rows": sorted({r for r, _ in comp}),
                "cols": sorted({c for _, c in comp}),
            }
        )
    return islands, ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey island by the colors of the edge lighthouses that can see it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    islands, island_ids = _analyze_islands(g)
    visible: list[list[tuple[int, int]]] = [[] for _ in islands]

    for order, lighthouse in enumerate(_ordered_lighthouses(g)):
        r, c = lighthouse["pos"]  # type: ignore[index]
        dr, dc = lighthouse["dir"]  # type: ignore[index]
        color = int(lighthouse["color"])  # type: ignore[index]
        r += dr
        c += dc
        seen_islands: set[int] = set()
        while 0 <= r < h and 0 <= c < w:
            val = int(g[r, c])
            if val == WALL:
                break
            if val == ISLAND:
                island_idx = int(island_ids[r, c])
                if island_idx >= 0 and island_idx not in seen_islands:
                    visible[island_idx].append((order, color))
                    seen_islands.add(island_idx)
            r += dr
            c += dc

    for island_idx, island in enumerate(islands):
        sightings = visible[island_idx]
        if not sightings:
            continue

        first_seen: dict[int, int] = {}
        for order, color in sightings:
            if color not in first_seen or order < first_seen[color]:
                first_seen[color] = order
        colors = [color for color, _ in sorted(first_seen.items(), key=lambda item: item[1])]

        cells = island["cells"]  # type: ignore[assignment]
        r0, c0, r1, c1 = island["bbox"]  # type: ignore[assignment]
        if len(colors) == 1:
            fill = colors[0]
            for r, c in cells:
                out[r, c] = fill
            continue

        axis_rows = (r1 - r0) >= (c1 - c0)
        for r, c in cells:
            idx = (r - r0) if axis_rows else (c - c0)
            out[r, c] = colors[idx % len(colors)]

    return out


def _random_spans(limit: int, sizes: list[int], border: int, gap: int, rng: random.Random) -> list[int] | None:
    base_gaps = [border] + [gap] * (len(sizes) - 1) + [border]
    slack = limit - sum(sizes) - sum(base_gaps)
    if slack < 0:
        return None
    extras = [0] * len(base_gaps)
    for _ in range(slack):
        extras[rng.randrange(len(extras))] += 1
    gaps = [base_gaps[i] + extras[i] for i in range(len(base_gaps))]
    starts: list[int] = []
    pos = gaps[0]
    for idx, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if idx < len(sizes) - 1:
            pos += gaps[idx + 1]
    return starts


def _make_island_shape(height: int, width: int, rng: random.Random) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(height) for c in range(width)}
    if height >= 3 and width >= 3:
        notch_h = rng.randint(1, height - 1)
        notch_w = rng.randint(1, width - 1)
        corner = rng.choice(("tl", "tr", "bl", "br"))
        removed: set[tuple[int, int]] = set()
        for r in range(notch_h):
            for c in range(notch_w):
                rr = r if corner in ("tl", "tr") else height - 1 - r
                cc = c if corner in ("tl", "bl") else width - 1 - c
                removed.add((rr, cc))
        if len(cells) - len(removed) >= max(4, min(height, width)):
            cells -= removed
    return cells


def _side_coordinate(island: dict[str, object], side: str, rng: random.Random) -> int:
    if side in ("top", "bottom"):
        cols = island["cols"]  # type: ignore[assignment]
        return int(rng.choice(cols))
    rows = island["rows"]  # type: ignore[assignment]
    return int(rng.choice(rows))


def _global_side_pos(shape: tuple[int, int], side: str, coord: int) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, coord
    if side == "bottom":
        return h - 1, coord
    if side == "left":
        return coord, 0
    return coord, w - 1


def _build_wall_plan(
    island: dict[str, object],
    visible_side_coords: dict[str, int],
    protected_rows: set[int],
    other_vertical_cols: set[int],
    all_island_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[str, int, list[tuple[int, int]]] | None:
    h, w = grid_shape
    cells = island["cells"]  # type: ignore[assignment]
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in cells:
        by_col[c].append(r)

    preferred: list[str]
    if "top" in visible_side_coords and "bottom" not in visible_side_coords:
        preferred = ["bottom", "top"]
    elif "bottom" in visible_side_coords and "top" not in visible_side_coords:
        preferred = ["top", "bottom"]
    else:
        preferred = ["top", "bottom"] if rng.random() < 0.5 else ["bottom", "top"]

    for side in preferred:
        cols = island["cols"][:]  # type: ignore[index]
        rng.shuffle(cols)
        if side in visible_side_coords:
            vis_col = visible_side_coords[side]
            cols = [c for c in cols if c != vis_col] + [c for c in cols if c == vis_col]

        for col in cols:
            rows_here = by_col[int(col)]
            if side == "top":
                row_options = list(range(1, min(rows_here) - 1))
            else:
                row_options = list(range(max(rows_here) + 2, h - 1))
            row_options = [r for r in row_options if r not in protected_rows]
            rng.shuffle(row_options)

            for wall_row in row_options:
                for length in (3, 2, 1):
                    start_min = max(1, int(col) - length + 1)
                    start_max = min(int(col), w - 1 - length)
                    starts = list(range(start_min, start_max + 1))
                    rng.shuffle(starts)
                    for start in starts:
                        wall = [(wall_row, c) for c in range(start, start + length)]
                        if int(col) not in {c for _, c in wall}:
                            continue
                        if other_vertical_cols & {c for _, c in wall}:
                            continue
                        if not _wall_is_valid(wall, all_island_cells, wall_cells):
                            continue
                        return side, int(col), wall
    return None


def _wall_is_valid(
    wall: list[tuple[int, int]],
    all_island_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> bool:
    for r, c in wall:
        if (r, c) in all_island_cells:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in all_island_cells:
                    return False
                if (nr, nc) in wall_cells and nr != r:
                    return False
    for r, c in wall:
        if (r, c) in wall_cells:
            continue
        if any((r, nc) in wall_cells for nc in (c - 1, c + 1)):
            continue
    return True


def _choose_color(palette: list[int], forbidden: set[int], rng: random.Random) -> int:
    choices = [c for c in palette if c not in forbidden]
    if not choices:
        choices = palette[:]
    return int(rng.choice(choices))


def _inspect_output(grid: np.ndarray, out: np.ndarray, islands: list[dict[str, object]]) -> list[int]:
    branch_kinds: list[int] = []
    for island in islands:
        colors = {int(out[r, c]) for r, c in island["cells"]}  # type: ignore[index]
        if colors == {ISLAND}:
            branch_kinds.append(0)
        elif len(colors) == 1:
            branch_kinds.append(1)
        else:
            branch_kinds.append(2)
    return branch_kinds


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_islands = rng.randint(2, 4)
    row_sizes = [rng.randint(3, 6) for _ in range(n_islands)]
    col_sizes = [rng.randint(3, 6) for _ in range(n_islands)]
    min_h = sum(row_sizes) + 3 * (n_islands - 1) + 6
    min_w = sum(col_sizes) + 3 * (n_islands - 1) + 6
    if min_h > 28 or min_w > 28:
        return None

    h = rng.randint(max(14, min_h), 28)
    w = rng.randint(max(14, min_w), 28)
    row_starts = _random_spans(h, row_sizes, border=3, gap=3, rng=rng)
    col_starts = _random_spans(w, col_sizes, border=3, gap=3, rng=rng)
    if row_starts is None or col_starts is None:
        return None

    perm = rng.sample(range(n_islands), n_islands)
    islands: list[dict[str, object]] = []
    all_island_cells: set[tuple[int, int]] = set()
    grid = np.full((h, w), SEA, dtype=int)
    for idx in range(n_islands):
        top = row_starts[idx]
        left = col_starts[perm[idx]]
        height = row_sizes[idx]
        width = col_sizes[perm[idx]]
        shape = _make_island_shape(height, width, rng)
        abs_cells = {(top + r, left + c) for r, c in shape}
        if abs_cells & all_island_cells:
            return None
        for r, c in abs_cells:
            grid[r, c] = ISLAND
        rows = sorted({r for r, _ in abs_cells})
        cols = sorted({c for _, c in abs_cells})
        islands.append(
            {
                "cells": sorted(abs_cells),
                "bbox": _bbox(abs_cells),
                "rows": rows,
                "cols": cols,
            }
        )
        all_island_cells |= abs_cells

    offset = rng.randrange(3)
    kinds = [((offset + i) % 3) for i in range(n_islands)]
    rng.shuffle(kinds)
    palette = list(rng.sample(LIGHTHOUSE_COLORS, rng.randint(3, 4)))

    lighthouse_plans: list[dict[str, object]] = []
    side_usage: set[tuple[str, int]] = set()
    visible_lr_rows: set[int] = set()
    visible_tb_cols_by_island: list[set[int]] = [set() for _ in islands]

    for idx, island in enumerate(islands):
        kind = kinds[idx]
        if kind == 0:
            visible_sides: list[str] = []
        elif kind == 1:
            visible_sides = [rng.choice(SIDES)]
        else:
            visible_sides = list(rng.sample(SIDES, 2))

        side_coords: dict[str, int] = {}
        for side in visible_sides:
            for _ in range(12):
                coord = _side_coordinate(island, side, rng)
                token = (side, coord)
                if token not in side_usage:
                    side_usage.add(token)
                    side_coords[side] = coord
                    break
            else:
                return None

        visible_colors: dict[str, int] = {}
        if kind == 1:
            visible_colors[visible_sides[0]] = _choose_color(palette, set(), rng)
        elif kind == 2:
            pair = rng.sample(palette, 2)
            visible_colors[visible_sides[0]] = pair[0]
            visible_colors[visible_sides[1]] = pair[1]

        for side in visible_sides:
            if side in ("left", "right"):
                visible_lr_rows.add(side_coords[side])
            else:
                visible_tb_cols_by_island[idx].add(side_coords[side])

        lighthouse_plans.append(
            {
                "kind": kind,
                "visible_sides": visible_sides,
                "visible_coords": side_coords,
                "visible_colors": visible_colors,
                "blocked": None,
            }
        )

    wall_cells: set[tuple[int, int]] = set()
    all_visible_tb_cols: set[int] = set()
    for cols in visible_tb_cols_by_island:
        all_visible_tb_cols |= cols

    for idx, island in enumerate(islands):
        plan = lighthouse_plans[idx]
        own_visible_tb = {
            coord
            for side, coord in plan["visible_coords"].items()  # type: ignore[union-attr]
            if side in ("top", "bottom")
        }
        blocked = _build_wall_plan(
            island=island,
            visible_side_coords=plan["visible_coords"],  # type: ignore[arg-type]
            protected_rows=visible_lr_rows,
            other_vertical_cols=all_visible_tb_cols - own_visible_tb,
            all_island_cells=all_island_cells,
            wall_cells=wall_cells,
            grid_shape=grid.shape,
            rng=rng,
        )
        if blocked is None:
            return None
        block_side, block_coord, wall = blocked
        for r, c in wall:
            grid[r, c] = WALL
            wall_cells.add((r, c))

        forbidden = set(plan["visible_colors"].values())  # type: ignore[union-attr]
        block_color = _choose_color(palette, forbidden, rng)
        token = (block_side, block_coord)
        if token in side_usage:
            return None
        side_usage.add(token)
        plan["blocked"] = {"side": block_side, "coord": block_coord, "color": block_color}

    for plan in lighthouse_plans:
        for side, coord in plan["visible_coords"].items():  # type: ignore[union-attr]
            r, c = _global_side_pos(grid.shape, side, coord)
            if grid[r, c] != SEA:
                return None
            grid[r, c] = plan["visible_colors"][side]  # type: ignore[index]
        blocked = plan["blocked"]  # type: ignore[assignment]
        br, bc = _global_side_pos(grid.shape, blocked["side"], blocked["coord"])
        if grid[br, bc] != SEA:
            return None
        grid[br, bc] = blocked["color"]

    out = main(grid)
    produced = _inspect_output(grid, out, islands)
    if produced != kinds:
        return None
    if not wall_cells:
        return None
    if not _ordered_lighthouses(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid lighthouse/island puzzle")
