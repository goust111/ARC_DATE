from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
GREEN = 3

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
SHAPE_FAMILIES = ("u", "bottom_narrow", "bottom_wide")


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _row_runs(cols: list[int]) -> list[tuple[int, int]]:
    ordered = sorted(int(c) for c in cols)
    if not ordered:
        return []
    runs: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for col in ordered[1:]:
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    return runs


def _extract_container(coords: np.ndarray) -> dict[str, object] | None:
    by_row: dict[int, list[int]] = {}
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        by_row.setdefault(r, []).append(c)

    interior_rows: list[tuple[int, int, int]] = []
    for r in sorted(by_row):
        runs = _row_runs(by_row[r])
        if len(runs) < 2:
            continue
        left_wall = runs[0][1]
        right_wall = runs[-1][0]
        if right_wall - left_wall <= 1:
            continue
        interior_rows.append((int(r), int(left_wall + 1), int(right_wall - 1)))

    if not interior_rows:
        return None

    row_ids = [r for r, _c0, _c1 in interior_rows]
    if row_ids != list(range(row_ids[0], row_ids[-1] + 1)):
        return None

    bottom_r, bottom_c0, bottom_c1 = interior_rows[-1]
    floor_r = bottom_r + 1
    floor_runs = _row_runs(by_row.get(floor_r, []))
    need_left = bottom_c0 - 1
    need_right = bottom_c1 + 1
    closed = any(run_c0 <= need_left and run_c1 >= need_right for run_c0, run_c1 in floor_runs)
    if not closed:
        return None

    top_r, open_c0, open_c1 = interior_rows[0]
    return {
        "rows": interior_rows,
        "top_row": int(top_r),
        "opening": (int(open_c0), int(open_c1)),
    }


def _find_containers(grid: np.ndarray) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for comp in _components(grid == GREEN, connectivity=4):
        container = _extract_container(comp)
        if container is not None:
            containers.append(container)
    containers.sort(key=lambda item: (int(item["top_row"]), int(item["opening"][0])))
    return containers


def _find_sources(grid: np.ndarray) -> list[dict[str, int]]:
    sources: list[dict[str, int]] = []
    for comp in _components(grid == BLUE, connectivity=4):
        rows = sorted({int(r) for r, _c in comp})
        if len(rows) != 1:
            continue
        cols = sorted(int(c) for _r, c in comp)
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        sources.append(
            {
                "row": int(rows[0]),
                "c0": int(cols[0]),
                "c1": int(cols[-1]),
                "count": int(len(cols)),
            }
        )
    sources.sort(key=lambda item: (int(item["row"]), int(item["c0"])))
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each open-top green container from the bottom up using the length of the
    blue source segment floating above its opening.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    containers = _find_containers(g)
    sources = _find_sources(g)

    for src in sources:
        best_idx = None
        best_key = None
        for idx, container in enumerate(containers):
            open_c0, open_c1 = container["opening"]
            overlap = min(int(src["c1"]), int(open_c1)) - max(int(src["c0"]), int(open_c0)) + 1
            if overlap <= 0 or int(src["row"]) >= int(container["top_row"]):
                continue
            gap = int(container["top_row"]) - int(src["row"])
            key = (int(overlap), -int(gap))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            continue

        rows = containers[best_idx]["rows"]
        fill_rows = min(int(src["count"]), len(rows))
        for row, c0, c1 in rows[-fill_rows:]:
            out[int(row), int(c0) : int(c1) + 1] = BLUE

    return out


def _positive_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    prev = 0
    result: list[int] = []
    for cut in cuts + [int(total)]:
        result.append(int(cut - prev))
        prev = cut
    return result


def _monotone_sequence(start: int, end: int, length: int, rng: np.random.Generator) -> list[int]:
    delta = int(end - start)
    if delta == 0 or length == 1:
        return [int(start)] * int(length)

    sign = 1 if delta > 0 else -1
    amount = abs(delta)
    max_changes = min(amount, length - 1)
    num_changes = int(rng.integers(1, max_changes + 1))
    rows = sorted(int(x) for x in rng.choice(np.arange(1, length), size=num_changes, replace=False))
    parts = _positive_composition(amount, num_changes, rng)

    vals = np.full(length, int(start), dtype=int)
    for row, part in zip(rows, parts):
        vals[row:] += sign * int(part)
    return [int(v) for v in vals]


def _split_extra(extra: int, rng: np.random.Generator) -> tuple[int, int]:
    if extra <= 0:
        return 0, 0
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return int(extra), 0
    if mode == 1:
        return 0, int(extra)
    if extra == 1:
        return 1, 0
    left = int(rng.integers(1, extra))
    return left, int(extra - left)


def _build_container_spec(rng: np.random.Generator) -> dict[str, object]:
    overflow = bool(int(rng.integers(0, 2)))
    if overflow:
        depth = int(rng.integers(2, 5))
        source_len = int(rng.integers(depth + 1, 6))
    else:
        depth = int(rng.integers(2, 7))
        source_len = int(rng.integers(1, min(depth, 5) + 1))

    family = SHAPE_FAMILIES[int(rng.integers(0, len(SHAPE_FAMILIES)))]

    if family == "u":
        width = int(rng.integers(max(2, source_len), 10))
        lefts = [0] * depth
        rights = [width + 1] * depth
    elif family == "bottom_narrow":
        top_width = int(rng.integers(max(3, source_len), 10))
        extra = int(rng.integers(1, min(4, top_width - 1) + 1))
        bottom_width = int(top_width - extra)
        left_extra, right_extra = _split_extra(extra, rng)
        lefts = _monotone_sequence(-left_extra, 0, depth, rng)
        rights = _monotone_sequence(bottom_width + 1 + right_extra, bottom_width + 1, depth, rng)
    else:
        top_width = int(rng.integers(max(2, source_len), 7))
        extra = int(rng.integers(1, 5))
        left_extra, right_extra = _split_extra(extra, rng)
        lefts = _monotone_sequence(0, -left_extra, depth, rng)
        rights = _monotone_sequence(top_width + 1, top_width + 1 + right_extra, depth, rng)

    shift = -min(lefts)
    lefts = [int(v + shift) for v in lefts]
    rights = [int(v + shift) for v in rights]

    wall_cells: set[tuple[int, int]] = set()
    for row in range(depth):
        wall_cells.add((row, lefts[row]))
        wall_cells.add((row, rights[row]))

    for row in range(depth - 1):
        if lefts[row] != lefts[row + 1]:
            bridge_row = row if lefts[row] > lefts[row + 1] else row + 1
            for col in range(min(lefts[row], lefts[row + 1]), max(lefts[row], lefts[row + 1]) + 1):
                wall_cells.add((bridge_row, col))
        if rights[row] != rights[row + 1]:
            bridge_row = row if rights[row] < rights[row + 1] else row + 1
            for col in range(min(rights[row], rights[row + 1]), max(rights[row], rights[row + 1]) + 1):
                wall_cells.add((bridge_row, col))

    for col in range(lefts[-1], rights[-1] + 1):
        wall_cells.add((depth, col))

    top_open_c0 = lefts[0] + 1
    top_open_c1 = rights[0] - 1
    source_gap = int(rng.integers(1, 4))
    source_row = -source_gap - 1
    source_start = int(rng.integers(top_open_c0, top_open_c1 - source_len + 2))
    source_cells = [(source_row, col) for col in range(source_start, source_start + source_len)]

    interior_rows = [(row, lefts[row] + 1, rights[row] - 1) for row in range(depth)]
    all_cells = list(wall_cells) + source_cells
    rows = [r for r, _c in all_cells]
    cols = [c for _r, c in all_cells]

    return {
        "cells": all_cells,
        "wall_cells": list(wall_cells),
        "source_cells": source_cells,
        "source_len": int(source_len),
        "interior_rows": interior_rows,
        "bbox": (int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))),
    }


def _mark_padding(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _place_objects(
    specs: list[dict[str, object]],
    height: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    placed_specs: list[dict[str, object]] = []

    for spec in specs:
        min_r, min_c, max_r, max_c = spec["bbox"]
        row_lo = -int(min_r)
        row_hi = height - int(max_r)
        col_lo = -int(min_c)
        col_hi = width - int(max_c)
        if row_lo >= row_hi or col_lo >= col_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            dr = int(rng.integers(row_lo, row_hi))
            dc = int(rng.integers(col_lo, col_hi))
            shifted_cells = [(int(r + dr), int(c + dc)) for r, c in spec["cells"]]
            if any(occupied[r, c] for r, c in shifted_cells):
                continue

            wall_cells = [(int(r + dr), int(c + dc)) for r, c in spec["wall_cells"]]
            source_cells = [(int(r + dr), int(c + dc)) for r, c in spec["source_cells"]]
            for r, c in wall_cells:
                grid[r, c] = GREEN
            for r, c in source_cells:
                grid[r, c] = BLUE

            _mark_padding(occupied, shifted_cells)
            placed_specs.append(
                {
                    "source_len": int(spec["source_len"]),
                    "interior_rows": [
                        (int(row + dr), int(c0 + dc), int(c1 + dc))
                        for row, c0, c1 in spec["interior_rows"]
                    ],
                }
            )
            placed = True
            break

        if not placed:
            return None

    return grid, placed_specs


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        rows = spec["interior_rows"]
        fill_rows = min(int(spec["source_len"]), len(rows))
        for row, c0, c1 in rows[-fill_rows:]:
            out[int(row), int(c0) : int(c1) + 1] = BLUE
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(14, 25))
    width = int(rng.integers(18, 33))
    num_objects = int(rng.integers(1, 5))

    specs = [_build_container_spec(rng) for _ in range(num_objects)]
    placed = _place_objects(specs, height, width, rng)
    if placed is None:
        return None
    grid, placed_specs = placed

    containers = _find_containers(grid)
    sources = _find_sources(grid)
    if len(containers) != num_objects or len(sources) != num_objects:
        return None

    for src in sources:
        matches = 0
        for container in containers:
            open_c0, open_c1 = container["opening"]
            overlap = min(int(src["c1"]), int(open_c1)) - max(int(src["c0"]), int(open_c0)) + 1
            if overlap > 0 and int(src["row"]) < int(container["top_row"]):
                matches += 1
        if matches != 1:
            return None

    expected = _expected_output(grid, placed_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-filling input")
