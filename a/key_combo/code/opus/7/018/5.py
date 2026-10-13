import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
SEED_COLORS = (1, 3, 4, 6, 7, 8, 9)

ALL_INTERIOR_DIMS = [
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
    (3, 2),
    (4, 2),
]
VISIBLE_ACTIVE_DIMS = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2)]
MAX_GEN_RETRIES = 600


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _find_seeds(grid: np.ndarray) -> list[dict]:
    seeds: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BLACK, GREY)):
        for comp in _components(grid == color, connectivity=8):
            if comp.shape[0] == 1:
                r, c = map(int, comp[0])
                seeds.append({"r": r, "c": c, "color": int(color)})
    seeds.sort(key=lambda seed: (seed["r"], seed["c"], seed["color"]))
    return seeds


def _paint_crosses(grid: np.ndarray, seeds: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    claims: dict[tuple[int, int], list[int]] = {}

    for idx, seed in enumerate(seeds):
        r = seed["r"]
        c = seed["c"]
        for cc in range(w):
            if int(grid[r, cc]) == BLACK:
                claims.setdefault((r, cc), []).append(idx)
        for rr in range(h):
            if int(grid[rr, c]) == BLACK:
                claims.setdefault((rr, c), []).append(idx)

    for (r, c), idxs in claims.items():
        colors = {seeds[idx]["color"] for idx in idxs}
        out[r, c] = RED if len(colors) >= 2 else seeds[idxs[0]]["color"]
    return out


def _enclosed_regions(grid: np.ndarray) -> list[np.ndarray]:
    regions = []
    for comp in _components(grid != GREY, connectivity=4):
        if not _touches_border(comp, grid.shape):
            regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate enclosed grey-ring regions using red cross intersections, then recolor them by the row-contributing seed.

    Input structure:
    - A black background contains 2-3 isolated non-grey seed pixels of distinct colors, with no two seeds sharing a row or column.
    - The same grid also contains 2-3 disjoint grey one-cell-thick closed rings whose interiors are small 4-connected regions.
    - Seed pixels lie outside every ring; ring interiors start as background.

    Rule:
    - Identify every non-black, non-grey singleton seed and record its color and `(row, col)` position.
    - For each seed, recolor every original background cell on that seed's whole row and whole column to the seed color, without overwriting grey ring cells or seed pixels.
    - If a background cell is claimed by two different seeds because it lies at one seed's row and another seed's column, color that cell red.
    - Find every 4-connected non-grey region fully enclosed by grey cells. If such a region contains at least one red cell, determine which seed row passes through that region and recolor every non-red cell in the region to that seed's color while leaving red cells and grey borders unchanged.
    - Enclosed regions without red cells stay exactly as produced by the row/column painting step.

    Key insight:
    - The full row and column extensions only matter because their cross intersections create red activation markers inside holes, and the final fill color is chosen by the seed contributing the row, not the column.
    """
    g = np.asarray(grid, dtype=int)
    seeds = _find_seeds(g)
    painted = _paint_crosses(g, seeds)
    out = painted.copy()

    for region in _enclosed_regions(painted):
        rows = set(int(r) for r in region[:, 0])
        row_seeds = [seed for seed in seeds if seed["r"] in rows]
        values = painted[region[:, 0], region[:, 1]]
        if RED not in set(map(int, values)):
            continue
        if len(row_seeds) != 1:
            continue
        fill_color = row_seeds[0]["color"]
        for r, c in region:
            if int(out[r, c]) != RED:
                out[r, c] = fill_color
    return out


def _sample_spaced_values(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    random.shuffle(values)
    chosen: list[int] = []
    for value in values:
        if all(abs(value - prev) >= gap for prev in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _choose_seeds(h: int, w: int) -> list[dict] | None:
    count = random.choice([2, 2, 3])
    rows = _sample_spaced_values(2, h - 3, count, gap=2)
    cols = _sample_spaced_values(2, w - 3, count, gap=2)
    if rows is None or cols is None:
        return None
    colors = random.sample(SEED_COLORS, count)
    random.shuffle(cols)
    seeds = [{"r": r, "c": c, "color": color} for r, c, color in zip(rows, cols, colors)]
    random.shuffle(seeds)
    return seeds


def _box_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _seed_too_close(seed: dict, box: tuple[int, int, int, int], margin: int = 1) -> bool:
    return (
        box[0] - margin <= seed["r"] <= box[2] + margin
        and box[1] - margin <= seed["c"] <= box[3] + margin
    )


def _spec_matches(
    rows_interior: set[int],
    cols_interior: set[int],
    seeds: list[dict],
    spec: dict,
) -> bool:
    row_ids = [idx for idx, seed in enumerate(seeds) if seed["r"] in rows_interior]
    col_ids = [idx for idx, seed in enumerate(seeds) if seed["c"] in cols_interior]

    kind = spec["kind"]
    if kind in {"active_visible", "active"}:
        return row_ids == [spec["row_seed"]] and col_ids == [spec["col_seed"]]
    if kind == "inactive_row":
        return row_ids == [spec["row_seed"]] and not col_ids
    if kind == "inactive_col":
        return not row_ids and col_ids == [spec["col_seed"]]
    return not row_ids and not col_ids


def _candidate_boxes(h: int, w: int, seeds: list[dict], spec: dict) -> list[dict]:
    dims = list(VISIBLE_ACTIVE_DIMS if spec["kind"] == "active_visible" else ALL_INTERIOR_DIMS)
    random.shuffle(dims)
    candidates: list[dict] = []

    for inner_h, inner_w in dims:
        for ir0 in range(1, h - inner_h):
            ir1 = ir0 + inner_h - 1
            rows_interior = set(range(ir0, ir1 + 1))
            for ic0 in range(1, w - inner_w):
                ic1 = ic0 + inner_w - 1
                cols_interior = set(range(ic0, ic1 + 1))
                if not _spec_matches(rows_interior, cols_interior, seeds, spec):
                    continue
                box = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
                if any(_seed_too_close(seed, box, margin=1) for seed in seeds):
                    continue
                candidates.append(
                    {
                        "inner": (ir0, ic0, ir1, ic1),
                        "box": box,
                        "kind": spec["kind"],
                    }
                )
    random.shuffle(candidates)
    return candidates


def _place_specs(h: int, w: int, seeds: list[dict], specs: list[dict]) -> list[dict] | None:
    candidate_lists = [(_candidate_boxes(h, w, seeds, spec), spec) for spec in specs]
    if any(not candidates for candidates, _ in candidate_lists):
        return None

    order = sorted(range(len(specs)), key=lambda idx: len(candidate_lists[idx][0]))
    placements: list[dict | None] = [None] * len(specs)
    used_boxes: list[tuple[int, int, int, int]] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        spec_idx = order[pos]
        candidates = candidate_lists[spec_idx][0]
        for cand in candidates:
            if any(_box_overlaps(cand["box"], other, margin=1) for other in used_boxes):
                continue
            used_boxes.append(cand["box"])
            placements[spec_idx] = cand
            if backtrack(pos + 1):
                return True
            used_boxes.pop()
            placements[spec_idx] = None
        return False

    if not backtrack(0):
        return None
    return [placement for placement in placements if placement is not None]


def _build_specs(seeds: list[dict], ring_count: int) -> list[dict]:
    count = len(seeds)
    row_seed = random.randrange(count)
    col_seed = random.choice([idx for idx in range(count) if idx != row_seed])
    specs = [{"kind": "active_visible", "row_seed": row_seed, "col_seed": col_seed}]

    if random.random() < 0.5:
        specs.append({"kind": "inactive_row", "row_seed": random.randrange(count)})
    else:
        specs.append({"kind": "inactive_col", "col_seed": random.randrange(count)})

    if ring_count == 3:
        choice = random.random()
        if choice < 0.45:
            other_rows = [idx for idx in range(count) if idx != row_seed] or [row_seed]
            extra_row = random.choice(other_rows)
            extra_col = random.choice([idx for idx in range(count) if idx != extra_row])
            specs.append({"kind": "active", "row_seed": extra_row, "col_seed": extra_col})
        elif choice < 0.80:
            if random.random() < 0.5:
                specs.append({"kind": "inactive_row", "row_seed": random.randrange(count)})
            else:
                specs.append({"kind": "inactive_col", "col_seed": random.randrange(count)})
        else:
            specs.append({"kind": "inactive_empty"})
    return specs


def _paint_ring(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _ring_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components(grid == GREY, connectivity=4)


def _is_rect_ring(comp: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 - 1
    w = c1 - c0 - 1
    if h < 1 or w < 1 or h * w < 3 or h * w > 8:
        return False
    ring = set((int(r), int(c)) for r, c in comp)
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return ring == expected


def _validate_example(grid: np.ndarray) -> bool:
    seeds = _find_seeds(grid)
    if len(seeds) not in (2, 3):
        return False
    if len({seed["color"] for seed in seeds}) != len(seeds):
        return False
    if len({seed["r"] for seed in seeds}) != len(seeds) or len({seed["c"] for seed in seeds}) != len(seeds):
        return False

    rings = _ring_components(grid)
    if len(rings) not in (2, 3) or not all(_is_rect_ring(comp) for comp in rings):
        return False

    painted = _paint_crosses(grid, seeds)
    regions = _enclosed_regions(painted)
    if len(regions) != len(rings):
        return False

    active_count = 0
    inactive_count = 0
    inactive_has_line = False
    active_overwrites_column = False

    out = main(grid)
    for region in regions:
        rows = set(int(r) for r in region[:, 0])
        values = painted[region[:, 0], region[:, 1]]
        output_values = out[region[:, 0], region[:, 1]]
        has_red = bool(np.any(values == RED))
        row_seeds = [seed for seed in seeds if seed["r"] in rows]

        if has_red:
            active_count += 1
            if len(row_seeds) != 1:
                return False
            row_color = row_seeds[0]["color"]
            if not np.all((output_values == RED) | (output_values == row_color)):
                return False
            if np.any((values != RED) & (values != row_color) & (values != BLACK)):
                active_overwrites_column = True
        else:
            inactive_count += 1
            if not np.array_equal(values, output_values):
                return False
            if np.any(values != BLACK):
                inactive_has_line = True

    return active_count >= 1 and inactive_count >= 1 and inactive_has_line and active_overwrites_column


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        h = random.randint(14, 20)
        w = random.randint(14, 20)
        seeds = _choose_seeds(h, w)
        if seeds is None:
            continue

        ring_count = random.choice([2, 3])
        specs = _build_specs(seeds, ring_count)
        placements = _place_specs(h, w, seeds, specs)
        if placements is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for placement in placements:
            _paint_ring(grid, placement["box"])
        for seed in seeds:
            grid[seed["r"], seed["c"]] = seed["color"]

        if not _validate_example(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid cross-activation enclosure example")
