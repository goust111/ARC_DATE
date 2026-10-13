import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
ARROW = 9
RAY_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
MISSING_BY_DIR = {
    "up": (0, 0),
    "right": (0, 1),
    "down": (1, 1),
    "left": (1, 0),
}
DIR_FROM_MISSING = {v: k for k, v in MISSING_BY_DIR.items()}
SHAPE_BY_DIR = {
    "up": frozenset({(0, 1), (1, 0), (1, 1)}),
    "right": frozenset({(0, 0), (1, 0), (1, 1)}),
    "down": frozenset({(0, 0), (0, 1), (1, 0)}),
    "left": frozenset({(0, 0), (0, 1), (1, 1)}),
}
FULL_2X2 = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})

GEN_RETRIES = 500
REGION_RETRIES = 80


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR_VECS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...] | frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _wall_lines(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    wall_rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] != bg))]
    wall_cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] != bg))]
    return wall_rows, wall_cols


def _region_boxes(wall_rows: list[int], wall_cols: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(len(wall_rows) - 1):
        r0, r1 = int(wall_rows[i]), int(wall_rows[i + 1])
        if r1 - r0 <= 1:
            continue
        for j in range(len(wall_cols) - 1):
            c0, c1 = int(wall_cols[j]), int(wall_cols[j + 1])
            if c1 - c0 <= 1:
                continue
            boxes.append((r0, c0, r1, c1))
    return boxes


def _arrow_info(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    assert len(coords) == 3, "arrow must contain 3 cells"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 == 1 and c1 - c0 == 1, "arrow must fit inside a 2x2 box"
    norm = frozenset((r - r0, c - c0) for r, c in coords)
    missing = list(FULL_2X2 - norm)
    assert len(missing) == 1, "arrow must miss exactly one corner"
    miss = missing[0]
    assert miss in DIR_FROM_MISSING, "unexpected missing corner"
    return DIR_FROM_MISSING[miss], (r0 + miss[0], c0 + miss[1])


def _region_emphasis_color(grid: np.ndarray, box: tuple[int, int, int, int], wall_color: int) -> int:
    r0, c0, r1, c1 = box
    segments = [
        grid[r0, c0 + 1 : c1],
        grid[r1, c0 + 1 : c1],
        grid[r0 + 1 : r1, c0],
        grid[r0 + 1 : r1, c1],
    ]
    found: set[int] = set()
    for seg in segments:
        for value in np.unique(seg):
            color = int(value)
            if color != wall_color:
                found.add(color)
    assert len(found) == 1, "each region should see exactly one emphasized border color"
    return next(iter(found))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    wall_rows, wall_cols = _wall_lines(g, bg)
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected partition walls"

    wall_mask = np.zeros_like(g, dtype=bool)
    wall_mask[wall_rows, :] = True
    wall_mask[:, wall_cols] = True
    wall_color = Counter(map(int, g[wall_mask])).most_common(1)[0][0]

    out = g.copy()
    for box in _region_boxes(wall_rows, wall_cols):
        r0, c0, r1, c1 = box
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        comps = _components(inner != bg)
        assert comps, "each region must contain at least one arrow"

        arrows = []
        directions = set()
        for comp in comps:
            global_comp = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in comp]
            direction, base = _arrow_info(global_comp)
            directions.add(direction)
            arrows.append({"base": base, "direction": direction})

        assert len(directions) == 1, "all arrows in a region must point the same way"
        direction = next(iter(directions))
        ray_color = _region_emphasis_color(g, box, wall_color)
        ray_len = len(arrows)
        dr, dc = DIR_VECS[direction]

        for arrow in arrows:
            br, bc = arrow["base"]
            for step in range(1, ray_len + 1):
                rr = br + dr * step
                cc = bc + dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if int(g[rr, cc]) != bg:
                    break
                out[rr, cc] = ray_color

    return out


def _axis_sizes(count: int, rng: random.Random) -> list[int]:
    if count == 1:
        lo, hi = 6, 9
    elif count == 2:
        lo, hi = 5, 8
    elif count == 3:
        lo, hi = 5, 7
    else:
        lo, hi = 5, 6
    return [rng.randint(lo, hi) for _ in range(count)]


def _choose_layout(rng: random.Random) -> tuple[int, int, list[int], list[int]]:
    mode = rng.choice(("row_strip", "col_strip"))
    if mode == "row_strip":
        rows, cols = 1, rng.randint(2, 4)
    else:
        rows, cols = rng.randint(2, 4), 1

    row_heights = _axis_sizes(rows, rng)
    col_widths = _axis_sizes(cols, rng)
    return rows, cols, row_heights, col_widths


def _build_canvas(rows: int, cols: int, row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    total_h = sum(row_heights) + rows + 1
    total_w = sum(col_widths) + cols + 1

    grid = np.full((total_h, total_w), BG, dtype=int)
    wall_rows = [0]
    wall_cols = [0]

    pos = 0
    for height in row_heights:
        pos += height + 1
        wall_rows.append(pos)

    pos = 0
    for width in col_widths:
        pos += width + 1
        wall_cols.append(pos)

    for r in wall_rows:
        grid[r, :] = WALL
    for c in wall_cols:
        grid[:, c] = WALL

    return grid, wall_rows, wall_cols


def _outer_sides(ri: int, ci: int, rows: int, cols: int) -> list[str]:
    sides: list[str] = []
    if ri == 0:
        sides.append("top")
    if ri == rows - 1:
        sides.append("bottom")
    if ci == 0:
        sides.append("left")
    if ci == cols - 1:
        sides.append("right")
    return sides


def _paint_side(grid: np.ndarray, box: tuple[int, int, int, int], side: str, color: int) -> None:
    r0, c0, r1, c1 = box
    if side == "top":
        grid[r0, c0 + 1 : c1] = color
    elif side == "bottom":
        grid[r1, c0 + 1 : c1] = color
    elif side == "left":
        grid[r0 + 1 : r1, c0] = color
    else:
        grid[r0 + 1 : r1, c1] = color


def _free_to_wall(base: tuple[int, int], direction: str, h: int, w: int) -> int:
    dr, dc = DIR_VECS[direction]
    rr, cc = base
    steps = 0
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            return steps
        steps += 1


def _arrow_candidates(h: int, w: int, direction: str) -> list[dict]:
    shape = SHAPE_BY_DIR[direction]
    miss_r, miss_c = MISSING_BY_DIR[direction]
    out: list[dict] = []
    next_id = 0
    for r in range(h - 1):
        for c in range(w - 1):
            cells = frozenset((r + dr, c + dc) for dr, dc in shape)
            base = (r + miss_r, c + miss_c)
            out.append(
                {
                    "id": next_id,
                    "cells": cells,
                    "base": base,
                    "free_to_wall": _free_to_wall(base, direction, h, w),
                }
            )
            next_id += 1
    return out


def _arrow_conflict(a: dict, b: dict) -> bool:
    if a["id"] == b["id"]:
        return True
    if a["base"] == b["base"]:
        return True
    if a["base"] in b["cells"] or b["base"] in a["cells"]:
        return True
    if a["cells"] & b["cells"]:
        return True
    for ra, ca in a["cells"]:
        for rb, cb in b["cells"]:
            if abs(ra - rb) + abs(ca - cb) == 1:
                return True
    return False


def _ray_outcomes(chosen: list[dict], direction: str, ray_len: int, h: int, w: int) -> list[dict]:
    occupied = set()
    for arrow in chosen:
        occupied.update(arrow["cells"])

    dr, dc = DIR_VECS[direction]
    outcomes: list[dict] = []
    for arrow in chosen:
        rr, cc = arrow["base"]
        painted = 0
        reason = "full"
        for step in range(1, ray_len + 1):
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                reason = "wall"
                break
            if (rr, cc) in occupied:
                reason = "arrow"
                break
            painted = step
        else:
            painted = ray_len
        outcomes.append({"length": painted, "reason": reason})
    return outcomes


def _branch_of(outcomes: list[dict], ray_len: int) -> str:
    if any(item["reason"] == "arrow" for item in outcomes):
        return "arrow"
    if any(item["reason"] == "wall" and item["length"] < ray_len for item in outcomes):
        return "wall"
    return "full"


def _search_nonconflicting(
    pool: list[dict],
    target_n: int,
    rng: random.Random,
    validator,
    initial: list[dict] | None = None,
) -> list[dict] | None:
    initial = [] if initial is None else list(initial)
    if len(initial) > target_n:
        return None
    initial_ids = {item["id"] for item in initial}
    for _ in range(REGION_RETRIES):
        chosen = list(initial)
        rest = list(pool)
        rng.shuffle(rest)
        for cand in rest:
            if cand["id"] in initial_ids:
                continue
            if all(not _arrow_conflict(cand, other) for other in chosen):
                chosen.append(cand)
                if len(chosen) == target_n:
                    break
        if len(chosen) != target_n:
            continue
        if validator(chosen):
            return chosen
    return None


def _search_full(cands: list[dict], ray_len: int, direction: str, h: int, w: int, rng: random.Random) -> list[dict] | None:
    pool = [cand for cand in cands if cand["free_to_wall"] >= ray_len]
    if len(pool) < ray_len:
        return None
    return _search_nonconflicting(
        pool,
        ray_len,
        rng,
        lambda chosen: all(item["reason"] == "full" for item in _ray_outcomes(chosen, direction, ray_len, h, w)),
    )


def _search_wall(cands: list[dict], ray_len: int, direction: str, h: int, w: int, rng: random.Random) -> list[dict] | None:
    wallers = [cand for cand in cands if 1 <= cand["free_to_wall"] < ray_len]
    pool = [cand for cand in cands if cand["free_to_wall"] >= 1]
    if not wallers or len(pool) < ray_len:
        return None
    rng.shuffle(wallers)
    for waller in wallers[:20]:
        chosen = _search_nonconflicting(
            pool,
            ray_len,
            rng,
            lambda trial: (
                (lambda outcomes: _branch_of(outcomes, ray_len) == "wall" and any(item["length"] > 0 for item in outcomes))(
                    _ray_outcomes(trial, direction, ray_len, h, w)
                )
            ),
            initial=[waller],
        )
        if chosen is not None:
            return chosen
    return None


def _hit_step(shooter: dict, blocker: dict, direction: str, ray_len: int, h: int, w: int) -> int | None:
    dr, dc = DIR_VECS[direction]
    rr, cc = shooter["base"]
    for step in range(1, ray_len + 1):
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if (rr, cc) in blocker["cells"]:
            return step
    return None


def _search_arrow(cands: list[dict], ray_len: int, direction: str, h: int, w: int, rng: random.Random) -> list[dict] | None:
    pairs: list[tuple[dict, dict]] = []
    fallback_pairs: list[tuple[dict, dict]] = []
    for shooter in cands:
        for blocker in cands:
            if _arrow_conflict(shooter, blocker):
                continue
            step = _hit_step(shooter, blocker, direction, ray_len, h, w)
            if step is None:
                continue
            if step >= 2:
                pairs.append((shooter, blocker))
            else:
                fallback_pairs.append((shooter, blocker))
    if not pairs:
        pairs = fallback_pairs
    if not pairs:
        return None

    rng.shuffle(pairs)
    for shooter, blocker in pairs[:40]:
        chosen = _search_nonconflicting(
            cands,
            ray_len,
            rng,
            lambda trial: (
                (lambda outcomes: _branch_of(outcomes, ray_len) == "arrow" and any(item["length"] > 0 for item in outcomes))(
                    _ray_outcomes(trial, direction, ray_len, h, w)
                )
            ),
            initial=[shooter, blocker],
        )
        if chosen is not None:
            return chosen
    return None


def _place_region_arrows(h: int, w: int, target_branch: str, rng: random.Random) -> dict | None:
    directions = list(DIR_VECS.keys())
    rng.shuffle(directions)
    for direction in directions:
        cands = _arrow_candidates(h, w, direction)
        max_n = min(4, len(cands))
        if max_n < 1:
            continue
        ray_lengths = list(range(1, max_n + 1))
        if target_branch in {"wall", "arrow"}:
            ray_lengths = [n for n in ray_lengths if n >= 2]
        rng.shuffle(ray_lengths)
        for ray_len in ray_lengths:
            if target_branch == "full":
                chosen = _search_full(cands, ray_len, direction, h, w, rng)
            elif target_branch == "wall":
                chosen = _search_wall(cands, ray_len, direction, h, w, rng)
            else:
                chosen = _search_arrow(cands, ray_len, direction, h, w, rng)
            if chosen is None:
                continue
            outcomes = _ray_outcomes(chosen, direction, ray_len, h, w)
            return {
                "direction": direction,
                "count": ray_len,
                "arrows": chosen,
                "branch": _branch_of(outcomes, ray_len),
                "outcomes": outcomes,
            }
    return None


def _branch_targets(n_regions: int, rng: random.Random) -> list[str]:
    targets: list[str] = []
    base = ["full", "wall", "arrow"]
    if n_regions >= 3:
        rng.shuffle(base)
        targets.extend(base)
    else:
        targets.extend(rng.sample(base, n_regions))
    while len(targets) < n_regions:
        targets.append(rng.choice(base))
    rng.shuffle(targets)
    return targets


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    for spec in specs:
        dr, dc = DIR_VECS[spec["direction"]]
        ray_len = len(spec["arrows"])
        for arrow in spec["arrows"]:
            br, bc = arrow["base_abs"]
            for step in range(1, ray_len + 1):
                rr = br + dr * step
                cc = bc + dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if int(grid[rr, cc]) != BG:
                    break
                out[rr, cc] = spec["color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols, row_heights, col_widths = _choose_layout(rng)
    grid, wall_rows, wall_cols = _build_canvas(rows, cols, row_heights, col_widths)
    boxes = _region_boxes(wall_rows, wall_cols)
    n_regions = len(boxes)
    if n_regions < 2:
        return None

    palette_size = rng.randint(2, min(5, len(RAY_COLORS), n_regions))
    palette = rng.sample(RAY_COLORS, palette_size)
    branch_targets = _branch_targets(n_regions, rng)
    specs: list[dict] = []

    for idx, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        h = r1 - r0 - 1
        w = c1 - c0 - 1
        placed = _place_region_arrows(h, w, branch_targets[idx], rng)
        if placed is None:
            return None

        ri = next(i for i in range(len(wall_rows) - 1) if wall_rows[i] == r0 and wall_rows[i + 1] == r1)
        ci = next(j for j in range(len(wall_cols) - 1) if wall_cols[j] == c0 and wall_cols[j + 1] == c1)
        side = rng.choice(_outer_sides(ri, ci, rows, cols))
        color = rng.choice(palette)
        _paint_side(grid, box, side, color)

        abs_arrows = []
        for arrow in placed["arrows"]:
            for rr, cc in arrow["cells"]:
                grid[r0 + 1 + rr, c0 + 1 + cc] = ARROW
            abs_arrows.append(
                {
                    "cells_abs": frozenset((r0 + 1 + rr, c0 + 1 + cc) for rr, cc in arrow["cells"]),
                    "base_abs": (r0 + 1 + arrow["base"][0], c0 + 1 + arrow["base"][1]),
                }
            )

        specs.append(
            {
                "box": box,
                "side": side,
                "color": color,
                "direction": placed["direction"],
                "branch": placed["branch"],
                "arrows": abs_arrows,
            }
        )

    expected = _render_expected(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    counts = [len(spec["arrows"]) for spec in specs]
    branches = {spec["branch"] for spec in specs}
    directions = {spec["direction"] for spec in specs}
    colors = {spec["color"] for spec in specs}

    if max(counts) < 2:
        return None
    if n_regions >= 2 and len(branches) < 2:
        return None
    if n_regions >= 3 and not {"full", "wall", "arrow"}.issubset(branches):
        return None
    if len(directions) < 2:
        return None
    if len(colors) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
