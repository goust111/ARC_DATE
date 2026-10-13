import random
import sys
from pathlib import Path

import numpy as np

ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(ARC_ROOT) not in sys.path:
    sys.path.append(str(ARC_ROOT))

from solvers.arc_utils import find_components, get_bg


BG = 0
PIVOT = 1
SPRITE_COLORS = [2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (12, 16)
GRID_W_RANGE = (12, 16)
PAIR_COUNT_RANGE = (3, 5)
AREA_RANGE = (3, 6)
GEN_RETRIES = 400
PLACE_RETRIES = 120


def _center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + r1, c0 + c1


def _matching_count(sprites: list[dict], pivots: list[tuple[int, int]]) -> int:
    candidates: dict[int, list[tuple[int, str]]] = {}
    for idx, sprite in enumerate(sprites):
        coords = np.asarray(sprite["coords"], dtype=int)
        rows = {int(r) for r in coords[:, 0]}
        cols = {int(c) for c in coords[:, 1]}
        options: list[tuple[int, str]] = []
        for j, (pr, pc) in enumerate(pivots):
            same_row = int(pr) in rows
            same_col = int(pc) in cols
            if same_row ^ same_col:
                options.append((j, "h" if same_row else "v"))
        if not options:
            return 0
        candidates[idx] = options

    ordered = sorted(candidates, key=lambda idx: (len(candidates[idx]), sprites[idx]["bbox"][0], sprites[idx]["bbox"][1]))
    count = 0

    def search(i: int, used: set[int]) -> None:
        nonlocal count
        if count > 1:
            return
        if i == len(ordered):
            count += 1
            return
        sprite_idx = ordered[i]
        for pivot_idx, _axis in candidates[sprite_idx]:
            if pivot_idx in used:
                continue
            used.add(pivot_idx)
            search(i + 1, used)
            used.remove(pivot_idx)

    search(0, set())
    return count


def _unique_pairs(sprites: list[dict], pivots: list[tuple[int, int]]) -> list[dict]:
    candidates: dict[int, list[tuple[int, str]]] = {}
    for idx, sprite in enumerate(sprites):
        coords = np.asarray(sprite["coords"], dtype=int)
        rows = {int(r) for r in coords[:, 0]}
        cols = {int(c) for c in coords[:, 1]}
        options: list[tuple[int, str]] = []
        for j, (pr, pc) in enumerate(pivots):
            same_row = int(pr) in rows
            same_col = int(pc) in cols
            if same_row ^ same_col:
                options.append((j, "h" if same_row else "v"))
        assert options, "each sprite must align with at least one pivot"
        candidates[idx] = options

    ordered = sorted(candidates, key=lambda idx: (len(candidates[idx]), sprites[idx]["bbox"][0], sprites[idx]["bbox"][1]))
    solutions: list[dict[int, tuple[int, str]]] = []

    def search(i: int, used: set[int], assign: dict[int, tuple[int, str]]) -> None:
        if len(solutions) > 1:
            return
        if i == len(ordered):
            solutions.append(assign.copy())
            return
        sprite_idx = ordered[i]
        for pivot_idx, axis in candidates[sprite_idx]:
            if pivot_idx in used:
                continue
            used.add(pivot_idx)
            assign[sprite_idx] = (pivot_idx, axis)
            search(i + 1, used, assign)
            del assign[sprite_idx]
            used.remove(pivot_idx)

    search(0, set(), {})
    assert len(solutions) == 1, "expected a unique one-to-one sprite/pivot matching"

    pairs: list[dict] = []
    for sprite_idx, sprite in enumerate(sprites):
        pivot_idx, axis = solutions[0][sprite_idx]
        pr, pc = pivots[pivot_idx]
        center2_r, center2_c = _center2(sprite["bbox"])
        pairs.append(
            {
                "sprite_idx": int(sprite_idx),
                "pivot_idx": int(pivot_idx),
                "axis": axis,
                "pivot": (int(pr), int(pc)),
                "color": int(sprite["color"]),
                "coords": np.asarray(sprite["coords"], dtype=int),
                "bbox": tuple(map(int, sprite["bbox"])),
                "center2": (int(center2_r), int(center2_c)),
                "distance2": int(abs(center2_r - 2 * pr) + abs(center2_c - 2 * pc)),
            }
        )
    return pairs


def _target_shift(pair: dict) -> int:
    pr, pc = pair["pivot"]
    r0, c0, r1, c1 = pair["bbox"]
    if pair["axis"] == "h":
        if pc < c0:
            return int(pc + 1 - c0)
        return int(pc - 1 - c1)
    if pr < r0:
        return int(pr + 1 - r0)
    return int(pr - 1 - r1)


def _shift_coords(coords: np.ndarray, axis: str, delta: int) -> np.ndarray:
    if axis == "h":
        return coords + np.array([0, int(delta)], dtype=int)
    return coords + np.array([int(delta), 0], dtype=int)


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    sprites = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=SPRITE_COLORS, return_format="dict")
    pivots = [(int(r), int(c)) for r, c in np.argwhere(g == PIVOT)]
    assert len(sprites) == len(pivots) > 0, "sprite/pivot counts must match and be nonzero"

    pairs = _unique_pairs(sprites, pivots)
    ordered = sorted(pairs, key=lambda item: (item["distance2"], item["center2"][0], item["center2"][1]))

    out = np.full_like(g, bg)
    for pr, pc in pivots:
        out[pr, pc] = PIVOT

    occupied = np.zeros_like(g, dtype=bool)
    for pair in ordered:
        target = _target_shift(pair)
        actual = 0
        if target != 0:
            step = 1 if target > 0 else -1
            for delta in range(step, target + step, step):
                moved = _shift_coords(pair["coords"], pair["axis"], delta)
                assert bool(np.all((0 <= moved[:, 0]) & (moved[:, 0] < g.shape[0])))
                assert bool(np.all((0 <= moved[:, 1]) & (moved[:, 1] < g.shape[1])))
                if bool(occupied[moved[:, 0], moved[:, 1]].any()):
                    break
                actual = int(delta)
            else:
                actual = int(target)

        moved = _shift_coords(pair["coords"], pair["axis"], actual)
        assert not bool(occupied[moved[:, 0], moved[:, 1]].any())
        out[moved[:, 0], moved[:, 1]] = int(pair["color"])
        occupied[moved[:, 0], moved[:, 1]] = True
        pair["target_shift"] = int(target)
        pair["actual_shift"] = int(actual)
        pair["blocked"] = bool(actual != target)
        pair["final_coords"] = moved

    for pair in pairs:
        pr, pc = pair["pivot"]
        out[pr, pc] = int(pair["color"])

    return out, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored sprite toward its uniquely aligned blue pivot, letting nearer sprites block farther ones, then recolor the pivots.

    Input structure:
    - The background is black (0).
    - There are 3 to 5 isolated blue pivot pixels (1).
    - There are equally many small monochrome 4-connected sprites of distinct non-blue colors.
    - Each sprite can be paired one-to-one with exactly one pivot because it shares either a row or a column with that pivot, but not both.

    Rule:
    - Compute the unique sprite/pivot pairing from row-or-column alignment.
    - For each pair, measure the Manhattan distance from the pivot to the sprite bbox center and sort pairs by increasing distance; break ties by smaller center row, then smaller center column.
    - In that order, translate the whole sprite along its alignment axis toward the pivot until the nearest bbox edge would become adjacent to the pivot, unless an already-settled sprite is hit first, in which case stop one cell before contact.
    - After all slides, repaint every blue pivot pixel with the color of its paired sprite.

    Key insight:
    The pivots choose both the movement axis and the target side, but the final arrangement also depends on the global distance order because earlier arrivals become solid obstacles for later sprites.
    """
    out, _pairs = _analyze(np.asarray(grid, dtype=int))
    return out


def _random_connected_mask(
    h: int,
    w: int,
    area: int,
    *,
    require_rows: set[int] | None = None,
    require_cols: set[int] | None = None,
) -> np.ndarray | None:
    require_rows = set() if require_rows is None else {int(x) for x in require_rows}
    require_cols = set() if require_cols is None else {int(x) for x in require_cols}
    area = int(area)
    for _ in range(80):
        if require_rows and require_cols:
            seeds = [(r, c) for r in require_rows for c in require_cols if 0 <= r < h and 0 <= c < w]
        elif require_rows:
            seeds = [(r, c) for r in require_rows for c in range(w) if 0 <= r < h]
        elif require_cols:
            seeds = [(r, c) for r in range(h) for c in require_cols if 0 <= c < w]
        else:
            seeds = [(r, c) for r in range(h) for c in range(w)]
        if not seeds:
            return None

        cells = {random.choice(seeds)}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))

        add_frontier(*next(iter(cells)))
        while len(cells) < area and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            cells.add((rr, cc))
            add_frontier(rr, cc)

        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        if any(not bool(mask[r, :].any()) for r in require_rows):
            continue
        if any(not bool(mask[:, c].any()) for c in require_cols):
            continue
        return mask
    return None


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    return {(int(top + r), int(left + c)) for r, c in coords}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                out.add((int(rr), int(cc)))
    return out


def _build_conflict_pair(h: int, w: int, colors: list[int], axis: str, direction: int) -> list[dict] | None:
    if axis == "h":
        base = random.randint(0, h - 3)
        w1 = random.randint(2, 3)
        w2 = random.randint(2, 3)
        area1 = random.randint(max(3, w1), min(5, 2 * w1))
        area2 = random.randint(max(3, w2), min(5, 2 * w2))
        m1 = _random_connected_mask(2, w1, area1, require_rows={0, 1})
        m2 = _random_connected_mask(2, w2, area2, require_rows={0, 1})
        if m1 is None or m2 is None or not bool(m1[0, :].any()) or not bool(m2[1, :].any()):
            return None

        if direction > 0:
            min_left = max(5, w2 + 4)
            max_left = min(w - w1 - 4, w - w2 - 4)
            if min_left > max_left:
                return None
            final_left = random.randint(min_left, max_left)
            d1 = random.randint(2, min(4, final_left - 2))
            d2_low = d1 + w2 + 1
            d2_high = min(d2_low + 3, final_left)
            if d2_low > d2_high:
                return None
            d2 = random.randint(d2_low, d2_high)
            left1 = final_left - d1
            left2 = final_left - d2
            if left2 + w2 > left1 - 1:
                return None
            p1 = (base, final_left + w1)
            p2 = (base + 2, final_left + w2)
        else:
            max_left = min(w - w2 - 5, w - w1 - 5)
            min_left = 1
            if min_left > max_left:
                return None
            final_left = random.randint(min_left, max_left)
            d1 = random.randint(2, min(4, w - (final_left + w1) - 2))
            d2_low = d1 + w1 + 1
            d2_high = min(d2_low + 3, w - (final_left + w2) - 1)
            if d2_low > d2_high:
                return None
            d2 = random.randint(d2_low, d2_high)
            left1 = final_left + d1
            left2 = final_left + d2
            if left1 + w1 > left2 - 1:
                return None
            p1 = (base, final_left - 1)
            p2 = (base + 2, final_left - 1)

        cells1 = _cells_from_mask(m1, base, left1)
        cells2 = _cells_from_mask(m2, base + 1, left2)
    else:
        base = random.randint(0, w - 3)
        h1 = random.randint(2, 3)
        h2 = random.randint(2, 3)
        area1 = random.randint(max(3, h1), min(5, 2 * h1))
        area2 = random.randint(max(3, h2), min(5, 2 * h2))
        m1 = _random_connected_mask(h1, 2, area1, require_cols={0, 1})
        m2 = _random_connected_mask(h2, 2, area2, require_cols={0, 1})
        if m1 is None or m2 is None or not bool(m1[:, 0].any()) or not bool(m2[:, 1].any()):
            return None

        if direction > 0:
            min_top = max(5, h2 + 4)
            max_top = min(h - h1 - 4, h - h2 - 4)
            if min_top > max_top:
                return None
            final_top = random.randint(min_top, max_top)
            d1 = random.randint(2, min(4, final_top - 2))
            d2_low = d1 + h2 + 1
            d2_high = min(d2_low + 3, final_top)
            if d2_low > d2_high:
                return None
            d2 = random.randint(d2_low, d2_high)
            top1 = final_top - d1
            top2 = final_top - d2
            if top2 + h2 > top1 - 1:
                return None
            p1 = (final_top + h1, base)
            p2 = (final_top + h2, base + 2)
        else:
            max_top = min(h - h2 - 5, h - h1 - 5)
            min_top = 1
            if min_top > max_top:
                return None
            final_top = random.randint(min_top, max_top)
            d1 = random.randint(2, min(4, h - (final_top + h1) - 2))
            d2_low = d1 + h1 + 1
            d2_high = min(d2_low + 3, h - (final_top + h2) - 1)
            if d2_low > d2_high:
                return None
            d2 = random.randint(d2_low, d2_high)
            top1 = final_top + d1
            top2 = final_top + d2
            if top1 + h1 > top2 - 1:
                return None
            p1 = (final_top - 1, base)
            p2 = (final_top - 1, base + 2)

        cells1 = _cells_from_mask(m1, top1, base)
        cells2 = _cells_from_mask(m2, top2, base + 1)

    if cells1 & cells2:
        return None
    if p1 in cells1 or p1 in cells2 or p2 in cells1 or p2 in cells2 or p1 == p2:
        return None
    halo1 = _halo(cells1 | {p1}, h, w)
    if halo1 & (cells2 | {p2}):
        return None

    pair1 = {"cells": cells1, "pivot": p1, "color": colors[0]}
    pair2 = {"cells": cells2, "pivot": p2, "color": colors[1]}
    if not _can_place(pair2["cells"], pair2["pivot"], [pair1], h, w):
        return None
    return [pair1, pair2]


def _can_place(cells: set[tuple[int, int]], pivot: tuple[int, int], placed: list[dict], h: int, w: int) -> bool:
    bundle = set(cells) | {pivot}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in bundle):
        return False
    halo = _halo(bundle, h, w)
    rows = {int(r) for r, _c in cells}
    cols = {int(c) for _r, c in cells}
    for item in placed:
        other = set(item["cells"]) | {item["pivot"]}
        if halo & other:
            return False
        other_rows = {int(r) for r, _c in item["cells"]}
        other_cols = {int(c) for _r, c in item["cells"]}
        opr, opc = item["pivot"]
        if int(opr) in rows or int(opc) in cols:
            return False
        if int(pivot[0]) in other_rows or int(pivot[1]) in other_cols:
            return False
    return True


def _random_extra_object(
    h: int,
    w: int,
    color: int,
    axis: str,
    placed: list[dict],
) -> dict | None:
    for _ in range(PLACE_RETRIES):
        if axis == "h":
            ph = random.randint(1, 3)
            pw = random.randint(2, 4)
            max_area = min(AREA_RANGE[1], ph * pw)
            min_area = min(max_area, AREA_RANGE[0])
            if min_area > max_area:
                continue
            area = random.randint(min_area, max_area)
            mask = _random_connected_mask(ph, pw, area, require_rows={random.randrange(ph)})
            if mask is None:
                continue
            occupied_rows = [r for r in range(ph) if bool(mask[r, :].any())]
            align_r = random.choice(occupied_rows)
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            direction = random.choice([-1, 1])
            if direction > 0:
                max_gap = w - (left + pw) - 1
                if max_gap < 2:
                    continue
                gap = random.randint(2, min(6, max_gap))
                pivot = (top + align_r, left + pw - 1 + gap + 1)
            else:
                max_gap = left
                if max_gap < 2:
                    continue
                gap = random.randint(2, min(6, max_gap))
                pivot = (top + align_r, left - gap - 1)
            cells = _cells_from_mask(mask, top, left)
        else:
            ph = random.randint(2, 4)
            pw = random.randint(1, 3)
            max_area = min(AREA_RANGE[1], ph * pw)
            min_area = min(max_area, AREA_RANGE[0])
            if min_area > max_area:
                continue
            area = random.randint(min_area, max_area)
            mask = _random_connected_mask(ph, pw, area, require_cols={random.randrange(pw)})
            if mask is None:
                continue
            occupied_cols = [c for c in range(pw) if bool(mask[:, c].any())]
            align_c = random.choice(occupied_cols)
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            direction = random.choice([-1, 1])
            if direction > 0:
                max_gap = h - (top + ph) - 1
                if max_gap < 2:
                    continue
                gap = random.randint(2, min(6, max_gap))
                pivot = (top + ph - 1 + gap + 1, left + align_c)
            else:
                max_gap = top
                if max_gap < 2:
                    continue
                gap = random.randint(2, min(6, max_gap))
                pivot = (top - gap - 1, left + align_c)
            cells = _cells_from_mask(mask, top, left)

        if not _can_place(cells, pivot, placed, h, w):
            continue
        return {"cells": cells, "pivot": pivot, "color": color}
    return None


def _render(h: int, w: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for item in objects:
        for r, c in item["cells"]:
            grid[r, c] = int(item["color"])
        pr, pc = item["pivot"]
        grid[pr, pc] = PIVOT
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        out, pairs = _analyze(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if not (PAIR_COUNT_RANGE[0] <= len(pairs) <= PAIR_COUNT_RANGE[1]):
        return False

    colors = [int(pair["color"]) for pair in pairs]
    if len(colors) != len(set(colors)):
        return False

    axes = {pair["axis"] for pair in pairs}
    if axes != {"h", "v"}:
        return False

    if not any(bool(pair["blocked"]) for pair in pairs):
        return False
    if not any(not bool(pair["blocked"]) for pair in pairs):
        return False

    if _matching_count(
        find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=SPRITE_COLORS, return_format="dict"),
        [(int(r), int(c)) for r, c in np.argwhere(grid == PIVOT)],
    ) != 1:
        return False

    for pair in pairs:
        coords = np.asarray(pair["coords"], dtype=int)
        if not (AREA_RANGE[0] <= len(coords) <= AREA_RANGE[1]):
            return False
        pr, pc = pair["pivot"]
        rows = {int(r) for r in coords[:, 0]}
        cols = {int(c) for c in coords[:, 1]}
        if ((pr in rows) + (pc in cols)) != 1:
            return False
        if pair["target_shift"] == 0:
            return False

    blobs = list(
        find_components(
            grid,
            bg=BG,
            connectivity=4,
            monochromatic=True,
            colors=SPRITE_COLORS,
            return_format="dict",
        )
    )
    singles = [{"cells": {(int(r), int(c))}, "pivot": None} for r, c in np.argwhere(grid == PIVOT)]
    pieces = [{"cells": {(int(r), int(c)) for r, c in comp["coords"]}, "pivot": None} for comp in blobs]
    every = pieces + singles
    for i, a in enumerate(every):
        halo = _halo(set(a["cells"]), grid.shape[0], grid.shape[1])
        for b in every[i + 1 :]:
            if halo & set(b["cells"]):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(SPRITE_COLORS, count)
    conflict_axis = random.choice(["h", "v"])
    conflict_direction = random.choice([-1, 1])
    placed = _build_conflict_pair(h, w, colors[:2], conflict_axis, conflict_direction)
    if placed is None:
        return None

    extra_axes = ["v" if conflict_axis == "h" else "h"]
    while len(extra_axes) < count - 2:
        extra_axes.append(random.choice(["h", "v"]))
    random.shuffle(extra_axes)

    for i, axis in enumerate(extra_axes, start=2):
        obj = _random_extra_object(h, w, colors[i], axis, placed)
        if obj is None:
            return None
        placed.append(obj)

    grid = _render(h, w, placed)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pivot-drag sliding input")
