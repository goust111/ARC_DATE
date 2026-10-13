from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

MAX_RAY_LEN = 3
CLEARANCE = MAX_RAY_LEN + 1
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _find_fences(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    fences: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set(_border_cells(r0, c0, r1, c1))
        if set(comp) != border:
            continue
        fences.append((r0, c0, r1, c1))
    fences.sort()
    return fences


def _segment_for_cell(r: int, c: int, box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    top = r - (r0 + 1)
    bottom = (r1 - 1) - r
    left = c - (c0 + 1)
    right = (c1 - 1) - c

    best_h = min(top, bottom)
    best_v = min(left, right)
    if best_h <= best_v:
        return "top" if top <= bottom else "bottom"
    return "left" if left <= right else "right"


def _segment_cells(box: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    out = {side: [] for side in SIDES}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            out[_segment_for_cell(r, c, box)].append((r, c))
    return out


def _ray_spec(side: str, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "top":
        return r0 - 1, mid_c, -1, 0
    if side == "bottom":
        return r1 + 1, mid_c, 1, 0
    if side == "left":
        return mid_r, c0 - 1, 0, -1
    return mid_r, c1 + 1, 0, 1


def _ray_cells(side: str, box: tuple[int, int, int, int], length: int) -> list[tuple[int, int]]:
    sr, sc, dr, dc = _ray_spec(side, box)
    return [(sr + dr * step, sc + dc * step) for step in range(length)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fence segment according to the side nearest to its seeds, then draw
    outward rays whose lengths equal the number of seeds assigned to that side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fences = _find_fences(g)
    assert fences, "expected at least one grey rectangular fence"

    out = g.copy()
    for box in fences:
        seg_cells = _segment_cells(box)
        seed_colors: dict[str, list[int]] = {side: [] for side in SIDES}

        for side, cells in seg_cells.items():
            for r, c in cells:
                val = int(g[r, c])
                if val not in (BLACK, GREY):
                    seed_colors[side].append(val)

        for side in SIDES:
            colors = seed_colors[side]
            if not colors:
                continue
            unique = sorted(set(colors))
            if len(unique) != 1:
                continue

            color = int(unique[0])
            for r, c in seg_cells[side]:
                if int(g[r, c]) == BLACK:
                    out[r, c] = color

            for r, c in _ray_cells(side, box, len(colors)):
                if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                    out[r, c] = color

    return out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sample_spaced_cells(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    if len(candidates) < count:
        return None

    for _ in range(30):
        order = list(range(len(candidates)))
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for idx in order:
            cell = candidates[idx]
            if all(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) > 1 for other in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    picks = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(i)] for i in picks]


def _tie_cells(side: str, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    cells = []
    for r, c in _segment_cells(box)[side]:
        r0, c0, r1, c1 = box
        top = r - (r0 + 1)
        bottom = (r1 - 1) - r
        left = c - (c0 + 1)
        right = (c1 - 1) - c
        if min(top, bottom) == min(left, right):
            cells.append((r, c))
    return cells


def _choose_counts(
    rng: np.random.Generator,
    seg_cells: dict[str, list[tuple[int, int]]],
) -> dict[str, int] | None:
    counts: dict[str, int] = {}
    for side in SIDES:
        limit = len(seg_cells[side])
        if limit <= 0:
            return None
        max_count = min(MAX_RAY_LEN, limit if limit == 1 else limit - 1)
        counts[side] = int(rng.integers(0, max_count + 1))

    if sum(counts.values()) == 0:
        candidates = [side for side in SIDES if len(seg_cells[side]) >= 1]
        chosen = candidates[int(rng.integers(len(candidates)))]
        counts[chosen] = 1

    if all(count <= 1 for count in counts.values()):
        multi_candidates = [
            side
            for side in SIDES
            if min(MAX_RAY_LEN, len(seg_cells[side]) if len(seg_cells[side]) == 1 else len(seg_cells[side]) - 1) >= 2
        ]
        if multi_candidates and rng.random() < 0.65:
            chosen = multi_candidates[int(rng.integers(len(multi_candidates)))]
            counts[chosen] = 2

    return counts


def _pick_side_seeds(
    rng: np.random.Generator,
    box: tuple[int, int, int, int],
    side: str,
    count: int,
) -> list[tuple[int, int]] | None:
    cells = _segment_cells(box)[side]
    if count == 0:
        return []

    forced: list[tuple[int, int]] = []
    if side in ("top", "bottom") and count > 0 and rng.random() < 0.4:
        ties = _tie_cells(side, box)
        if ties:
            forced = [ties[int(rng.integers(len(ties)))]]
            cells = [cell for cell in cells if cell != forced[0]]

    rest = _sample_spaced_cells(rng, cells, count - len(forced))
    if rest is None:
        return None
    return forced + rest


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(18, 31))
    width = int(rng.integers(18, 31))
    n_fences = int(rng.choice(np.array([1, 2, 3]), p=np.array([0.25, 0.45, 0.30])))

    fences: list[tuple[int, int, int, int]] = []
    for _ in range(n_fences):
        placed = False
        for _ in range(120):
            outer_h = int(rng.integers(6, 11))
            outer_w = int(rng.integers(6, 11))
            if height - outer_h - 2 * CLEARANCE < 0 or width - outer_w - 2 * CLEARANCE < 0:
                continue
            r0 = int(rng.integers(CLEARANCE, height - outer_h - CLEARANCE + 1))
            c0 = int(rng.integers(CLEARANCE, width - outer_w - CLEARANCE + 1))
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if all(_boxes_separated(box, other, gap=2) for other in fences):
                fences.append(box)
                placed = True
                break
        if not placed:
            return None

    fence_data: list[dict[str, object]] = []
    static_cells: set[tuple[int, int]] = set()
    for box in fences:
        for cell in _border_cells(*box):
            static_cells.add(cell)

    for box in fences:
        seg_cells = _segment_cells(box)
        counts = _choose_counts(rng, seg_cells)
        if counts is None:
            return None

        occupied_sides = [side for side in SIDES if counts[side] > 0]
        colors = rng.choice(np.array(PALETTE), size=len(occupied_sides), replace=False)
        side_colors = {side: int(color) for side, color in zip(occupied_sides, colors)}

        side_seeds: dict[str, list[tuple[int, int]]] = {}
        for side in SIDES:
            picks = _pick_side_seeds(rng, box, side, counts[side])
            if picks is None:
                return None
            side_seeds[side] = picks
            for cell in picks:
                if cell in static_cells:
                    return None
                static_cells.add(cell)

        fence_data.append({"box": box, "counts": counts, "side_colors": side_colors, "side_seeds": side_seeds})

    planned_rays: dict[tuple[int, int], int] = {}
    for item in fence_data:
        box = item["box"]
        counts = item["counts"]
        side_colors = item["side_colors"]
        for side in SIDES:
            count = counts[side]
            if count <= 0:
                continue
            color = side_colors[side]
            for cell in _ray_cells(side, box, count):
                r, c = cell
                if not (0 <= r < height and 0 <= c < width):
                    return None
                if cell in static_cells:
                    return None
                prev = planned_rays.get(cell)
                if prev is not None and prev != color:
                    return None
                planned_rays[cell] = color

    grid = np.full((height, width), BLACK, dtype=int)
    for box in fences:
        for r, c in _border_cells(*box):
            grid[r, c] = GREY

    for item in fence_data:
        side_colors = item["side_colors"]
        side_seeds = item["side_seeds"]
        for side in SIDES:
            if side not in side_colors:
                continue
            color = side_colors[side]
            for r, c in side_seeds[side]:
                grid[r, c] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
