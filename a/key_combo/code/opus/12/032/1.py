import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5

GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _region_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    regions = []
    for comp in _components(grid != GREY):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        regions.append((int(r0), int(c0), int(r1), int(c1)))
    regions.sort()
    return regions


def _simulate_region(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    blue_blocks: bool = False,
    forced_parity: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    blue_cells = np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE)
    assert blue_cells.shape[0] == 1
    br, bc = map(int, blue_cells[0])
    br += r0
    bc += c0

    if forced_parity is None:
        p_r, p_c = br % 2, bc % 2
    else:
        p_r, p_c = forced_parity

    landing_rows: dict[int, list[int]] = {}
    for c in range(c0, c1 + 1):
        rows = []
        for r in range(r0, r1 + 1):
            if r % 2 != p_r or c % 2 != p_c:
                continue
            if r == br and c == bc:
                continue
            rows.append(r)
        landing_rows[c] = rows

    movers = [(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW) if r0 <= r <= r1 and c0 <= c <= c1]
    movers.sort(key=lambda rc: (-rc[0], rc[1]))

    occupied: set[tuple[int, int]] = set()
    settled: list[tuple[int, int]] = []
    for sr, sc in movers:
        candidates = []
        for rr in landing_rows.get(sc, []):
            if rr < sr:
                continue
            if blue_blocks and sc == bc and min(sr, rr) <= br <= max(sr, rr):
                continue
            if (rr, sc) not in occupied:
                candidates.append(rr)
        assert candidates, f"yellow at {(sr, sc)} has no valid landing site in {bbox}"
        dest = (max(candidates), sc)
        occupied.add(dest)
        settled.append(dest)
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop yellow pixels independently inside grey-walled regions onto the parity-defined lattice set by each blue marker.

    Input structure:
    - Black background is split by one-cell-wide grey walls that run across whole rows or columns, creating 3 or 4 rectangular regions.
    - Each region contains exactly one blue marker and several yellow movable pixels above the lower landing area.
    - Walls stay fixed; blue markers stay fixed and only define each region's local parity lattice.

    Rule:
    - Process each grey-walled region independently.
    - Read the absolute blue-marker coordinate `(r_m, c_m)` in that region and use `(r_m % 2, c_m % 2)` as the region's active parity.
    - In that region, only cells on the matching row-and-column parity lattice are valid yellow landing sites; the blue cell itself never blocks and cannot be occupied.
    - For the yellow pixels of that region, process lower ones first and move each straight downward to the deepest still-free landing site in its column.

    Key insight:
    - The blue marker does not paint cells directly; instead its absolute parity turns one checkerboard sublattice into an invisible ground pattern, and each region can have a different ground topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    out[out == YELLOW] = BLACK
    for bbox in _region_bboxes(g):
        for r, c in _simulate_region(g, bbox):
            out[r, c] = YELLOW
    return out


def _bounded_composition(total: int, parts: int, minimum: int, rng: random.Random) -> list[int] | None:
    if total < parts * minimum:
        return None
    vals = [minimum] * parts
    extra = total - parts * minimum
    for _ in range(extra):
        vals[rng.randrange(parts)] += 1
    rng.shuffle(vals)
    return vals


def _build_layout(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    layout = rng.choice(["4grid", "3rows", "3cols"])
    min_region_h = 6
    min_region_w = 5

    if layout == "4grid":
        h = rng.randint(14, 18)
        w = rng.randint(14, 18)
        heights = _bounded_composition(h - 1, 2, min_region_h, rng)
        widths = _bounded_composition(w - 1, 2, min_region_w, rng)
    elif layout == "3rows":
        h = rng.randint(20, 22)
        w = rng.randint(13, 18)
        heights = _bounded_composition(h - 2, 3, min_region_h, rng)
        widths = [w]
    else:
        h = rng.randint(13, 18)
        w = rng.randint(17, 20)
        heights = [h]
        widths = _bounded_composition(w - 2, 3, min_region_w, rng)

    assert heights is not None and widths is not None
    grid = np.full((sum(heights) + len(heights) - 1, sum(widths) + len(widths) - 1), BLACK, dtype=int)

    r = 0
    for i, hh in enumerate(heights):
        if i:
            grid[r - 1, :] = GREY
        r += hh + (1 if i else 0)

    c = 0
    for j, ww in enumerate(widths):
        if j:
            grid[:, c - 1] = GREY
        c += ww + (1 if j else 0)

    regions = []
    r0 = 0
    for i, hh in enumerate(heights):
        c0 = 0
        for j, ww in enumerate(widths):
            regions.append((r0, c0, r0 + hh - 1, c0 + ww - 1))
            c0 += ww + 1
        r0 += hh + 1
    return grid, regions


def _pick_blue(
    bbox: tuple[int, int, int, int],
    rng: random.Random,
    need_cross: bool,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    upper_end = r0 + h // 2 - 1

    blue_rows = []
    for r in range(r0 + 1, r1):
        parity_rows = [rr for rr in range(r0, r1 + 1) if rr % 2 == r % 2]
        upper_rows = [rr for rr in parity_rows if rr <= upper_end and rr != r]
        if len(upper_rows) < 2:
            continue
        if need_cross and not any(rr > r for rr in parity_rows):
            continue
        blue_rows.append(r)
    blue_cols = list(range(c0 + 1, c1))
    assert blue_rows and blue_cols and h >= 6 and w >= 5
    return rng.choice(blue_rows), rng.choice(blue_cols)


def _nonadjacent(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                return False
    return True


def _fill_region(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rng: random.Random,
    *,
    need_stack: bool,
    need_cross: bool,
) -> dict:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    upper_end = r0 + h // 2 - 1
    br, bc = _pick_blue(bbox, rng, need_cross)
    p_r, p_c = br % 2, bc % 2

    parity_rows = [r for r in range(r0, r1 + 1) if r % 2 == p_r]
    parity_cols = [c for c in range(c0, c1 + 1) if c % 2 == p_c]
    input_rows = [r for r in parity_rows if r <= upper_end and r != br]
    assert len(input_rows) >= 2

    landing_rows = {
        c: [r for r in parity_rows if not (r == br and c == bc)]
        for c in parity_cols
    }
    bottom_rows = {c: rows[-1] for c, rows in landing_rows.items() if rows}
    stack_cols = [c for c, rows in landing_rows.items() if len(rows) >= 2 and len([r for r in input_rows if r < rows[-2]]) >= 2]

    max_yellows = min(4, len(parity_cols) + (1 if need_stack else 0), len(input_rows) * len(parity_cols))
    min_yellows = 2
    num_yellows = rng.randint(min_yellows, max_yellows)

    counts = {c: 0 for c in parity_cols}
    if need_stack:
        assert stack_cols
        stack_col = rng.choice(stack_cols)
        counts[stack_col] = 2
    else:
        stack_col = None

    if need_cross:
        assert bc in parity_cols
        counts[bc] = max(counts[bc], 1)

    remaining = num_yellows - sum(counts.values())
    candidate_cols = [c for c in parity_cols if landing_rows[c]]
    while remaining > 0:
        choices = [c for c in candidate_cols if counts[c] == 0 or (c == stack_col and counts[c] < 2)]
        if not choices:
            break
        pick = rng.choice(choices)
        counts[pick] += 1
        remaining -= 1

    if remaining != 0:
        raise AssertionError("failed to allocate yellow columns")

    final_positions: list[tuple[int, int]] = []
    initial_positions: list[tuple[int, int]] = []
    for c in sorted(parity_cols):
        k = counts[c]
        if not k:
            continue
        rows = landing_rows[c]
        assert len(rows) >= k
        chosen_final = rows[-k:]
        final_positions.extend((r, c) for r in chosen_final)

        if need_cross and c == bc:
            usable_inputs = [r for r in input_rows if r < br]
        else:
            usable_inputs = list(input_rows)
        usable_inputs = [r for r in usable_inputs if r < chosen_final[0]]
        if k == 2:
            usable_inputs = [r for r in usable_inputs if r < chosen_final[1]]
        if len(usable_inputs) < k:
            raise AssertionError("not enough reverse-slide rows")
        start_rows = usable_inputs[:]
        rng.shuffle(start_rows)
        start_rows = sorted(start_rows[:k])
        while len(start_rows) == 2 and start_rows[1] - start_rows[0] == 1:
            rng.shuffle(usable_inputs)
            start_rows = sorted(usable_inputs[:k])
        if len(start_rows) == 2 and start_rows[1] - start_rows[0] == 1:
            raise AssertionError("adjacent stacked starts")
        initial_positions.extend((r, c) for r in start_rows)

    if not _nonadjacent(initial_positions):
        raise AssertionError("adjacent input yellows")
    if any(r == br for r, _c in initial_positions):
        raise AssertionError("yellow shares row with blue")
    if any(r > upper_end for r, _c in initial_positions):
        raise AssertionError("yellow not in upper half")
    if need_cross and not any(c == bc and r < br for r, c in initial_positions):
        raise AssertionError("missing blue-cross start")
    if need_cross and not any(c == bc and r > br for r, c in final_positions):
        raise AssertionError("missing blue-cross landing")

    grid[br, bc] = BLUE
    for r, c in initial_positions:
        grid[r, c] = YELLOW

    return {
        "blue": (br, bc),
        "initial": sorted(initial_positions),
        "final": sorted(final_positions),
        "parity": (p_r, p_c),
        "has_stack": any(counts[c] >= 2 for c in counts),
        "has_cross": any(c == bc and r < br for r, c in initial_positions)
        and any(c == bc and r > br for r, c in final_positions),
    }


def _wrong_global_parity(grid: np.ndarray) -> np.ndarray:
    regions = _region_bboxes(grid)
    first_box = regions[0]
    blue_cells = np.argwhere(grid[first_box[0] : first_box[2] + 1, first_box[1] : first_box[3] + 1] == BLUE)
    br, bc = map(int, blue_cells[0])
    forced = ((br + first_box[0]) % 2, (bc + first_box[1]) % 2)

    out = grid.copy()
    out[out == YELLOW] = BLACK
    for bbox in regions:
        for r, c in _simulate_region(grid, bbox, forced_parity=forced):
            out[r, c] = YELLOW
    return out


def _blue_blocking_variant(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[out == YELLOW] = BLACK
    for bbox in _region_bboxes(grid):
        for r, c in _simulate_region(grid, bbox, blue_blocks=True):
            out[r, c] = YELLOW
    return out


def _matches_wrong_semantics(grid: np.ndarray, out: np.ndarray, variant) -> bool:
    try:
        wrong = variant(grid)
    except AssertionError:
        return False
    return np.array_equal(out, wrong)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid, regions = _build_layout(rng)
            region_indices = list(range(len(regions)))
            stack_region = rng.choice(region_indices)
            cross_choices = [idx for idx in region_indices if idx != stack_region]
            cross_region = rng.choice(cross_choices or region_indices)

            infos = []
            for idx, bbox in enumerate(regions):
                infos.append(
                    _fill_region(
                        grid,
                        bbox,
                        rng,
                        need_stack=(idx == stack_region),
                        need_cross=(idx == cross_region),
                    )
                )

            parities = {info["parity"] for info in infos}
            if len(parities) < 2:
                continue

            out = main(grid)
            if np.array_equal(out, grid):
                continue
            if _matches_wrong_semantics(grid, out, _wrong_global_parity):
                continue
            if _matches_wrong_semantics(grid, out, _blue_blocking_variant):
                continue

            for bbox in regions:
                region = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
                assert np.count_nonzero(region == BLUE) == 1
                y_count = int(np.count_nonzero(region == YELLOW))
                assert 2 <= y_count <= 4

            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
