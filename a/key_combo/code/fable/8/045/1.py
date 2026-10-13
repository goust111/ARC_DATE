from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
FILL_COLOR = 2
REMNANT_COLOR = 5

GEN_RETRIES = 500
GEN_HEIGHT_RANGE = (14, 26)
GEN_WIDTH_RANGE = (14, 26)
GEN_TRIPLE_HEIGHT_RANGE = (20, 28)
GEN_TRIPLE_WIDTH_RANGE = (20, 28)
GEN_EDGE_MARGIN = 1
GEN_NEST_GAP = 2
GEN_MIN_CORNER_SPAN = 3
GEN_MIN_RED_PIXELS = 6

ROLE_TL = "tl"
ROLE_TR = "tr"
ROLE_BL = "bl"
ROLE_BR = "br"

CARDINAL_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MISSING_TO_ROLE = {
    (1, 1): ROLE_TL,
    (1, 0): ROLE_TR,
    (0, 1): ROLE_BL,
    (0, 0): ROLE_BR,
}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in CARDINAL_STEPS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _rect_remnants(rect: tuple[int, int, int, int]) -> dict[str, set[tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    return {
        ROLE_TL: {(r0, c0), (r0, c0 + 1), (r0 + 1, c0)},
        ROLE_TR: {(r0, c1), (r0, c1 - 1), (r0 + 1, c1)},
        ROLE_BL: {(r1, c0), (r1, c0 + 1), (r1 - 1, c0)},
        ROLE_BR: {(r1, c1), (r1, c1 - 1), (r1 - 1, c1)},
    }


def _extract_remnants(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    assert len(colors) == 1, "expected a single remnant color on the background"
    remnant_color = int(colors[0])

    remnants: list[dict[str, object]] = []
    for comp in _components4(g == remnant_color):
        assert comp.shape[0] == 3, "every remnant must have exactly three pixels"
        r0, c0 = np.min(comp, axis=0)
        r1, c1 = np.max(comp, axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        assert r1 == r0 + 1 and c1 == c0 + 1, "every remnant must fit in a 2x2 corner window"
        cells = {tuple(map(int, cell)) for cell in comp}
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        missing = [pt for pt in corners if pt not in cells]
        assert len(missing) == 1
        mr, mc = missing[0]
        role = MISSING_TO_ROLE[(mr - r0, mc - c0)]
        if role == ROLE_TL:
            corner = (r0, c0)
        elif role == ROLE_TR:
            corner = (r0, c1)
        elif role == ROLE_BL:
            corner = (r1, c0)
        else:
            corner = (r1, c1)
        remnants.append({"role": role, "corner": corner})

    assert remnants and len(remnants) % 4 == 0, "the grid must contain complete remnant quartets"
    role_counts = {role: 0 for role in (ROLE_TL, ROLE_TR, ROLE_BL, ROLE_BR)}
    for rem in remnants:
        role_counts[str(rem["role"])] += 1
    assert len(set(role_counts.values())) == 1, "each role must appear equally often"

    remnants.sort(key=lambda item: (item["corner"], item["role"]))  # type: ignore[index]
    for idx, rem in enumerate(remnants):
        rem["id"] = idx
    return remnant_color, remnants


def _candidate_rectangles(remnants: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup: dict[tuple[str, tuple[int, int]], int] = {}
    tls: list[dict[str, object]] = []
    brs: list[dict[str, object]] = []
    for rem in remnants:
        key = (str(rem["role"]), tuple(rem["corner"]))  # type: ignore[arg-type]
        assert key not in lookup, "duplicate remnant role/corner"
        lookup[key] = int(rem["id"])
        if rem["role"] == ROLE_TL:
            tls.append(rem)
        elif rem["role"] == ROLE_BR:
            brs.append(rem)

    seen: set[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = set()
    candidates: list[dict[str, object]] = []
    for tl in sorted(tls, key=lambda item: tuple(item["corner"])):  # type: ignore[arg-type]
        r0, c0 = map(int, tl["corner"])  # type: ignore[arg-type]
        for br in sorted(brs, key=lambda item: tuple(item["corner"])):  # type: ignore[arg-type]
            r1, c1 = map(int, br["corner"])  # type: ignore[arg-type]
            if r1 - r0 < GEN_MIN_CORNER_SPAN or c1 - c0 < GEN_MIN_CORNER_SPAN:
                continue
            tr_id = lookup.get((ROLE_TR, (r0, c1)))
            bl_id = lookup.get((ROLE_BL, (r1, c0)))
            if tr_id is None or bl_id is None:
                continue
            rect = (r0, c0, r1, c1)
            ids = tuple(sorted((int(tl["id"]), int(tr_id), int(bl_id), int(br["id"]))))
            key = (rect, ids)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"rect": rect, "ids": ids})
    candidates.sort(key=lambda item: tuple(item["rect"]))  # type: ignore[arg-type]
    assert candidates, "no rectangle is compatible with the remnants"
    return candidates


def _solve_rectangles(remnants: list[dict[str, object]]) -> list[tuple[int, int, int, int]]:
    candidates = _candidate_rectangles(remnants)
    all_ids = {int(rem["id"]) for rem in remnants}
    rem_to_cands = {rid: [] for rid in all_ids}
    for idx, cand in enumerate(candidates):
        for rid in cand["ids"]:  # type: ignore[index]
            rem_to_cands[int(rid)].append(idx)

    solutions: list[list[int]] = []

    def backtrack(remaining: set[int], chosen: list[int]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return

        best_options: list[int] | None = None
        for rid in sorted(remaining):
            options = [
                idx
                for idx in rem_to_cands[rid]
                if set(int(x) for x in candidates[idx]["ids"]).issubset(remaining)
            ]
            if best_options is None or len(options) < len(best_options):
                best_options = options
        if not best_options:
            return

        for idx in best_options:
            chosen.append(idx)
            backtrack(remaining - set(int(x) for x in candidates[idx]["ids"]), chosen)
            chosen.pop()

    backtrack(all_ids, [])
    assert len(solutions) == 1, "remnants must determine a unique rectangle grouping"
    rects = [tuple(map(int, candidates[idx]["rect"])) for idx in solutions[0]]  # type: ignore[arg-type]
    rects.sort()
    return rects


def _coverage_masks(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    border = np.zeros(shape, dtype=bool)
    cover = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        cover[r0 + 1 : r1, c0 + 1 : c1] += 1
    return border, cover


def _render_output(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int]], border_color: int
) -> np.ndarray:
    border, cover = _coverage_masks(shape, rects)
    out = np.full(shape, BG_COLOR, dtype=int)
    out[cover % 2 == 1] = FILL_COLOR
    out[border] = int(border_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden rectangles from their corner remnants, then fill by odd coverage parity.

    Input structure:
    - A black background contains several disconnected three-pixel grey L remnants.
    - Each remnant marks one corner of a missing axis-aligned rectangle, and its open quadrant points toward that rectangle's interior.
    - Every rectangle contributes exactly four remnants, one per corner, while different rectangles may nest or overlap.

    Rule:
    - Group four remnants into a rectangle only when their opening directions and corner row/column alignments agree as top-left, top-right, bottom-left, and bottom-right corners of the same axis-aligned box.
    - Draw every reconstructed rectangle border in grey.
    - For each non-border cell, count how many reconstructed rectangles strictly contain it; color the cell red iff that count is odd, otherwise leave it black. Border cells stay grey.

    Key insight:
    - The remnants do not identify rectangles by proximity; the missing quadrant inside each 2x2 L is the reliable cue for each corner role, and the final fill is an XOR-like parity over all strict interiors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    remnant_color, remnants = _extract_remnants(g)
    rects = _solve_rectangles(remnants)
    return _render_output(g.shape, rects, remnant_color)


def _sample_spaced_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    usable_high = high - min_gap * (count - 1)
    if usable_high < low:
        return None
    base = sorted(random.sample(range(low, usable_high + 1), count))
    return [base[i] + i * min_gap for i in range(count)]


def _gen_nested_chain(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    def rec(
        rmin: int,
        rmax: int,
        cmin: int,
        cmax: int,
        remaining: int,
    ) -> list[tuple[int, int, int, int]] | None:
        min_span = GEN_MIN_CORNER_SPAN + 2 * GEN_NEST_GAP * (remaining - 1)
        if rmax - rmin < min_span or cmax - cmin < min_span:
            return None
        for _ in range(80):
            row_span = random.randint(min_span, min(rmax - rmin, min_span + 8))
            col_span = random.randint(min_span, min(cmax - cmin, min_span + 8))
            r0 = random.randint(rmin, rmax - row_span)
            r1 = r0 + row_span
            c0 = random.randint(cmin, cmax - col_span)
            c1 = c0 + col_span
            rect = (r0, c0, r1, c1)
            if remaining == 1:
                return [rect]
            inner = rec(r0 + GEN_NEST_GAP, r1 - GEN_NEST_GAP, c0 + GEN_NEST_GAP, c1 - GEN_NEST_GAP, remaining - 1)
            if inner is not None:
                return [rect] + inner
        return None

    return rec(GEN_EDGE_MARGIN, height - 1 - GEN_EDGE_MARGIN, GEN_EDGE_MARGIN, width - 1 - GEN_EDGE_MARGIN, count)


def _gen_partial_pair(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    rows = _sample_spaced_positions(GEN_EDGE_MARGIN, height - 1 - GEN_EDGE_MARGIN, 4, 2)
    cols = _sample_spaced_positions(GEN_EDGE_MARGIN, width - 1 - GEN_EDGE_MARGIN, 4, 2)
    if rows is None or cols is None:
        return None
    a, b, c, d = rows
    p, q, r, s = cols
    if random.random() < 0.5:
        rects = [(a, p, c, r), (b, q, d, s)]
    else:
        rects = [(a, q, c, s), (b, p, d, r)]
    rects.sort()
    return rects


def _touches_existing_8(cells: set[tuple[int, int]], prior: list[set[tuple[int, int]]]) -> bool:
    for other in prior:
        for r0, c0 in cells:
            for r1, c1 in other:
                if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                    return True
    return False


def _render_input(
    height: int, width: int, rects: list[tuple[int, int, int, int]]
) -> np.ndarray | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    remnants: list[set[tuple[int, int]]] = []
    for rect in rects:
        for cells in _rect_remnants(rect).values():
            if _touches_existing_8(cells, remnants):
                return None
            remnants.append(cells)
    for cells in remnants:
        for r, c in cells:
            grid[r, c] = REMNANT_COLOR
    if len(_components4(grid == REMNANT_COLOR)) != 4 * len(rects):
        return None
    return grid


def _canonical_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(tuple(map(int, rect)) for rect in rects)


def _instance_valid(height: int, width: int, rects: list[tuple[int, int, int, int]]) -> bool:
    rects = _canonical_rects(rects)
    if len(rects) < 2 or len(set(rects)) != len(rects):
        return False
    for r0, c0, r1, c1 in rects:
        if not (GEN_EDGE_MARGIN <= r0 < r1 <= height - 1 - GEN_EDGE_MARGIN):
            return False
        if not (GEN_EDGE_MARGIN <= c0 < c1 <= width - 1 - GEN_EDGE_MARGIN):
            return False
        if r1 - r0 < GEN_MIN_CORNER_SPAN or c1 - c0 < GEN_MIN_CORNER_SPAN:
            return False

    inp = _render_input(height, width, rects)
    if inp is None:
        return False

    try:
        remnant_color, remnants = _extract_remnants(inp)
        solved_rects = _solve_rectangles(remnants)
    except Exception:
        return False
    if remnant_color != REMNANT_COLOR or solved_rects != rects:
        return False

    border, cover = _coverage_masks((height, width), rects)
    interior = ~border
    if not np.any(cover[interior] == 1):
        return False
    if not np.any(cover[interior] >= 2):
        return False
    if not np.any(border & (cover > 0)):
        return False

    out = _render_output((height, width), rects, REMNANT_COLOR)
    red_count = int(np.count_nonzero(out == FILL_COLOR))
    grey_count = int(np.count_nonzero(out == REMNANT_COLOR))
    if red_count < GEN_MIN_RED_PIXELS or grey_count < 4 * len(rects):
        return False
    if red_count + grey_count >= int(0.65 * height * width):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, out))


def _gen_attempt() -> np.ndarray | None:
    family = random.choices(
        ["nested_pair", "partial_pair", "triple_nested"],
        weights=[4, 4, 3],
        k=1,
    )[0]
    if family == "triple_nested":
        height = random.randint(*GEN_TRIPLE_HEIGHT_RANGE)
        width = random.randint(*GEN_TRIPLE_WIDTH_RANGE)
    else:
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)

    if family == "nested_pair":
        rects = _gen_nested_chain(height, width, 2)
    elif family == "partial_pair":
        rects = _gen_partial_pair(height, width)
    else:
        rects = _gen_nested_chain(height, width, 3)

    if rects is None or not _instance_valid(height, width, rects):
        return None
    return _render_input(height, width, rects)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-remnant reconstruction input")
