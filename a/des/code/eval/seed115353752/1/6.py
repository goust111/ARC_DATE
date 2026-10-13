import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))
GENERATION_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_anchors(
    grid: np.ndarray, bg: int, orientation: str
) -> tuple[Optional[Dict[int, int]], np.ndarray, int]:
    h, w = grid.shape
    anchor_mask = np.zeros((h, w), dtype=bool)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if orientation == "vertical":
        if h < 2:
            return None, anchor_mask, 0
        for r in (0, h - 1):
            inward = 1 if r == 0 else h - 2
            c = 0
            while c < w - 1:
                if int(grid[r, c]) != bg and int(grid[r, c + 1]) != bg:
                    isolated = (c == 0 or int(grid[r, c - 1]) == bg) and (
                        c + 2 == w or int(grid[r, c + 2]) == bg
                    )
                    clear_inside = int(grid[inward, c]) == bg and int(grid[inward, c + 1]) == bg
                    if isolated and clear_inside:
                        pairs.append(((r, c), (r, c + 1)))
                        anchor_mask[r, c] = True
                        anchor_mask[r, c + 1] = True
                        c += 2
                        continue
                c += 1
    else:
        if w < 2:
            return None, anchor_mask, 0
        for c in (0, w - 1):
            inward = 1 if c == 0 else w - 2
            r = 0
            while r < h - 1:
                if int(grid[r, c]) != bg and int(grid[r + 1, c]) != bg:
                    isolated = (r == 0 or int(grid[r - 1, c]) == bg) and (
                        r + 2 == h or int(grid[r + 2, c]) == bg
                    )
                    clear_inside = int(grid[r, inward]) == bg and int(grid[r + 1, inward]) == bg
                    if isolated and clear_inside:
                        pairs.append(((r, c), (r + 1, c)))
                        anchor_mask[r, c] = True
                        anchor_mask[r + 1, c] = True
                        r += 2
                        continue
                r += 1

    if not pairs:
        return None, anchor_mask, 0

    mapping: dict[int, int] = {}
    inverse_seen: dict[int, int] = {}
    for p0, p1 in pairs:
        src = int(grid[p0])
        dst = int(grid[p1])
        if src in mapping and mapping[src] != dst:
            return None, anchor_mask, 0
        if dst in inverse_seen and inverse_seen[dst] != src:
            return None, anchor_mask, 0
        mapping[src] = dst
        inverse_seen[dst] = src
    return mapping, anchor_mask, len(pairs)


def _mirror_point(
    r: int, c: int, orientation: str, axis: int
) -> tuple[int, int]:
    if orientation == "vertical":
        return r, 2 * axis + 1 - c
    return 2 * axis + 1 - r, c


def _evaluate_candidate(
    grid: np.ndarray,
    bg: int,
    orientation: str,
    axis: int,
    mapping: Dict[int, int],
    inverse: Dict[int, int],
    anchor_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    exact = 0
    holes_low = 0
    holes_high = 0
    conflicts = 0

    pattern_cells = np.argwhere((grid != bg) & (~anchor_mask))
    for r0, c0 in pattern_cells:
        r = int(r0)
        c = int(c0)
        rr, cc = _mirror_point(r, c, orientation, axis)
        if not (0 <= rr < h and 0 <= cc < w):
            conflicts += 1
            continue
        if anchor_mask[rr, cc]:
            conflicts += 1
            continue

        color = int(grid[r, c])
        low_side = c <= axis if orientation == "vertical" else r <= axis
        if low_side:
            expected = mapping.get(color)
            if expected is None:
                conflicts += 1
                continue
            actual = int(grid[rr, cc])
            if actual == expected:
                exact += 1
            elif actual == bg:
                holes_high += 1
            else:
                conflicts += 1
        else:
            expected = inverse.get(color)
            if expected is None:
                conflicts += 1
                continue
            actual = int(grid[rr, cc])
            if actual == expected:
                exact += 1
            elif actual == bg:
                holes_low += 1
            else:
                conflicts += 1

    return exact, holes_low, holes_high, conflicts


def _candidate_rankings(
    grid: np.ndarray,
) -> list[tuple[tuple[int, int, int, int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    ranked: list[tuple[tuple[int, int, int, int, int], dict]] = []

    for orientation in ("vertical", "horizontal"):
        mapping, anchor_mask, anchor_count = _extract_anchors(g, bg, orientation)
        if not mapping or anchor_count == 0:
            continue
        inverse = {dst: src for src, dst in mapping.items()}
        if len(inverse) != len(mapping):
            continue

        axis_limit = g.shape[1] - 1 if orientation == "vertical" else g.shape[0] - 1
        for axis in range(axis_limit):
            exact, holes_low, holes_high, conflicts = _evaluate_candidate(
                g, bg, orientation, axis, mapping, inverse, anchor_mask
            )
            for missing_side in ("low", "high"):
                holes_here = holes_low if missing_side == "low" else holes_high
                opposite_holes = holes_high if missing_side == "low" else holes_low
                bad = conflicts + opposite_holes
                good = exact + holes_here
                score = (-bad, good, exact, holes_here, anchor_count)
                ranked.append(
                    (
                        score,
                        {
                            "bg": bg,
                            "orientation": orientation,
                            "axis": axis,
                            "missing_side": missing_side,
                            "mapping": mapping,
                            "inverse": inverse,
                            "anchor_mask": anchor_mask,
                        },
                    )
                )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _best_spec(grid: np.ndarray) -> dict:
    rankings = _candidate_rankings(grid)
    assert rankings, "expected edge anchors and a recoverable symmetry axis"
    best_score, best = rankings[0]
    if len(rankings) > 1:
        assert best_score > rankings[1][0], "expected a unique best symmetry candidate"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a recolored mirror pattern by inferring the hidden axis and anchor mapping."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    spec = _best_spec(g)
    bg = int(spec["bg"])
    orientation = spec["orientation"]
    axis = int(spec["axis"])
    missing_side = spec["missing_side"]
    mapping = spec["mapping"]
    inverse = spec["inverse"]
    anchor_mask = spec["anchor_mask"]

    out = g.copy()
    pattern_cells = np.argwhere((g != bg) & (~anchor_mask))
    for r0, c0 in pattern_cells:
        r = int(r0)
        c = int(c0)
        rr, cc = _mirror_point(r, c, orientation, axis)
        if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
            continue
        if anchor_mask[rr, cc] or out[rr, cc] != bg:
            continue

        color = int(g[r, c])
        low_side = c <= axis if orientation == "vertical" else r <= axis
        if missing_side == "high" and low_side:
            out[rr, cc] = mapping[color]
        elif missing_side == "low" and not low_side:
            out[rr, cc] = inverse[color]

    return out


def _grow_blob(
    height: int,
    width: int,
    target_size: int,
    mandatory: Iterable[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]]:
    cells = {tuple(cell) for cell in mandatory}
    while len(cells) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return cells


def _color_blob(
    cells: set[tuple[int, int]], colors: list[int], rng: random.Random
) -> dict[tuple[int, int], int]:
    pts = list(cells)
    seed_pts = rng.sample(pts, len(colors))
    colored: dict[tuple[int, int], int] = {}
    for pt in pts:
        best_i = min(
            range(len(colors)),
            key=lambda i: (
                abs(pt[0] - seed_pts[i][0]) + abs(pt[1] - seed_pts[i][1]),
                i,
            ),
        )
        colored[pt] = colors[best_i]
    return colored


def _make_mapping(colors: list[int], rng: random.Random) -> dict[int, int]:
    if len(colors) == 2:
        return {colors[0]: colors[1], colors[1]: colors[0]}

    perm = colors[:]
    if rng.random() < 0.5:
        for _ in range(60):
            rng.shuffle(perm)
            if all(src != dst for src, dst in zip(colors, perm)):
                return dict(zip(colors, perm))

    for _ in range(60):
        rng.shuffle(perm)
        if any(src != dst for src, dst in zip(colors, perm)):
            return dict(zip(colors, perm))
    return dict(zip(colors, perm))


def _pick_edge_starts(length: int, count: int, rng: random.Random) -> list[int]:
    starts: list[int] = []
    while len(starts) < count:
        options = [
            s
            for s in range(1, length - 2)
            if all(abs(s - other) >= 3 for other in starts)
        ]
        if not options:
            break
        starts.append(rng.choice(options))
    return starts


def _grow_hole_subset(
    pool: set[tuple[int, int]], target_size: int, rng: random.Random
) -> set[tuple[int, int]]:
    holes: set[tuple[int, int]] = set()
    seed = rng.choice(list(pool))
    holes.add(seed)
    while len(holes) < target_size:
        if rng.random() < 0.85:
            base = rng.choice(list(holes))
            choices = [
                (base[0] + dr, base[1] + dc)
                for dr, dc in DIRS4
                if (base[0] + dr, base[1] + dc) in pool
                and (base[0] + dr, base[1] + dc) not in holes
            ]
            if choices:
                holes.add(rng.choice(choices))
                continue
        holes.add(rng.choice(list(pool - holes)))
    return holes


def _random_transform(
    inp: np.ndarray, solved: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        inp = inp.T.copy()
        solved = solved.T.copy()
    if rng.random() < 0.5:
        inp = np.flipud(inp).copy()
        solved = np.flipud(solved).copy()
    if rng.random() < 0.5:
        inp = np.fliplr(inp).copy()
        solved = np.fliplr(solved).copy()
    return inp, solved


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    bg = rng.choice(PALETTE)
    object_palette = [c for c in PALETTE if c != bg]

    h = rng.randint(12, 18)
    w = rng.randint(13, 19)
    axis_options = []
    for axis in range(4, w - 5):
        max_span = min(7, axis - 1, w - axis - 3)
        if max_span >= 4:
            axis_options.append((axis, max_span))
    if not axis_options:
        return None
    axis, max_span = rng.choice(axis_options)
    span = rng.randint(4, max_span)

    top = rng.randint(2, h - 7)
    bottom = rng.randint(top + 4, h - 3)
    box_h = bottom - top + 1
    box_w = span
    box_area = box_h * box_w

    color_cap = min(4, max(2, box_area // 6))
    num_colors = rng.randint(2, color_cap)
    colors = rng.sample(object_palette, num_colors)

    seed_row = rng.randrange(box_h)
    mandatory = {(seed_row, box_w - 1)}
    if box_h >= 6 and rng.random() < 0.7:
        other_row = min(box_h - 1, max(0, seed_row + rng.choice((-1, 1))))
        mandatory.add((other_row, box_w - 1))

    min_cells = max(num_colors * 4, box_area // 3)
    max_cells = min(box_area - 1, (box_area * 3) // 4)
    if min_cells > max_cells:
        return None
    target_size = rng.randint(min_cells, max_cells)
    blob = _grow_blob(box_h, box_w, target_size, mandatory, rng)
    rows = [r for r, _ in blob]
    cols = [c for _, c in blob]
    if len(set(rows)) < 3 or len(set(cols)) < 3:
        return None

    left_colors = _color_blob(blob, colors, rng)
    used_colors = sorted({left_colors[cell] for cell in blob})
    if len(used_colors) != num_colors:
        return None
    mapping = _make_mapping(used_colors, rng)

    solved = np.full((h, w), bg, dtype=int)
    left_cells: set[tuple[int, int]] = set()
    right_cells: set[tuple[int, int]] = set()
    for (lr, lc), color in left_colors.items():
        gr = top + lr
        gc = axis - span + 1 + lc
        mr, mc = _mirror_point(gr, gc, "vertical", axis)
        left_cells.add((gr, gc))
        right_cells.add((mr, mc))
        solved[gr, gc] = color
        solved[mr, mc] = mapping[color]

    top_slots = _pick_edge_starts(w, num_colors + rng.randint(0, 1), rng)
    bottom_slots = _pick_edge_starts(w, num_colors + rng.randint(0, 1), rng)
    if len(top_slots) + len(bottom_slots) < num_colors:
        return None

    anchor_specs = [(color, mapping[color]) for color in used_colors]
    rng.shuffle(anchor_specs)
    top_used = 0
    bottom_used = 0
    for src, dst in anchor_specs:
        if top_used < len(top_slots) and bottom_used < len(bottom_slots):
            use_top = rng.random() < 0.5
        else:
            use_top = top_used < len(top_slots)
        if use_top:
            start = top_slots[top_used]
            top_used += 1
            solved[0, start] = src
            solved[0, start + 1] = dst
        else:
            start = bottom_slots[bottom_used]
            bottom_used += 1
            solved[h - 1, start] = src
            solved[h - 1, start + 1] = dst

    extras = rng.randint(0, max(1, num_colors - 1))
    for _ in range(extras):
        src = rng.choice(used_colors)
        dst = mapping[src]
        choose_top = rng.random() < 0.5
        if choose_top and top_used < len(top_slots):
            start = top_slots[top_used]
            top_used += 1
            solved[0, start] = src
            solved[0, start + 1] = dst
        elif (not choose_top) and bottom_used < len(bottom_slots):
            start = bottom_slots[bottom_used]
            bottom_used += 1
            solved[h - 1, start] = src
            solved[h - 1, start + 1] = dst

    missing_side = rng.choice(("low", "high"))
    hole_pool = left_cells if missing_side == "low" else right_cells
    hole_count_lo = max(2, len(hole_pool) // 8)
    hole_count_hi = min(len(hole_pool) - 1, max(hole_count_lo, len(hole_pool) // 3))
    if hole_count_lo > hole_count_hi:
        return None
    holes = _grow_hole_subset(hole_pool, rng.randint(hole_count_lo, hole_count_hi), rng)

    inp = solved.copy()
    for r, c in holes:
        inp[r, c] = bg

    inp, solved = _random_transform(inp, solved, rng)
    if np.count_nonzero(inp == _background_color(inp)) <= inp.size // 2:
        return None

    rankings = _candidate_rankings(inp)
    if len(rankings) < 2 or rankings[0][0] == rankings[1][0]:
        return None
    if not np.array_equal(main(inp), solved):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-symmetry input")
