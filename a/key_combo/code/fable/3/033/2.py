import random

import numpy as np


PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIZE_CHOICES = [(4, 4), (4, 5), (5, 4), (5, 5)]
MAX_GEN_ATTEMPTS = 500


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _mirror_point(kind: str, axis_sum: int, r: int, c: int) -> tuple[int, int] | None:
    if kind == "vertical":
        rr, cc = int(r), int(axis_sum - c)
    else:
        rr, cc = int(axis_sum - r), int(c)
    return (rr, cc) if rr >= 0 and cc >= 0 else None


def _evaluate_axis(grid: np.ndarray, bg: int, kind: str, axis_sum: int) -> dict | None:
    h, w = grid.shape
    repaired = np.full_like(grid, bg)
    breaches: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            q = _mirror_point(kind, axis_sum, r, c)
            if q is None:
                if int(grid[r, c]) != bg:
                    return None
                continue
            rr, cc = q
            if rr >= h or cc >= w:
                if int(grid[r, c]) != bg:
                    return None
                continue
            if (rr, cc) < (r, c):
                continue

            a = int(grid[r, c])
            b = int(grid[rr, cc])
            if (rr, cc) == (r, c):
                if a != bg:
                    repaired[r, c] = a
                continue
            if a == bg and b == bg:
                continue
            if a != bg and b != bg:
                if a != b:
                    return None
                repaired[r, c] = a
                repaired[rr, cc] = a
                continue
            if a != bg:
                breaches.append((r, c))
            else:
                breaches.append((rr, cc))

    repaired_area = int(np.count_nonzero(repaired != bg))
    if repaired_area == 0 or not breaches:
        return None
    return {
        "kind": kind,
        "axis_sum": int(axis_sum),
        "bg": int(bg),
        "repaired": repaired,
        "breaches": tuple(sorted(breaches)),
        "repaired_area": repaired_area,
    }


def _best_candidate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    candidates: list[dict] = []

    for axis_sum in range(2 * w - 1):
        cand = _evaluate_axis(g, bg, "vertical", axis_sum)
        if cand is not None:
            candidates.append(cand)
    for axis_sum in range(2 * h - 1):
        cand = _evaluate_axis(g, bg, "horizontal", axis_sum)
        if cand is not None:
            candidates.append(cand)

    assert candidates, "expected at least one symmetry-repair candidate"
    min_breaches = min(len(c["breaches"]) for c in candidates)
    candidates = [c for c in candidates if len(c["breaches"]) == min_breaches]
    max_area = max(c["repaired_area"] for c in candidates)
    candidates = [c for c in candidates if c["repaired_area"] == max_area]
    assert len(candidates) == 1, "expected a unique best symmetry axis"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the repaired symmetric pattern into the macro-cells named by the symmetry-breaking pixels.

    Input structure:
    - A small background grid contains one multicolor pattern that is almost mirror-symmetric about one vertical or horizontal axis.
    - A few colored breach pixels violate that symmetry: each breach pixel has a mirrored location that is background.
    - Removing those breach pixels leaves a fully symmetric repaired pattern.

    Rule:
    - Find the unique mirror axis that becomes exact after deleting the few asymmetric colored pixels.
    - Erase those breach pixels to recover the repaired symmetric pattern.
    - Build an output macro-grid with the same HxW layout as the input, where each macro-cell is itself an HxW grid.
    - For every breach pixel position in the input, stamp one copy of the repaired pattern into the corresponding macro-cell; all other macro-cells stay background.

    Key insight:
    - The control cells are defined by global symmetry failure rather than by a fixed marker color, and the stamp is the repaired pattern instead of the original input.
    """
    g = np.asarray(grid, dtype=int)
    cand = _best_candidate(g)
    repaired = cand["repaired"]
    breaches = cand["breaches"]
    bg = cand["bg"]
    h, w = g.shape

    out = np.full((h * h, w * w), bg, dtype=int)
    for r, c in breaches:
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = repaired
    return out


def _make_repaired_pattern(
    h: int, w: int, kind: str, rng: random.Random
) -> tuple[np.ndarray, int, list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    grid = np.zeros((h, w), dtype=int)

    if kind == "vertical":
        while True:
            box_h = rng.randint(2, h)
            box_w = rng.randint(2, w)
            if box_h * (box_w // 2) >= 3:
                break
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        axis_sum = 2 * c0 + box_w - 1
        off_pairs = []
        axis_cells = []
        for r in range(r0, r0 + box_h):
            for c in range(c0, c0 + box_w):
                mc = axis_sum - c
                if c > mc:
                    continue
                if c == mc:
                    axis_cells.append((r, c))
                else:
                    off_pairs.append(((r, c), (r, mc)))
    else:
        while True:
            box_h = rng.randint(2, h)
            box_w = rng.randint(2, w)
            if box_w * (box_h // 2) >= 3:
                break
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        axis_sum = 2 * r0 + box_h - 1
        off_pairs = []
        axis_cells = []
        for r in range(r0, r0 + box_h):
            mr = axis_sum - r
            if r > mr:
                continue
            for c in range(c0, c0 + box_w):
                if r == mr:
                    axis_cells.append((r, c))
                else:
                    off_pairs.append(((r, c), (mr, c)))

    max_fill = min(4, len(off_pairs) - 1)
    filled_pair_count = rng.randint(2, max_fill)
    filled_pair_ids = set(rng.sample(range(len(off_pairs)), filled_pair_count))
    palette = rng.sample(PALETTE, rng.randint(2, 4))
    painted_groups: list[list[tuple[int, int]]] = []

    for idx, pair in enumerate(off_pairs):
        if idx not in filled_pair_ids:
            continue
        color = rng.choice(palette)
        for r, c in pair:
            grid[r, c] = color
        painted_groups.append(list(pair))

    if axis_cells:
        axis_fill_count = rng.randint(0, min(2, len(axis_cells)))
        for cell in rng.sample(axis_cells, axis_fill_count):
            grid[cell] = rng.choice(palette)
            painted_groups.append([cell])

    used = {int(grid[r, c]) for group in painted_groups for r, c in group if int(grid[r, c]) != 0}
    if len(used) == 1 and len(painted_groups) >= 2:
        recolor = rng.choice([c for c in palette if c != next(iter(used))])
        for r, c in painted_groups[-1]:
            grid[r, c] = recolor

    empty_pairs = [off_pairs[idx] for idx in range(len(off_pairs)) if idx not in filled_pair_ids]
    return grid, axis_sum, empty_pairs, axis_cells


def _add_breaches(
    repaired: np.ndarray,
    empty_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    rng: random.Random,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    grid = repaired.copy()
    breach_count = rng.randint(1, min(3, len(empty_pairs)))
    chosen_pairs = rng.sample(empty_pairs, breach_count)
    breaches = []
    for pair in chosen_pairs:
        cell = pair[rng.randrange(2)]
        grid[cell] = rng.choice(PALETTE)
        breaches.append(cell)
    return grid, tuple(sorted(breaches))


def _background_is_unique_majority(grid: np.ndarray, bg: int) -> bool:
    vals, cnts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(vals, cnts)}
    bg_count = count_map.get(bg, 0)
    return all(bg_count > count for color, count in count_map.items() if color != bg)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = rng.choice(SIZE_CHOICES)
    kind = rng.choice(["vertical", "horizontal"])

    repaired, axis_sum, empty_pairs, _axis_cells = _make_repaired_pattern(h, w, kind, rng)
    if not empty_pairs:
        return None

    grid, breaches = _add_breaches(repaired, empty_pairs, rng)
    bg = 0
    if not _background_is_unique_majority(grid, bg):
        return None

    try:
        cand = _best_candidate(grid)
    except AssertionError:
        return None

    if cand["kind"] != kind or cand["axis_sum"] != axis_sum:
        return None
    if cand["breaches"] != breaches:
        return None
    if not np.array_equal(cand["repaired"], repaired):
        return None
    if max(main(grid).shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
