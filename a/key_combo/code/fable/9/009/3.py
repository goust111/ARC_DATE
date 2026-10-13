from __future__ import annotations

from collections import Counter
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _shift_coords(coords: Iterable[Coord], dr: int, dc: int) -> set[Coord]:
    return {(int(r) + int(dr), int(c) + int(dc)) for r, c in coords}


def _normalize(coords: Iterable[Coord]) -> set[Coord]:
    pts = {(int(r), int(c)) for r, c in coords}
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return {(r - min_r, c - min_c) for r, c in pts}


def _bbox(coords: Iterable[Coord]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: Iterable[Coord]) -> np.ndarray:
    pts = _normalize(coords)
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return mask


def _is_connected(coords: Iterable[Coord]) -> bool:
    pts = {(int(r), int(c)) for r, c in coords}
    if not pts:
        return False
    todo = [next(iter(pts))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == len(pts)


def _is_asymmetric(coords: Iterable[Coord]) -> bool:
    mask = _mask_from_coords(coords)
    return not (
        np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.fliplr(mask))
        or (mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T))
        or (mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.rot90(mask)))
    )


def _best_shift_info(a: set[Coord], b: set[Coord]) -> tuple[list[Coord], int, Counter]:
    counts: Counter = Counter()
    for ar, ac in a:
        for br, bc in b:
            counts[(ar - br, ac - bc)] += 1
    best = max(counts.values())
    winners = [delta for delta, count in counts.items() if count == best]
    return winners, best, counts


def _best_shift(a: set[Coord], b: set[Coord]) -> tuple[Coord, int, Counter]:
    winners, best, counts = _best_shift_info(a, b)
    assert len(winners) == 1, "pairwise translation alignment must be unique"
    return winners[0], best, counts


def _infer_alignment(color_coords: dict[int, set[Coord]]) -> tuple[dict[int, Coord], set[Coord]]:
    colors = list(color_coords)
    assert len(colors) >= 2, "expected multiple colored fragments"
    ref = colors[0]

    shifts_to_ref: dict[int, Coord] = {ref: (0, 0)}
    pair_shift: dict[tuple[int, int], Coord] = {}
    pair_overlap: dict[tuple[int, int], int] = {}
    for i, color_i in enumerate(colors):
        for color_j in colors[i + 1 :]:
            delta, overlap, _ = _best_shift(color_coords[color_i], color_coords[color_j])
            pair_shift[(color_i, color_j)] = delta
            pair_shift[(color_j, color_i)] = (-delta[0], -delta[1])
            pair_overlap[(color_i, color_j)] = overlap
            pair_overlap[(color_j, color_i)] = overlap
            assert overlap >= 2, "copies must share visible pixels after correct alignment"

    for color in colors[1:]:
        shifts_to_ref[color] = pair_shift[(ref, color)]

    for color_i in colors:
        for color_j in colors:
            expected = (
                shifts_to_ref[color_j][0] - shifts_to_ref[color_i][0],
                shifts_to_ref[color_j][1] - shifts_to_ref[color_i][1],
            )
            if color_i == color_j:
                assert expected == (0, 0)
            else:
                assert pair_shift[(color_i, color_j)] == expected, "pairwise shifts must be globally consistent"

    aligned = {
        color: _shift_coords(coords, *shifts_to_ref[color])
        for color, coords in color_coords.items()
    }
    template = set().union(*aligned.values())
    assert template, "template must be non-empty"

    for i, color_i in enumerate(colors):
        for color_j in colors[i + 1 :]:
            overlap = len(aligned[color_i] & aligned[color_j])
            assert overlap == pair_overlap[(color_i, color_j)], "alignment overlap should match the modal shift vote"

    return shifts_to_ref, template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color fragment by recovering the hidden template shared by all translated copies.

    Input structure:
    - The grid has one dominant background color and several foreground colors.
    - Each foreground color marks one incomplete translated copy of the same hidden shape.
    - No color shows the full shape, but different colors preserve different subsets of it.

    Rule:
    - Group pixels by color and infer the unique translation that aligns every pair of colors as compatible subsets of one template.
    - In that common aligned frame, take the union of all visible pixels to recover the hidden template.
    - Shift the recovered template back to each color's original position and fill its missing cells with that same color.

    Key insight:
    - There is no complete master copy; the template exists only in the mutual alignment of several incomplete copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) >= 3, "expected at least three incomplete colored copies"

    color_coords = {
        color: {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        for color in fg_colors
    }
    for coords in color_coords.values():
        assert coords, "each color must be present"

    shifts_to_ref, template = _infer_alignment(color_coords)
    template_area = len(template)
    assert all(len(coords) < template_area for coords in color_coords.values()), "no input color may already be complete"

    out = g.copy()
    painted_by: dict[Coord, int] = {}
    for color in fg_colors:
        dr, dc = shifts_to_ref[color]
        target = _shift_coords(template, -dr, -dc)
        assert color_coords[color] <= target, "observed pixels must belong to the completed translated template"
        for cell in target:
            other = painted_by.get(cell)
            assert other in (None, color), "completed copies must not overlap each other"
            painted_by[cell] = color
        for r, c in target:
            out[r, c] = color

    return out


GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
TEMPLATE_H_RANGE = (4, 6)
TEMPLATE_W_RANGE = (4, 6)
TEMPLATE_AREA_RANGE = (8, 12)
COLOR_COUNT_CHOICES = (3, 4, 5)
ANCHOR_SIZE_CHOICES = (3, 4)
MISSING_PER_COPY_RANGE = (1, 4)
PLACEMENT_GAP = 1
GENERATION_RETRIES = 400


def _random_template() -> set[Coord] | None:
    h = random.randint(*TEMPLATE_H_RANGE)
    w = random.randint(*TEMPLATE_W_RANGE)
    area = random.randint(TEMPLATE_AREA_RANGE[0], min(TEMPLATE_AREA_RANGE[1], h * w - 1))

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in list(cells):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    cells = _normalize(cells)
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    if not _is_connected(cells):
        return None
    if not _is_asymmetric(cells):
        return None
    return cells


def _pick_anchor(template: set[Coord], count: int) -> set[Coord] | None:
    cells = list(template)
    random.shuffle(cells)
    for _ in range(40):
        anchor = set(random.sample(cells, count))
        if _is_connected(anchor):
            return anchor
    return None


def _random_missing_sets(extra_cells: list[Coord], k: int) -> list[set[Coord]] | None:
    upper = min(MISSING_PER_COPY_RANGE[1], len(extra_cells))
    if upper < 1:
        return None

    for _ in range(120):
        missing_sets = []
        seen = set()
        for _color_idx in range(k):
            size = random.randint(MISSING_PER_COPY_RANGE[0], upper)
            choices = tuple(sorted(random.sample(extra_cells, size)))
            if choices in seen:
                break
            seen.add(choices)
            missing_sets.append(set(choices))
        else:
            if all(any(cell not in missing for missing in missing_sets) for cell in extra_cells):
                return missing_sets
    return None


def _subsets_from_template(template: set[Coord], color_count: int) -> list[set[Coord]] | None:
    max_anchor = min(max(ANCHOR_SIZE_CHOICES), len(template) - 2)
    anchor_choices = [size for size in ANCHOR_SIZE_CHOICES if size <= max_anchor]
    if not anchor_choices:
        return None

    for _ in range(150):
        anchor = _pick_anchor(template, random.choice(anchor_choices))
        if anchor is None:
            continue
        extra_cells = [cell for cell in template if cell not in anchor]
        if len(extra_cells) < 2:
            continue
        missing_sets = _random_missing_sets(extra_cells, color_count)
        if missing_sets is None:
            continue

        subsets = [set(template) - missing for missing in missing_sets]
        if any(subset == template for subset in subsets):
            continue
        if any(len(subset) <= 2 for subset in subsets):
            continue
        if set().union(*subsets) != template:
            continue

        sizes = [len(subset) for subset in subsets]
        largest = max(sizes)
        for subset in [subsets[i] for i, size in enumerate(sizes) if size == largest]:
            missing = template - subset
            if not missing <= set().union(*(s for s in subsets if s is not subset)):
                break
        else:
            ok = True
            for i in range(color_count):
                for j in range(i + 1, color_count):
                    winners, overlap, _ = _best_shift_info(subsets[i], subsets[j])
                    if winners != [(0, 0)] or overlap < len(anchor):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return subsets
    return None


def _fits(cells: set[Coord], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _touching(a: set[Coord], b: set[Coord]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in b:
                    return True
    return False


def _place_templates(template: set[Coord], count: int, H: int, W: int) -> list[Coord] | None:
    r0, c0, r1, c1 = _bbox(template)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h + 2 > H or w + 2 > W:
        return None

    placements: list[Coord] = []
    placed_sets: list[set[Coord]] = []
    for _ in range(300):
        placements.clear()
        placed_sets.clear()
        for _copy_idx in range(count):
            candidates = []
            for top in range(1, H - h):
                for left in range(1, W - w):
                    shifted = _shift_coords(template, top, left)
                    if not _fits(shifted, H, W):
                        continue
                    if any(_touching(shifted, other) for other in placed_sets):
                        continue
                    candidates.append((top, left, shifted))
            if not candidates:
                break
            top, left, shifted = random.choice(candidates)
            placements.append((top, left))
            placed_sets.append(shifted)
        if len(placements) == count:
            return placements
    return None


def _build_grid(
    H: int,
    W: int,
    bg: int,
    colors: list[int],
    placements: list[Coord],
    subsets: list[set[Coord]],
) -> np.ndarray:
    grid = np.full((H, W), bg, dtype=int)
    for color, (top, left), subset in zip(colors, placements, subsets):
        for r, c in subset:
            grid[top + r, left + c] = color
    return grid


def _expected_output(
    H: int,
    W: int,
    bg: int,
    colors: list[int],
    placements: list[Coord],
    template: set[Coord],
) -> np.ndarray:
    grid = np.full((H, W), bg, dtype=int)
    for color, (top, left) in zip(colors, placements):
        for r, c in template:
            grid[top + r, left + c] = color
    return grid


def _validate_generated_instance(
    inp: np.ndarray,
    template: set[Coord],
    subsets: list[set[Coord]],
    colors: list[int],
    placements: list[Coord],
    bg: int,
) -> bool:
    expected = _expected_output(inp.shape[0], inp.shape[1], bg, colors, placements, template)
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(colors_present(inp, exclude={bg})) != len(colors):
        return False
    if len(set(map(frozenset, subsets))) != len(subsets):
        return False
    if any(len(subset) >= len(template) for subset in subsets):
        return False
    if set().union(*subsets) != template:
        return False

    sizes = [len(subset) for subset in subsets]
    if max(sizes) == len(template):
        return False

    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            winners, overlap, counts = _best_shift_info(subsets[i], subsets[j])
            if winners != [(0, 0)]:
                return False
            if overlap < 3:
                return False
            best = max(counts.values())
            if sum(1 for v in counts.values() if v == best) != 1:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randrange(10)
    color_count = random.choice(COLOR_COUNT_CHOICES)
    colors = random.sample([c for c in range(10) if c != bg], color_count)

    template = _random_template()
    if template is None:
        return None

    subsets = _subsets_from_template(template, color_count)
    if subsets is None:
        return None

    placements = _place_templates(template, color_count, H, W)
    if placements is None:
        return None

    inp = _build_grid(H, W, bg, colors, placements, subsets)
    if np.count_nonzero(inp != bg) >= inp.size // 2:
        return None

    if not _validate_generated_instance(inp, template, subsets, colors, placements, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-template-complement input")
