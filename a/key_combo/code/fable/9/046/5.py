from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

_TEMPLATE_CATALOG = (
    {"name": "gap_domino", "centers": ((0, 0), (0, 2)), "redundant": True},
    {"name": "adj_domino", "centers": ((0, 0), (0, 1)), "redundant": False},
    {"name": "l_gap", "centers": ((0, 0), (0, 2), (2, 0)), "redundant": True},
    {"name": "l_adj", "centers": ((0, 0), (0, 1), (1, 0)), "redundant": False},
    {"name": "diag3", "centers": ((0, 0), (1, 1), (2, 2)), "redundant": False},
    {"name": "zig3", "centers": ((0, 0), (0, 2), (1, 1)), "redundant": True},
    {"name": "square4", "centers": ((0, 0), (0, 2), (2, 0), (2, 2)), "redundant": True},
    {"name": "asym4", "centers": ((0, 0), (0, 1), (1, 0), (1, 2)), "redundant": True},
)

_COMPONENT_COUNT_CHOICES = (1, 2, 3)
_COMPONENT_COUNT_WEIGHTS = (2, 5, 3)
_GRID_SIZE_RANGE = (12, 24)
_TOTAL_STAMPS_RANGE = (4, 8)
_PLACEMENT_GAP = 1
_MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _transform_centers(centers: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(centers, dtype=int)
    if random.choice((False, True)):
        arr[:, 1] *= -1
    rot = random.randrange(4)
    for _ in range(rot):
        arr = np.stack((-arr[:, 1], arr[:, 0]), axis=1)
    arr -= arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _render_component(
    centers: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    norm = np.array(centers, dtype=int)
    norm -= norm.min(axis=0)
    norm += 1
    h = int(norm[:, 0].max()) + 2
    w = int(norm[:, 1].max()) + 2
    mask = np.zeros((h, w), dtype=bool)
    center_list = []
    for r, c in norm:
        rr = int(r)
        cc = int(c)
        mask[rr - 1 : rr + 2, cc - 1 : cc + 2] = True
        center_list.append((rr, cc))
    return mask, tuple(sorted(center_list))


def _candidate_stamps(
    mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], int]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return [], [], 0
    index = {cell: i for i, cell in enumerate(cells)}
    centers: list[tuple[int, int]] = []
    covers: list[int] = []
    h, w = mask.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not np.all(mask[r - 1 : r + 2, c - 1 : c + 2]):
                continue
            cover = 0
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    cover |= 1 << index[(rr, cc)]
            centers.append((r, c))
            covers.append(cover)
    return centers, covers, len(cells)


def _solve_component(mask: np.ndarray) -> dict:
    centers, covers, cell_count = _candidate_stamps(mask)
    assert cell_count > 0, "empty component is invalid"
    assert centers, "component cannot be decomposed into 3x3 solid stamps"

    full = (1 << cell_count) - 1
    cell_to_candidates: list[tuple[int, ...]] = []
    for cell in range(cell_count):
        options = tuple(idx for idx, cover in enumerate(covers) if (cover >> cell) & 1)
        assert options, "some blue cells are uncovered by every candidate stamp"
        cell_to_candidates.append(options)

    option_count = [len(options) for options in cell_to_candidates]
    ordered_options = [
        tuple(sorted(options, key=lambda idx: covers[idx].bit_count(), reverse=True))
        for options in cell_to_candidates
    ]
    memo: dict[int, tuple[int, int, tuple[int, ...]]] = {}

    def dp(remaining: int) -> tuple[int, int, tuple[int, ...]]:
        forced: list[int] = []
        while remaining:
            chosen = None
            probe = remaining
            while probe:
                bit = probe & -probe
                cell = bit.bit_length() - 1
                if option_count[cell] == 1:
                    chosen = cell_to_candidates[cell][0]
                    break
                probe ^= bit
            if chosen is None:
                break
            next_remaining = remaining & ~covers[chosen]
            assert next_remaining != remaining, "forced step must remove uncovered cells"
            forced.append(chosen)
            remaining = next_remaining

        forced_tuple = tuple(forced)
        if remaining == 0:
            return len(forced_tuple), 1, forced_tuple

        cached = memo.get(remaining)
        if cached is not None:
            return len(forced_tuple) + cached[0], cached[1], forced_tuple + cached[2]

        pivot = None
        pivot_options: tuple[int, ...] | None = None
        probe = remaining
        while probe:
            bit = probe & -probe
            cell = bit.bit_length() - 1
            options = ordered_options[cell]
            if pivot_options is None or len(options) < len(pivot_options):
                pivot = cell
                pivot_options = options
                if len(pivot_options) == 2:
                    break
            probe ^= bit
        assert pivot is not None and pivot_options is not None

        best_count = 10 ** 9
        best_solutions = 0
        best_choice: tuple[int, ...] = ()
        for cand in pivot_options:
            next_remaining = remaining & ~covers[cand]
            if next_remaining == remaining:
                continue
            sub_count, sub_solutions, sub_choice = dp(next_remaining)
            total = 1 + sub_count
            if total < best_count:
                best_count = total
                best_solutions = sub_solutions
                best_choice = (cand,) + sub_choice
            elif total == best_count:
                best_solutions = min(2, best_solutions + sub_solutions)

        assert best_count < 10 ** 9, "no exact cover found"
        memo[remaining] = (best_count, best_solutions, best_choice)
        return len(forced_tuple) + best_count, best_solutions, forced_tuple + best_choice

    min_count, solution_count, choice = dp(full)
    chosen_centers = tuple(sorted(centers[idx] for idx in choice))
    assert len(chosen_centers) == min_count
    return {
        "candidate_count": len(centers),
        "min_count": min_count,
        "solution_count": solution_count,
        "centers": chosen_centers,
    }


def _solve_grid(mask: np.ndarray) -> dict:
    all_centers: list[tuple[int, int]] = []
    total_candidates = 0
    total_min = 0
    comps = _components4(mask)
    assert comps, "expected at least one blue component"
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        crop = mask[r0 : r1 + 1, c0 : c1 + 1]
        solved = _solve_component(crop)
        assert solved["solution_count"] == 1, "minimum exact decomposition must be unique"
        total_candidates += int(solved["candidate_count"])
        total_min += int(solved["min_count"])
        for rr, cc in solved["centers"]:
            all_centers.append((r0 + int(rr), c0 + int(cc)))
    return {
        "component_count": len(comps),
        "candidate_count": total_candidates,
        "min_count": total_min,
        "centers": tuple(sorted(all_centers)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose the blue union into the unique minimum set of 3x3 solid stamps and mark each stamp center red.

    Input structure:
    - A black background contains one or more separated blue blob regions.
    - Every blue blob is exactly the union of one or more axis-aligned 3x3 solid square stamps, with overlaps allowed.
    - Stamp boundaries are erased by overlap, so the visible input only shows the merged blue residue.

    Rule:
    - Find all 3x3 stamp placements whose union matches the blue cells exactly.
    - Among those exact reconstructions, keep the one using the fewest stamps; the puzzle guarantees this minimum solution is unique.
    - Recolor the center cell of each chosen stamp red and leave every other blue cell unchanged.

    Key insight:
    - Not every fully-blue 3x3 window is a true generator: some are redundant interior placements, so the solution must be the unique minimum exact union reconstruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]
    assert fg != RED, "foreground color must differ from the red marker color"

    blue_mask = g == fg
    solved = _solve_grid(blue_mask)

    out = g.copy()
    for r, c in solved["centers"]:
        out[r, c] = RED
    return out


def _sample_component_specs() -> list[dict] | None:
    count = random.choices(_COMPONENT_COUNT_CHOICES, weights=_COMPONENT_COUNT_WEIGHTS, k=1)[0]
    specs = []
    for _ in range(count):
        template = random.choice(_TEMPLATE_CATALOG)
        transformed = _transform_centers(template["centers"])
        mask, centers = _render_component(transformed)
        specs.append(
            {
                "name": template["name"],
                "redundant": template["redundant"],
                "mask": mask,
                "centers": centers,
                "stamp_count": len(centers),
            }
        )

    total_stamps = sum(spec["stamp_count"] for spec in specs)
    if not (_TOTAL_STAMPS_RANGE[0] <= total_stamps <= _TOTAL_STAMPS_RANGE[1]):
        return None
    if total_stamps <= count:
        return None
    if not any(spec["redundant"] for spec in specs):
        return None
    return specs


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_components(specs: list[dict], h: int, w: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]] | None:
    grid = np.full((h, w), BLACK, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    all_centers: list[tuple[int, int]] = []

    ordered_specs = sorted(specs, key=lambda spec: int(spec["mask"].sum()), reverse=True)
    for spec in ordered_specs:
        mask = spec["mask"]
        mh, mw = mask.shape
        options = []
        for r0 in range(h - mh + 1):
            for c0 in range(w - mw + 1):
                box = (r0, c0, r0 + mh - 1, c0 + mw - 1)
                if all(_boxes_separated(box, other, _PLACEMENT_GAP) for other in placed_boxes):
                    options.append((r0, c0))
        if not options:
            return None
        r0, c0 = random.choice(options)
        rr, cc = np.where(mask)
        grid[r0 + rr, c0 + cc] = BLUE
        placed_boxes.append((r0, c0, r0 + mh - 1, c0 + mw - 1))
        for cr, cc0 in spec["centers"]:
            all_centers.append((r0 + int(cr), c0 + int(cc0)))

    return grid, tuple(sorted(all_centers))


def _expected_output(inp: np.ndarray, centers: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = inp.copy()
    for r, c in centers:
        out[r, c] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_component_specs()
    if specs is None:
        return None

    h = random.randint(*_GRID_SIZE_RANGE)
    w = random.randint(*_GRID_SIZE_RANGE)
    placed = _place_components(specs, h, w)
    if placed is None:
        return None
    inp, expected_centers = placed

    solved = _solve_grid(inp == BLUE)
    if solved["centers"] != expected_centers:
        return None
    if solved["min_count"] != len(expected_centers):
        return None
    if solved["candidate_count"] <= solved["min_count"]:
        return None
    if solved["min_count"] <= solved["component_count"]:
        return None

    expected = _expected_output(inp, expected_centers)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-decomposition input")
