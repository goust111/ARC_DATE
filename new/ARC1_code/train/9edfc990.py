from collections import deque

import numpy as np

from arc_utils import Color, components_mask, get_bg, in_bounds, neighbors4
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Flood-fill black background regions connected to original blue pixels, turning them blue while leaving blockers unchanged.

    Input structure:
    - A rectangular grid has black background cells, one or more blue seed pixels, and other non-black colors acting as blockers.
    Rule:
    - Starting from every original blue pixel, traverse 4-neighbor black cells and recolor each reached black cell blue.
    - Leave every non-black pixel unchanged.
    Key insight:
    - Black cells remain black exactly when no orthogonal path of black cells connects them to an original blue seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == Color.BLACK

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.BLUE)]
    assert seeds

    out = g.copy()
    seen = np.zeros(g.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        seen[r, c] = True
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if not in_bounds(g.shape, rr, cc) or seen[rr, cc] or g[rr, cc] != Color.BLACK:
                continue
            seen[rr, cc] = True
            out[rr, cc] = Color.BLUE
            q.append((rr, cc))

    return out

# --------------- generator (o5x) ---------------

_GEN_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_BLACK, _GEN_BLUE, _GEN_BLOCKER_COLORS = 0, 1, tuple(range(2, 10))

# Canvas and cell-composition bounds.
_GEN_SIZE_MIN, _GEN_SIZE_MAX = 13, 16
_GEN_BLACK_FRAC_MIN, _GEN_BLACK_FRAC_MAX = 0.39, 0.56
_GEN_BLUE_FRAC_MIN, _GEN_BLUE_FRAC_MAX = 0.03, 0.07
_GEN_BLUE_COUNT_MIN, _GEN_BLUE_COUNT_MAX = 5, 17

# Blocker scatter, color counts, and connectivity targets.
_GEN_BLOCKER_ABS_MIN, _GEN_CLUSTER_DIVISOR = 40, 40
_GEN_BLOCKER_FRAC_MIN, _GEN_BLOCKER_FRAC_MAX = 0.41, 0.54
_GEN_BLOCKER_COMPONENT_MIN, _GEN_BLOCKER_COMPONENT_MAX = 18, 35
_GEN_BLOCKER_NEIGHBOR_AVG_MIN, _GEN_BLOCKER_NEIGHBOR_AVG_MAX = 1.4, 2.0
_GEN_COLOR_COUNT_MIN, _GEN_COLOR_COUNT_MAX = 5, 24

# Reachability and unreached-region targets.
_GEN_BLACK_COMPONENT_REJECT_MIN, _GEN_BLACK_COMPONENT_SELECT_MIN = 2, 5
_GEN_REACHABLE_COMPONENT_MIN, _GEN_REACHABLE_FIXUP_FACTOR, _GEN_REACHABLE_PERTURB_MAX = 1, 2, 6
_GEN_UNREACHED_CELL_MIN, _GEN_UNREACHED_CELL_MAX = 5, 45
_GEN_UNREACHED_FRAC_MIN, _GEN_UNREACHED_FRAC_MAX = 0.03, 0.18
_GEN_UNREACHED_COMPONENT_MIN, _GEN_UNREACHED_COMPONENT_MAX = 4, 20

# Seed placement and retry limits.
_GEN_SEED_PAIR_PROB, _GEN_SEED_BLACK_NEIGHBOR_MIN, _GEN_MAX_ATTEMPTS = 0.18, 0.5, 300

def _gen_neighbor_count(mask, r, c):
    h, w = mask.shape
    total = 0
    for dr, dc in _GEN_DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            total += 1
    return total


def _gen_blocker_neighbor_average(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0.0
    return sum(_gen_neighbor_count(mask, int(r), int(c)) for r, c in coords) / len(coords)


def _gen_unreached_bounds(n_cells):
    lo = max(_GEN_UNREACHED_CELL_MIN, int(np.floor(_GEN_UNREACHED_FRAC_MIN * n_cells)))
    hi = min(_GEN_UNREACHED_CELL_MAX, int(np.floor(_GEN_UNREACHED_FRAC_MAX * n_cells)))
    return lo, hi


def _gen_scattered_blockers(size, n_blockers):
    mask = np.zeros((size, size), dtype=bool)
    cells = [(r, c) for r in range(size) for c in range(size)]
    for r, c in random.sample(cells, n_blockers):
        mask[r, c] = True

    # A few local moves add light random-walk clustering without making solid walls.
    for _ in range(random.randint(0, max(1, n_blockers // _GEN_CLUSTER_DIVISOR))):
        blockers = [tuple(rc) for rc in np.argwhere(mask)]
        isolated = [rc for rc in blockers if _gen_neighbor_count(mask, rc[0], rc[1]) == 0]
        if not isolated:
            break
        r, c = random.choice(blockers)
        nbrs = []
        for dr, dc in _GEN_DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size and not mask[rr, cc]:
                nbrs.append((rr, cc))
        if not nbrs:
            continue
        old = random.choice(isolated)
        new = random.choice(nbrs)
        mask[old] = False
        mask[new] = True
    return mask


def _gen_color_counts(total):
    counts = []
    remaining = total
    for i in range(len(_GEN_BLOCKER_COLORS)):
        left = len(_GEN_BLOCKER_COLORS) - 1 - i
        lo = max(_GEN_COLOR_COUNT_MIN, remaining - _GEN_COLOR_COUNT_MAX * left)
        hi = min(_GEN_COLOR_COUNT_MAX, remaining - _GEN_COLOR_COUNT_MIN * left)
        counts.append(random.randint(lo, hi))
        remaining -= counts[-1]
    random.shuffle(counts)
    return counts


def _gen_paint_blockers(mask):
    grid = np.zeros(mask.shape, dtype=int)
    coords = [tuple(rc) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    counts = _gen_color_counts(len(coords))
    k = 0
    for color, count in zip(_GEN_BLOCKER_COLORS, counts):
        for r, c in coords[k:k + count]:
            grid[r, c] = color
        k += count
    return grid


def _gen_reachable_swap_options(sizes, reachable, min_total, max_total, current_total, require_growth):
    options = []
    for r_idx in reachable:
        for u_idx in range(len(sizes)):
            if u_idx in reachable:
                continue
            delta = sizes[r_idx] - sizes[u_idx]
            new_total = current_total + delta
            if require_growth:
                if delta > 0 and new_total <= max_total:
                    options.append((r_idx, u_idx, delta))
            elif min_total <= new_total <= max_total:
                options.append((r_idx, u_idx, new_total))
    return options


def _gen_choose_reachable_components(components, seed_cap, min_unreached, max_unreached):
    n_comps = len(components)
    if n_comps < _GEN_BLACK_COMPONENT_SELECT_MIN:
        return None

    sizes = [len(comp) for comp in components]
    total_black = sum(sizes)
    order = sorted(range(n_comps), key=lambda i: sizes[i], reverse=True)
    min_k = max(_GEN_REACHABLE_COMPONENT_MIN, n_comps - _GEN_UNREACHED_COMPONENT_MAX)
    max_k = min(seed_cap, n_comps - _GEN_UNREACHED_COMPONENT_MIN)
    if min_k > max_k:
        return None

    feasible = []
    for k in range(min_k, max_k + 1):
        smallest_unreached = total_black - sum(sizes[i] for i in order[:k])
        largest_unreached = total_black - sum(sizes[i] for i in order[-k:])
        if smallest_unreached <= max_unreached and largest_unreached >= min_unreached:
            feasible.append(k)
    if not feasible:
        return None

    k = random.choice(feasible)
    reachable = set(order[:k])
    unreached_total = total_black - sum(sizes[i] for i in reachable)

    for _ in range(n_comps * _GEN_REACHABLE_FIXUP_FACTOR):
        if unreached_total >= min_unreached:
            break
        options = _gen_reachable_swap_options(sizes, reachable, min_unreached, max_unreached, unreached_total, True)
        if not options:
            break
        r_idx, u_idx, delta = random.choice(options)
        reachable.remove(r_idx)
        reachable.add(u_idx)
        unreached_total += delta

    if not (min_unreached <= unreached_total <= max_unreached):
        return None

    for _ in range(random.randint(0, min(_GEN_REACHABLE_PERTURB_MAX, n_comps))):
        options = _gen_reachable_swap_options(sizes, reachable, min_unreached, max_unreached, unreached_total, False)
        if not options:
            break
        r_idx, u_idx, unreached_total = random.choice(options)
        reachable.remove(r_idx)
        reachable.add(u_idx)

    unreached_count = n_comps - len(reachable)
    if not (_GEN_UNREACHED_COMPONENT_MIN <= unreached_count <= _GEN_UNREACHED_COMPONENT_MAX):
        return None
    return reachable


def _gen_has_comp_neighbor(cell, comp_cells):
    r, c = cell
    return any((r + dr, c + dc) in comp_cells for dr, dc in _GEN_DIRS4)


def _gen_seed_cells(components, reachable, seed_count):
    seeds = set()
    all_reachable = []
    for idx in random.sample(list(reachable), len(reachable)):
        coords = [tuple(map(int, rc)) for rc in components[idx]]
        comp_cells = set(coords)
        neighbor_cells = [rc for rc in coords if _gen_has_comp_neighbor(rc, comp_cells)]
        choice_pool = neighbor_cells or coords
        cell = random.choice(choice_pool)
        seeds.add(cell)
        all_reachable.extend(coords)

    while len(seeds) < seed_count:
        available = [rc for rc in all_reachable if rc not in seeds]
        if not available:
            return None
        make_pair = random.random() < _GEN_SEED_PAIR_PROB
        adjacent = [rc for rc in available if _gen_has_comp_neighbor(rc, seeds)]
        isolated = [rc for rc in available if not _gen_has_comp_neighbor(rc, seeds)]
        if make_pair and adjacent:
            cell = random.choice(adjacent)
        elif isolated:
            cell = random.choice(isolated)
        else:
            cell = random.choice(available)
        seeds.add(cell)

    if len(seeds) != seed_count:
        return None
    return seeds


def _gen_seed_black_neighbor_fraction(grid):
    seeds = np.argwhere(grid == _GEN_BLUE)
    if len(seeds) == 0:
        return 0.0
    with_black_neighbor = 0
    h, w = grid.shape
    for r, c in seeds:
        r, c = int(r), int(c)
        for dr, dc in _GEN_DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == _GEN_BLACK:
                with_black_neighbor += 1
                break
    return with_black_neighbor / len(seeds)


def _gen_sample_blocker_layout():
    size = random.randint(_GEN_SIZE_MIN, _GEN_SIZE_MAX)
    n_cells = size * size
    min_blockers = max(_GEN_BLOCKER_ABS_MIN, int(np.ceil(_GEN_BLOCKER_FRAC_MIN * n_cells)))
    max_blockers = int(np.floor(_GEN_BLOCKER_FRAC_MAX * n_cells))
    n_blockers = random.randint(min_blockers, max_blockers)
    return size, n_cells, _gen_scattered_blockers(size, n_blockers)


def _gen_valid_blocker_layout(blocker_mask):
    blocker_components = components_mask(blocker_mask, connectivity=4)
    avg_blocker_neighbors = _gen_blocker_neighbor_average(blocker_mask)
    return (
        _GEN_BLOCKER_COMPONENT_MIN <= len(blocker_components) <= _GEN_BLOCKER_COMPONENT_MAX
        and _GEN_BLOCKER_NEIGHBOR_AVG_MIN <= avg_blocker_neighbors <= _GEN_BLOCKER_NEIGHBOR_AVG_MAX)


def _gen_design_reachability(blocker_mask, n_cells):
    black_components = components_mask(~blocker_mask, connectivity=4)
    if len(black_components) < _GEN_BLACK_COMPONENT_REJECT_MIN:
        return None

    seed_cap = min(_GEN_BLUE_COUNT_MAX, int(np.floor(_GEN_BLUE_FRAC_MAX * n_cells)))
    min_unreached, max_unreached = _gen_unreached_bounds(n_cells)
    reachable = _gen_choose_reachable_components(black_components, seed_cap, min_unreached, max_unreached)
    if reachable is None:
        return None

    min_seeds = max(_GEN_BLUE_COUNT_MIN, int(np.ceil(_GEN_BLUE_FRAC_MIN * n_cells)), len(reachable))
    max_seeds = min(seed_cap, sum(len(black_components[i]) for i in reachable))
    if min_seeds > max_seeds:
        return None
    n_seeds = random.randint(min_seeds, max_seeds)
    seeds = _gen_seed_cells(black_components, reachable, n_seeds)
    if seeds is None:
        return None
    return black_components, reachable, seeds, min_unreached, max_unreached


def _gen_reachable_mask(shape, black_components, reachable):
    mask = np.zeros(shape, dtype=bool)
    for idx in reachable:
        for r, c in black_components[idx]:
            mask[int(r), int(c)] = True
    return mask


def _gen_render_pair(blocker_mask, black_components, reachable, seeds):
    out = _gen_paint_blockers(blocker_mask)
    reachable_mask = _gen_reachable_mask(blocker_mask.shape, black_components, reachable)
    out[reachable_mask] = _GEN_BLUE

    inp = out.copy()
    inp[reachable_mask] = _GEN_BLACK
    for r, c in seeds:
        inp[r, c] = _GEN_BLUE
    return inp, out


def _gen_valid_cell_mix(inp, n_cells):
    black_count = int((inp == _GEN_BLACK).sum())
    blue_count = int((inp == _GEN_BLUE).sum())
    blocker_count = int(((inp >= _GEN_BLOCKER_COLORS[0]) & (inp <= _GEN_BLOCKER_COLORS[-1])).sum())
    return (
        _GEN_BLACK_FRAC_MIN <= black_count / n_cells <= _GEN_BLACK_FRAC_MAX
        and _GEN_BLUE_FRAC_MIN <= blue_count / n_cells <= _GEN_BLUE_FRAC_MAX
        and _GEN_BLUE_COUNT_MIN <= blue_count <= _GEN_BLUE_COUNT_MAX
        and _GEN_BLOCKER_FRAC_MIN <= blocker_count / n_cells <= _GEN_BLOCKER_FRAC_MAX)


def _gen_valid_unreached_regions(out, min_unreached, max_unreached):
    unreached_count = int((out == _GEN_BLACK).sum())
    unreached_components = components_mask(out == _GEN_BLACK, connectivity=4)
    return (
        min_unreached <= unreached_count <= max_unreached
        and _GEN_UNREACHED_COMPONENT_MIN <= len(unreached_components) <= _GEN_UNREACHED_COMPONENT_MAX)


def _gen_attempt():
    _, n_cells, blocker_mask = _gen_sample_blocker_layout()
    if not _gen_valid_blocker_layout(blocker_mask):
        return None

    design = _gen_design_reachability(blocker_mask, n_cells)
    if design is None:
        return None
    black_components, reachable, seeds, min_unreached, max_unreached = design

    inp, out = _gen_render_pair(blocker_mask, black_components, reachable, seeds)
    if _gen_seed_black_neighbor_fraction(inp) < _GEN_SEED_BLACK_NEIGHBOR_MIN:
        return None
    if not _gen_valid_cell_mix(inp, n_cells):
        return None
    if not _gen_valid_unreached_regions(out, min_unreached, max_unreached):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
