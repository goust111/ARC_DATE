from collections import Counter, deque

import numpy as np

from arc_utils import components_mask as _gen_components_mask, get_bg, neighbors4
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Flood-fill each marked chamber bounded by walls with its marker color.

    Input structure
    A same-size grid has a dominant background/open color, one repeated wall color,
    and one-cell colored markers sitting inside some wall-bounded open chambers.

    Rule
    Infer the wall as the unique non-background color that appears more than once.
    Treat wall pixels as barriers. For every 4-connected non-wall component, leave
    it unchanged if it has no marker; otherwise assert it has exactly one marker and
    recolor the whole component with that marker's color.

    Key insight
    The repeated color is a maze wall that partitions the background into chambers;
    the singleton marker inside a chamber supplies the fill color for that chamber.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)

    counts = Counter(map(int, g.ravel()))
    wall_colors = [color for color, count in counts.items() if color != bg and count > 1]
    assert len(wall_colors) == 1
    wall = wall_colors[0]

    marker_colors = [color for color in counts if color not in (bg, wall)]
    assert all(counts[color] == 1 for color in marker_colors)

    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or g[sr, sc] == wall:
                continue

            component = []
            markers = []
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                if g[r, c] != bg:
                    markers.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != wall:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            if markers:
                assert len(markers) == 1
                color = int(g[markers[0]])
                for r, c in component:
                    out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

# Neighborhood and sentinel values.
_GEN_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_EMPTY_LABEL, _GEN_BG_COLOR, _GEN_WALL_COLOR = -1, 0, 5
# Canvas, component counts, and wall budgets.
_GEN_GRID_SIZES = [7, 9]
_GEN_MARKED_COUNTS = [2, 3]
_GEN_UNMARKED_COUNTS_7 = [5, 5, 6, 6, 7, 8]
_GEN_UNMARKED_COUNT_RANGE_9 = (5, 8)
_GEN_WALL_COUNTS_7 = [19, 20, 21]
_GEN_WALL_COUNTS_9 = [32, 33, 34, 35, 36]
_GEN_MAX_WALL_COMPONENT_7, _GEN_MAX_WALL_COMPONENT_9 = 9, 12
# Chamber size ranges.
_GEN_MARKED_SIZE_RANGE, _GEN_UNMARKED_SIZE_RANGE = (3, 13), (1, 10)
# Validation thresholds.
_GEN_MIN_WALL_4_COMPONENTS = 5
_GEN_WALL_8_COMPONENT_MIN_FRACTION = 0.75
_GEN_BORDER_WALL_RANGE = (8, 12)
# Retry limits and marker palette.
_GEN_START_TRIAL_LIMIT, _GEN_GROW_RESTARTS, _GEN_RETRY_LIMIT = 80, 8, 300
_GEN_MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _gen_sample_counts(n):
    marked_count = random.choice(_GEN_MARKED_COUNTS)
    if n == 7:
        unmarked_count = random.choice(_GEN_UNMARKED_COUNTS_7)
        wall_count = random.choice(_GEN_WALL_COUNTS_7)
        max_wall_component = _GEN_MAX_WALL_COMPONENT_7
    else:
        unmarked_count = random.randint(*_GEN_UNMARKED_COUNT_RANGE_9)
        wall_count = random.choice(_GEN_WALL_COUNTS_9)
        max_wall_component = _GEN_MAX_WALL_COMPONENT_9
    return marked_count, unmarked_count, wall_count, max_wall_component


def _gen_sample_component_sizes(total_open, marked_count, unmarked_count):
    marked_min, marked_max = _GEN_MARKED_SIZE_RANGE
    unmarked_min, unmarked_max = _GEN_UNMARKED_SIZE_RANGE
    mins = [marked_min] * marked_count + [unmarked_min] * unmarked_count
    maxs = [marked_max] * marked_count + [unmarked_max] * unmarked_count
    order = list(range(len(mins)))
    random.shuffle(order)

    values = [None] * len(mins)
    remaining = total_open
    for pos, idx in enumerate(order):
        rest = order[pos + 1 :]
        lo = max(mins[idx], remaining - sum(maxs[j] for j in rest))
        hi = min(maxs[idx], remaining - sum(mins[j] for j in rest))
        if lo > hi:
            return None
        values[idx] = random.randint(lo, hi)
        remaining -= values[idx]

    return values[:marked_count], values[marked_count:]


def _gen_can_use_cell(cell, labels, label):
    empty, dirs = _GEN_EMPTY_LABEL, _GEN_DIRS
    n = labels.shape[0]
    r, c = cell
    if not (0 <= r < n and 0 <= c < n) or labels[r, c] != empty:
        return False
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            neighbor_label = labels[nr, nc]
            if neighbor_label != empty and neighbor_label != label:
                return False
    return True


def _gen_place_component(labels, size, label, *, must_touch_border=False):
    empty, dirs = _GEN_EMPTY_LABEL, _GEN_DIRS
    n = labels.shape[0]
    starts = [(r, c) for r in range(n) for c in range(n) if labels[r, c] == empty]
    if must_touch_border:
        starts = [cell for cell in starts if cell[0] in (0, n - 1) or cell[1] in (0, n - 1)]
    random.shuffle(starts)

    for seed in starts[:_GEN_START_TRIAL_LIMIT]:
        for _ in range(_GEN_GROW_RESTARTS):
            if not _gen_can_use_cell(seed, labels, label):
                continue

            cells = {seed}
            labels[seed] = label
            ok = True
            for _ in range(size - 1):
                frontier = []
                for r, c in cells:
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if not _gen_can_use_cell((nr, nc), labels, label):
                            continue
                        same_neighbors = 0
                        for er, ec in dirs:
                            rr, cc = nr + er, nc + ec
                            if 0 <= rr < n and 0 <= cc < n and labels[rr, cc] == label:
                                same_neighbors += 1
                        frontier.extend([(nr, nc)] * (1 + same_neighbors))

                if not frontier:
                    ok = False
                    break

                cell = random.choice(frontier)
                cells.add(cell)
                labels[cell] = label

            if ok:
                return cells

            for r, c in cells:
                labels[r, c] = empty

    return None


def _gen_place_components(labels, marked_sizes, unmarked_sizes):
    """Place all chamber labels; returns marked/unmarked component sets or None."""
    items = [("marked", size) for size in marked_sizes]
    items.extend(("unmarked", size) for size in unmarked_sizes)
    random.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)

    marked_components = []
    unmarked_components = []
    for label, (role, size) in enumerate(items):
        component = _gen_place_component(
            labels,
            size,
            label,
            must_touch_border=(role == "marked"),
        )
        if component is None: return None
        (marked_components if role == "marked" else unmarked_components).append(component)

    return marked_components, unmarked_components


def _gen_border_wall_count(wall_mask):
    total = wall_mask[0].sum() + wall_mask[-1].sum()
    return int(total + wall_mask[1:-1, 0].sum() + wall_mask[1:-1, -1].sum())


def _gen_layout_is_valid(labels, marked_sizes, unmarked_sizes, wall_count, max_wall_component):
    """Validate chamber sizes and wall connectivity against the sampled plan."""
    wall_mask = labels == _GEN_EMPTY_LABEL
    actual_sizes = sorted(len(comp) for comp in _gen_components_mask(~wall_mask, connectivity=4))
    if actual_sizes != sorted(marked_sizes + unmarked_sizes):
        return False

    wall_components4 = _gen_components_mask(wall_mask, connectivity=4)
    if len(wall_components4) < _GEN_MIN_WALL_4_COMPONENTS or max(len(comp) for comp in wall_components4) > max_wall_component:
        return False

    wall_components8 = _gen_components_mask(wall_mask, connectivity=8)
    if max(len(comp) for comp in wall_components8) < int(wall_count * _GEN_WALL_8_COMPONENT_MIN_FRACTION):
        return False

    border_min, border_max = _GEN_BORDER_WALL_RANGE
    if not (border_min <= _gen_border_wall_count(wall_mask) <= border_max):
        return False

    return True


def _gen_render_pair(n, marked_components, unmarked_components):
    """Build the solved grid and corresponding input with one visible seed per marked chamber."""
    out = np.full((n, n), _GEN_WALL_COLOR, dtype=int)
    for component in unmarked_components:
        for r, c in component:
            out[r, c] = _GEN_BG_COLOR

    marker_colors = random.sample(_GEN_MARKER_COLORS, len(marked_components))
    inp = out.copy()
    for color, component in zip(marker_colors, marked_components):
        cells = list(component)
        for r, c in cells:
            out[r, c] = color
            inp[r, c] = _GEN_BG_COLOR
        r, c = random.choice(cells)
        inp[r, c] = color

    return inp, out


def _gen_attempt(_gen_n=None):
    n = _gen_n if _gen_n is not None else random.choice(_GEN_GRID_SIZES)
    marked_count, unmarked_count, wall_count, max_wall_component = _gen_sample_counts(n)
    sampled = _gen_sample_component_sizes(n * n - wall_count, marked_count, unmarked_count)
    if sampled is None: return None
    marked_sizes, unmarked_sizes = sampled

    labels = np.full((n, n), _GEN_EMPTY_LABEL, dtype=int)
    placed = _gen_place_components(labels, marked_sizes, unmarked_sizes)
    if placed is None:
        return None
    marked_components, unmarked_components = placed

    if not _gen_layout_is_valid(labels, marked_sizes, unmarked_sizes, wall_count, max_wall_component):
        return None

    inp, out = _gen_render_pair(n, marked_components, unmarked_components)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None

    return inp


def generate_input():
    n = random.choice(_GEN_GRID_SIZES)
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt(n)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e69241bd input")
