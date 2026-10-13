import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

RNG = np.random.default_rng()
GEN_ATTEMPTS = 400


def _find_separator(grid: np.ndarray) -> int:
    cols = np.where(np.all(grid == YELLOW, axis=0))[0]
    if len(cols) != 1:
        raise AssertionError("expected exactly one full-height yellow separator column")
    return int(cols[0])


def _swap_color(color: int) -> int:
    if int(color) == BLUE:
        return RED
    if int(color) == RED:
        return BLUE
    return int(color)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


def _build_swap_mask(shape: tuple[int, int], markers: np.ndarray, sep: int) -> np.ndarray:
    h, _w = shape
    mask = np.zeros(shape, dtype=bool)
    for r, c in markers:
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(int(sep), int(c) + 2)
        mask[r0:r1, c0:c1] = True
    mask[:, int(sep) :] = False
    return mask


def _apply_rule(grid: np.ndarray, markers_override: np.ndarray | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sep = _find_separator(g)
    left = g[:, :sep]
    markers = np.argwhere(left == GREY) if markers_override is None else np.asarray(markers_override, dtype=int)
    swap_mask = _build_swap_mask(g.shape, markers, sep)

    for r, c in np.argwhere(np.isin(left, (BLUE, RED))):
        rr = int(r)
        cc = int(c)
        mirror_c = 2 * int(sep) - cc
        if not (0 <= mirror_c < g.shape[1]):
            raise AssertionError("mirrored pixel falls outside the grid")
        color = int(g[rr, cc])
        if bool(swap_mask[rr, cc]):
            color = _swap_color(color)
        out[rr, mirror_c] = color
    return out


def _random_connected_mask(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(RNG.integers(height)), int(RNG.integers(width)))
    mask[start] = True
    while int(mask.sum()) < int(target):
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = list(frontier)[int(RNG.integers(len(frontier)))]
        mask[nr, nc] = True
    return mask


def _random_scattered_mask(height: int, width: int, target: int) -> np.ndarray | None:
    area = height * width
    if target > area:
        return None
    for _ in range(80):
        flat = np.zeros(area, dtype=bool)
        idx = RNG.choice(area, size=target, replace=False)
        flat[idx] = True
        mask = flat.reshape(height, width)
        sizes = _component_sizes(mask)
        if len(sizes) not in (2, 3):
            continue
        if min(sizes) < 2:
            continue
        return mask
    return None


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) < 2:
        raise AssertionError("need at least two pattern cells to colorize")

    seeds = coords[RNG.choice(len(coords), size=2, replace=False)]
    sub = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        d0 = abs(int(r) - int(seeds[0, 0])) + abs(int(c) - int(seeds[0, 1]))
        d1 = abs(int(r) - int(seeds[1, 0])) + abs(int(c) - int(seeds[1, 1]))
        if d0 < d1:
            color = BLUE
        elif d1 < d0:
            color = RED
        else:
            color = BLUE if ((int(r) + int(c)) % 2 == 0) else RED
        sub[int(r), int(c)] = color

    flip_cap = min(3, len(coords) // 3)
    if flip_cap > 0:
        flip_count = int(RNG.integers(0, flip_cap + 1))
        if flip_count:
            for rr, cc in coords[RNG.choice(len(coords), size=flip_count, replace=False)]:
                sub[int(rr), int(cc)] = _swap_color(int(sub[int(rr), int(cc)]))

    used = set(map(int, sub[mask]))
    if used != {BLUE, RED}:
        raise AssertionError("pattern must use exactly blue and red")
    return sub


def _marker_candidates(pattern_coords: set[tuple[int, int]], height: int, sep: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], set[tuple[int, int]]]]:
    inside = sorted(pattern_coords)
    ring = set()
    for r, c in pattern_coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < sep:
                    ring.add((nr, nc))
    outside = sorted(ring - pattern_coords)

    coverage = {}
    for pos in inside + outside:
        covered = {
            (pr, pc)
            for pr, pc in pattern_coords
            if max(abs(pr - pos[0]), abs(pc - pos[1])) <= 1
        }
        if covered:
            coverage[pos] = covered
    return inside, outside, coverage


def _choose_markers(pattern_coords: set[tuple[int, int]], height: int, sep: int) -> list[tuple[int, int]] | None:
    inside, outside, coverage = _marker_candidates(pattern_coords, height, sep)
    all_candidates = inside + outside
    if not all_candidates:
        return None

    counts = np.array([1, 2, 3], dtype=int)
    probs = np.array([0.42, 0.34, 0.24], dtype=float)
    probs = probs[counts <= len(all_candidates)]
    counts = counts[counts <= len(all_candidates)]
    probs = probs / probs.sum()

    diagonal_outside = []
    for pos in outside:
        if any(abs(pr - pos[0]) == 1 and abs(pc - pos[1]) == 1 for pr, pc in coverage[pos]):
            diagonal_outside.append(pos)

    for _ in range(120):
        count = int(RNG.choice(counts, p=probs))
        picked = []
        available = list(all_candidates)

        if diagonal_outside and RNG.random() < 0.7:
            first = diagonal_outside[int(RNG.integers(len(diagonal_outside)))]
            picked.append(first)
            available.remove(first)

        if inside and outside and len(picked) < count and RNG.random() < 0.35:
            inside_options = [pos for pos in inside if pos not in picked]
            if inside_options:
                first_inside = inside_options[int(RNG.integers(len(inside_options)))]
                picked.append(first_inside)
                if first_inside in available:
                    available.remove(first_inside)

        covered = set().union(*(coverage[pos] for pos in picked)) if picked else set()
        while len(picked) < count:
            options = [pos for pos in available if coverage[pos] - covered]
            if not options:
                break
            pos = options[int(RNG.integers(len(options)))]
            picked.append(pos)
            available.remove(pos)
            covered |= coverage[pos]

        if len(picked) == count:
            return picked
    return None


def _make_pattern_block(height: int, width: int) -> np.ndarray | None:
    area = int(height * width)
    min_cells = max(6, area // 2)
    max_cells = area - 1
    if max_cells < min_cells:
        max_cells = area
    if max_cells < min_cells:
        return None

    connected = bool(RNG.random() < 0.7)
    for _ in range(80):
        target = int(RNG.integers(min_cells, max_cells + 1))
        if connected:
            mask = _random_connected_mask(height, width, target)
            sizes = _component_sizes(mask)
            if len(sizes) != 1:
                continue
        else:
            mask = _random_scattered_mask(height, width, target)
            if mask is None:
                continue
        try:
            sub = _colorize_mask(mask)
        except AssertionError:
            continue
        if np.array_equal(sub, np.fliplr(sub)):
            continue
        return sub
    return None


def _extract_visible_pattern(left: np.ndarray) -> np.ndarray:
    mask = np.isin(left, (BLUE, RED))
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = box
    crop = left[r0 : r1 + 1, c0 : c1 + 1].copy()
    crop[~np.isin(crop, (BLUE, RED))] = BLACK
    return crop


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    try:
        sep = _find_separator(g)
    except AssertionError:
        return False
    if sep <= 0 or sep >= g.shape[1] - 1:
        return False
    if not np.all(g[:, sep + 1 :] == BLACK):
        return False

    left = g[:, :sep]
    pattern_mask = np.isin(left, (BLUE, RED))
    marker_mask = left == GREY
    if not pattern_mask.any():
        return False

    colors = set(map(int, left[pattern_mask]))
    if colors != {BLUE, RED}:
        return False

    markers = np.argwhere(marker_mask)
    if not (1 <= len(markers) <= 3):
        return False

    box = _bbox(pattern_mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if not (3 <= r1 - r0 + 1 <= 5 and 3 <= c1 - c0 + 1 <= 5):
        return False

    visible = _extract_visible_pattern(left)
    if visible.size == 0 or np.array_equal(visible, np.fliplr(visible)):
        return False

    sizes = _component_sizes(pattern_mask)
    if not sizes or len(sizes) > 3 or min(sizes) < 2:
        return False

    pattern_coords = {(int(r), int(c)) for r, c in np.argwhere(pattern_mask)}
    swap_mask = _build_swap_mask(g.shape, markers, sep)[:, :sep]
    masked = pattern_mask & swap_mask
    unmasked = pattern_mask & ~swap_mask
    masked_count = int(masked.sum())
    unmasked_count = int(unmasked.sum())
    if masked_count < 2 or unmasked_count < 2:
        return False

    if set(map(int, left[masked])) != {BLUE, RED}:
        return False
    if set(map(int, left[unmasked])) != {BLUE, RED}:
        return False

    diagonal_only = []
    marker_list = [tuple(map(int, rc)) for rc in markers]
    for r, c in pattern_coords:
        cheb = [max(abs(r - mr), abs(c - mc)) <= 1 for mr, mc in marker_list]
        manh = [abs(r - mr) + abs(c - mc) <= 1 for mr, mc in marker_list]
        if any(cheb) and not any(manh):
            diagonal_only.append((r, c))
    if not diagonal_only:
        return False

    expected = _apply_rule(g)
    changed = []
    unchanged = []
    for r, c in pattern_coords:
        mirror_c = 2 * sep - c
        src = int(left[r, c])
        dst = int(expected[r, mirror_c])
        if dst == _swap_color(src):
            changed.append((r, c))
        elif dst == src:
            unchanged.append((r, c))
        else:
            return False
    if len(changed) != masked_count or len(unchanged) != unmasked_count:
        return False

    plain_mirror = g.copy()
    for r, c in pattern_coords:
        plain_mirror[r, 2 * sep - c] = int(left[r, c])
    full_swap = g.copy()
    for r, c in pattern_coords:
        full_swap[r, 2 * sep - c] = _swap_color(int(left[r, c]))
    if np.array_equal(expected, plain_mirror) or np.array_equal(expected, full_swap):
        return False

    for idx in range(len(markers)):
        reduced = np.delete(markers, idx, axis=0)
        if np.array_equal(expected, _apply_rule(g, reduced)):
            return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the left two-color pattern across the yellow separator while swapping colors only inside marker-defined dilation zones.

    Input structure:
    - A black grid contains exactly one full-height yellow separator column.
    - The left half contains a blue/red pattern plus 1-3 grey marker pixels; the right half is empty.
    - Grey markers may sit on or next to the pattern, and only blue/red pattern pixels are mirrored.

    Rule:
    - Identify the yellow separator and all grey markers on its left side.
    - Take the union of each marker's 3x3 neighborhood, clipped to the left half, as the swap mask.
    - Mirror every left-side blue/red pixel across the separator onto the right: pixels inside the swap mask change blue<->red, while pixels outside keep their original color.
    - Keep the original left pattern, grey markers, and yellow separator unchanged.

    Key insight:
    - Grey markers do not belong to the mirrored motif; their dilated spatial influence selectively toggles whether the mirrored copy preserves or swaps each source color.
    """
    return _apply_rule(grid)


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(10, 15))
    width = int(RNG.integers(14, 19))

    sep_low = max(5, width // 2 - 1)
    sep_high = min(width - 6, width // 2 + 1)
    if sep_low > sep_high:
        return None
    sep = int(RNG.integers(sep_low, sep_high + 1))

    mirrorable_left = max(0, 2 * sep - (width - 1))
    max_pattern_width = min(5, sep - 1 - mirrorable_left)
    if max_pattern_width < 3:
        return None

    pattern_height = int(RNG.integers(3, 6))
    pattern_width = int(RNG.integers(3, max_pattern_width + 1))

    row_max = height - pattern_height - 1
    col_max = sep - pattern_width - 1
    if row_max < 1 or col_max < mirrorable_left:
        return None
    row0 = int(RNG.integers(1, row_max + 1))
    col0 = int(RNG.integers(mirrorable_left, col_max + 1))

    block = _make_pattern_block(pattern_height, pattern_width)
    if block is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, sep] = YELLOW
    sub_rows, sub_cols = np.where(np.isin(block, (BLUE, RED)))
    pattern_coords = set()
    for rr, cc in zip(sub_rows.tolist(), sub_cols.tolist()):
        ar = row0 + int(rr)
        ac = col0 + int(cc)
        grid[ar, ac] = int(block[rr, cc])
        pattern_coords.add((ar, ac))

    markers = _choose_markers(pattern_coords, height, sep)
    if markers is None:
        return None
    for r, c in markers:
        grid[int(r), int(c)] = GREY

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-swap mirror input")
