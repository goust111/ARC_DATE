import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

STAMP_COUNT_RANGE = (2, 4)
INTERIOR_HEIGHT_RANGE = (4, 8)
INTERIOR_WIDTH_RANGE = (4, 8)
COLOR_COUNT_RANGE = (2, 4)
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _scan_distinct_colors(arr: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in arr.ravel():
        color = int(value)
        if color != BLACK and color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _bar_segments(bar: np.ndarray) -> list[int]:
    segments: list[int] = []
    prev = None
    for value in bar:
        color = int(value)
        assert color != BLACK
        if color != prev:
            segments.append(color)
            prev = color
    assert segments
    assert len(segments) == len(set(segments))
    return segments


def _recolor(interior: np.ndarray, src_order: list[int], dst_order: list[int]) -> np.ndarray:
    assert len(src_order) == len(dst_order)
    out = np.array(interior, copy=True)
    for src, dst in zip(src_order, dst_order):
        out[interior == src] = dst
    return out


def _find_stamps(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    stamps: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border_h = r1 - r0 + 1
        border_w = c1 - c0 + 1
        perimeter = 2 * border_h + 2 * border_w - 4
        if len(comp) != perimeter:
            continue

        border_ok = True
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border != (g[rr, cc] == GREY):
                    border_ok = False
                    break
            if not border_ok:
                break
        if not border_ok or r1 + 1 >= h:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue

        bar = g[r1 + 1, c0 + 1 : c1]
        if bar.size != interior.shape[1] or np.any(bar == BLACK):
            continue

        src_order = _scan_distinct_colors(interior)
        dst_order = _bar_segments(bar)
        if len(src_order) < 2 or len(src_order) != len(dst_order):
            continue

        stamps.append((r0, c0, interior.copy(), bar.copy()))

    assert stamps, "expected at least one stamp"
    return sorted(stamps, key=lambda item: (item[0], item[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each framed stamp from its bar, remove the frame, then center-stack
    the transformed interiors from largest non-black area to smallest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    patterns: list[tuple[int, int, int, np.ndarray]] = []
    for r0, c0, interior, bar in _find_stamps(g):
        src_order = _scan_distinct_colors(interior)
        dst_order = _bar_segments(bar)
        transformed = _recolor(interior, src_order, dst_order)
        area = int(np.count_nonzero(transformed))
        patterns.append((area, r0, c0, transformed))

    patterns.sort(key=lambda item: (-item[0], item[1], item[2]))
    canvas_h = max(patterns[0][3].shape[0], max(pat.shape[0] for _, _, _, pat in patterns))
    canvas_w = max(patterns[0][3].shape[1], max(pat.shape[1] for _, _, _, pat in patterns))

    out = np.zeros((canvas_h, canvas_w), dtype=int)
    for _area, _r0, _c0, pattern in patterns:
        ph, pw = pattern.shape
        top = (canvas_h - ph) // 2
        left = (canvas_w - pw) // 2
        out[top : top + ph, left : left + pw] = pattern
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    cuts = sorted(random.sample(range(1, total), parts - 1))
    last = 0
    widths: list[int] = []
    for cut in cuts + [total]:
        widths.append(cut - last)
        last = cut
    return widths


def _grow_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < area:
        if frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
        else:
            remaining = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
            cell = random.choice(remaining)
        r, c = cell
        if mask[r, c]:
            continue
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _random_mask(h: int, w: int, area: int) -> np.ndarray:
    total = h * w
    if area == total:
        return np.ones((h, w), dtype=bool)
    if random.random() < 0.5:
        return _grow_mask(h, w, area)
    carved = _grow_mask(h, w, total - area)
    return ~carved


def _assign_colors(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = mask.shape
    coords = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
    assert len(coords) >= len(colors)

    seed_coords = random.sample(coords, len(colors))
    priorities = list(range(len(colors)))
    random.shuffle(priorities)

    out = np.zeros((h, w), dtype=int)
    for r, c in coords:
        best_idx = min(
            range(len(colors)),
            key=lambda idx: (
                abs(r - seed_coords[idx][0]) + abs(c - seed_coords[idx][1]),
                priorities[idx],
            ),
        )
        out[r, c] = colors[best_idx]
    return out


def _make_bar(width: int, colors: list[int]) -> np.ndarray:
    widths = _random_composition(width, len(colors))
    bar = np.zeros(width, dtype=int)
    start = 0
    for color, seg_w in zip(colors, widths):
        bar[start : start + seg_w] = color
        start += seg_w
    return bar


def _make_stamp(h: int, w: int, n_colors: int, area: int) -> tuple[np.ndarray, np.ndarray]:
    mask = _random_mask(h, w, area)
    src_colors = random.sample(COLOR_POOL, n_colors)
    interior = _assign_colors(mask, src_colors)

    src_order = _scan_distinct_colors(interior)
    assert len(src_order) == n_colors

    dst_order = random.sample(COLOR_POOL, n_colors)
    if dst_order == src_order:
        dst_order = dst_order[1:] + dst_order[:1]
    transformed = _recolor(interior, src_order, dst_order)

    stamp = np.zeros((h + 3, w + 2), dtype=int)
    stamp[0, :] = GREY
    stamp[h + 1, :] = GREY
    stamp[: h + 2, 0] = GREY
    stamp[: h + 2, w + 1] = GREY
    stamp[1 : h + 1, 1 : w + 1] = interior
    stamp[h + 2, 1 : w + 1] = _make_bar(w, dst_order)
    return stamp, transformed


def _rectangles_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int
) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_stamps(patches: list[np.ndarray], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph > grid_h or pw > grid_w:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, grid_h - ph)
            c0 = random.randint(0, grid_w - pw)
            rect = (r0, c0, ph, pw)
            if any(_rectangles_overlap(rect, other, PLACEMENT_PADDING) for other in rects):
                continue
            rects.append(rect)
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _stack_patterns(items: list[tuple[int, int, int, np.ndarray]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (-item[0], item[1], item[2]))
    canvas_h = max(ordered[0][3].shape[0], max(pat.shape[0] for _, _, _, pat in ordered))
    canvas_w = max(ordered[0][3].shape[1], max(pat.shape[1] for _, _, _, pat in ordered))
    out = np.zeros((canvas_h, canvas_w), dtype=int)
    for _area, _r0, _c0, pattern in ordered:
        ph, pw = pattern.shape
        top = (canvas_h - ph) // 2
        left = (canvas_w - pw) // 2
        out[top : top + ph, left : left + pw] = pattern
    return out


def _gen_attempt() -> np.ndarray | None:
    n_stamps = random.randint(*STAMP_COUNT_RANGE)
    largest_h = random.randint(*INTERIOR_HEIGHT_RANGE)
    largest_w = random.randint(*INTERIOR_WIDTH_RANGE)
    largest_colors = random.randint(COLOR_COUNT_RANGE[0], min(COLOR_COUNT_RANGE[1], largest_h, largest_w))

    largest_area_min = max(largest_colors + 2, int(np.ceil(0.55 * largest_h * largest_w)))
    largest_area = random.randint(largest_area_min, largest_h * largest_w)

    patches: list[np.ndarray] = []
    transformed_items: list[tuple[int, int, int, np.ndarray]] = []
    used_areas = {largest_area}

    stamp, transformed = _make_stamp(largest_h, largest_w, largest_colors, largest_area)
    patches.append(stamp)
    transformed_items.append((largest_area, 0, 0, transformed))

    for _ in range(n_stamps - 1):
        for _retry in range(80):
            h = random.randint(INTERIOR_HEIGHT_RANGE[0], largest_h)
            w = random.randint(INTERIOR_WIDTH_RANGE[0], largest_w)
            n_colors = random.randint(COLOR_COUNT_RANGE[0], min(COLOR_COUNT_RANGE[1], h, w))
            min_area = max(n_colors + 1, int(np.ceil(0.35 * h * w)))
            max_area = min(h * w, largest_area - 1)
            available = [area for area in range(min_area, max_area + 1) if area not in used_areas]
            if not available:
                continue
            area = random.choice(available)
            stamp, transformed = _make_stamp(h, w, n_colors, area)
            used_areas.add(area)
            patches.append(stamp)
            transformed_items.append((area, 0, 0, transformed))
            break
        else:
            return None

    max_patch_h = max(patch.shape[0] for patch in patches)
    max_patch_w = max(patch.shape[1] for patch in patches)
    grid_h = random.randint(max(GRID_HEIGHT_RANGE[0], max_patch_h + 2), GRID_HEIGHT_RANGE[1])
    grid_w = random.randint(max(GRID_WIDTH_RANGE[0], max_patch_w + 2), GRID_WIDTH_RANGE[1])

    positions = _place_stamps(patches, grid_h, grid_w)
    if positions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    stacked_items: list[tuple[int, int, int, np.ndarray]] = []
    for patch, pos, item in zip(patches, positions, transformed_items):
        r0, c0 = pos
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        area, _, _, transformed = item
        stacked_items.append((area, r0, c0, transformed))

    expected = _stack_patterns(stacked_items)
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
