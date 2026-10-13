import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
TEMPLATE_BORDER = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GRID_SIDE = 28
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_single_component(mask: np.ndarray) -> bool:
    comps = _components_mask(mask)
    return len(comps) == 1


def _find_frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    out: list[dict] = []
    seen = set()
    for color in colors:
        for comp in _components_mask(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            key = (r0, c0, r1, c1, color)
            if key in seen:
                continue
            if not bool((g[r0, c0 : c1 + 1] == color).all()):
                continue
            if not bool((g[r1, c0 : c1 + 1] == color).all()):
                continue
            if not bool((g[r0 : r1 + 1, c0] == color).all()):
                continue
            if not bool((g[r0 : r1 + 1, c1] == color).all()):
                continue
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or bool((inner == color).all()):
                continue
            out.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "outer_shape": (r1 - r0 + 1, c1 - c0 + 1),
                    "inner": inner.copy(),
                }
            )
            seen.add(key)
    return out


def _source_like(candidate: dict, bg: int) -> bool:
    inner = np.asarray(candidate["inner"], dtype=int)
    vals = set(map(int, np.unique(inner))) - {bg}
    return vals.issubset({int(candidate["color"])}) and np.any(inner == int(candidate["color"]))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates = _find_frame_candidates(g, bg)
    assert candidates, "expected at least one bordered rectangle"

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for cand in candidates:
        groups[tuple(cand["outer_shape"])].append(cand)

    solutions: dict[bytes, np.ndarray] = {}
    for group in groups.values():
        if not (3 <= len(group) <= 5):
            continue
        for template_idx, template in enumerate(group):
            sources = [cand for i, cand in enumerate(group) if i != template_idx]
            if not (2 <= len(sources) <= 4):
                continue
            if len({int(src["color"]) for src in sources}) != len(sources):
                continue
            if any(not _source_like(src, bg) for src in sources):
                continue

            template_inner = np.asarray(template["inner"], dtype=int)
            template_colors = set(map(int, np.unique(template_inner))) - {bg, int(template["color"])}
            source_by_color = {int(src["color"]): src for src in sources}
            if template_colors != set(source_by_color):
                continue
            if any(not _is_single_component(template_inner == color) for color in template_colors):
                continue

            out = np.full(template_inner.shape, bg, dtype=int)
            valid = True
            for color, src in source_by_color.items():
                src_inner = np.asarray(src["inner"], dtype=int)
                if src_inner.shape != template_inner.shape:
                    valid = False
                    break
                mask = template_inner == color
                out[mask] = src_inner[mask]
            if valid:
                solutions[out.tobytes() + str(out.shape).encode()] = out

    assert len(solutions) == 1, "failed to isolate a unique template/source grouping"
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compose a same-sized mosaic by using the template's colored regions as source selectors.

    Each colored region inside the template points to the source rectangle with the same border
    color. The output cell at a template position is copied from the matching source's interior at
    the same relative position.
    """
    return _solve(grid)


def _grow_connected_mask(h: int, w: int, target: int, rng: random.Random) -> np.ndarray:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(cells) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _draw_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, value: bool) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = value


def _make_source_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    area = h * w
    for _ in range(80):
        family = rng.randrange(3)
        mask = np.zeros((h, w), dtype=bool)
        if family == 0:
            target = rng.randint(max(4, area // 4), min(area - 4, (3 * area) // 4))
            mask = _grow_connected_mask(h, w, target, rng)
            if rng.random() < 0.5 and h >= 4 and w >= 4:
                rr = rng.randint(1, h - 2)
                cc = rng.randint(1, w - 2)
                mask[rr, cc] = False
        elif family == 1:
            step = rng.choice((2, 3))
            row_offset = rng.randrange(step)
            col_offset = rng.randrange(step)
            mask[row_offset::step, :] = True
            if rng.random() < 0.7:
                mask[:, col_offset::step] ^= True
            if rng.random() < 0.6:
                rr0 = rng.randrange(h)
                rr1 = rng.randrange(rr0, h)
                mask[rr0 : rr1 + 1, rng.randrange(w)] = True
            if rng.random() < 0.5 and h >= 3 and w >= 3:
                cut_h = rng.randint(1, max(1, h // 2))
                cut_w = rng.randint(1, max(1, w // 2))
                r0 = rng.randint(0, h - cut_h)
                c0 = rng.randint(0, w - cut_w)
                _draw_rect(mask, r0, c0, r0 + cut_h - 1, c0 + cut_w - 1, False)
        else:
            for _ in range(rng.randint(2, 4)):
                if rng.random() < 0.5:
                    rr = rng.randrange(h)
                    c0 = rng.randrange(w)
                    c1 = rng.randrange(c0, w)
                    mask[rr, c0 : c1 + 1] = True
                else:
                    cc = rng.randrange(w)
                    r0 = rng.randrange(h)
                    r1 = rng.randrange(r0, h)
                    mask[r0 : r1 + 1, cc] = True
            for _ in range(rng.randint(1, 3)):
                rh = rng.randint(1, max(1, min(3, h)))
                rw = rng.randint(1, max(1, min(3, w)))
                r0 = rng.randint(0, h - rh)
                c0 = rng.randint(0, w - rw)
                _draw_rect(mask, r0, c0, r0 + rh - 1, c0 + rw - 1, True)
            if rng.random() < 0.4 and h >= 4 and w >= 4:
                rh = rng.randint(1, max(1, h // 2))
                rw = rng.randint(1, max(1, w // 2))
                r0 = rng.randint(0, h - rh)
                c0 = rng.randint(0, w - rw)
                _draw_rect(mask, r0, c0, r0 + rh - 1, c0 + rw - 1, False)

        if h == w:
            for _ in range(rng.randrange(4)):
                mask = np.rot90(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)

        fg = int(mask.sum())
        if 3 <= fg <= area - 3:
            return mask
    raise RuntimeError("failed to create a non-degenerate source mask")


def _partition_connected(h: int, w: int, k: int, rng: random.Random) -> np.ndarray:
    total = h * w
    min_area = max(3, total // (3 * k))
    for _ in range(120):
        labels = np.full((h, w), -1, dtype=int)
        cells = [(r, c) for r in range(h) for c in range(w)]
        seeds = rng.sample(cells, k)
        frontiers = [set() for _ in range(k)]
        areas = [0] * k
        unassigned = total

        for idx, (r, c) in enumerate(seeds):
            labels[r, c] = idx
            areas[idx] = 1
            unassigned -= 1
        for idx, (r, c) in enumerate(seeds):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                    frontiers[idx].add((nr, nc))

        def grow(idx: int) -> bool:
            if not frontiers[idx]:
                return False
            cell = rng.choice(tuple(frontiers[idx]))
            frontiers[idx].remove(cell)
            r, c = cell
            if labels[r, c] != -1:
                return False
            labels[r, c] = idx
            areas[idx] += 1
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                    frontiers[idx].add((nr, nc))
            return True

        while unassigned > 0 and any(area < min_area for area in areas):
            progress = False
            order = list(range(k))
            rng.shuffle(order)
            for idx in order:
                if areas[idx] < min_area and grow(idx):
                    unassigned -= 1
                    progress = True
            if not progress:
                break
        if unassigned == 0 and all(area >= min_area for area in areas):
            pass
        else:
            while unassigned > 0:
                choices = [idx for idx in range(k) if frontiers[idx]]
                if not choices:
                    break
                avg = total / k
                weights = [max(0.25, avg - areas[idx] + 1.0) for idx in choices]
                idx = rng.choices(choices, weights=weights, k=1)[0]
                if grow(idx):
                    unassigned -= 1
            if unassigned != 0:
                continue

        if any(not _is_single_component(labels == idx) for idx in range(k)):
            continue
        if min(np.sum(labels == idx) for idx in range(k)) < min_area:
            continue
        return labels
    raise RuntimeError("failed to partition template into connected regions")


def _source_grid_shape(n: int, layout: str, rng: random.Random) -> tuple[int, int]:
    if layout == "tb":
        options = [(1, n)]
        if n > 1:
            options.append((2, (n + 1) // 2))
        if n == 4:
            options = [(2, 2)]
    else:
        options = [(n, 1)]
        if n > 1:
            options.append(((n + 1) // 2, 2))
        if n == 4:
            options = [(2, 2)]
    return rng.choice(options)


def _pad_canvas(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = grid.shape
    extra_h = rng.randint(2, max(2, MAX_GRID_SIDE - h))
    extra_w = rng.randint(2, max(2, MAX_GRID_SIDE - w))
    top = rng.randint(1, extra_h - 1)
    bottom = extra_h - top
    left = rng.randint(1, extra_w - 1)
    right = extra_w - left
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=BG)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_sources = rng.randint(2, 4)
    layout = rng.choice(("tb", "lr"))
    src_rows, src_cols = _source_grid_shape(n_sources, layout, rng)
    ih = rng.randint(4, 7)
    iw = rng.randint(4, 7)
    oh, ow = ih + 2, iw + 2
    gap = rng.randint(1, 2)
    sep = rng.randint(2, 3)

    if layout == "tb":
        min_h = src_rows * oh + (src_rows - 1) * gap + sep + oh
        min_w = src_cols * ow + (src_cols - 1) * gap
    else:
        min_h = src_rows * oh + (src_rows - 1) * gap
        min_w = src_cols * ow + (src_cols - 1) * gap + sep + ow
    if min_h > MAX_GRID_SIDE - 2 or min_w > MAX_GRID_SIDE - 2:
        return None

    source_colors = rng.sample(SOURCE_COLORS, n_sources)
    labels = _partition_connected(ih, iw, n_sources, rng)
    template_inner = np.zeros((ih, iw), dtype=int)
    for idx, color in enumerate(source_colors):
        template_inner[labels == idx] = color

    source_masks = []
    source_inners = []
    for idx, color in enumerate(source_colors):
        region = labels == idx
        mask = _make_source_mask(ih, iw, rng)
        if np.array_equal(mask, region) or np.array_equal(mask, ~region):
            return None
        if int(np.sum(mask[region])) == 0 or int(np.sum(~mask[region])) == 0:
            return None
        if any(np.array_equal(mask, prev) for prev in source_masks):
            return None
        source_masks.append(mask)
        source_inners.append(np.where(mask, color, BG))

    source_slots = []
    if layout == "tb":
        block_w = src_cols * ow + (src_cols - 1) * gap
        block_h = src_rows * oh + (src_rows - 1) * gap
        for idx in range(n_sources):
            rr = idx // src_cols
            cc = idx % src_cols
            source_slots.append((rr * (oh + gap), cc * (ow + gap)))
        template_left = rng.randint(0, block_w - ow)
        template_top = block_h + sep
        core = np.full((block_h + sep + oh, block_w), BG, dtype=int)
    else:
        block_w = src_cols * ow + (src_cols - 1) * gap
        block_h = src_rows * oh + (src_rows - 1) * gap
        for idx in range(n_sources):
            rr = idx // src_cols
            cc = idx % src_cols
            source_slots.append((rr * (oh + gap), cc * (ow + gap)))
        template_top = rng.randint(0, block_h - oh)
        template_left = block_w + sep
        core = np.full((block_h, block_w + sep + ow), BG, dtype=int)

    rng.shuffle(source_slots)
    for (top, left), color, inner in zip(source_slots, source_colors, source_inners):
        core[top : top + oh, left : left + ow] = color
        core[top + 1 : top + 1 + ih, left + 1 : left + 1 + iw] = inner

    core[template_top : template_top + oh, template_left : template_left + ow] = TEMPLATE_BORDER
    core[template_top + 1 : template_top + 1 + ih, template_left + 1 : template_left + 1 + iw] = template_inner

    grid = _pad_canvas(core, rng)
    if grid.shape[0] > MAX_GRID_SIDE or grid.shape[1] > MAX_GRID_SIDE:
        return None
    if _get_bg(grid) != BG:
        return None

    expected = np.full((ih, iw), BG, dtype=int)
    for color, inner in zip(source_colors, source_inners):
        mask = template_inner == color
        expected[mask] = inner[mask]

    visible = set(map(int, np.unique(expected))) - {BG}
    if visible != set(source_colors):
        return None
    if any(np.array_equal(expected, inner) for inner in source_inners):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
