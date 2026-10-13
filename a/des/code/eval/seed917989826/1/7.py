import random
from collections import deque

import numpy as np


BG = 0
MIRROR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = NEIGHBORS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_mirror(component: list[tuple[int, int]], grid: np.ndarray) -> dict:
    rows = sorted({r for r, _ in component})
    cols = sorted({c for _, c in component})

    if len(rows) == 1:
        row = rows[0]
        line_cols = sorted(c for _, c in component)
        if line_cols != list(range(line_cols[0], line_cols[-1] + 1)):
            raise AssertionError("horizontal mirror must be contiguous")
        if len(line_cols) < 3:
            raise AssertionError("mirror too short")
        ep0 = (row, line_cols[0] - 1)
        ep1 = (row, line_cols[-1] + 1)
        if not (_in_bounds(grid.shape, *ep0) and _in_bounds(grid.shape, *ep1)):
            raise AssertionError("horizontal mirror endpoints out of bounds")
        colors = (int(grid[ep0]), int(grid[ep1]))
        if colors[0] in {BG, MIRROR} or colors[1] in {BG, MIRROR} or colors[0] == colors[1]:
            raise AssertionError("invalid horizontal mirror markers")
        return {
            "orientation": "h",
            "axis": row,
            "span": (line_cols[0], line_cols[-1]),
            "endpoints": (ep0, ep1),
            "sort_key": (row, line_cols[0]),
        }

    if len(cols) == 1:
        col = cols[0]
        line_rows = sorted(r for r, _ in component)
        if line_rows != list(range(line_rows[0], line_rows[-1] + 1)):
            raise AssertionError("vertical mirror must be contiguous")
        if len(line_rows) < 3:
            raise AssertionError("mirror too short")
        ep0 = (line_rows[0] - 1, col)
        ep1 = (line_rows[-1] + 1, col)
        if not (_in_bounds(grid.shape, *ep0) and _in_bounds(grid.shape, *ep1)):
            raise AssertionError("vertical mirror endpoints out of bounds")
        colors = (int(grid[ep0]), int(grid[ep1]))
        if colors[0] in {BG, MIRROR} or colors[1] in {BG, MIRROR} or colors[0] == colors[1]:
            raise AssertionError("invalid vertical mirror markers")
        return {
            "orientation": "v",
            "axis": col,
            "span": (line_rows[0], line_rows[-1]),
            "endpoints": (ep0, ep1),
            "sort_key": (line_rows[0], col),
        }

    raise AssertionError("mirror must be straight")


def _adjacent_source_seeds(grid: np.ndarray, mirror: dict) -> list[tuple[int, int]]:
    if mirror["orientation"] == "h":
        row = int(mirror["axis"])
        c0, c1 = map(int, mirror["span"])
        above = []
        below = []
        if row > 0:
            above = [(row - 1, c) for c in range(c0, c1 + 1) if int(grid[row - 1, c]) not in {BG, MIRROR}]
        if row + 1 < grid.shape[0]:
            below = [(row + 1, c) for c in range(c0, c1 + 1) if int(grid[row + 1, c]) not in {BG, MIRROR}]
        sides = [above, below]
    else:
        col = int(mirror["axis"])
        r0, r1 = map(int, mirror["span"])
        left = []
        right = []
        if col > 0:
            left = [(r, col - 1) for r in range(r0, r1 + 1) if int(grid[r, col - 1]) not in {BG, MIRROR}]
        if col + 1 < grid.shape[1]:
            right = [(r, col + 1) for r in range(r0, r1 + 1) if int(grid[r, col + 1]) not in {BG, MIRROR}]
        sides = [left, right]

    nonempty = [cells for cells in sides if cells]
    if len(nonempty) != 1:
        raise AssertionError("mirror must touch exactly one source side")
    return nonempty[0]


def _collect_component(
    grid: np.ndarray,
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if start in blocked or int(grid[start]) in {BG, MIRROR}:
        raise AssertionError("invalid component seed")
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in blocked or (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) in {BG, MIRROR}:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _reflect(mirror: dict, r: int, c: int) -> tuple[int, int]:
    if mirror["orientation"] == "h":
        return 2 * int(mirror["axis"]) - r, c
    return r, 2 * int(mirror["axis"]) - c


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    mirrors = sorted(
        (_parse_mirror(comp, g) for comp in _mask_components(g == MIRROR)),
        key=lambda item: item["sort_key"],
    )
    assert mirrors, "expected at least one mirror"

    out = g.copy()
    for mirror in mirrors:
        seeds = _adjacent_source_seeds(g, mirror)
        component = _collect_component(g, seeds[0], set(mirror["endpoints"]))
        if any(seed not in component for seed in seeds):
            raise AssertionError("adjacent source cells must form one connected object")

        ep0, ep1 = mirror["endpoints"]
        d0 = min(abs(r - ep0[0]) + abs(c - ep0[1]) for r, c in component)
        d1 = min(abs(r - ep1[0]) + abs(c - ep1[1]) for r, c in component)
        if d0 == d1:
            raise AssertionError("endpoint direction must be unambiguous")

        src_ep, dst_ep = (ep0, ep1) if d0 < d1 else (ep1, ep0)
        src_key = int(g[src_ep])
        dst_key = int(g[dst_ep])
        if not any(int(g[r, c]) == src_key for r, c in component):
            raise AssertionError("source-key color must appear inside the source pattern")

        for r, c in component:
            rr, cc = _reflect(mirror, r, c)
            if not _in_bounds(g.shape, rr, cc):
                raise AssertionError("reflection left the canvas")
            color = dst_key if int(g[r, c]) == src_key else int(g[r, c])
            existing = int(out[rr, cc])
            if existing not in {BG, color}:
                raise AssertionError("reflections must not conflict")
            out[rr, cc] = color

    return out


def _connected_mask(
    h: int,
    w: int,
    required: set[tuple[int, int]],
    area: int,
) -> np.ndarray:
    cells = set(required)
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            raise AssertionError("failed to grow connected shape")
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _paint_source(mask: np.ndarray, anchor: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.randint(2, min(3, len(coords)))
    source_colors = random.sample(list(PALETTE), color_count)
    source_key = random.choice(source_colors)
    target_color = random.choice([color for color in PALETTE if color not in source_colors])

    patch = np.full(mask.shape, BG, dtype=int)
    patch[anchor] = source_key

    remaining = [coord for coord in coords if coord != anchor]
    random.shuffle(remaining)

    required_colors: list[int] = [color for color in source_colors if color != source_key]
    if len(coords) >= 5:
        required_colors.insert(0, source_key)

    for coord, color in zip(remaining, required_colors):
        patch[coord] = color

    for coord in remaining:
        if patch[coord] == BG:
            patch[coord] = random.choice(source_colors)

    return patch, source_key, target_color


def _paste_nonzero(dest: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    region = dest[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BG
    if np.any(region[mask] != BG):
        raise AssertionError("patch overlap")
    region[mask] = patch[mask]


def _make_horizontal_pair(source_above: bool, near_left: bool) -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(2, 4)
    w = random.randint(2, 5)
    area = random.randint(4, min(8, h * w))

    boundary_row = h - 1 if source_above else 0
    anchor = (boundary_row, 0 if near_left else w - 1)
    required = {anchor}
    if w >= 3 and random.random() < 0.6:
        step = 1 if near_left else -1
        required.add((boundary_row, anchor[1] + step))

    mask = _connected_mask(h, w, required, area)
    source_patch, source_key, target_color = _paint_source(mask, anchor)

    short_gap = random.randint(0, 1)
    long_gap = random.randint(2, 4)
    left_gap, right_gap = (short_gap, long_gap) if near_left else (long_gap, short_gap)
    mirror_len = left_gap + w + right_gap

    patch_h = 2 * h + 1
    patch_w = mirror_len + 2
    mirror_r = h
    source_r0 = 0 if source_above else h + 1
    source_c0 = 1 + left_gap

    left_marker = source_key if near_left else target_color
    right_marker = target_color if near_left else source_key

    inp = np.full((patch_h, patch_w), BG, dtype=int)
    out = np.full_like(inp, BG)
    inp[mirror_r, 1 : mirror_len + 1] = MIRROR
    out[mirror_r, 1 : mirror_len + 1] = MIRROR
    inp[mirror_r, 0] = left_marker
    inp[mirror_r, patch_w - 1] = right_marker
    out[mirror_r, 0] = left_marker
    out[mirror_r, patch_w - 1] = right_marker

    _paste_nonzero(inp, source_patch, source_r0, source_c0)
    _paste_nonzero(out, source_patch, source_r0, source_c0)

    for r, c in np.argwhere(source_patch != BG):
        rr = 2 * mirror_r - (source_r0 + int(r))
        cc = source_c0 + int(c)
        color = int(source_patch[r, c])
        out[rr, cc] = target_color if color == source_key else color

    return inp, out


def _make_vertical_pair(source_left: bool, near_top: bool) -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(2, 5)
    w = random.randint(2, 4)
    area = random.randint(4, min(8, h * w))

    boundary_col = w - 1 if source_left else 0
    anchor = (0 if near_top else h - 1, boundary_col)
    required = {anchor}
    if h >= 3 and random.random() < 0.6:
        step = 1 if near_top else -1
        required.add((anchor[0] + step, boundary_col))

    mask = _connected_mask(h, w, required, area)
    source_patch, source_key, target_color = _paint_source(mask, anchor)

    short_gap = random.randint(0, 1)
    long_gap = random.randint(2, 4)
    top_gap, bottom_gap = (short_gap, long_gap) if near_top else (long_gap, short_gap)
    mirror_len = top_gap + h + bottom_gap

    patch_h = mirror_len + 2
    patch_w = 2 * w + 1
    mirror_c = w
    source_r0 = 1 + top_gap
    source_c0 = 0 if source_left else w + 1

    top_marker = source_key if near_top else target_color
    bottom_marker = target_color if near_top else source_key

    inp = np.full((patch_h, patch_w), BG, dtype=int)
    out = np.full_like(inp, BG)
    inp[1 : mirror_len + 1, mirror_c] = MIRROR
    out[1 : mirror_len + 1, mirror_c] = MIRROR
    inp[0, mirror_c] = top_marker
    inp[patch_h - 1, mirror_c] = bottom_marker
    out[0, mirror_c] = top_marker
    out[patch_h - 1, mirror_c] = bottom_marker

    _paste_nonzero(inp, source_patch, source_r0, source_c0)
    _paste_nonzero(out, source_patch, source_r0, source_c0)

    for r, c in np.argwhere(source_patch != BG):
        rr = source_r0 + int(r)
        cc = 2 * mirror_c - (source_c0 + int(c))
        color = int(source_patch[r, c])
        out[rr, cc] = target_color if color == source_key else color

    return inp, out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = max(out.shape[0] for _, out in pairs)
    max_w = max(out.shape[1] for _, out in pairs)
    total_area = sum(out.shape[0] * out.shape[1] for _, out in pairs)

    h_lo = max(18, max_h + 2)
    w_lo = max(18, max_w + 2)
    h_hi = min(30, h_lo + 8)
    w_hi = min(30, w_lo + 10)
    if h_lo > 30 or w_lo > 30:
        return None

    height = random.randint(h_lo, max(h_lo, h_hi))
    width = random.randint(w_lo, max(w_lo, w_hi))
    while height * width < total_area * 2 and (height < 30 or width < 30):
        if height <= width and height < 30:
            height += 1
        elif width < 30:
            width += 1
        elif height < 30:
            height += 1
        else:
            break

    inp = np.full((height, width), BG, dtype=int)
    out = np.full_like(inp, BG)
    occupied = np.zeros((height, width), dtype=bool)

    ordered = sorted(pairs, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    for inp_patch, out_patch in ordered:
        mask = out_patch != BG
        ph, pw = out_patch.shape
        rows = range(1 if height - ph >= 2 else 0, height - ph + 1 - (1 if height - ph >= 2 else 0))
        cols = range(1 if width - pw >= 2 else 0, width - pw + 1 - (1 if width - pw >= 2 else 0))
        dilated = _dilate(occupied)
        options: list[tuple[int, int]] = []
        for r0 in rows:
            for c0 in cols:
                if not np.any(dilated[r0 : r0 + ph, c0 : c0 + pw] & mask):
                    options.append((r0, c0))
        if not options:
            return None
        r0, c0 = random.choice(options)
        _paste_nonzero(inp, inp_patch, r0, c0)
        _paste_nonzero(out, out_patch, r0, c0)
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3))

    horizontal_branches = [("h", True, True), ("h", True, False), ("h", False, True), ("h", False, False)]
    vertical_branches = [("v", True, True), ("v", True, False), ("v", False, True), ("v", False, False)]
    random.shuffle(horizontal_branches)
    random.shuffle(vertical_branches)
    branches = [horizontal_branches.pop(), vertical_branches.pop()]
    remainder = horizontal_branches + vertical_branches
    random.shuffle(remainder)
    branches.extend(remainder[: pair_count - 2])
    random.shuffle(branches)

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for orientation, flag0, flag1 in branches:
        if orientation == "h":
            pairs.append(_make_horizontal_pair(source_above=flag0, near_left=flag1))
        else:
            pairs.append(_make_vertical_pair(source_left=flag0, near_top=flag1))

    placed = _place_pairs(pairs)
    if placed is None:
        return None
    inp, expected = placed
    if np.array_equal(inp, expected):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-reflection puzzle")
