import random

import numpy as np


BG = 8
GREY = 5
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in _neighbors4(r, c, h, w):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_edge = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True
                for rr, cc in _neighbors4(r, c, h, w):
                    if bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_edge:
                holes += 1
    return holes


def _outside_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return outside


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return mask.copy()
    return np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)


def _random_polyomino(
    cell_count: int,
    min_h: int,
    min_w: int,
    max_h: int,
    max_w: int,
    attempts: int = 80,
) -> np.ndarray | None:
    for _ in range(attempts):
        cells = {(0, 0)}
        frontier = set()
        frontier.update({(1, 0), (-1, 0), (0, 1), (0, -1)})
        while len(cells) < cell_count and frontier:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            cells.add((r, c))
            for dr, dc in DIRS4:
                nb = (r + dr, c + dc)
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != cell_count:
            continue
        coords = np.array(list(cells), dtype=int)
        coords[:, 0] -= int(coords[:, 0].min())
        coords[:, 1] -= int(coords[:, 1].min())
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        mask = _random_transform(mask)
        h, w = mask.shape
        if h < min_h or w < min_w or h > max_h or w > max_w:
            continue
        if len(_components(mask)) != 1:
            continue
        if _hole_count(mask) != 0:
            continue
        return mask
    return None


def _ring_mask(inner_h: int, inner_w: int) -> np.ndarray:
    mask = np.ones((inner_h + 2, inner_w + 2), dtype=bool)
    mask[1 : 1 + inner_h, 1 : 1 + inner_w] = False
    return mask


def _double_hole_masks() -> list[np.ndarray]:
    return [
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ]


def _try_add_arm(mask: np.ndarray, hole_count: int, max_length: int) -> np.ndarray | None:
    padded = np.pad(mask, max_length + 1)
    outside = _outside_background(padded)
    candidates = []
    h, w = padded.shape
    for r, c in np.argwhere(padded):
        for dr, dc in DIRS4:
            rr, cc = int(r + dr), int(c + dc)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if not outside[rr, cc]:
                continue
            run = 0
            nr, nc = rr, cc
            while 0 <= nr < h and 0 <= nc < w and outside[nr, nc] and run < max_length:
                run += 1
                nr += dr
                nc += dc
            if run > 0:
                candidates.append((int(r), int(c), dr, dc, run))
    random.shuffle(candidates)
    for r, c, dr, dc, run in candidates[:24]:
        length = random.randint(1, run)
        grown = padded.copy()
        for step in range(1, length + 1):
            grown[r + dr * step, c + dc * step] = True
        grown = _crop(grown)
        if _hole_count(grown) == hole_count and len(_components(grown)) == 1:
            return grown
    return None


def _make_legend_mask(hole_count: int) -> np.ndarray | None:
    for _ in range(100):
        if hole_count == 0:
            mask = _random_polyomino(
                cell_count=random.randint(5, 8),
                min_h=3,
                min_w=3,
                max_h=4,
                max_w=4,
            )
        elif hole_count == 1:
            inner_h, inner_w = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
            mask = _ring_mask(inner_h, inner_w)
            if random.random() < 0.55:
                grown = _try_add_arm(mask, 1, 1)
                if grown is not None and grown.shape[0] <= 4 and grown.shape[1] <= 4:
                    mask = grown
            mask = _random_transform(mask)
        else:
            mask = _random_transform(random.choice(_double_hole_masks()[:2]))
        if mask is None:
            continue
        mask = _crop(mask)
        h, w = mask.shape
        if 3 <= h <= 4 and 3 <= w <= 4 and _hole_count(mask) == hole_count:
            return mask
    return None


def _make_object_mask(hole_count: int) -> np.ndarray | None:
    for _ in range(120):
        if hole_count == 0:
            mask = _random_polyomino(
                cell_count=random.randint(5, 10),
                min_h=2,
                min_w=2,
                max_h=5,
                max_w=6,
            )
            scale = random.choice([1, 2, 2, 3])
        elif hole_count == 1:
            inner_h, inner_w = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
            mask = _random_transform(_ring_mask(inner_h, inner_w))
            scale = random.choice([1, 1, 2, 2])
        else:
            mask = _random_transform(random.choice(_double_hole_masks()))
            scale = random.choice([1, 2, 2])
        if mask is None:
            continue
        mask = _scale_mask(mask, scale)
        arm_budget = random.randint(0, 2 if hole_count == 0 else 1)
        for _ in range(arm_budget):
            grown = _try_add_arm(mask, hole_count, scale + random.randint(1, 3))
            if grown is not None:
                mask = grown
        mask = _crop(mask)
        h, w = mask.shape
        if _hole_count(mask) != hole_count:
            continue
        if len(_components(mask)) != 1:
            continue
        if h < 2 or w < 2 or h > 11 or w > 11:
            continue
        if int(mask.sum()) < 5:
            continue
        return mask
    return None


def _paint(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = int(color)


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, top: int, left: int, gap: int) -> None:
    h, w = blocked.shape
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        r0 = max(0, rr - gap)
        r1 = min(h, rr + gap + 1)
        c0 = max(0, cc - gap)
        c1 = min(w, cc + gap + 1)
        blocked[r0:r1, c0:c1] = True


def _place_object(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    output_color: int,
) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    rows = list(range(1, gh - h))
    cols = list(range(1, gw - w))
    if not rows or not cols:
        return False
    positions = [(r, c) for r in rows for c in cols]
    random.shuffle(positions)
    for top, left in positions:
        region = blocked[top : top + h, left : left + w]
        if np.any(region & mask):
            continue
        _paint(grid, mask, top, left, GREY)
        _paint(expected, mask, top, left, output_color)
        _mark_blocked(blocked, mask, top, left, gap=1)
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    legend: dict[int, int] = {}
    for comp in _components((g != bg) & (g != GREY)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        holes = _hole_count(mask)
        assert holes not in legend
        legend[holes] = color

    out = g.copy()
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        holes = _hole_count(mask)
        color = int(legend[holes])
        for r, c in comp:
            out[r, c] = color
    return out


def _legend_layout(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray
) -> tuple[dict[int, int], list[np.ndarray]] | None:
    gh, gw = grid.shape
    side = random.choice(["left", "right"])
    strip_w = 7
    gap_w = 2
    if gw <= strip_w + gap_w + 10:
        return None

    if side == "left":
        strip_c0 = 0
        blocked[:, : strip_w + gap_w] = True
    else:
        strip_c0 = gw - strip_w
        blocked[:, gw - strip_w - gap_w :] = True

    row_order = [0, 1, 2]
    random.shuffle(row_order)
    colors = random.sample(LEGEND_COLORS, len(row_order))
    masks = []
    for holes in row_order:
        mask = _make_legend_mask(holes)
        if mask is None:
            return None
        masks.append(mask)

    row_gaps = [random.randint(1, 2) for _ in range(len(masks) - 1)]
    total_h = sum(mask.shape[0] for mask in masks) + sum(row_gaps)
    if total_h > gh - 2:
        return None
    top = random.randint(1, gh - 1 - total_h)

    mapping: dict[int, int] = {}
    for idx, holes in enumerate(row_order):
        mask = masks[idx]
        mh, mw = mask.shape
        left_low = strip_c0 + 1
        left_high = strip_c0 + strip_w - mw - 1
        if left_low > left_high:
            return None
        left = random.randint(left_low, left_high)
        color = colors[idx]
        _paint(grid, mask, top, left, color)
        _paint(expected, mask, top, left, color)
        mapping[holes] = color
        top += mh
        if idx + 1 < len(masks):
            top += row_gaps[idx]

    return mapping, masks


def _grey_hole_sequence(count: int) -> list[int]:
    seq = [0, 1, 2]
    while len(seq) < count:
        seq.append(random.choice([0, 1, 2]))
    random.shuffle(seq)
    return seq


def _generate_attempt() -> np.ndarray | None:
    gh = random.randint(20, 28)
    gw = random.randint(22, 30)
    grid = np.full((gh, gw), BG, dtype=int)
    expected = grid.copy()
    blocked = np.zeros((gh, gw), dtype=bool)

    legend = _legend_layout(grid, expected, blocked)
    if legend is None:
        return None
    mapping, _masks = legend

    obj_count = random.randint(5, 8)
    items = []
    for holes in _grey_hole_sequence(obj_count):
        mask = _make_object_mask(holes)
        if mask is None:
            return None
        items.append((int(mask.shape[0] * mask.shape[1]), holes, mask))
    items.sort(key=lambda item: item[0], reverse=True)

    for _area, holes, mask in items:
        if not _place_object(grid, expected, blocked, mask, mapping[holes]):
            return None

    if int(np.sum(grid == BG)) <= int(np.sum(grid != BG)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
