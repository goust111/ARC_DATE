import numpy as np
from collections import Counter, deque


FENCE_COLOR = 9
GUIDE_COLOR = 4
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
GROUP_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _border_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _door_side(door: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = door
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "up"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("door must be a non-corner border gap")


def _find_guide(grid: np.ndarray, door: tuple[int, int], side: str, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    dr, dc = DIRS[side]
    r, c = door
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        val = int(grid[r, c])
        if val == bg:
            r += dr
            c += dc
            continue
        assert val == GUIDE_COLOR, "door ray must hit its guide pixel first"
        return r, c
    raise AssertionError("guide pixel missing")


def _extract_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _components(g == FENCE_COLOR)
    fences: list[dict] = []
    seen_guides: set[tuple[int, int]] = set()
    for coords in comps:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "fence must enclose an interior"

        border = _border_positions(box)
        missing = [(r, c) for (r, c) in border if int(g[r, c]) != FENCE_COLOR]
        assert len(missing) == 1, "each fence must have exactly one door"
        door = missing[0]
        assert int(g[door]) == bg, "door cell must be background"
        side = _door_side(door, box)
        guide = _find_guide(g, door, side, bg)
        assert guide not in seen_guides, "guides must be unique"
        seen_guides.add(guide)

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(x) for x in np.unique(interior) if int(x) != bg]
        assert all(color not in (FENCE_COLOR, GUIDE_COLOR) for color in colors)
        assert len(colors) <= 1, "each fence interior must contain at most one group color"
        group_color = colors[0] if colors else None
        group_coords = []
        if group_color is not None:
            group_coords = [tuple(map(int, p)) for p in np.argwhere(g == group_color) if r0 < int(p[0]) < r1 and c0 < int(p[1]) < c1]
        fences.append(
            {
                "bbox": box,
                "door": door,
                "side": side,
                "guide": guide,
                "group_color": group_color,
                "group_coords": group_coords,
                "group_count": len(group_coords),
            }
        )
    return fences


def _pair_fences(fences: list[dict]) -> list[tuple[int, int]]:
    horizontal_edges: list[tuple[int, int, int]] = []
    vertical_edges: list[tuple[int, int, int]] = []
    for i, left in enumerate(fences):
        lr0, lc0, lr1, lc1 = left["bbox"]
        for j, right in enumerate(fences):
            if i == j:
                continue
            rr0, rc0, rr1, rc1 = right["bbox"]
            if left["side"] == "right" and right["side"] == "left":
                if lc1 < rc0 and _interval_overlap(lr0, lr1, rr0, rr1):
                    horizontal_edges.append((rc0 - lc1, i, j))
            if left["side"] == "down" and right["side"] == "up":
                if lr1 < rr0 and _interval_overlap(lc0, lc1, rc0, rc1):
                    vertical_edges.append((rr0 - lr1, i, j))

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in sorted(horizontal_edges + vertical_edges):
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j))
    assert len(used) == len(fences), "all fences must belong to exactly one facing pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare group sizes inside facing fence pairs, remove the smaller group,
    and redraw that losing group as a straight line beyond its guide pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fences = _extract_fences(g)
    pairs = _pair_fences(fences)

    out = g.copy()
    for i, j in pairs:
        a = fences[i]
        b = fences[j]
        assert a["group_count"] != b["group_count"], "paired group sizes must differ"
        loser = a if a["group_count"] < b["group_count"] else b
        color = loser["group_color"]
        assert color is not None and loser["group_count"] > 0

        for r, c in loser["group_coords"]:
            out[r, c] = bg

        dr, dc = DIRS[loser["side"]]
        gr, gc = loser["guide"]
        for step in range(1, loser["group_count"] + 1):
            rr = gr + dr * step
            cc = gc + dc * step
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            assert int(out[rr, cc]) == bg
            out[rr, cc] = color

    return out


def _draw_fence(grid: np.ndarray, top: int, left: int, height: int, width: int, side: str, offset: int) -> tuple[int, int]:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = FENCE_COLOR
    grid[bottom, left : right + 1] = FENCE_COLOR
    grid[top : bottom + 1, left] = FENCE_COLOR
    grid[top : bottom + 1, right] = FENCE_COLOR
    if side == "up":
        door = (top, left + offset)
    elif side == "down":
        door = (bottom, left + offset)
    elif side == "left":
        door = (top + offset, left)
    else:
        door = (top + offset, right)
    grid[door] = 0
    return door


def _sample_group_cells(rng: np.random.Generator, height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    if rng.random() < 0.5:
        return [tuple(map(int, p)) for p in rng.choice(np.asarray(cells, dtype=int), size=count, replace=False)]

    start = cells[int(rng.integers(len(cells)))]
    chosen = {start}
    frontier = {start}
    while len(chosen) < count:
        candidates = []
        for r, c in frontier:
            for dr, dc in DIRS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    candidates.append((nr, nc))
        if not candidates:
            remaining = [cell for cell in cells if cell not in chosen]
            pick = remaining[int(rng.integers(len(remaining)))]
        else:
            pick = candidates[int(rng.integers(len(candidates)))]
        chosen.add(pick)
        frontier.add(pick)
    return list(chosen)


def _build_horizontal_pair(rng: np.random.Generator, color_a: int, color_b: int) -> np.ndarray:
    h1 = int(rng.integers(4, 8))
    h2 = int(rng.integers(4, 8))
    w1 = int(rng.integers(4, 8))
    w2 = int(rng.integers(4, 8))
    door_row = int(rng.integers(1, min(h1, h2) - 1))

    cap1 = (h1 - 2) * (w1 - 2)
    cap2 = (h2 - 2) * (w2 - 2)
    n1 = int(rng.integers(2, min(7, cap1) + 1))
    n2 = int(rng.integers(2, min(7, cap2) + 1))
    while n2 == n1:
        n2 = int(rng.integers(2, min(7, cap2) + 1))

    gap = max(n1, n2) + int(rng.integers(4, 8))
    patch = np.zeros((max(h1, h2), w1 + gap + w2), dtype=int)

    _draw_fence(patch, 0, 0, h1, w1, "right", door_row)
    right_left = w1 + gap
    _draw_fence(patch, 0, right_left, h2, w2, "left", door_row)

    patch[door_row, w1 + 1] = GUIDE_COLOR
    patch[door_row, right_left - 2] = GUIDE_COLOR

    for rr, cc in _sample_group_cells(rng, h1 - 2, w1 - 2, n1):
        patch[1 + rr, 1 + cc] = color_a
    for rr, cc in _sample_group_cells(rng, h2 - 2, w2 - 2, n2):
        patch[1 + rr, right_left + 1 + cc] = color_b

    return patch


def _build_vertical_pair(rng: np.random.Generator, color_a: int, color_b: int) -> np.ndarray:
    h1 = int(rng.integers(4, 8))
    h2 = int(rng.integers(4, 8))
    w1 = int(rng.integers(4, 8))
    w2 = int(rng.integers(4, 8))
    door_col = int(rng.integers(1, min(w1, w2) - 1))

    cap1 = (h1 - 2) * (w1 - 2)
    cap2 = (h2 - 2) * (w2 - 2)
    n1 = int(rng.integers(2, min(7, cap1) + 1))
    n2 = int(rng.integers(2, min(7, cap2) + 1))
    while n2 == n1:
        n2 = int(rng.integers(2, min(7, cap2) + 1))

    gap = max(n1, n2) + int(rng.integers(4, 8))
    patch = np.zeros((h1 + gap + h2, max(w1, w2)), dtype=int)

    _draw_fence(patch, 0, 0, h1, w1, "down", door_col)
    lower_top = h1 + gap
    _draw_fence(patch, lower_top, 0, h2, w2, "up", door_col)

    patch[h1 + 1, door_col] = GUIDE_COLOR
    patch[lower_top - 2, door_col] = GUIDE_COLOR

    for rr, cc in _sample_group_cells(rng, h1 - 2, w1 - 2, n1):
        patch[1 + rr, 1 + cc] = color_a
    for rr, cc in _sample_group_cells(rng, h2 - 2, w2 - 2, n2):
        patch[lower_top + 1 + rr, 1 + cc] = color_b

    return patch


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = "horizontal" if rng.random() < 0.5 else "vertical"
    pair_count = int(rng.integers(1, 4))
    color_count = 2 * pair_count
    colors = list(map(int, rng.choice(np.asarray(GROUP_COLOR_POOL, dtype=int), size=color_count, replace=False)))

    patches = []
    for idx in range(pair_count):
        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        if orientation == "horizontal":
            patches.append(_build_horizontal_pair(rng, color_a, color_b))
        else:
            patches.append(_build_vertical_pair(rng, color_a, color_b))

    margin_top = int(rng.integers(1, 5))
    margin_bottom = int(rng.integers(1, 5))
    margin_left = int(rng.integers(1, 5))
    margin_right = int(rng.integers(1, 5))

    if orientation == "horizontal":
        band_gaps = [int(rng.integers(2, 6)) for _ in range(pair_count - 1)]
        height = margin_top + sum(p.shape[0] for p in patches) + sum(band_gaps) + margin_bottom
        width = margin_left + max(p.shape[1] for p in patches) + margin_right + int(rng.integers(0, 7))
        grid = np.zeros((height, width), dtype=int)
        row = margin_top
        for idx, patch in enumerate(patches):
            max_col = width - margin_right - patch.shape[1]
            col = int(rng.integers(margin_left, max_col + 1))
            grid[row : row + patch.shape[0], col : col + patch.shape[1]] |= patch
            row += patch.shape[0]
            if idx < len(band_gaps):
                row += band_gaps[idx]
    else:
        band_gaps = [int(rng.integers(2, 6)) for _ in range(pair_count - 1)]
        height = margin_top + max(p.shape[0] for p in patches) + margin_bottom + int(rng.integers(0, 7))
        width = margin_left + sum(p.shape[1] for p in patches) + sum(band_gaps) + margin_right
        grid = np.zeros((height, width), dtype=int)
        col = margin_left
        for idx, patch in enumerate(patches):
            max_row = height - margin_bottom - patch.shape[0]
            row = int(rng.integers(margin_top, max_row + 1))
            grid[row : row + patch.shape[0], col : col + patch.shape[1]] |= patch
            col += patch.shape[1]
            if idx < len(band_gaps):
                col += band_gaps[idx]

    try:
        fences = _extract_fences(grid)
        pairs = _pair_fences(fences)
        if len(fences) != 2 * pair_count or len(pairs) != pair_count:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
