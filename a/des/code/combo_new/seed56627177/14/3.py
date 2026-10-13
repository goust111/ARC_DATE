import numpy as np


DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
WALL_COLOR = 5
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR_TO_VEC.values():
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords})
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _is_interval_expansion(prev: tuple[int, int], cur: tuple[int, int]) -> bool:
    pa, pb = prev
    ca, cb = cur
    if not (ca <= pa <= pb <= cb):
        return False
    return (pa - ca) + (cb - pb) == 1


def _analyze_direction(mask: np.ndarray, direction: str) -> dict | None:
    h, w = mask.shape
    counts: list[int] = []
    intervals: list[tuple[int, int]] = []

    if direction == "right":
        axis = range(w - 1, -1, -1)
        for c in axis:
            rows = np.flatnonzero(mask[:, c])
            if rows.size == 0:
                return None
            if int(rows[-1] - rows[0] + 1) != int(rows.size):
                return None
            intervals.append((int(rows[0]), int(rows[-1])))
            counts.append(int(rows.size))
        tip = (intervals[0][0], w - 1)
    elif direction == "left":
        axis = range(w)
        for c in axis:
            rows = np.flatnonzero(mask[:, c])
            if rows.size == 0:
                return None
            if int(rows[-1] - rows[0] + 1) != int(rows.size):
                return None
            intervals.append((int(rows[0]), int(rows[-1])))
            counts.append(int(rows.size))
        tip = (intervals[0][0], 0)
    elif direction == "down":
        axis = range(h - 1, -1, -1)
        for r in axis:
            cols = np.flatnonzero(mask[r])
            if cols.size == 0:
                return None
            if int(cols[-1] - cols[0] + 1) != int(cols.size):
                return None
            intervals.append((int(cols[0]), int(cols[-1])))
            counts.append(int(cols.size))
        tip = (h - 1, intervals[0][0])
    else:
        axis = range(h)
        for r in axis:
            cols = np.flatnonzero(mask[r])
            if cols.size == 0:
                return None
            if int(cols[-1] - cols[0] + 1) != int(cols.size):
                return None
            intervals.append((int(cols[0]), int(cols[-1])))
            counts.append(int(cols.size))
        tip = (0, intervals[0][0])

    if not counts or counts[0] != 1:
        return None

    width = max(counts)
    if width < 2:
        return None

    prev = intervals[0]
    idx = 1
    expected = 1
    while expected < width:
        if idx >= len(counts) or counts[idx] != expected + 1:
            return None
        cur = intervals[idx]
        if not _is_interval_expansion(prev, cur):
            return None
        prev = cur
        expected += 1
        idx += 1

    if idx >= len(counts):
        return None

    tail_interval = prev
    if counts[idx] != width:
        return None
    while idx < len(counts):
        if counts[idx] != width or intervals[idx] != tail_interval:
            return None
        idx += 1

    return {
        "direction": direction,
        "vec": DIR_TO_VEC[direction],
        "tip_local": tip,
        "width": int(width),
    }


def _detect_arrow(mask: np.ndarray) -> dict | None:
    matches = []
    for direction in ("up", "down", "left", "right"):
        info = _analyze_direction(mask, direction)
        if info is not None:
            matches.append(info)
    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each arrow from its tip by a same-color ray whose length is encoded by the tail width.

    Input structure:
    - The background is black.
    - The foreground contains several monochrome arrow shapes, each formed by a rectangular tail and a shrinking triangular head pointing in one of the four cardinal directions.
    - Additional single-pixel wall obstacles are scattered in the grid and remain fixed.

    Rule:
    - For every monochrome component that matches the arrow geometry, infer its pointing direction from the unique tip side.
    - Read the ray length as the tail width, i.e. the maximum cross-section size perpendicular to the arrow direction.
    - Starting from the cell immediately beyond the tip, draw a one-pixel-wide ray in that direction for that many cells, stopping early if the next cell would leave the grid or hit a wall obstacle.

    Key insight:
    - The arrow encodes two different pieces of information at once: the tip gives the direction, while the constant-width tail gives the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    out = g.copy()

    obstacle_mask = np.zeros_like(g, dtype=bool)
    arrows: list[dict] = []
    for comp in _components_by_color(g, bg):
        coords = comp["coords"]
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[int(r - r0), int(c - c0)] = True
        info = _detect_arrow(mask)
        if info is None:
            for r, c in coords:
                obstacle_mask[int(r), int(c)] = True
            continue
        tip_r, tip_c = info["tip_local"]
        arrows.append(
            {
                "color": int(comp["color"]),
                "tip": (int(r0 + tip_r), int(c0 + tip_c)),
                "vec": info["vec"],
                "width": int(info["width"]),
            }
        )

    h, w = g.shape
    for arrow in arrows:
        color = arrow["color"]
        tip_r, tip_c = arrow["tip"]
        dr, dc = arrow["vec"]
        for step in range(1, int(arrow["width"]) + 1):
            nr = tip_r + dr * step
            nc = tip_c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if obstacle_mask[nr, nc]:
                break
            out[nr, nc] = color

    return out


def _centered_intervals(width: int, bias: int) -> list[tuple[int, int]]:
    top = 0
    bottom = width - 1
    intervals = [(top, bottom)]
    take_top = bool(bias)
    while bottom > top:
        if take_top:
            top += 1
        else:
            bottom -= 1
        intervals.append((top, bottom))
        take_top = not take_top
    return intervals


def _build_arrow(width: int, tail_len: int, direction: str, bias: int) -> tuple[np.ndarray, tuple[int, int]]:
    mask = np.zeros((width, tail_len + width), dtype=bool)
    mask[:, :tail_len] = True
    intervals = _centered_intervals(width, bias)
    for offset, (top, bottom) in enumerate(intervals):
        mask[top : bottom + 1, tail_len + offset] = True
    tip = (intervals[-1][0], tail_len + width - 1)

    if direction == "right":
        return mask, tip
    if direction == "left":
        out = np.fliplr(mask)
        return out, (tip[0], mask.shape[1] - 1 - tip[1])
    if direction == "up":
        out = np.rot90(mask, 1)
        return out, (mask.shape[1] - 1 - tip[1], tip[0])
    out = np.rot90(mask, -1)
    return out, (tip[1], mask.shape[0] - 1 - tip[0])


def _sample_distinct(seq: tuple, n: int, rng: np.random.Generator) -> list:
    values = list(seq)
    rng.shuffle(values)
    if n <= len(values):
        return values[:n]
    extra = rng.choice(seq, size=n - len(values), replace=True).tolist()
    out = values + extra
    rng.shuffle(out)
    return out


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in np.argwhere(mask)]


def _ray_plan(
    tip: tuple[int, int],
    direction: str,
    width: int,
    shape: tuple[int, int],
    mode: str,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None] | None:
    h, w = shape
    tr, tc = tip
    dr, dc = DIR_TO_VEC[direction]
    avail = 0
    nr = tr + dr
    nc = tc + dc
    while 0 <= nr < h and 0 <= nc < w:
        avail += 1
        nr += dr
        nc += dc

    if mode == "free":
        if avail < width:
            return None
        ray = [(int(tr + dr * step), int(tc + dc * step)) for step in range(1, width + 1)]
        return ray, None

    if mode == "edge":
        if not (1 <= avail < width):
            return None
        ray = [(int(tr + dr * step), int(tc + dc * step)) for step in range(1, avail + 1)]
        return ray, None

    max_wall_dist = min(width, avail)
    if max_wall_dist < 2:
        return None
    wall_dist = int(rng.integers(2, max_wall_dist + 1))
    ray = [(int(tr + dr * step), int(tc + dc * step)) for step in range(1, wall_dist)]
    wall = (int(tr + dr * wall_dist), int(tc + dc * wall_dist))
    return ray, wall


def _try_place_arrow(
    grid: np.ndarray,
    reserved: np.ndarray,
    direction: str,
    mode: str,
    color: int,
    rng: np.random.Generator,
) -> tuple[bool, dict | None]:
    h, w = grid.shape
    width = int(rng.integers(2, 6))
    tail_len = int(rng.integers(2, 7))
    bias = int(rng.integers(0, 2))
    mask, tip_local = _build_arrow(width, tail_len, direction, bias)
    ah, aw = mask.shape
    if ah > h or aw > w:
        return False, None

    positions = [(r, c) for r in range(h - ah + 1) for c in range(w - aw + 1)]
    rng.shuffle(positions)
    for top, left in positions[:200]:
        tip = (int(top + tip_local[0]), int(left + tip_local[1]))
        plan = _ray_plan(tip, direction, width, (h, w), mode, rng)
        if plan is None:
            continue
        ray_cells, wall_cell = plan
        existing_clear = _dilate8(reserved)
        arrow_cells = _cells_from_mask(mask, top, left)

        candidate_ok = True
        for r, c in arrow_cells:
            if existing_clear[r, c]:
                candidate_ok = False
                break
        if not candidate_ok:
            continue

        if wall_cell is not None and existing_clear[wall_cell]:
            continue

        for r, c in ray_cells:
            if existing_clear[r, c]:
                candidate_ok = False
                break
        if not candidate_ok:
            continue

        for r, c in arrow_cells:
            grid[r, c] = color
            reserved[r, c] = True
        for r, c in ray_cells:
            reserved[r, c] = True
        if wall_cell is not None:
            grid[wall_cell] = WALL_COLOR
            reserved[wall_cell] = True

        return True, {"width": width, "mode": mode}

    return False, None


def _place_extra_walls(grid: np.ndarray, reserved: np.ndarray, rng: np.random.Generator) -> None:
    h, w = grid.shape
    target = int(rng.integers(1, 7))
    for _ in range(target * 20):
        if target <= 0:
            break
        clear = _dilate8(reserved)
        candidates = np.argwhere(~clear)
        if candidates.size == 0:
            break
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        r = int(r)
        c = int(c)
        grid[r, c] = WALL_COLOR
        reserved[r, c] = True
        target -= 1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    n_arrows = int(rng.integers(2, 6))
    directions = _sample_distinct(("up", "down", "left", "right"), n_arrows, rng)
    modes = _sample_distinct(("free", "wall", "edge"), n_arrows, rng)
    colors = rng.choice(np.array(ARROW_COLORS, dtype=int), size=n_arrows, replace=False).tolist()

    placed = []
    for direction, mode, color in zip(directions, modes, colors):
        ok, info = _try_place_arrow(grid, reserved, direction, mode, int(color), rng)
        if not ok:
            return None
        placed.append(info)

    _place_extra_walls(grid, reserved, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
