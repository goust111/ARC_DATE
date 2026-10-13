import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_IDS = tuple(range(10))
MAX_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr})
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert w >= 4, "grid must leave room for a 2-column legend and object area"
    candidates = []
    for side in ("left", "right"):
        if side == "left":
            legend_cols = slice(0, 2)
            rest = grid[:, 2:]
        else:
            legend_cols = slice(w - 2, w)
            rest = grid[:, : w - 2]

        row_mask = np.all(grid[:, legend_cols] != bg, axis=1)
        if rest.size:
            row_mask &= np.all(rest == bg, axis=1)

        for r0, r1 in _runs(row_mask):
            if r1 - r0 < 2:
                continue
            sub = grid[r0:r1, legend_cols]
            if np.any(sub[:, 0] == sub[:, 1]):
                continue

            mapping: dict[int, int] = {}
            ok = True
            for left_color, right_color in sub.tolist():
                a = int(left_color)
                b = int(right_color)
                if a == bg or b == bg or a == b:
                    ok = False
                    break
                if (a in mapping and mapping[a] != b) or (b in mapping and mapping[b] != a):
                    ok = False
                    break
                mapping[a] = b
                mapping[b] = a
            if not ok or not mapping:
                continue

            mask = np.zeros_like(grid, dtype=bool)
            mask[r0:r1, legend_cols] = True
            candidates.append((r1 - r0, 1 if side == "left" else 0, mapping, mask))

    assert candidates, "legend not found"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_len = candidates[0][0]
    best = [item for item in candidates if item[0] == best_len]
    assert len(best) == 1, "legend must be uniquely identifiable"
    _, _side_rank, mapping, mask = best[0]
    return mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """Translate object colors using a two-column edge legend, then erase the legend."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    mapping, legend_mask = _find_legend(g, bg)

    work = g.copy()
    work[legend_mask] = bg
    out = work.copy()

    for comp in _find_components(work, bg):
        color = int(comp["color"])
        new_color = mapping.get(color, color)
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color
    return out


def _random_mask(rng: np.random.Generator, height: int, width: int, area: int) -> np.ndarray | None:
    total = height * width
    if area > total:
        return None
    start = (int(rng.integers(height)), int(rng.integers(width)))
    cells = {start}
    frontier = {start}
    while len(cells) < area and frontier:
        base = tuple(list(frontier)[int(rng.integers(len(frontier)))])
        frontier.discard(base)
        nbrs = []
        for dr, dc in NEIGHBORS4:
            nr, nc = base[0] + dr, base[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        rng.shuffle(nbrs)
        for nbr in nbrs:
            cells.add(nbr)
            frontier.add(nbr)
            if len(cells) == area:
                break
    if len(cells) != area:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    mask = mask[rows][:, cols]
    if mask.shape[0] == 1 and mask.shape[1] == 1:
        return None
    if np.count_nonzero(mask) < 3:
        return None
    return mask


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if row_hi - row_lo < mh or col_hi - col_lo < mw:
        return False

    row_choices = np.arange(row_lo, row_hi - mh + 1)
    col_choices = np.arange(col_lo, col_hi - mw + 1)
    if row_choices.size == 0 or col_choices.size == 0:
        return False

    order = [(int(r), int(c)) for r in row_choices for c in col_choices]
    rng.shuffle(order)
    for r0, c0 in order:
        pr0 = max(0, r0 - 1)
        pc0 = max(0, c0 - 1)
        pr1 = min(h, r0 + mh + 1)
        pc1 = min(w, c0 + mw + 1)
        if np.any(reserved[pr0:pr1, pc0:pc1]):
            continue
        cells = np.argwhere(mask) + np.array([r0, c0])
        grid[cells[:, 0], cells[:, 1]] = color
        reserved[pr0:pr1, pc0:pc1] = True
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(10, 17))
    width = int(rng.integers(11, 18))
    bg = int(rng.choice(COLOR_IDS))
    grid = np.full((height, width), bg, dtype=int)

    side = "left" if rng.random() < 0.5 else "right"
    legend_pairs = int(rng.integers(2, 4))
    legend_top = int(rng.integers(1, height - legend_pairs - 1))

    usable_colors = [c for c in COLOR_IDS if c != bg]
    rng.shuffle(usable_colors)
    mapped_colors = usable_colors[: 2 * legend_pairs]
    unmapped_colors = usable_colors[2 * legend_pairs :]
    if len(unmapped_colors) < 1:
        return None

    pairs = [(mapped_colors[2 * i], mapped_colors[2 * i + 1]) for i in range(legend_pairs)]
    if side == "left":
        grid[legend_top : legend_top + legend_pairs, 0] = [a for a, _ in pairs]
        grid[legend_top : legend_top + legend_pairs, 1] = [b for _, b in pairs]
        obj_col_lo, obj_col_hi = 3, width - 1
    else:
        grid[legend_top : legend_top + legend_pairs, width - 2] = [a for a, _ in pairs]
        grid[legend_top : legend_top + legend_pairs, width - 1] = [b for _, b in pairs]
        obj_col_lo, obj_col_hi = 1, width - 3

    reserved = np.zeros_like(grid, dtype=bool)
    reserved[legend_top : legend_top + legend_pairs, :] = True

    showcase = pairs[int(rng.integers(len(pairs)))]
    mandatory_colors = [showcase[0], showcase[1], int(rng.choice(unmapped_colors))]
    total_objects = int(rng.integers(4, 7))
    extra_pool = mapped_colors + unmapped_colors
    object_colors = mandatory_colors[:]
    while len(object_colors) < total_objects:
        object_colors.append(int(rng.choice(extra_pool)))
    rng.shuffle(object_colors)

    row_lo, row_hi = 1, height - 1
    for color in object_colors:
        placed = False
        for _ in range(60):
            box_h = int(rng.integers(2, 5))
            box_w = int(rng.integers(2, 5))
            area = int(rng.integers(3, min(box_h * box_w, 6) + 1))
            mask = _random_mask(rng, box_h, box_w, area)
            if mask is None:
                continue
            if _place_object(grid, reserved, mask, color, row_lo, row_hi, obj_col_lo, obj_col_hi, rng):
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    legend_mask = (grid != out) & (out == bg)
    if np.count_nonzero(legend_mask) < 2 * legend_pairs:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_RETRIES} attempts")
