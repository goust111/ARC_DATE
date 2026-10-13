import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
FILL_COLORS = [2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = _neighbors4(0, 0) if connectivity == 4 else _neighbors8(0, 0)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_legend(grid: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    special = {BLACK, BLUE, GREY}
    for r, c in np.argwhere(~np.isin(grid, list(special))):
        rr = int(r)
        cc = int(c)
        count = 0
        scan = cc - 1
        while scan >= 0 and int(grid[rr, scan]) == GREY:
            count += 1
            scan -= 1
        assert count not in mapping, "legend counts must be unique"
        mapping[count] = int(grid[rr, cc])
    assert mapping, "expected a non-empty legend"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey closed border according to the number of blue squares inside it.

    The legend is a vertical stack of colored pixels; each row's grey run length on the
    left encodes the hole count for that color. Closed regions are detected as connected
    black/blue areas that do not touch the outer grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping = _parse_legend(g)
    out = g.copy()

    interior_mask = (g == BLACK) | (g == BLUE)
    for comp in _components(interior_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True

        hole_count = len(_components(comp_mask & (g == BLUE), connectivity=4))
        assert hole_count in mapping, "legend must cover every hole count"

        border_pixels: set[tuple[int, int]] = set()
        for r, c in comp:
            if int(g[r, c]) == BLUE:
                out[r, c] = BLACK
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GREY:
                    border_pixels.add((nr, nc))

        assert border_pixels, "enclosed region must have a grey border"
        fill_color = mapping[hole_count]
        for r, c in border_pixels:
            out[r, c] = fill_color

    return out


def _place_legend(
    grid: np.ndarray,
    rng: np.random.Generator,
    max_count: int,
    colors: list[int],
) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    h, w = grid.shape
    n_rows = max_count + 1
    step = int(rng.choice(np.array([1, 1, 2], dtype=int)))
    total_height = 1 + step * (n_rows - 1)
    corner = str(rng.choice(np.array(["tl", "tr", "bl", "br"])))

    if corner in ("tl", "bl"):
        color_col = int(rng.integers(max_count + 1, max_count + 3))
    else:
        color_col = int(rng.integers(w - 3, w - 1))

    if corner in ("tl", "tr"):
        start_row = int(rng.integers(1, 3))
        rows = [start_row + step * i for i in range(n_rows)]
    else:
        start_row = int(rng.integers(h - total_height - 2, h - total_height))
        rows = [start_row + step * i for i in range(n_rows)]

    counts = list(range(n_rows))
    rng.shuffle(counts)
    mapping = {count: color for count, color in zip(range(n_rows), colors, strict=False)}
    row_to_count = {row: count for row, count in zip(rows, counts, strict=False)}

    c0 = color_col - max_count
    c1 = color_col
    r0 = min(rows)
    r1 = max(rows)
    for row in rows:
        count = row_to_count[row]
        grid[row, color_col] = mapping[count]
        for k in range(count):
            grid[row, color_col - 1 - k] = GREY

    return mapping, (r0, c0, r1, c1)


def _try_place_squares(
    inner_h: int,
    inner_w: int,
    hole_count: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    holes = np.zeros((inner_h, inner_w), dtype=bool)
    for _ in range(120):
        holes.fill(False)
        ok = True
        for _hole_index in range(hole_count):
            size = int(rng.choice(np.array([1, 1, 2], dtype=int)))
            candidates: list[tuple[int, int, int]] = []
            for r in range(1, inner_h - size):
                for c in range(1, inner_w - size):
                    r0 = max(0, r - 1)
                    r1 = min(inner_h, r + size + 1)
                    c0 = max(0, c - 1)
                    c1 = min(inner_w, c + size + 1)
                    if holes[r0:r1, c0:c1].any():
                        continue
                    candidates.append((r, c, size))
            if not candidates:
                ok = False
                break
            r, c, size = candidates[int(rng.integers(len(candidates)))]
            holes[r : r + size, c : c + size] = True
        if ok:
            return holes.copy()
    return None


def _build_object(
    hole_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(120):
        min_h = 3 if hole_count == 0 else max(5, hole_count + 3)
        min_w = 3 if hole_count == 0 else max(5, hole_count + 3)
        inner_h = int(rng.integers(min_h, 10))
        inner_w = int(rng.integers(min_w, 12))
        holes = _try_place_squares(inner_h, inner_w, hole_count, rng)
        if holes is None:
            continue

        patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        patch[border] = GREY
        patch[1:-1, 1:-1][holes] = BLUE
        return patch, border, holes
    return None


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if _can_place(blocked, top, left, ph, pw):
                positions.append((top, left))
    if not positions:
        return None
    top, left = positions[int(rng.integers(len(positions)))]
    region = patch != BLACK
    grid[top : top + ph, left : left + pw][region] = patch[region]
    _mark_blocked(blocked, top, left, ph, pw)
    return top, left


def _manual_expected(
    grid: np.ndarray,
    placed: list[dict[str, object]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for obj in placed:
        top = int(obj["top"])
        left = int(obj["left"])
        border = obj["border"]
        holes = obj["holes"]
        count = int(obj["count"])
        ph, pw = border.shape
        out[top : top + ph, left : left + pw][border] = mapping[count]
        out[top + 1 : top + 1 + holes.shape[0], left + 1 : left + 1 + holes.shape[1]][holes] = BLACK
    return out


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    max_count = int(rng.integers(2, 5))
    if w < max_count + 6 or h < max_count + 8:
        return None

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend_colors = list(rng.choice(np.array(FILL_COLORS, dtype=int), size=max_count + 1, replace=False))
    mapping, legend_box = _place_legend(grid, rng, max_count, legend_colors)
    lr0, lc0, lr1, lc1 = legend_box
    _mark_blocked(blocked, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1, pad=2)

    obj_count = int(rng.integers(2, min(6, max_count + 4)))
    distinct = min(obj_count, max_count + 1, int(rng.integers(2, max_count + 2)))
    counts = list(rng.choice(np.arange(max_count + 1), size=distinct, replace=False))
    while len(counts) < obj_count:
        counts.append(int(rng.integers(0, max_count + 1)))
    rng.shuffle(counts)

    specs: list[dict[str, object]] = []
    for count in counts:
        built = _build_object(int(count), rng)
        if built is None:
            return None
        patch, border, holes = built
        specs.append({"count": int(count), "patch": patch, "border": border, "holes": holes})

    specs.sort(key=lambda obj: -int(np.prod(obj["patch"].shape)))
    placed: list[dict[str, object]] = []
    for obj in specs:
        position = _place_patch(grid, blocked, obj["patch"], rng)
        if position is None:
            return None
        top, left = position
        placed.append(
            {
                "count": obj["count"],
                "top": top,
                "left": left,
                "border": obj["border"],
                "holes": obj["holes"],
            }
        )

    expected = _manual_expected(grid, placed, mapping)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
