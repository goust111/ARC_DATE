import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIZES = (2, 3, 4)
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
GRID_MIN = 20
GRID_MAX = 27
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _legend_rows(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    rows: list[int] = []
    if w < 2:
        return rows
    for r in range(h):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left in (BG, GREY) or right in (BG, GREY) or left == right:
            continue
        if np.any(grid[r, 2:] != BG):
            continue
        rows.append(r)
    return rows


def _parse_scene(grid: np.ndarray) -> tuple[dict[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    rows = sorted(_legend_rows(g))
    assert 2 <= len(rows) <= 3
    assert all(rows[i + 1] - rows[i] == 2 for i in range(len(rows) - 1))

    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    used_colors: set[int] = set()
    for r in rows:
        src = int(g[r, 0])
        dst = int(g[r, 1])
        assert src not in used_colors and dst not in used_colors
        used_colors.add(src)
        used_colors.add(dst)
        mapping[src] = dst
        legend_cells.add((r, 0))
        legend_cells.add((r, 1))

    grey_cells = {tuple(pt) for pt in np.argwhere(g == GREY)}
    objects: list[dict] = []
    used_greys: set[tuple[int, int]] = set()

    for src, dst in mapping.items():
        comps = _components(g == src)
        big = [coords for coords in comps if len(coords) > 1]
        singles = [coords for coords in comps if len(coords) == 1]
        assert len(big) == 1
        assert len(singles) == 1
        assert tuple(map(int, singles[0][0])) in legend_cells

        coords = big[0]
        r0, c0, r1, c1 = _bbox(coords)
        size = r1 - r0 + 1
        assert size == c1 - c0 + 1
        assert size in SIZES
        assert len(coords) == size * size
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == src)

        corners = {
            (r0 - 1, c0 - 1): (-1, -1),
            (r0 - 1, c1 + 1): (-1, 1),
            (r1 + 1, c0 - 1): (1, -1),
            (r1 + 1, c1 + 1): (1, 1),
        }
        matches = [pos for pos in corners if pos in grey_cells]
        assert len(matches) == 1
        grey = matches[0]
        used_greys.add(grey)

        objects.append(
            {
                "source": int(src),
                "target": int(dst),
                "top": int(r0),
                "left": int(c0),
                "size": int(size),
                "direction": corners[grey],
                "grey": grey,
            }
        )

    assert used_greys == grey_cells
    return mapping, objects


def _paint_clipped_square(out: np.ndarray, top: int, left: int, size: int, color: int) -> bool:
    h, w = out.shape
    if top >= h or left >= w or top + size <= 0 or left + size <= 0:
        return False
    r0 = max(0, top)
    r1 = min(h, top + size)
    c0 = max(0, left)
    c1 = min(w, left + size)
    out[r0:r1, c0:c1] = color
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid square via the bottom-left legend, then copy it repeatedly
    along the diagonal indicated by its grey corner marker.
    """
    g = np.asarray(grid, dtype=int)
    _, objects = _parse_scene(g)
    out = np.zeros_like(g)

    for obj in objects:
        top = int(obj["top"])
        left = int(obj["left"])
        size = int(obj["size"])
        dr, dc = obj["direction"]
        color = int(obj["target"])
        step_r = 2 * size * dr
        step_c = 2 * size * dc

        k = 0
        while True:
            rr = top + k * step_r
            cc = left + k * step_c
            if not _paint_clipped_square(out, rr, cc, size, color):
                break
            k += 1

    return out


# --------------- generator (codex) ---------------

OBJECT_COUNTS = (2, 3)


def _legend_layout(h: int, count: int) -> tuple[list[int], set[tuple[int, int]]]:
    rows = [h - 1 - 2 * i for i in range(count)]
    band_top = min(rows)
    reserved = {(r, c) for r in range(band_top, h) for c in range(2)}
    return rows, reserved


def _path_cells(
    top: int, left: int, size: int, direction: tuple[int, int], h: int, w: int
) -> tuple[set[tuple[int, int]], bool]:
    dr, dc = direction
    step_r = 2 * size * dr
    step_c = 2 * size * dc
    rr = top
    cc = left
    cells: set[tuple[int, int]] = set()
    last_was_full = True

    while rr < h and cc < w and rr + size > 0 and cc + size > 0:
        full = rr >= 0 and cc >= 0 and rr + size <= h and cc + size <= w
        r0 = max(0, rr)
        r1 = min(h, rr + size)
        c0 = max(0, cc)
        c1 = min(w, cc + size)
        for r in range(r0, r1):
            for c in range(c0, c1):
                cells.add((r, c))
        last_was_full = full
        rr += step_r
        cc += step_c

    return cells, not last_was_full


def _candidate_positions(
    reserved: set[tuple[int, int]],
    h: int,
    w: int,
    size: int,
    direction: tuple[int, int],
) -> list[dict]:
    dr, dc = direction
    positions: list[dict] = []
    row_start = 1 if dr < 0 else 0
    row_stop = h - size if dr > 0 else h - size + 1
    col_start = 1 if dc < 0 else 0
    col_stop = w - size if dc > 0 else w - size + 1

    for top in range(row_start, row_stop):
        for left in range(col_start, col_stop):
            grey = (top - 1 if dr < 0 else top + size, left - 1 if dc < 0 else left + size)
            path, has_partial = _path_cells(top, left, size, direction, h, w)
            footprint = set(path)
            footprint.add(grey)
            if footprint.isdisjoint(reserved):
                positions.append(
                    {
                        "top": top,
                        "left": left,
                        "grey": grey,
                        "path": path,
                        "has_partial": has_partial,
                    }
                )
    return positions


def _choose_positions(rng: np.random.Generator, specs: list[dict], h: int, w: int) -> list[dict] | None:
    _, reserved = _legend_layout(h, len(specs))
    ordered = sorted(specs, key=lambda obj: -int(obj["size"]))
    placed: list[dict] = []

    for spec in ordered:
        candidates = _candidate_positions(reserved, h, w, int(spec["size"]), spec["direction"])
        if not candidates:
            return None

        want_partial = bool(rng.integers(0, 2))
        preferred = [cand for cand in candidates if cand["has_partial"] == want_partial]
        pool = preferred if preferred else candidates
        chosen = pool[int(rng.integers(len(pool)))]

        reserved.update(chosen["path"])
        reserved.add(chosen["grey"])
        merged = dict(spec)
        merged.update(chosen)
        placed.append(merged)

    rng.shuffle(placed)
    return placed


def _build_grid(h: int, w: int, objects: list[dict], legend_rows: list[int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    order = list(range(len(objects)))
    for idx, obj_idx in enumerate(order):
        obj = objects[obj_idx]
        row = legend_rows[idx]
        grid[row, 0] = int(obj["source"])
        grid[row, 1] = int(obj["target"])

    for obj in objects:
        top = int(obj["top"])
        left = int(obj["left"])
        size = int(obj["size"])
        src = int(obj["source"])
        grey_r, grey_c = obj["grey"]
        grid[top : top + size, left : left + size] = src
        grid[grey_r, grey_c] = GREY

    return grid


def _gen_attempt(rng: np.random.Generator, count: int) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX))
    w = int(rng.integers(GRID_MIN, GRID_MAX))
    sizes = list(rng.choice(np.array(SIZES), size=count, replace=False))

    chosen_colors = list(rng.choice(np.array(PALETTE), size=2 * count, replace=False))
    rng.shuffle(chosen_colors)
    sources = chosen_colors[:count]
    targets = chosen_colors[count:]

    specs = []
    for idx in range(count):
        specs.append(
            {
                "size": int(sizes[idx]),
                "direction": DIRECTIONS[int(rng.integers(len(DIRECTIONS)))],
                "source": int(sources[idx]),
                "target": int(targets[idx]),
            }
        )

    placed = _choose_positions(rng, specs, h, w)
    if placed is None:
        return None

    legend_rows, _ = _legend_layout(h, count)
    rng.shuffle(legend_rows)
    grid = _build_grid(h, w, placed, legend_rows)

    try:
        mapping, objects = _parse_scene(grid)
        assert len(mapping) == count
        assert len(objects) == count
        assert len({int(obj["size"]) for obj in objects}) == count
        out = main(grid)
        assert out.shape == grid.shape
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    count = int(OBJECT_COUNTS[int(rng.integers(len(OBJECT_COUNTS)))])
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng, count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
