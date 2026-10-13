import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("tl", "tr", "bl", "br")


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _corner_bbox(h: int, w: int, side: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, side, side
    if corner == "tr":
        return 0, w - side, side, w
    if corner == "bl":
        return h - side, 0, h, side
    if corner == "br":
        return h - side, w - side, h, w
    raise ValueError(f"unknown corner {corner}")


def _extract_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, dict[int, int]]:
    h, w = grid.shape
    for corner in CORNERS:
        if corner == "tl":
            color = int(grid[0, 0])
            if color == BG:
                continue
            row_run = 0
            while row_run < w and int(grid[0, row_run]) == color:
                row_run += 1
            col_run = 0
            while col_run < h and int(grid[col_run, 0]) == color:
                col_run += 1
        elif corner == "tr":
            color = int(grid[0, w - 1])
            if color == BG:
                continue
            row_run = 0
            while row_run < w and int(grid[0, w - 1 - row_run]) == color:
                row_run += 1
            col_run = 0
            while col_run < h and int(grid[col_run, w - 1]) == color:
                col_run += 1
        elif corner == "bl":
            color = int(grid[h - 1, 0])
            if color == BG:
                continue
            row_run = 0
            while row_run < w and int(grid[h - 1, row_run]) == color:
                row_run += 1
            col_run = 0
            while col_run < h and int(grid[h - 1 - col_run, 0]) == color:
                col_run += 1
        else:
            color = int(grid[h - 1, w - 1])
            if color == BG:
                continue
            row_run = 0
            while row_run < w and int(grid[h - 1, w - 1 - row_run]) == color:
                row_run += 1
            col_run = 0
            while col_run < h and int(grid[h - 1 - col_run, w - 1]) == color:
                col_run += 1

        side = min(row_run, col_run)
        if side < 4:
            continue

        r0, c0, r1, c1 = _corner_bbox(h, w, side, corner)
        box = grid[r0:r1, c0:c1]
        if not (
            np.all(box[0, :] == color)
            and np.all(box[-1, :] == color)
            and np.all(box[:, 0] == color)
            and np.all(box[:, -1] == color)
        ):
            continue

        inner = box[1:-1, 1:-1]
        pts = np.argwhere(inner != BG)
        if pts.shape[0] not in (4, 6):
            continue
        rr0, cc0 = pts.min(axis=0)
        rr1, cc1 = pts.max(axis=0)
        height = int(rr1 - rr0 + 1)
        width = int(cc1 - cc0 + 1)
        if height != 2 or width not in (2, 3):
            continue
        block = inner[rr0 : rr1 + 1, cc0 : cc1 + 1]
        if np.any(block == BG):
            continue
        mask = np.zeros_like(inner, dtype=bool)
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        if np.any((inner != BG) & ~mask):
            continue

        top = [int(v) for v in block[0]]
        bottom = [int(v) for v in block[1]]
        if len(set(top)) != len(top):
            continue
        return (r0, c0, r1, c1), color, dict(zip(top, bottom))

    raise AssertionError("legend not found")


def _snake_sequence(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], start_color: int) -> list[int]:
    h, w = grid.shape
    mask = grid != BG
    r0, c0, r1, c1 = legend_bbox
    mask[r0:r1, c0:c1] = False

    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    assert coords, "snake is missing"
    cell_set = set(coords)

    start = coords[0]
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nxt in _neighbors4(r, c, h, w):
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert len(seen) == len(cell_set), "snake must be a single component"

    endpoints = []
    for cell in cell_set:
        degree = sum((nbr in cell_set) for nbr in _neighbors4(cell[0], cell[1], h, w))
        assert degree in (1, 2), "snake must be branchless"
        if degree == 1:
            endpoints.append(cell)
    assert len(endpoints) == 2, "snake must have exactly two endpoints"

    starts = [cell for cell in endpoints if int(grid[cell]) == start_color]
    assert len(starts) == 1, "legend color must select exactly one endpoint"

    path = []
    visited = set()
    prev = None
    curr = starts[0]
    while True:
        path.append(int(grid[curr]))
        visited.add(curr)
        nxts = [nbr for nbr in _neighbors4(curr[0], curr[1], h, w) if nbr in cell_set and nbr != prev]
        nxts = [nbr for nbr in nxts if nbr not in visited]
        assert len(nxts) <= 1, "snake traversal became ambiguous"
        if not nxts:
            break
        prev, curr = curr, nxts[0]

    assert len(visited) == len(cell_set), "traversal must cover the full snake"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_bbox, border_color, mapping = _extract_legend(g)
    seq = _snake_sequence(g, legend_bbox, border_color)
    remapped = [mapping.get(color, color) for color in seq]
    return np.asarray(remapped, dtype=int).reshape(-1, 1)


def _can_extend(
    cell: tuple[int, int],
    prev: tuple[int, int],
    visited: set[tuple[int, int]],
    allowed: np.ndarray,
) -> bool:
    r, c = cell
    if not allowed[r, c] or cell in visited:
        return False
    touches = 0
    h, w = allowed.shape
    for nbr in _neighbors4(r, c, h, w):
        if nbr in visited:
            if nbr != prev:
                return False
            touches += 1
    return touches == 1


def _valid_extensions(
    curr: tuple[int, int], visited: set[tuple[int, int]], allowed: np.ndarray
) -> list[tuple[int, int]]:
    h, w = allowed.shape
    return [
        nbr
        for nbr in _neighbors4(curr[0], curr[1], h, w)
        if _can_extend(nbr, curr, visited, allowed)
    ]


def _build_induced_path(
    allowed: np.ndarray, length: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    h, w = allowed.shape
    starts = [tuple(map(int, p)) for p in np.argwhere(allowed)]
    if len(starts) < length:
        return None

    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    starts.sort(key=lambda rc: abs(rc[0] - center[0]) + abs(rc[1] - center[1]) + float(rng.random()))

    for start in starts[: min(len(starts), 40)]:
        path = [start]
        visited = {start}

        def dfs(curr: tuple[int, int]) -> bool:
            if len(path) == length:
                return True

            cand = _valid_extensions(curr, visited, allowed)
            if not cand:
                return False

            rng.shuffle(cand)
            scored = []
            for nxt in cand:
                visited.add(nxt)
                path.append(nxt)
                score = len(_valid_extensions(nxt, visited, allowed))
                path.pop()
                visited.remove(nxt)
                scored.append((score, float(rng.random()), nxt))
            scored.sort(key=lambda item: (item[0], item[1]))

            for _score, _noise, nxt in scored:
                visited.add(nxt)
                path.append(nxt)
                if dfs(nxt):
                    return True
                path.pop()
                visited.remove(nxt)
            return False

        if dfs(start):
            return path

    return None


def _random_color_sequence(
    length: int, colors: list[int], rng: np.random.Generator
) -> list[int]:
    seq = [int(rng.choice(colors)) for _ in range(length)]
    ends = rng.choice(colors, size=2, replace=False)
    seq[0] = int(ends[0])
    seq[-1] = int(ends[1])

    needed = [color for color in colors if color not in seq]
    slots = list(range(1, length - 1))
    rng.shuffle(slots)
    for color, idx in zip(needed, slots):
        seq[idx] = color
    return seq


def _draw_legend(
    grid: np.ndarray,
    corner: str,
    border_color: int,
    top_colors: list[int],
    bottom_colors: list[int],
) -> tuple[int, int, int, int]:
    k = len(top_colors)
    side = 4 if k == 2 else 5
    r0, c0, r1, c1 = _corner_bbox(grid.shape[0], grid.shape[1], side, corner)
    grid[r0:r1, c0:c1] = BG
    grid[r0:r1, c0] = border_color
    grid[r0:r1, c1 - 1] = border_color
    grid[r0, c0:c1] = border_color
    grid[r1 - 1, c0:c1] = border_color
    grid[r0 + 1, c0 + 1 : c0 + 1 + k] = np.asarray(top_colors, dtype=int)
    grid[r0 + 2, c0 + 1 : c0 + 1 + k] = np.asarray(bottom_colors, dtype=int)
    return r0, c0, r1, c1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 19))
    w = int(rng.integers(12, 19))

    if bool(rng.integers(0, 2)):
        k = 2
        all_mapped = False
        snake_color_count = int(rng.integers(3, 6))
    else:
        k = 3
        all_mapped = bool(rng.integers(0, 2))
        snake_color_count = 3 if all_mapped else int(rng.integers(4, 6))

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    snake_colors = color_pool[:snake_color_count]
    repl_colors = color_pool[snake_color_count : snake_color_count + k]

    length = int(rng.integers(8, 31))
    corner = str(rng.choice(np.array(CORNERS)))
    grid = np.zeros((h, w), dtype=int)

    legend_side = 4 if k == 2 else 5
    legend_bbox = _corner_bbox(h, w, legend_side, corner)
    pad_r0 = max(0, legend_bbox[0] - 1)
    pad_c0 = max(0, legend_bbox[1] - 1)
    pad_r1 = min(h, legend_bbox[2] + 1)
    pad_c1 = min(w, legend_bbox[3] + 1)

    allowed = np.zeros((h, w), dtype=bool)
    allowed[2 : h - 2, 2 : w - 2] = True
    allowed[pad_r0:pad_r1, pad_c0:pad_c1] = False
    if int(allowed.sum()) < length:
        return None

    path = _build_induced_path(allowed, length, rng)
    if path is None:
        return None

    seq = _random_color_sequence(length, snake_colors, rng)
    if seq[0] == seq[-1]:
        return None

    border_from_head = bool(rng.integers(0, 2))
    border_color = seq[0] if border_from_head else seq[-1]

    if all_mapped:
        top_colors = list(snake_colors)
    else:
        include_border = bool(rng.integers(0, 2))
        selectable = list(snake_colors)
        if include_border and border_color in selectable:
            selectable.remove(border_color)
            chosen = [border_color]
            need = k - 1
        else:
            chosen = []
            need = k
        rng.shuffle(selectable)
        chosen.extend(selectable[:need])
        if len(chosen) != k:
            return None
        top_colors = chosen

    rng.shuffle(repl_colors)
    bottom_colors = repl_colors[:k]
    if len(set(top_colors)) != k or len(set(bottom_colors)) != k:
        return None

    _draw_legend(grid, corner, border_color, top_colors, bottom_colors)
    for cell, color in zip(path, seq):
        grid[cell] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (length, 1):
        return None
    if len(set(seq)) != snake_color_count:
        return None
    if all_mapped != (set(top_colors) == set(snake_colors)):
        return None
    if border_color not in (seq[0], seq[-1]) or seq[0] == seq[-1]:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
