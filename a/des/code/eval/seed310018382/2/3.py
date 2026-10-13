import random
from collections import Counter, deque

import numpy as np


OP_NAMES = ("hflip", "vflip", "rot90", "invert")
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    corners = {
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    }
    if len(corners) == 1:
        return next(iter(corners))
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            out.append(coords)
    return out


def _frame_from_component(coords: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    coord_set = set(coords)
    if coord_set != expected:
        return None
    return {"bbox": (r0, c0, r1, c1), "area": len(coords)}


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for comp in _color_components(grid, color):
            frame = _frame_from_component(comp)
            if frame is None:
                continue
            frame["color"] = int(color)
            frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _divider_row(interior: np.ndarray, bg: int) -> int | None:
    candidates = []
    for r in range(interior.shape[0]):
        row = interior[r]
        vals = set(map(int, row))
        if int(bg) in vals:
            continue
        if len(vals) >= 2:
            candidates.append(r)
    if len(candidates) != 1:
        return None
    return int(candidates[0])


def _apply_rule(canvas: np.ndarray, name: str) -> np.ndarray:
    if name == "hflip":
        return np.fliplr(canvas)
    if name == "vflip":
        return np.flipud(canvas)
    if name == "rot90":
        if canvas.shape[0] != canvas.shape[1]:
            raise ValueError("rot90 requires a square canvas")
        return np.rot90(canvas, -1)
    if name == "invert":
        vals = list(map(int, np.unique(canvas)))
        if len(vals) != 2:
            raise ValueError("invert requires exactly two colors")
        a, b = vals
        out = canvas.copy()
        out[canvas == a] = b
        out[canvas == b] = a
        return out
    raise ValueError(f"unknown rule: {name}")


def _infer_rule(src: np.ndarray, dst: np.ndarray) -> str:
    matches = []
    for name in OP_NAMES:
        try:
            cand = _apply_rule(src, name)
        except ValueError:
            continue
        if cand.shape == dst.shape and np.array_equal(cand, dst):
            matches.append(name)
    if len(matches) != 1:
        raise AssertionError(f"ambiguous or missing rule: {matches}")
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn one transform per frame color from example regions, then apply the
    matching transform to every ordinary region with the same frame color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _find_frames(g, bg)

    rules: dict[int, str] = {}
    example_ids: set[int] = set()

    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        divider = _divider_row(interior, bg)
        if divider is None:
            continue
        top = interior[:divider, :]
        bottom = interior[divider + 1 :, :]
        if top.shape != bottom.shape or top.size == 0:
            raise AssertionError("example halves must be equal and non-empty")
        rule = _infer_rule(top, bottom)
        color = int(frame["color"])
        if color in rules and rules[color] != rule:
            raise AssertionError("conflicting examples for one frame color")
        rules[color] = rule
        example_ids.add(idx)

    out = g.copy()
    for idx, frame in enumerate(frames):
        if idx in example_ids:
            continue
        rule = rules.get(int(frame["color"]))
        if rule is None:
            raise AssertionError("missing example for frame color")
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        transformed = _apply_rule(interior, rule)
        if transformed.shape != interior.shape:
            raise AssertionError("transform changed region shape")
        out[r0 + 1 : r1, c0 + 1 : c1] = transformed
    return out


def _binary_rule(mask: np.ndarray, name: str) -> np.ndarray:
    return _apply_rule(mask.astype(int), name)


def _grow_mask(size: int, area: int) -> np.ndarray:
    start = (random.randrange(size), random.randrange(size))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((size, size), dtype=int)
    for r, c in cells:
        mask[r, c] = 1
    return mask


def _sample_mask(size: int, rule: str, *, require_unique: bool) -> np.ndarray | None:
    min_area = max(4, size)
    max_area = min(size * size - 4, size * size - 1)
    if min_area > max_area:
        return None
    for _ in range(200):
        area = random.randint(min_area, max_area)
        mask = _grow_mask(size, area)
        if mask.sum() in (0, size * size):
            continue
        try:
            transformed = _binary_rule(mask, rule)
        except ValueError:
            continue
        if np.array_equal(mask, transformed):
            continue
        if require_unique:
            matches = []
            for name in OP_NAMES:
                try:
                    cand = _binary_rule(mask, name)
                except ValueError:
                    continue
                if np.array_equal(cand, transformed):
                    matches.append(name)
            if matches != [rule]:
                continue
        return mask
    return None


def _paint_mask(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[mask.astype(bool)] = int(fg)
    return out


def _guide_row(width: int, color_a: int, color_b: int) -> np.ndarray:
    offset = random.randint(0, 1)
    return np.array(
        [color_a if (i + offset) % 2 == 0 else color_b for i in range(width)],
        dtype=int,
    )


def _frame_patch(inner: np.ndarray, frame_color: int, bg: int) -> np.ndarray:
    h, w = inner.shape
    patch = np.full((h + 2, w + 2), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    patch[1:-1, 1:-1] = inner
    return patch


def _split_rows(items: list[dict]) -> list[list[dict]]:
    if len(items) <= 3 or random.random() < 0.45:
        return [items]
    cut = random.randint(1, len(items) - 1)
    return [items[:cut], items[cut:]]


def _row_extent(items: list[dict], gaps: list[int]) -> tuple[int, int]:
    width = sum(int(item["patch"].shape[1]) for item in items) + sum(gaps)
    height = max(int(item["patch"].shape[0]) for item in items)
    return height, width


def _expected_output(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        if item["kind"] != "regular":
            continue
        r0, c0, r1, c1 = item["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        out[r0 + 1 : r1, c0 + 1 : c1] = _apply_rule(inner, item["rule"])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    inner_size = random.randint(4, 7)
    n_rules = random.choice((2, 3))
    rule_names = random.sample(OP_NAMES, n_rules)

    palette = random.sample(list(range(1, 10)), n_rules + 3)
    pattern_color = int(palette[0])
    divider_a = int(palette[1])
    divider_b = int(palette[2])
    frame_colors = [int(x) for x in palette[3:]]

    example_items: list[dict] = []
    regular_items: list[dict] = []

    for frame_color, rule in zip(frame_colors, rule_names):
        top_mask = _sample_mask(inner_size, rule, require_unique=True)
        if top_mask is None:
            return None
        top = _paint_mask(top_mask, bg, pattern_color)
        bottom = _apply_rule(top, rule)
        guide = _guide_row(inner_size, divider_a, divider_b)[None, :]
        example_inner = np.vstack([top, guide, bottom])
        example_items.append(
            {
                "kind": "example",
                "rule": rule,
                "frame_color": frame_color,
                "patch": _frame_patch(example_inner, frame_color, bg),
            }
        )

        copies = random.randint(1, 3)
        for _ in range(copies):
            mask = _sample_mask(inner_size, rule, require_unique=False)
            if mask is None:
                return None
            inner = _paint_mask(mask, bg, pattern_color)
            regular_items.append(
                {
                    "kind": "regular",
                    "rule": rule,
                    "frame_color": frame_color,
                    "patch": _frame_patch(inner, frame_color, bg),
                }
            )

    random.shuffle(example_items)
    random.shuffle(regular_items)
    regular_rows = _split_rows(regular_items)
    row_groups = [example_items] + regular_rows

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    between_rows = [random.randint(1, 3) for _ in range(len(row_groups) - 1)]
    row_gaps = [[random.randint(1, 3) for _ in range(len(row) - 1)] for row in row_groups]

    row_sizes = [_row_extent(row, gaps) for row, gaps in zip(row_groups, row_gaps)]
    canvas_h = top_margin + bottom_margin + sum(h for h, _ in row_sizes) + sum(between_rows)
    canvas_w = left_margin + right_margin + max(w for _, w in row_sizes)
    grid = np.full((canvas_h, canvas_w), int(bg), dtype=int)

    placements: list[dict] = []
    y = top_margin
    for row_idx, (row, gaps) in enumerate(zip(row_groups, row_gaps)):
        x = left_margin
        row_h = max(int(item["patch"].shape[0]) for item in row)
        for idx, item in enumerate(row):
            patch = item["patch"]
            ph, pw = patch.shape
            grid[y : y + ph, x : x + pw] = patch
            placements.append(
                {
                    "kind": item["kind"],
                    "rule": item["rule"],
                    "bbox": (y, x, y + ph - 1, x + pw - 1),
                }
            )
            x += pw
            if idx + 1 < len(row):
                x += gaps[idx]
        y += row_h
        if row_idx + 1 < len(row_groups):
            y += between_rows[row_idx]

    expected = _expected_output(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
