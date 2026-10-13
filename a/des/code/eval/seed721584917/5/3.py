import numpy as np


BG = 0
SEPARATOR = 5
ANCHOR = 2
SPRITE_PALETTE = (1, 3, 4, 6, 7, 8, 9)

DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

LAYOUT_CHOICES = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
)

REGION_SIZE_MIN = 6
REGION_SIZE_MAX = 9
MAX_GENERATION_ATTEMPTS = 400
GROW_ATTEMPTS = 120


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR_TO_DELTA.values():
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return parts


def _separator_spec(grid: np.ndarray, bg: int) -> tuple[int, set[int], set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, set[int], set[int]]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        rows = {r for r in range(h) if np.all(g[r, :] == color)}
        cols = {c for c in range(w) if np.all(g[:, c] == color)}
        if not rows and not cols:
            continue
        coords = np.argwhere(g == color)
        if all(int(r) in rows or int(c) in cols for r, c in coords):
            candidates.append((int(coords.shape[0]), color, rows, cols))

    assert candidates, "expected a separator color with full rows/cols"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, color, rows, cols = candidates[0]
    return int(color), rows, cols


def _regions_from_separators(
    h: int,
    w: int,
    sep_rows: set[int],
    sep_cols: set[int],
) -> list[tuple[int, int, int, int]]:
    row_breaks = [-1] + sorted(sep_rows) + [h]
    col_breaks = [-1] + sorted(sep_cols) + [w]
    regions: list[tuple[int, int, int, int]] = []
    for i in range(len(row_breaks) - 1):
        r0 = row_breaks[i] + 1
        r1 = row_breaks[i + 1] - 1
        if r0 > r1:
            continue
        for j in range(len(col_breaks) - 1):
            c0 = col_breaks[j] + 1
            c1 = col_breaks[j + 1] - 1
            if c0 > c1:
                continue
            regions.append((r0, c0, r1, c1))
    return regions


def _anchor_direction(anchor: tuple[int, int], bounds: tuple[int, int, int, int]) -> str:
    r, c = anchor
    r0, c0, r1, c1 = bounds
    hits = []
    if r == r0:
        hits.append("up")
    if r == r1:
        hits.append("down")
    if c == c0:
        hits.append("left")
    if c == c1:
        hits.append("right")
    assert len(hits) == 1, "anchor must lie on exactly one region edge"
    return hits[0]


def _shift_limit(
    sprite: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    direction: str,
) -> int:
    r0, c0, r1, c1 = bounds
    if direction == "up":
        return min(r - r0 for r, _ in sprite)
    if direction == "down":
        return min(r1 - r for r, _ in sprite)
    if direction == "left":
        return min(c - c0 for _, c in sprite)
    assert direction == "right"
    return min(c1 - c for _, c in sprite)


def _analyze_scene(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    separator, sep_rows, sep_cols = _separator_spec(g, bg)
    regions = _regions_from_separators(g.shape[0], g.shape[1], sep_rows, sep_cols)
    assert regions, "expected at least one region"

    parsed: list[dict[str, object]] = []
    for r0, c0, r1, c1 in regions:
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = (sub != bg) & (sub != separator)
        comps = _components(mask)
        assert len(comps) == 2, "each region must contain exactly one anchor and one sprite"

        global_comps = [[(r0 + lr, c0 + lc) for lr, lc in comp] for comp in comps]
        anchors = [comp for comp in global_comps if len(comp) == 1]
        sprites = [comp for comp in global_comps if 3 <= len(comp) <= 6]
        assert len(anchors) == 1 and len(sprites) == 1

        anchor = anchors[0][0]
        sprite = sprites[0]
        anchor_color = int(g[anchor])
        sprite_colors = {int(g[r, c]) for r, c in sprite}
        assert len(sprite_colors) == 1
        sprite_color = next(iter(sprite_colors))
        assert sprite_color not in {bg, separator, anchor_color}

        direction = _anchor_direction(anchor, (r0, c0, r1, c1))
        parsed.append(
            {
                "bounds": (r0, c0, r1, c1),
                "anchor": anchor,
                "anchor_color": anchor_color,
                "sprite": sprite,
                "sprite_color": sprite_color,
                "direction": direction,
            }
        )

    return bg, separator, parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each regional sprite toward the edge marked by its anchor.

    The separator color forms full horizontal/vertical partition lines. Every
    resulting rectangle contains one singleton anchor on an edge and one
    connected monochrome sprite of size 3-6. The sprite moves toward the
    anchor's edge by its own area, clipped by the region boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg, _separator, regions = _analyze_scene(g)

    out = g.copy()
    for region in regions:
        for r, c in region["sprite"]:
            out[int(r), int(c)] = bg

    for region in regions:
        sprite = [(int(r), int(c)) for r, c in region["sprite"]]
        bounds = tuple(map(int, region["bounds"]))
        direction = str(region["direction"])
        dr, dc = DIR_TO_DELTA[direction]
        shift = min(len(sprite), _shift_limit(sprite, bounds, direction))
        moved = [(r + dr * shift, c + dc * shift) for r, c in sprite]
        anchor = tuple(map(int, region["anchor"]))
        assert anchor not in moved, "anchor and sprite must stay distinct"
        for r, c in moved:
            out[r, c] = int(region["sprite_color"])
        out[anchor] = int(region["anchor_color"])

    return out


def _repeated_shuffle(rng: np.random.Generator, items: tuple[object, ...], count: int) -> list[object]:
    pool: list[object] = []
    while len(pool) < count:
        chunk = list(items)
        rng.shuffle(chunk)
        pool.extend(chunk)
    return pool[:count]


def _grow_connected(
    rng: np.random.Generator,
    allowed: set[tuple[int, int]],
    count: int,
    must_touch: set[tuple[int, int]] | None,
) -> list[tuple[int, int]] | None:
    if len(allowed) < count:
        return None

    starts = list(must_touch if must_touch is not None else allowed)
    if not starts:
        return None

    for _ in range(GROW_ATTEMPTS):
        start = starts[int(rng.integers(len(starts)))]
        cells = {start}
        frontier = set()
        for dr, dc in DIR_TO_DELTA.values():
            nbr = (start[0] + dr, start[1] + dc)
            if nbr in allowed:
                frontier.add(nbr)

        while len(cells) < count and frontier:
            choices = list(frontier)
            cell = choices[int(rng.integers(len(choices)))]
            frontier.remove(cell)
            if cell in cells:
                continue
            cells.add(cell)
            for dr, dc in DIR_TO_DELTA.values():
                nbr = (cell[0] + dr, cell[1] + dc)
                if nbr in allowed and nbr not in cells:
                    frontier.add(nbr)

        if len(cells) == count:
            return sorted(cells)
    return None


def _choose_anchor(
    rng: np.random.Generator,
    h: int,
    w: int,
    direction: str,
    sprite: list[tuple[int, int]],
) -> tuple[int, int] | None:
    rows = {r for r, _ in sprite}
    cols = {c for _, c in sprite}
    if direction == "up":
        options = [(0, c) for c in range(1, w - 1) if c not in cols]
    elif direction == "down":
        options = [(h - 1, c) for c in range(1, w - 1) if c not in cols]
    elif direction == "left":
        options = [(r, 0) for r in range(1, h - 1) if r not in rows]
    else:
        assert direction == "right"
        options = [(r, w - 1) for r in range(1, h - 1) if r not in rows]
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _place_sprite_and_anchor(
    rng: np.random.Generator,
    h: int,
    w: int,
    direction: str,
    clipped: bool,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    along = h if direction in ("up", "down") else w
    size_choices = list(range(3, 7))
    rng.shuffle(size_choices)

    for sprite_size in size_choices:
        if clipped:
            max_distance = min(sprite_size - 1, along - 1)
            if max_distance < 1:
                continue
            distance_choices = list(range(1, max_distance + 1))
            rng.shuffle(distance_choices)
        else:
            if along <= sprite_size:
                continue
            distance_choices = [sprite_size]

        for distance in distance_choices:
            if direction == "up":
                allowed = {(r, c) for r in range(distance, h) for c in range(w)}
                must_touch = {(distance, c) for c in range(w)} if clipped else None
            elif direction == "down":
                edge_row = h - 1 - distance
                allowed = {(r, c) for r in range(edge_row + 1) for c in range(w)}
                must_touch = {(edge_row, c) for c in range(w)} if clipped else None
            elif direction == "left":
                allowed = {(r, c) for r in range(h) for c in range(distance, w)}
                must_touch = {(r, distance) for r in range(h)} if clipped else None
            else:
                assert direction == "right"
                edge_col = w - 1 - distance
                allowed = {(r, c) for r in range(h) for c in range(edge_col + 1)}
                must_touch = {(r, edge_col) for r in range(h)} if clipped else None

            sprite = _grow_connected(rng, allowed, sprite_size, must_touch)
            if sprite is None:
                continue
            anchor = _choose_anchor(rng, h, w, direction, sprite)
            if anchor is None:
                continue

            limit = _shift_limit(sprite, (0, 0, h - 1, w - 1), direction)
            if clipped:
                if not (1 <= limit < sprite_size):
                    continue
            else:
                if limit < sprite_size:
                    continue
            return sprite, anchor

    return None


def _sample_partition_sizes(rng: np.random.Generator, count: int) -> list[int]:
    return [int(rng.integers(REGION_SIZE_MIN, REGION_SIZE_MAX + 1)) for _ in range(count)]


def _region_boxes(row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int, int, int]]:
    row_boxes: list[tuple[int, int]] = []
    cursor = 0
    for i, size in enumerate(row_sizes):
        row_boxes.append((cursor, cursor + size - 1))
        cursor += size
        if i < len(row_sizes) - 1:
            cursor += 1

    col_boxes: list[tuple[int, int]] = []
    cursor = 0
    for i, size in enumerate(col_sizes):
        col_boxes.append((cursor, cursor + size - 1))
        cursor += size
        if i < len(col_sizes) - 1:
            cursor += 1

    boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_boxes:
        for c0, c1 in col_boxes:
            boxes.append((r0, c0, r1, c1))
    return boxes


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    region_rows, region_cols = LAYOUT_CHOICES[int(rng.integers(len(LAYOUT_CHOICES)))]
    row_sizes = _sample_partition_sizes(rng, region_rows)
    col_sizes = _sample_partition_sizes(rng, region_cols)

    h = sum(row_sizes) + region_rows - 1
    w = sum(col_sizes) + region_cols - 1
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)

    row_cursor = row_sizes[0]
    for size in row_sizes[1:]:
        grid[row_cursor, :] = SEPARATOR
        row_cursor += 1 + size

    col_cursor = col_sizes[0]
    for size in col_sizes[1:]:
        grid[:, col_cursor] = SEPARATOR
        col_cursor += 1 + size

    boxes = _region_boxes(row_sizes, col_sizes)
    region_count = len(boxes)
    sprite_colors = list(rng.choice(np.array(SPRITE_PALETTE, dtype=int), size=region_count, replace=False))
    directions = _repeated_shuffle(rng, ("up", "down", "left", "right"), region_count)
    clipped_flags = _repeated_shuffle(rng, (False, True), region_count)

    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        direction = str(directions[idx])
        clipped = bool(clipped_flags[idx])
        local_h = r1 - r0 + 1
        local_w = c1 - c0 + 1
        placed = _place_sprite_and_anchor(rng, local_h, local_w, direction, clipped)
        if placed is None:
            return None
        sprite, anchor = placed

        grid[r0 + anchor[0], c0 + anchor[1]] = ANCHOR
        for lr, lc in sprite:
            grid[r0 + lr, c0 + lc] = int(sprite_colors[idx])

    try:
        _bg, separator, parsed = _analyze_scene(grid)
        assert separator == SEPARATOR
        assert len(parsed) == region_count
        moved = main(grid)
        assert moved.shape == grid.shape
        assert not np.array_equal(moved, grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
