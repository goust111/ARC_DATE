import numpy as np


BG = 0
LINE = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GENERATE_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored rectangle along its attached arrow until it rests against the nearest grey partition line.

    Input structure:
    - The background is black, and one grey color forms complete horizontal and vertical partition lines.
    - Each non-grey foreground component is monochromatic and consists of a solid axis-aligned rectangle plus a same-colored 4-cell arrow arm attached to one side.
    - The arrow arm points up, down, left, or right inside one partition cell bounded by grey lines.

    Rule:
    - Keep every grey partition line unchanged.
    - Remove every attached arrow arm and erase each rectangle from its original position.
    - Move each rectangle straight in the arrow direction until its leading edge becomes 4-adjacent to the nearest grey line in that direction.
    - Paint only the translated rectangle at its final location.

    Key insight:
    - The same-colored protruding 4-cell arm identifies both the moving rectangle and the slide direction; the stopping point is the nearest full grey boundary line beyond that rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    line_color, line_rows, line_cols = _find_partition_lines(g, bg)
    components = _find_components(g, excluded={bg, line_color})
    assert components, "expected at least one movable object"

    out = g.copy()
    out[(g != bg) & (g != line_color)] = bg

    for comp in components:
        body_bbox, direction = _decompose_composite(comp["coords"])
        r0, c0, r1, c1 = body_bbox
        if direction == "right":
            target_col = min(col for col in line_cols if col > c1)
            dy, dx = 0, target_col - c1 - 1
        elif direction == "left":
            target_col = max(col for col in line_cols if col < c0)
            dy, dx = 0, target_col - c0 + 1
        elif direction == "down":
            target_row = min(row for row in line_rows if row > r1)
            dy, dx = target_row - r1 - 1, 0
        else:
            assert direction == "up"
            target_row = max(row for row in line_rows if row < r0)
            dy, dx = target_row - r0 + 1, 0
        _paint_rect(out, (r0 + dy, c0 + dx, r1 + dy, c1 + dx), comp["color"])

    return out


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_partition_lines(grid: np.ndarray, bg: int) -> tuple[int, set[int], set[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, set[int], set[int]]] = []
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        rows = {r for r in range(h) if np.all(grid[r, :] == color)}
        cols = {c for c in range(w) if np.all(grid[:, c] == color)}
        if rows or cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one partition-line color"
    return candidates[0]


def _find_components(grid: np.ndarray, excluded: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _decompose_composite(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], str]:
    pts = {(int(r), int(c)) for r, c in coords}
    overall = _bbox(pts)
    matches: list[tuple[tuple[int, int, int, int], str]] = []
    for direction in ("right", "left", "down", "up"):
        body = _try_decompose(pts, overall, direction)
        if body is not None:
            matches.append((body, direction))
    assert len(matches) == 1, "expected exactly one rectangle-plus-arrow decomposition"
    return matches[0]


def _try_decompose(
    pts: set[tuple[int, int]],
    overall: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = overall

    if direction == "right":
        if c1 - c0 + 1 < 6:
            return None
        body = (r0, c0, r1, c1 - 4)
    elif direction == "left":
        if c1 - c0 + 1 < 6:
            return None
        body = (r0, c0 + 4, r1, c1)
    elif direction == "down":
        if r1 - r0 + 1 < 6:
            return None
        body = (r0, c0, r1 - 4, c1)
    else:
        assert direction == "up"
        if r1 - r0 + 1 < 6:
            return None
        body = (r0 + 4, c0, r1, c1)

    br0, bc0, br1, bc1 = body
    if br1 - br0 + 1 < 2 or bc1 - bc0 + 1 < 2:
        return None

    body_cells = {(rr, cc) for rr in range(br0, br1 + 1) for cc in range(bc0, bc1 + 1)}
    if not body_cells <= pts:
        return None

    extra = pts - body_cells
    if len(extra) != 4 or len(pts) != len(body_cells) + 4:
        return None

    if direction == "right":
        rows = {r for r, _ in extra}
        cols = sorted(c for _, c in extra)
        if len(rows) != 1 or cols != list(range(bc1 + 1, bc1 + 5)):
            return None
        row = next(iter(rows))
        if not (br0 <= row <= br1):
            return None
    elif direction == "left":
        rows = {r for r, _ in extra}
        cols = sorted(c for _, c in extra)
        if len(rows) != 1 or cols != list(range(bc0 - 4, bc0)):
            return None
        row = next(iter(rows))
        if not (br0 <= row <= br1):
            return None
    elif direction == "down":
        cols = {c for _, c in extra}
        rows = sorted(r for r, _ in extra)
        if len(cols) != 1 or rows != list(range(br1 + 1, br1 + 5)):
            return None
        col = next(iter(cols))
        if not (bc0 <= col <= bc1):
            return None
    else:
        cols = {c for _, c in extra}
        rows = sorted(r for r, _ in extra)
        if len(cols) != 1 or rows != list(range(br0 - 4, br0)):
            return None
        col = next(iter(cols))
        if not (bc0 <= col <= bc1):
            return None

    return body


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_count = int(rng.integers(2, 5))
    col_count = int(rng.integers(2, 5))
    row_sizes = _sample_zone_sizes(rng, row_count)
    col_sizes = _sample_zone_sizes(rng, col_count)
    if row_sizes is None or col_sizes is None:
        return None

    row_lines, row_spans = _partition_geometry(row_sizes)
    col_lines, col_spans = _partition_geometry(col_sizes)
    h = row_lines[-1] + 1
    w = col_lines[-1] + 1
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[row_lines, :] = LINE
    grid[:, col_lines] = LINE

    zones = [(r0, c0, r1, c1) for r0, r1 in row_spans for c0, c1 in col_spans]
    max_objects = min(5, len(zones))
    num_objects = int(rng.integers(1, max_objects + 1))
    zone_indices = rng.choice(len(zones), size=num_objects, replace=False)
    directions = _sample_directions(rng, num_objects)
    colors = rng.choice(PALETTE, size=num_objects, replace=True)

    specs: list[dict[str, object]] = []
    for idx, direction, color in zip(zone_indices, directions, colors):
        spec = _place_object(grid, zones[int(idx)], direction, int(color), rng)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_output(grid, specs)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def _sample_zone_sizes(rng: np.random.Generator, count: int) -> list[int] | None:
    min_size = 6
    max_size = 11
    max_total = 30 - (count + 1)
    low_total = count * min_size
    high_total = min(count * max_size, max_total)
    if low_total > high_total:
        return None

    total = int(rng.integers(low_total, high_total + 1))
    sizes = [min_size] * count
    capacities = [max_size - min_size] * count
    remaining = total - low_total
    while remaining > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        i = int(rng.choice(choices))
        sizes[i] += 1
        capacities[i] -= 1
        remaining -= 1
    rng.shuffle(sizes)
    return sizes


def _partition_geometry(sizes: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    lines = [0]
    spans: list[tuple[int, int]] = []
    line_pos = 0
    for size in sizes:
        start = line_pos + 1
        end = start + size - 1
        spans.append((start, end))
        line_pos = end + 1
        lines.append(line_pos)
    return lines, spans


def _sample_directions(rng: np.random.Generator, count: int) -> list[str]:
    directions: list[str] = []
    base = ["up", "down", "left", "right"]
    while len(directions) < count:
        chunk = base.copy()
        rng.shuffle(chunk)
        directions.extend(chunk)
    return directions[:count]


def _place_object(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    direction: str,
    color: int,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1

    if direction in {"left", "right"}:
        max_rw = zw - 4
        if max_rw < 2:
            return None
        rh = int(rng.integers(2, min(zh, 6) + 1))
        rw = int(rng.integers(2, min(max_rw, 6) + 1))
        rr = int(rng.integers(0, zh - rh + 1))
        if direction == "right":
            cc = int(rng.integers(0, zw - rw - 4 + 1))
        else:
            cc = int(rng.integers(4, zw - rw + 1))
        arrow_row = rr + int(rng.integers(0, rh))
        body = (zr0 + rr, zc0 + cc, zr0 + rr + rh - 1, zc0 + cc + rw - 1)
        if direction == "right":
            arrow = [(arrow_row + zr0, zc0 + cc + rw + offset) for offset in range(4)]
        else:
            arrow = [(arrow_row + zr0, zc0 + cc - 4 + offset) for offset in range(4)]
    else:
        max_rh = zh - 4
        if max_rh < 2:
            return None
        rh = int(rng.integers(2, min(max_rh, 6) + 1))
        rw = int(rng.integers(2, min(zw, 6) + 1))
        cc = int(rng.integers(0, zw - rw + 1))
        if direction == "down":
            rr = int(rng.integers(0, zh - rh - 4 + 1))
        else:
            rr = int(rng.integers(4, zh - rh + 1))
        arrow_col = cc + int(rng.integers(0, rw))
        body = (zr0 + rr, zc0 + cc, zr0 + rr + rh - 1, zc0 + cc + rw - 1)
        if direction == "down":
            arrow = [(zr0 + rr + rh + offset, zc0 + arrow_col) for offset in range(4)]
        else:
            arrow = [(zr0 + rr - 4 + offset, zc0 + arrow_col) for offset in range(4)]

    r0, c0, r1, c1 = body
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    cells.update(arrow)
    if any(int(grid[r, c]) != BG for r, c in cells):
        return None

    for r, c in cells:
        grid[r, c] = color

    return {"zone": zone, "body": body, "direction": direction, "color": color}


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    out[(out != BG) & (out != LINE)] = BG

    for spec in specs:
        zr0, zc0, zr1, zc1 = spec["zone"]
        r0, c0, r1, c1 = spec["body"]
        direction = spec["direction"]
        color = spec["color"]

        if direction == "right":
            dy, dx = 0, zc1 - c1
        elif direction == "left":
            dy, dx = 0, zc0 - c0
        elif direction == "down":
            dy, dx = zr1 - r1, 0
        else:
            assert direction == "up"
            dy, dx = zr0 - r0, 0

        _paint_rect(out, (r0 + dy, c0 + dx, r1 + dy, c1 + dx), int(color))

    return out
