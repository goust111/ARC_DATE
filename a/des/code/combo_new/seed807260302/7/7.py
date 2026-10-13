import random
from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MOVE_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


GEN_BG = 0
GEN_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_GRID_MIN = 12
GEN_GRID_MAX = 28
GEN_OBJECT_COUNT_MIN = 1
GEN_OBJECT_COUNT_MAX = 4
GEN_COPY_COUNT_MIN = 1
GEN_COPY_COUNT_MAX = 4
GEN_SHAPE_RETRIES = 80
GEN_PLACEMENT_RETRIES = 140
GEN_ATTEMPTS = 400
GEN_DIRECTIONS = ("up", "down", "left", "right")


def get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _normalize_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _row_extents(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    extents: dict[int, tuple[int, int]] = {}
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        if r not in extents:
            extents[r] = (c, c)
        else:
            lo, hi = extents[r]
            extents[r] = (min(lo, c), max(hi, c))
    return extents


def _col_extents(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    extents: dict[int, tuple[int, int]] = {}
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        if c not in extents:
            extents[c] = (r, r)
        else:
            lo, hi = extents[c]
            extents[c] = (min(lo, r), max(hi, r))
    return extents


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                    "row_extents": _row_extents(coords),
                    "col_extents": _col_extents(coords),
                }
            )
    return components


def _line_info(component: dict) -> dict | None:
    coords = component["coords"]
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if rows.size == 1:
        row = int(rows[0])
        span = sorted(int(c) for c in coords[:, 1])
        if span == list(range(span[0], span[-1] + 1)):
            return {"orientation": "H", "row": row, "start": span[0], "end": span[-1], "length": len(span)}

    if cols.size == 1:
        col = int(cols[0])
        span = sorted(int(r) for r in coords[:, 0])
        if span == list(range(span[0], span[-1] + 1)):
            return {"orientation": "V", "col": col, "start": span[0], "end": span[-1], "length": len(span)}

    return None


def _relation_direction(body: dict, line: dict, grid: np.ndarray, bg: int) -> str | None:
    if body["color"] != line["color"]:
        return None

    info = line["line_info"]
    if info is None:
        return None

    h, w = grid.shape
    candidates: list[str] = []

    if info["orientation"] == "H":
        row = int(info["row"])
        if row in body["row_extents"]:
            left_edge, right_edge = body["row_extents"][row]
            gap_right = right_edge + 1
            gap_left = left_edge - 1

            if info["start"] == right_edge + 2 and 0 <= gap_right < w and int(grid[row, gap_right]) == bg:
                candidates.append("right")
            if info["end"] == left_edge - 2 and 0 <= gap_left < w and int(grid[row, gap_left]) == bg:
                candidates.append("left")

    if info["orientation"] == "V":
        col = int(info["col"])
        if col in body["col_extents"]:
            top_edge, bottom_edge = body["col_extents"][col]
            gap_down = bottom_edge + 1
            gap_up = top_edge - 1

            if info["start"] == bottom_edge + 2 and 0 <= gap_down < h and int(grid[gap_down, col]) == bg:
                candidates.append("down")
            if info["end"] == top_edge - 2 and 0 <= gap_up < h and int(grid[gap_up, col]) == bg:
                candidates.append("up")

    if len(candidates) == 1:
        return candidates[0]
    return None


def _pair_components(grid: np.ndarray, bg: int, components: list[dict]) -> list[tuple[int, int, str]]:
    n = len(components)
    assert n > 0 and n % 2 == 0, "expected one guide line per object"

    for component in components:
        component["line_info"] = _line_info(component)

    outgoing: dict[int, list[tuple[int, str]]] = {idx: [] for idx in range(n)}
    incoming: dict[int, list[tuple[int, str]]] = {idx: [] for idx in range(n)}

    for line_idx, line in enumerate(components):
        if line["line_info"] is None:
            continue
        for body_idx, body in enumerate(components):
            if body_idx == line_idx:
                continue
            direction = _relation_direction(body, line, grid, bg)
            if direction is None:
                continue
            outgoing[line_idx].append((body_idx, direction))
            incoming[body_idx].append((line_idx, direction))

    solutions: list[list[tuple[int, int, str]]] = []
    used = [False] * n
    pairs: list[tuple[int, int, str]] = []

    def options_for(idx: int) -> list[tuple[int, int, str]]:
        opts: list[tuple[int, int, str]] = []
        for line_idx, direction in incoming[idx]:
            if not used[line_idx]:
                opts.append((idx, line_idx, direction))
        if components[idx]["line_info"] is not None:
            for body_idx, direction in outgoing[idx]:
                if not used[body_idx]:
                    opts.append((body_idx, idx, direction))
        return opts

    def search() -> None:
        if len(solutions) >= 2:
            return
        if len(pairs) == n // 2:
            if all(used):
                solutions.append(list(pairs))
            return

        pivot = -1
        pivot_options: list[tuple[int, int, str]] | None = None
        for idx in range(n):
            if used[idx]:
                continue
            opts = [opt for opt in options_for(idx) if not used[opt[0]] and not used[opt[1]]]
            if not opts:
                return
            if pivot_options is None or len(opts) < len(pivot_options):
                pivot = idx
                pivot_options = opts

        assert pivot >= 0 and pivot_options is not None
        seen_pairs: set[tuple[int, int]] = set()
        for body_idx, line_idx, direction in pivot_options:
            key = (body_idx, line_idx)
            if key in seen_pairs or used[body_idx] or used[line_idx]:
                continue
            seen_pairs.add(key)
            used[body_idx] = True
            used[line_idx] = True
            pairs.append((body_idx, line_idx, direction))
            search()
            pairs.pop()
            used[body_idx] = False
            used[line_idx] = False

    search()
    assert len(solutions) == 1, "scene must admit one unique object-guide pairing"
    return solutions[0]


def _span_for_direction(component: dict, direction: str) -> int:
    r0, c0, r1, c1 = component["bbox"]
    if direction in ("left", "right"):
        return int(c1 - c0 + 1)
    return int(r1 - r0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each detached guide line to stamp translated copies of its matching object.

    Input structure:
    - The background is black.
    - Foreground pixels form monochrome 4-connected components.
    - Each stamp object has exactly one disconnected same-colored straight guide line, with a single black cell between the line and one side of the object on a shared row or column.

    Rule:
    - Split the foreground into components, identify which components are one-pixel-wide horizontal or vertical guide lines, and pair each such line with the same-colored object that is exactly one black cell away on the matching row or column.
    - The side containing the guide line gives the translation direction, and the guide-line length gives the number of copies to create.
    - Keep the original object, erase the guide line, and place the requested number of translated copies using a step equal to the object's bounding-box width for left/right moves or height for up/down moves.

    Key insight:
    - The detached line is an instruction token rather than part of the picture: its position encodes direction and its pixel count encodes how many tightly packed copies of the object must be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = int(get_bg(g))

    components = _find_components(g, bg)
    pairs = _pair_components(g, bg, components)

    out = g.copy()
    occupied = out != bg

    for _body_idx, line_idx, _direction in pairs:
        line = components[line_idx]
        for rr, cc in line["coords"]:
            r = int(rr)
            c = int(cc)
            out[r, c] = bg
            occupied[r, c] = False

    for body_idx, line_idx, direction in pairs:
        body = components[body_idx]
        line = components[line_idx]
        dr, dc = MOVE_VECTORS[direction]
        step_size = _span_for_direction(body, direction)
        copies = int(line["area"])

        for step in range(1, copies + 1):
            shift_r = dr * step_size * step
            shift_c = dc * step_size * step
            shifted = body["coords"] + np.array([shift_r, shift_c], dtype=int)
            assert bool(np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < h)))
            assert bool(np.all((0 <= shifted[:, 1]) & (shifted[:, 1] < w)))

            for rr, cc in shifted:
                r = int(rr)
                c = int(cc)
                assert not occupied[r, c], "stamped copies must not overlap"
                occupied[r, c] = True
                out[r, c] = int(body["color"])

    return out


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return _normalize_coords([(int(r), int(c)) for r, c in np.argwhere(mask)])


def _connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_polyomino_mask() -> np.ndarray:
    for _ in range(GEN_SHAPE_RETRIES):
        mode = random.choice(("rect", "notch", "poly"))

        if mode == "rect":
            height = random.randint(2, 4)
            width = random.randint(2, 4)
            mask = np.ones((height, width), dtype=bool)
        elif mode == "notch":
            height = random.randint(2, 4)
            width = random.randint(2, 4)
            mask = np.ones((height, width), dtype=bool)
            removable = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
            random.shuffle(removable)
            remove_count = random.randint(1, min(2, len(removable)))
            changed = False
            for rr, cc in removable[:remove_count]:
                trial = mask.copy()
                trial[rr, cc] = False
                if trial.sum() >= 3 and _connected_mask(trial):
                    mask = trial
                    changed = True
            if not changed:
                continue
        else:
            height = random.randint(2, 4)
            width = random.randint(2, 4)
            target = random.randint(3, min(7, height * width))
            start = (random.randrange(height), random.randrange(width))
            cells = {start}
            while len(cells) < target:
                frontier: list[tuple[int, int]] = []
                for r, c in cells:
                    for dr, dc in DIR4:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(random.choice(frontier))
            if len(cells) != target:
                continue
            mask = np.zeros((height, width), dtype=bool)
            for r, c in cells:
                mask[r, c] = True

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        mask = _trim_mask(mask)
        if mask.sum() < 3 or min(mask.shape) < 2 or not _connected_mask(mask):
            continue
        return mask

    raise RuntimeError("failed to build a non-degenerate stamp shape")


def _random_stamp() -> np.ndarray:
    return _mask_to_coords(_random_polyomino_mask())


def _build_local_spec(shape: np.ndarray, direction: str, copies: int, attach_index: int) -> dict:
    body = [(int(r), int(c)) for r, c in shape]
    row_extents = _row_extents(shape)
    col_extents = _col_extents(shape)
    r0, c0, r1, c1 = _bbox(shape)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)

    if direction == "right":
        row = attach_index
        right_edge = row_extents[row][1]
        line = [(row, right_edge + 2 + offset) for offset in range(copies)]
        gap = [(row, right_edge + 1)]
        output = set(body)
        for step in range(1, copies + 1):
            shift = step * width
            output.update((r, c + shift) for r, c in body)
    elif direction == "left":
        row = attach_index
        left_edge = row_extents[row][0]
        line = [(row, left_edge - 2 - offset) for offset in range(copies)]
        gap = [(row, left_edge - 1)]
        output = set(body)
        for step in range(1, copies + 1):
            shift = step * width
            output.update((r, c - shift) for r, c in body)
    elif direction == "down":
        col = attach_index
        bottom_edge = col_extents[col][1]
        line = [(bottom_edge + 2 + offset, col) for offset in range(copies)]
        gap = [(bottom_edge + 1, col)]
        output = set(body)
        for step in range(1, copies + 1):
            shift = step * height
            output.update((r + shift, c) for r, c in body)
    else:
        col = attach_index
        top_edge = col_extents[col][0]
        line = [(top_edge - 2 - offset, col) for offset in range(copies)]
        gap = [(top_edge - 1, col)]
        output = set(body)
        for step in range(1, copies + 1):
            shift = step * height
            output.update((r - shift, c) for r, c in body)

    colored = set(body) | set(line)
    return {
        "body": set(body),
        "line": set(line),
        "gap": set(gap),
        "colored": colored,
        "output": output,
    }


def _anchor_ranges(local_sets: list[set[tuple[int, int]]], height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rows = [r for cells in local_sets for r, _c in cells]
    cols = [c for cells in local_sets for _r, c in cells]
    min_anchor_r = -min(rows)
    max_anchor_r = height - 1 - max(rows)
    min_anchor_c = -min(cols)
    max_anchor_c = width - 1 - max(cols)
    if min_anchor_r > max_anchor_r or min_anchor_c > max_anchor_c:
        return None
    return (min_anchor_r, max_anchor_r), (min_anchor_c, max_anchor_c)


def _translate(cells: set[tuple[int, int]], anchor_r: int, anchor_c: int) -> set[tuple[int, int]]:
    return {(anchor_r + r, anchor_c + c) for r, c in cells}


def _mark_forbidden(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(GEN_OBJECT_COUNT_MIN, GEN_OBJECT_COUNT_MAX)
    size_floor = GEN_GRID_MIN + 2 * (object_count - 1)
    height = random.randint(size_floor, GEN_GRID_MAX)
    width = random.randint(size_floor, GEN_GRID_MAX)
    grid = np.full((height, width), GEN_BG, dtype=int)

    input_forbidden = np.zeros((height, width), dtype=bool)
    output_forbidden = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []
    colors = random.sample(GEN_PALETTE, object_count)

    for color in colors:
        placed_ok = False
        for _ in range(GEN_PLACEMENT_RETRIES):
            shape = _random_stamp()
            direction = random.choice(GEN_DIRECTIONS)
            copies = random.randint(GEN_COPY_COUNT_MIN, GEN_COPY_COUNT_MAX)

            if direction in ("left", "right"):
                attach_index = random.choice(sorted(_row_extents(shape)))
            else:
                attach_index = random.choice(sorted(_col_extents(shape)))

            local = _build_local_spec(shape, direction, copies, attach_index)
            ranges = _anchor_ranges([local["colored"], local["output"], local["gap"]], height, width)
            if ranges is None:
                continue
            (r_lo, r_hi), (c_lo, c_hi) = ranges
            anchor_r = random.randint(r_lo, r_hi)
            anchor_c = random.randint(c_lo, c_hi)

            colored = _translate(local["colored"], anchor_r, anchor_c)
            output = _translate(local["output"], anchor_r, anchor_c)
            gap = _translate(local["gap"], anchor_r, anchor_c)

            if any(input_forbidden[r, c] for r, c in colored):
                continue
            if any(output_forbidden[r, c] for r, c in output):
                continue
            if any(grid[r, c] != GEN_BG for r, c in gap):
                continue

            _paint(grid, colored, color)
            _mark_forbidden(input_forbidden, colored)
            _mark_forbidden(output_forbidden, output)
            placed.append({"color": color, "output": output})
            placed_ok = True
            break

        if not placed_ok:
            return None

    expected = np.full_like(grid, GEN_BG)
    for spec in placed:
        color = int(spec["color"])
        for r, c in spec["output"]:
            if expected[r, c] != GEN_BG:
                return None
            expected[r, c] = color

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamping scene")
