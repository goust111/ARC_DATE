import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVE = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_COLORS = tuple(range(10))
MAX_GEN_ATTEMPTS = 500
MAX_PLACEMENT_ATTEMPTS = 150
MAX_LOCAL_ATTEMPTS = 150
MAX_SHIFT = 4
MAX_OBJECT_SPAN = 4
RESERVE_EXTRA = MAX_SHIFT + MAX_OBJECT_SPAN


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _analyze_fence(
    coords: list[tuple[int, int]],
    color: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        rr = r - r0
        cc = c - c0
        mask[rr, cc] = True
        if 0 < rr < h - 1 and 0 < cc < w - 1:
            return None

    sides = {
        "up": mask[0, :],
        "down": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }

    incomplete: list[tuple[str, int, int]] = []
    for name, side in sides.items():
        missing = np.flatnonzero(~side)
        if missing.size == 0:
            continue
        if missing[0] == 0 or missing[-1] == len(side) - 1:
            return None
        if int(missing[-1] - missing[0] + 1) != int(missing.size):
            return None
        incomplete.append((name, int(missing[0]), int(missing[-1])))

    if len(incomplete) != 1:
        return None

    direction, gap0, gap1 = incomplete[0]
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": [(int(r), int(c)) for r, c in coords],
        "direction": direction,
        "gap_span": (int(gap0), int(gap1)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each enclosed fragment toward its frame gap by a distance encoded by same-color anchor pixels.

    Input structure:
    - The background is the dominant color.
    - The grid contains one or more single-color rectangular fences, each drawn as a 1-pixel border with exactly one continuous gap on one side.
    - Inside each fence are one or more monochrome 4-connected fragment objects, and each fragment color may also appear as isolated same-color single-pixel anchors elsewhere in that fence interior.

    Rule:
    - Detect every fence and infer its gap direction from the unique incomplete side of the border.
    - For each fragment, count how many isolated same-color anchor pixels lie inside the same fence; call that count N.
    - Remove all interior fragments and anchors, then translate each fragment N cells in the gap direction while preserving its shape.
    - Keep the fences unchanged, and clip translated fragment pixels that would leave the grid.

    Key insight:
    - The gap only determines direction; the actual travel distance is stored separately as the number of same-color anchor pixels paired with each fragment.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = g.shape
    bg = _bg_color(g)

    fences: list[dict[str, object]] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components_from_mask(g == color):
            info = _analyze_fence(comp, color)
            if info is not None:
                fences.append(info)

    if not fences:
        return g.copy()

    fences.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]

    out = g.copy()
    fence_mask = np.zeros_like(g, dtype=bool)
    clear_cells: set[tuple[int, int]] = set()
    moves: list[tuple[int, list[tuple[int, int]]]] = []

    for fence in fences:
        for r, c in fence["coords"]:  # type: ignore[index]
            fence_mask[r, c] = True

        r0, c0, r1, c1 = fence["bbox"]  # type: ignore[index]
        direction = fence["direction"]  # type: ignore[index]
        fence_color = int(fence["color"])
        interior = g[r0 + 1:r1, c0 + 1:c1]

        for rr, cc in np.argwhere((interior != bg) & (interior != fence_color)):
            clear_cells.add((int(r0 + 1 + rr), int(c0 + 1 + cc)))

        for color in map(int, np.unique(interior)):
            if color in (bg, fence_color):
                continue
            comps = _components_from_mask(interior == color)
            if not comps:
                continue

            sizes = [len(comp) for comp in comps]
            max_size = max(sizes)
            object_ids = [idx for idx, size in enumerate(sizes) if size == max_size]
            if len(object_ids) != 1:
                raise AssertionError("fragment color must have a unique largest component")

            object_idx = object_ids[0]
            anchor_count = 0
            for idx, comp in enumerate(comps):
                if idx == object_idx:
                    continue
                if len(comp) != 1:
                    raise AssertionError("non-fragment same-color components must be single-pixel anchors")
                anchor_count += 1

            dr, dc = MOVE[str(direction)]
            shifted: list[tuple[int, int]] = []
            for rr, cc in comps[object_idx]:
                shifted.append(
                    (
                        int(r0 + 1 + rr + dr * anchor_count),
                        int(c0 + 1 + cc + dc * anchor_count),
                    )
                )
            moves.append((int(color), shifted))

    for r, c in clear_cells:
        out[r, c] = bg

    for color, coords in moves:
        for r, c in coords:
            if 0 <= r < h and 0 <= c < w and not fence_mask[r, c]:
                out[r, c] = int(color)

    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(map(int, rng.integers(0, total + 1, size=parts - 1)))
    values = []
    prev = 0
    for cut in cuts:
        values.append(int(cut - prev))
        prev = cut
    values.append(int(total - prev))
    return values


def _random_connected_mask(
    rng: np.random.Generator,
    max_h: int,
    max_w: int,
    max_cells: int = 6,
) -> np.ndarray | None:
    for _ in range(100):
        h = int(rng.integers(1, max_h + 1))
        w = int(rng.integers(1, max_w + 1))
        if h * w < 2:
            continue
        target = int(rng.integers(2, min(max_cells, h * w) + 1))
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) < 2:
            continue

        arr = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            arr[r, c] = True
        rows, cols = np.where(arr)
        arr = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        return arr
    return None


def _can_place(local: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    h, w = local.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if local[r, c] != bg:
            return False
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and local[nr, nc] != bg:
                return False
    return True


def _segment_layout(length: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    if count < 1 or length < 2 * count - 1:
        return None
    for _ in range(100):
        widths = [int(rng.integers(1, min(3, length) + 1)) for _ in range(count)]
        required = sum(widths) + (count - 1)
        if required > length:
            continue
        extras = _random_composition(length - required, count + 1, rng)
        segments: list[tuple[int, int]] = []
        cursor = extras[0]
        for idx, width in enumerate(widths):
            segments.append((int(cursor), int(width)))
            cursor += width
            if idx < count - 1:
                cursor += 1 + extras[idx + 1]
        return segments
    return None


def _place_anchor(
    local: np.ndarray,
    color: int,
    rng: np.random.Generator,
    bg: int,
) -> bool:
    candidates = []
    h, w = local.shape
    for r in range(h):
        for c in range(w):
            if _can_place(local, [(r, c)], bg):
                candidates.append((int(r), int(c)))
    if not candidates:
        return False
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    local[r, c] = int(color)
    return True


def _reserve_box(
    bbox: tuple[int, int, int, int],
    direction: str,
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    top = 1
    bottom = 1
    left = 1
    right = 1
    if direction == "up":
        top += RESERVE_EXTRA
    elif direction == "down":
        bottom += RESERVE_EXTRA
    elif direction == "left":
        left += RESERVE_EXTRA
    else:
        right += RESERVE_EXTRA
    return (
        max(0, r0 - top),
        max(0, c0 - left),
        min(h - 1, r1 + bottom),
        min(w - 1, c1 + right),
    )


def _populate_fence(
    local: np.ndarray,
    direction: str,
    colors: list[int],
    rng: np.random.Generator,
    bg: int,
) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    ih, iw = local.shape
    vertical = direction in ("up", "down")
    span_len = iw if vertical else ih
    max_objects = min(3, (span_len + 1) // 2)
    if max_objects < 1:
        return None

    for _ in range(MAX_LOCAL_ATTEMPTS):
        local[:, :] = bg
        obj_count = int(rng.integers(1, max_objects + 1))
        if len(colors) < obj_count:
            return None

        segments = _segment_layout(span_len, obj_count, rng)
        if segments is None:
            continue
        chosen = list(map(int, rng.choice(np.array(colors), size=obj_count, replace=False)))
        objects: list[dict[str, object]] = []
        ok = True

        for idx, (start, width) in enumerate(segments):
            anchor_count = int(rng.integers(1, MAX_SHIFT + 1))
            if vertical:
                shape = _random_connected_mask(rng, min(MAX_OBJECT_SPAN, ih), width)
            else:
                shape = _random_connected_mask(rng, width, min(MAX_OBJECT_SPAN, iw))
            if shape is None:
                ok = False
                break
            sh, sw = shape.shape

            if direction == "up":
                max_row = min(ih - sh, anchor_count - 1)
                if max_row < 0:
                    ok = False
                    break
                row = int(rng.integers(0, max_row + 1))
                col = int(start + rng.integers(0, width - sw + 1))
            elif direction == "down":
                min_row = max(0, ih - anchor_count - sh + 1)
                max_row = ih - sh
                if min_row > max_row:
                    ok = False
                    break
                row = int(rng.integers(min_row, max_row + 1))
                col = int(start + rng.integers(0, width - sw + 1))
            elif direction == "left":
                row = int(start + rng.integers(0, width - sh + 1))
                max_col = min(iw - sw, anchor_count - 1)
                if max_col < 0:
                    ok = False
                    break
                col = int(rng.integers(0, max_col + 1))
            else:
                row = int(start + rng.integers(0, width - sh + 1))
                min_col = max(0, iw - anchor_count - sw + 1)
                max_col = iw - sw
                if min_col > max_col:
                    ok = False
                    break
                col = int(rng.integers(min_col, max_col + 1))

            cells = [(int(row + rr), int(col + cc)) for rr, cc in zip(*np.where(shape))]
            if not _can_place(local, cells, bg):
                ok = False
                break
            for rr, cc in cells:
                local[rr, cc] = int(chosen[idx])
            objects.append(
                {
                    "color": int(chosen[idx]),
                    "anchors": int(anchor_count),
                }
            )

        if not ok:
            continue

        for obj in objects:
            for _anchor_idx in range(int(obj["anchors"])):
                if not _place_anchor(local, int(obj["color"]), rng, bg):
                    ok = False
                    break
            if not ok:
                break

        if not ok:
            continue

        span0 = min(start for start, _ in segments)
        span1 = max(start + width - 1 for start, width in segments)
        return (
            [(int(r), int(c), int(local[r, c])) for r, c in zip(*np.where(local != bg))],
            (int(span0), int(span1)),
        )

    return None


def _draw_fence(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    gap_span: tuple[int, int],
    frame_color: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0:r1 + 1, c0] = frame_color
    grid[r0:r1 + 1, c1] = frame_color
    grid[r0, c0:c1 + 1] = frame_color
    grid[r1, c0:c1 + 1] = frame_color

    a, b = gap_span
    if direction == "up":
        grid[r0, c0 + a + 1:c0 + b + 2] = bg
    elif direction == "down":
        grid[r1, c0 + a + 1:c0 + b + 2] = bg
    elif direction == "left":
        grid[r0 + a + 1:r0 + b + 2, c0] = bg
    else:
        grid[r0 + a + 1:r0 + b + 2, c1] = bg


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    fence_target = int(rng.choice(np.array([1, 2, 3]), p=np.array([0.35, 0.4, 0.25])))
    base = 11 + 5 * (fence_target - 1)
    h = int(rng.integers(base, 31))
    w = int(rng.integers(base, 31))

    bg = int(rng.integers(0, 10))
    other_colors = [color for color in ALL_COLORS if color != bg]
    frame_color = int(rng.choice(np.array(other_colors)))
    object_colors = [color for color in other_colors if color != frame_color]

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    directions = list(map(str, rng.permutation(np.array(["up", "down", "left", "right"]))[:fence_target]))

    for direction in directions:
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            outer_h = int(rng.integers(6, min(12, h) + 1))
            outer_w = int(rng.integers(6, min(12, w) + 1))
            max_r0 = h - outer_h
            max_c0 = w - outer_w
            if max_r0 < 0 or max_c0 < 0:
                return None

            clip_bias = bool(rng.integers(0, 2))
            if direction == "up" and clip_bias:
                r0 = int(rng.integers(0, min(3, max_r0) + 1))
            elif direction == "down" and clip_bias:
                low = max(0, max_r0 - 2)
                r0 = int(rng.integers(low, max_r0 + 1))
            else:
                r0 = int(rng.integers(0, max_r0 + 1))

            if direction == "left" and clip_bias:
                c0 = int(rng.integers(0, min(3, max_c0) + 1))
            elif direction == "right" and clip_bias:
                low = max(0, max_c0 - 2)
                c0 = int(rng.integers(low, max_c0 + 1))
            else:
                c0 = int(rng.integers(0, max_c0 + 1))

            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            reserve = _reserve_box((r0, c0, r1, c1), direction, h, w)
            rr0, cc0, rr1, cc1 = reserve
            if reserved[rr0:rr1 + 1, cc0:cc1 + 1].any():
                continue

            interior = np.full((outer_h - 2, outer_w - 2), bg, dtype=int)
            built = _populate_fence(interior, direction, object_colors, rng, bg)
            if built is None:
                continue
            colored_cells, gap_span = built
            if not colored_cells:
                continue

            _draw_fence(grid, (r0, c0, r1, c1), direction, gap_span, frame_color, bg)
            for rr, cc, color in colored_cells:
                grid[r0 + 1 + rr, c0 + 1 + cc] = int(color)
            reserved[rr0:rr1 + 1, cc0:cc1 + 1] = True
            placed = True
            break

        if not placed:
            return None

    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
