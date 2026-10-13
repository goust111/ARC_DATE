import numpy as np


AZURE = 8
GREY = 5
ALL_COLORS = tuple(range(10))
BG_CHOICES = (0, 1, 2, 3, 4, 6, 7, 9)
TEMPLATE_SPANS = (3, 5)
BOTTOM_H_RANGE = (9, 15)
BOTTOM_W_RANGE = (10, 18)
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray, exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    vals, counts = np.unique(grid, return_counts=True)
    best_color = None
    best_count = -1
    for val, count in zip(vals, counts):
        color = int(val)
        if color in exclude:
            continue
        if int(count) > best_count:
            best_color = color
            best_count = int(count)
    assert best_color is not None
    return best_color


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == AZURE, axis=1))[0]
    assert len(rows) == 1, "expected exactly one azure separator row"
    row = int(rows[0])
    assert 0 < row < grid.shape[0] - 1, "separator must split the grid into two non-empty parts"
    return row


def _extract_template(template_area: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    fg_coords = np.argwhere(template_area != bg)
    assert fg_coords.size > 0, "template area must contain a foreground pattern"
    colors = {int(v) for v in np.unique(template_area[template_area != bg])}
    assert len(colors) == 1, "template area must use exactly one foreground color"
    template_color = next(iter(colors))

    mins = fg_coords.min(axis=0)
    maxs = fg_coords.max(axis=0)
    r0, c0 = (int(x) for x in mins)
    r1, c1 = (int(x) for x in maxs)
    patch = template_area[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == template_color
    assert _is_connected(mask), "template pattern must be 4-connected"

    center = np.array([mask.shape[0] // 2, mask.shape[1] // 2], dtype=int)
    rel = np.argwhere(mask).astype(int) - center
    return template_color, rel


def _find_crosshair_center(instruction_area: np.ndarray) -> tuple[int, int]:
    grey_mask = instruction_area == GREY
    coords = np.argwhere(grey_mask)
    assert coords.shape[0] == 5, "crosshair must use exactly five grey pixels"

    h, w = instruction_area.shape
    centers = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        if 0 < r < h - 1 and 0 < c < w - 1:
            if (
                grey_mask[r - 1, c]
                and grey_mask[r + 1, c]
                and grey_mask[r, c - 1]
                and grey_mask[r, c + 1]
            ):
                centers.append((r, c))
    assert len(centers) == 1, "expected exactly one crosshair center"
    center = centers[0]
    expected = {
        center,
        (center[0] - 1, center[1]),
        (center[0] + 1, center[1]),
        (center[0], center[1] - 1),
        (center[0], center[1] + 1),
    }
    assert expected == set((int(r), int(c)) for r, c in coords), "grey pixels must form a plus"
    return center


def _stamp(out: np.ndarray, rel_cells: np.ndarray, center: tuple[int, int], color: int) -> None:
    h, w = out.shape
    cr, cc = center
    for dr, dc in rel_cells:
        rr = cr + int(dr)
        cc2 = cc + int(dc)
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = color


def _render_output(template_area: np.ndarray, instruction_area: np.ndarray, bg: int) -> np.ndarray:
    template_color, rel_cells = _extract_template(template_area, bg)
    origin = _find_crosshair_center(instruction_area)

    out = np.full(instruction_area.shape, bg, dtype=int)
    _stamp(out, rel_cells, origin, template_color)

    colored = np.argwhere((instruction_area != bg) & (instruction_area != GREY))
    for r, c in colored:
        _stamp(out, rel_cells, (int(r), int(c)), int(instruction_area[r, c]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a template and an instruction map into a layered copy composition.

    The azure row splits the input into a template region above and an
    instruction region below. The grey crosshair gives the output origin. The
    template is stamped once at the origin with its original color, then stamped
    again at every colored instruction pixel with that pixel's color, in row
    scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5 and g.shape[1] >= 7

    sep = _find_separator_row(g)
    bg = _most_common_color(g, exclude={AZURE})
    assert bg not in (AZURE, GREY), "background must differ from special marker colors"

    template_area = g[:sep, :]
    instruction_area = g[sep + 1 :, :]
    return _render_output(template_area, instruction_area, bg)


def _random_path(
    start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        steps = []
        if r < tr:
            steps.append((r + 1, c))
        elif r > tr:
            steps.append((r - 1, c))
        if c < tc:
            steps.append((r, c + 1))
        elif c > tc:
            steps.append((r, c - 1))
        nr, nc = steps[int(rng.integers(len(steps)))]
        r, c = nr, nc
        path.append((r, c))
    return path


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_template_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.choice(TEMPLATE_SPANS))
        w = int(rng.choice(TEMPLATE_SPANS))
        center = (h // 2, w // 2)
        anchors = {
            center,
            (0, int(rng.integers(w))),
            (h - 1, int(rng.integers(w))),
            (int(rng.integers(h)), 0),
            (int(rng.integers(h)), w - 1),
        }

        mask = np.zeros((h, w), dtype=bool)
        for anchor in anchors:
            for r, c in _random_path(anchor, center, rng):
                mask[r, c] = True

        extra_growth = int(rng.integers(0, h + w + 1))
        for _ in range(extra_growth):
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(mask[rr, cc] for rr, cc in _neighbors4(r, c, h, w)):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = frontier[int(rng.integers(len(frontier)))]
            mask[rr, cc] = True

        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if int(mask.sum()) >= h * w:
            continue
        if int(mask.sum()) < max(4, (h + w) // 2):
            continue
        return mask
    raise RuntimeError("failed to generate a valid template mask")


def _rel_cells_from_mask(mask: np.ndarray) -> np.ndarray:
    center = np.array([mask.shape[0] // 2, mask.shape[1] // 2], dtype=int)
    return np.argwhere(mask).astype(int) - center


def _copy_cells(rel_cells: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + int(dr), cc + int(dc)) for dr, dc in rel_cells}


def _copies_overlap(rel_cells: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not _copy_cells(rel_cells, a).isdisjoint(_copy_cells(rel_cells, b))


def _coverage_map(
    shape: tuple[int, int], rel_cells: np.ndarray, centers: list[tuple[int, int]]
) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for center in centers:
        for r, c in _copy_cells(rel_cells, center):
            cover[r, c] += 1
    return cover


def _safe_center_ranges(rel_cells: np.ndarray, shape: tuple[int, int]) -> tuple[range, range]:
    h, w = shape
    rmin = int(rel_cells[:, 0].min())
    rmax = int(rel_cells[:, 0].max())
    cmin = int(rel_cells[:, 1].min())
    cmax = int(rel_cells[:, 1].max())
    rows = range(-rmin, h - rmax)
    cols = range(-cmin, w - cmax)
    return rows, cols


def _choose_origin(
    rel_cells: np.ndarray, shape: tuple[int, int], mode: str, rng: np.random.Generator
) -> tuple[int, int] | None:
    rows, cols = _safe_center_ranges(rel_cells, shape)
    candidates = [
        (r, c)
        for r in rows
        for c in cols
        if 0 < r < shape[0] - 1 and 0 < c < shape[1] - 1
    ]
    if not candidates:
        return None

    order = rng.permutation(len(candidates))
    for idx in order:
        origin = candidates[int(idx)]
        crosshair = {
            origin,
            (origin[0] - 1, origin[1]),
            (origin[0] + 1, origin[1]),
            (origin[0], origin[1] - 1),
            (origin[0], origin[1] + 1),
        }
        usable = [p for p in candidates if p not in crosshair]
        if mode == "overlap":
            if any(_copies_overlap(rel_cells, origin, p) for p in usable):
                return origin
        else:
            return origin
    return None


def _choose_centers(
    rel_cells: np.ndarray,
    shape: tuple[int, int],
    origin: tuple[int, int],
    num_points: int,
    mode: str,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    rows, cols = _safe_center_ranges(rel_cells, shape)
    crosshair = {
        origin,
        (origin[0] - 1, origin[1]),
        (origin[0] + 1, origin[1]),
        (origin[0], origin[1] - 1),
        (origin[0], origin[1] + 1),
    }
    valid = [(r, c) for r in rows for c in cols if (r, c) not in crosshair]
    if len(valid) < num_points:
        return None

    if mode == "overlap":
        overlap_pool = [p for p in valid if _copies_overlap(rel_cells, origin, p)]
        if not overlap_pool:
            return None
        first = overlap_pool[int(rng.integers(len(overlap_pool)))]
        chosen = [first]
        chosen_set = {first}
        shuffled = list(valid)
        perm = rng.permutation(len(shuffled))
        for idx in perm:
            cand = shuffled[int(idx)]
            if cand in chosen_set:
                continue
            chosen.append(cand)
            chosen_set.add(cand)
            if len(chosen) == num_points:
                break
        if len(chosen) != num_points:
            return None
        if _coverage_map(shape, rel_cells, [origin] + chosen).max() < 2:
            return None
        return chosen

    for _ in range(200):
        chosen: list[tuple[int, int]] = []
        shuffled = list(valid)
        perm = rng.permutation(len(shuffled))
        for idx in perm:
            cand = shuffled[int(idx)]
            if any(_copies_overlap(rel_cells, cand, other) for other in [origin] + chosen):
                continue
            if any(abs(cand[0] - other[0]) + abs(cand[1] - other[1]) < 2 for other in chosen):
                continue
            chosen.append(cand)
            if len(chosen) == num_points:
                break
        if len(chosen) == num_points:
            if _coverage_map(shape, rel_cells, [origin] + chosen).max() == 1:
                return chosen
    return None


def _choose_template_position(span: int, item: int, rng: np.random.Generator) -> int:
    if span - item >= 2:
        return int(rng.integers(1, span - item))
    return int(rng.integers(0, span - item + 1))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template_mask = _make_template_mask(rng)
    rel_cells = _rel_cells_from_mask(template_mask)
    template_h, template_w = template_mask.shape

    bottom_h = int(rng.integers(max(BOTTOM_H_RANGE[0], template_h + 4), BOTTOM_H_RANGE[1] + 1))
    bottom_w = int(rng.integers(max(BOTTOM_W_RANGE[0], template_w + 4), BOTTOM_W_RANGE[1] + 1))
    mode = "overlap" if float(rng.random()) < 0.5 else "separate"

    origin = _choose_origin(rel_cells, (bottom_h, bottom_w), mode, rng)
    if origin is None:
        return None

    max_points = 6
    min_points = 2 if mode == "overlap" else 1
    num_points = int(rng.integers(min_points, max_points + 1))
    centers = _choose_centers(rel_cells, (bottom_h, bottom_w), origin, num_points, mode, rng)
    if centers is None:
        return None

    bg = int(rng.choice(BG_CHOICES))
    palette = [c for c in ALL_COLORS if c not in {bg, AZURE, GREY}]
    template_color = int(palette[int(rng.integers(len(palette)))])
    point_palette = [c for c in palette if c != template_color]
    if len(point_palette) < num_points:
        return None
    point_colors = [int(point_palette[i]) for i in rng.permutation(len(point_palette))[:num_points]]

    top_h = int(rng.integers(template_h + 2, template_h + 7))
    grid = np.full((top_h + 1 + bottom_h, bottom_w), bg, dtype=int)

    top_r = _choose_template_position(top_h, template_h, rng)
    top_c = _choose_template_position(bottom_w, template_w, rng)
    template_slice = grid[top_r : top_r + template_h, top_c : top_c + template_w]
    template_slice[template_mask] = template_color

    sep = top_h
    grid[sep, :] = AZURE
    bottom = grid[sep + 1 :, :]

    for rr, cc in {
        origin,
        (origin[0] - 1, origin[1]),
        (origin[0] + 1, origin[1]),
        (origin[0], origin[1] - 1),
        (origin[0], origin[1] + 1),
    }:
        bottom[rr, cc] = GREY

    for (r, c), color in zip(centers, point_colors):
        bottom[r, c] = color

    cover = _coverage_map((bottom_h, bottom_w), rel_cells, [origin] + centers)
    if mode == "overlap" and int(cover.max()) < 2:
        return None
    if mode == "separate" and int(cover.max()) != 1:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (bottom_h, bottom_w):
        return None
    if len(np.unique(out)) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
