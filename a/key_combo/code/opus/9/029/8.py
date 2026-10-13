import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEP = 5
MAGENTA = 6
CYAN = 8

RIGHT_COLORS = (BLUE, GREEN, MAGENTA)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
ENCODE_POSITIONS = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
MAX_GENERATION_ATTEMPTS = 700


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(sorted(cells), dtype=int)
    return _normalize_coords(arr)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int = 1) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _connected_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _encode_count(n: int) -> np.ndarray:
    out = np.zeros((3, 3), dtype=int)
    for r, c in ENCODE_POSITIONS[:n]:
        out[r, c] = CYAN
    return out


def _rotate90(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _w = _shape_dims(shape)
    return _normalize_shape({(c, h - 1 - r) for r, c in shape})


def _flip_lr(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _h, w = _shape_dims(shape)
    return _normalize_shape({(r, w - 1 - c) for r, c in shape})


def _mirror_pairs(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, _w = _shape_dims(shape)
    cell_set = set(shape)
    pairs = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in shape:
        partner = (h - 1 - r, c)
        if partner == (r, c):
            continue
        assert partner in cell_set, "template must be horizontally symmetric"
        pair = tuple(sorted(((r, c), partner)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _random_connected_shape(
    area: int,
    *,
    max_h: int = 6,
    max_w: int = 6,
    exact_dims: tuple[int, int] | None = None,
    reject: set[tuple[tuple[int, int], ...]] | None = None,
) -> tuple[tuple[int, int], ...] | None:
    reject = set() if reject is None else set(reject)
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    cand = (r + dr, c + dc)
                    if cand not in cells:
                        frontier.add(cand)
            cells.add(random.choice(tuple(frontier)))

        shape = _normalize_shape(cells)
        h, w = _shape_dims(shape)
        if h > max_h or w > max_w:
            continue
        if exact_dims is not None and (h, w) != exact_dims:
            continue
        if shape in reject:
            continue
        return shape
    return None


def _random_symmetric_template() -> tuple[tuple[int, int], ...]:
    for _ in range(500):
        height = random.randint(3, 6)
        top_rows = (height + 1) // 2
        left = 0
        right = random.randint(1, 3)
        intervals = [(left, right)]

        ok = True
        for _ in range(1, top_rows):
            candidates = []
            for new_left in range(left - 1, left + 2):
                for new_right in range(right - 1, right + 2):
                    width = new_right - new_left + 1
                    overlap = min(right, new_right) - max(left, new_left) + 1
                    if width < 1 or width > 5 or overlap < 1:
                        continue
                    candidates.append((new_left, new_right))
            if not candidates:
                ok = False
                break
            left, right = random.choice(candidates)
            intervals.append((left, right))
        if not ok:
            continue

        min_left = min(l for l, _r in intervals)
        intervals = [(l - min_left, r - min_left) for l, r in intervals]
        mirrored = intervals[:-1][::-1] if height % 2 else intervals[::-1]
        rows = intervals + mirrored

        cells = set()
        for r, (c0, c1) in enumerate(rows):
            for c in range(c0, c1 + 1):
                cells.add((r, c))

        shape = _normalize_shape(cells)
        h, w = _shape_dims(shape)
        area = len(shape)
        pair_count = len(_mirror_pairs(shape))
        if not (6 <= area <= 10):
            continue
        if h * w == area:
            continue
        if pair_count < 2:
            continue
        if len({tuple(row) for row in rows}) < 2:
            continue
        if _rotate90(shape) == shape:
            continue
        return shape
    raise RuntimeError("failed to build a symmetric template")


def _build_mismatch_shapes(
    template: tuple[tuple[int, int], ...],
    count: int,
) -> list[tuple[tuple[int, int], ...]]:
    assert count >= 1
    mismatches: list[tuple[tuple[int, int], ...]] = []
    template_area = len(template)
    template_dims = _shape_dims(template)

    same_bbox = _random_connected_shape(
        template_area,
        max_h=6,
        max_w=6,
        exact_dims=template_dims,
        reject={template},
    )
    if same_bbox is None:
        same_bbox = _random_connected_shape(template_area, max_h=6, max_w=6, reject={template})
    assert same_bbox is not None
    mismatches.append(same_bbox)

    for special in (_rotate90(template), _flip_lr(template)):
        if special != template and special not in mismatches:
            mismatches.append(special)
        if len(mismatches) >= count:
            return mismatches[:count]

    reject = {template, *mismatches}
    while len(mismatches) < count:
        if random.random() < 0.6:
            area = template_area
        else:
            area = random.randint(4, 10)
        shape = _random_connected_shape(area, max_h=6, max_w=6, reject=reject)
        if shape is None:
            continue
        reject.add(shape)
        mismatches.append(shape)
    return mismatches


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    for pr, pc in np.argwhere(patch):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        for dr, dc in [(0, 0), *NEIGHBORS8]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _place_right_components(
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    patches = [_shape_to_patch(shape, color) for shape, color in zip(shapes, colors)]
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)

    for _ in range(120):
        height = random.randint(max(14, max_h + 4), 22)
        width = random.randint(max(8, max_w + 4), 12)
        region = np.zeros((height, width), dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        placements: list[dict] = []
        order = list(range(len(patches)))
        order.sort(key=lambda i: int(np.count_nonzero(patches[i])), reverse=True)

        success = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            row_lo = 1 if height - ph >= 2 else 0
            row_hi = height - ph - 1 if height - ph >= 2 else height - ph
            col_lo = 1 if width - pw >= 2 else 0
            col_hi = width - pw - 1 if width - pw >= 2 else width - pw
            origins = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
            random.shuffle(origins)
            placed = False
            for r0, c0 in origins:
                if not _can_place_patch(occupied, patch, r0, c0):
                    continue
                for pr, pc in np.argwhere(patch):
                    rr = int(r0 + pr)
                    cc = int(c0 + pc)
                    region[rr, cc] = int(patch[pr, pc])
                    occupied[rr, cc] = True
                placements.append({"shape": shapes[idx], "color": colors[idx]})
                placed = True
                break
            if not placed:
                success = False
                break

        if success and 4 <= len(_connected_components(region != BG, connectivity=4)) <= 6:
            return region, placements
    return None


def _count_matches_in_region(region: np.ndarray, target_shape: tuple[tuple[int, int], ...]) -> int:
    comps = _connected_components(region != BG, connectivity=4)
    return sum(_normalize_coords(comp) == target_shape for comp in comps)


def _build_left_region(
    template: tuple[tuple[int, int], ...],
    height: int,
) -> np.ndarray | None:
    th, tw = _shape_dims(template)
    left_w = random.randint(max(6, tw + 2), max(8, tw + 3))
    if left_w > 10 or height < th + 2:
        return None

    region = np.zeros((height, left_w), dtype=int)
    top = random.randint(1, height - th - 1)
    left = random.randint(1, left_w - tw - 1)

    for r, c in template:
        region[top + r, left + c] = RED

    pairs = _mirror_pairs(template)
    marker_count = random.randint(2, min(4, len(pairs)))
    chosen_pairs = random.sample(pairs, marker_count)
    for a, b in chosen_pairs:
        yellow_cell, red_cell = random.choice(((a, b), (b, a)))
        yr, yc = yellow_cell
        rr, rc = red_cell
        region[top + yr, left + yc] = YELLOW
        region[top + rr, left + rc] = RED

    return region


def _semantic_check(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) == 1
    sep = sep_cols[0]
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    left_union = np.isin(left, [RED, YELLOW])
    left_comps = _connected_components(left_union, connectivity=4)
    assert len(left_comps) == 1
    template = _normalize_coords(left_comps[0])
    area = len(template)
    assert 6 <= area <= 10

    bbox = _bbox(left_comps[0])
    r0, c0, r1, c1 = bbox
    left_set = {tuple(map(int, cell)) for cell in left_comps[0]}
    for r, c in left_set:
        assert (r0 + r1 - r, c) in left_set

    yellow_coords = np.argwhere(left == YELLOW)
    assert 2 <= len(yellow_coords) <= 4
    for r, c in yellow_coords:
        mr = r0 + r1 - int(r)
        assert 0 <= mr < left.shape[0]
        assert int(left[mr, int(c)]) == RED

    right_comps = _connected_components(right != BG, connectivity=4)
    assert 4 <= len(right_comps) <= 6
    colors = []
    match_colors = set()
    mismatch_colors = set()
    for comp in right_comps:
        vals = {int(right[r, c]) for r, c in comp}
        assert len(vals) == 1
        color = vals.pop()
        assert color in RIGHT_COLORS
        colors.append(color)
        if _normalize_coords(comp) == template:
            match_colors.add(color)
        else:
            mismatch_colors.add(color)

    match_count = sum(_normalize_coords(comp) == template for comp in right_comps)
    assert 1 <= match_count <= 5

    visible_red = np.argwhere(left == RED)
    visible_shape = _normalize_coords(visible_red)
    assert visible_shape != template
    assert sum(_normalize_coords(comp) == visible_shape for comp in right_comps) != match_count

    template_area = len(template)
    assert any(len(_normalize_coords(comp)) == template_area and _normalize_coords(comp) != template for comp in right_comps)
    assert len(set(colors)) < len(colors)
    assert match_colors & mismatch_colors

    expected = _encode_count(match_count)
    assert np.array_equal(main(g), expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the horizontally symmetric left template, count matching right-side shapes, and encode that count in a fixed 3x3 pattern.

    Input structure:
    - A black background grid is split by exactly one full-height grey separator column.
    - Left of the separator is one 4-connected red figure with 2-4 yellow replacement cells; the full figure is horizontally symmetric about its own center line.
    - Right of the separator are 4-6 detached monochrome 4-connected components in blue, green, or magenta.

    Rule:
    - Treat yellow cells on the left as missing red pixels and restore them by horizontal mirror symmetry within the figure.
    - Use the completed left figure's translation-normalized shape as the template.
    - Translation-normalize every right-side connected component and count how many shapes exactly equal the template.
    - Return a 3x3 black grid and fill the first n cells of the fixed order (0,0), (0,2), (1,1), (2,0), (2,2) with cyan, where n is the match count.

    Key insight:
    - The left object is an encrypted template: its yellow holes must be interpreted through symmetry before the restored shape can be used to select right-side matches and drive the final count encoding.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) == 1, "expected one full-height grey separator column"
    sep = sep_cols[0]

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0

    left_union = np.isin(left, [RED, YELLOW])
    left_comps = _connected_components(left_union, connectivity=4)
    assert len(left_comps) == 1, "expected exactly one left-side marked figure"
    left_coords = left_comps[0]
    r0, c0, r1, c1 = _bbox(left_coords)
    left_set = {tuple(map(int, cell)) for cell in left_coords}

    for r, c in left_set:
        assert (r0 + r1 - r, c) in left_set, "left figure must be horizontally symmetric after restoration"

    for r, c in np.argwhere(left == YELLOW):
        mr = r0 + r1 - int(r)
        assert int(left[mr, int(c)]) == RED, "each yellow cell must mirror a red cell"

    template = _normalize_coords(left_coords)

    right_comps = _connected_components(right != BG, connectivity=4)
    assert 4 <= len(right_comps) <= 6, "expected 4-6 right-side components"
    match_count = 0
    for comp in right_comps:
        colors = {int(right[r, c]) for r, c in comp}
        assert len(colors) == 1, "right-side components must be monochrome"
        if _normalize_coords(comp) == template:
            match_count += 1

    assert 1 <= match_count <= 5
    return _encode_count(match_count)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        template = _random_symmetric_template()
        match_count = random.randint(1, 5)
        total_components = random.randint(max(4, match_count + 1), 6)
        mismatch_count = total_components - match_count

        mismatch_shapes = _build_mismatch_shapes(template, mismatch_count)
        shapes = [template] * match_count + mismatch_shapes
        random.shuffle(shapes)

        colors = [random.choice(RIGHT_COLORS) for _ in shapes]
        match_indices = [i for i, shape in enumerate(shapes) if shape == template]
        mismatch_indices = [i for i, shape in enumerate(shapes) if shape != template]
        if mismatch_indices:
            colors[mismatch_indices[0]] = colors[match_indices[0]]

        placed = _place_right_components(shapes, colors)
        if placed is None:
            continue
        right_region, _placements = placed
        if _count_matches_in_region(right_region, template) != match_count:
            continue

        left_region = _build_left_region(template, right_region.shape[0])
        if left_region is None:
            continue

        grid = np.zeros((right_region.shape[0], left_region.shape[1] + 1 + right_region.shape[1]), dtype=int)
        grid[:, left_region.shape[1]] = SEP
        grid[:, : left_region.shape[1]] = left_region
        grid[:, left_region.shape[1] + 1 :] = right_region

        try:
            _semantic_check(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid new puzzle input")
