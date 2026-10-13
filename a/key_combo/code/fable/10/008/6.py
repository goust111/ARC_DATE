import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEOMS = ("rot90", "rot270", "rot180", "flip_lr", "flip_ud")
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int):
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _components(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _swap_colors(patch: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = patch.copy()
    mask_a = out == color_a
    mask_b = out == color_b
    out[mask_a] = color_b
    out[mask_b] = color_a
    return out


def _apply_geom(patch: np.ndarray, geom: str) -> np.ndarray:
    if geom == "rot90":
        return np.rot90(patch, 1)
    if geom == "rot270":
        return np.rot90(patch, 3)
    if geom == "rot180":
        return np.rot90(patch, 2)
    if geom == "flip_lr":
        return patch[:, ::-1].copy()
    if geom == "flip_ud":
        return patch[::-1, :].copy()
    raise ValueError(f"unknown geometry: {geom}")


def _apply_step(patch: np.ndarray, geom: str, do_swap: bool, color_a: int, color_b: int) -> np.ndarray:
    out = _apply_geom(patch, geom)
    if do_swap:
        out = _swap_colors(out, color_a, color_b)
    return out


def _repeat_step(patch: np.ndarray, count: int, geom: str, do_swap: bool, color_a: int, color_b: int) -> np.ndarray:
    out = patch.copy()
    for _ in range(int(count)):
        out = _apply_step(out, geom, do_swap, color_a, color_b)
    return out


def _infer_operator(before: np.ndarray, after: np.ndarray, color_a: int, color_b: int):
    matches = []
    for geom in GEOMS:
        for do_swap in (False, True):
            candidate = _apply_step(before, geom, do_swap, color_a, color_b)
            if np.array_equal(candidate, after):
                matches.append((geom, do_swap))
    assert len(matches) == 1, "demo pair must define a unique operator"
    return matches[0]


def _operator_order(geom: str) -> int:
    return 4 if geom in {"rot90", "rot270"} else 2


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the single-step transform shown in the gray demo frame and iterate it on each counted free object.

    Input structure:
    - A black background contains one gray hollow demo frame.
    - Inside the frame sit two square asymmetric multicolor objects: a before-state on the left and its one-step after-state on the right, separated by one empty column.
    - Outside the frame are several non-touching square-bbox free objects in the same two object colors, each with one to three isolated dots of one extra marker color on the ring just outside its bbox.

    Rule:
    - Compare the left and right demo objects to infer the unique one-step operator: one rotation or mirror, optionally followed by swapping the two object colors.
    - For every free object, count its attached dots and apply that same one-step operator exactly that many times to the object's square crop, writing the result back in place.
    - Keep the demo frame, the demo pair, the marker dots, and the background unchanged.

    Key insight:
    - The demo pair encodes only a single application of the operator; each object's dots specify the exponent, so even counts can cancel mirrors or 180-degree turns while color swapping toggles on each odd step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_box = _bbox(g == FRAME)
    assert frame_box is not None, "missing demo frame"
    fr0, fc0, fr1, fc1 = frame_box
    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    demo_mask = inner != BG
    demo_comps = _components(demo_mask)
    assert len(demo_comps) == 2, "demo frame must contain exactly two objects"

    demo_boxes = []
    for cells in demo_comps:
        coords = np.array(cells, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        demo_boxes.append((int(r0), int(c0), int(r1), int(c1)))
    demo_boxes.sort(key=lambda box: box[1])

    left = _crop(inner, demo_boxes[0])
    right = _crop(inner, demo_boxes[1])
    assert left.shape == right.shape and left.shape[0] == left.shape[1], "demo objects must share one square bbox"

    object_colors = sorted(int(x) for x in np.unique(left[left != BG]))
    assert len(object_colors) == 2
    color_a, color_b = object_colors
    assert set(int(x) for x in np.unique(right[right != BG])) == {color_a, color_b}

    geom, do_swap = _infer_operator(left, right, color_a, color_b)

    outside = np.ones_like(g, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    free_mask = outside & np.isin(g, object_colors)
    free_comps = _components(free_mask)
    assert free_comps, "need at least one free object"

    extra_colors = set(int(x) for x in np.unique(g[outside]))
    extra_colors.discard(BG)
    extra_colors -= set(object_colors)
    assert len(extra_colors) == 1, "marker color must be the only non-demo color outside the frame"
    marker_color = next(iter(extra_colors))

    out = g.copy()
    for cells in free_comps:
        coords = np.array(cells, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert r1 - r0 == c1 - c0, "free object bbox must be square"
        patch = _crop(g, (int(r0), int(c0), int(r1), int(c1)))

        count = 0
        for rr in range(r0 - 1, r1 + 2):
            for cc in range(c0 - 1, c1 + 2):
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    if r0 <= rr <= r1 and c0 <= cc <= c1:
                        continue
                    count += int(g[rr, cc] == marker_color)
        assert 1 <= count <= 3, "each free object must have one to three marker dots"
        transformed = _repeat_step(patch, count, geom, do_swap, color_a, color_b)
        _paste(out, transformed, int(r0), int(c0))

    return out


def _random_path(start, end):
    r, c = start
    er, ec = end
    cells = [(r, c)]
    while (r, c) != (er, ec):
        moves = []
        if r < er:
            moves.append((1, 0))
        elif r > er:
            moves.append((-1, 0))
        if c < ec:
            moves.append((0, 1))
        elif c > ec:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _generate_mask(size: int) -> np.ndarray:
    total = size * size
    min_area = max(size + 3, int(np.ceil(total * 0.35)))
    max_area = min(total - 1, int(np.floor(total * 0.62)))
    center = (random.randrange(size), random.randrange(size))
    seeds = [
        (0, random.randrange(size)),
        (size - 1, random.randrange(size)),
        (random.randrange(size), 0),
        (random.randrange(size), size - 1),
    ]

    cells = {center}
    for seed in seeds:
        cells.update(_random_path(seed, center))

    lower = max(min_area, len(cells))
    upper = max(max_area, lower)
    target = random.randint(lower, upper)
    frontier = list(cells)
    while len(cells) < target:
        r, c = random.choice(frontier)
        candidates = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells
        ]
        if not candidates:
            continue
        new_cell = random.choice(candidates)
        cells.add(new_cell)
        frontier.append(new_cell)

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _colorize_mask(mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    coords = list(map(tuple, np.argwhere(mask)))
    patch = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        patch[r, c] = color_a
    count_b = random.randint(max(1, len(coords) // 4), min(len(coords) - 1, max(2, (3 * len(coords)) // 5)))
    for r, c in random.sample(coords, count_b):
        patch[r, c] = color_b
    return patch


def _generate_object_patch(size: int, color_a: int, color_b: int) -> np.ndarray:
    for _ in range(200):
        mask = _generate_mask(size)
        patch = _colorize_mask(mask, color_a, color_b)
        if np.count_nonzero(patch == color_a) == 0 or np.count_nonzero(patch == color_b) == 0:
            continue
        if len(_components(patch != BG)) != 1:
            continue
        if _bbox(patch != BG) != (0, 0, size - 1, size - 1):
            continue
        return patch
    raise RuntimeError("failed to generate object patch")


def _patch_supports_count(patch: np.ndarray, count: int, geom: str, do_swap: bool, color_a: int, color_b: int) -> bool:
    step1 = _repeat_step(patch, 1, geom, do_swap, color_a, color_b)
    target = _repeat_step(patch, count, geom, do_swap, color_a, color_b)
    if np.array_equal(step1, patch):
        return False
    if count > 1 and np.array_equal(target, step1):
        return False
    if count > 2:
        step2 = _repeat_step(patch, 2, geom, do_swap, color_a, color_b)
        if np.array_equal(target, step2):
            return False
    return True


def _build_demo(size: int, geom: str, do_swap: bool, color_a: int, color_b: int):
    for _ in range(300):
        before = _generate_object_patch(size, color_a, color_b)
        after = _apply_step(before, geom, do_swap, color_a, color_b)
        if np.array_equal(before, after):
            continue
        try:
            inferred = _infer_operator(before, after, color_a, color_b)
        except AssertionError:
            continue
        if inferred == (geom, do_swap):
            return before, after
    raise RuntimeError("failed to build unique demo pair")


def _choose_count_profile(order: int):
    if order == 4:
        counts = [1, 2, 3]
        if random.random() < 0.5:
            counts.append(random.choice([1, 2, 3]))
        random.shuffle(counts)
        return counts
    counts = random.choice([[1, 2, 1], [1, 2, 2]])
    if random.random() < 0.5:
        counts.append(random.choice([1, 2]))
    random.shuffle(counts)
    return counts


def _choose_marker_positions(size: int, count: int):
    choices = list(range(size))
    for _ in range(100):
        picks = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    raise RuntimeError("failed to place marker dots")


def _rect_overlaps(a, b, gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_rect(existing, h: int, w: int, grid_h: int, grid_w: int):
    if h + 2 > grid_h or w + 2 > grid_w:
        return None
    for _ in range(400):
        r0 = random.randint(1, grid_h - h - 1)
        c0 = random.randint(1, grid_w - w - 1)
        rect = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_rect_overlaps(rect, other, gap=1) for other in existing):
            continue
        return rect
    return None


def _independent_expected(grid: np.ndarray, geom: str, do_swap: bool, color_a: int, color_b: int, free_specs):
    out = grid.copy()
    for spec in free_specs:
        transformed = _repeat_step(spec["patch"], spec["count"], geom, do_swap, color_a, color_b)
        _paste(out, transformed, spec["r0"], spec["c0"])
    return out


def _validate_sample(inp: np.ndarray, expected: np.ndarray, geom: str, do_swap: bool, color_a: int, color_b: int, free_specs):
    demo_box = _bbox(inp == FRAME)
    assert demo_box is not None
    fr0, fc0, fr1, fc1 = demo_box
    inner = inp[fr0 + 1 : fr1, fc0 + 1 : fc1]
    demo_boxes = []
    for cells in _components(inner != BG):
        coords = np.array(cells, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        demo_boxes.append((int(r0), int(c0), int(r1), int(c1)))
    demo_boxes.sort(key=lambda box: box[1])
    left = _crop(inner, demo_boxes[0])
    right = _crop(inner, demo_boxes[1])
    assert _infer_operator(left, right, color_a, color_b) == (geom, do_swap)

    marker_cells = set(map(tuple, np.argwhere(~np.isin(inp, [BG, FRAME, color_a, color_b]))))
    seen_markers = set()
    assert any(spec["count"] > 1 for spec in free_specs)
    for spec in free_specs:
        r0, c0 = spec["r0"], spec["c0"]
        size = spec["patch"].shape[0]
        ring = set()
        for rr in range(r0 - 1, r0 + size + 1):
            for cc in range(c0 - 1, c0 + size + 1):
                if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
                    if r0 <= rr < r0 + size and c0 <= cc < c0 + size:
                        continue
                    ring.add((rr, cc))
        attached = ring & marker_cells
        assert len(attached) == spec["count"]
        assert seen_markers.isdisjoint(attached)
        seen_markers |= attached
        assert _patch_supports_count(spec["patch"], spec["count"], geom, do_swap, color_a, color_b)
    assert seen_markers == marker_cells
    assert np.array_equal(main(inp), expected)


def _generate_one():
    color_a, color_b, marker_color = random.sample(PALETTE, 3)
    geom = random.choices(GEOMS, weights=[2, 2, 1, 1, 1], k=1)[0]
    do_swap = random.random() < 0.6
    order = _operator_order(geom)

    demo_size = random.randint(5, 7)
    demo_before, demo_after = _build_demo(demo_size, geom, do_swap, color_a, color_b)
    counts = _choose_count_profile(order)

    free_specs = []
    for count in counts:
        for _ in range(200):
            size = random.randint(5, 7)
            patch = _generate_object_patch(size, color_a, color_b)
            if _patch_supports_count(patch, count, geom, do_swap, color_a, color_b):
                free_specs.append({"patch": patch, "count": count, "size": size})
                break
        else:
            raise RuntimeError("failed to generate a nondegenerate free object")

    grid_h = random.randint(23, 29)
    grid_w = random.randint(23, 29)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    demo_h = demo_size + 2
    demo_w = 2 * demo_size + 3
    placed = []
    demo_rect = _place_rect(placed, demo_h, demo_w, grid_h, grid_w)
    if demo_rect is None:
        return None
    placed.append(demo_rect)

    dr0, dc0, dr1, dc1 = demo_rect
    grid[dr0, dc0 : dc1 + 1] = FRAME
    grid[dr1, dc0 : dc1 + 1] = FRAME
    grid[dr0 : dr1 + 1, dc0] = FRAME
    grid[dr0 : dr1 + 1, dc1] = FRAME
    _paste(grid, demo_before, dr0 + 1, dc0 + 1)
    _paste(grid, demo_after, dr0 + 1, dc0 + demo_size + 2)

    for spec in free_specs:
        reserve = _place_rect(placed, spec["size"] + 2, spec["size"] + 2, grid_h, grid_w)
        if reserve is None:
            return None
        placed.append(reserve)
        rr0, rc0, _rr1, _rc1 = reserve
        spec["r0"] = rr0 + 1
        spec["c0"] = rc0 + 1
        _paste(grid, spec["patch"], spec["r0"], spec["c0"])

        side = random.choice(["top", "bottom", "left", "right"])
        positions = _choose_marker_positions(spec["size"], spec["count"])
        spec["marker_side"] = side
        spec["marker_positions"] = positions
        if side == "top":
            for pos in positions:
                grid[spec["r0"] - 1, spec["c0"] + pos] = marker_color
        elif side == "bottom":
            for pos in positions:
                grid[spec["r0"] + spec["size"], spec["c0"] + pos] = marker_color
        elif side == "left":
            for pos in positions:
                grid[spec["r0"] + pos, spec["c0"] - 1] = marker_color
        else:
            for pos in positions:
                grid[spec["r0"] + pos, spec["c0"] + spec["size"]] = marker_color

    expected = _independent_expected(grid, geom, do_swap, color_a, color_b, free_specs)
    _validate_sample(grid, expected, geom, do_swap, color_a, color_b, free_specs)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
