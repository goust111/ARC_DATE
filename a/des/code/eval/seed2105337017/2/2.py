import numpy as np


BG = 0
_NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_AXES = ("vertical", "horizontal", "diag", "anti")


def _components(mask: np.ndarray) -> list[frozenset[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[frozenset[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(frozenset(pts))
    return comps


def _bbox(points: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _translate(
    points: frozenset[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int
) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in points)


def _reflect_point(point: tuple[int, int], axis_name: str, param: int) -> tuple[int, int]:
    r, c = point
    if axis_name == "vertical":
        return r, param - c
    if axis_name == "horizontal":
        return param - r, c
    if axis_name == "diag":
        return c + param, r - param
    return param - c, param - r


def _symmetry_axes(
    points: frozenset[tuple[int, int]] | set[tuple[int, int]]
) -> list[tuple[str, int]]:
    pts = set(points)
    r0, c0, r1, c1 = _bbox(pts)
    axes: list[tuple[str, int]] = []

    c_sum = c0 + c1
    if all((r, c_sum - c) in pts for r, c in pts):
        axes.append(("vertical", c_sum))

    r_sum = r0 + r1
    if all((r_sum - r, c) in pts for r, c in pts):
        axes.append(("horizontal", r_sum))

    diffs = [r - c for r, c in pts]
    diff_sum = min(diffs) + max(diffs)
    if diff_sum % 2 == 0:
        b = diff_sum // 2
        if all((c + b, r - b) in pts for r, c in pts):
            axes.append(("diag", b))

    sums = [r + c for r, c in pts]
    axis_sum = min(sums) + max(sums)
    if axis_sum % 2 == 0:
        b = axis_sum // 2
        if all((b - c, b - r) in pts for r, c in pts):
            axes.append(("anti", b))

    return axes


def _is_connected(points: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    pts = set(points)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS8:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _is_cross_mirror(
    comp_a: frozenset[tuple[int, int]],
    comp_b: frozenset[tuple[int, int]],
    axis_name: str,
    param: int,
) -> bool:
    set_a = set(comp_a)
    set_b = set(comp_b)
    for point in set_a:
        reflected = _reflect_point(point, axis_name, param)
        if reflected != point and reflected not in set_b:
            return False
    for point in set_b:
        reflected = _reflect_point(point, axis_name, param)
        if reflected != point and reflected not in set_a:
            return False
    return True


def _crop_patch(
    points: frozenset[tuple[int, int]] | set[tuple[int, int]], color: int
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(points)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in points:
        patch[r - r0, c - c0] = color
    return patch, patch != BG


def _boxes_touch(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box1
    rr0, cc0, rr1, cc1 = box2
    separated = r1 + 1 < rr0 or rr1 + 1 < r0 or c1 + 1 < cc0 or cc1 + 1 < c0
    return not separated


def _find_unique_merge(
    comp_a: frozenset[tuple[int, int]], comp_b: frozenset[tuple[int, int]]
) -> frozenset[tuple[int, int]]:
    a_r0, a_c0, a_r1, a_c1 = _bbox(comp_a)
    b_r0, b_c0, b_r1, b_c1 = _bbox(comp_b)
    a_h = a_r1 - a_r0 + 1
    a_w = a_c1 - a_c0 + 1
    b_h = b_r1 - b_r0 + 1
    b_w = b_c1 - b_c0 + 1

    dr_min = a_r0 - b_h - 1 - b_r1
    dr_max = a_r1 + a_h + 1 - b_r0
    dc_min = a_c0 - b_w - 1 - b_c1
    dc_max = a_c1 + a_w + 1 - b_c0

    solutions: list[frozenset[tuple[int, int]]] = []
    for dr in range(dr_min, dr_max + 1):
        for dc in range(dc_min, dc_max + 1):
            moved_b = _translate(comp_b, dr, dc)
            merged = frozenset(set(comp_a) | set(moved_b))
            if not _is_connected(merged):
                continue
            for axis_name, param in _symmetry_axes(merged):
                if _is_cross_mirror(comp_a, moved_b, axis_name, param):
                    solutions.append(merged)
                    break

    unique = []
    seen = set()
    for merged in solutions:
        key = tuple(sorted(merged))
        if key not in seen:
            seen.add(key)
            unique.append(merged)

    assert len(unique) == 1, f"expected exactly one symmetric merge, got {len(unique)}"
    return unique[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one foreground color"

    merged_specs: list[tuple[int, int, int, bytes, np.ndarray]] = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, f"color {color} must have exactly two components"
        merged = _find_unique_merge(comps[0], comps[1])
        patch, mask = _crop_patch(merged, color)
        h, w = patch.shape
        merged_specs.append((len(merged), h, w, mask.tobytes(), patch))

    merged_specs.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    out_h = max(item[1] for item in merged_specs)
    out_w = sum(item[2] for item in merged_specs) + len(merged_specs) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c0 = 0
    for idx, (_area, h, w, _mask_key, patch) in enumerate(merged_specs):
        out[:h, c0 : c0 + w] = patch
        c0 += w
        if idx != len(merged_specs) - 1:
            c0 += 1
    return out


# --------------- generator ---------------

_PAIR_RETRIES = 80
_PLACE_RETRIES = 200
_GEN_RETRIES = 400


def _grow_connected(
    valid: set[tuple[int, int]],
    target_size: int,
    preferred_starts: list[tuple[int, int]],
    rng: np.random.Generator,
) -> frozenset[tuple[int, int]] | None:
    starts = [pt for pt in preferred_starts if pt in valid]
    if not starts:
        starts = list(valid)
    if not starts:
        return None

    cells = {starts[int(rng.integers(0, len(starts)))]}
    while len(cells) < target_size:
        frontier = set()
        for r, c in cells:
            for dr, dc in _NEIGHBORS8:
                nxt = (r + dr, c + dc)
                if nxt in valid and nxt not in cells:
                    frontier.add(nxt)
        if not frontier:
            return None
        options = list(frontier)
        cells.add(options[int(rng.integers(0, len(options)))])
    return frozenset(cells)


def _validate_local_pair(
    comp_a: frozenset[tuple[int, int]], comp_b: frozenset[tuple[int, int]], axis_name: str
) -> bool:
    merged = frozenset(set(comp_a) | set(comp_b))
    if not _is_connected(merged):
        return False
    axes = _symmetry_axes(merged)
    if len(axes) != 1 or axes[0][0] != axis_name:
        return False
    try:
        recovered = _find_unique_merge(comp_a, comp_b)
    except AssertionError:
        return False
    return recovered == merged


def _make_vertical_pair(rng: np.random.Generator) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    for _ in range(_PAIR_RETRIES):
        h = int(rng.integers(3, 7))
        half_w = int(rng.integers(2, 5))
        max_cells = h * half_w
        target_max = min(max_cells, 9)
        if target_max < 4:
            continue
        target_size = int(rng.integers(4, target_max + 1))
        valid = {(r, c) for r in range(h) for c in range(half_w)}
        preferred = [(r, half_w - 1) for r in range(h)]
        left = _grow_connected(valid, target_size, preferred, rng)
        if left is None:
            continue
        boundary_rows = sorted({r for r, c in left if c == half_w - 1})
        if not boundary_rows:
            continue
        axis_row = boundary_rows[int(rng.integers(0, len(boundary_rows)))]
        comp_a = frozenset(set(left) | {(axis_row, half_w)})
        comp_b = frozenset((r, 2 * half_w - c) for r, c in left)
        if _validate_local_pair(comp_a, comp_b, "vertical"):
            return comp_a, comp_b
    return None


def _make_horizontal_pair(rng: np.random.Generator) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    for _ in range(_PAIR_RETRIES):
        half_h = int(rng.integers(2, 5))
        w = int(rng.integers(3, 7))
        max_cells = half_h * w
        target_max = min(max_cells, 9)
        if target_max < 4:
            continue
        target_size = int(rng.integers(4, target_max + 1))
        valid = {(r, c) for r in range(half_h) for c in range(w)}
        preferred = [(half_h - 1, c) for c in range(w)]
        top = _grow_connected(valid, target_size, preferred, rng)
        if top is None:
            continue
        boundary_cols = sorted({c for r, c in top if r == half_h - 1})
        if not boundary_cols:
            continue
        axis_col = boundary_cols[int(rng.integers(0, len(boundary_cols)))]
        comp_a = frozenset(set(top) | {(half_h, axis_col)})
        comp_b = frozenset((2 * half_h - r, c) for r, c in top)
        if _validate_local_pair(comp_a, comp_b, "horizontal"):
            return comp_a, comp_b
    return None


def _make_diag_pair(rng: np.random.Generator) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    for _ in range(_PAIR_RETRIES):
        side = int(rng.integers(4, 7))
        valid = {(r, c) for r in range(side) for c in range(side) if r > c}
        target_max = min(len(valid), 10)
        if target_max < 5:
            continue
        target_size = int(rng.integers(5, target_max + 1))
        preferred = [(r, c) for r in range(side) for c in range(side) if r == c + 1]
        lower = _grow_connected(valid, target_size, preferred, rng)
        if lower is None:
            continue
        boundary_diag = sorted({c for r, c in lower if r == c + 1})
        if not boundary_diag:
            continue
        diag_idx = boundary_diag[int(rng.integers(0, len(boundary_diag)))]
        comp_a = frozenset(set(lower) | {(diag_idx, diag_idx)})
        comp_b = frozenset((c, r) for r, c in lower)
        if _validate_local_pair(comp_a, comp_b, "diag"):
            return comp_a, comp_b
    return None


def _make_anti_pair(rng: np.random.Generator) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    for _ in range(_PAIR_RETRIES):
        side = int(rng.integers(4, 7))
        valid = {(r, c) for r in range(side) for c in range(side) if r + c < side - 1}
        target_max = min(len(valid), 10)
        if target_max < 5:
            continue
        target_size = int(rng.integers(5, target_max + 1))
        preferred = [(r, c) for r in range(side) for c in range(side) if r + c == side - 2]
        lower = _grow_connected(valid, target_size, preferred, rng)
        if lower is None:
            continue
        boundary_axis = sorted([pt for pt in lower if pt[0] + pt[1] == side - 2])
        if not boundary_axis:
            continue
        br, bc = boundary_axis[int(rng.integers(0, len(boundary_axis)))]
        comp_a = frozenset(set(lower) | {(br, bc + 1)})
        comp_b = frozenset((side - 1 - c, side - 1 - r) for r, c in lower)
        if _validate_local_pair(comp_a, comp_b, "anti"):
            return comp_a, comp_b
    return None


def _make_pair(
    axis_name: str, rng: np.random.Generator
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    if axis_name == "vertical":
        made = _make_vertical_pair(rng)
    elif axis_name == "horizontal":
        made = _make_horizontal_pair(rng)
    elif axis_name == "diag":
        made = _make_diag_pair(rng)
    else:
        made = _make_anti_pair(rng)
    if made is None:
        return None
    comp_a, comp_b = made
    return comp_a, comp_b, frozenset(set(comp_a) | set(comp_b))


def _can_place_box(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    return all(not _boxes_touch(box, prev) for prev in boxes)


def _place_patch(
    grid: np.ndarray, patch: np.ndarray, boxes: list[tuple[int, int, int, int]], rng: np.random.Generator
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    color = int(patch.max())
    coords = np.argwhere(patch == color)
    for _ in range(_PLACE_RETRIES):
        r0 = int(rng.integers(0, h - ph + 1))
        c0 = int(rng.integers(0, w - pw + 1))
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not _can_place_box(box, boxes):
            continue
        abs_coords = coords + np.array([r0, c0], dtype=int)
        if np.any(grid[abs_coords[:, 0], abs_coords[:, 1]] != BG):
            continue
        grid[abs_coords[:, 0], abs_coords[:, 1]] = color
        boxes.append(box)
        return True
    return False


def _render_expected(pair_specs: list[dict]) -> np.ndarray:
    specs = []
    for spec in pair_specs:
        merged_patch = spec["merged_patch"]
        mask = merged_patch != BG
        h, w = merged_patch.shape
        specs.append((spec["area"], h, w, mask.tobytes(), merged_patch))
    specs.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    out_h = max(item[1] for item in specs)
    out_w = sum(item[2] for item in specs) + len(specs) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    c0 = 0
    for idx, (_area, h, w, _mask_key, patch) in enumerate(specs):
        out[:h, c0 : c0 + w] = patch
        c0 += w
        if idx != len(specs) - 1:
            c0 += 1
    return out


def _valid_input(grid: np.ndarray, expected: np.ndarray) -> bool:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    if not colors:
        return False
    for color in colors:
        comps = _components(grid == color)
        if len(comps) != 2:
            return False
        if _boxes_touch(_bbox(comps[0]), _bbox(comps[1])):
            return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    return np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_pairs = int(rng.integers(2, 5))
    colors = list(map(int, rng.choice(np.arange(1, 10), size=n_pairs, replace=False)))
    axis_indices = rng.choice(len(_AXES), size=n_pairs, replace=False)
    axes = [_AXES[int(idx)] for idx in axis_indices]

    pair_specs: list[dict] = []
    used_areas: set[int] = set()
    for color, axis_name in zip(colors, axes, strict=False):
        pair = _make_pair(axis_name, rng)
        if pair is None:
            return None
        comp_a, comp_b, merged = pair
        area = len(merged)
        if area in used_areas:
            return None
        used_areas.add(area)
        frag_a, _ = _crop_patch(comp_a, color)
        frag_b, _ = _crop_patch(comp_b, color)
        merged_patch, _ = _crop_patch(merged, color)
        pair_specs.append(
            {
                "color": color,
                "frag_a": frag_a,
                "frag_b": frag_b,
                "merged_patch": merged_patch,
                "area": area,
            }
        )

    total_bbox_area = sum(spec["frag_a"].size + spec["frag_b"].size for spec in pair_specs)
    max_h = max(
        max(spec["frag_a"].shape[0], spec["frag_b"].shape[0], spec["merged_patch"].shape[0]) for spec in pair_specs
    )
    max_w = max(
        max(spec["frag_a"].shape[1], spec["frag_b"].shape[1], spec["merged_patch"].shape[1]) for spec in pair_specs
    )
    span = max(max_h + 6, max_w + 6, int(np.ceil(np.sqrt(total_bbox_area * 2.4))))
    h = int(rng.integers(span, span + 6))
    w = int(rng.integers(span, span + 6))
    grid = np.full((h, w), BG, dtype=int)

    fragments = []
    for spec in pair_specs:
        fragments.append(spec["frag_a"])
        fragments.append(spec["frag_b"])
    fragments.sort(key=lambda patch: (patch.shape[0] * patch.shape[1], patch.shape[0], patch.shape[1]), reverse=True)

    boxes: list[tuple[int, int, int, int]] = []
    for patch in fragments:
        if not _place_patch(grid, patch, boxes, rng):
            return None

    expected = _render_expected(pair_specs)
    if not _valid_input(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
