import numpy as np


BG = 0
AZURE = 8
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    if h > 2 and w > 2 and mask[1:-1, 1:-1].any():
        return False
    return int(mask.sum()) == 2 * h + 2 * w - 4


def _find_plus_center(patch: np.ndarray) -> tuple[int, int] | None:
    azure = np.argwhere(patch == AZURE)
    if azure.size == 0:
        return None
    assert azure.shape[0] == 5, "azure marker must be a 5-cell plus"
    azure_set = {tuple(map(int, pt)) for pt in azure}
    centers = []
    h, w = patch.shape
    for r, c in azure_set:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            continue
        plus = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        if plus == azure_set:
            centers.append((r, c))
    assert len(centers) == 1, "azure marker must form exactly one plus"
    return centers[0]


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    assert factor >= 1
    if factor == 1:
        return patch.copy()
    return np.repeat(np.repeat(patch, factor, axis=0), factor, axis=1)


def _extract_specs(grid: np.ndarray) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, np.unique(g))) - {BG, AZURE})
    specs: dict[int, dict] = {}
    frame_colors = set(colors)

    for color in colors:
        frame_bboxes = []
        for comp in _components(g == color):
            if _is_frame_component(comp):
                frame_bboxes.append(_bbox(comp))
        assert len(frame_bboxes) == 1, "each border color must define exactly one frame"

        r0, c0, r1, c1 = frame_bboxes[0]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        interior = patch[1:-1, 1:-1]
        marker_cells = []
        for rr, cc in np.argwhere(interior != BG):
            rr = int(rr) + 1
            cc = int(cc) + 1
            value = int(patch[rr, cc])
            if value == AZURE:
                continue
            assert value in frame_colors and value != color, "marker must reference another frame color"
            marker_cells.append((rr, cc, value))
        assert len(marker_cells) <= 1, "each container can have at most one parent marker"

        specs[color] = {
            "color": color,
            "shape": patch.shape,
            "parent": None if not marker_cells else int(marker_cells[0][2]),
            "cross": _find_plus_center(patch),
        }
    return specs


def _build_tree(specs: dict[int, dict]) -> tuple[int, dict[int, list[int]]]:
    children = {color: [] for color in specs}
    roots = []
    for color, spec in specs.items():
        parent = spec["parent"]
        if parent is None:
            roots.append(color)
            continue
        assert parent in specs and parent != color
        children[parent].append(color)

    assert len(roots) == 1, "must have exactly one root container"
    for color, kids in children.items():
        assert len(kids) <= 1, "container hierarchy must be an unambiguous nesting chain"
        if kids:
            assert specs[color]["cross"] is not None, "parent container needs an azure placement cross"
    return roots[0], children


def _assemble_from_specs(specs: dict[int, dict]) -> np.ndarray:
    root, children = _build_tree(specs)

    def build(color: int) -> np.ndarray:
        spec = specs[color]
        h, w = map(int, spec["shape"])
        out = np.full((h, w), BG, dtype=int)
        out[0, :] = color
        out[-1, :] = color
        out[:, 0] = color
        out[:, -1] = color

        if children[color]:
            child = build(children[color][0])
            scale = min((h - 2) // child.shape[0], (w - 2) // child.shape[1])
            assert scale >= 1
            scaled = _scale_patch(child, scale)
            sh, sw = scaled.shape
            cr, cc = map(int, spec["cross"])
            top = cr - sh // 2
            left = cc - sw // 2
            assert 1 <= top and top + sh <= h - 1
            assert 1 <= left and left + sw <= w - 1
            out[top : top + sh, left : left + sw] = scaled

        return out

    return build(root)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered color-coded containers into one nested structure.

    Each frame color is unique. A singleton marker inside one container points to the
    border color of its parent container. Parents carry one azure plus that marks
    where the assembled child should be centered after integer scaling.
    """
    specs = _extract_specs(np.asarray(grid, dtype=int))
    return _assemble_from_specs(specs)


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [v for v in range(lo, hi + 1) if v % 2 == 1]


def _rand_odd(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.choice(_odd_choices(lo, hi)))


def _choose_even_slack(rng: np.random.Generator, child_dim: int, scale: int) -> int:
    max_slack = min(child_dim - 1, 2 if scale > 1 else 4)
    options = [v for v in range(0, max_slack + 1, 2)]
    return int(rng.choice(options))


def _candidate_cross_centers(parent_shape: tuple[int, int], scaled_shape: tuple[int, int]) -> list[tuple[int, int]]:
    ph, pw = map(int, parent_shape)
    sh, sw = map(int, scaled_shape)
    centers = []
    for r in range(2, ph - 2):
        for c in range(2, pw - 2):
            top = r - sh // 2
            left = c - sw // 2
            if 1 <= top and top + sh <= ph - 1 and 1 <= left and left + sw <= pw - 1:
                centers.append((r, c))
    return centers


def _free_marker_cells(shape: tuple[int, int], cross: tuple[int, int] | None) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    occupied = set()
    if cross is not None:
        r, c = map(int, cross)
        occupied = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    strict = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if (r, c) not in occupied
    ]
    if strict:
        return strict
    return [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in occupied
    ]


def _make_container_patch(shape: tuple[int, int], color: int, parent_color: int | None, cross: tuple[int, int] | None,
                          marker_pos: tuple[int, int] | None) -> np.ndarray:
    h, w = map(int, shape)
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if cross is not None:
        r, c = map(int, cross)
        patch[r, c] = AZURE
        patch[r - 1, c] = AZURE
        patch[r + 1, c] = AZURE
        patch[r, c - 1] = AZURE
        patch[r, c + 1] = AZURE
    if parent_color is not None:
        assert marker_pos is not None
        mr, mc = map(int, marker_pos)
        patch[mr, mc] = parent_color
    return patch


def _pack_patches(patches: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    count = len(patches)
    max_rows = min(3, count)
    row_count = 1 if count <= 2 else int(rng.integers(1, max_rows + 1))
    order = list(map(int, rng.permutation(count)))
    rows = [[] for _ in range(row_count)]
    for i, idx in enumerate(order):
        rows[i % row_count].append(patches[idx])
    rows = [row for row in rows if row]

    row_layouts = []
    max_row_width = 0
    for row in rows:
        gaps = [int(rng.integers(2, 6)) for _ in range(len(row) - 1)]
        row_height = max(int(p.shape[0]) for p in row)
        row_width = sum(int(p.shape[1]) for p in row) + sum(gaps)
        row_layouts.append((row, gaps, row_height, row_width))
        max_row_width = max(max_row_width, row_width)

    top_pad = int(rng.integers(1, 5))
    bottom_pad = int(rng.integers(1, 5))
    left_pad = int(rng.integers(1, 5))
    right_pad = int(rng.integers(1, 5))
    extra_width = int(rng.integers(0, 8))
    row_gaps = [int(rng.integers(2, 6)) for _ in range(len(row_layouts) - 1)]

    grid_h = top_pad + bottom_pad + sum(layout[2] for layout in row_layouts) + sum(row_gaps)
    grid_w = left_pad + right_pad + max_row_width + extra_width
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    y = top_pad
    inner_width = grid_w - left_pad - right_pad
    for row_idx, (row, gaps, row_height, row_width) in enumerate(row_layouts):
        x = left_pad + int(rng.integers(0, inner_width - row_width + 1))
        for patch_idx, patch in enumerate(row):
            ph, pw = map(int, patch.shape)
            dy = int(rng.integers(0, row_height - ph + 1))
            grid[y + dy : y + dy + ph, x : x + pw] = patch
            x += pw
            if patch_idx < len(gaps):
                x += gaps[patch_idx]
        if row_idx < len(row_gaps):
            y += row_height + row_gaps[row_idx]
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    depth = int(rng.integers(2, 6))
    scales = [1] * (depth - 1)
    if depth >= 2 and rng.random() < 0.5:
        scales[int(rng.integers(0, depth - 1))] = 3

    leaf_hi = 7 if depth >= 4 else 9
    dims_leaf_to_root = [(_rand_odd(rng, 5, leaf_hi), _rand_odd(rng, 5, leaf_hi))]
    for level, scale in enumerate(scales):
        ch, cw = dims_leaf_to_root[-1]
        inner_h = scale * ch + _choose_even_slack(rng, ch, scale)
        inner_w = scale * cw + _choose_even_slack(rng, cw, scale)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h > 31 or outer_w > 31:
            return None
        dims_leaf_to_root.append((outer_h, outer_w))

    dims_root_to_leaf = list(reversed(dims_leaf_to_root))
    if dims_root_to_leaf[0][0] < 7 or dims_root_to_leaf[0][1] < 7:
        return None

    colors = list(map(int, rng.choice(FRAME_COLORS, size=depth, replace=False)))
    specs: dict[int, dict] = {}
    patches: list[np.ndarray] = []

    for idx, color in enumerate(colors):
        shape = dims_root_to_leaf[idx]
        parent_color = None if idx == 0 else colors[idx - 1]
        cross = None
        if idx < depth - 1:
            child_shape = dims_root_to_leaf[idx + 1]
            scale = min((shape[0] - 2) // child_shape[0], (shape[1] - 2) // child_shape[1])
            if scale < 1:
                return None
            centers = _candidate_cross_centers(shape, (child_shape[0] * scale, child_shape[1] * scale))
            if not centers:
                return None
            cross = centers[int(rng.integers(0, len(centers)))]

        marker_choices = _free_marker_cells(shape, cross)
        if parent_color is not None and not marker_choices:
            return None
        marker_pos = None if parent_color is None else marker_choices[int(rng.integers(0, len(marker_choices)))]

        specs[color] = {
            "color": color,
            "shape": shape,
            "parent": parent_color,
            "cross": cross,
        }
        patches.append(_make_container_patch(shape, color, parent_color, cross, marker_pos))

    if sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in patches) > 2200:
        return None

    grid = _pack_patches(patches, rng)
    expected = _assemble_from_specs(specs)
    actual = main(grid)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
