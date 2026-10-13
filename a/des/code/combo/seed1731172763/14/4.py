from __future__ import annotations

import numpy as np


BG = 0
RNG = np.random.default_rng()

# Generation bounds.
MAX_GENERATION_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (3, 4)
COMPLEXITY_PROFILES = {
    3: ((4, 3, 1), (4, 2, 2), (3, 3, 1), (4, 4, 2)),
    4: ((4, 3, 2, 1), (4, 4, 2, 1), (4, 3, 3, 1), (3, 3, 2, 2)),
}
INNER_HEIGHT_RANGE = (4, 7)
INNER_WIDTH_RANGE = (4, 7)
CANVAS_MIN_SIDE = 22
CANVAS_MAX_SIDE = 30
PLACEMENT_MARGIN = 1
PLACEMENT_TRIES = 250

# Small pattern shapes; all stay visually simple and never form a frame by themselves.
PATTERN_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[int(RNG.integers(0, len(seq)))]


def _sample(seq, k: int) -> list[int]:
    idx = RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _same_color_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})

    return comps


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    rel = coords - np.array([r0, c0], dtype=int)
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    if not bool(np.all(on_border)):
        return False

    expected = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != expected:
        return False

    comp_mask = np.zeros((h, w), dtype=bool)
    comp_mask[rel[:, 0], rel[:, 1]] = True

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    return bool(np.array_equal(comp_mask, border_mask))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict] = []
    for comp in _same_color_components(g, bg=BG):
        coords = comp["coords"]
        if not _is_rectangular_frame(coords):
            continue

        r0, c0, r1, c1 = comp["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        pattern_colors = [int(color) for color in np.unique(interior) if int(color) != BG]
        if not pattern_colors:
            continue

        candidates.append(
            {
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "interior": interior,
                "complexity": int(len(pattern_colors)),
            }
        )

    frames = [
        cand
        for cand in candidates
        if not any(
            other is not cand and _strictly_contains(other["bbox"], cand["bbox"])
            for other in candidates
        )
    ]
    assert len(frames) >= 2, "expected at least two non-nested frames"

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            assert not _bboxes_overlap(frames[i]["bbox"], frames[j]["bbox"]), "frames must be disjoint"

    return frames


def _frame_patch(color: int, interior: np.ndarray) -> np.ndarray:
    ih, iw = map(int, interior.shape)
    patch = np.zeros((ih + 2, iw + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1] = interior
    return patch


def _frame_patch_from_item(item: dict) -> np.ndarray:
    return _frame_patch(int(item["color"]), np.asarray(item["interior"], dtype=int))


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, src.shape)
    sub = dst[top : top + h, left : left + w]
    mask = src != BG
    sub[mask] = src[mask]


def _nest_centered(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    ph, pw = map(int, parent.shape)
    ch, cw = map(int, child.shape)
    assert ch <= ph - 4 and cw <= pw - 4, "child frame must fit with a one-pixel gap"

    top = int((ph - ch) // 2)
    left = int((pw - cw) // 2)
    assert top >= 2 and left >= 2
    assert ph - (top + ch) >= 2 and pw - (left + cw) >= 2

    out = parent.copy()
    _paste_nonzero(out, child, top, left)
    return out


def _expanded_outer_patch(item: dict, target_h: int, target_w: int) -> np.ndarray:
    interior = np.asarray(item["interior"], dtype=int)
    ih, iw = map(int, interior.shape)
    assert target_h >= ih + 2 and target_w >= iw + 2

    patch = np.zeros((int(target_h), int(target_w)), dtype=int)
    color = int(item["color"])
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    top = 1 + (int(target_h) - 2 - ih) // 2
    left = 1 + (int(target_w) - 2 - iw) // 2
    patch[top : top + ih, left : left + iw] = interior
    return patch


def _assemble_sorted_items(items: list[dict]) -> np.ndarray:
    assert items, "expected at least one frame"
    if len(items) == 1:
        return _frame_patch_from_item(items[0])

    composite = _frame_patch_from_item(items[-1])
    for item in reversed(items[1:-1]):
        composite = _nest_centered(_frame_patch_from_item(item), composite)

    outer = items[0]
    target_h = max(int(outer["height"]), int(composite.shape[0]) + 4)
    target_w = max(int(outer["width"]), int(composite.shape[1]) + 4)
    outer_patch = _expanded_outer_patch(outer, target_h, target_w)
    return _nest_centered(outer_patch, composite)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort rectangular frames by the number of colors used in their interior
    pattern, then rebuild them as a centered nesting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    items = _extract_frames(g)
    items.sort(
        key=lambda item: (
            -int(item["complexity"]),
            -int(item["area"]),
            int(item["bbox"][0]),
            int(item["bbox"][1]),
        )
    )
    return _assemble_sorted_items(items)


def _center_box(outer_h: int, outer_w: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    top = (int(outer_h) - int(inner_h)) // 2
    left = (int(outer_w) - int(inner_w)) // 2
    return int(top), int(left), int(top + inner_h - 1), int(left + inner_w - 1)


def _cluster_fits(
    shape: tuple[tuple[int, int], ...],
    anchor_r: int,
    anchor_c: int,
    allowed: np.ndarray,
    occupied: np.ndarray,
) -> bool:
    h, w = allowed.shape
    for dr, dc in shape:
        rr = int(anchor_r + dr)
        cc = int(anchor_c + dc)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if not bool(allowed[rr, cc]) or bool(occupied[rr, cc]):
            return False
    return True


def _make_pattern(
    interior_h: int,
    interior_w: int,
    complexity: int,
    palette: list[int],
    blocked: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    pattern = np.zeros((int(interior_h), int(interior_w)), dtype=int)
    allowed = np.ones((int(interior_h), int(interior_w)), dtype=bool)
    if blocked is not None:
        r0, c0, r1, c1 = blocked
        allowed[r0 : r1 + 1, c0 : c1 + 1] = False
    if int(allowed.sum()) < int(complexity):
        return None

    colors = _sample(palette, int(complexity))
    occupied = np.zeros_like(allowed)

    for color in colors:
        anchors = [(int(r), int(c)) for r, c in np.argwhere(allowed & ~occupied)]
        RNG.shuffle(anchors)
        shapes = list(PATTERN_SHAPES)
        RNG.shuffle(shapes)

        placed = False
        for shape in shapes:
            for anchor_r, anchor_c in anchors:
                if not _cluster_fits(shape, anchor_r, anchor_c, allowed, occupied):
                    continue
                for dr, dc in shape:
                    rr = int(anchor_r + dr)
                    cc = int(anchor_c + dc)
                    pattern[rr, cc] = int(color)
                    occupied[rr, cc] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return pattern


def _sorted_specs() -> list[dict] | None:
    count = int(_choice(FRAME_COUNT_CHOICES))
    complexities = [int(v) for v in _choice(COMPLEXITY_PROFILES[count])]

    sizes: list[tuple[int, int]] = [(0, 0)] * count
    inner_h = _randint(*INNER_HEIGHT_RANGE)
    inner_w = _randint(*INNER_WIDTH_RANGE)
    sizes[-1] = (int(inner_h), int(inner_w))
    for idx in range(count - 2, 0, -1):
        child_h, child_w = sizes[idx + 1]
        sizes[idx] = (int(child_h + 4), int(child_w + 4))

    if count >= 2:
        child_h, child_w = sizes[1]
        dh = int(_choice((3, 4)))
        dw = int(_choice((3, 4)))
        if dh == 4 and dw == 4:
            if _randint(0, 1) == 0:
                dh = 3
            else:
                dw = 3
        sizes[0] = (int(child_h + dh), int(child_w + dw))

    frame_colors = _sample(list(range(1, 10)), count)
    pattern_palette = [color for color in range(1, 10) if color not in frame_colors]

    specs: list[dict] = []
    for idx, ((height, width), complexity, frame_color) in enumerate(
        zip(sizes, complexities, frame_colors, strict=True)
    ):
        interior_h = int(height - 2)
        interior_w = int(width - 2)
        blocked = None
        if idx + 1 < count:
            child_h, child_w = sizes[idx + 1]
            blocked = _center_box(interior_h, interior_w, child_h, child_w)
        pattern = _make_pattern(interior_h, interior_w, complexity, pattern_palette, blocked)
        if pattern is None:
            return None
        specs.append(
            {
                "color": int(frame_color),
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
                "complexity": int(complexity),
                "interior": pattern,
            }
        )

    keys = [(int(spec["complexity"]), int(spec["area"])) for spec in specs]
    assert keys == sorted(keys, reverse=True), "generator specs must already match the target sort order"
    return specs


def _choose_canvas(patches: list[np.ndarray]) -> tuple[int, int] | None:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    padded_area = sum(int((patch.shape[0] + 2) * (patch.shape[1] + 2)) for patch in patches)
    base = max(
        CANVAS_MIN_SIDE,
        max_h + 2,
        max_w + 2,
        int(np.ceil(np.sqrt(float(padded_area) * 1.45))),
    )
    if base > CANVAS_MAX_SIDE:
        return None
    return _randint(base, CANVAS_MAX_SIDE), _randint(base, CANVAS_MAX_SIDE)


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, int(top) - PLACEMENT_MARGIN)
    c0 = max(0, int(left) - PLACEMENT_MARGIN)
    r1 = min(h, int(top) + int(height) + PLACEMENT_MARGIN)
    c1 = min(w, int(left) + int(width) + PLACEMENT_MARGIN)
    return not bool(occupied[r0:r1, c0:c1].any())


def _render_input(specs: list[dict]) -> np.ndarray | None:
    patches = [_frame_patch(int(spec["color"]), np.asarray(spec["interior"], dtype=int)) for spec in specs]
    canvas = _choose_canvas(patches)
    if canvas is None:
        return None
    grid_h, grid_w = canvas
    grid = np.zeros((int(grid_h), int(grid_w)), dtype=int)
    occupied = np.zeros((int(grid_h), int(grid_w)), dtype=bool)

    order = sorted(range(len(patches)), key=lambda idx: int(np.prod(patches[idx].shape)), reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = _randint(0, grid_h - ph)
            left = _randint(0, grid_w - pw)
            if not _can_place(occupied, top, left, ph, pw):
                continue
            sub = grid[top : top + ph, left : left + pw]
            mask = patch != BG
            if bool(np.any(sub[mask] != BG)):
                continue
            sub[mask] = patch[mask]
            occupied[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _assemble_specs(specs: list[dict]) -> np.ndarray:
    return _assemble_sorted_items(
        [
            {
                "color": int(spec["color"]),
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "area": int(spec["area"]),
                "complexity": int(spec["complexity"]),
                "interior": np.asarray(spec["interior"], dtype=int),
                "bbox": (0, 0, int(spec["height"]) - 1, int(spec["width"]) - 1),
            }
            for spec in specs
        ]
    )


def _gen_attempt() -> np.ndarray | None:
    specs = _sorted_specs()
    if specs is None:
        return None

    grid = _render_input(specs)
    if grid is None:
        return None

    extracted = _extract_frames(grid)
    if len(extracted) != len(specs):
        return None

    expected = _assemble_specs(specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
