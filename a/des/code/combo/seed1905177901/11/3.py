import numpy as np


BG = 0
YELLOW = 4
MAROON = 9
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8)
GENERATION_ATTEMPTS = 300
SHAPE_ATTEMPTS = 120


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = int(rng.integers(5, 9))
        w = int(rng.integers(5, 9))
        mask = np.ones((h, w), dtype=bool)
        notch_count = int(rng.integers(1, 5))
        changed = False

        for _ in range(notch_count):
            side = str(rng.choice(("top", "bottom", "left", "right")))
            if side in ("top", "bottom"):
                max_depth = min(2, h - 2)
                max_span = w - 2
                depth = int(rng.integers(1, max_depth + 1))
                span = int(rng.integers(1, max_span + 1))
                start = int(rng.integers(1, w - span))
                sl = (slice(0, depth), slice(start, start + span))
                if side == "bottom":
                    sl = (slice(h - depth, h), slice(start, start + span))
            else:
                max_depth = min(2, w - 2)
                max_span = h - 2
                depth = int(rng.integers(1, max_depth + 1))
                span = int(rng.integers(1, max_span + 1))
                start = int(rng.integers(1, h - span))
                sl = (slice(start, start + span), slice(0, depth))
                if side == "right":
                    sl = (slice(start, start + span), slice(w - depth, w))

            before = mask.copy()
            mask[sl] = False
            changed = changed or bool(np.any(before != mask))

        if not changed or bool(mask.all()):
            continue
        if int(mask.sum()) < int(np.ceil(0.58 * h * w)):
            continue
        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue

        k = int(rng.integers(0, 4))
        mask = np.rot90(mask, k)
        if bool(rng.integers(0, 2)):
            mask = np.flipud(mask)
        if bool(rng.integers(0, 2)):
            mask = np.fliplr(mask)
        return mask.copy()

    raise RuntimeError("failed to build a valid template shape")


def _sample_shifts(
    rng: np.random.Generator, shape_h: int, shape_w: int, count: int
) -> list[tuple[int, int]]:
    row_step = int(rng.integers(1, min(2, shape_h - 2) + 1))
    col_step = int(rng.integers(1, min(2, shape_w - 2) + 1))

    patterns: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
        2: (
            ((-1, 0), (1, 0)),
            ((0, -1), (0, 1)),
            ((-1, -1), (1, 1)),
            ((-1, 1), (1, -1)),
        ),
        3: (
            ((-1, 0), (0, 0), (1, 0)),
            ((0, -1), (0, 0), (0, 1)),
            ((-1, -1), (0, 0), (1, 1)),
            ((-1, 1), (0, 0), (1, -1)),
            ((-1, 0), (0, 1), (1, -1)),
            ((-1, 1), (0, -1), (1, 0)),
        ),
        4: (
            ((-1, 0), (0, -1), (0, 1), (1, 0)),
            ((-1, -1), (-1, 1), (1, -1), (1, 1)),
            ((-1, 0), (-1, 1), (1, -1), (1, 0)),
            ((-1, -1), (0, 1), (0, -1), (1, 1)),
        ),
    }

    base = patterns[count][int(rng.integers(0, len(patterns[count])))]
    return [(int(dr * row_step), int(dc * col_step)) for dr, dc in base]


def _render_overlay(
    shape_mask: np.ndarray,
    positions: list[tuple[int, int]],
    colors: list[int],
    anchor: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    shape_h, shape_w = map(int, shape_mask.shape)
    if anchor is None:
        anchor = (
            int(round(float(np.mean([r for r, _ in positions])))),
            int(round(float(np.mean([c for _, c in positions])))),
        )

    rel_positions = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in positions]
    min_r = min(r for r, _ in rel_positions)
    min_c = min(c for _, c in rel_positions)
    max_r = max(r + shape_h - 1 for r, _ in rel_positions)
    max_c = max(c + shape_w - 1 for _, c in rel_positions)

    out_h = int(max_r - min_r + 1)
    out_w = int(max_c - min_c + 1)
    counts = np.zeros((out_h, out_w), dtype=int)
    first_color = np.zeros((out_h, out_w), dtype=int)
    shape_int = shape_mask.astype(int)

    for (r0, c0), color in zip(rel_positions, colors):
        rr = int(r0 - min_r)
        cc = int(c0 - min_c)
        view_counts = counts[rr : rr + shape_h, cc : cc + shape_w]
        view_first = first_color[rr : rr + shape_h, cc : cc + shape_w]
        uncovered = shape_mask & (view_counts == 0)
        view_first[uncovered] = int(color)
        view_counts += shape_int

    out = np.full((out_h, out_w), BG, dtype=int)
    out[counts == 1] = first_color[counts == 1]
    out[counts == 2] = MAROON
    out[counts >= 3] = YELLOW
    return out, counts


def main(grid: np.ndarray) -> np.ndarray:
    """Merge same-shape colored objects and recolor overlap counts.

    Input structure
    Background is black. Several monochrome foreground objects share the same
    normalized shape, but they are drawn at different positions and may occlude
    one another because later objects overwrite earlier colors.

    Rule
    Recover the shared template shape, recover each object's original bbox
    position, then redraw all full objects in a common frame whose origin is the
    average bbox center (equivalently, the average bbox top-left because all
    bboxes have the same size). Pixels covered once keep the object's color,
    pixels covered twice become maroon, and pixels covered three or more times
    become yellow. Return the tight crop of that merged overlay.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one foreground object"

    objects = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        rel = coords - np.array([r0, c0], dtype=int)
        objects.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(len(coords)),
                "rel": rel,
            }
        )

    template_obj = max(objects, key=lambda item: int(item["area"]))
    tr0, tc0, tr1, tc1 = template_obj["bbox"]
    shape_h = int(tr1 - tr0 + 1)
    shape_w = int(tc1 - tc0 + 1)
    shape_mask = np.zeros((shape_h, shape_w), dtype=bool)
    shape_mask[template_obj["rel"][:, 0], template_obj["rel"][:, 1]] = True

    positions: list[tuple[int, int]] = []
    ordered_colors: list[int] = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert int(r1 - r0 + 1) == shape_h and int(c1 - c0 + 1) == shape_w
        rel = obj["rel"]
        assert np.all(shape_mask[rel[:, 0], rel[:, 1]]), "visible cells must fit the template"
        positions.append((int(r0), int(c0)))
        ordered_colors.append(int(obj["color"]))

    avg_top = (
        int(round(float(np.mean([r for r, _ in positions])))),
        int(round(float(np.mean([c for _, c in positions])))),
    )
    out, _counts = _render_overlay(shape_mask, positions, ordered_colors, anchor=avg_top)
    assert np.any(out != BG)
    return out


def _gen_attempt(
    rng: np.random.Generator, return_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    shape_mask = _random_shape(rng)
    shape_h, shape_w = map(int, shape_mask.shape)
    object_count = int(rng.choice([2, 3, 4]))
    shifts = _sample_shifts(rng, shape_h, shape_w, object_count)

    full_rows = [dr for dr, _ in shifts]
    full_cols = [dc for _, dc in shifts]
    margin_top = int(rng.integers(1, 4))
    margin_bottom = int(rng.integers(1, 4))
    margin_left = int(rng.integers(1, 4))
    margin_right = int(rng.integers(1, 4))

    union_h = int(max(full_rows) - min(full_rows) + shape_h)
    union_w = int(max(full_cols) - min(full_cols) + shape_w)
    grid_h = union_h + margin_top + margin_bottom
    grid_w = union_w + margin_left + margin_right
    anchor_r = margin_top - min(full_rows)
    anchor_c = margin_left - min(full_cols)
    positions = [(int(anchor_r + dr), int(anchor_c + dc)) for dr, dc in shifts]

    colors = list(map(int, rng.choice(INPUT_COLORS, size=object_count, replace=False)))
    top_idx = int(rng.integers(0, object_count))
    draw_order = [idx for idx in rng.permutation(object_count).tolist() if idx != top_idx] + [top_idx]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for idx in draw_order:
        r0, c0 = positions[idx]
        view = grid[r0 : r0 + shape_h, c0 : c0 + shape_w]
        view[shape_mask] = int(colors[idx])

    overlay, counts = _render_overlay(shape_mask, positions, colors, anchor=(anchor_r, anchor_c))
    shape_area = int(shape_mask.sum())

    if not np.any(counts == 2):
        return None
    if object_count >= 3 and not np.any(counts >= 3):
        return None
    if not np.any(counts == 1):
        return None

    for idx, color in enumerate(colors):
        visible = np.argwhere(grid == int(color))
        if visible.size == 0:
            return None
        bbox = _bbox_from_coords(visible)
        r0, c0 = positions[idx]
        expected_bbox = (int(r0), int(c0), int(r0 + shape_h - 1), int(c0 + shape_w - 1))
        if bbox != expected_bbox:
            return None
        if int(len(visible)) < int(np.ceil(0.25 * shape_area)):
            return None

    if not np.array_equal(main(grid), overlay):
        return None

    if return_meta:
        return grid, {"overlay": overlay, "shape_mask": shape_mask, "positions": positions, "colors": colors}
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
