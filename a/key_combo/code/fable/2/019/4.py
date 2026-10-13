import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


BG = Color.BLACK
GREY = Color.GREY
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 160
RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if colors is None:
        colors_iter = sorted(int(c) for c in np.unique(g) if int(c) != int(bg))
    else:
        colors_iter = [int(c) for c in colors if int(c) != int(bg)]

    out: list[dict] = []
    for color in colors_iter:
        mask = g == int(color)
        if not bool(mask.any()):
            continue
        seen = np.zeros(mask.shape, dtype=bool)
        for start_r, start_c in np.argwhere(mask):
            start_r = int(start_r)
            start_c = int(start_c)
            if seen[start_r, start_c]:
                continue
            stack = [(start_r, start_c)]
            coords = []
            seen[start_r, start_c] = True
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return out


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    packed = tuple(int(x) for x in np.asarray(mask, dtype=np.uint8).ravel())
    return tuple(int(v) for v in mask.shape), packed


def _d4_transforms(mask: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    transforms = []
    for k in range(4):
        rot = np.rot90(arr, k)
        transforms.append(rot)
        transforms.append(np.fliplr(rot))

    unique: list[np.ndarray] = []
    seen = set()
    for transformed in transforms:
        sig = _mask_signature(transformed)
        if sig not in seen:
            seen.add(sig)
            unique.append(transformed.copy())
    return unique


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_signature(transformed) for transformed in _d4_transforms(mask))


def _apply_d4(mask: np.ndarray, code: int) -> np.ndarray:
    transformed = np.rot90(np.asarray(mask, dtype=bool), int(code) % 4)
    if int(code) >= 4:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _make_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _build_shape_library() -> tuple[np.ndarray, ...]:
    candidates = [
        ["10", "11", "10"],
        ["10", "10", "11"],
        ["111", "010", "010"],
        ["011", "110", "010"],
        ["101", "111"],
        ["11", "11", "10"],
        ["100", "110", "011"],
        ["110", "010", "011"],
        ["111", "001", "010"],
        ["111", "101", "100"],
        ["1110", "0011"],
    ]
    out = []
    seen = set()
    for rows in candidates:
        mask = _make_mask(rows)
        if not _is_connected(mask):
            continue
        sig = _canonical_signature(mask)
        if sig not in seen:
            seen.add(sig)
            out.append(mask)
    return tuple(out)


SHAPE_LIBRARY = _build_shape_library()


def _separator_info(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    row_candidates = [r for r in range(g.shape[0]) if int(g[r, 0]) != int(bg) and np.all(g[r] == g[r, 0])]
    col_candidates = [c for c in range(g.shape[1]) if int(g[0, c]) != int(bg) and np.all(g[:, c] == g[0, c])]
    assert len(row_candidates) + len(col_candidates) == 1, "expected exactly one separator line"
    if row_candidates:
        r = row_candidates[0]
        return "row", int(r), int(g[r, 0])
    c = col_candidates[0]
    return "col", int(c), int(g[0, c])


def _split_regions(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    kind, idx, _sep = _separator_info(grid, bg)
    if kind == "row":
        first = grid[:idx, :]
        second = grid[idx + 1 :, :]
    else:
        first = grid[:, :idx]
        second = grid[:, idx + 1 :]
    first_has_grey = bool(np.any(first == GREY))
    second_has_grey = bool(np.any(second == GREY))
    assert first_has_grey != second_has_grey, "expected grey copies in exactly one region"
    return (second, first) if first_has_grey else (first, second)


def _legend_dictionary(legend: np.ndarray, bg: int) -> dict:
    colors = [int(c) for c in np.unique(legend) if int(c) not in {int(bg), GREY}]
    components = find_components(legend, bg=bg, colors=colors)
    assert len(components) >= 2, "expected multiple legend shapes"

    mapping: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    for component in components:
        color = int(component["color"])
        mask = crop_bbox(legend == color, component["bbox"])
        sig = _canonical_signature(mask)
        assert sig not in mapping, "legend shapes must be distinct up to D4 symmetry"
        mapping[sig] = color
    return mapping


def _adjacent_label_color(region: np.ndarray, component: dict, bg: int) -> int:
    coords = np.asarray(component["coords"], dtype=int)
    h, w = region.shape
    label_cells = set()
    for r, c in coords:
        for dr, dc in NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                value = int(region[nr, nc])
                if value not in {int(bg), GREY}:
                    label_cells.add((nr, nc))
    assert len(label_cells) == 1, "each grey copy must have exactly one adjacent label pixel"
    rr, cc = next(iter(label_cells))
    return int(region[rr, cc])


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    legend, search = _split_regions(g, bg)
    shape_to_color = _legend_dictionary(legend, bg)

    grey_components = find_components(search, bg=bg, colors=[GREY])
    assert grey_components, "expected at least one grey copy"

    liars = []
    for component in grey_components:
        mask = crop_bbox(search == GREY, component["bbox"])
        sig = _canonical_signature(mask)
        assert sig in shape_to_color, "grey copy must match exactly one legend shape"
        true_color = int(shape_to_color[sig])
        label_color = _adjacent_label_color(search, component, bg)
        assert label_color in set(shape_to_color.values()), "label color must name a legend entry"
        if label_color != true_color:
            liars.append({"mask": mask, "true_color": true_color, "label_color": label_color})

    assert len(liars) == 1, "expected exactly one mismatched label"
    return {"bg": int(bg), "liar": liars[0]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the only grey copy whose label lies about its identity and return that sprite in its true legend color.

    Input structure:
    - One constant-color separator line splits the grid into a legend region and a search region.
    - The legend region contains several disconnected monochrome shapes in distinct colors, defining a color-to-shape dictionary.
    - The search region contains disconnected grey copies of those shapes, each with one adjacent single-pixel colored label.

    Rule:
    - Build the legend dictionary by cropping each colored legend shape.
    - Match every grey search sprite to a legend shape up to rotation or mirror equivalence, then read its true legend color.
    - Find the unique sprite whose adjacent label color disagrees with that true color, crop that sprite's search-region orientation, and recolor its grey cells to the true color.

    Key insight:
    - The label color is only a claimed identity; the real identity comes from D4-equivalent shape matching against the legend.
    """
    info = _analyze(grid)
    bg = int(info["bg"])
    liar = info["liar"]
    mask = np.asarray(liar["mask"], dtype=bool)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(liar["true_color"])
    return out


def _box_is_free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, int(r0) - margin)
    cc0 = max(0, int(c0) - margin)
    rr1 = min(h - 1, int(r1) + margin)
    cc1 = min(w - 1, int(c1) + margin)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_box(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, int(r0) - margin)
    cc0 = max(0, int(c0) - margin)
    rr1 = min(h - 1, int(r1) + margin)
    cc1 = min(w - 1, int(c1) + margin)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_legend_shape(region: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = region.shape
    ph, pw = mask.shape
    top_min = 1 if h - ph >= 2 else 0
    top_max = h - ph - 1 if h - ph >= 2 else h - ph
    left_min = 1 if w - pw >= 2 else 0
    left_max = w - pw - 1 if w - pw >= 2 else w - pw
    if top_max < top_min or left_max < left_min:
        return False

    for _ in range(PLACE_ATTEMPTS):
        top = int(RNG.integers(top_min, top_max + 1))
        left = int(RNG.integers(left_min, left_max + 1))
        if not _box_is_free(occupied, top, left, top + ph - 1, left + pw - 1):
            continue
        patch = region[top : top + ph, left : left + pw]
        patch[mask] = int(color)
        _reserve_box(occupied, top, left, top + ph - 1, left + pw - 1)
        return True
    return False


def _external_label_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates = set()
    for c in range(w):
        if mask[0, c]:
            candidates.add((-1, c))
        if mask[h - 1, c]:
            candidates.add((h, c))
    for r in range(h):
        if mask[r, 0]:
            candidates.add((r, -1))
        if mask[r, w - 1]:
            candidates.add((r, w))
    return list(candidates)


def _place_search_item(
    region: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    label_color: int,
) -> bool:
    h, w = region.shape
    ph, pw = mask.shape
    label_positions = _external_label_positions(mask)
    RNG.shuffle(label_positions)

    for label_r, label_c in label_positions:
        top_min = max(0, -label_r)
        left_min = max(0, -label_c)
        top_max = min(h - ph, h - 1 - label_r)
        left_max = min(w - pw, w - 1 - label_c)
        if top_max < top_min or left_max < left_min:
            continue

        for _ in range(PLACE_ATTEMPTS):
            top = int(RNG.integers(top_min, top_max + 1))
            left = int(RNG.integers(left_min, left_max + 1))
            lr = top + label_r
            lc = left + label_c
            r0 = min(top, lr)
            c0 = min(left, lc)
            r1 = max(top + ph - 1, lr)
            c1 = max(left + pw - 1, lc)
            if not _box_is_free(occupied, r0, c0, r1, c1):
                continue

            patch = region[top : top + ph, left : left + pw]
            patch[mask] = GREY
            region[lr, lc] = int(label_color)
            _reserve_box(occupied, r0, c0, r1, c1)
            return True
    return False


def _sample_transform_code(kind: str) -> int:
    if kind == "identity":
        return 0
    if kind == "rotate":
        return int(RNG.integers(1, 4))
    if kind == "mirror":
        return int(RNG.integers(4, 8))
    raise ValueError(f"unknown transform class: {kind}")


def _compose_regions(k: int, item_count: int) -> tuple[np.ndarray, np.ndarray, int, str, bool] | None:
    orientation = str(RNG.choice(["vertical", "horizontal"]))
    legend_first = bool(RNG.integers(0, 2))

    if orientation == "vertical":
        rows = int(RNG.integers(18, 25))
        legend_cols = int(RNG.integers(8, 11))
        search_cols = int(RNG.integers(12 + max(0, item_count - 5), 17))
        total_cols = legend_cols + 1 + search_cols
        if total_cols > 30:
            return None
        legend = np.full((rows, legend_cols), BG, dtype=int)
        search = np.full((rows, search_cols), BG, dtype=int)
    else:
        cols = int(RNG.integers(18, 25))
        legend_rows = int(RNG.integers(8, 11))
        search_rows = int(RNG.integers(12 + max(0, item_count - 5), 17))
        total_rows = legend_rows + 1 + search_rows
        if total_rows > 30:
            return None
        legend = np.full((legend_rows, cols), BG, dtype=int)
        search = np.full((search_rows, cols), BG, dtype=int)

    return legend, search, int(BG), orientation, legend_first


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    shape_count = int(RNG.integers(3, 5))
    chosen_indices = RNG.choice(len(SHAPE_LIBRARY), size=shape_count, replace=False)
    chosen_shapes = [SHAPE_LIBRARY[int(i)] for i in chosen_indices]

    legend_colors = list(RNG.choice(np.array(PALETTE, dtype=int), size=shape_count, replace=False))
    remaining = [color for color in PALETTE if color not in legend_colors]
    separator_color = int(RNG.choice(np.array(remaining, dtype=int)))

    liar_identity = int(RNG.integers(0, shape_count))
    extras = int(RNG.integers(1, 4))
    identities = list(range(shape_count)) + [liar_identity]
    while len(identities) < shape_count + extras:
        identities.append(int(RNG.integers(0, shape_count)))
    RNG.shuffle(identities)
    item_count = len(identities)

    composed = _compose_regions(shape_count, item_count)
    if composed is None:
        return None
    legend_region, search_region, bg, orientation, legend_first = composed
    legend_occupied = np.zeros(legend_region.shape, dtype=bool)
    search_occupied = np.zeros(search_region.shape, dtype=bool)

    transformed_legend_shapes = []
    for mask in chosen_shapes:
        transformed_legend_shapes.append(_apply_d4(mask, int(RNG.integers(0, 8))))

    legend_order = sorted(range(shape_count), key=lambda i: transformed_legend_shapes[i].size, reverse=True)
    for idx in legend_order:
        if not _place_legend_shape(legend_region, legend_occupied, transformed_legend_shapes[idx], int(legend_colors[idx])):
            return None

    liar_positions = [i for i, identity in enumerate(identities) if identity == liar_identity]
    liar_pos = int(RNG.choice(np.array(liar_positions, dtype=int)))

    transform_kinds = ["identity", "rotate", "mirror"]
    while len(transform_kinds) < item_count:
        transform_kinds.append(str(RNG.choice(["identity", "rotate", "mirror"])))
    RNG.shuffle(transform_kinds)

    expected_output = None
    item_order = list(range(item_count))
    RNG.shuffle(item_order)
    item_order.sort(
        key=lambda idx: -chosen_shapes[identities[idx]].shape[0] * chosen_shapes[identities[idx]].shape[1]
    )

    item_specs = {}
    for idx in range(item_count):
        identity = identities[idx]
        base_mask = chosen_shapes[identity]
        transform_code = _sample_transform_code(transform_kinds[idx])
        transformed_mask = _apply_d4(base_mask, transform_code)
        true_color = int(legend_colors[identity])
        label_color = true_color
        if idx == liar_pos:
            alternatives = [color for color in legend_colors if int(color) != true_color]
            label_color = int(RNG.choice(np.array(alternatives, dtype=int)))
            expected_output = np.full(transformed_mask.shape, bg, dtype=int)
            expected_output[transformed_mask] = true_color
        item_specs[idx] = {"mask": transformed_mask, "label_color": label_color}

    for idx in item_order:
        spec = item_specs[idx]
        if not _place_search_item(search_region, search_occupied, spec["mask"], spec["label_color"]):
            return None

    assert expected_output is not None

    if orientation == "vertical":
        if legend_first:
            full = np.full((legend_region.shape[0], legend_region.shape[1] + 1 + search_region.shape[1]), bg, dtype=int)
            full[:, : legend_region.shape[1]] = legend_region
            full[:, legend_region.shape[1]] = separator_color
            full[:, legend_region.shape[1] + 1 :] = search_region
        else:
            full = np.full((legend_region.shape[0], search_region.shape[1] + 1 + legend_region.shape[1]), bg, dtype=int)
            full[:, : search_region.shape[1]] = search_region
            full[:, search_region.shape[1]] = separator_color
            full[:, search_region.shape[1] + 1 :] = legend_region
    else:
        if legend_first:
            full = np.full((legend_region.shape[0] + 1 + search_region.shape[0], legend_region.shape[1]), bg, dtype=int)
            full[: legend_region.shape[0], :] = legend_region
            full[legend_region.shape[0], :] = separator_color
            full[legend_region.shape[0] + 1 :, :] = search_region
        else:
            full = np.full((search_region.shape[0] + 1 + legend_region.shape[0], legend_region.shape[1]), bg, dtype=int)
            full[: search_region.shape[0], :] = search_region
            full[search_region.shape[0], :] = separator_color
            full[search_region.shape[0] + 1 :, :] = legend_region

    if max(full.shape) > 30:
        return None
    if not np.array_equal(main(full), expected_output):
        return None
    return full, expected_output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid liar-label instance")
