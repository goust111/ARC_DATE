import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return components


def _parse_l_fragment(mask: np.ndarray) -> dict[str, object]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    assert h >= 2 and w >= 2, "each fragment must span at least 2x2"

    missing = ~mask
    assert bool(missing.any()), "fragment must miss at least one cell from its bbox"

    miss_rows, miss_cols = np.where(missing)
    r0 = int(miss_rows.min())
    r1 = int(miss_rows.max())
    c0 = int(miss_cols.min())
    c1 = int(miss_cols.max())
    hole_h = int(r1 - r0 + 1)
    hole_w = int(c1 - c0 + 1)

    assert hole_h < h and hole_w < w, "removed corner must be smaller than the outer rectangle"
    assert int(missing.sum()) == hole_h * hole_w, "missing cells must form a solid rectangle"
    assert bool(missing[r0 : r1 + 1, c0 : c1 + 1].all()), "missing area must be rectangular"

    corner = None
    if r0 == 0 and c0 == 0:
        corner = "TL"
    elif r0 == 0 and c1 == w - 1:
        corner = "TR"
    elif r1 == h - 1 and c0 == 0:
        corner = "BL"
    elif r1 == h - 1 and c1 == w - 1:
        corner = "BR"
    assert corner is not None, "missing rectangle must sit in a bbox corner"

    full = np.ones((h, w), dtype=bool)
    full[r0 : r1 + 1, c0 : c1 + 1] = False
    assert np.array_equal(mask, full), "fragment must equal outer rectangle minus one corner rectangle"

    return {
        "shape": (h, w),
        "mask": mask,
        "corner": corner,
        "hole_shape": (hole_h, hole_w),
        "area": int(mask.sum()),
    }


def _assembled_rectangles(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground fragment"

    by_color: dict[int, list[dict[str, object]]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    rectangles: list[tuple[int, int, int, int]] = []
    for color, parts in by_color.items():
        assert len(parts) == 2, "each color must appear on exactly two fragments"

        parsed = [_parse_l_fragment(np.asarray(part["mask"], dtype=bool)) for part in parts]
        shape0 = parsed[0]["shape"]
        shape1 = parsed[1]["shape"]
        assert shape0 == shape1, "paired fragments must reconstruct the same outer rectangle"

        mask0 = np.asarray(parsed[0]["mask"], dtype=bool)
        mask1 = np.asarray(parsed[1]["mask"], dtype=bool)
        assembled = mask0 | np.rot90(mask1, 2)
        assert bool(assembled.all()), "one fragment rotated 180 degrees must complete the other"

        rect_h, rect_w = map(int, shape0)
        rectangles.append((rect_h * rect_w, rect_h, rect_w, int(color)))

    rectangles.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return bg, rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored L fragments, rotate one piece by 180 degrees, and stack
    the completed rectangles in ascending area order.
    """
    bg, rectangles = _assembled_rectangles(grid)

    out_h = sum(rect_h for _, rect_h, _, _ in rectangles)
    out_w = max(rect_w for _, _, rect_w, _ in rectangles)
    out = np.full((out_h, out_w), bg, dtype=int)

    r0 = 0
    for _, rect_h, rect_w, color in rectangles:
        out[r0 : r0 + rect_h, :rect_w] = color
        r0 += rect_h

    return out


# --------------- generator (o3) ---------------

_RNG = np.random.default_rng()

PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 5
SIDE_MIN = 2
SIDE_MAX = 6
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 30
EDGE_MARGIN = 1
PLACEMENT_TRIES = 250
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))
_CORNERS = ("TL", "TR", "BL", "BR")


def _make_l_mask(height: int, width: int, hole_h: int, hole_w: int, corner: str) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if corner == "TL":
        mask[:hole_h, :hole_w] = False
    elif corner == "TR":
        mask[:hole_h, width - hole_w :] = False
    elif corner == "BL":
        mask[height - hole_h :, :hole_w] = False
    else:
        assert corner == "BR"
        mask[height - hole_h :, width - hole_w :] = False
    return mask


def _sample_pairs(bg: int) -> list[dict[str, object]] | None:
    n_pairs = int(_RNG.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    available = [color for color in PALETTE if color != bg]
    colors = [int(c) for c in _RNG.choice(np.array(available, dtype=int), size=n_pairs, replace=False)]

    used_areas: set[int] = set()
    pairs: list[dict[str, object]] = []

    for color in colors:
        for _ in range(200):
            height = int(_RNG.integers(SIDE_MIN, SIDE_MAX + 1))
            width = int(_RNG.integers(SIDE_MIN, SIDE_MAX + 1))
            area = height * width
            if area in used_areas:
                continue

            corner = _CORNERS[int(_RNG.integers(0, len(_CORNERS)))]
            hole1_h = int(_RNG.integers(1, height))
            hole1_w = int(_RNG.integers(1, width))
            hole2_h = int(_RNG.integers(1, height))
            hole2_w = int(_RNG.integers(1, width))
            mask1 = _make_l_mask(height, width, hole1_h, hole1_w, corner)
            mask2 = _make_l_mask(height, width, hole2_h, hole2_w, corner)

            if not bool((mask1 | np.rot90(mask2, 2)).all()):
                continue

            pairs.append(
                {
                    "color": color,
                    "height": height,
                    "width": width,
                    "area": area,
                    "masks": (mask1, mask2),
                }
            )
            used_areas.add(area)
            break
        else:
            return None

    if sum(int(pair["height"]) for pair in pairs) > GRID_SIDE_MAX:
        return None
    return pairs


def _pick_grid_shape(pairs: list[dict[str, object]]) -> tuple[int, int] | None:
    max_h = max(int(pair["height"]) for pair in pairs)
    max_w = max(int(pair["width"]) for pair in pairs)
    fg_pixels = sum(int(mask.sum()) for pair in pairs for mask in pair["masks"])
    max_color_pixels = max(int(pair["masks"][0].sum() + pair["masks"][1].sum()) for pair in pairs)
    padded_bbox_area = sum((int(pair["height"]) + 2) * (int(pair["width"]) + 2) * 2 for pair in pairs)
    required_area = max(fg_pixels + max_color_pixels + 1, padded_bbox_area)

    min_h = max(GRID_SIDE_MIN, max_h + 2 * EDGE_MARGIN + 1)
    min_w = max(GRID_SIDE_MIN, max_w + 2 * EDGE_MARGIN + 1)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    for _ in range(150):
        grid_h = int(_RNG.integers(min_h, GRID_SIDE_MAX + 1))
        grid_w = int(_RNG.integers(min_w, GRID_SIDE_MAX + 1))
        if grid_h * grid_w >= required_area:
            return grid_h, grid_w
    return None


def _place_fragments(
    pairs: list[dict[str, object]], bg: int, grid_h: int, grid_w: int
) -> np.ndarray | None:
    fragments: list[dict[str, object]] = []
    for pair in pairs:
        for mask in pair["masks"]:
            fragments.append(
                {
                    "color": int(pair["color"]),
                    "height": int(pair["height"]),
                    "width": int(pair["width"]),
                    "mask": np.asarray(mask, dtype=bool),
                }
            )

    order = list(range(len(fragments)))
    _RNG.shuffle(order)
    order.sort(
        key=lambda idx: int(fragments[idx]["height"]) * int(fragments[idx]["width"]),
        reverse=True,
    )

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for idx in order:
        fragment = fragments[idx]
        height = int(fragment["height"])
        width = int(fragment["width"])
        mask = np.asarray(fragment["mask"], dtype=bool)

        min_r = EDGE_MARGIN
        min_c = EDGE_MARGIN
        max_r = grid_h - height - EDGE_MARGIN
        max_c = grid_w - width - EDGE_MARGIN
        if max_r < min_r or max_c < min_c:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = int(_RNG.integers(min_r, max_r + 1))
            c0 = int(_RNG.integers(min_c, max_c + 1))
            if bool(blocked[r0 : r0 + height, c0 : c0 + width].any()):
                continue

            patch = grid[r0 : r0 + height, c0 : c0 + width]
            patch[mask] = int(fragment["color"])

            br0 = max(0, r0 - 1)
            bc0 = max(0, c0 - 1)
            br1 = min(grid_h, r0 + height + 1)
            bc1 = min(grid_w, c0 + width + 1)
            blocked[br0:br1, bc0:bc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _expected_output(bg: int, pairs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(
        (
            (int(pair["area"]), int(pair["height"]), int(pair["width"]), int(pair["color"]))
            for pair in pairs
        ),
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    out_h = sum(rect_h for _, rect_h, _, _ in ordered)
    out_w = max(rect_w for _, _, rect_w, _ in ordered)
    out = np.full((out_h, out_w), bg, dtype=int)

    r0 = 0
    for _, rect_h, rect_w, color in ordered:
        out[r0 : r0 + rect_h, :rect_w] = color
        r0 += rect_h
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.choice(np.array(PALETTE, dtype=int)))
    pairs = _sample_pairs(bg)
    if pairs is None:
        return None

    shape = _pick_grid_shape(pairs)
    if shape is None:
        return None
    grid_h, grid_w = shape

    grid = _place_fragments(pairs, bg, grid_h, grid_w)
    if grid is None:
        return None

    bg_count = int(np.sum(grid == bg))
    fg_counts = [int(np.sum(grid == int(pair["color"]))) for pair in pairs]
    if bg_count <= max(fg_counts):
        return None

    expected = _expected_output(bg, pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
