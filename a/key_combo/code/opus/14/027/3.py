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


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

MARKER_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)

GEN_RETRY_LIMIT = 400
GEN_MIN_SIDE = 14
GEN_MAX_SIDE = 21
GEN_PADDING_GAP = 1
GEN_PARTIAL_CLIP_PROB = 0.45


def _neighbors(r: int, c: int, connectivity: int) -> tuple[Coord, ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_block_clipped(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    h, w = grid.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r0) + int(size))
    cc1 = min(w, int(c0) + int(size))
    if rr0 < rr1 and cc0 < cc1:
        grid[rr0:rr1, cc0:cc1] = int(color)


def _visible_rect_area(shape: tuple[int, int], r0: int, c0: int, size: int) -> int:
    h, w = shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r0) + int(size))
    cc1 = min(w, int(c0) + int(size))
    return max(0, rr1 - rr0) * max(0, cc1 - cc0)


def _parse_object(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h % 2 == 0, "object bbox must be an even square"
    size = h // 2
    assert size in (2, 3), "block size must be 2 or 3"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    markers: dict[tuple[int, int], int] = {}

    for br in range(2):
        for bc in range(2):
            block = sub[br * size : (br + 1) * size, bc * size : (bc + 1) * size]
            if np.all(block == Color.BLACK):
                continue
            assert np.all(block != Color.BLACK), "occupied blocks must be solid"
            non_green = block[block != Color.GREEN]
            assert non_green.size == 1, "each green block must contain exactly one marker pixel"
            marker_color = int(non_green[0])
            assert marker_color not in (Color.BLACK, Color.GREEN)
            markers[(br, bc)] = marker_color

    occupied = set(markers)
    assert occupied in ({(0, 0), (1, 1)}, {(0, 1), (1, 0)}), "expected one diagonal pair"
    assert len(set(markers.values())) == 2, "markers inside one object must have distinct colors"

    return {
        "bbox": (r0, c0, r1, c1),
        "size": size,
        "markers": markers,
        "orientation": "nw_se" if occupied == {(0, 0), (1, 1)} else "ne_sw",
    }


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components_from_mask(g != Color.BLACK, connectivity=8)
    assert 2 <= len(comps) <= 3, "expected 2-3 isolated objects"
    objects = [_parse_object(g, comp) for comp in comps]
    objects.sort(key=lambda obj: obj["bbox"])
    return objects


def _target_specs(obj: dict, mode: str = "cross") -> list[tuple[int, int, int]]:
    r0, c0, _, _ = obj["bbox"]
    size = obj["size"]
    markers = obj["markers"]

    # The prompt's per-corner bullet list is inconsistent with the legal occupied diagonals.
    # This uses the consistent reading implied by "inherit from the farthest diagonal source":
    # each outer stamp takes the marker color from the occupied core block diagonally farthest from it.
    if obj["orientation"] == "nw_se":
        if mode == "cross":
            return [
                (r0 - size, c0 + size, markers[(1, 1)]),  # NE outer <- SE marker
                (r0 + 2 * size, c0 - size, markers[(0, 0)]),  # SW outer <- NW marker
            ]
        if mode == "row":
            return [
                (r0 - size, c0 + size, markers[(0, 0)]),
                (r0 + 2 * size, c0 - size, markers[(1, 1)]),
            ]
    else:
        if mode == "cross":
            return [
                (r0 - size, c0 - size, markers[(1, 0)]),  # NW outer <- SW marker
                (r0 + 2 * size, c0 + 2 * size, markers[(0, 1)]),  # SE outer <- NE marker
            ]
        if mode == "row":
            return [
                (r0 - size, c0 - size, markers[(0, 1)]),
                (r0 + 2 * size, c0 + 2 * size, markers[(1, 0)]),
            ]
    raise ValueError(f"unknown mode: {mode}")


def _apply_rule(grid: np.ndarray, mode: str = "cross") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for obj in _parse_objects(g):
        for tr, tc, color in _target_specs(obj, mode=mode):
            _paint_block_clipped(out, tr, tc, obj["size"], color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the complementary outer squares and let each one inherit the opposite diagonal marker color.

    Input structure:
    - The grid is a black background containing 2-3 isolated 8-connected objects.
    - Each object is a 2x2 layout of equal square blocks of side 2 or 3.
    - Exactly two diagonal blocks are solid green, and each green block contains exactly one non-black, non-green marker pixel.

    Rule:
    - Preserve every original pixel.
    - If the occupied diagonal is NW+SE, stamp same-size blocks just outside the NE and SW corners; color the NE stamp with the SE marker color and the SW stamp with the NW marker color.
    - If the occupied diagonal is NE+SW, stamp same-size blocks just outside the NW and SE corners; color the NW stamp with the SW marker color and the SE stamp with the NE marker color.
    - Clip stamped blocks at the grid boundary.

    Key insight:
    - Geometry comes from the diagonal-complement stamping rule, but color comes from the diagonally farthest occupied source block rather than from a nearby block.
    """
    return _apply_rule(grid, mode="cross")


def _expanded_box(shape: tuple[int, int], r0: int, c0: int, size: int) -> BBox:
    h, w = shape
    rr0 = max(0, r0 - size)
    cc0 = max(0, c0 - size)
    rr1 = min(h - 1, r0 + 3 * size - 1)
    cc1 = min(w - 1, c0 + 3 * size - 1)
    return int(rr0), int(cc0), int(rr1), int(cc1)


def _boxes_too_close(a: BBox, b: BBox, gap: int = 1) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _has_partial_clip(r0: int, c0: int, size: int, h: int, w: int) -> bool:
    return (
        0 < r0 < size
        or 0 < c0 < size
        or h - 3 * size < r0 < h - 2 * size
        or w - 3 * size < c0 < w - 2 * size
    )


def _core_spec(r0: int, c0: int, size: int, orientation: str, colors: tuple[int, int]) -> dict:
    if orientation == "nw_se":
        markers = {(0, 0): int(colors[0]), (1, 1): int(colors[1])}
    else:
        markers = {(0, 1): int(colors[0]), (1, 0): int(colors[1])}
    return {
        "bbox": (r0, c0, r0 + 2 * size - 1, c0 + 2 * size - 1),
        "size": size,
        "markers": markers,
        "orientation": orientation,
    }


def _draw_object(
    grid: np.ndarray,
    obj: dict,
    marker_offsets: dict[tuple[int, int], tuple[int, int]],
) -> None:
    r0, c0, _, _ = obj["bbox"]
    size = obj["size"]
    for (br, bc), color in obj["markers"].items():
        rr = r0 + br * size
        cc = c0 + bc * size
        grid[rr : rr + size, cc : cc + size] = Color.GREEN
        mr, mc = marker_offsets[(br, bc)]
        grid[rr + mr, cc + mc] = int(color)


def _validate_example(inp: np.ndarray) -> None:
    objects = _parse_objects(inp)
    orientations = {obj["orientation"] for obj in objects}
    assert orientations == {"nw_se", "ne_sw"}, "both diagonal orientations should appear"

    out = main(inp)
    expected = _apply_rule(inp, mode="cross")
    assert np.array_equal(out, expected), "main output must match the semantic rule"
    assert np.array_equal(out[inp != Color.BLACK], inp[inp != Color.BLACK]), "input pixels must be preserved"
    assert np.any(out != inp), "the transform must add at least one visible stamp"

    for obj in objects:
        for tr, tc, color in _target_specs(obj, mode="cross"):
            assert _visible_rect_area(inp.shape, tr, tc, obj["size"]) > 0, "every stamp must remain at least partly visible"
            rr0 = max(0, tr)
            cc0 = max(0, tc)
            rr1 = min(inp.shape[0], tr + obj["size"])
            cc1 = min(inp.shape[1], tc + obj["size"])
            assert np.all(inp[rr0:rr1, cc0:cc1] == Color.BLACK), "stamp targets must be blank in the input"
            assert np.all(out[rr0:rr1, cc0:cc1] == color), "stamp color must match the inherited marker color"

    wrong_row = _apply_rule(inp, mode="row")
    assert not np.array_equal(out, wrong_row), "examples must rule out same-row color inheritance"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(2, 4))
    h = int(rng.integers(GEN_MIN_SIDE, GEN_MAX_SIDE + 1))
    w = int(rng.integers(GEN_MIN_SIDE, GEN_MAX_SIDE + 1))
    grid = np.full((h, w), Color.BLACK, dtype=int)

    if count == 2:
        orientations = ["nw_se", "ne_sw"]
    else:
        orientations = ["nw_se", "ne_sw", "nw_se" if rng.random() < 0.5 else "ne_sw"]
    rng.shuffle(orientations)

    sizes = []
    for _ in range(count):
        if count == 3 and rng.random() < 0.65:
            sizes.append(2)
        else:
            sizes.append(int(rng.integers(2, 4)))

    clip_index = int(rng.integers(count)) if rng.random() < GEN_PARTIAL_CLIP_PROB else None
    place_order = list(range(count))
    if clip_index is not None:
        place_order.remove(clip_index)
        place_order.insert(0, clip_index)

    placed_boxes: list[BBox] = []
    chosen_specs: dict[int, tuple[dict, dict[tuple[int, int], tuple[int, int]]]] = {}

    for idx in place_order:
        size = sizes[idx]
        orientation = orientations[idx]
        color_pair = tuple(int(v) for v in rng.choice(MARKER_COLORS, size=2, replace=False))
        marker_offsets: dict[tuple[int, int], tuple[int, int]] = {}
        if orientation == "nw_se":
            blocks = [(0, 0), (1, 1)]
        else:
            blocks = [(0, 1), (1, 0)]
        for br, bc in blocks:
            marker_offsets[(br, bc)] = (
                int(rng.integers(0, size)),
                int(rng.integers(0, size)),
            )

        positions = [(r0, c0) for r0 in range(0, h - 2 * size + 1) for c0 in range(0, w - 2 * size + 1)]
        rng.shuffle(positions)

        need_partial_clip = idx == clip_index
        chosen = None
        for r0, c0 in positions:
            obj = _core_spec(r0, c0, size, orientation, color_pair)
            targets = _target_specs(obj, mode="cross")
            if any(_visible_rect_area((h, w), tr, tc, size) == 0 for tr, tc, _ in targets):
                continue
            if need_partial_clip and not _has_partial_clip(r0, c0, size, h, w):
                continue
            box = _expanded_box((h, w), r0, c0, size)
            if any(_boxes_too_close(box, prev, gap=GEN_PADDING_GAP) for prev in placed_boxes):
                continue
            chosen = (obj, marker_offsets, box)
            break

        if chosen is None:
            return None

        obj, offsets, box = chosen
        chosen_specs[idx] = (obj, offsets)
        placed_boxes.append(box)

    for idx in range(count):
        obj, offsets = chosen_specs[idx]
        _draw_object(grid, obj, offsets)

    try:
        _validate_example(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
