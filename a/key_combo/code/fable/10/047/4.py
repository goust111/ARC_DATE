import random
from collections import Counter

import numpy as np


BG = 0
QUADRANTS = ("TL", "TR", "BL", "BR")
RECT_SHAPES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(
    grid: np.ndarray,
    bg: int,
    blocked: set[tuple[int, int]] | None = None,
) -> list[dict]:
    h, w = grid.shape
    blocked = blocked or set()
    seen = np.zeros((h, w), dtype=bool)
    for r, c in blocked:
        seen[r, c] = True

    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "bbox": _bbox(arr), "area": int(len(coords))})
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            vals = [int(v) for v in block.ravel()]
            if bg in vals:
                continue
            if len(set(vals)) == 4:
                candidates.append((r, c))
    assert len(candidates) == 1, "expected exactly one 2x2 four-color legend block"
    return candidates[0]


def _quadrant_bounds(shape: tuple[int, int], anchor_r: int, anchor_c: int, quad: str) -> tuple[int, int, int, int]:
    h, w = shape
    if quad == "TL":
        return 0, anchor_r - 1, 0, anchor_c - 1
    if quad == "TR":
        return 0, anchor_r - 1, anchor_c + 2, w - 1
    if quad == "BL":
        return anchor_r + 2, h - 1, 0, anchor_c - 1
    if quad == "BR":
        return anchor_r + 2, h - 1, anchor_c + 2, w - 1
    raise ValueError(quad)


def _quadrant_of_bbox(bbox: tuple[int, int, int, int], anchor_r: int, anchor_c: int) -> str:
    r0, c0, r1, c1 = bbox
    top = r1 < anchor_r
    bottom = r0 > anchor_r + 1
    left = c1 < anchor_c
    right = c0 > anchor_c + 1
    assert top or bottom, "object must lie fully above or below the legend"
    assert left or right, "object must lie fully left or right of the legend"
    if top and left:
        return "TL"
    if top and right:
        return "TR"
    if bottom and left:
        return "BL"
    if bottom and right:
        return "BR"
    raise AssertionError("object must lie entirely inside one quadrant")


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _outer_corner(shape: tuple[int, int], quad: str) -> tuple[int, int]:
    h, w = shape
    if quad == "TL":
        return 0, 0
    if quad == "TR":
        return 0, w - 1
    if quad == "BL":
        return h - 1, 0
    if quad == "BR":
        return h - 1, w - 1
    raise ValueError(quad)


def _inner_corner(anchor_r: int, anchor_c: int, quad: str) -> tuple[int, int]:
    if quad == "TL":
        return anchor_r - 1, anchor_c - 1
    if quad == "TR":
        return anchor_r - 1, anchor_c + 2
    if quad == "BL":
        return anchor_r + 2, anchor_c - 1
    if quad == "BR":
        return anchor_r + 2, anchor_c + 2
    raise ValueError(quad)


def _bbox_ref_corner(bbox: tuple[int, int, int, int], quad: str, toward_outer: bool) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if toward_outer:
        if quad == "TL":
            return r0, c0
        if quad == "TR":
            return r0, c1
        if quad == "BL":
            return r1, c0
        if quad == "BR":
            return r1, c1
    else:
        if quad == "TL":
            return r1, c1
        if quad == "TR":
            return r1, c0
        if quad == "BL":
            return r0, c1
        if quad == "BR":
            return r0, c0
    raise ValueError(quad)


def _bbox_from_outer_offset(
    shape: tuple[int, int],
    quad: str,
    off_r: int,
    off_c: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    h, w = shape
    if quad == "TL":
        r0, c0 = off_r, off_c
    elif quad == "TR":
        r0, c1 = off_r, w - 1 - off_c
        c0 = c1 - width + 1
    elif quad == "BL":
        r1, c0 = h - 1 - off_r, off_c
        r0 = r1 - height + 1
    elif quad == "BR":
        r1, c1 = h - 1 - off_r, w - 1 - off_c
        r0 = r1 - height + 1
        c0 = c1 - width + 1
    else:
        raise ValueError(quad)
    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def _bbox_from_inner_offset(
    shape: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    quad: str,
    off_r: int,
    off_c: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    ir, ic = _inner_corner(anchor_r, anchor_c, quad)
    if quad == "TL":
        r1, c1 = ir - off_r, ic - off_c
        r0, c0 = r1 - height + 1, c1 - width + 1
    elif quad == "TR":
        r1, c0 = ir - off_r, ic + off_c
        r0 = r1 - height + 1
    elif quad == "BL":
        r0, c1 = ir + off_r, ic - off_c
        c0 = c1 - width + 1
    elif quad == "BR":
        r0, c0 = ir + off_r, ic + off_c
    else:
        raise ValueError(quad)
    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def _bbox_fits_in_quadrant(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    quad: str,
) -> bool:
    qr0, qr1, qc0, qc1 = _quadrant_bounds(shape, anchor_r, anchor_c, quad)
    r0, c0, r1, c1 = bbox
    return qr0 <= r0 <= r1 <= qr1 and qc0 <= c0 <= c1 <= qc1


def _component_to_target_bbox(
    comp: dict,
    source_quad: str,
    target_quad: str,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    bbox = comp["bbox"]
    height, width = _bbox_size(bbox)
    ref_r, ref_c = _bbox_ref_corner(bbox, source_quad, toward_outer=True)
    cor_r, cor_c = _outer_corner(shape, source_quad)
    off_r = abs(ref_r - cor_r)
    off_c = abs(ref_c - cor_c)
    return _bbox_from_outer_offset(shape, target_quad, off_r, off_c, height, width)


def _coords_from_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored object into the quadrant declared by the matching cell of the unique 2x2 color key.

    Input structure:
    - A black-background grid contains exactly one 2x2 block whose four cells are four different colors.
    - The two rows and two columns occupied by that 2x2 block split the canvas into four outer quadrants.
    - Elsewhere there are several separated monochrome objects whose colors all come from the 2x2 block.

    Rule:
    - Read the 2x2 block as a legend: its upper-left color points to the upper-left quadrant, its upper-right color to the upper-right quadrant, and likewise for the two lower cells.
    - Any object already inside its own color's target quadrant stays where it is.
    - Otherwise move the whole object by translation: measure the row offset and column offset from the object's outer-corner-side bounding-box corner to its current quadrant's outer grid corner, then place the object in the target quadrant with the same two offsets from that target quadrant's outer grid corner.
    - Leave the 2x2 legend block and the background unchanged.

    Key insight:
    - The 2x2 block is both a color legend and the partition origin, while relocation uses the outer canvas corners rather than reflection around the 2x2 block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    anchor_r, anchor_c = _find_legend(g, bg)

    legend = g[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2]
    color_to_quad = {
        int(legend[0, 0]): "TL",
        int(legend[0, 1]): "TR",
        int(legend[1, 0]): "BL",
        int(legend[1, 1]): "BR",
    }

    blocked = {
        (anchor_r, anchor_c),
        (anchor_r, anchor_c + 1),
        (anchor_r + 1, anchor_c),
        (anchor_r + 1, anchor_c + 1),
    }
    components = _find_components(g, bg=bg, blocked=blocked)
    assert components, "expected at least one movable object"
    assert {int(comp["color"]) for comp in components}.issubset(set(color_to_quad)), "unexpected object color"

    out = g.copy()
    occupied_target = set()
    for comp in components:
        color = int(comp["color"])
        bbox = comp["bbox"]
        coord_set = {tuple(map(int, rc)) for rc in comp["coords"]}
        source_quad = _quadrant_of_bbox(comp["bbox"], anchor_r, anchor_c)
        target_quad = color_to_quad[color]
        target_bbox = bbox if source_quad == target_quad else _component_to_target_bbox(comp, source_quad, target_quad, g.shape)
        assert _bbox_fits_in_quadrant(target_bbox, g.shape, anchor_r, anchor_c, target_quad), "translated object left target quadrant"

        height, width = _bbox_size(bbox)
        tr0, tc0, tr1, tc1 = target_bbox
        assert tr1 - tr0 + 1 == height and tc1 - tc0 + 1 == width
        delta_r = tr0 - bbox[0]
        delta_c = tc0 - bbox[1]
        target_cells = [(r + delta_r, c + delta_c) for r, c in coord_set]
        target_arr = np.asarray(target_cells, dtype=int)
        assert _bbox(target_arr) == target_bbox, "translated shape no longer matches expected outer-corner bbox"
        for cell in target_cells:
            assert cell not in blocked, "translated object overlaps legend"
            if cell not in coord_set:
                assert int(g[cell]) == bg, "translated target cells must be empty in the input"
            assert cell not in occupied_target, "two objects map to the same target region"
        occupied_target.update(target_cells)

        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
        out[target_arr[:, 0], target_arr[:, 1]] = color

    out[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = legend
    return out


def _sample_anchor(h: int, w: int) -> tuple[int, int] | None:
    choices = []
    for r in range(4, h - 5):
        top = r
        bottom = h - r - 2
        if min(top, bottom) < 4:
            continue
        for c in range(4, w - 5):
            left = c
            right = w - c - 2
            if min(left, right) < 4:
                continue
            if top == bottom and left == right:
                continue
            if abs(top - bottom) + abs(left - right) < 3:
                continue
            choices.append((r, c))
    if not choices:
        return None
    return random.choice(choices)


def _touches_occupied(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _max_offsets_for_quad(
    shape: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    quad: str,
    height: int,
    width: int,
) -> tuple[int, int]:
    qr0, qr1, qc0, qc1 = _quadrant_bounds(shape, anchor_r, anchor_c, quad)
    return qr1 - qr0 - height + 1, qc1 - qc0 - width + 1


def _reflection_differs(
    source_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    source_quad: str,
    target_quad: str,
    anchor_r: int,
    anchor_c: int,
) -> bool:
    height, width = _bbox_size(source_bbox)
    ref_r, ref_c = _bbox_ref_corner(source_bbox, source_quad, toward_outer=False)
    inner_r, inner_c = _inner_corner(anchor_r, anchor_c, source_quad)
    off_r = abs(inner_r - ref_r)
    off_c = abs(inner_c - ref_c)
    mirrored = _bbox_from_inner_offset(
        (target_bbox[2] + 1, target_bbox[3] + 1),
        anchor_r,
        anchor_c,
        target_quad,
        off_r,
        off_c,
        height,
        width,
    )
    return mirrored != target_bbox


def _count_unique_legends(grid: np.ndarray, bg: int) -> int:
    count = 0
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            vals = [int(v) for v in grid[r : r + 2, c : c + 2].ravel()]
            if bg not in vals and len(set(vals)) == 4:
                count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    anchor = _sample_anchor(h, w)
    if anchor is None:
        return None
    anchor_r, anchor_c = anchor

    colors = random.sample(range(1, 10), 4)
    color_to_quad = {
        colors[0]: "TL",
        colors[1]: "TR",
        colors[2]: "BL",
        colors[3]: "BR",
    }
    quad_to_color = {quad: color for color, quad in color_to_quad.items()}

    object_count = random.randint(5, 8)
    object_colors = list(colors)
    while len(object_colors) < object_count:
        object_colors.append(random.choice(colors))
    random.shuffle(object_colors)

    moved_target = random.randint(2, object_count - 1)
    move_flags = [True] * moved_target + [False] * (object_count - moved_target)
    random.shuffle(move_flags)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    legend = np.array([[quad_to_color["TL"], quad_to_color["TR"]], [quad_to_color["BL"], quad_to_color["BR"]]], dtype=int)
    inp[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = legend
    out[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = legend

    occupied_in = np.zeros((h, w), dtype=bool)
    occupied_out = np.zeros((h, w), dtype=bool)
    occupied_in[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = True
    occupied_out[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = True

    moved_records = []
    stationary_count = 0

    for idx, color in enumerate(object_colors):
        target_quad = color_to_quad[color]
        want_move = move_flags[idx]
        placed = False
        for _ in range(120):
            height, width = random.choice(RECT_SHAPES)
            max_r_target, max_c_target = _max_offsets_for_quad((h, w), anchor_r, anchor_c, target_quad, height, width)
            if max_r_target < 0 or max_c_target < 0:
                continue

            source_options = [target_quad] if not want_move else [q for q in QUADRANTS if q != target_quad]
            random.shuffle(source_options)
            for source_quad in source_options:
                max_r_source, max_c_source = _max_offsets_for_quad((h, w), anchor_r, anchor_c, source_quad, height, width)
                if max_r_source < 0 or max_c_source < 0:
                    continue

                max_r = min(max_r_target, max_r_source)
                max_c = min(max_c_target, max_c_source)
                if max_r < 0 or max_c < 0:
                    continue

                candidate_offsets = [(rr, cc) for rr in range(max_r + 1) for cc in range(max_c + 1)]
                random.shuffle(candidate_offsets)
                for off_r, off_c in candidate_offsets[: min(len(candidate_offsets), 20)]:
                    target_bbox = _bbox_from_outer_offset((h, w), target_quad, off_r, off_c, height, width)
                    source_bbox = _bbox_from_outer_offset((h, w), source_quad, off_r, off_c, height, width)
                    if _touches_occupied(occupied_out, target_bbox) or _touches_occupied(occupied_in, source_bbox):
                        continue

                    if want_move and not _reflection_differs(
                        source_bbox, target_bbox, source_quad, target_quad, anchor_r, anchor_c
                    ):
                        continue

                    _paint_rect(out, target_bbox, color)
                    _paint_rect(inp, source_bbox, color)
                    _paint_rect(occupied_out, target_bbox, True)
                    _paint_rect(occupied_in, source_bbox, True)
                    if want_move:
                        moved_records.append((source_bbox, target_bbox, source_quad, target_quad))
                    else:
                        stationary_count += 1
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if stationary_count == 0 or len(moved_records) < 2:
        return None
    if not all(np.any(inp == color) for color in colors):
        return None
    if _count_unique_legends(inp, BG) != 1:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None

    comps_in = _find_components(inp, bg=BG, blocked={(anchor_r, anchor_c), (anchor_r, anchor_c + 1), (anchor_r + 1, anchor_c), (anchor_r + 1, anchor_c + 1)})
    comps_out = _find_components(out, bg=BG, blocked={(anchor_r, anchor_c), (anchor_r, anchor_c + 1), (anchor_r + 1, anchor_c), (anchor_r + 1, anchor_c + 1)})
    if len(comps_in) != len(comps_out):
        return None
    for comp in comps_in:
        quad = _quadrant_of_bbox(comp["bbox"], anchor_r, anchor_c)
        target_quad = color_to_quad[int(comp["color"])]
        if quad != target_quad:
            expected_bbox = _component_to_target_bbox(comp, quad, target_quad, inp.shape)
            matched = [other for other in comps_out if int(other["color"]) == int(comp["color"]) and other["bbox"] == expected_bbox]
            if len(matched) != 1:
                return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
