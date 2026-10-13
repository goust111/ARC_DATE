from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

INSTRUCTION_COLORS = [RED, BLUE, YELLOW, ORANGE, MAGENTA, AZURE]
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue

        stack = [(start_r, start_c)]
        seen[start_r, start_c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))

            if r > 0 and mask[r - 1, c] and not seen[r - 1, c]:
                seen[r - 1, c] = True
                stack.append((r - 1, c))
            if r + 1 < height and mask[r + 1, c] and not seen[r + 1, c]:
                seen[r + 1, c] = True
                stack.append((r + 1, c))
            if c > 0 and mask[r, c - 1] and not seen[r, c - 1]:
                seen[r, c - 1] = True
                stack.append((r, c - 1))
            if c + 1 < width and mask[r, c + 1] and not seen[r, c + 1]:
                seen[r, c + 1] = True
                stack.append((r, c + 1))

        comps.append(np.array(pts, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _paint_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _collect_rings(grid: np.ndarray) -> list[dict]:
    rings: list[dict] = []
    for coords in _components(grid == GREY):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "each grey component must be a rectangular border"

        local = np.zeros((height, width), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = np.zeros_like(local)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(local, expected), "grey components must be 1-cell-thick rectangle rings"

        rings.append({"bbox": box, "area": _bbox_area(box)})

    assert rings, "expected at least one nested-frame group"
    return rings


def _group_rings(rings: list[dict]) -> list[list[dict]]:
    outer_indices = [
        i
        for i, ring in enumerate(rings)
        if not any(
            _strictly_contains(other["bbox"], ring["bbox"])
            for j, other in enumerate(rings)
            if j != i
        )
    ]
    assert outer_indices, "expected at least one outer ring"

    groups: list[list[dict]] = []
    assigned: set[int] = set()
    for outer_idx in sorted(outer_indices, key=lambda i: rings[i]["bbox"]):
        outer_box = rings[outer_idx]["bbox"]
        member_indices = [
            i
            for i, ring in enumerate(rings)
            if i not in assigned and (i == outer_idx or _strictly_contains(outer_box, ring["bbox"]))
        ]
        assert member_indices, "outer ring must own at least itself"

        members = [rings[i] for i in member_indices]
        members.sort(key=lambda ring: (ring["area"], ring["bbox"][0], ring["bbox"][1]))
        for inner, outer in zip(members, members[1:]):
            assert _strictly_contains(outer["bbox"], inner["bbox"]), "group rings must form one nesting chain"

        assigned.update(member_indices)
        groups.append(members)

    assert len(assigned) == len(rings), "every grey ring must belong to exactly one group"
    return groups


def _extract_sequence(
    grid: np.ndarray, inner_box: tuple[int, int, int, int]
) -> tuple[int, list[int], list[int]]:
    r0, c0, r1, c1 = inner_box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "innermost ring must enclose an interior"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != BLACK) & (interior != GREY) & (interior != MAROON)
    coords = np.argwhere(mask)
    assert 1 <= len(coords) <= 3, "instruction sequence must contain 1 to 3 colored pixels"

    rows = coords[:, 0]
    assert np.all(rows == rows[0]), "instruction pixels must lie on one horizontal row"
    order = np.argsort(coords[:, 1])
    local_row = int(rows[0])
    local_cols = [int(coords[idx, 1]) for idx in order]
    assert local_cols == list(range(local_cols[0], local_cols[0] + len(local_cols))), "sequence must be contiguous"

    colors = [int(interior[local_row, col]) for col in local_cols]
    assert len(set(colors)) == len(colors), "sequence colors must be distinct"

    global_row = r0 + 1 + local_row
    global_cols = [c0 + 1 + col for col in local_cols]
    return int(global_row), global_cols, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular container from inner to outer using the innermost color sequence, with the maroon marker deciding whether that sequence is read forward or backward.

    Input structure:
    - Black background with several disjoint groups of 2-4 nested 1-cell-thick grey rectangular borders.
    - Each group's innermost enclosed area contains a short horizontal sequence of 1-3 distinct colored pixels.
    - There is exactly one isolated maroon pixel outside all nested groups.

    Rule:
    - Split the grey pixels into rectangle-ring components and group them into nesting chains from innermost ring to outermost ring.
    - Read the colored instruction sequence inside each innermost ring from left to right.
    - If the maroon marker lies above or left of that group's outer ring, map the sequence forward from inner ring to outer ring; otherwise map it backward from inner ring to outer ring.
    - Recolor each ring with its assigned sequence color, leaving any extra outer rings grey when the sequence is shorter than the nesting depth, and remove the instruction pixels by turning them black.

    Key insight:
    - The colored pixels inside the smallest container are not scene contents but a ring-coloring program, and the otherwise unrelated maroon marker determines whether to read that program in normal or reversed order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_positions = np.argwhere(g == MAROON)
    assert len(marker_positions) == 1, "expected exactly one maroon marker"
    marker_r, marker_c = map(int, marker_positions[0])

    rings = _collect_rings(g)
    groups = _group_rings(rings)

    out = g.copy()
    seen_instruction = np.zeros_like(g, dtype=bool)

    for group in groups:
        inner_box = group[0]["bbox"]
        outer_box = group[-1]["bbox"]
        or0, oc0, or1, oc1 = outer_box
        assert not (or0 <= marker_r <= or1 and oc0 <= marker_c <= oc1), "maroon marker must sit outside every group"

        seq_row, seq_cols, seq_colors = _extract_sequence(g, inner_box)
        seen_instruction[seq_row, seq_cols] = True

        colors_inner_to_outer = seq_colors if (marker_r < or0 or marker_c < oc0) else list(reversed(seq_colors))

        for depth, ring in enumerate(group):
            ring_color = colors_inner_to_outer[depth] if depth < len(colors_inner_to_outer) else GREY
            _paint_ring(out, ring["bbox"], ring_color)

        out[seq_row, seq_cols] = BLACK

    colored_mask = (g != BLACK) & (g != GREY) & (g != MAROON)
    assert np.array_equal(colored_mask, seen_instruction), "all non-grey non-maroon colors must belong to instruction sequences"
    return out


def _sample_group(group_count: int) -> dict:
    max_layers = 3 if group_count == 3 and random.random() < 0.8 else 4
    layers = random.randint(2, max_layers)
    seq_len = random.randint(1, min(3, layers))
    inner_h = random.choice([1, 3])
    extra_w = random.randint(0, 1 if group_count == 3 else 2)
    left_pad = random.randint(0, extra_w)
    right_pad = extra_w - left_pad
    inner_w = seq_len + left_pad + right_pad
    seq_colors = random.sample(INSTRUCTION_COLORS, seq_len)
    outer_h = inner_h + 2 + 4 * (layers - 1)
    outer_w = inner_w + 2 + 4 * (layers - 1)
    return {
        "layers": layers,
        "seq_len": seq_len,
        "seq_colors": seq_colors,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "left_pad": left_pad,
        "outer_h": outer_h,
        "outer_w": outer_w,
    }


def _boxes_are_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _place_groups(specs: list[dict], mode: str) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    if mode == "tl":
        marker = (random.randint(0, 1), random.randint(0, 1))
    else:
        marker = (random.randint(CANVAS_SIZE - 2, CANVAS_SIZE - 1), random.randint(CANVAS_SIZE - 2, CANVAS_SIZE - 1))

    boxes: dict[int, tuple[int, int, int, int]] = {}
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"],
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        height = int(spec["outer_h"])
        width = int(spec["outer_w"])

        if mode == "tl":
            r_lo = marker[0] + 2
            c_lo = marker[1] + 2
            r_hi = CANVAS_SIZE - height
            c_hi = CANVAS_SIZE - width
        else:
            r_lo = 0
            c_lo = 0
            r_hi = marker[0] - height - 1
            c_hi = marker[1] - width - 1

        if r_hi < r_lo or c_hi < c_lo:
            return None

        placed_here = False
        for _ in range(150):
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if all(_boxes_are_separated(box, other, pad=2) for other in placed):
                boxes[idx] = box
                placed.append(box)
                placed_here = True
                break
        if not placed_here:
            return None

    return marker, [boxes[idx] for idx in range(len(specs))]


def _render_group(grid: np.ndarray, spec: dict, outer_box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = outer_box
    layers = int(spec["layers"])

    for offset in range(0, 2 * layers, 2):
        _paint_ring(grid, (r0 + offset, c0 + offset, r1 - offset, c1 - offset), GREY)

    inner_r0 = r0 + 2 * (layers - 1)
    inner_c0 = c0 + 2 * (layers - 1)
    seq_r = inner_r0 + 1 + spec["inner_h"] // 2
    seq_c0 = inner_c0 + 1 + spec["left_pad"]
    for idx, color in enumerate(spec["seq_colors"]):
        grid[seq_r, seq_c0 + idx] = int(color)


def _expected_output(
    inp: np.ndarray,
    specs: list[dict],
    boxes: list[tuple[int, int, int, int]],
    mode: str,
) -> np.ndarray:
    out = inp.copy()
    for spec, outer_box in zip(specs, boxes):
        layers = int(spec["layers"])
        colors = list(spec["seq_colors"]) if mode == "tl" else list(reversed(spec["seq_colors"]))

        for depth in range(layers):
            inset = 2 * (layers - 1 - depth)
            ring_box = (
                outer_box[0] + inset,
                outer_box[1] + inset,
                outer_box[2] - inset,
                outer_box[3] - inset,
            )
            ring_color = colors[depth] if depth < len(colors) else GREY
            _paint_ring(out, ring_box, ring_color)

        inner_r0 = outer_box[0] + 2 * (layers - 1)
        inner_c0 = outer_box[1] + 2 * (layers - 1)
        seq_r = inner_r0 + 1 + spec["inner_h"] // 2
        seq_c0 = inner_c0 + 1 + spec["left_pad"]
        out[seq_r, seq_c0 : seq_c0 + spec["seq_len"]] = BLACK

    return out


def _crop_bounds(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    assert len(coords) > 0
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))

    top_margin = random.randint(0, min(3, r0))
    left_margin = random.randint(0, min(3, c0))
    bottom_margin = random.randint(0, min(3, CANVAS_SIZE - 1 - r1))
    right_margin = random.randint(0, min(3, CANVAS_SIZE - 1 - c1))
    return (
        r0 - top_margin,
        c0 - left_margin,
        r1 + bottom_margin,
        c1 + right_margin,
    )


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choices([1, 2, 3], weights=[3, 4, 2], k=1)[0]
    specs = [_sample_group(group_count) for _ in range(group_count)]

    footprint = sum((spec["outer_h"] + 2) * (spec["outer_w"] + 2) for spec in specs)
    if footprint > 520:
        return None

    mode = random.choice(["tl", "br"])
    placed = _place_groups(specs, mode)
    if placed is None:
        return None
    marker, boxes = placed

    full_inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    full_inp[marker] = MAROON
    for spec, outer_box in zip(specs, boxes):
        _render_group(full_inp, spec, outer_box)

    full_out = _expected_output(full_inp, specs, boxes, mode)
    crop_r0, crop_c0, crop_r1, crop_c1 = _crop_bounds(full_inp)
    inp = full_inp[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    expected = full_out[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame recoloring input")
