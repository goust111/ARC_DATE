from __future__ import annotations

import random

import numpy as np


_DIRECTIONS = ("top", "right", "bottom", "left")


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(order) > 0
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "background must be uniquely most common"
    return int(values[order[0]])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive (r0, c0, r1, c1) bbox, following arc_utils conventions."""
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_scene(grid: np.ndarray) -> tuple[int, int, tuple[int, int], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background(g)

    objects: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        mask = g == color
        box = _bbox(mask)
        r0, c0, r1, c1 = box
        area = int(mask.sum())
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "each color must form one solid rectangle"
        objects.append({"color": color, "area": area, "bbox": box})

    anchors = [obj for obj in objects if int(obj["area"]) == 1]
    assert len(anchors) == 1, "expected one uniquely identifiable single-pixel anchor"
    anchor = anchors[0]
    ar, ac, _, _ = anchor["bbox"]  # type: ignore[misc]
    anchor_color = int(anchor["color"])

    blocks: list[dict[str, int]] = []
    for obj in objects:
        if obj is anchor:
            continue
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[misc]
        h, w = r1 - r0 + 1, c1 - c0 + 1
        candidates: list[tuple[str, int, int, int]] = []

        if r1 < ar and c0 <= ac <= c1:
            candidates.append(("top", ar - r1 - 1, h, w))
        if c0 > ac and r0 <= ar <= r1:
            candidates.append(("right", c0 - ac - 1, w, h))
        if r0 > ar and c0 <= ac <= c1:
            candidates.append(("bottom", r0 - ar - 1, h, w))
        if c1 < ac and r0 <= ar <= r1:
            candidates.append(("left", ac - c1 - 1, w, h))

        assert len(candidates) == 1, "every block must align with the anchor along exactly one cardinal axis"
        direction, gap, thickness, cross_width = candidates[0]
        assert gap >= 1, "a visible background gap must separate each block from the anchor"
        blocks.append(
            {
                "color": int(obj["color"]),
                "direction": _DIRECTIONS.index(direction),
                "gap": int(gap),
                "thickness": int(thickness),
                "cross_width": int(cross_width),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
            }
        )

    assert blocks, "expected at least one sliding rectangle"
    gaps = [block["gap"] for block in blocks]
    assert len(gaps) == len(set(gaps)), "arrival gaps must be globally unique"
    assert 1 + 2 * sum(block["thickness"] for block in blocks) <= 30, "output exceeds ARC size"
    return bg, anchor_color, (int(ar), int(ac)), blocks


def _wrap_layers(anchor_color: int, blocks: list[dict[str, int]]) -> np.ndarray:
    out = np.array([[int(anchor_color)]], dtype=int)
    for block in sorted(blocks, key=lambda item: item["gap"]):
        thickness = int(block["thickness"])
        n = int(out.shape[0] + 2 * thickness)
        wrapped = np.full((n, n), int(block["color"]), dtype=int)
        wrapped[thickness : thickness + out.shape[0], thickness : thickness + out.shape[1]] = out
        out = wrapped
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Turn axis-aligned arriving blocks into distance-ordered concentric square layers.

    Input structure:
    - A uniquely most-common background contains one single-pixel colored anchor.
    - Several differently colored solid rectangles lie above, below, left, or right
      of it; each rectangle's span contains the anchor column or anchor row.
    - At least one background cell separates every rectangle from the anchor, and
      all resulting edge-to-anchor gap distances are different.

    Rule:
    - Measure every gap in the original full-grid row/column coordinate system and
      sort all rectangles globally by increasing gap, as if they slid simultaneously.
    - Starting with the anchor-colored center cell, wrap one square layer per block
      from inner to outer.  Its color is the block color and its thickness is the
      block length parallel to that block's cardinal sliding direction.
    - Discard the block's perpendicular width; the output side is one plus twice the
      sum of all layer thicknesses.

    Key insight:
    Arrival distance controls the global layer order, while orientation decides
    which rectangle dimension survives as thickness; neither direction, color,
    area, nor the perpendicular width orders the onion.
    """
    _bg, anchor_color, _anchor, blocks = _decode_scene(np.asarray(grid, dtype=int))
    return _wrap_layers(anchor_color, blocks)


def _relative_box(direction: str, gap: int, thickness: int, cross_width: int) -> tuple[int, int, int, int]:
    before = random.randint(0, cross_width - 1)
    after = cross_width - 1 - before
    if direction == "top":
        return -(gap + thickness), -before, -(gap + 1), after
    if direction == "right":
        return -before, gap + 1, after, gap + thickness
    if direction == "bottom":
        return gap + 1, -before, gap + thickness, after
    if direction == "left":
        return -before, -(gap + thickness), after, -(gap + 1)
    raise AssertionError("unknown direction")


def _well_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True when the two rectangles have at least one full background cell between them."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _oracle(anchor_color: int, ordered: list[dict[str, int]], size_key: str = "thickness") -> np.ndarray:
    """Independent Chebyshev-distance renderer used only for generator validation."""
    total = sum(int(item[size_key]) for item in ordered)
    n = 2 * total + 1
    center = total
    rr, cc = np.indices((n, n))
    radius = np.maximum(abs(rr - center), abs(cc - center))
    out = np.full((n, n), int(anchor_color), dtype=int)
    inner = 0
    for item in ordered:
        outer = inner + int(item[size_key])
        out[(radius > inner) & (radius <= outer)] = int(item["color"])
        inner = outer
    return out


def _same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _validate_generated(
    grid: np.ndarray,
    bg: int,
    anchor_color: int,
    anchor: tuple[int, int],
    specs: list[dict[str, int]],
) -> bool:
    """Check scene semantics plus explicit counterexamples to common shortcut solvers."""
    try:
        if grid.ndim != 2 or max(grid.shape) > 30 or _background(grid) != int(bg):
            return False
        parsed_bg, parsed_anchor_color, parsed_anchor, parsed = _decode_scene(grid)
        if (parsed_bg, parsed_anchor_color, parsed_anchor) != (int(bg), int(anchor_color), anchor):
            return False
        if len(parsed) != len(specs) or {item["direction"] for item in parsed} != set(range(4)):
            return False

        expected_by_color = {int(item["color"]): item for item in specs}
        for item in parsed:
            expected = expected_by_color.get(int(item["color"]))
            if expected is None:
                return False
            for key in ("direction", "gap", "thickness", "cross_width", "r0", "c0", "r1", "c1"):
                if int(item[key]) != int(expected[key]):
                    return False

        boxes = [(item["r0"], item["c0"], item["r1"], item["c1"]) for item in parsed]
        boxes.append((anchor[0], anchor[1], anchor[0], anchor[1]))
        if any(not _well_separated(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i)):
            return False

        # Equal-area rectangles with unequal axial lengths rule out using area as thickness.
        if not any(
            a["thickness"] * a["cross_width"] == b["thickness"] * b["cross_width"]
            and a["thickness"] != b["thickness"]
            for i, a in enumerate(parsed)
            for b in parsed[i + 1 :]
        ):
            return False

        ordered = sorted(parsed, key=lambda item: item["gap"])
        expected_out = _oracle(anchor_color, ordered)
        if not _same_grid(main(grid), expected_out):
            return False

        # These alternatives encode visible but wrong shortcuts.  Reject any scene
        # where one happens to agree with the stated global arrival rule.
        alternatives = [
            _oracle(anchor_color, sorted(parsed, key=lambda item: item["color"])),
            _oracle(anchor_color, sorted(parsed, key=lambda item: (item["thickness"], item["gap"]))),
            _oracle(anchor_color, sorted(parsed, key=lambda item: (item["direction"], item["gap"]))),
            _oracle(anchor_color, ordered, size_key="cross_width"),
        ]
        if any(_same_grid(expected_out, alternative) for alternative in alternatives):
            return False
        return True
    except (AssertionError, KeyError, ValueError):
        return False


def _generate_attempt() -> np.ndarray | None:
    n_blocks = random.randint(4, 7)

    # All four arrival branches appear. Extra blocks are spread across different
    # sides, so no side contains more than two rectangles.
    directions = list(_DIRECTIONS)
    directions += random.sample(list(_DIRECTIONS), n_blocks - 4)
    random.shuffle(directions)

    # The first two shapes have equal area (4) but different sliding-axis lengths.
    # This makes both the orientation choice and the discarded cross width visible.
    shapes = [(1, 4), (2, 2)]
    for _ in range(n_blocks - 2):
        thickness = random.randint(1, 3)
        cross_width = random.randint(1, 4)
        if thickness == cross_width == 1:
            cross_width = random.randint(2, 4)
        shapes.append((thickness, cross_width))
    random.shuffle(shapes)
    if sum(thickness for thickness, _ in shapes) > 14:
        return None

    gaps = random.sample(range(1, 10), n_blocks)
    palette = random.sample(range(10), n_blocks + 2)
    bg, anchor_color = palette[:2]
    block_colors = palette[2:]

    specs: list[dict[str, int]] = []
    relative_boxes: list[tuple[int, int, int, int]] = []
    for direction, (thickness, cross_width), gap, color in zip(directions, shapes, gaps, block_colors):
        box = _relative_box(direction, gap, thickness, cross_width)
        if any(not _well_separated(box, old) for old in relative_boxes):
            return None
        if not _well_separated(box, (0, 0, 0, 0)):
            return None
        relative_boxes.append(box)
        specs.append(
            {
                "color": int(color),
                "direction": _DIRECTIONS.index(direction),
                "gap": int(gap),
                "thickness": int(thickness),
                "cross_width": int(cross_width),
            }
        )

    min_r = min(0, *(box[0] for box in relative_boxes))
    min_c = min(0, *(box[1] for box in relative_boxes))
    max_r = max(0, *(box[2] for box in relative_boxes))
    max_c = max(0, *(box[3] for box in relative_boxes))
    pad_top, pad_bottom, pad_left, pad_right = [random.randint(1, 3) for _ in range(4)]
    height = max_r - min_r + 1 + pad_top + pad_bottom
    width = max_c - min_c + 1 + pad_left + pad_right
    if height > 30 or width > 30:
        return None

    ar, ac = pad_top - min_r, pad_left - min_c
    grid = np.full((height, width), int(bg), dtype=int)
    grid[ar, ac] = int(anchor_color)
    for spec, rel_box in zip(specs, relative_boxes):
        rr0, cc0, rr1, cc1 = rel_box
        r0, c0, r1, c1 = ar + rr0, ac + cc0, ar + rr1, ac + cc1
        assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg))
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
        spec.update({"r0": r0, "c0": c0, "r1": r1, "c1": c1})

    if _validate_generated(grid, bg, anchor_color, (ar, ac), specs):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(2000):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate arrival-order onion scene")
