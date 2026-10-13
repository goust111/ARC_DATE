from __future__ import annotations

from collections import Counter
import random

import numpy as np


# All coordinates and inclusive bounding boxes follow arc_utils' (row, column)
# convention.  The few primitives needed here stay local so SourceFileLoader can
# load this task without depending on sys.path configuration.


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    """Return 4-connected foreground components with inclusive bounding boxes."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for start_r, start_c in np.argwhere(grid != bg):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        stack = [(start_r, start_c)]
        seen[start_r, start_c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and not seen[rr, cc]
                    and int(grid[rr, cc]) != bg
                ):
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        arr = np.asarray(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        values = {int(grid[r, c]) for r, c in coords}
        assert len(values) == 1, "each queue object must be monochromatic"
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] != bg
        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "color": values.pop(),
                "mask": mask,
            }
        )
    return objects


def _scaled_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    assert factor >= 1
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def _decode(grid: np.ndarray) -> dict:
    """Decode the baseline queue, its one missing slot, and the remote template."""
    bg = _background(grid)
    objects = _components(grid, bg)
    assert len(objects) == 4, "expected three queue objects and one remote template"

    bottom_counts = Counter(obj["bbox"][2] for obj in objects)
    baseline, count = bottom_counts.most_common(1)[0]
    assert count == 3 and list(bottom_counts.values()).count(3) == 1
    queue = sorted(
        [obj for obj in objects if obj["bbox"][2] == baseline],
        key=lambda obj: obj["bbox"][1],
    )
    templates = [obj for obj in objects if obj["bbox"][2] != baseline]
    assert len(queue) == 3 and len(templates) == 1
    template = templates[0]
    template_mask = template["mask"]
    template_h, template_w = template_mask.shape
    assert template_h >= 2 and template_w >= 2

    scales: list[int] = []
    queue_colors: list[int] = []
    for obj in queue:
        r0, c0, r1, c1 = obj["bbox"]
        obj_h, obj_w = r1 - r0 + 1, c1 - c0 + 1
        assert obj_h % template_h == 0 and obj_w % template_w == 0
        scale_h, scale_w = obj_h // template_h, obj_w // template_w
        assert scale_h == scale_w and scale_h >= 1
        assert np.array_equal(obj["mask"], _scaled_mask(template_mask, scale_h))
        scales.append(scale_h)
        queue_colors.append(int(obj["color"]))

    gaps = [
        queue[i + 1]["bbox"][1] - queue[i]["bbox"][3] - 1
        for i in range(len(queue) - 1)
    ]
    assert len(gaps) == 2 and min(gaps) >= 1 and max(gaps) > min(gaps)
    normal_gap = min(gaps)
    large_gap_index = int(np.argmax(gaps))
    missing_index = large_gap_index + 1

    # The visible ranks correspond to four logical queue positions with one
    # internal omission.  Fit s(i) = first_scale + step*i in that coordinate
    # system rather than treating visible objects as adjacent.
    logical_indices = [i if i < missing_index else i + 1 for i in range(3)]
    numerator = scales[1] - scales[0]
    denominator = logical_indices[1] - logical_indices[0]
    assert numerator % denominator == 0
    step = numerator // denominator
    first_scale = scales[0] - step * logical_indices[0]
    assert step != 0
    assert all(
        scale == first_scale + step * index
        for scale, index in zip(scales, logical_indices)
    )
    missing_scale = first_scale + step * missing_index
    assert missing_scale >= 1 and missing_scale not in scales

    expected_large_gap = template_w * missing_scale + 2 * normal_gap
    assert gaps[large_gap_index] == expected_large_gap

    # The visible equality pattern identifies a two-color cycle without using
    # color ids as ordered or numerical values.
    parity_to_color: dict[int, int] = {}
    for index, color in zip(logical_indices, queue_colors):
        parity = index % 2
        if parity in parity_to_color:
            assert parity_to_color[parity] == color
        else:
            parity_to_color[parity] = color
    assert set(parity_to_color) == {0, 1}
    assert parity_to_color[0] != parity_to_color[1]
    assert template["color"] not in set(parity_to_color.values())
    missing_color = parity_to_color[missing_index % 2]

    left_obj = queue[large_gap_index]
    right_obj = queue[large_gap_index + 1]
    target_left = left_obj["bbox"][3] + normal_gap + 1
    target_top = baseline - template_h * missing_scale + 1
    assert target_left + template_w * missing_scale + normal_gap == right_obj["bbox"][1]
    assert 0 <= target_top and target_left >= 0
    assert target_top + template_h * missing_scale <= grid.shape[0]
    assert target_left + template_w * missing_scale <= grid.shape[1]

    return {
        "bg": bg,
        "queue": queue,
        "template": template,
        "template_mask": template_mask,
        "baseline": baseline,
        "normal_gap": normal_gap,
        "logical_indices": logical_indices,
        "scales": scales,
        "colors": queue_colors,
        "missing_index": missing_index,
        "missing_scale": missing_scale,
        "missing_color": missing_color,
        "target_top": target_top,
        "target_left": target_left,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the one empty queue slot by scaling and recoloring the remote template.

    Input structure:
    - A dominant background contains three monochrome copies of one shape on a shared
      horizontal baseline and a separate one-scale template in an upper corner.
    - The three baseline objects occupy four regularly spaced logical slots with one
      internal slot empty; their sizes come from an integer arithmetic progression.
    - Queue colors alternate in a visible two-color cycle, while the template uses a
      third color outside that cycle.

    Rule:
    - Locate the empty slot from the unique enlarged horizontal gap, fit the size
      progression in the four-slot queue coordinate system, and infer the omitted
      color from the alternating equality pattern.
    - Pixel-scale the template mask by the missing factor, recolor it with the inferred
      queue color, and add it in the slot with its bottom edge on the shared baseline.
      Preserve the template and every existing pixel.

    Key insight:
    - Geometry fixes which term is absent, while the arithmetic size sequence and the
      independent color cycle respectively supply its scale and color; the remote
      object's own size and color only identify the shape template.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    info = _decode(g)
    out = g.copy()

    mask = _scaled_mask(info["template_mask"], info["missing_scale"])
    r0, c0 = info["target_top"], info["target_left"]
    region = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    assert np.all(region[mask] == info["bg"]), "the inferred slot must be empty"
    region[mask] = info["missing_color"]
    return out


_BASE_SHAPES = (
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    assert patch.shape == mask.shape
    patch[mask] = color


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    base_mask: np.ndarray,
    scales: list[int],
    colors: list[int],
    missing_index: int,
    template_color: int,
    template_top: int,
    template_left: int,
) -> None:
    """Check the intended semantics independently of a mere generator/solver loop."""
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert expected.shape == inp.shape and not np.array_equal(inp, expected)
    assert missing_index in (1, 2)
    assert scales[missing_index] not in [s for i, s in enumerate(scales) if i != missing_index]
    assert 2 * scales[missing_index] == scales[missing_index - 1] + scales[missing_index + 1]
    assert colors[missing_index] == colors[missing_index % 2]
    assert colors[missing_index] != colors[missing_index - 1]
    assert colors[missing_index] != colors[missing_index + 1]
    assert template_color not in set(colors)
    assert np.any(~base_mask), "a non-solid template makes shape transfer necessary"

    changed = inp != expected
    assert np.any(changed)
    bg = _background(inp)
    assert np.all(inp[changed] == bg)
    assert np.all(expected[changed] == colors[missing_index])
    expected_mask = _scaled_mask(base_mask, scales[missing_index])
    changed_coords = np.argwhere(changed)
    r0, c0 = changed_coords.min(axis=0)
    r1, c1 = changed_coords.max(axis=0)
    assert np.array_equal(changed[r0 : r1 + 1, c0 : c1 + 1], expected_mask)

    # The template is retained exactly, and neither its lure color nor either
    # neighboring size/color can replace the intended dual inference.
    template_region = inp[
        template_top : template_top + base_mask.shape[0],
        template_left : template_left + base_mask.shape[1],
    ]
    assert np.array_equal(template_region == template_color, base_mask)
    assert np.array_equal(
        expected[
            template_top : template_top + base_mask.shape[0],
            template_left : template_left + base_mask.shape[1],
        ],
        template_region,
    )
    wrong_color = colors[missing_index - 1]
    assert wrong_color != colors[missing_index]
    assert not np.all(expected[changed] == wrong_color)
    assert _scaled_mask(base_mask, scales[missing_index - 1]).shape != expected_mask.shape

    decoded = _decode(inp)
    assert decoded["missing_index"] == missing_index
    assert decoded["missing_scale"] == scales[missing_index]
    assert decoded["missing_color"] == colors[missing_index]
    assert np.array_equal(main(inp), expected)


def _generation_attempt(ascending: bool, missing_index: int) -> np.ndarray | None:
    base_mask = random.choice(_BASE_SHAPES).copy()
    base_h, base_w = base_mask.shape
    normal_gap = random.choice((1, 2))

    scales = [1, 2, 3, 4]
    if not ascending:
        scales.reverse()

    bg, color_a, color_b, template_color = random.sample(range(10), 4)
    if random.choice((True, False)):
        color_a, color_b = color_b, color_a
    colors = [color_a if i % 2 == 0 else color_b for i in range(4)]

    full_span = sum(base_w * scale for scale in scales) + 3 * normal_gap
    width = random.randint(max(26, full_span + 2), 30)
    height = random.randint(20, 28)
    queue_left = random.randint(1, width - full_span - 1)
    baseline = random.randint(17, height - 2)

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()

    slot_lefts: list[int] = []
    cursor = queue_left
    for scale in scales:
        slot_lefts.append(cursor)
        cursor += base_w * scale + normal_gap

    for index, (scale, color, left) in enumerate(zip(scales, colors, slot_lefts)):
        mask = _scaled_mask(base_mask, scale)
        top = baseline - mask.shape[0] + 1
        _paint_mask(expected, mask, top, left, color)
        if index != missing_index:
            _paint_mask(inp, mask, top, left, color)

    template_top = random.choice((1, 2))
    template_left = random.choice((1, width - base_w - 1))
    template_region = inp[
        template_top : template_top + base_h,
        template_left : template_left + base_w,
    ]
    if np.any(template_region != bg):
        return None
    _paint_mask(inp, base_mask, template_top, template_left, template_color)
    _paint_mask(expected, base_mask, template_top, template_left, template_color)

    try:
        _validate_generated(
            inp,
            expected,
            base_mask=base_mask,
            scales=scales,
            colors=colors,
            missing_index=missing_index,
            template_color=template_color,
            template_top=template_top,
            template_left=template_left,
        )
    except (AssertionError, ValueError):
        return None
    return inp


_BRANCH_SCHEDULE = ((True, 1), (False, 2), (True, 2), (False, 1))
_generation_calls = 0


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked queue-completion instance."""
    global _generation_calls
    ascending, missing_index = _BRANCH_SCHEDULE[_generation_calls % len(_BRANCH_SCHEDULE)]
    for _ in range(200):
        grid = _generation_attempt(ascending, missing_index)
        if grid is not None:
            _generation_calls += 1
            return grid
    raise RuntimeError("generate_input: failed to build a valid queue instance")
