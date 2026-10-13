from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _op_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a & b


def _op_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a | b


def _op_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a ^ b


def _op_a_only(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a & ~b


def _op_b_only(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b & ~a


OPS = {
    "intersection": _op_and,
    "union": _op_or,
    "xor": _op_xor,
    "left_only": _op_a_only,
    "right_only": _op_b_only,
}


def _get_bg(grid: np.ndarray) -> int:
    """Local copy of arc_utils.get_bg semantics."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Local copy of arc_utils.bbox_coords semantics."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Local copy of arc_utils.crop_bbox semantics."""
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask, returned as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_divider(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []
    for r in range(h):
        if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0]):
            candidates.append(("horizontal", r, int(grid[r, 0])))
    for c in range(w):
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c]):
            candidates.append(("vertical", c, int(grid[0, c])))
    assert len(candidates) == 1, "expected one full divider line"
    return candidates[0]


def _window_boxes(grid: np.ndarray, bg: int, divider_color: int) -> list[tuple[int, int, int, int]]:
    mask = (grid != bg) & (grid != divider_color)
    boxes: list[tuple[int, int, int, int]] = []
    for coords in _components4(mask):
        box = _bbox_coords(coords)
        sub = _crop_bbox(mask, box)
        assert np.all(sub), "each panel should be a solid rectangle"
        boxes.append(box)
    return boxes


def _panel_mask_and_color(panel: np.ndarray) -> tuple[np.ndarray, int]:
    fill = _get_bg(panel)
    colors = sorted(set(map(int, np.unique(panel))) - {fill})
    assert len(colors) == 1, "each panel should contain exactly one foreground color"
    return panel != fill, colors[0]


def _matching_ops(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> list[str]:
    return [name for name, func in OPS.items() if np.array_equal(func(a, b), out)]


def _combo_counts(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    return {
        "00": int((~a & ~b).sum()),
        "10": int((a & ~b).sum()),
        "01": int((~a & b).sum()),
        "11": int((a & b).sum()),
    }


def _all_truth_cases_present(a: np.ndarray, b: np.ndarray) -> bool:
    counts = _combo_counts(a, b)
    return all(counts[key] > 0 for key in ("00", "10", "01", "11"))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the inline demo to identify a boolean mask operation, then apply it to the main mask pair.

    Input structure:
    - The grid has a dominant background and one full-row or full-column divider line.
    - Away from the divider are five solid rectangular windows on a neutral fill color: two equal larger windows hold the main masks, and three equal smaller windows in one corner hold a demo as small-mask, small-mask, small-result.
    - Each mask window contains one foreground color painted over its local fill; the third demo window uses the output color that should appear in the answer.

    Rule:
    - Detect the three demo windows, extract their boolean masks relative to each window's local fill, and compare the two small inputs to the small result.
    - Among the candidate cellwise operations {intersection, union, xor, left-only, right-only}, keep the unique operation whose truth table matches the demo.
    - Remove the divider and window scaffolding conceptually, align the two large masks cell by cell, and apply that inferred operation.

    Key insight:
    - The transformation rule is not fixed across examples: the small triple is an explicit worked example that selects the boolean operator, and the large pair cannot be solved correctly without reading that demonstration first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    divider_orientation, divider_index, divider_color = _find_divider(g, bg)
    boxes = _window_boxes(g, bg, divider_color)

    by_size: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for box in boxes:
        h = box[2] - box[0] + 1
        w = box[3] - box[1] + 1
        by_size.setdefault((h, w), []).append(box)

    counts = sorted(len(group) for group in by_size.values())
    assert counts == [2, 3], "expected two main windows and three demo windows"

    main_boxes = next(group for group in by_size.values() if len(group) == 2)
    demo_boxes = next(group for group in by_size.values() if len(group) == 3)

    if divider_orientation == "vertical":
        main_boxes = sorted(main_boxes, key=lambda b: b[1])
        assert main_boxes[0][3] < divider_index < main_boxes[1][1], "main windows must straddle the divider"
    else:
        main_boxes = sorted(main_boxes, key=lambda b: b[0])
        assert main_boxes[0][2] < divider_index < main_boxes[1][0], "main windows must straddle the divider"

    demo_boxes = sorted(demo_boxes, key=lambda b: (b[1], b[0]))
    assert len({box[0] for box in demo_boxes}) == 1, "demo windows should be side by side"

    demo_a, _demo_a_color = _panel_mask_and_color(_crop_bbox(g, demo_boxes[0]))
    demo_b, _demo_b_color = _panel_mask_and_color(_crop_bbox(g, demo_boxes[1]))
    demo_out, result_color = _panel_mask_and_color(_crop_bbox(g, demo_boxes[2]))

    matches = _matching_ops(demo_a, demo_b, demo_out)
    assert len(matches) == 1, "demo must determine a unique boolean operation"
    op = OPS[matches[0]]

    main_a, _main_a_color = _panel_mask_and_color(_crop_bbox(g, main_boxes[0]))
    main_b, _main_b_color = _panel_mask_and_color(_crop_bbox(g, main_boxes[1]))
    assert main_a.shape == main_b.shape, "main windows must have the same size"

    out_mask = op(main_a, main_b)
    out = np.full(main_a.shape, bg, dtype=int)
    out[out_mask] = int(result_color)
    return out


def _random_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    min_cells = max(3, area // 6)
    max_cells = min((area - 1) // 2, max(min_cells, int(round(area * 0.42))))
    assert max_cells >= min_cells

    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        shape_count = random.randint(2, 4)
        for _ in range(shape_count):
            kind = random.choice(("rect", "rect", "hbar", "vbar", "walk", "point"))
            if kind == "rect":
                rh = random.randint(1, max(1, h - 1))
                rw = random.randint(1, max(1, w - 1))
                r0 = random.randint(0, h - rh)
                c0 = random.randint(0, w - rw)
                mask[r0 : r0 + rh, c0 : c0 + rw] = True
            elif kind == "hbar":
                row = random.randrange(h)
                c0 = random.randrange(w)
                c1 = random.randint(c0, w - 1)
                mask[row, c0 : c1 + 1] = True
            elif kind == "vbar":
                col = random.randrange(w)
                r0 = random.randrange(h)
                r1 = random.randint(r0, h - 1)
                mask[r0 : r1 + 1, col] = True
            elif kind == "walk":
                steps = random.randint(2, max(2, min(area // 3, 7)))
                rr = random.randrange(h)
                cc = random.randrange(w)
                mask[rr, cc] = True
                for _ in range(steps - 1):
                    dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                    rr = min(max(rr + dr, 0), h - 1)
                    cc = min(max(cc + dc, 0), w - 1)
                    mask[rr, cc] = True
            else:
                mask[random.randrange(h), random.randrange(w)] = True

        count = int(mask.sum())
        if not (min_cells <= count <= max_cells):
            continue
        rows = np.any(mask, axis=1).sum()
        cols = np.any(mask, axis=0).sum()
        if rows < 2 or cols < 2:
            continue
        return mask

    raise RuntimeError("failed to build a non-degenerate mask")


def _random_demo_pair(h: int, w: int, op_name: str) -> tuple[np.ndarray, np.ndarray]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    area = h * w
    max_fg = (area - 1) // 2
    assert max_fg >= 3

    for _ in range(200):
        random.shuffle(cells)
        a = np.zeros((h, w), dtype=bool)
        b = np.zeros((h, w), dtype=bool)

        both, a_only, b_only = cells[:3]
        a[both] = True
        b[both] = True
        a[a_only] = True
        b[b_only] = True

        extra_cells = cells[3 : 3 + random.randint(0, min(3, area - 4))]
        for r, c in extra_cells:
            combo = random.choice(("00", "10", "01", "11"))
            if combo in ("10", "11"):
                a[r, c] = True
            if combo in ("01", "11"):
                b[r, c] = True

        if not _all_truth_cases_present(a, b):
            continue
        if not (3 <= int(a.sum()) <= max_fg and 3 <= int(b.sum()) <= max_fg):
            continue
        out = OPS[op_name](a, b)
        if not (0 < int(out.sum()) <= max_fg):
            continue
        if len(_matching_ops(a, b, out)) != 1:
            continue
        return a, b

    raise RuntimeError("failed to build a distinguishing demo")


def _random_main_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        a = _random_mask(h, w)
        b = _random_mask(h, w)
        if not _all_truth_cases_present(a, b):
            continue
        outputs = {OPS[name](a, b).tobytes() for name in OPS}
        if len(outputs) != len(OPS):
            continue
        return a, b
    raise RuntimeError("failed to build a distinguishing main pair")


def _draw_panel(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    fill_color: int,
    fg_color: int,
) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w] = int(fill_color)
    grid[top : top + h, left : left + w][mask] = int(fg_color)


def _layout_boxes(
    main_shape: tuple[int, int], demo_shape: tuple[int, int], orientation: str, corner: str
) -> tuple[tuple[int, int], tuple[str, int], list[tuple[int, int]], list[tuple[int, int]]]:
    main_h, main_w = main_shape
    demo_h, demo_w = demo_shape
    strip_w = 3 * demo_w + 2

    if orientation == "vertical":
        demo_side = "left" if corner.endswith("l") else "right"
        demo_vertical = "top" if corner.startswith("t") else "bottom"

        left_capacity = max(main_w, strip_w if demo_side == "left" else 0)
        right_capacity = max(main_w, strip_w if demo_side == "right" else 0)
        left_half_w = left_capacity + 2
        right_half_w = right_capacity + 2

        top_extra = demo_h + 1 if demo_vertical == "top" else 0
        bottom_extra = demo_h + 1 if demo_vertical == "bottom" else 0
        total_h = top_extra + main_h + bottom_extra + 2
        total_w = left_half_w + 1 + right_half_w

        main_y = 1 + top_extra
        left_x = 1 + random.randint(0, left_capacity - main_w)
        right_x = left_half_w + 2 + random.randint(0, right_capacity - main_w)
        demo_y = 1 if demo_vertical == "top" else total_h - 1 - demo_h
        demo_x = 1 if demo_side == "left" else total_w - 1 - strip_w

        divider = ("vertical", left_half_w)
        main_panels = [(main_y, left_x), (main_y, right_x)]
    else:
        demo_side = "left" if corner.endswith("l") else "right"
        demo_vertical = "top" if corner.startswith("t") else "bottom"

        capacity_w = max(main_w, strip_w)
        top_above = demo_h + 1 if demo_vertical == "top" else 0
        bottom_below = demo_h + 1 if demo_vertical == "bottom" else 0
        top_half_h = top_above + main_h + 2
        bottom_half_h = main_h + bottom_below + 2
        total_h = top_half_h + 1 + bottom_half_h
        total_w = capacity_w + 2

        main_x = 1 + random.randint(0, capacity_w - main_w)
        top_y = 1 + top_above
        bottom_y = top_half_h + 2
        demo_y = 1 if demo_vertical == "top" else total_h - 1 - demo_h
        demo_x = 1 if demo_side == "left" else total_w - 1 - strip_w

        divider = ("horizontal", top_half_h)
        main_panels = [(top_y, main_x), (bottom_y, main_x)]

    demo_panels = [(demo_y, demo_x + idx * (demo_w + 1)) for idx in range(3)]
    assert total_h <= 30 and total_w <= 30
    return (total_h, total_w), divider, main_panels, demo_panels


def _expected_output(main_a: np.ndarray, main_b: np.ndarray, op_name: str, result_color: int) -> np.ndarray:
    out = np.full(main_a.shape, BACKGROUND, dtype=int)
    out[OPS[op_name](main_a, main_b)] = int(result_color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        op_name = random.choice(tuple(OPS))
        orientation = random.choice(("vertical", "horizontal"))
        corner = random.choice(("tl", "tr", "bl", "br"))

        main_shape = (random.randint(5, 8), random.randint(5, 8))
        demo_shape = (random.randint(3, 4), random.randint(3, 4))

        colors = random.sample(PALETTE, 5)
        divider_color, panel_fill, main_a_color, main_b_color, result_color = colors

        main_a, main_b = _random_main_pair(*main_shape)
        demo_a, demo_b = _random_demo_pair(*demo_shape, op_name)
        demo_out = OPS[op_name](demo_a, demo_b)

        if not _all_truth_cases_present(demo_a, demo_b):
            continue
        if not _all_truth_cases_present(main_a, main_b):
            continue
        if len({OPS[name](main_a, main_b).tobytes() for name in OPS}) != len(OPS):
            continue

        shape, divider, main_panels, demo_panels = _layout_boxes(main_shape, demo_shape, orientation, corner)
        grid = np.full(shape, BACKGROUND, dtype=int)

        if divider[0] == "vertical":
            grid[:, divider[1]] = divider_color
        else:
            grid[divider[1], :] = divider_color

        _draw_panel(grid, main_panels[0][0], main_panels[0][1], main_a, panel_fill, main_a_color)
        _draw_panel(grid, main_panels[1][0], main_panels[1][1], main_b, panel_fill, main_b_color)
        _draw_panel(grid, demo_panels[0][0], demo_panels[0][1], demo_a, panel_fill, main_a_color)
        _draw_panel(grid, demo_panels[1][0], demo_panels[1][1], demo_b, panel_fill, main_b_color)
        _draw_panel(grid, demo_panels[2][0], demo_panels[2][1], demo_out, panel_fill, result_color)

        expected = _expected_output(main_a, main_b, op_name, result_color)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid demo-selected boolean-op task")
