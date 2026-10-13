import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


ANCHOR_COLORS = (Color.RED, Color.GREEN, Color.YELLOW)
FILL_COLOR_POOL = (Color.BLUE, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
GENERATE_RETRIES = 400


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, box)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _bbox_intersection(a, b) is not None


def _paint_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _is_solid_rect(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == color))


def _corner_boxes(rect: tuple[int, int, int, int], size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, rect)
    h, w = map(int, size)
    return [
        (r0, c0, r0 + h - 1, c0 + w - 1),
        (r0, c1 - w + 1, r0 + h - 1, c1),
        (r1 - h + 1, c0, r1, c0 + w - 1),
        (r1 - h + 1, c1 - w + 1, r1, c1),
    ]


def _expand_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _parse_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert int(bg) == Color.BLACK
    h, w = g.shape
    assert h >= 4 and w >= 4

    legend: list[dict] = []
    seen_anchor_colors: set[int] = set()
    for row in range(3):
        anchor_color = int(g[row, 0])
        fill_color = int(g[row, 1])
        assert anchor_color != bg and fill_color != bg
        assert anchor_color not in seen_anchor_colors
        seen_anchor_colors.add(anchor_color)
        legend.append({"priority": row, "anchor_color": anchor_color, "fill_color": fill_color})

    work = g.copy()
    work[:3, :2] = bg

    specs: list[dict] = []
    for entry in legend:
        color = int(entry["anchor_color"])
        comps = [
            comp
            for comp in find_components(work, bg=bg, connectivity=4, monochromatic=True, colors=[color], return_format="dict")
            if _is_solid_rect(work, comp)
        ]
        assert len(comps) == 2, f"expected two solid anchors for color {color}"
        comps.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
        a, b = comps
        ah, aw = bbox_h(a["bbox"]), bbox_w(a["bbox"])
        bh, bw = bbox_h(b["bbox"]), bbox_w(b["bbox"])
        assert ah == bh and aw == bw
        ar0, ac0, ar1, ac1 = map(int, a["bbox"])
        br0, bc0, br1, bc1 = map(int, b["bbox"])
        assert ar0 < br0 and ac0 < bc0
        rect = (ar0, ac0, br1, bc1)
        specs.append(
            {
                "priority": int(entry["priority"]),
                "anchor_color": color,
                "fill_color": int(entry["fill_color"]),
                "size": (int(ah), int(aw)),
                "rect": rect,
                "input_corners": [a["bbox"], b["bbox"]],
            }
        )
    return specs


def _render_from_specs(shape: tuple[int, int], specs: list[dict], *, input_mode: bool) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    if input_mode:
        for spec in sorted(specs, key=lambda item: item["priority"]):
            row = int(spec["priority"])
            out[row, 0] = int(spec["anchor_color"])
            out[row, 1] = int(spec["fill_color"])
            for box in spec["input_corners"]:
                _paint_rect(out, box, int(spec["anchor_color"]))
        return out

    anchor_mask = np.zeros(shape, dtype=bool)
    for spec in specs:
        for box in _corner_boxes(spec["rect"], spec["size"]):
            r0, c0, r1, c1 = box
            out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["anchor_color"])
            anchor_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    for spec in sorted(specs, key=lambda item: item["priority"], reverse=True):
        fill_color = int(spec["fill_color"])
        r0, c0, r1, c1 = _expand_box(spec["rect"], shape)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = ~anchor_mask[r0 : r1 + 1, c0 : c1 + 1]
        patch[mask] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete diagonal anchor pairs into four-corner rectangles, then expand and layer their legend-driven fills.

    Input structure:
    - Black background with a 3x2 legend in the top-left corner.
    - Each legend row stores an anchor color in column 0 and that anchor's fill color in column 1.
    - For each legend anchor color, the grid contains exactly two same-color solid rectangles of the same size, placed at the upper-left and lower-right corners of an implied larger rectangle.

    Rule:
    - Erase the legend area to background, but keep each row's anchor->fill mapping and its row order.
    - For every same-color diagonal anchor pair, infer the implied large rectangle from the upper-left anchor's upper-left corner and the lower-right anchor's lower-right corner.
    - Copy the anchor rectangle into the other two corners of that implied rectangle using the same anchor color.
    - Expand each implied rectangle's bounding box by one cell in all four directions, clipped to the grid.
    - Fill every non-anchor cell inside each expanded box with that row's legend fill color; where multiple expanded boxes overlap, earlier legend rows override later ones, while anchor rectangles always stay on top.

    Key insight: The visible anchor pair only reveals two diagonal corners; the legend simultaneously supplies the fill color and the overlap priority needed to finish the completed rectangles and resolve their layered expansions.
    """
    g = np.asarray(grid, dtype=int)
    specs = _parse_specs(g)
    return _render_from_specs(g.shape, specs, input_mode=False)


def _sample_spaced_values(rng: random.Random, upper: int) -> list[int] | None:
    candidates = list(range(1, int(upper) + 1))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for value in candidates:
        if all(abs(value - old) >= 2 for old in chosen):
            chosen.append(int(value))
            if len(chosen) == 3:
                return sorted(chosen)
    return None


def _spec_has_visible_conflict(out: np.ndarray, specs: list[dict]) -> bool:
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            contenders = []
            anchor_here = False
            for spec in specs:
                if _bbox_overlaps((r, c, r, c), _expand_box(spec["rect"], out.shape)):
                    contenders.append(spec)
                for box in _corner_boxes(spec["rect"], spec["size"]):
                    if _bbox_overlaps((r, c, r, c), box):
                        anchor_here = True
                        break
                if anchor_here:
                    break
            if anchor_here or len(contenders) < 2:
                continue
            contenders.sort(key=lambda item: item["priority"])
            if int(out[r, c]) == int(contenders[0]["fill_color"]):
                return True
    return False


def _validate_specs(shape: tuple[int, int], specs: list[dict], out: np.ndarray) -> bool:
    corner_boxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        rect = spec["rect"]
        size = spec["size"]
        rect_area = _bbox_area(rect)
        if rect_area <= 4 * size[0] * size[1]:
            return False
        for box in _corner_boxes(rect, size):
            r0, c0, r1, c1 = box
            if not (0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]):
                return False
            corner_boxes.append(box)

    for i in range(len(corner_boxes)):
        for j in range(i + 1, len(corner_boxes)):
            if _bbox_overlaps(corner_boxes[i], corner_boxes[j]):
                return False

    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            inter = _bbox_intersection(specs[i]["rect"], specs[j]["rect"])
            if inter is None:
                return False
            inter_area = _bbox_area(inter)
            if inter_area == _bbox_area(specs[i]["rect"]) or inter_area == _bbox_area(specs[j]["rect"]):
                return False

    for spec in specs:
        anchor_pixels = 4 * int(spec["size"][0]) * int(spec["size"][1])
        if int(np.count_nonzero(out == int(spec["anchor_color"]))) != anchor_pixels:
            return False
        if int(np.count_nonzero(out == int(spec["fill_color"]))) == 0:
            return False

    if not _spec_has_visible_conflict(out, specs):
        return False
    return True


def _make_specs(rng: random.Random, shape: tuple[int, int]) -> list[dict] | None:
    h, w = map(int, shape)
    fill_colors = rng.sample(FILL_COLOR_POOL, 3)
    fill_by_anchor = {int(anchor): int(fill) for anchor, fill in zip(ANCHOR_COLORS, fill_colors)}
    row_order = rng.sample(list(ANCHOR_COLORS), len(ANCHOR_COLORS))
    priority_of = {int(anchor): row for row, anchor in enumerate(row_order)}
    sizes = {int(anchor): (rng.randint(2, 3), rng.randint(2, 3)) for anchor in ANCHOR_COLORS}

    side_cap = min(7, h - 14, w - 13)
    if side_cap < 4:
        return None
    top_vals = _sample_spaced_values(rng, side_cap)
    left_vals = _sample_spaced_values(rng, side_cap)
    bottom_vals = _sample_spaced_values(rng, side_cap)
    right_vals = _sample_spaced_values(rng, side_cap)
    if any(values is None for values in (top_vals, left_vals, bottom_vals, right_vals)):
        return None

    core_h = rng.randint(2, 4)
    core_w = rng.randint(2, 4)
    max_top = max(top_vals)
    max_left = max(left_vals)
    max_bottom = max(bottom_vals)
    max_right = max(right_vals)
    core_r0_min = 4 + max_top
    core_c0_min = 3 + max_left
    core_r0_max = h - core_h - 1 - max_bottom
    core_c0_max = w - core_w - 1 - max_right
    if core_r0_min > core_r0_max or core_c0_min > core_c0_max:
        return None

    core_r0 = rng.randint(core_r0_min, core_r0_max)
    core_c0 = rng.randint(core_c0_min, core_c0_max)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    top_perm = rng.sample(range(3), 3)
    left_perm = rng.sample(range(3), 3)
    bottom_perm = rng.sample(range(3), 3)
    right_perm = rng.sample(range(3), 3)

    touch_edge = rng.random() < 0.45
    touch_anchor = int(rng.choice(ANCHOR_COLORS)) if touch_edge else None
    touch_axis = rng.choice(("bottom", "right", "both")) if touch_edge else None

    specs: list[dict] = []
    for i, anchor_color in enumerate(ANCHOR_COLORS):
        anchor_color = int(anchor_color)
        ah, aw = map(int, sizes[anchor_color])
        r0 = core_r0 - top_vals[top_perm[i]]
        c0 = core_c0 - left_vals[left_perm[i]]
        r1 = core_r1 + bottom_vals[bottom_perm[i]]
        c1 = core_c1 + right_vals[right_perm[i]]
        if touch_anchor == anchor_color:
            if touch_axis in {"bottom", "both"}:
                r1 = h - 1
            if touch_axis in {"right", "both"}:
                c1 = w - 1
        if r1 - r0 + 1 < 2 * ah + 1 or c1 - c0 + 1 < 2 * aw + 1:
            return None
        specs.append(
            {
                "priority": int(priority_of[anchor_color]),
                "anchor_color": anchor_color,
                "fill_color": int(fill_by_anchor[anchor_color]),
                "size": (ah, aw),
                "rect": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    for spec in specs:
        corners = _corner_boxes(spec["rect"], spec["size"])
        spec["input_corners"] = [corners[0], corners[3]]
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(22, 29), rng.randint(22, 29))
    specs = _make_specs(rng, shape)
    if specs is None:
        return None

    inp = _render_from_specs(shape, specs, input_mode=True)
    expected = _render_from_specs(shape, specs, input_mode=False)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _validate_specs(shape, specs, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
