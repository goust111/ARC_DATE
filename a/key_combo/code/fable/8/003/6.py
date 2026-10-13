import random
from collections import Counter

import numpy as np


ARC_BG = 0
ARC_COLORS = tuple(range(1, 10))
MAX_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_from_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"missing color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_rectangles(
    shape: tuple[int, int],
    rectangles: list[dict],
    order: list[int],
    *,
    bg: int = ARC_BG,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    by_color = {int(rect["color"]): rect for rect in rectangles}
    for color in order:
        _paint_rect(out, by_color[int(color)]["bbox"], int(color))
    return out


def _visible_diagonal_pair(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return (
        int(grid[r0, c0]) == int(color) and int(grid[r1, c1]) == int(color)
    ) or (
        int(grid[r0, c1]) == int(color) and int(grid[r1, c0]) == int(color)
    )


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return 0
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _scene_stats_from_input(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    stats = []
    for color in _colors_present(grid, exclude={bg}):
        bbox = _bbox_from_color(grid, color)
        visible = int(np.count_nonzero(grid == int(color)))
        full_area = _bbox_area(bbox)
        hidden = full_area - visible
        stats.append(
            {
                "color": int(color),
                "bbox": bbox,
                "visible": visible,
                "full_area": full_area,
                "hidden": hidden,
            }
        )
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially occluded solid rectangle, then restack them so more-hidden rectangles rise higher.

    Input structure:
    - The grid has a black background and several differently colored solid rectangles that were drawn with overlap.
    - Each color belongs to exactly one rectangle, and each rectangle still shows a diagonal pair of corners.
    - Upper rectangles hide parts of lower rectangles, so the visible same-color cells may be only fragments of the full rectangle.

    Rule:
    - For each non-background color, take the bounding box of its visible cells as the rectangle's full extent.
    - Compute hidden area = full rectangle area minus the number of currently visible cells of that color.
    - Redraw the recovered full rectangles on the same canvas from smaller hidden area to larger hidden area.
    - Keep every rectangle's position, size, and color unchanged; only the ownership of overlap cells changes.

    Key insight:
    - The new stacking order is determined by an invisible derived quantity, how much of each rectangle was hidden, so solving requires both amodal rectangle completion and ordering by that recovered loss.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected multiple rectangles"

    rectangles = []
    hidden_values = []
    for color in colors:
        bbox = _bbox_from_color(g, color)
        visible = int(np.count_nonzero(g == int(color)))
        full_area = _bbox_area(bbox)
        hidden = full_area - visible
        assert hidden >= 0
        assert _visible_diagonal_pair(g, color, bbox), "each rectangle must expose one diagonal pair of corners"
        rectangles.append({"color": int(color), "bbox": bbox, "hidden": hidden})
        hidden_values.append(hidden)

    assert len(set(hidden_values)) == len(hidden_values), "hidden areas must be unique"

    order = [rect["color"] for rect in sorted(rectangles, key=lambda rect: (rect["hidden"], rect["color"]))]
    return _render_rectangles(g.shape, rectangles, order, bg=bg)


def _rect_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _bbox_union(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _sample_scene(shape: tuple[int, int], order: list[int]) -> list[dict]:
    h, w = shape
    top_to_bottom = list(reversed(order))
    rectangles: list[dict] = []
    seen = set()
    upper_masks: list[np.ndarray] = []

    for index, color in enumerate(top_to_bottom):
        for _ in range(240):
            rect_h = random.randint(5, min(11, h - 4))
            rect_w = random.randint(5, min(11, w - 4))

            if index == 0:
                r0 = random.randint(2, h - rect_h - 2)
                c0 = random.randint(2, w - rect_w - 2)
            else:
                upper_bboxes = [rect["bbox"] for rect in rectangles]
                ur0, uc0, ur1, uc1 = _bbox_union(upper_bboxes)
                r_lo = max(1, ur0 - rect_h + 2)
                r_hi = min(h - rect_h - 1, ur1 - 1)
                c_lo = max(1, uc0 - rect_w + 2)
                c_hi = min(w - rect_w - 1, uc1 - 1)
                if r_lo > r_hi or c_lo > c_hi:
                    continue
                r0 = random.randint(r_lo, r_hi)
                c0 = random.randint(c_lo, c_hi)

            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if bbox in seen or _bbox_area(bbox) > 96:
                continue

            mask = _rect_mask(shape, bbox)
            if index > 0:
                upper_union = np.any(np.stack(upper_masks, axis=0), axis=0)
                hidden_area = int(np.count_nonzero(mask & upper_union))
                visible_area = int(mask.sum()) - hidden_area
                if hidden_area < 2 or visible_area < 8:
                    continue

                corners = [
                    ((bbox[0], bbox[1]), (bbox[2], bbox[3])),
                    ((bbox[0], bbox[3]), (bbox[2], bbox[1])),
                ]
                if not any((not upper_union[a] and not upper_union[b]) for a, b in corners):
                    continue

            rectangles.append({"color": int(color), "bbox": bbox})
            seen.add(bbox)
            upper_masks.append(mask)
            break
        else:
            raise AssertionError("failed to sample a valid stacked scene")

    return rectangles


def _hidden_order(stats: list[dict]) -> list[int]:
    return [item["color"] for item in sorted(stats, key=lambda item: (item["hidden"], item["color"]))]


def _order_by_key(stats: list[dict], key: str, reverse: bool = False) -> list[int]:
    return [
        item["color"]
        for item in sorted(
            stats,
            key=lambda item: (item[key], item["color"]),
            reverse=bool(reverse),
        )
    ]


def _output_for_scene(shape: tuple[int, int], stats: list[dict], order: list[int]) -> np.ndarray:
    rectangles = [{"color": item["color"], "bbox": item["bbox"]} for item in stats]
    return _render_rectangles(shape, rectangles, order, bg=ARC_BG)


def _validate_scene(rectangles: list[dict], order: list[int], inp: np.ndarray) -> bool:
    if int(np.count_nonzero(inp == ARC_BG)) <= 0:
        return False
    if int(np.count_nonzero(inp == ARC_BG)) <= max(int(np.count_nonzero(inp == rect["color"])) for rect in rectangles):
        return False

    stats = _scene_stats_from_input(inp)
    if len(stats) != len(rectangles):
        return False

    original_bboxes = {int(rect["color"]): rect["bbox"] for rect in rectangles}
    for item in stats:
        if item["bbox"] != original_bboxes[item["color"]]:
            return False
        if item["hidden"] == 0:
            continue
        if not _visible_diagonal_pair(inp, item["color"], item["bbox"]):
            return False

    hidden_values = [item["hidden"] for item in stats]
    if len(set(hidden_values)) != len(hidden_values):
        return False
    if sum(value > 0 for value in hidden_values) < 2:
        return False

    pair_overlaps = {
        item["color"]: 0
        for item in stats
    }
    for i, left in enumerate(rectangles):
        for right in rectangles[i + 1 :]:
            area = _intersection_area(left["bbox"], right["bbox"])
            if area > 0:
                pair_overlaps[left["color"]] += 1
                pair_overlaps[right["color"]] += 1
    if any(count == 0 for count in pair_overlaps.values()):
        return False

    target_order = _hidden_order(stats)
    if target_order == order or target_order == list(reversed(order)):
        return False
    if target_order == _order_by_key(stats, "full_area"):
        return False
    if target_order == _order_by_key(stats, "visible"):
        return False

    target = _output_for_scene(inp.shape, stats, target_order)
    if np.array_equal(target, inp):
        return False

    alt_orders = [
        order,
        list(reversed(order)),
        _order_by_key(stats, "full_area"),
        _order_by_key(stats, "full_area", reverse=True),
        _order_by_key(stats, "visible"),
        _order_by_key(stats, "visible", reverse=True),
    ]
    for alt in alt_orders:
        if alt == target_order:
            continue
        if np.array_equal(target, _output_for_scene(inp.shape, stats, alt)):
            return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, target)


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(15, 24), random.randint(15, 24))
    count = random.choice([3, 4])
    order = random.sample(random.sample(ARC_COLORS, count), count)
    try:
        rectangles = _sample_scene(shape, order)
    except AssertionError:
        return None
    inp = _render_rectangles(shape, rectangles, order, bg=ARC_BG)
    if not _validate_scene(rectangles, order, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
