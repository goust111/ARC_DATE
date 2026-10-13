from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_2_025_3",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


MIN_RECT_COUNT = 3
MAX_RECT_COUNT = 5
MIN_RECT_SIDE = 3
MAX_RECT_SIDE = 6
MIN_CANVAS_H = 20
MIN_CANVAS_W = 24
MAX_CANVAS = 30
ZONE_ROWS = 2
ZONE_COLS = 3
ZONE_MARGIN = 1
ZONE_GAP = 2
MAX_GENERATION_ATTEMPTS = 500
LAYOUT_TEMPLATES = (
    (2, 0, 4, 1, 3, 5),
    (3, 5, 1, 0, 4, 2),
    (1, 5, 0, 4, 2, 3),
)
_GEN_CALL_COUNT = 0


def _parse_rectangle(grid: np.ndarray, comp: dict, bg: int) -> dict:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    h, w = map(int, patch.shape)
    assert np.all(patch != bg), "rectangle bbox must be fully filled"

    colors, counts = np.unique(patch, return_counts=True)
    colors = [int(color) for color in colors]
    counts_by_color = {int(color): int(count) for color, count in zip(colors, counts)}
    assert bg not in counts_by_color, "rectangle must not contain background holes"
    assert 1 <= len(colors) <= 2, "each rectangle must have one body color and at most one marker color"

    if len(colors) == 1:
        body_color = colors[0]
        marker_color = None
        marker_pos = None
    else:
        marker_candidates = [color for color, count in counts_by_color.items() if count == 1]
        body_candidates = [color for color, count in counts_by_color.items() if count == h * w - 1]
        assert len(marker_candidates) == 1 and len(body_candidates) == 1, "marker must be exactly one pixel"
        marker_color = marker_candidates[0]
        body_color = body_candidates[0]
        marker_pos = tuple(map(int, np.argwhere(patch == marker_color)[0]))
        assert 0 < marker_pos[0] < h - 1 and 0 < marker_pos[1] < w - 1, "marker must be strictly inside the rectangle"

    return {
        "body_color": int(body_color),
        "marker_color": None if marker_color is None else int(marker_color),
        "marker_pos": marker_pos,
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "area": h * w,
    }


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert comps, "expected at least one rectangle"
    rects = [_parse_rectangle(grid, comp, bg) for comp in comps]

    body_colors = [rect["body_color"] for rect in rects]
    assert len(body_colors) == len(set(body_colors)), "body colors must be unique"
    marker_colors = [rect["marker_color"] for rect in rects if rect["marker_color"] is not None]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be unique"
    for rect in rects:
        marker_color = rect["marker_color"]
        if marker_color is not None:
            assert marker_color in body_colors, "each marker must point to another rectangle's body color"
            assert marker_color != rect["body_color"], "marker may not point to its own rectangle"
    return rects


def _recover_chain(rects: list[dict]) -> list[dict]:
    by_color = {rect["body_color"]: rect for rect in rects}
    incoming = {color: 0 for color in by_color}
    tails = []
    for rect in rects:
        marker_color = rect["marker_color"]
        if marker_color is None:
            tails.append(rect)
            continue
        incoming[marker_color] += 1
    assert all(count in (0, 1) for count in incoming.values()), "chain must have indegree <= 1 everywhere"
    heads = [color for color, count in incoming.items() if count == 0]
    assert len(heads) == 1, "chain must have exactly one head"
    assert len(tails) == 1, "chain must have exactly one tail"

    chain = []
    seen = set()
    color = heads[0]
    while True:
        assert color not in seen, "chain may not contain a cycle"
        rect = by_color[color]
        chain.append(rect)
        seen.add(color)
        if rect["marker_color"] is None:
            break
        color = rect["marker_color"]

    assert len(chain) == len(rects), "chain must cover every rectangle exactly once"
    assert chain[-1]["body_color"] == tails[0]["body_color"], "tail must be the markerless rectangle"
    return chain


def _render_chain(shape: tuple[int, int], bg: int, chain: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    row = 0
    col = 0
    for rect in chain:
        h, w = map(int, rect["shape"])
        body_color = int(rect["body_color"])
        assert row + h <= shape[0] and col + w <= shape[1], "diagonal chain must fit the canvas"
        patch = np.full((h, w), body_color, dtype=int)
        paste(out, patch, row, col)
        row += h - 1
        col += w - 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain scattered rectangles through their hidden color pointers, then restack them into an upper-left diagonal stair.

    Input structure:
    - The background is the most common color, and the foreground consists of separated filled rectangles with distinct body colors.
    - Every rectangle except one contains exactly one interior marker pixel of a different color; that marker color equals another rectangle's body color.
    - Marker colors are unique, and no rectangle points to itself.

    Rule:
    - Treat each marker color as the body color of the next rectangle, reconstructing the unique chain whose head is never referenced and whose tail has no marker.
    - Clear the whole canvas to background, erase the markers, and place solid rectangles in chain order starting with the head at the upper-left corner.
    - After placing a rectangle of height h and width w, place the next rectangle with its upper-left corner on the same cell as the previous rectangle's lower-right corner, so the later rectangle overwrites that shared corner.

    Key insight:
    - The visible positions of the rectangles are bait; the true ordering is encoded only by the single-pixel color pointers hidden inside the rectangle bodies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    chain = _recover_chain(_extract_rectangles(g))
    return _render_chain(g.shape, bg, chain)


def _partition_axis(length: int, parts: int) -> tuple[list[int], list[int]]:
    usable = length - 2 * ZONE_MARGIN - ZONE_GAP * (parts - 1)
    assert usable >= parts
    base, rem = divmod(usable, parts)
    sizes = [base + (1 if i < rem else 0) for i in range(parts)]
    starts = []
    pos = ZONE_MARGIN
    for size in sizes:
        starts.append(pos)
        pos += size + ZONE_GAP
    return sizes, starts


def _build_zones(height: int, width: int) -> list[dict]:
    row_sizes, row_starts = _partition_axis(height, ZONE_ROWS)
    col_sizes, col_starts = _partition_axis(width, ZONE_COLS)
    zones = []
    zone_id = 0
    for row_index, (row_size, r0) in enumerate(zip(row_sizes, row_starts)):
        for col_index, (col_size, c0) in enumerate(zip(col_sizes, col_starts)):
            zones.append(
                {
                    "id": zone_id,
                    "row_index": row_index,
                    "col_index": col_index,
                    "bbox": (r0, c0, r0 + row_size - 1, c0 + col_size - 1),
                    "shape": (row_size, col_size),
                }
            )
            zone_id += 1
    return zones


def _sample_shapes(n: int) -> list[tuple[int, int]] | None:
    shapes = []
    areas = []
    for _ in range(n):
        h = int(np.random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE + 1))
        w = int(np.random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE + 1))
        shapes.append((h, w))
        areas.append(h * w)
    if len(set(areas)) != n:
        return None
    order = tuple(range(n))
    area_asc = tuple(sorted(range(n), key=lambda idx: areas[idx]))
    if area_asc == order or area_asc[::-1] == order:
        return None
    return shapes


def _sample_body_colors(bg: int, n: int) -> list[int] | None:
    palette = [color for color in Color.ALL_COLORS if int(color) != int(bg)]
    colors = [int(color) for color in np.random.choice(palette, size=n, replace=False)]
    order = tuple(range(n))
    color_asc = tuple(sorted(range(n), key=lambda idx: colors[idx]))
    if color_asc == order or color_asc[::-1] == order:
        return None
    return colors


def _expected_extent(shapes: list[tuple[int, int]]) -> tuple[int, int]:
    out_h = 1 + sum(h - 1 for h, _ in shapes)
    out_w = 1 + sum(w - 1 for _, w in shapes)
    return int(out_h), int(out_w)


def _place_patch_in_zone(zone: dict, patch: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, zone["bbox"])
    h, w = map(int, patch.shape)
    max_r = r1 - h + 1
    max_c = c1 - w + 1
    assert r0 <= max_r and c0 <= max_c, "patch must fit inside its zone"
    row = int(np.random.randint(r0, max_r + 1))
    col = int(np.random.randint(c0, max_c + 1))
    return row, col


def _make_input_patch(shape: tuple[int, int], body_color: int, marker_color: int | None) -> tuple[np.ndarray, tuple[int, int] | None]:
    h, w = map(int, shape)
    patch = np.full((h, w), int(body_color), dtype=int)
    if marker_color is None:
        return patch, None
    marker_r = int(np.random.randint(1, h - 1))
    marker_c = int(np.random.randint(1, w - 1))
    patch[marker_r, marker_c] = int(marker_color)
    return patch, (marker_r, marker_c)


def _order_by_row_major(rects: list[dict]) -> tuple[int, ...]:
    return tuple(sorted(range(len(rects)), key=lambda idx: (rects[idx]["bbox"][0], rects[idx]["bbox"][1])))


def _order_by_left_edge(rects: list[dict]) -> tuple[int, ...]:
    return tuple(sorted(range(len(rects)), key=lambda idx: (rects[idx]["bbox"][1], rects[idx]["bbox"][0])))


def _breaks_shortcuts(rects: list[dict]) -> bool:
    true_order = tuple(range(len(rects)))
    row_major = _order_by_row_major(rects)
    left_edge = _order_by_left_edge(rects)
    if true_order in {row_major, row_major[::-1], left_edge, left_edge[::-1]}:
        return False

    colors = [int(rect["body_color"]) for rect in rects]
    color_order = tuple(sorted(range(len(rects)), key=lambda idx: colors[idx]))
    if true_order in {color_order, color_order[::-1]}:
        return False

    areas = [int(rect["area"]) for rect in rects]
    if len(set(areas)) == len(areas):
        area_order = tuple(sorted(range(len(rects)), key=lambda idx: areas[idx]))
        if true_order in {area_order, area_order[::-1]}:
            return False
    return True


def _passes_oracle(grid: np.ndarray, chain_specs: list[dict], expected: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=max(Color.ALL_COLORS) + 1)
    bg = int(get_bg(grid))
    if counts[bg] <= max(count for color, count in enumerate(counts) if color != bg):
        return False

    try:
        parsed_chain = _recover_chain(_extract_rectangles(grid))
        actual = main(grid)
    except Exception:
        return False

    expected_sig = [
        (int(rect["body_color"]), tuple(map(int, rect["shape"])), rect["marker_color"])
        for rect in chain_specs
    ]
    parsed_sig = [
        (int(rect["body_color"]), tuple(map(int, rect["shape"])), rect["marker_color"])
        for rect in parsed_chain
    ]
    if parsed_sig != expected_sig:
        return False
    if not _breaks_shortcuts(chain_specs):
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(mode: int) -> np.ndarray | None:
    n = int(np.random.randint(MIN_RECT_COUNT, MAX_RECT_COUNT + 1))
    shapes = _sample_shapes(n)
    if shapes is None:
        return None

    out_h, out_w = _expected_extent(shapes)
    min_h = max(MIN_CANVAS_H, out_h)
    min_w = max(MIN_CANVAS_W, out_w)
    if min_h > MAX_CANVAS or min_w > MAX_CANVAS:
        return None
    height = int(np.random.randint(min_h, MAX_CANVAS + 1))
    width = int(np.random.randint(min_w, MAX_CANVAS + 1))

    zones = _build_zones(height, width)
    min_zone_h = min(zone["shape"][0] for zone in zones)
    min_zone_w = min(zone["shape"][1] for zone in zones)
    if max(h for h, _ in shapes) > min_zone_h or max(w for _, w in shapes) > min_zone_w:
        return None

    bg = int(np.random.choice(Color.ALL_COLORS))
    body_colors = _sample_body_colors(bg, n)
    if body_colors is None:
        return None

    zone_ids = LAYOUT_TEMPLATES[mode % len(LAYOUT_TEMPLATES)][:n]
    grid = np.full((height, width), bg, dtype=int)
    chain_specs = []

    for idx, zone_id in enumerate(zone_ids):
        body_color = int(body_colors[idx])
        marker_color = None if idx == n - 1 else int(body_colors[idx + 1])
        patch, marker_pos = _make_input_patch(shapes[idx], body_color, marker_color)
        row, col = _place_patch_in_zone(zones[zone_id], patch)
        paste(grid, patch, row, col)
        h, w = map(int, patch.shape)
        chain_specs.append(
            {
                "body_color": body_color,
                "marker_color": marker_color,
                "marker_pos": marker_pos,
                "bbox": (row, col, row + h - 1, col + w - 1),
                "shape": (h, w),
                "area": h * w,
            }
        )

    expected = _render_chain(grid.shape, bg, chain_specs)
    return grid if _passes_oracle(grid, chain_specs, expected) else None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        mode = (_GEN_CALL_COUNT + attempt) % len(LAYOUT_TEMPLATES)
        grid = _gen_attempt(mode)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid linked-rectangle chain input")
