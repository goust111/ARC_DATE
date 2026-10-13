from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_035_2",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
components_mask = _ARC_UTILS.components_mask
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BOX_SIZE = 3
BOX_AREA = 8
BG = Color.BLACK
LARGEST_FILL = Color.GREY
SMALLEST_FILL = Color.RED
GEN_RETRY_LIMIT = 400

EDGE_BY_COLOR = {
    int(Color.GREEN): "left",
    int(Color.YELLOW): "down",
    int(Color.PINK): "up",
    int(Color.TEAL): "right",
}
BOX_COLORS = tuple(EDGE_BY_COLOR)


def _hollow_box(color: int) -> np.ndarray:
    sprite = np.full((BOX_SIZE, BOX_SIZE), int(color), dtype=int)
    sprite[1, 1] = BG
    return sprite


def _box_bbox(box: dict) -> tuple[int, int, int, int]:
    r0 = int(box["r0"])
    c0 = int(box["c0"])
    return r0, c0, r0 + BOX_SIZE - 1, c0 + BOX_SIZE - 1


def _boxes_conflict(box_a: dict, box_b: dict, gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = _box_bbox(box_a)
    br0, bc0, br1, bc1 = _box_bbox(box_b)
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _render_boxes(shape: tuple[int, int], boxes: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for box in boxes:
        r0 = int(box["r0"])
        c0 = int(box["c0"])
        sprite = _hollow_box(int(box["color"]))
        fg = sprite != BG
        target = occupied[r0 : r0 + BOX_SIZE, c0 : c0 + BOX_SIZE]
        assert not bool(target[fg].any()), "boxes must not overlap"
        target[fg] = True
        paste(out, sprite, r0, c0)
    return out


def _is_hollow_square(grid: np.ndarray, obj: dict, bg: int) -> bool:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    color = int(obj["color"])
    if int(obj["area"]) != BOX_AREA:
        return False
    if (r1 - r0 + 1, c1 - c0 + 1) != (BOX_SIZE, BOX_SIZE):
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return np.array_equal(patch, _hollow_box(color).astype(grid.dtype, copy=False))


def _extract_boxes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=BOX_COLORS,
        return_format="dict",
    )
    boxes = []
    for obj in objects:
        if _is_hollow_square(g, obj, bg):
            r0, c0, _r1, _c1 = map(int, obj["bbox"])
            boxes.append({"color": int(obj["color"]), "r0": r0, "c0": c0})

    assert 3 <= len(boxes) <= 4, "expected three or four hollow boxes"
    assert np.array_equal(_render_boxes(g.shape, boxes), g), "expected only hollow-box sprites"
    return boxes


def _move_box_to_edge(box: dict, shape: tuple[int, int]) -> dict:
    height, width = map(int, shape)
    color = int(box["color"])
    r0 = int(box["r0"])
    c0 = int(box["c0"])
    edge = EDGE_BY_COLOR[color]

    if edge == "left":
        return {"color": color, "r0": r0, "c0": 0}
    if edge == "right":
        return {"color": color, "r0": r0, "c0": width - BOX_SIZE}
    if edge == "up":
        return {"color": color, "r0": 0, "c0": c0}
    return {"color": color, "r0": height - BOX_SIZE, "c0": c0}


def _candidate_regions(grid: np.ndarray, boxes: list[dict]) -> list[dict]:
    hole_centers = {(int(box["r0"]) + 1, int(box["c0"]) + 1) for box in boxes}
    regions: list[dict] = []

    for coords in components_mask(grid == BG, connectivity=4):
        if coords.shape[0] == 1:
            only = tuple(map(int, coords[0]))
            # Ignore the trivial one-cell cavities trapped inside the hollow boxes.
            if only in hole_centers:
                continue
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        anchor = tuple(map(int, coords[order[0]]))
        regions.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "anchor": anchor,
            }
        )
    return regions


def _apply_checker_fill(out: np.ndarray, region: dict) -> None:
    ar, ac = map(int, region["anchor"])
    for r, c in region["coords"]:
        rr = int(r)
        cc = int(c)
        if (rr - ar + cc - ac) % 2 == 0:
            out[rr, cc] = LARGEST_FILL


def _apply_solid_fill(out: np.ndarray, region: dict, color: int) -> None:
    coords = region["coords"]
    out[coords[:, 0], coords[:, 1]] = int(color)


def _extreme_regions(regions: list[dict]) -> tuple[dict | None, dict | None]:
    if not regions:
        return None, None
    largest = min(
        regions,
        key=lambda region: (
            -int(region["area"]),
            int(region["anchor"][0]),
            int(region["anchor"][1]),
        ),
    )
    smallest = min(
        regions,
        key=lambda region: (
            int(region["area"]),
            int(region["anchor"][0]),
            int(region["anchor"][1]),
        ),
    )
    return largest, smallest


def _final_output(shape: tuple[int, int], boxes: list[dict]) -> np.ndarray:
    moved = [_move_box_to_edge(box, shape) for box in boxes]
    base = _render_boxes(shape, moved)
    regions = _candidate_regions(base, moved)
    largest, smallest = _extreme_regions(regions)

    out = base.copy()
    if smallest is not None and (
        largest is None or tuple(smallest["anchor"]) != tuple(largest["anchor"])
    ):
        _apply_solid_fill(out, smallest, SMALLEST_FILL)
    if largest is not None:
        _apply_checker_fill(out, largest)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the direction-coded hollow squares to the border, then fill the largest and smallest non-hole black regions differently.

    Input structure:
    - The background is black.
    - The foreground contains 3 or 4 non-overlapping 3x3 hollow-square sprites colored green, yellow, pink, and/or teal.
    - Every sprite sits away from the border, and each color encodes one slide direction.

    Rule:
    - Move every hollow square independently until its 3x3 bbox is flush with the border indicated by its color: green left, yellow down, pink up, teal right.
    - Erase the original locations and keep only the simultaneous final placements.
    - In that slid layout, find the 4-connected black canvas regions outside the one-cell box holes.
    - Fill the largest such region with a checkerboard anchored at its lexicographically first cell, painting even-offset cells grey and leaving odd-offset cells black.
    - Fill the smallest such region solid red; leave all other black regions and all box pixels unchanged.

    Key insight:
    - The colors only tell you where each box lands; the actual fill depends on the partition topology created jointly by all final edge placements, while the trivial one-cell holes inside the hollow boxes should be ignored for area ranking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    boxes = _extract_boxes(g)
    return _final_output(g.shape, boxes)


def _final_layout_is_valid(shape: tuple[int, int], boxes: list[dict]) -> bool:
    try:
        grid = _render_boxes(shape, boxes)
    except AssertionError:
        return False

    all_black_regions = components_mask(grid == BG, connectivity=4)
    ranked_regions = _candidate_regions(grid, boxes)
    if len(all_black_regions) < 4 or len(ranked_regions) < 2:
        return False

    areas = [int(region["area"]) for region in ranked_regions]
    return areas.count(max(areas)) == 1 and areas.count(min(areas)) == 1


def _rand_inclusive(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(int(lo), int(hi)))


def _sample_final_boxes(rng: random.Random, shape: tuple[int, int], colors: list[int]) -> list[dict]:
    height, width = map(int, shape)
    colors_set = set(map(int, colors))
    boxes: list[dict] = []

    have_nw_pair = Color.GREEN in colors_set and Color.PINK in colors_set
    have_se_pair = Color.YELLOW in colors_set and Color.TEAL in colors_set

    if have_nw_pair:
        left_r = _rand_inclusive(rng, 2, min(4, height - 5))
        top_c = _rand_inclusive(rng, 3, min(6, width - 5))
        boxes.append({"color": int(Color.GREEN), "r0": left_r, "c0": 0})
        boxes.append({"color": int(Color.PINK), "r0": 0, "c0": top_c})
    else:
        if Color.GREEN in colors_set:
            boxes.append(
                {
                    "color": int(Color.GREEN),
                    "r0": _rand_inclusive(rng, 2, min(4, height - 5)),
                    "c0": 0,
                }
            )
        if Color.PINK in colors_set:
            boxes.append(
                {
                    "color": int(Color.PINK),
                    "r0": 0,
                    "c0": _rand_inclusive(rng, 2, min(4, width - 5)),
                }
            )

    if have_se_pair:
        se_h = _rand_inclusive(rng, 3, min(5, height - 6))
        se_w = _rand_inclusive(rng, 3, min(5, width - 6))
        boxes.append(
            {
                "color": int(Color.TEAL),
                "r0": height - BOX_SIZE - se_h,
                "c0": width - BOX_SIZE,
            }
        )
        boxes.append(
            {
                "color": int(Color.YELLOW),
                "r0": height - BOX_SIZE,
                "c0": width - BOX_SIZE - se_w,
            }
        )
    else:
        if Color.TEAL in colors_set:
            boxes.append(
                {
                    "color": int(Color.TEAL),
                    "r0": _rand_inclusive(rng, max(4, height - 8), height - 6),
                    "c0": width - BOX_SIZE,
                }
            )
        if Color.YELLOW in colors_set:
            boxes.append(
                {
                    "color": int(Color.YELLOW),
                    "r0": height - BOX_SIZE,
                    "c0": _rand_inclusive(rng, max(4, width - 8), width - 6),
                }
            )

    return boxes


def _axis_candidates(box: dict, shape: tuple[int, int]) -> list[dict]:
    height, width = map(int, shape)
    color = int(box["color"])
    r0 = int(box["r0"])
    c0 = int(box["c0"])
    edge = EDGE_BY_COLOR[color]
    candidates: list[dict] = []

    if edge in {"left", "right"}:
        for nc0 in range(2, width - 4):
            candidates.append({"color": color, "r0": r0, "c0": int(nc0)})
    else:
        for nr0 in range(2, height - 4):
            candidates.append({"color": color, "r0": int(nr0), "c0": c0})
    return candidates


def _sample_initial_boxes(
    rng: random.Random,
    shape: tuple[int, int],
    final_boxes: list[dict],
) -> list[dict] | None:
    for _ in range(200):
        order = final_boxes[:]
        rng.shuffle(order)
        placed: list[dict] = []
        for final_box in order:
            candidates = _axis_candidates(final_box, shape)
            rng.shuffle(candidates)
            chosen = None
            for candidate in candidates:
                if any(_boxes_conflict(candidate, other, gap=1) for other in placed):
                    continue
                chosen = candidate
                break
            if chosen is None:
                break
            placed.append(chosen)
        if len(placed) == len(final_boxes):
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = _rand_inclusive(rng, 12, 18)
    width = _rand_inclusive(rng, 12, 18)
    shape = (height, width)

    box_count = 4 if rng.random() < 0.65 else 3
    colors = rng.sample(list(BOX_COLORS), box_count)
    final_boxes = _sample_final_boxes(rng, shape, colors)
    if not _final_layout_is_valid(shape, final_boxes):
        return None

    initial_boxes = _sample_initial_boxes(rng, shape, final_boxes)
    if initial_boxes is None:
        return None

    inp = _render_boxes(shape, initial_boxes)
    expected = _final_output(shape, initial_boxes)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
