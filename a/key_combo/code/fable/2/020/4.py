from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PALETTE = tuple(range(1, 10))
ORIENTATIONS = ("TL", "TR", "BL", "BR")
SPRITE_SHAPES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
)
GEN_ATTEMPTS = 300


def _bbox_coords(coords: list[Coord] | np.ndarray) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_inside(inner: BBox, outer: BBox) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 <= ir0 <= ir1 <= or1 and oc0 <= ic0 <= ic1 <= oc1


def _interior_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return int(max(0, r1 - r0 - 1) * max(0, c1 - c0 - 1))


def _cells_for_bracket_bbox(bbox: BBox, orient: str) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    if orient == "TL":
        cells = {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    elif orient == "TR":
        cells = {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    elif orient == "BL":
        cells = {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    elif orient == "BR":
        cells = {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    else:
        raise ValueError(f"unknown bracket orientation: {orient}")
    return sorted((int(r), int(c)) for r, c in cells)


def _elbow_of(bbox: BBox, orient: str) -> Coord:
    r0, c0, r1, c1 = bbox
    if orient == "TL":
        return int(r0), int(c0)
    if orient == "TR":
        return int(r0), int(c1)
    if orient == "BL":
        return int(r1), int(c0)
    if orient == "BR":
        return int(r1), int(c1)
    raise ValueError(f"unknown bracket orientation: {orient}")


def _classify_bracket(component: dict) -> dict | None:
    coords = {tuple(map(int, p)) for p in component["coords"]}
    bbox = tuple(map(int, component["bbox"]))
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or int(component["area"]) != h + w - 1:
        return None
    for orient in ORIENTATIONS:
        cells = _cells_for_bracket_bbox(bbox, orient)
        if coords == set(cells):
            return {
                "color": int(component["color"]),
                "bbox": bbox,
                "coords": cells,
                "orient": orient,
                "elbow": _elbow_of(bbox, orient),
            }
    return None


def _region_from_pair(a: dict, b: dict) -> dict | None:
    if int(a["color"]) != int(b["color"]):
        return None

    pair = (a["orient"], b["orient"])
    if pair == ("TL", "BR"):
        top, left = a["elbow"]
        bottom, right = b["elbow"]
    elif pair == ("BR", "TL"):
        top, left = b["elbow"]
        bottom, right = a["elbow"]
    elif pair == ("TR", "BL"):
        top, right = a["elbow"]
        bottom, left = b["elbow"]
    elif pair == ("BL", "TR"):
        top, right = b["elbow"]
        bottom, left = a["elbow"]
    else:
        return None

    bbox = (int(top), int(left), int(bottom), int(right))
    if _interior_area(bbox) <= 0:
        return None
    if not _bbox_inside(a["bbox"], bbox) or not _bbox_inside(b["bbox"], bbox):
        return None
    return {
        "color": int(a["color"]),
        "bbox": bbox,
        "area": _interior_area(bbox),
        "brackets": (a["bbox"], b["bbox"]),
    }


def _inside_region(cells: list[Coord], bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    return all(r0 < r < r1 and c0 < c < c1 for r, c in cells)


def _intersects_region_interior(cells: list[Coord], bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 < r < r1 and c0 < c < c1 for r, c in cells)


def _winner_region(cells: list[Coord], regions: list[dict]) -> dict | None:
    containing = [region for region in regions if _inside_region(cells, region["bbox"])]
    if not containing:
        return None
    return min(
        containing,
        key=lambda region: (
            int(region["area"]),
            tuple(map(int, region["bbox"])),
            int(region["color"]),
        ),
    )


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    raw_brackets = find_components(g, bg=bg, connectivity=4, monochromatic=True, min_area=3)
    brackets = []
    bracket_mask = np.zeros_like(g, dtype=bool)
    for component in raw_brackets:
        bracket = _classify_bracket(component)
        if bracket is None:
            continue
        brackets.append(bracket)
        for r, c in bracket["coords"]:
            bracket_mask[r, c] = True

    region_map: dict[tuple[int, int, int, int, int], dict] = {}
    for i in range(len(brackets)):
        for j in range(i + 1, len(brackets)):
            region = _region_from_pair(brackets[i], brackets[j])
            if region is None:
                continue
            key = (int(region["color"]), *map(int, region["bbox"]))
            region_map[key] = region
    regions = sorted(
        region_map.values(),
        key=lambda region: (
            int(region["area"]),
            tuple(map(int, region["bbox"])),
            int(region["color"]),
        ),
    )

    sprite_mask = (g != bg) & (~bracket_mask)
    sprites = []
    for pts in components_mask(sprite_mask, connectivity=8):
        cells = [tuple(map(int, p)) for p in pts]
        bbox = _bbox_coords(cells)
        containing = [region for region in regions if _inside_region(cells, region["bbox"])]
        winner = _winner_region(cells, regions)
        partial = winner is None and any(_intersects_region_interior(cells, region["bbox"]) for region in regions)
        sprites.append(
            {
                "coords": cells,
                "bbox": bbox,
                "containing": containing,
                "winner": winner,
                "partial": bool(partial),
            }
        )

    return {
        "bg": bg,
        "brackets": brackets,
        "regions": regions,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only sprites fully enclosed by invisible rectangles implied by diagonal-facing same-color corner brackets.

    Input structure:
    - The background is black, with several monochrome L-shaped corner brackets and several small sprite objects.
    - A valid hidden selection comes from two same-color brackets on opposite rectangle corners whose openings face each other.
    - Other brackets are distractors, and different hidden rectangles may overlap.

    Rule:
    - Find every same-color bracket pair that forms an axis-aligned invisible rectangle.
    - Any sprite fully inside no rectangle is erased; a sprite fully inside one rectangle is recolored to that rectangle's bracket color.
    - If a sprite lies fully inside multiple overlapping rectangles, recolor it with the smallest rectangle's color.
    - Leave all brackets unchanged and never draw the rectangle borders themselves.

    Key insight:
    - The brackets are the only generators of the hidden regions: geometry and facing determine the invisible rectangles, and those rectangles alone decide which sprites survive and what color they become.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    analysis = _analyze_grid(g)
    bg = int(analysis["bg"])
    out = np.full_like(g, bg)

    for bracket in analysis["brackets"]:
        for r, c in bracket["coords"]:
            out[r, c] = int(bracket["color"])

    for sprite in analysis["sprites"]:
        winner = sprite["winner"]
        if winner is None:
            continue
        color = int(winner["color"])
        for r, c in sprite["coords"]:
            out[r, c] = color

    return out


def _expand_cells(cells: list[Coord], shape: tuple[int, int], pad: int = 1) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _place_cells(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _render_oracle(shape: tuple[int, int], brackets: list[dict], sprites: list[dict], regions: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for bracket in brackets:
        _place_cells(out, bracket["coords"], bracket["color"])
    for sprite in sprites:
        winner = _winner_region(sprite["coords"], regions)
        if winner is None:
            continue
        _place_cells(out, sprite["coords"], winner["color"])
    return out


def _random_region_pair_type() -> str:
    return random.choice(("TLBR", "TRBL"))


def _build_region_brackets(region: dict) -> list[dict]:
    top, left, bottom, right = region["bbox"]
    height = bottom - top + 1
    width = right - left + 1
    arm_h_max = min(4, height - 1)
    arm_w_max = min(4, width - 1)
    h1 = random.randint(2, arm_h_max)
    w1 = random.randint(2, arm_w_max)
    h2 = random.randint(2, arm_h_max)
    w2 = random.randint(2, arm_w_max)

    if region["pair_type"] == "TLBR":
        specs = [
            {"color": region["color"], "bbox": (top, left, top + h1 - 1, left + w1 - 1), "orient": "TL"},
            {"color": region["color"], "bbox": (bottom - h2 + 1, right - w2 + 1, bottom, right), "orient": "BR"},
        ]
    else:
        specs = [
            {"color": region["color"], "bbox": (top, right - w1 + 1, top + h1 - 1, right), "orient": "TR"},
            {"color": region["color"], "bbox": (bottom - h2 + 1, left, bottom, left + w2 - 1), "orient": "BL"},
        ]

    for spec in specs:
        spec["coords"] = _cells_for_bracket_bbox(spec["bbox"], spec["orient"])
    return specs


def _sample_regions(h: int, w: int, include_third: bool) -> list[dict] | None:
    for _ in range(80):
        top1 = random.randint(1, 3)
        left1 = random.randint(1, 3)
        right1 = random.randint(w - 11, w - 8)
        bottom1 = random.randint(h - 5, h - 3)
        if bottom1 - top1 < 12 or right1 - left1 < 11:
            continue

        top2 = random.randint(top1 + 4, top1 + 6)
        left2 = random.randint(left1 + 4, left1 + 7)
        bottom2 = random.randint(bottom1 - 6, bottom1 - 4)
        right2 = random.randint(right1 - 7, right1 - 4)
        if bottom2 - top2 < 6 or right2 - left2 < 6:
            continue

        regions = [
            {"bbox": (top1, left1, bottom1, right1), "pair_type": _random_region_pair_type()},
            {"bbox": (top2, left2, bottom2, right2), "pair_type": _random_region_pair_type()},
        ]

        if include_third:
            left3_min = right1 + 3
            if left3_min > w - 6:
                continue
            left3 = random.randint(left3_min, w - 6)
            right3 = random.randint(left3 + 4, w - 2)
            top3 = random.randint(2, max(2, h // 2))
            bottom3 = random.randint(top3 + 5, min(h - 2, top3 + 9))
            if bottom3 - top3 < 5 or right3 - left3 < 4:
                continue
            regions.append({"bbox": (top3, left3, bottom3, right3), "pair_type": _random_region_pair_type()})

        return regions
    return None


def _try_add_bracket(
    shape: tuple[int, int],
    blocked: set[Coord],
    color: int,
    orient: str,
    size: tuple[int, int],
) -> dict | None:
    h, w = shape
    bh, bw = size
    positions = [(r0, c0) for r0 in range(h - bh + 1) for c0 in range(w - bw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        cells = _cells_for_bracket_bbox(bbox, orient)
        if any(cell in blocked for cell in cells):
            continue
        return {"color": int(color), "bbox": bbox, "orient": orient, "coords": cells}
    return None


def _sprite_cells(mask: np.ndarray, r0: int, c0: int) -> list[Coord]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _search_sprite(
    shape: tuple[int, int],
    blocked: set[Coord],
    regions: list[dict],
    predicate,
    *,
    shape_choices: tuple[np.ndarray, ...] = SPRITE_SHAPES,
) -> list[Coord] | None:
    h, w = shape
    masks = list(shape_choices)
    random.shuffle(masks)
    for mask in masks:
        mh, mw = mask.shape
        positions = [(r0, c0) for r0 in range(h - mh + 1) for c0 in range(w - mw + 1)]
        random.shuffle(positions)
        for r0, c0 in positions:
            cells = _sprite_cells(mask, r0, c0)
            if any(cell in blocked for cell in cells):
                continue
            if predicate(cells, regions):
                return cells
    return None


def _contains_exactly(cells: list[Coord], regions: list[dict], target_bbox: BBox) -> bool:
    containing = [region["bbox"] for region in regions if _inside_region(cells, region["bbox"])]
    return containing == [target_bbox]


def _overlap_candidate(cells: list[Coord], regions: list[dict], required_bboxes: set[BBox]) -> bool:
    containing = {region["bbox"] for region in regions if _inside_region(cells, region["bbox"])}
    return required_bboxes.issubset(containing) and len(containing) >= 2


def _partial_candidate(cells: list[Coord], regions: list[dict]) -> bool:
    if any(_inside_region(cells, region["bbox"]) for region in regions):
        return False
    return any(_intersects_region_interior(cells, region["bbox"]) for region in regions)


def _outside_candidate(cells: list[Coord], regions: list[dict]) -> bool:
    return not any(_intersects_region_interior(cells, region["bbox"]) for region in regions)


def _intended_region_key(region: dict) -> tuple[int, int, int, int, int]:
    return (int(region["color"]), *map(int, region["bbox"]))


def _valid_categories(sprites: list[dict]) -> bool:
    has_single = any(sprite["winner"] is not None and len(sprite["containing"]) == 1 for sprite in sprites)
    has_multi = any(sprite["winner"] is not None and len(sprite["containing"]) >= 2 for sprite in sprites)
    has_partial = any(sprite["partial"] for sprite in sprites)
    has_outside = any(sprite["winner"] is None and not sprite["partial"] for sprite in sprites)
    return has_single and has_multi and has_partial and has_outside


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 29)
    w = random.randint(24, 29)
    include_third = random.random() < 0.7
    regions = _sample_regions(h, w, include_third)
    if regions is None:
        return None

    needed_colors = len(regions) + 4 + (1 if include_third else 0)
    if needed_colors > len(PALETTE):
        return None
    colors = random.sample(PALETTE, needed_colors)

    for region, color in zip(regions, colors):
        region["color"] = int(color)
        region["area"] = _interior_area(region["bbox"])

    region_brackets: list[dict] = []
    blocked: set[Coord] = set()
    for region in regions:
        for bracket in _build_region_brackets(region):
            if any(cell in blocked for cell in bracket["coords"]):
                return None
            region_brackets.append(bracket)
            blocked |= _expand_cells(bracket["coords"], (h, w), pad=1)

    distractor_target = random.choice(regions)
    if distractor_target["pair_type"] == "TLBR":
        distractor_orient = random.choice(("TR", "BL"))
    else:
        distractor_orient = random.choice(("TL", "BR"))
    distractor_size = (random.randint(2, 4), random.randint(2, 4))
    distractor = _try_add_bracket((h, w), blocked, distractor_target["color"], distractor_orient, distractor_size)
    if distractor is None:
        return None
    distractor_brackets = [distractor]
    blocked |= _expand_cells(distractor["coords"], (h, w), pad=1)

    extra_distractor = None
    if random.random() < 0.5:
        remaining = [c for c in PALETTE if c not in {region["color"] for region in regions}]
        if remaining:
            extra_distractor = _try_add_bracket(
                (h, w),
                blocked,
                random.choice(remaining),
                random.choice(ORIENTATIONS),
                (random.randint(2, 4), random.randint(2, 4)),
            )
            if extra_distractor is not None:
                distractor_brackets.append(extra_distractor)
                blocked |= _expand_cells(extra_distractor["coords"], (h, w), pad=1)

    big_bbox = regions[0]["bbox"]
    small_bbox = regions[1]["bbox"]
    third_bbox = regions[2]["bbox"] if include_third and len(regions) == 3 else None

    sprite_colors = colors[len(regions) :]
    sprites: list[dict] = []

    overlap_cells = _search_sprite(
        (h, w),
        blocked,
        regions,
        lambda cells, rs: _overlap_candidate(cells, rs, {big_bbox, small_bbox}),
        shape_choices=SPRITE_SHAPES[1:],
    )
    if overlap_cells is None:
        return None
    sprites.append({"coords": overlap_cells, "color": sprite_colors[0], "tag": "overlap"})
    blocked |= _expand_cells(overlap_cells, (h, w), pad=1)

    single_target = third_bbox if third_bbox is not None and random.random() < 0.5 else big_bbox
    single_cells = _search_sprite(
        (h, w),
        blocked,
        regions,
        lambda cells, rs: _contains_exactly(cells, rs, single_target),
    )
    if single_cells is None and single_target != big_bbox:
        single_target = big_bbox
        single_cells = _search_sprite(
            (h, w),
            blocked,
            regions,
            lambda cells, rs: _contains_exactly(cells, rs, single_target),
        )
    if single_cells is None:
        return None
    sprites.append({"coords": single_cells, "color": sprite_colors[1], "tag": "single"})
    blocked |= _expand_cells(single_cells, (h, w), pad=1)

    partial_cells = _search_sprite(
        (h, w),
        blocked,
        regions,
        _partial_candidate,
        shape_choices=SPRITE_SHAPES[:4],
    )
    if partial_cells is None:
        return None
    sprites.append({"coords": partial_cells, "color": sprite_colors[2], "tag": "partial"})
    blocked |= _expand_cells(partial_cells, (h, w), pad=1)

    outside_cells = _search_sprite(
        (h, w),
        blocked,
        regions,
        _outside_candidate,
    )
    if outside_cells is None:
        return None
    sprites.append({"coords": outside_cells, "color": sprite_colors[3], "tag": "outside"})
    blocked |= _expand_cells(outside_cells, (h, w), pad=1)

    if include_third and len(sprite_colors) > 4 and random.random() < 0.6:
        extra_single = _search_sprite(
            (h, w),
            blocked,
            regions,
            lambda cells, rs: _contains_exactly(cells, rs, third_bbox),
        )
        if extra_single is not None:
            sprites.append({"coords": extra_single, "color": sprite_colors[4], "tag": "single_extra"})
            blocked |= _expand_cells(extra_single, (h, w), pad=1)

    grid = np.full((h, w), BG, dtype=int)
    all_brackets = region_brackets + distractor_brackets
    for bracket in all_brackets:
        _place_cells(grid, bracket["coords"], bracket["color"])
    for sprite in sprites:
        _place_cells(grid, sprite["coords"], sprite["color"])

    intended_region_keys = {_intended_region_key(region) for region in regions}
    oracle = _render_oracle((h, w), all_brackets, sprites, regions)
    actual = _analyze_grid(grid)

    actual_region_keys = {_intended_region_key(region) for region in actual["regions"]}
    if actual_region_keys != intended_region_keys:
        return None
    if len(actual["brackets"]) != len(all_brackets):
        return None
    if len(actual["sprites"]) != len(sprites):
        return None
    if not _valid_categories(actual["sprites"]):
        return None
    if not any(
        sprite["winner"] is not None
        and len(sprite["containing"]) >= 2
        and int(sprite["winner"]["color"]) == int(regions[1]["color"])
        for sprite in actual["sprites"]
    ):
        return None
    if not np.array_equal(main(grid), oracle):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for hidden-rectangle ARC task")
