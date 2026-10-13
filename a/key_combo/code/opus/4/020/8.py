from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, find_components, get_bg, neighbors4


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

GRID_HEIGHTS = (10, 12, 14)
GRID_WIDTH_RANGE = (10, 14)
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY)
MAX_GEN_ATTEMPTS = 500
MAX_LOCAL_ATTEMPTS = 60
MAX_PLACE_ATTEMPTS = 80


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the foreground object with the largest row-flip mismatch count.

    Input structure:
    - The grid has a black background and 3 to 5 separated single-color foreground objects.
    - Each foreground object is a 4-connected irregular shape with a unique color.
    - Objects may sit near or far from the horizontal midline, so some overlap their own row-flipped copy more than others.

    Rule:
    - Flip the whole grid vertically by reversing the row order.
    - Mark every cell where the original grid and the flipped grid differ.
    - For each foreground object in the original grid, count how many of that object's pixels lie on marked mismatch cells.
    - Select the unique object with the largest mismatch count and return its tight bounding-box crop.

    Key insight:
    - The target is defined by global vertical asymmetry with respect to the canvas, so it can only be identified by comparing the original grid to its row-flipped version before scoring each object.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected at least one foreground component"

    diff = g != np.flipud(g)
    scores = []
    for comp in components:
        coords = comp["coords"]
        score = int(diff[coords[:, 0], coords[:, 1]].sum())
        scores.append(score)

    best = max(scores)
    winners = [comp for comp, score in zip(components, scores) if score == best]
    assert len(winners) == 1, "expected a unique maximum mismatch count"
    return crop_bbox(g, winners[0]["bbox"])


def _coords_bbox(coords: set[Coord]) -> BBox:
    arr = np.array(sorted(coords), dtype=int)
    return bbox_coords(arr)


def _normalize_coords(coords: set[Coord]) -> set[Coord]:
    r0, c0, _r1, _c1 = _coords_bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _normalize_many(*coord_sets: set[Coord]) -> list[set[Coord]]:
    union = set().union(*coord_sets)
    r0, c0, _r1, _c1 = _coords_bbox(union)
    return [{(r - r0, c - c0) for r, c in coords} for coords in coord_sets]


def _bbox_dims(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _boxes_separated(a: BBox, b: BBox, gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _can_place(coords: set[Coord], existing_boxes: list[BBox], h: int, w: int) -> bool:
    if not coords:
        return False
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
    box = _coords_bbox(coords)
    return all(_boxes_separated(box, other) for other in existing_boxes)


def _grow_connected(
    seed_cells: set[Coord],
    allowed: set[Coord],
    target_size: int,
    rng: np.random.Generator,
) -> set[Coord] | None:
    cells = set(seed_cells)
    if not cells or not cells.issubset(allowed) or len(cells) > int(target_size):
        return None

    while len(cells) < int(target_size):
        frontier: set[Coord] = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                nbr = (int(nr), int(nc))
                if nbr in allowed and nbr not in cells:
                    frontier.add(nbr)
        if not frontier:
            return None
        picks = sorted(frontier)
        cells.add(picks[int(rng.integers(len(picks)))])
    return cells


def _make_connected_local(
    area: int,
    rng: np.random.Generator,
    *,
    depth_range: range,
    width_range: range,
    seed_bottom: bool,
) -> set[Coord] | None:
    dims = [(depth, width) for depth in depth_range for width in width_range if depth * width >= int(area)]
    if not dims:
        return None

    for idx in rng.permutation(len(dims)):
        depth, width = dims[int(idx)]
        allowed = {(r, c) for r in range(depth) for c in range(width)}
        for _ in range(MAX_LOCAL_ATTEMPTS):
            seed_r = depth - 1 if seed_bottom else int(rng.integers(depth))
            seed_c = int(rng.integers(width))
            coords = _grow_connected({(seed_r, seed_c)}, allowed, area, rng)
            if coords is None:
                continue
            coords = _normalize_coords(coords)
            box = _coords_bbox(coords)
            box_h, box_w = _bbox_dims(box)
            if box_h < 2 or box_w < 2:
                continue
            return coords
    return None


def _make_axis_asym_local(
    base_size: int,
    extra_size: int,
    rng: np.random.Generator,
) -> tuple[set[Coord], set[Coord]] | None:
    total_top = int(base_size + extra_size)
    dims = [(depth, width) for depth in range(2, 4) for width in range(2, 5) if depth * width >= total_top]
    if not dims:
        return None

    for idx in rng.permutation(len(dims)):
        depth, width = dims[int(idx)]
        allowed = {(r, c) for r in range(depth) for c in range(width)}
        for _ in range(MAX_LOCAL_ATTEMPTS):
            seed = {(depth - 1, int(rng.integers(width)))}
            base = _grow_connected(seed, allowed, base_size, rng)
            if base is None:
                continue
            top_all = _grow_connected(base, allowed, total_top, rng)
            if top_all is None:
                continue
            base_norm, top_norm = _normalize_many(base, top_all)
            extras = top_norm - base_norm
            box = _coords_bbox(top_norm)
            box_h, box_w = _bbox_dims(box)
            if box_h < 2 or box_w < 2 or not extras:
                continue
            return base_norm, extras
    return None


def _local_to_global(local_coords: set[Coord], row0: int, col0: int) -> set[Coord]:
    return {(row0 + r, col0 + c) for r, c in local_coords}


def _place_axis_symmetric(
    existing_boxes: list[BBox],
    h: int,
    w: int,
    top_area: int,
    rng: np.random.Generator,
) -> dict | None:
    for _ in range(MAX_PLACE_ATTEMPTS):
        top_local = _make_connected_local(
            top_area,
            rng,
            depth_range=range(1, 4),
            width_range=range(2, 5),
            seed_bottom=True,
        )
        if top_local is None:
            continue
        _top_h, top_w = _bbox_dims(_coords_bbox(top_local))
        if top_w > w:
            continue
        col0 = int(rng.integers(0, w - top_w + 1))
        row0 = (h // 2) - _top_h
        top_global = _local_to_global(top_local, row0, col0)
        coords = set(top_global)
        coords.update((h - 1 - r, c) for r, c in top_global)
        if _can_place(coords, existing_boxes, h, w):
            return {"coords": coords, "score": 0, "area": len(coords), "kind": "sym"}
    return None


def _place_axis_asymmetric(
    existing_boxes: list[BBox],
    h: int,
    w: int,
    base_size: int,
    score: int,
    rng: np.random.Generator,
) -> dict | None:
    for _ in range(MAX_PLACE_ATTEMPTS):
        built = _make_axis_asym_local(base_size, score, rng)
        if built is None:
            continue
        base_local, extras_local = built
        top_local = base_local | extras_local
        _top_h, top_w = _bbox_dims(_coords_bbox(top_local))
        if top_w > w:
            continue
        col0 = int(rng.integers(0, w - top_w + 1))
        row0 = (h // 2) - _top_h
        base_global = _local_to_global(base_local, row0, col0)
        extras_global = _local_to_global(extras_local, row0, col0)
        coords = set(base_global)
        coords.update((h - 1 - r, c) for r, c in base_global)
        coords.update(extras_global)
        if _can_place(coords, existing_boxes, h, w):
            return {"coords": coords, "score": score, "area": len(coords), "kind": "asym"}
    return None


def _place_far(
    existing_boxes: list[BBox],
    h: int,
    w: int,
    area: int,
    side: str,
    rng: np.random.Generator,
) -> dict | None:
    for _ in range(MAX_PLACE_ATTEMPTS):
        local = _make_connected_local(
            area,
            rng,
            depth_range=range(2, 5),
            width_range=range(2, 6),
            seed_bottom=False,
        )
        if local is None:
            continue
        patch_h, patch_w = _bbox_dims(_coords_bbox(local))
        if patch_w > w:
            continue
        if side == "top":
            row_lo, row_hi = 0, (h // 2) - patch_h
        else:
            row_lo, row_hi = h // 2, h - patch_h
        if row_hi < row_lo:
            continue
        row0 = int(rng.integers(row_lo, row_hi + 1))
        col0 = int(rng.integers(0, w - patch_w + 1))
        coords = {(row0 + r, col0 + c) for r, c in local}
        if _can_place(coords, existing_boxes, h, w):
            return {"coords": coords, "score": area, "area": area, "kind": "far"}
    return None


def _base_size_for_score(score: int, rng: np.random.Generator) -> int:
    candidates = [base for base in (1, 2) if 4 <= (2 * base) + int(score) <= 10]
    return int(candidates[int(rng.integers(len(candidates)))])


def _render_grid(h: int, w: int, objects: list[dict], rng: np.random.Generator) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    colors = rng.permutation(np.array(PALETTE, dtype=int))[: len(objects)]
    for color, obj in zip(colors, objects):
        coords = np.array(sorted(obj["coords"]), dtype=int)
        grid[coords[:, 0], coords[:, 1]] = int(color)
    return grid


def _solver_components_with_scores(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    diff = grid != np.flipud(grid)
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    scored = []
    for comp in comps:
        coords = comp["coords"]
        comp = dict(comp)
        comp["score"] = int(diff[coords[:, 0], coords[:, 1]].sum())
        scored.append(comp)
    return scored


def _validate_scene(grid: np.ndarray, mode: str) -> bool:
    h, w = map(int, grid.shape)
    assert h in GRID_HEIGHTS and h % 2 == 0
    assert GRID_WIDTH_RANGE[0] <= w <= GRID_WIDTH_RANGE[1]

    comps = _solver_components_with_scores(grid)
    assert 3 <= len(comps) <= 5

    colors = [int(comp["color"]) for comp in comps]
    assert len(set(colors)) == len(colors)
    assert set(colors).issubset(set(PALETTE))

    scores = [int(comp["score"]) for comp in comps]
    areas = [int(comp["area"]) for comp in comps]
    bboxes = [tuple(comp["bbox"]) for comp in comps]

    for comp in comps:
        area = int(comp["area"])
        assert 4 <= area <= 10
        box = tuple(comp["bbox"])
        box_h, box_w = _bbox_dims(box)
        assert box_h >= 2 and box_w >= 2
        crop = crop_bbox(grid, box)
        allowed = {0, int(comp["color"])}
        assert set(map(int, crop.flatten())).issubset(allowed)

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            assert _boxes_separated(bboxes[i], bboxes[j])

    best = max(scores)
    assert scores.count(best) == 1
    winner = comps[int(np.argmax(scores))]
    assert np.array_equal(main(grid), crop_bbox(grid, winner["bbox"]))
    assert any(score < area for score, area in zip(scores, areas))

    winner_area = int(winner["area"])
    winner_score = int(winner["score"])
    if mode == "far_winner":
        assert winner_score == winner_area
        assert any(area > winner_area for area in areas if area != winner_area)
    else:
        assert winner_score < winner_area

    return True


def _build_specs(mode: str, n_objects: int, rng: np.random.Generator) -> list[dict]:
    specs: list[dict] = []
    if mode == "far_winner":
        winner_score = int(rng.integers(6, 9))
        top_area_choices = [top_area for top_area in range(3, 6) if 2 * top_area > winner_score]
        specs.append({"kind": "sym", "top_area": int(top_area_choices[int(rng.integers(len(top_area_choices)))])})
        low_score = int(rng.integers(1, 4))
        specs.append({"kind": "asym", "score": low_score, "base_size": _base_size_for_score(low_score, rng)})
        specs.append({"kind": "far", "score": winner_score, "side": "top" if rng.random() < 0.5 else "bottom", "winner": True})
        if n_objects >= 4:
            specs.append({"kind": "far", "score": int(rng.integers(4, winner_score)), "side": "bottom" if specs[-1]["side"] == "top" else "top"})
        if n_objects == 5:
            if rng.random() < 0.5:
                specs.append({"kind": "sym", "top_area": int(rng.integers(2, 4))})
            else:
                filler_score = int(rng.integers(1, min(4, winner_score)))
                specs.append({"kind": "asym", "score": filler_score, "base_size": _base_size_for_score(filler_score, rng)})
    else:
        winner_score = int(rng.integers(5, 7))
        specs.append({"kind": "sym", "top_area": int(rng.integers(4, 6))})
        specs.append({"kind": "asym", "score": winner_score, "base_size": _base_size_for_score(winner_score, rng), "winner": True})
        specs.append({"kind": "far", "score": winner_score - 1, "side": "top" if rng.random() < 0.5 else "bottom"})
        if n_objects >= 4:
            filler_score = int(rng.integers(1, 4))
            specs.append({"kind": "asym", "score": filler_score, "base_size": _base_size_for_score(filler_score, rng)})
        if n_objects == 5:
            if rng.random() < 0.5:
                specs.append({"kind": "sym", "top_area": int(rng.integers(2, 4))})
            else:
                specs.append({"kind": "far", "score": 4, "side": "bottom" if specs[2]["side"] == "top" else "top"})
    return specs[:n_objects]


def _place_spec(spec: dict, existing_boxes: list[BBox], h: int, w: int, rng: np.random.Generator) -> dict | None:
    kind = spec["kind"]
    if kind == "sym":
        return _place_axis_symmetric(existing_boxes, h, w, int(spec["top_area"]), rng)
    if kind == "asym":
        return _place_axis_asymmetric(existing_boxes, h, w, int(spec["base_size"]), int(spec["score"]), rng)
    return _place_far(existing_boxes, h, w, int(spec["score"]), str(spec["side"]), rng)


def _generate_one(rng: np.random.Generator) -> np.ndarray | None:
    h = int(np.array(GRID_HEIGHTS, dtype=int)[int(rng.integers(len(GRID_HEIGHTS)))])
    w = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    mode = "far_winner" if rng.random() < 0.5 else "axis_winner"
    n_objects = int(rng.integers(3, 6))

    specs = _build_specs(mode, n_objects, rng)
    axis_specs = [spec for spec in specs if spec["kind"] in {"sym", "asym"}]
    far_specs = [spec for spec in specs if spec["kind"] == "far"]
    ordered_specs = axis_specs + far_specs

    objects: list[dict] = []
    boxes: list[BBox] = []
    for spec in ordered_specs:
        placed = _place_spec(spec, boxes, h, w, rng)
        if placed is None:
            return None
        objects.append(placed)
        boxes.append(_coords_bbox(placed["coords"]))

    grid = _render_grid(h, w, objects, rng)
    try:
        if _validate_scene(grid, mode):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_one(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
