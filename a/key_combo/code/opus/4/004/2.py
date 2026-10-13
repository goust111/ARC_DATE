import math
from itertools import permutations, product

import numpy as np


BG = 0
BODY_COLORS = (2, 3, 4, 5)
MARKER_COLORS = (1, 7)
_RNG = np.random.default_rng()

_GRID_MIN = 18
_GRID_MAX = 24
_MAX_GENERATION_ATTEMPTS = 1000

_SPLIT_QUARTET_SPECS = (
    {
        "pairs": {1: (0, 3), 7: (1, 2)},
        "coords": (
            ((0, 2), (1, 2), (2, 2), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)),
            ((0, 2), (1, 2), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)),
            ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
            ((0, 1), (1, 0), (1, 1), (2, 0), (3, 0)),
        ),
    },
    {
        "pairs": {1: (0, 2), 7: (1, 3)},
        "coords": (
            ((0, 3), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (3, 2)),
            ((0, 3), (1, 3), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3)),
            ((0, 0), (0, 1), (0, 2), (1, 0)),
            ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)),
        ),
    },
)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in arr:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _find_foreground_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0])})
    return comps


def _aligned_pixel_map(crop: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(r), int(c)): int(crop[r, c]) for r, c in np.argwhere(crop != BG)}


def _analyze(grid: np.ndarray) -> tuple[list[dict], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_foreground_components(g, bg=BG)
    assert len(components) == 4

    color_positions: dict[int, list[tuple[int, int]]] = {}
    color_components: dict[int, set[int]] = {}
    for idx, comp in enumerate(components):
        for rr, cc in comp["coords"]:
            color = int(g[rr, cc])
            color_positions.setdefault(color, []).append((int(rr), int(cc)))
            color_components.setdefault(color, set()).add(idx)

    marker_colors = sorted(
        color
        for color, positions in color_positions.items()
        if color != BG and len(positions) == 2 and len(color_components[color]) == 2
    )
    assert marker_colors == [1, 7]

    pair_distances = {color: manhattan(*color_positions[color]) for color in marker_colors}
    assert pair_distances[1] != pair_distances[7]

    sprites: list[dict] = []
    for idx, comp in enumerate(components):
        bbox = comp["bbox"]
        crop = crop_bbox(g, bbox)
        colors = {int(g[r, c]) for r, c in comp["coords"]}
        marker_set = tuple(sorted(color for color in marker_colors if color in colors))
        body_colors = sorted(colors - {BG, *marker_colors})
        assert len(body_colors) == 1
        distance = min(pair_distances[color] for color in marker_set) if marker_set else math.inf
        sprites.append(
            {
                "index": idx,
                "bbox": bbox,
                "crop": crop,
                "body_color": int(body_colors[0]),
                "marker_colors": marker_set,
                "distance": float(distance),
                "aligned_pixels": _aligned_pixel_map(crop),
            }
        )
    return sprites, pair_distances


def _compose_from_order(sprites: list[dict], order: list[int]) -> np.ndarray:
    out_h = max(sprite["crop"].shape[0] for sprite in sprites)
    out_w = max(sprite["crop"].shape[1] for sprite in sprites)
    out = np.zeros((out_h, out_w), dtype=int)
    for idx in order:
        patch = sprites[idx]["crop"]
        mask = patch != BG
        out[: patch.shape[0], : patch.shape[1]][mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay all cropped sprites after sorting them by connector-pair distance.

    Input structure:
    - The grid has black background and exactly four scattered 4-connected sprites.
    - Each sprite has one body color from {2,3,4,5}; some boundary pixels are replaced by marker colors 1 or 7.
    - Each marker color appears exactly twice, on two different sprites, forming two connector pairs.

    Rule:
    - Find the four multicolor sprites and crop each to its tight bounding box.
    - For each marker color, compute the Manhattan distance between its two occurrences.
    - Give every sprite the minimum distance among the marker colors it carries; sprites with no marker get infinite distance.
    - Create a canvas whose size is the maximum sprite height by maximum sprite width, align every crop to the canvas upper-left, and draw them from larger distance to smaller distance, treating black as transparent.

    Key insight:
    - The marker pairs are not assembled physically; their pairwise distances encode the layer order for the final transparent overlay.
    """
    sprites, _pair_distances = _analyze(np.asarray(grid, dtype=int))
    order = sorted(range(len(sprites)), key=lambda i: sprites[i]["distance"], reverse=True)
    return _compose_from_order(sprites, order)


def _shape_boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if any(not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc] for nr, nc in neighbors4(rr, cc)):
            cells.append((rr, cc))
    return cells


def _generate_shape() -> np.ndarray | None:
    for _ in range(200):
        h = int(_RNG.integers(2, 6))
        w = int(_RNG.integers(2, 6))
        max_area = min(8, h * w - 1)
        if max_area < 3:
            continue
        target = int(_RNG.integers(3, max_area + 1))
        start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(_RNG.integers(len(frontier)))])
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 3 or int(mask.sum()) > 8:
            continue
        if np.all(mask):
            continue
        if len(_shape_boundary_cells(mask)) < 2:
            continue
        return mask
    return None


def _shape_overlap(shape_a: dict, shape_b: dict) -> bool:
    return any(pos in shape_b["pixels"] for pos in shape_a["pixels"])


def _search_plan(shapes: list[dict]) -> list[dict]:
    overlaps = {
        (i, j): _shape_overlap(shapes[i], shapes[j])
        for i in range(len(shapes))
        for j in range(len(shapes))
        if i != j
    }
    plans = []

    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    for pair_a, pair_b in pairings:
        if overlaps[pair_a] or overlaps[pair_b]:
            continue
        if not any(overlaps[(i, j)] for i in pair_a for j in pair_b):
            continue
        plans.append({"kind": "split", "pairs": {1: pair_a, 7: pair_b}})

    for hub, small_partner, large_partner, none_idx in permutations(range(4), 4):
        if overlaps[(hub, small_partner)]:
            continue
        groups = [{hub, small_partner}, {large_partner}, {none_idx}]
        diff_overlap = False
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                if any(overlaps[(a, b)] for a in groups[gi] for b in groups[gj]):
                    diff_overlap = True
        if not diff_overlap:
            continue
        plans.append(
            {
                "kind": "hub",
                "pairs": {1: (hub, small_partner), 7: (hub, large_partner)},
                "hub": hub,
                "small_partner": small_partner,
                "large_partner": large_partner,
                "none": none_idx,
            }
        )

    return plans


def _bboxes_far_enough(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int], gap: int = 2
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_shapes(shapes: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda i: len(shapes[i]["pixels"]), reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    bboxes: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        sh = shapes[idx]["mask"].shape
        placed = False
        for _ in range(300):
            r0 = int(_RNG.integers(0, h - sh[0] + 1))
            c0 = int(_RNG.integers(0, w - sh[1] + 1))
            bbox = (r0, c0, r0 + sh[0] - 1, c0 + sh[1] - 1)
            if all(_bboxes_far_enough(bbox, bboxes[other], gap=2) for other in bboxes):
                placements[idx] = (r0, c0)
                bboxes[idx] = bbox
                placed = True
                break
        if not placed:
            return None
    return [placements[i] for i in range(len(shapes))]


def _marker_assignments_for_plan(
    shapes: list[dict], placements: list[tuple[int, int]], plan: dict
) -> list[tuple[dict[int, tuple[int, int]], tuple[int, int]]] | None:
    boundary_globals = []
    for shape, (r0, c0) in zip(shapes, placements):
        boundary_globals.append([(r0 + r, c0 + c) for r, c in shape["boundary"]])

    if plan["kind"] == "split":
        blue_a, blue_b = plan["pairs"][1]
        orange_a, orange_b = plan["pairs"][7]
        candidates = []
        for pos_ba, pos_bb, pos_oa, pos_ob in product(
            boundary_globals[blue_a],
            boundary_globals[blue_b],
            boundary_globals[orange_a],
            boundary_globals[orange_b],
        ):
            d_blue = manhattan(pos_ba, pos_bb)
            d_orange = manhattan(pos_oa, pos_ob)
            if d_blue == d_orange or abs(d_blue - d_orange) < 2:
                continue
            assignments = {
                blue_a: {1: pos_ba},
                blue_b: {1: pos_bb},
                orange_a: {7: pos_oa},
                orange_b: {7: pos_ob},
            }
            candidates.append((assignments, (d_blue, d_orange)))
        if not candidates:
            return None
        _RNG.shuffle(candidates)
        candidates.sort(key=lambda item: abs(item[1][0] - item[1][1]), reverse=True)
        return candidates[:40]

    hub = plan["hub"]
    small_partner = plan["small_partner"]
    large_partner = plan["large_partner"]
    candidates = []
    for hub_blue in boundary_globals[hub]:
        for hub_orange in boundary_globals[hub]:
            if hub_blue == hub_orange:
                continue
            for pos_small in boundary_globals[small_partner]:
                for pos_large in boundary_globals[large_partner]:
                    d_blue = manhattan(hub_blue, pos_small)
                    d_orange = manhattan(hub_orange, pos_large)
                    if d_blue >= d_orange or d_blue == d_orange or d_orange - d_blue < 2:
                        continue
                    assignments = {
                        hub: {1: hub_blue, 7: hub_orange},
                        small_partner: {1: pos_small},
                        large_partner: {7: pos_large},
                    }
                    candidates.append((assignments, (d_blue, d_orange)))
    if not candidates:
        return None
    _RNG.shuffle(candidates)
    candidates.sort(key=lambda item: item[1][1] - item[1][0], reverse=True)
    return candidates[:40]


def _render_input(
    shapes: list[dict],
    placements: list[tuple[int, int]],
    marker_assignments: dict[int, dict[int, tuple[int, int]]],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(canvas_shape, dtype=int)

    for idx, (shape, (r0, c0)) in enumerate(zip(shapes, placements)):
        grid[r0 : r0 + shape["mask"].shape[0], c0 : c0 + shape["mask"].shape[1]][shape["mask"]] = shape["body_color"]
        for color, pos in marker_assignments.get(idx, {}).items():
            grid[pos] = color
    return grid


def _aligned_overlap_is_order_sensitive(sprite_a: dict, sprite_b: dict) -> bool:
    common = set(sprite_a["aligned_pixels"]) & set(sprite_b["aligned_pixels"])
    return any(sprite_a["aligned_pixels"][pos] != sprite_b["aligned_pixels"][pos] for pos in common)


def _proxy_orders(sprites: list[dict]) -> list[list[int]]:
    indices = list(range(len(sprites)))
    return [
        list(sorted(indices, key=lambda i: sprites[i]["distance"])),
        list(sorted(indices, key=lambda i: (sprites[i]["bbox"][0], sprites[i]["bbox"][1]))),
        list(sorted(indices, key=lambda i: (sprites[i]["bbox"][0], sprites[i]["bbox"][1]), reverse=True)),
        list(sorted(indices, key=lambda i: (sprites[i]["bbox"][1], sprites[i]["bbox"][0]))),
        list(sorted(indices, key=lambda i: (sprites[i]["bbox"][1], sprites[i]["bbox"][0]), reverse=True)),
        list(sorted(indices, key=lambda i: (sprites[i]["crop"].shape[0] * sprites[i]["crop"].shape[1], i))),
        list(sorted(indices, key=lambda i: (sprites[i]["crop"].shape[0] * sprites[i]["crop"].shape[1], i), reverse=True)),
        list(sorted(indices, key=lambda i: sprites[i]["body_color"])),
        list(sorted(indices, key=lambda i: sprites[i]["body_color"], reverse=True)),
        list(sorted(indices, key=lambda i: len(sprites[i]["marker_colors"]))),
        list(sorted(indices, key=lambda i: len(sprites[i]["marker_colors"]), reverse=True)),
    ]


def _validate_candidate(grid: np.ndarray) -> bool:
    sprites, pair_distances = _analyze(grid)
    assert sorted(pair_distances.values()) == list(sorted(pair_distances.values()))
    assert pair_distances[1] != pair_distances[7]

    true_order = sorted(range(len(sprites)), key=lambda i: sprites[i]["distance"], reverse=True)
    out = _compose_from_order(sprites, true_order)
    assert np.array_equal(main(grid), out)

    different_distance_overlap = False
    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            same_distance = sprites[i]["distance"] == sprites[j]["distance"]
            if same_distance:
                assert not (set(sprites[i]["aligned_pixels"]) & set(sprites[j]["aligned_pixels"]))
            else:
                if _aligned_overlap_is_order_sensitive(sprites[i], sprites[j]):
                    different_distance_overlap = True
    assert different_distance_overlap

    proxy_outputs = {_compose_from_order(sprites, order).tobytes() for order in _proxy_orders(sprites)}
    assert out.tobytes() not in proxy_outputs
    return True


def _build_shape_dicts(masks: list[np.ndarray]) -> list[dict]:
    shapes: list[dict] = []
    for body_color, mask in zip(_RNG.permutation(BODY_COLORS), masks):
        shapes.append(
            {
                "mask": mask.copy(),
                "pixels": {tuple(map(int, rc)) for rc in np.argwhere(mask)},
                "boundary": _shape_boundary_cells(mask),
                "body_color": int(body_color),
            }
        )
    return shapes


def _random_shape_set() -> list[dict] | None:
    masks = []
    for _ in range(4):
        mask = _generate_shape()
        if mask is None:
            return None
        masks.append(mask)
    return _build_shape_dicts(masks)


def _split_shape_set_and_plan() -> tuple[list[dict], dict]:
    spec = _SPLIT_QUARTET_SPECS[int(_RNG.integers(len(_SPLIT_QUARTET_SPECS)))]
    masks = [_mask_from_coords(coords) for coords in spec["coords"]]
    plan = {"kind": "split", "pairs": dict(spec["pairs"])}
    if float(_RNG.random()) < 0.5:
        plan["pairs"] = {1: plan["pairs"][7], 7: plan["pairs"][1]}
    return _build_shape_dicts(masks), plan


def _gen_attempt() -> np.ndarray | None:
    if float(_RNG.random()) < 0.97:
        shapes = _random_shape_set()
        if shapes is None:
            return None
        plans = [plan for plan in _search_plan(shapes) if plan["kind"] == "hub"]
        if not plans:
            return None
        _RNG.shuffle(plans)
        plan_weighted = plans
    else:
        shapes, split_plan = _split_shape_set_and_plan()
        plan_weighted = [split_plan]

    for plan in plan_weighted[:20]:
        grid_h = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
        grid_w = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
        placements = _place_shapes(shapes, grid_h, grid_w)
        if placements is None:
            continue
        marker_candidates = _marker_assignments_for_plan(shapes, placements, plan)
        if not marker_candidates:
            continue

        for marker_assignments, _pair_distances in marker_candidates:
            grid = _render_input(shapes, placements, marker_assignments, (grid_h, grid_w))
            if max(grid.shape) > 30:
                continue
            try:
                if _validate_candidate(grid):
                    return grid
            except AssertionError:
                continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid distance-layer overlay puzzle")
