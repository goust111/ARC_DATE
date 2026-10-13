from __future__ import annotations

import importlib.util
import random
from itertools import combinations
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox_coords = _ARC.bbox_coords
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
components_mask = _ARC.components_mask
crop_bbox = _ARC.crop_bbox
draw_line = _ARC.draw_line
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste


GREY = Color.GREY
BG = Color.BLACK
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
SPRITE_DIMS = [
    (1, 3),
    (3, 1),
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (1, 6),
    (6, 1),
    (1, 7),
    (7, 1),
    (1, 8),
    (8, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
]
TRANSFORMS = [
    lambda g: g,
    lambda g: np.rot90(g, 1),
    lambda g: np.rot90(g, 2),
    lambda g: np.rot90(g, 3),
    lambda g: np.fliplr(g),
    lambda g: np.flipud(g),
    lambda g: g.T,
    lambda g: np.rot90(g.T, 2),
]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _full_solid_block(arr: np.ndarray, color: int) -> bool:
    return bool(arr.size and np.all(arr == int(color)))


def _find_anchor(g: np.ndarray) -> dict:
    grey_objects = find_components(g, bg=get_bg(g), connectivity=4, monochromatic=True, colors=[GREY])
    anchors = []
    for comp in grey_objects:
        if int(comp["area"]) != 9:
            continue
        if bbox_h(comp["bbox"]) != 3 or bbox_w(comp["bbox"]) != 3:
            continue
        if np.all(crop_bbox(g, comp["bbox"]) == GREY):
            anchors.append(comp)
    assert len(anchors) == 1, "expected exactly one solid 3x3 grey anchor"
    return anchors[0]


def _marker_role(coord: tuple[int, int], anchor_bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]] | None:
    r, c = coord
    r0, c0, r1, c1 = anchor_bbox
    if r0 <= r <= r1 and c < c0:
        return "left", (r, c0 - 1)
    if r0 <= r <= r1 and c > c1:
        return "right", (r, c1 + 1)
    if c0 <= c <= c1 and r < r0:
        return "top", (r0 - 1, c)
    if c0 <= c <= c1 and r > r1:
        return "bottom", (r1 + 1, c)
    return None


def _ray_grid(grid: np.ndarray) -> tuple[np.ndarray, dict, list[tuple[int, int]], list[tuple[str, tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    anchor = _find_anchor(g)
    anchor_mask = np.zeros(g.shape, dtype=bool)
    coords = anchor["coords"]
    anchor_mask[coords[:, 0], coords[:, 1]] = True

    work = g.copy()
    grey_coords = np.argwhere((g == GREY) & (~anchor_mask))
    work[grey_coords[:, 0], grey_coords[:, 1]] = BG

    aligned: list[tuple[str, tuple[int, int]]] = []
    markers = [tuple(map(int, rc)) for rc in grey_coords.tolist()]
    for marker in markers:
        role = _marker_role(marker, anchor["bbox"])
        if role is None:
            continue
        side, endpoint = role
        draw_line(work, marker[0], marker[1], end_r=endpoint[0], end_c=endpoint[1], color=GREY)
        aligned.append((side, marker))
    return work, anchor, markers, aligned


def _open_regions(rayed_grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    mask = rayed_grid != GREY
    rectangles: list[dict] = []
    other: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        bbox = bbox_coords(coords)
        item = {
            "coords": coords,
            "bbox": bbox,
            "area": int(coords.shape[0]),
            "shape": (bbox_h(bbox), bbox_w(bbox)),
        }
        if int(item["area"]) == _bbox_area(bbox):
            rectangles.append(item)
        else:
            other.append(item)
    rectangles.sort(key=lambda item: item["bbox"])
    other.sort(key=lambda item: item["bbox"])
    return rectangles, other


def _extract_sprites(g: np.ndarray) -> list[dict]:
    sprites = []
    for comp in find_components(g, bg=get_bg(g), connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color in (BG, GREY):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        crop = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        rel = comp["coords"] - np.array([r0, c0])
        crop[rel[:, 0], rel[:, 1]] = color
        sprites.append(
            {
                "color": color,
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "crop": crop,
                "shape": crop.shape,
            }
        )
    sprites.sort(key=lambda item: item["bbox"])
    return sprites


def _can_tile_region(sprite_crop: np.ndarray, region_shape: tuple[int, int]) -> bool:
    rh, rw = region_shape
    sh, sw = sprite_crop.shape
    if rh % sh != 0 or rw % sw != 0:
        return False
    tiled = np.tile(sprite_crop, (rh // sh, rw // sw))
    return bool(np.all(tiled != BG))


def _unique_assignment(sprites: list[dict], regions: list[dict]) -> list[int]:
    n = len(sprites)
    assert n == len(regions) and n > 0
    compat = [[_can_tile_region(sprite["crop"], region["shape"]) for region in regions] for sprite in sprites]
    assert all(any(row) for row in compat), "every sprite must fit at least one region"
    assert all(any(compat[r][c] for r in range(n)) for c in range(n)), "every region must accept at least one sprite"

    solutions: list[list[int]] = []

    def search(i: int, used_regions: set[int], assignment: list[int]) -> None:
        if len(solutions) > 1:
            return
        if i == n:
            solutions.append(assignment.copy())
            return

        best_idx = None
        best_options = None
        for sprite_idx in range(n):
            if assignment[sprite_idx] != -1:
                continue
            options = [region_idx for region_idx in range(n) if compat[sprite_idx][region_idx] and region_idx not in used_regions]
            if best_options is None or len(options) < len(best_options):
                best_idx = sprite_idx
                best_options = options
        assert best_idx is not None and best_options is not None
        if not best_options:
            return

        for region_idx in best_options:
            assignment[best_idx] = region_idx
            used_regions.add(region_idx)
            search(i + 1, used_regions, assignment)
            used_regions.remove(region_idx)
            assignment[best_idx] = -1

    search(0, set(), [-1] * n)
    assert len(solutions) == 1, "sprite-to-region pairing must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Extend anchor-aligned grey rays, treat the resulting enclosed rectangles as containers, and tile-match each source sprite into its unique rectangle.

    Input structure:
    - The grid has a black background, one solid 3x3 grey anchor, several isolated grey marker pixels, and 2-3 small monochrome source sprites in other colors.
    - Some grey markers share one anchor row or one anchor column; other grey markers are unaligned noise.
    - The anchor, the valid marker rays, and the grid boundary jointly carve the non-grey-free space into a mix of open regions, of which exactly 2-3 are perfect rectangles.

    Rule:
    - Keep the 3x3 grey anchor and erase every original grey marker before drawing anything else.
    - For each grey marker aligned with an anchor row or anchor column, draw a grey horizontal or vertical ray from that marker to the cell adjacent to the anchor; ignore and erase unaligned grey markers.
    - On the ray-augmented grid, find every 4-connected non-grey region enclosed by grey and/or the grid boundary whose occupied cells form an exact rectangle.
    - For each source sprite, test whether its cropped shape can tile one rectangle by pure translation with no rotation or reflection; use the unique exact sprite-to-rectangle matching.
    - Erase the original sprite positions and fill each matched rectangle by repeating that sprite pattern from the rectangle's top-left corner.

    Key insight:
    - The destination containers are not explicit input objects: they emerge only after the aligned grey markers are extended into rays, and the final sprite placement is determined by geometric tiling compatibility rather than by color or source position.
    """
    g = np.asarray(grid, dtype=int)
    rayed, _anchor, _markers, _aligned = _ray_grid(g)
    regions, _other = _open_regions(rayed)
    sprites = _extract_sprites(g)
    assert len(regions) == len(sprites) in (2, 3), "expected 2-3 rectangular regions and the same number of sprites"

    assignment = _unique_assignment(sprites, regions)

    out = rayed.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for sprite_idx, region_idx in enumerate(assignment):
        sprite = sprites[sprite_idx]
        region = regions[region_idx]
        r0, c0, r1, c1 = region["bbox"]
        rh, rw = region["shape"]
        sh, sw = sprite["crop"].shape
        tiled = np.tile(sprite["crop"], (rh // sh, rw // sw))
        out[r0 : r1 + 1, c0 : c1 + 1] = tiled

    return out


def _count_unique_matchings(region_shapes: list[tuple[int, int]], sprite_dims: list[tuple[int, int]]) -> tuple[int, list[int] | None, list[list[bool]]]:
    n = len(region_shapes)
    compat = [[(rh % sh == 0 and rw % sw == 0) for rh, rw in region_shapes] for sh, sw in sprite_dims]
    solutions: list[list[int]] = []

    def search(i: int, used: set[int], assignment: list[int]) -> None:
        if len(solutions) > 1:
            return
        if i == n:
            solutions.append(assignment.copy())
            return
        best = None
        best_options = None
        for sprite_idx in range(n):
            if assignment[sprite_idx] != -1:
                continue
            options = [region_idx for region_idx in range(n) if compat[sprite_idx][region_idx] and region_idx not in used]
            if best_options is None or len(options) < len(best_options):
                best = sprite_idx
                best_options = options
        if best is None or best_options is None or not best_options:
            return
        for region_idx in best_options:
            assignment[best] = region_idx
            used.add(region_idx)
            search(i + 1, used, assignment)
            used.remove(region_idx)
            assignment[best] = -1

    search(0, set(), [-1] * n)
    return len(solutions), (solutions[0] if len(solutions) == 1 else None), compat


def _choose_sprite_dims(region_shapes: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[int]]:
    best_score = None
    best_choice = None
    for dims in combinations(SPRITE_DIMS, len(region_shapes)):
        count, assignment, compat = _count_unique_matchings(region_shapes, list(dims))
        if count != 1 or assignment is None:
            continue
        areas = [h * w for h, w in dims]
        duplicate_areas = len(areas) - len(set(areas))
        edge_count = sum(sum(row) for row in compat)
        ambiguous_rows = sum(sum(row) > 1 for row in compat)
        ambiguous_cols = sum(sum(compat[r][c] for r in range(len(dims))) > 1 for c in range(len(region_shapes)))
        score = duplicate_areas * 100 + edge_count * 10 + ambiguous_rows * 4 + ambiguous_cols * 4
        if best_score is None or score > best_score:
            best_score = score
            best_choice = (list(dims), assignment)
    if best_choice is None:
        raise ValueError(f"no unique sprite tiling set for regions {region_shapes}")
    return best_choice


def _place_rectangles(grid: np.ndarray, dims: list[tuple[int, int]], colors: list[int], safe_box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = safe_box
    order = list(range(len(dims)))
    order.sort(key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)
    occupied = np.zeros(grid.shape, dtype=bool)

    for idx in order:
        h, w = dims[idx]
        choices = []
        for rr in range(r0, r1 - h + 2):
            for cc in range(c0, c1 - w + 2):
                if np.any(grid[rr : rr + h, cc : cc + w] != BG):
                    continue
                rr0 = max(0, rr - 1)
                cc0 = max(0, cc - 1)
                rr1 = min(grid.shape[0], rr + h + 1)
                cc1 = min(grid.shape[1], cc + w + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                choices.append((rr, cc))
        if not choices:
            return False
        rr, cc = random.choice(choices)
        patch = np.full((h, w), colors[idx], dtype=int)
        paste(grid, patch, rr, cc)
        occupied[rr : rr + h, cc : cc + w] = True
    return True


def _place_noise_markers(grid: np.ndarray, anchor_bbox: tuple[int, int, int, int], count: int) -> None:
    r0, c0, r1, c1 = anchor_bbox
    anchor_rows = set(range(r0, r1 + 1))
    anchor_cols = set(range(c0, c1 + 1))
    placed = 0
    for _ in range(500):
        if placed >= count:
            return
        rr = random.randrange(grid.shape[0])
        cc = random.randrange(grid.shape[1])
        if rr in anchor_rows or cc in anchor_cols:
            continue
        if grid[rr, cc] != BG:
            continue
        neighborhood = grid[max(0, rr - 1) : rr + 2, max(0, cc - 1) : cc + 2]
        if np.any(neighborhood == GREY):
            continue
        grid[rr, cc] = GREY
        placed += 1


def _build_canonical_input() -> np.ndarray:
    family = random.choices(["two_left", "three_top", "three_bottom"], weights=[4, 3, 3])[0]

    H = random.randint(16, 24)
    W = random.randint(18, 26)
    ar = random.randint(4, 8)
    ac = random.randint(4, 7)
    rk = ar + random.randint(0, 2)
    c0 = ac
    c2 = ac + 2

    if family == "two_left":
        region_shapes = [(rk, c0), (H - rk - 1, c0)]
        aligned_markers = [(0, c0), (H - 1, c0), (rk, 0)]
    elif family == "three_top":
        region_shapes = [(rk, c0), (ar, 1), (H - rk - 1, c0)]
        aligned_markers = [(0, c0), (H - 1, c0), (0, c2), (rk, 0)]
    else:
        region_shapes = [(rk, c0), (H - (ar + 3), 1), (H - rk - 1, c0)]
        aligned_markers = [(0, c0), (H - 1, c0), (H - 1, c2), (rk, 0)]

    if min(h for h, _w in region_shapes) < 3 or min(w for _h, w in region_shapes) < 1:
        raise ValueError("degenerate region")
    sprite_dims, _assignment = _choose_sprite_dims(region_shapes)

    grid = np.zeros((H, W), dtype=int)
    grid[ar : ar + 3, ac : ac + 3] = GREY
    for rr, cc in aligned_markers:
        grid[rr, cc] = GREY

    colors = random.sample(SPRITE_COLORS, len(sprite_dims))
    safe_box = (1, ac + 4, H - 2, W - 2)
    if safe_box[1] > safe_box[3]:
        raise ValueError("safe zone too thin")
    if not _place_rectangles(grid, sprite_dims, colors, safe_box):
        raise ValueError("failed to place sprites")

    _place_noise_markers(grid, (ar, ac, ar + 2, ac + 2), random.randint(1, 3))
    return grid


def _semantic_checks(inp: np.ndarray) -> None:
    rayed, anchor, markers, aligned = _ray_grid(inp)
    rectangles, others = _open_regions(rayed)
    sprites = _extract_sprites(inp)
    directions = {side for side, _marker in aligned}

    assert len(directions) >= 2
    assert len(rectangles) == len(sprites) in (2, 3)
    assert others, "generator should leave a non-rectangular open area for the source sprites"
    assert all(_bbox_area(region["bbox"]) == int(region["area"]) for region in rectangles)
    assert len({sprite["color"] for sprite in sprites}) == len(sprites)
    assert all(_full_solid_block(sprite["crop"], sprite["color"]) for sprite in sprites)

    assignment = _unique_assignment(sprites, rectangles)
    out = main(inp)
    assert out.shape == inp.shape
    for marker in markers:
        role = _marker_role(marker, anchor["bbox"])
        if role is None:
            assert int(out[marker]) == BG
    for sprite in sprites:
        coords = sprite["coords"]
        assert np.all(out[coords[:, 0], coords[:, 1]] != sprite["color"])

    for sprite_idx, region_idx in enumerate(assignment):
        sprite = sprites[sprite_idx]
        region = rectangles[region_idx]
        r0, c0, r1, c1 = region["bbox"]
        rh, rw = region["shape"]
        sh, sw = sprite["crop"].shape
        expected = np.tile(sprite["crop"], (rh // sh, rw // sw))
        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid = _build_canonical_input()
        except (AssertionError, ValueError):
            continue
        grid = random.choice(TRANSFORMS)(grid.copy())
        try:
            _semantic_checks(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ray-split sprite-tiling instance")
