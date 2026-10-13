from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _UTILS_PATH)
assert _UTILS_SPEC is not None and _UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


RED = 2
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
MARKER_CELLS = {
    "top_left": frozenset({(0, 1), (1, 0), (1, 1)}),
    "top_right": frozenset({(0, 0), (1, 0), (1, 1)}),
    "bottom_left": frozenset({(0, 0), (0, 1), (1, 1)}),
    "bottom_right": frozenset({(0, 0), (0, 1), (1, 0)}),
}
MARKER_TO_CORNER = {cells: corner for corner, cells in MARKER_CELLS.items()}
MARKER_QUADRANTS = ("nw", "ne", "sw", "se")

GEN_ATTEMPTS = 500
GEN_PADDING = 1
GEN_BG_CHOICES = tuple(color for color in range(10) if color != RED)
GEN_MIN_SIZE = 16
GEN_MAX_SIZE = 30
GEN_COLOR_POOL = tuple(range(10))
GEN_STYLE_SOLID = "solid"
GEN_STYLE_FRAME = "frame"
GEN_STYLE_PUNCTURE = "puncture"


def _marker_corner_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, _r1, _c1 = bbox
    local = frozenset((int(r) - r0, int(c) - c0) for r, c in coords)
    assert local in MARKER_TO_CORNER, "red component must be a 3-pixel L in a 2x2 box"
    return MARKER_TO_CORNER[local]


def _touches8(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    for ra, ca in coords_a:
        for rb, cb in coords_b:
            if max(abs(int(ra) - int(rb)), abs(int(ca) - int(cb))) == 1:
                return True
    return False


def _bbox_corner_distance(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    corner: str,
) -> int:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    if corner == "top_left":
        return max(r0, c0)
    if corner == "top_right":
        return max(r0, w - 1 - c1)
    if corner == "bottom_left":
        return max(h - 1 - r1, c0)
    assert corner == "bottom_right"
    return max(h - 1 - r1, w - 1 - c1)


def _first_bbox(corner: str, patch_shape: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, patch_shape)
    gh, gw = map(int, grid_shape)
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, gw - w
    if corner == "bottom_left":
        return gh - h, 0
    assert corner == "bottom_right"
    return gh - h, gw - w


def _next_bbox(
    corner: str,
    prev_bbox: tuple[int, int, int, int],
    patch_shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = map(int, patch_shape)
    pr0, pc0, pr1, pc1 = map(int, prev_bbox)
    if corner == "top_left":
        return pr1 + 1, pc1 + 1
    if corner == "top_right":
        return pr1 + 1, pc0 - w
    if corner == "bottom_left":
        return pr0 - h, pc1 + 1
    assert corner == "bottom_right"
    return pr0 - h, pc0 - w


def _place_corner_queue(
    out: np.ndarray,
    items: list[dict],
    corner: str,
    bg: int,
) -> None:
    prev_bbox: tuple[int, int, int, int] | None = None
    for item in items:
        patch = item["patch"]
        ph, pw = map(int, patch.shape)
        if prev_bbox is None:
            r0, c0 = _first_bbox(corner, patch.shape, out.shape)
        else:
            r0, c0 = _next_bbox(corner, prev_bbox, patch.shape)
        r1 = r0 + ph - 1
        c1 = c0 + pw - 1
        assert 0 <= r0 <= r1 < out.shape[0]
        assert 0 <= c0 <= c1 < out.shape[1]
        paste(out, patch, r0, c0, transparent=bg, overwrite=False)
        prev_bbox = (r0, c0, r1, c1)


def _parse_routes(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(get_bg(grid))
    assert bg != RED, "red is reserved for route markers"

    red_components = [
        comp
        for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True)
        if int(comp["color"]) == RED
    ]
    assert red_components, "expected at least one red route marker"

    markers = []
    for comp in red_components:
        assert int(comp["area"]) == 3
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert (r1 - r0, c1 - c0) == (1, 1)
        markers.append(
            {
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "corner": _marker_corner_from_coords(comp["coords"], comp["bbox"]),
            }
        )

    sprites = [
        comp
        for comp in find_components(grid, bg=bg, connectivity=8, monochromatic=True)
        if int(comp["color"]) != RED
    ]
    assert len(sprites) == len(markers), "expected one non-red sprite per marker"

    routes = []
    used_sprites: set[int] = set()
    for marker in markers:
        candidate_ids = [
            idx
            for idx, sprite in enumerate(sprites)
            if _touches8(marker["coords"], sprite["coords"])
        ]
        assert len(candidate_ids) == 1, "each marker must diagonally touch exactly one sprite"
        sprite_idx = candidate_ids[0]
        assert sprite_idx not in used_sprites, "sprites must not share markers"
        used_sprites.add(sprite_idx)

        sprite = sprites[sprite_idx]
        color = int(sprite["color"])
        bbox = tuple(map(int, sprite["bbox"]))
        patch = crop_bbox(grid, bbox)
        assert set(map(int, np.unique(patch))) <= {bg, color}, "sprite bbox must not contain foreign colors"
        routes.append(
            {
                "corner": marker["corner"],
                "distance": _bbox_corner_distance(bbox, grid.shape, marker["corner"]),
                "patch": patch,
                "bbox": bbox,
                "color": color,
            }
        )

    assert len(used_sprites) == len(sprites)
    return bg, routes


def main(grid: np.ndarray) -> np.ndarray:
    """Route each sprite to the corner indicated by its red L marker and queue same-corner sprites along the inward diagonal.

    Input structure:
    - The grid has one dominant background color.
    - Non-red monochrome sprites appear as isolated connected components.
    - Each sprite has one red 3-pixel L marker in a 2x2 box that 8-touches the sprite diagonally; the missing cell of that L indicates one of the four grid corners.

    Rule:
    - Clear the canvas to the background color.
    - Pair every red L marker with its uniquely diagonal-adjacent non-red sprite.
    - Route the sprite to the corner indicated by the marker; among sprites going to the same corner, sort by the Chebyshev distance from that sprite bbox to the target corner in the input.
    - Place the closest sprite flush to the corner, then place later sprites along the diagonal toward the center so each new sprite bbox outer corner is diagonally adjacent to the previous sprite bbox inner corner.
    - Delete all red markers.

    Key insight:
    - The red L is a routing instruction: its missing cell chooses the destination corner, and original distance to that corner determines the diagonal queue order for sprites sharing a destination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, routes = _parse_routes(g)
    out = np.full_like(g, bg)

    for corner in CORNERS:
        corner_items = [item for item in routes if item["corner"] == corner]
        distances = [int(item["distance"]) for item in corner_items]
        assert len(distances) == len(set(distances)), "same-corner route distances must be unique"
        corner_items.sort(key=lambda item: int(item["distance"]))
        _place_corner_queue(out, corner_items, corner, bg)

    return out


def _make_sprite_patch(bg: int, color: int) -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    patch = np.full((h, w), color, dtype=int)

    styles = [GEN_STYLE_SOLID]
    if h >= 3 and w >= 3:
        styles.extend([GEN_STYLE_FRAME, GEN_STYLE_PUNCTURE])
    style = random.choice(styles)

    if style == GEN_STYLE_FRAME:
        patch[1:-1, 1:-1] = bg
    elif style == GEN_STYLE_PUNCTURE:
        hole_r = random.randint(1, h - 2)
        hole_c = random.randint(1, w - 2)
        patch[hole_r, hole_c] = bg

    return patch


def _make_marker_patch(bg: int, corner: str) -> np.ndarray:
    patch = np.full((2, 2), bg, dtype=int)
    for r, c in MARKER_CELLS[corner]:
        patch[r, c] = RED
    return patch


def _assemble_group_patch(
    sprite_patch: np.ndarray,
    corner: str,
    quadrant: str,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = map(int, sprite_patch.shape)
    group = np.full((h + 2, w + 2), bg, dtype=int)
    marker = _make_marker_patch(bg, corner)

    if quadrant == "nw":
        sprite_r, sprite_c = 2, 2
        marker_r, marker_c = 0, 0
    elif quadrant == "ne":
        sprite_r, sprite_c = 2, 0
        marker_r, marker_c = 0, w
    elif quadrant == "sw":
        sprite_r, sprite_c = 0, 2
        marker_r, marker_c = h, 0
    else:
        assert quadrant == "se"
        sprite_r, sprite_c = 0, 0
        marker_r, marker_c = h, w

    paste(group, sprite_patch, sprite_r, sprite_c, transparent=bg)
    paste(group, marker, marker_r, marker_c, transparent=bg)
    return group, (sprite_r, sprite_c)


def _corner_multiset() -> list[str]:
    active_count = random.randint(2, 4)
    active = random.sample(list(CORNERS), active_count)
    corners = active[:]
    for _ in range(random.randint(1, 3)):
        corners.append(random.choice(active))
    random.shuffle(corners)
    return corners


def _output_min_shape(specs: list[dict]) -> tuple[int, int] | None:
    total_h = {corner: 0 for corner in CORNERS}
    total_w = {corner: 0 for corner in CORNERS}
    for spec in specs:
        h, w = map(int, spec["sprite_patch"].shape)
        total_h[spec["corner"]] += h
        total_w[spec["corner"]] += w

    min_h = max(
        total_h["top_left"] + total_h["bottom_left"],
        total_h["top_right"] + total_h["bottom_right"],
        total_h["top_left"] + total_h["bottom_right"],
        total_h["top_right"] + total_h["bottom_left"],
        max(total_h.values()),
        GEN_MIN_SIZE,
    )
    min_w = max(
        total_w["top_left"] + total_w["top_right"],
        total_w["bottom_left"] + total_w["bottom_right"],
        total_w["top_left"] + total_w["bottom_right"],
        total_w["top_right"] + total_w["bottom_left"],
        max(total_w.values()),
        GEN_MIN_SIZE,
    )

    min_h += 2
    min_w += 2
    if min_h > GEN_MAX_SIZE or min_w > GEN_MAX_SIZE:
        return None
    return min_h, min_w


def _build_expected_output(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for corner in CORNERS:
        corner_items = [
            {"patch": spec["sprite_patch"], "distance": spec["distance"]}
            for spec in specs
            if spec["corner"] == corner
        ]
        distances = [int(item["distance"]) for item in corner_items]
        assert len(distances) == len(set(distances))
        corner_items.sort(key=lambda item: int(item["distance"]))
        _place_corner_queue(out, corner_items, corner, bg)
    return out


def _same_corner_distances_unique(specs: list[dict]) -> bool:
    for corner in CORNERS:
        distances = [int(spec["distance"]) for spec in specs if spec["corner"] == corner]
        if len(distances) != len(set(distances)):
            return False
    return True


def _semantic_check(grid: np.ndarray, specs: list[dict], bg: int, expected: np.ndarray) -> bool:
    red_components = [
        comp
        for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True)
        if int(comp["color"]) == RED
    ]
    if len(red_components) != len(specs):
        return False
    for comp in red_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if int(comp["area"]) != 3 or (r1 - r0, c1 - c0) != (1, 1):
            return False
        if _marker_corner_from_coords(comp["coords"], comp["bbox"]) not in CORNERS:
            return False

    sprite_components = [
        comp
        for comp in find_components(grid, bg=bg, connectivity=8, monochromatic=True)
        if int(comp["color"]) != RED
    ]
    if len(sprite_components) != len(specs):
        return False

    planned_distances = {corner: [] for corner in CORNERS}
    for spec in specs:
        planned_distances[spec["corner"]].append(int(spec["distance"]))
    if any(len(values) != len(set(values)) for values in planned_distances.values()):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(GEN_BG_CHOICES)
    corner_assignments = _corner_multiset()
    color_choices = random.sample([color for color in GEN_COLOR_POOL if color not in {bg, RED}], len(corner_assignments))

    specs = []
    for corner, color in zip(corner_assignments, color_choices):
        sprite_patch = _make_sprite_patch(bg, color)
        quadrant = random.choice(MARKER_QUADRANTS)
        group_patch, sprite_offset = _assemble_group_patch(sprite_patch, corner, quadrant, bg)
        specs.append(
            {
                "corner": corner,
                "color": color,
                "sprite_patch": sprite_patch,
                "group_patch": group_patch,
                "sprite_offset": sprite_offset,
            }
        )

    min_shape = _output_min_shape(specs)
    if min_shape is None:
        return None
    min_h, min_w = min_shape
    h = random.randint(min_h, GEN_MAX_SIZE)
    w = random.randint(min_w, GEN_MAX_SIZE)
    grid = np.full((h, w), bg, dtype=int)

    order = specs[:]
    random.shuffle(order)
    order.sort(key=lambda spec: int(np.count_nonzero(spec["group_patch"] != bg)), reverse=True)
    for spec in order:
        try:
            r0, c0 = random_free_location(
                grid,
                spec["group_patch"],
                transparent=bg,
                padding=GEN_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        spec["group_origin"] = (r0, c0)
        paste(grid, spec["group_patch"], r0, c0, transparent=bg)

    for spec in specs:
        sr, sc = spec["sprite_offset"]
        gh, gw = spec["group_origin"]
        sh, sw = spec["sprite_patch"].shape
        bbox = (gh + sr, gw + sc, gh + sr + sh - 1, gw + sc + sw - 1)
        spec["bbox"] = bbox
        spec["distance"] = _bbox_corner_distance(bbox, grid.shape, spec["corner"])

    if not _same_corner_distances_unique(specs):
        return None

    expected = _build_expected_output(grid.shape, bg, specs)
    if not _semantic_check(grid, specs, bg, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to construct a valid routed-corner ARC sample")
