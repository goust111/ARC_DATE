from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg


DIRS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

BG = Color.BLACK
RECT = Color.GREY
SLOT = 5
HOLE_STRIDE = 6
RECT_SIZE = 19
BASE_SIZE = 29
RECT_TOP = 5
RECT_LEFT = 5
INNER_TOP = RECT_TOP + 1
INNER_LEFT = RECT_LEFT + 1
RECT_CENTER = RECT_TOP + RECT_SIZE // 2
OUTER_CENTER_TOP = 12
OUTER_CENTER_LEFT = 12
PALETTE = [color for color in range(10) if color not in (BG, RECT)]
SHAPE_MIN_AREA = 12
SHAPE_MAX_AREA = 18
GEN_ATTEMPTS = 400


def _sgn(value: float) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _direction_from_point(point_r: float, point_c: float, center_r: float, center_c: float) -> tuple[int, int]:
    direction = (_sgn(point_r - center_r), _sgn(point_c - center_c))
    assert direction != (0, 0), "direction must be one of the eight compass offsets"
    return direction


def _occupied_patch(coords: np.ndarray, bbox_rc: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_rc)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = True
    return patch


def _color_counter(grid: np.ndarray, coords: np.ndarray) -> Counter:
    return Counter(int(grid[int(r), int(c)]) for r, c in coords)


def _extract_rectangle_and_fragments(grid: np.ndarray, bg: int) -> tuple[dict, list[dict]]:
    rect_coords = np.argwhere(grid == RECT)
    assert rect_coords.size > 0, "expected gray rectangle pixels"
    rr0, cc0 = rect_coords.min(axis=0)
    rr1, cc1 = rect_coords.max(axis=0)
    rect_comp = {
        "color": int(RECT),
        "coords": rect_coords,
        "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
        "area": int(rect_coords.shape[0]),
    }

    fragments_grid = grid.copy()
    for r, c in rect_coords:
        fragments_grid[int(r), int(c)] = bg
    fragments = find_components(fragments_grid, bg=bg, connectivity=8, monochromatic=False)
    assert fragments, "expected at least one fragment outside the rectangle"
    return rect_comp, fragments


def _extract_holes(grid: np.ndarray, rect_comp: dict, bg: int) -> dict[tuple[int, int], np.ndarray]:
    rect_bbox = rect_comp["bbox"]
    rect_patch = crop_bbox(grid, rect_bbox)
    assert np.all(rect_patch[0, :] == RECT) and np.all(rect_patch[-1, :] == RECT)
    assert np.all(rect_patch[:, 0] == RECT) and np.all(rect_patch[:, -1] == RECT)

    hole_grid = (rect_patch == bg).astype(int)
    hole_comps = find_components(hole_grid, bg=0, connectivity=4, monochromatic=True)
    rect_center_r = (rect_patch.shape[0] - 1) / 2.0
    rect_center_c = (rect_patch.shape[1] - 1) / 2.0
    holes: dict[tuple[int, int], np.ndarray] = {}
    rect_r0, rect_c0, _rect_r1, _rect_c1 = map(int, rect_bbox)

    for comp in hole_comps:
        hr0, hc0, hr1, hc1 = map(int, comp["bbox"])
        direction = _direction_from_point((hr0 + hr1) / 2.0, (hc0 + hc1) / 2.0, rect_center_r, rect_center_c)
        assert direction not in holes, "hole directions must be unique"
        holes[direction] = np.array([(rect_r0 + int(r), rect_c0 + int(c)) for r, c in comp["coords"]], dtype=int)

    assert holes, "expected at least one interior black hole"
    return holes


def _extract_fragment_info(grid: np.ndarray, comp: dict) -> dict:
    counts = _color_counter(grid, comp["coords"])
    assert len(counts) == 2, "every fragment must use one main color and one marker color"
    singleton_colors = [color for color, count in counts.items() if count == 1]
    assert len(singleton_colors) == 1, "expected exactly one marker pixel in each fragment"
    marker_color = singleton_colors[0]
    main_colors = [color for color, count in counts.items() if color != marker_color]
    assert len(main_colors) == 1 and counts[main_colors[0]] >= 2
    main_color = main_colors[0]

    marker_coord = next((int(r), int(c)) for r, c in comp["coords"] if int(grid[int(r), int(c)]) == marker_color)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    direction = _direction_from_point(marker_coord[0], marker_coord[1], (r0 + r1) / 2.0, (c0 + c1) / 2.0)

    return {
        "bbox": comp["bbox"],
        "coords": comp["coords"],
        "direction": direction,
        "main_color": int(main_color),
        "marker_color": int(marker_color),
        "shape": _occupied_patch(comp["coords"], comp["bbox"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Use each fragment's internal marker direction to color the matching hole inside the gray rectangle.

    Input structure:
    - A black/dominant background contains one large gray rectangle whose border stays solid gray.
    - Inside that rectangle are several disconnected black holes, each occupying a unique compass direction relative to the rectangle center.
    - Outside the rectangle are separated multicolor fragments; each fragment has one main color plus exactly one differently colored marker pixel.

    Rule:
    - For every fragment, compute the sign of the marker pixel's row/column offset from that fragment's bbox center to obtain one of the eight compass directions.
    - Find the hole whose own bbox center lies in the same direction from the gray rectangle center, and recolor every black cell of that hole with the fragment's main color.
    - Erase every fragment back to background, keep the gray rectangle unchanged, and leave any unaddressed holes black.

    Key insight:
    - The fragments are direction codes rather than puzzle pieces: shape matches can be decoys, while the only reliable address is the hidden marker pixel inside each fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    rect_comp, fragment_comps = _extract_rectangle_and_fragments(g, bg)
    holes = _extract_holes(g, rect_comp, bg)

    out = g.copy()
    used_dirs: set[tuple[int, int]] = set()
    for comp in fragment_comps:
        info = _extract_fragment_info(g, comp)
        direction = info["direction"]
        assert direction in holes, "every fragment direction must address an existing hole"
        assert direction not in used_dirs, "fragment directions must be unique"
        used_dirs.add(direction)

        for r, c in info["coords"]:
            out[int(r), int(c)] = bg
        for r, c in holes[direction]:
            out[int(r), int(c)] = info["main_color"]

    return out


def _dir_to_hole_top(direction: tuple[int, int]) -> tuple[int, int]:
    return INNER_TOP + (direction[0] + 1) * HOLE_STRIDE, INNER_LEFT + (direction[1] + 1) * HOLE_STRIDE


OUTER_SLOT_TOPS = {
    (-1, -1): (0, 0),
    (-1, 0): (0, OUTER_CENTER_LEFT),
    (-1, 1): (0, BASE_SIZE - SLOT),
    (0, -1): (OUTER_CENTER_TOP, 0),
    (0, 1): (OUTER_CENTER_TOP, BASE_SIZE - SLOT),
    (1, -1): (BASE_SIZE - SLOT, 0),
    (1, 0): (BASE_SIZE - SLOT, OUTER_CENTER_LEFT),
    (1, 1): (BASE_SIZE - SLOT, BASE_SIZE - SLOT),
}


BUCKET_CELLS = {
    direction: [
        (r, c)
        for r in range(SLOT)
        for c in range(SLOT)
        if (r, c) != (SLOT // 2, SLOT // 2)
        and _direction_from_point(r, c, SLOT // 2, SLOT // 2) == direction
    ]
    for direction in DIRS
}


def _mask_connected(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True)
    return len(comps) == 1


def _mask_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_has_all_buckets(mask: np.ndarray) -> bool:
    for direction, cells in BUCKET_CELLS.items():
        if not any(mask[r, c] for r, c in cells):
            return False
    return True


def _shape_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _shape_structure_ok(mask: np.ndarray) -> bool:
    return _mask_has_full_bbox(mask) and _mask_has_all_buckets(mask) and _mask_connected(mask)


def _shape_ok(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    return SHAPE_MIN_AREA <= area <= SHAPE_MAX_AREA and _shape_structure_ok(mask)


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(300):
        mask = np.ones((SLOT, SLOT), dtype=bool)
        cells = [(r, c) for r in range(SLOT) for c in range(SLOT)]
        rng.shuffle(cells)
        target_area = rng.randint(SHAPE_MIN_AREA, SHAPE_MAX_AREA)
        for r, c in cells:
            if int(mask.sum()) <= target_area:
                break
            mask[r, c] = False
            if not _shape_structure_ok(mask):
                mask[r, c] = True
        if _shape_ok(mask):
            return mask
    raise RuntimeError("failed to sample a valid 5x5 hole shape")


def _distinct_shapes(count: int, rng: random.Random) -> dict[tuple[int, int], np.ndarray]:
    shapes: dict[tuple[int, int], np.ndarray] = {}
    seen: set[bytes] = set()
    for direction in rng.sample(list(DIRS), count):
        while True:
            mask = _random_shape(rng)
            sig = _shape_signature(mask)
            if sig not in seen:
                seen.add(sig)
                shapes[direction] = mask
                break
    return shapes


def _marker_candidates(mask: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    return [(r, c) for r, c in BUCKET_CELLS[direction] if mask[r, c]]


def _deranged_targets(targets: list[tuple[int, int]], pool: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    assert len(targets) == len(pool) and len(targets) >= 2
    for _ in range(200):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        if any(a != b for a, b in zip(targets, shuffled)):
            return shuffled
    raise RuntimeError("failed to create a non-identity assignment")


def _render_fragment(mask: np.ndarray, direction: tuple[int, int], main_color: int, marker_color: int, rng: random.Random) -> np.ndarray:
    patch = np.zeros((SLOT, SLOT), dtype=int)
    patch[mask] = int(main_color)
    marker_r, marker_c = rng.choice(_marker_candidates(mask, direction))
    patch[marker_r, marker_c] = int(marker_color)
    return patch


def _render_base_input(meta: dict) -> np.ndarray:
    grid = np.zeros((BASE_SIZE, BASE_SIZE), dtype=int)
    grid[RECT_TOP : RECT_TOP + RECT_SIZE, RECT_LEFT : RECT_LEFT + RECT_SIZE] = RECT

    for direction, mask in meta["holes"].items():
        r0, c0 = _dir_to_hole_top(direction)
        view = grid[r0 : r0 + SLOT, c0 : c0 + SLOT]
        view[mask] = BG

    for frag in meta["fragments"]:
        r0, c0 = OUTER_SLOT_TOPS[frag["placement_dir"]]
        grid[r0 : r0 + SLOT, c0 : c0 + SLOT] = frag["patch"]
    return grid


def _expected_output(meta: dict, input_grid: np.ndarray) -> np.ndarray:
    out = input_grid.copy()
    row_off, col_off = meta["offset"]
    for frag in meta["fragments"]:
        r0, c0 = OUTER_SLOT_TOPS[frag["placement_dir"]]
        out[row_off + r0 : row_off + r0 + SLOT, col_off + c0 : col_off + c0 + SLOT] = BG
        hr0, hc0 = _dir_to_hole_top(frag["target_dir"])
        out[
            row_off + hr0 : row_off + hr0 + SLOT,
            col_off + hc0 : col_off + hc0 + SLOT,
        ][meta["holes"][frag["target_dir"]]] = frag["main_color"]
    return out


def _naive_shape_output(meta: dict, input_grid: np.ndarray) -> np.ndarray:
    out = input_grid.copy()
    row_off, col_off = meta["offset"]
    for frag in meta["fragments"]:
        r0, c0 = OUTER_SLOT_TOPS[frag["placement_dir"]]
        out[row_off + r0 : row_off + r0 + SLOT, col_off + c0 : col_off + c0 + SLOT] = BG
        hr0, hc0 = _dir_to_hole_top(frag["source_dir"])
        out[
            row_off + hr0 : row_off + hr0 + SLOT,
            col_off + hc0 : col_off + hc0 + SLOT,
        ][meta["holes"][frag["source_dir"]]] = frag["main_color"]
    return out


def _embed_with_padding(base: np.ndarray, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    options = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    top, bottom, left, right = rng.choice(options)
    grid = np.zeros((base.shape[0] + top + bottom, base.shape[1] + left + right), dtype=int)
    grid[top : top + base.shape[0], left : left + base.shape[1]] = base
    return grid, (top, left)


def _build_meta(rng: random.Random) -> dict:
    hole_count = rng.randint(4, 6)
    fragment_count = rng.randint(3, hole_count)

    hole_shapes = _distinct_shapes(hole_count, rng)
    hole_dirs = list(hole_shapes.keys())
    rng.shuffle(hole_dirs)
    target_dirs = hole_dirs[:fragment_count]

    source_pool = rng.sample(hole_dirs, fragment_count)
    source_dirs = _deranged_targets(target_dirs, source_pool, rng)
    placement_pool = rng.sample(list(DIRS), fragment_count)
    placement_dirs = _deranged_targets(target_dirs, placement_pool, rng)

    main_colors = rng.sample(PALETTE, fragment_count)
    fragments = []
    for target_dir, source_dir, placement_dir, main_color in zip(target_dirs, source_dirs, placement_dirs, main_colors):
        marker_color = rng.choice([color for color in PALETTE if color != main_color])
        patch = _render_fragment(hole_shapes[source_dir], target_dir, main_color, marker_color, rng)
        fragments.append(
            {
                "target_dir": target_dir,
                "source_dir": source_dir,
                "placement_dir": placement_dir,
                "main_color": int(main_color),
                "marker_color": int(marker_color),
                "patch": patch,
            }
        )

    return {"holes": hole_shapes, "fragments": fragments}


def _validate_semantics(grid: np.ndarray, meta: dict) -> bool:
    expected = _expected_output(meta, grid)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(_naive_shape_output(meta, grid), expected):
        return False
    if not any(frag["target_dir"] != frag["source_dir"] for frag in meta["fragments"]):
        return False
    if not any(frag["target_dir"] != frag["placement_dir"] for frag in meta["fragments"]):
        return False

    bg = int(get_bg(grid))
    rect_comp, fragment_comps = _extract_rectangle_and_fragments(grid, bg)
    holes = _extract_holes(grid, rect_comp, bg)
    if set(holes) != set(meta["holes"]):
        return False
    parsed_fragments = [_extract_fragment_info(grid, comp) for comp in fragment_comps]
    if {frag["direction"] for frag in parsed_fragments} != {frag["target_dir"] for frag in meta["fragments"]}:
        return False

    hole_signatures = {direction: _shape_signature(mask) for direction, mask in meta["holes"].items()}
    decoy_hits = 0
    for frag in parsed_fragments:
        matches = [direction for direction, sig in hole_signatures.items() if sig == _shape_signature(frag["shape"])]
        if len(matches) != 1:
            return False
        if matches[0] != frag["direction"]:
            decoy_hits += 1
    return decoy_hits >= 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        meta = _build_meta(rng)
        base = _render_base_input(meta)
        grid, offset = _embed_with_padding(base, rng)
        meta["offset"] = offset
        if _validate_semantics(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid direction-code hole-filling puzzle")
