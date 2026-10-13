from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import itertools
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_arc_utils_032_8", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"could not load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste


BG = 0
PROTO = 2
MATCH = 4
CAND = 8

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
TRANSFORMS = tuple((turns, flip) for turns in range(4) for flip in (False, True))

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (16, 22)
TOTAL_CANDIDATES_RANGE = (4, 6)
MATCHED_COUNT_RANGE = (2, 3)
MAX_RETRIES = 400
MIN_VISIBLE_RAY = 2


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _cropped_component_mask(color_mask: np.ndarray, coords: np.ndarray) -> np.ndarray:
    return crop_bbox(color_mask.astype(int), bbox_coords(coords)).astype(bool)


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    assert bool(mask.any())

    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    mask = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            col_keep.append(c)
    return mask[:, col_keep]


def _transform_mask(mask: np.ndarray, turns: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, int(turns))
    if flip:
        out = np.fliplr(out)
    return out


def _inverse_transform_mask(mask: np.ndarray, turns: int, flip: bool) -> np.ndarray:
    out = np.fliplr(mask) if flip else mask
    return np.rot90(out, -int(turns))


def _transform_direction(direction: tuple[int, int], turns: int, flip: bool) -> tuple[int, int]:
    dr, dc = map(int, direction)
    for _ in range(int(turns) % 4):
        dr, dc = -dc, dr
    if flip:
        dc = -dc
    return int(dr), int(dc)


def _candidate_direction_for_transform(turns: int, flip: bool) -> tuple[int, int]:
    for direction in DIRS:
        if _transform_direction(direction, turns, flip) == RIGHT:
            return direction
    raise AssertionError("direction lookup failed")


def _match_transform(candidate_mask: np.ndarray, prototype_mask: np.ndarray) -> tuple[int, bool] | None:
    for turns in range(4):
        for flip in (False, True):
            transformed = _transform_mask(candidate_mask, turns, flip)
            if transformed.shape == prototype_mask.shape and np.array_equal(transformed, prototype_mask):
                return int(turns), bool(flip)
    return None


def _side_center_from_bbox(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if direction == RIGHT:
        return (r0 + r1) // 2, c1
    if direction == LEFT:
        return (r0 + r1) // 2, c0
    if direction == DOWN:
        return r1, (c0 + c1) // 2
    if direction == UP:
        return r0, (c0 + c1) // 2
    raise ValueError("invalid direction")


def _draw_ray(out: np.ndarray, bbox: tuple[int, int, int, int], direction: tuple[int, int], color: int) -> None:
    r, c = _side_center_from_bbox(bbox, direction)
    dr, dc = direction
    h, w = out.shape
    while 0 <= r < h and 0 <= c < w:
        out[r, c] = int(color)
        r += dr
        c += dc


def _build_compressed_pool() -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_masks: list[np.ndarray] = []
    prototypes: list[np.ndarray] = []
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1, 1 << (h * w)):
                mask = np.array(
                    [(bits >> i) & 1 for i in range(h * w)],
                    dtype=bool,
                ).reshape(h, w)
                area = int(mask.sum())
                if not 3 <= area <= 6:
                    continue
                if area == mask.size:
                    continue
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if len(components_mask(mask, connectivity=4)) != 1:
                    continue
                all_masks.append(mask.copy())

                if not 4 <= area <= 5:
                    continue
                if int(mask[:, -1].sum()) <= int(mask[:, 0].sum()):
                    continue
                symmetry_count = sum(
                    np.array_equal(_transform_mask(mask, turns, flip), mask)
                    for turns, flip in TRANSFORMS
                )
                if symmetry_count == 1:
                    prototypes.append(mask.copy())
    return prototypes, all_masks


PROTO_POOL, COMPRESSED_POOL = _build_compressed_pool()
assert PROTO_POOL, "expected at least one prototype mask"


def _stretch_options(mask: np.ndarray) -> list[np.ndarray]:
    options: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    row_choices = itertools.product((1, 2), repeat=mask.shape[0])
    col_choice_list = list(itertools.product((1, 2), repeat=mask.shape[1]))
    for row_factors in row_choices:
        for col_factors in col_choice_list:
            stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
            if not 3 <= int(stretched.sum()) <= 8:
                continue
            if max(stretched.shape) > 5:
                continue
            key = _mask_key(stretched)
            if key in seen:
                continue
            seen.add(key)
            options.append(stretched.copy())
    return options


def _choose_raw_shape(
    compressed_mask: np.ndarray,
    *,
    direction: tuple[int, int] | None,
    require_stretched: bool,
) -> np.ndarray | None:
    options = _stretch_options(compressed_mask)
    if require_stretched:
        stretched = [option for option in options if option.shape != compressed_mask.shape]
        if not stretched:
            return None
        options = stretched

    if not options:
        return None

    def rank(option: np.ndarray) -> tuple[int, int, int, int]:
        side_score = 0
        odd_score = 0
        if direction is not None:
            sr, sc = _side_center_from_bbox((0, 0, option.shape[0] - 1, option.shape[1] - 1), direction)
            side_score = int(option[sr, sc])
            perp = option.shape[0] if direction in (LEFT, RIGHT) else option.shape[1]
            odd_score = int(perp % 2 == 1)
        stretched_score = int(option.shape != compressed_mask.shape)
        area_score = int(option.sum())
        return side_score, odd_score, stretched_score, area_score

    best = max(rank(option) for option in options)
    candidates = [option for option in options if rank(option) == best]
    return random.choice(candidates).copy()


def _bbox_for(top: int, left: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _ray_cells_outside_bbox(
    grid_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, grid_shape)
    r, c = _side_center_from_bbox(bbox, direction)
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if not (bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3]):
            cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _prototype_signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {
        _mask_key(_compress_runs(_transform_mask(mask, turns, flip)))
        for turns, flip in TRANSFORMS
    }


DIR_TO_TRANSFORMS = {
    direction: [
        (turns, flip)
        for turns, flip in TRANSFORMS
        if _candidate_direction_for_transform(turns, flip) == direction
    ]
    for direction in DIRS
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match run-compressed teal candidates to the red prototype, then recolor the matches and emit orientation-dependent rays.

    Input structure:
    - The grid has a black background, one small red prototype object near the upper-left, and several separated teal candidate objects elsewhere.
    - Prototype and candidates are all 4-connected and are processed via their cropped boolean masks.
    - Some teal candidates are stretched row/column-run variants of the prototype up to rotation or reflection; the rest are unrelated.

    Rule:
    - Crop the red prototype to its bbox and compress consecutive identical rows and then identical columns.
    - Crop and compress each teal candidate the same way, then find the first dihedral transform (0, 90, 180, 270 degrees, preferring no reflection within the same rotation) that maps the candidate's compressed mask onto the prototype's compressed mask.
    - Every matching candidate fires a teal ray from the center of the bbox side indicated by the inverse of that matching transform applied to the prototype's reference direction "right", and then the candidate itself is recolored yellow.
    - Nonmatching candidates stay teal, and the red prototype stays unchanged.

    Key insight:
    - The rotation/reflection used to certify compressed-shape equality is not just a yes/no witness: its inverse encodes the candidate's actual ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    proto_components = components_mask(g == PROTO, connectivity=4)
    cand_components = components_mask(g == CAND, connectivity=4)
    assert len(proto_components) == 1
    assert cand_components

    prototype_mask = _compress_runs(_cropped_component_mask(g == PROTO, proto_components[0]))

    out = g.copy()
    for coords in cand_components:
        bbox = bbox_coords(coords)
        candidate_mask = _compress_runs(_cropped_component_mask(g == CAND, coords))
        match = _match_transform(candidate_mask, prototype_mask)
        if match is None:
            continue
        turns, flip = match
        direction = _candidate_direction_for_transform(turns, flip)
        _draw_ray(out, bbox, direction, CAND)
        out[coords[:, 0], coords[:, 1]] = MATCH

    return out


def _choose_matched_specs(prototype_mask: np.ndarray, matched_count: int) -> list[dict] | None:
    directions = random.sample(list(DIRS), matched_count)
    force_reflection = random.random() < 0.65
    reflected_slot = random.randrange(matched_count) if force_reflection else -1

    specs: list[dict] = []
    stretched_seen = False
    for index, direction in enumerate(directions):
        transform_choices = DIR_TO_TRANSFORMS[direction]
        if index == reflected_slot:
            reflected_choices = [choice for choice in transform_choices if choice[1]]
            if reflected_choices:
                transform_choices = reflected_choices
        turns, flip = random.choice(transform_choices)
        compressed = _inverse_transform_mask(prototype_mask, turns, flip)
        require_stretched = not stretched_seen and index == matched_count - 1
        raw_mask = _choose_raw_shape(compressed, direction=direction, require_stretched=require_stretched)
        if raw_mask is None:
            return None
        stretched_seen = stretched_seen or raw_mask.shape != compressed.shape
        specs.append(
            {
                "matched": True,
                "raw_mask": raw_mask,
                "direction": direction,
            }
        )

    if not stretched_seen:
        for spec in specs:
            compressed = _compress_runs(spec["raw_mask"])
            stretched = _choose_raw_shape(compressed, direction=spec["direction"], require_stretched=True)
            if stretched is not None:
                spec["raw_mask"] = stretched
                stretched_seen = True
                break
    return specs if stretched_seen else None


def _choose_unmatched_specs(prototype_mask: np.ndarray, count: int) -> list[dict] | None:
    prototype_signatures = _prototype_signature_set(prototype_mask)
    unrelated = [
        mask
        for mask in COMPRESSED_POOL
        if _mask_key(_compress_runs(mask)) not in prototype_signatures
    ]
    if not unrelated:
        return None

    specs: list[dict] = []
    for _ in range(count):
        base = random.choice(unrelated)
        oriented = _transform_mask(base, random.randrange(4), random.choice((False, True)))
        raw_mask = _choose_raw_shape(oriented, direction=None, require_stretched=random.random() < 0.35)
        if raw_mask is None:
            return None
        specs.append({"matched": False, "raw_mask": raw_mask})
    return specs


def _try_place_scene(grid_shape: tuple[int, int], prototype_mask: np.ndarray, candidate_specs: list[dict]) -> list[dict] | None:
    h, w = map(int, grid_shape)
    proto_h, proto_w = map(int, prototype_mask.shape)
    proto_positions = [
        (r, c)
        for r in range(1, min(4, h - proto_h + 1))
        for c in range(1, min(4, w - proto_w + 1))
    ]
    random.shuffle(proto_positions)
    candidate_specs = sorted(candidate_specs, key=lambda spec: int(spec["raw_mask"].sum()), reverse=True)

    for proto_top, proto_left in proto_positions:
        placements = [
            {
                "matched": False,
                "is_prototype": True,
                "raw_mask": prototype_mask,
                "top": proto_top,
                "left": proto_left,
                "bbox": _bbox_for(proto_top, proto_left, prototype_mask.shape),
            }
        ]
        placed_boxes = [placements[0]["bbox"]]
        reserved_ray_cells: set[tuple[int, int]] = set()

        success = True
        for spec in candidate_specs:
            mask = spec["raw_mask"]
            obj_h, obj_w = map(int, mask.shape)
            positions = [
                (r, c)
                for r in range(0, h - obj_h + 1)
                for c in range(0, w - obj_w + 1)
            ]
            random.shuffle(positions)

            placed = None
            for top, left in positions:
                bbox = _bbox_for(top, left, mask.shape)
                if any(_boxes_overlap(_expand_bbox(bbox, 1), other) for other in placed_boxes):
                    continue

                bbox_cells = {
                    (rr, cc)
                    for rr in range(bbox[0], bbox[2] + 1)
                    for cc in range(bbox[1], bbox[3] + 1)
                }
                if bbox_cells & reserved_ray_cells:
                    continue

                new_ray_cells: list[tuple[int, int]] = []
                if spec["matched"]:
                    new_ray_cells = _ray_cells_outside_bbox(grid_shape, bbox, spec["direction"])
                    if len(new_ray_cells) < MIN_VISIBLE_RAY:
                        continue
                    if any(cell in reserved_ray_cells for cell in new_ray_cells):
                        continue
                    if any(
                        bbox0[0] <= rr <= bbox0[2] and bbox0[1] <= cc <= bbox0[3]
                        for rr, cc in new_ray_cells
                        for bbox0 in placed_boxes
                    ):
                        continue

                placed = {
                    **spec,
                    "top": top,
                    "left": left,
                    "bbox": bbox,
                    "ray_cells": new_ray_cells,
                }
                break

            if placed is None:
                success = False
                break

            placements.append(placed)
            placed_boxes.append(placed["bbox"])
            reserved_ray_cells.update(placed.get("ray_cells", []))

        if success:
            return placements
    return None


def _render_scene(grid_shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for placement in placements:
        color = PROTO if placement.get("is_prototype") else CAND
        patch = np.where(placement["raw_mask"], color, BG)
        paste(grid, patch, placement["top"], placement["left"], transparent=BG)
    return grid


def _expected_output(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        if placement.get("is_prototype") or not placement["matched"]:
            continue
        _draw_ray(out, placement["bbox"], placement["direction"], CAND)
        top, left = placement["top"], placement["left"]
        coords = np.argwhere(placement["raw_mask"])
        out[top + coords[:, 0], left + coords[:, 1]] = MATCH
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    prototype_mask = random.choice(PROTO_POOL).copy()

    total_candidates = random.randint(*TOTAL_CANDIDATES_RANGE)
    matched_count = random.randint(MATCHED_COUNT_RANGE[0], min(MATCHED_COUNT_RANGE[1], total_candidates - 1))
    unmatched_count = total_candidates - matched_count

    matched_specs = _choose_matched_specs(prototype_mask, matched_count)
    if matched_specs is None:
        return None
    unmatched_specs = _choose_unmatched_specs(prototype_mask, unmatched_count)
    if unmatched_specs is None:
        return None

    candidate_specs = matched_specs + unmatched_specs
    random.shuffle(candidate_specs)
    placements = _try_place_scene((h, w), prototype_mask, candidate_specs)
    if placements is None:
        return None

    grid = _render_scene((h, w), placements)
    expected = _expected_output(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
