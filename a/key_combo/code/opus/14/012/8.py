from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SPRITE_COLORS = (BLUE, RED, YELLOW)
GEN_ATTEMPTS = 400
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
FRAME_H_RANGE = (12, 14)
FRAME_W_RANGE = (11, 13)
RECT_H_RANGE = (4, 6)
RECT_W_RANGE = (4, 6)


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    variants: list[np.ndarray] = []
    base = mask.astype(bool)
    for flipped in (base, np.fliplr(base)):
        for k in range(4):
            variant = np.rot90(flipped, k)
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((variant.shape, variant.tobytes()) for variant in _all_variants(mask))


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return (int(r0), int(c0), int(r1), int(c1)), mask


def _row_overlap_len(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0])) + 1)


def _round_half_up_ratio(src_index: int, src_len: int, dst_len: int) -> int:
    if dst_len <= 1 or src_len <= 1:
        return 0
    num = 2 * int(src_index) * (int(dst_len) - 1) + (int(src_len) - 1)
    den = 2 * (int(src_len) - 1)
    return int(num // den)


def _map_marker_mask(src_marker: np.ndarray, dst_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = map(int, src_marker.shape)
    dst_h, dst_w = map(int, dst_shape)
    out = np.zeros((dst_h, dst_w), dtype=bool)
    for r, c in np.argwhere(src_marker):
        rr = _round_half_up_ratio(int(r), src_h, dst_h)
        cc = _round_half_up_ratio(int(c), src_w, dst_w)
        out[rr, cc] = True
    return out


def _find_frame(grid: np.ndarray) -> dict:
    comps = find_components(
        grid,
        bg=BLACK,
        colors=[GREEN],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected a green frame"
    max_area = max(int(comp["area"]) for comp in comps)
    winners = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(winners) == 1, "expected one largest green frame"
    return winners[0]


def _extract_holes(frame_sub: np.ndarray) -> list[dict]:
    holes: list[dict] = []
    for coords in components_mask(frame_sub == BLACK, connectivity=4):
        bbox, mask = _coords_to_mask(coords)
        holes.append(
            {
                "bbox": bbox,
                "mask": mask,
                "key": _canonical_key(mask),
            }
        )
    holes.sort(key=lambda hole: hole["bbox"])
    return holes


def _extract_sprites(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    sprites: list[dict] = []
    for comp in find_components(
        grid,
        bg=BLACK,
        colors=list(SPRITE_COLORS),
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if not (r1 < fr0 or r0 > fr1 or c1 < fc0 or c0 > fc1):
            continue
        _, mask = _coords_to_mask(comp["coords"])
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "key": _canonical_key(mask),
                "color": int(comp["color"]),
            }
        )
    return sprites


def _extract_marker_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for coords in components_mask(np.isin(grid, [GREY, ORANGE]), connectivity=4):
        bbox, _ = _coords_to_mask(coords)
        sub = crop_bbox(grid, bbox)
        if not np.all(np.isin(sub, [GREY, ORANGE])):
            continue
        if ORANGE not in set(map(int, sub.flat)):
            continue
        rects.append({"bbox": bbox, "sub": sub})
    return rects


def _solve(grid: np.ndarray, *, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BLACK, "expected black background"

    frame = _find_frame(g)
    frame_bbox = tuple(map(int, frame["bbox"]))
    fr0, fc0, fr1, fc1 = frame_bbox
    frame_sub = crop_bbox(g, frame_bbox)
    holes = _extract_holes(frame_sub)
    assert 2 <= len(holes) <= 3, "expected 2-3 holes"

    sprites = _extract_sprites(g, frame_bbox)
    assert len(sprites) >= len(holes), "expected at least one sprite per hole"
    sprites_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for sprite in sprites:
        sprites_by_key.setdefault(sprite["key"], []).append(sprite)

    rects = _extract_marker_rectangles(g)
    assert rects, "expected marker rectangles"
    overlaps = [(_row_overlap_len(rect["bbox"], frame_bbox), idx) for idx, rect in enumerate(rects)]
    best_overlap = max(value for value, _ in overlaps)
    winners = [idx for value, idx in overlaps if value == best_overlap]
    assert best_overlap > 0 and len(winners) == 1, "expected one marker rectangle with unique max row overlap"
    source = rects[winners[0]]
    source_marker = source["sub"] == ORANGE

    out = g.copy()
    details = {"frame_bbox": frame_bbox, "holes": [], "source_bbox": source["bbox"], "rects": rects}
    for hole in holes:
        matches = sprites_by_key.get(hole["key"], [])
        assert len(matches) == 1, "expected exactly one matching external sprite"
        base_color = int(matches[0]["color"])
        hr0, hc0, hr1, hc1 = map(int, hole["bbox"])
        hole_view = out[fr0 + hr0 : fr0 + hr1 + 1, fc0 + hc0 : fc0 + hc1 + 1]
        hole_view[hole["mask"]] = base_color

        mapped = _map_marker_mask(source_marker, hole["mask"].shape)
        applied = mapped & hole["mask"]
        hole_view[applied] = ORANGE

        details["holes"].append(
            {
                "bbox": hole["bbox"],
                "mask": hole["mask"],
                "base_color": base_color,
                "sprite_bbox": matches[0]["bbox"],
                "mapped_marker": mapped,
                "applied_marker": applied,
                "ignored_count": int(mapped.sum() - applied.sum()),
            }
        )

    if return_details:
        return out, details
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hole in the green template by combining shape-matched sprite color with edge-relative marker transfer from the best-overlapping rectangle.

    Input structure:
    - The grid has a black background with one large green solid rectangle containing 2-3 black hole components.
    - Outside the green rectangle are 2-3 small monochrome sprites in blue/red/yellow whose shapes match the holes under rotation or flip.
    - Outside the frame are grey dense rectangles carrying orange marker pixels; multiple such rectangles may exist.

    Rule:
    - Detect the large green rectangle and identify each black hole inside it.
    - For every hole, find the unique outside sprite with the same shape up to rotation or reflection, and use that sprite's color as the hole's base fill.
    - Among the grey-orange rectangles, choose the one whose row span overlaps the frame's row span the most.
    - Map that rectangle's orange marker pixels into each hole bbox by preserving edge-relative row and column positions with nearest-integer rounding.
    - Fill the whole hole with its matched base color, then overwrite mapped positions that still land inside the hole mask with orange; leave everything else unchanged.

    Key insight:
    - Shape matching answers which color each hole should receive, while the selected marker rectangle answers where orange accents should be placed; both mechanisms are required to recover the final filled holes.
    """
    return _solve(grid)


SHAPE_LIBRARY = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]

PATTERN_KINDS = (0, 1, 2, 3, 4)


def _pattern_mask(h: int, w: int, kind: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if kind == 0:
        mask[0, 0] = True
        mask[0, w - 1] = True
        mask[h - 1, 0] = True
        mask[h - 1, w - 1] = True
        mask[h // 2, min(1, w - 1)] = True
    elif kind == 1:
        mask[0, : min(3, w)] = True
        mask[: min(3, h), 0] = True
        mask[h - 1, w - 1] = True
    elif kind == 2:
        mask[: min(3, h), w - 1] = True
        mask[h - 1, max(0, w - 3) : w] = True
        mask[min(h - 1, 1), 0] = True
    elif kind == 3:
        mask[0, 0] = True
        mask[min(h - 1, 1), min(w - 1, 1)] = True
        mask[h - 1, w - 1] = True
        mask[max(0, h - 2), 0] = True
        mask[0, max(0, w - 2)] = True
    else:
        mask[0, w // 2] = True
        mask[h // 2, 0] = True
        mask[h - 1, w // 2] = True
        mask[h // 2, w - 1] = True
        mask[h - 1, w - 1] = True
    return mask


def _can_place_hole(reserved: np.ndarray, shape: np.ndarray, top: int, left: int, frame_shape: tuple[int, int]) -> bool:
    h, w = frame_shape
    sh, sw = shape.shape
    if top < 1 or left < 1 or top + sh > h - 1 or left + sw > w - 1:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + sh + 1)
    c1 = min(w, left + sw + 1)
    if reserved[r0:r1, c0:c1].any():
        return False
    return True


def _place_holes(rng: random.Random, frame_h: int, frame_w: int, count: int) -> list[dict] | None:
    chosen = rng.sample(SHAPE_LIBRARY, count)
    holes: list[dict] = []
    slots = [
        (1, 1, 4, 4),
        (1, frame_w - 5, 4, 4),
        (frame_h - 5, 2, 4, 4),
    ]
    selected_slots = rng.sample(slots, count)
    transformed_count = 0
    for base_shape, slot in zip(chosen, selected_slots):
        variants = _all_variants(base_shape)
        rng.shuffle(variants)
        slot_r0, slot_c0, slot_h, slot_w = slot
        shape = next((variant for variant in variants if variant.shape[0] <= slot_h and variant.shape[1] <= slot_w), None)
        if shape is None:
            return None
        sh, sw = shape.shape
        top = slot_r0 + rng.randint(0, slot_h - sh)
        left = slot_c0 + rng.randint(0, slot_w - sw)
        if not np.array_equal(shape, base_shape):
            transformed_count += 1
        holes.append(
            {
                "shape": shape.copy(),
                "bbox": (top, left, top + sh - 1, left + sw - 1),
                "key": _canonical_key(shape),
            }
        )
    if len({hole["key"] for hole in holes}) != count:
        return None
    if transformed_count == 0:
        return None
    return holes


def _choose_sprite_orientations(rng: random.Random, holes: list[dict]) -> list[dict]:
    sprites: list[dict] = []
    changed = 0
    colors = rng.sample(list(SPRITE_COLORS), len(holes))
    order = list(range(len(holes)))
    rng.shuffle(order)
    for color, idx in zip(colors, order):
        hole_shape = holes[idx]["shape"]
        variants = _all_variants(hole_shape)
        rng.shuffle(variants)
        sprite_shape = variants[0].copy()
        if not np.array_equal(sprite_shape, hole_shape):
            changed += 1
        sprites.append({"hole_index": idx, "color": color, "shape": sprite_shape})
    if changed == 0:
        for sprite in sprites:
            hole_shape = holes[sprite["hole_index"]]["shape"]
            alternatives = [variant for variant in _all_variants(hole_shape) if not np.array_equal(variant, hole_shape)]
            if alternatives:
                sprite["shape"] = rng.choice(alternatives).copy()
                changed = 1
                break
    if changed == 0:
        return []
    rng.shuffle(sprites)
    return sprites


def _stack_sprite_rows(rng: random.Random, frame_top: int, frame_h: int, sprites: list[dict]) -> list[int] | None:
    total = sum(int(sprite["shape"].shape[0]) for sprite in sprites) + (len(sprites) - 1)
    if total > frame_h:
        return None
    slack = frame_h - total
    gaps = [0] * (len(sprites) + 1)
    for _ in range(slack):
        gaps[rng.randrange(len(gaps))] += 1
    rows: list[int] = []
    cursor = frame_top + gaps[0]
    for idx, sprite in enumerate(sprites):
        rows.append(cursor)
        cursor += int(sprite["shape"].shape[0]) + 1 + gaps[idx + 1]
    return rows


def _render_case(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    holes: list[dict],
    sprites: list[dict],
    sprite_positions: list[tuple[int, int]],
    rects: list[dict],
) -> np.ndarray:
    h, w = grid_shape
    grid = np.zeros((h, w), dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = GREEN
    for hole in holes:
        hr0, hc0, _hr1, _hc1 = hole["bbox"]
        shape = hole["shape"]
        sh, sw = shape.shape
        view = grid[fr0 + hr0 : fr0 + hr0 + sh, fc0 + hc0 : fc0 + hc0 + sw]
        view[shape] = BLACK
    for sprite, (sr0, sc0) in zip(sprites, sprite_positions):
        shape = sprite["shape"]
        sh, sw = shape.shape
        grid[sr0 : sr0 + sh, sc0 : sc0 + sw][shape] = int(sprite["color"])
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0 : c1 + 1][rect["marker"]] = ORANGE
    return grid


def _latent_output(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    holes: list[dict],
    sprites: list[dict],
    source_rect: dict,
) -> tuple[np.ndarray, dict]:
    out = grid.copy()
    fr0, fc0, _fr1, _fc1 = frame_bbox
    by_hole = {sprite["hole_index"]: sprite for sprite in sprites}
    total_applied = 0
    total_ignored = 0
    hole_stats = []
    for idx, hole in enumerate(holes):
        hr0, hc0, hr1, hc1 = hole["bbox"]
        hole_shape = hole["shape"]
        sprite = by_hole[idx]
        view = out[fr0 + hr0 : fr0 + hr1 + 1, fc0 + hc0 : fc0 + hc1 + 1]
        view[hole_shape] = int(sprite["color"])
        mapped = _map_marker_mask(source_rect["marker"], hole_shape.shape)
        applied = mapped & hole_shape
        view[applied] = ORANGE
        applied_count = int(applied.sum())
        ignored_count = int(mapped.sum() - applied_count)
        total_applied += applied_count
        total_ignored += ignored_count
        hole_stats.append({"applied": applied_count, "ignored": ignored_count, "mapped": mapped})
    return out, {"applied": total_applied, "ignored": total_ignored, "holes": hole_stats}


def _patterns_distinguish_holes(source_rect: dict, distractor_rect: dict, holes: list[dict]) -> bool:
    for hole in holes:
        src = _map_marker_mask(source_rect["marker"], hole["shape"].shape) & hole["shape"]
        dst = _map_marker_mask(distractor_rect["marker"], hole["shape"].shape) & hole["shape"]
        if not np.array_equal(src, dst):
            return True
    return False


def _generate_case(rng: random.Random) -> np.ndarray | None:
    frame_h = rng.randint(*FRAME_H_RANGE)
    frame_w = rng.randint(*FRAME_W_RANGE)
    hole_count = rng.choice([2, 3])
    holes = _place_holes(rng, frame_h, frame_w, hole_count)
    if holes is None:
        return None

    sprites = _choose_sprite_orientations(rng, holes)
    if not sprites:
        return None

    max_sprite_w = max(int(sprite["shape"].shape[1]) for sprite in sprites)
    src_h = rng.randint(*RECT_H_RANGE)
    src_w = rng.randint(*RECT_W_RANGE)
    dst_h = rng.randint(*RECT_H_RANGE)
    dst_w = rng.randint(*RECT_W_RANGE)
    src_kind, dst_kind = rng.sample(PATTERN_KINDS, 2)
    source_marker = _pattern_mask(src_h, src_w, src_kind)
    distractor_marker = _pattern_mask(dst_h, dst_w, dst_kind)

    source_overlap = rng.randint(2, min(3, src_h, frame_h - 1))
    distractor_overlap = rng.choice([0, 0, 1, max(0, source_overlap - 1)])
    if distractor_overlap >= source_overlap:
        distractor_overlap = source_overlap - 1
    if distractor_overlap < 0:
        return None

    distractor_gap = rng.randint(1, 2) if distractor_overlap == 0 else 0
    if distractor_overlap == 0:
        bottom_extension = src_h - source_overlap + distractor_gap + dst_h
    else:
        bottom_extension = max(src_h - source_overlap, dst_h - distractor_overlap)
    col_gap = 2
    first_band_w = max(src_w, max_sprite_w)
    second_band_w = max(dst_w, max_sprite_w)
    right_band_w = first_band_w + col_gap + second_band_w + 1
    min_grid_h = max(GRID_H_RANGE[0], frame_h + bottom_extension + 4)
    min_grid_w = max(GRID_W_RANGE[0], frame_w + right_band_w + 5)
    if min_grid_h > GRID_H_RANGE[1] or min_grid_w > GRID_W_RANGE[1]:
        return None
    grid_h = rng.randint(min_grid_h, GRID_H_RANGE[1])
    grid_w = rng.randint(min_grid_w, GRID_W_RANGE[1])
    frame_top_max = grid_h - frame_h - bottom_extension - 2
    frame_left_max = grid_w - frame_w - right_band_w - 2
    if frame_top_max < 1 or frame_left_max < 1:
        return None
    frame_top = rng.randint(1, frame_top_max)
    frame_left = rng.randint(1, frame_left_max)
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    source_r0 = frame_bbox[2] - source_overlap + 1
    if distractor_overlap == 0:
        distractor_r0 = source_r0 + src_h + distractor_gap
    else:
        distractor_r0 = frame_bbox[2] - distractor_overlap + 1

    source_c0 = frame_bbox[3] + 3
    second_col = source_c0 + first_band_w + col_gap
    distractor_c0 = source_c0 if distractor_overlap == 0 else second_col
    if second_col + second_band_w >= grid_w - 1:
        return None

    sprite_top = frame_top + 1
    lower_sprite_top = sprite_top + 4
    max_sprite_h = max(int(sprite["shape"].shape[0]) for sprite in sprites)
    if lower_sprite_top + max_sprite_h >= source_r0 - 1:
        return None
    layout_slots = [(sprite_top, source_c0), (sprite_top, second_col), (lower_sprite_top, source_c0)]
    sprite_positions = [layout_slots[i] for i in range(len(sprites))]

    rects = [
        {
            "bbox": (source_r0, source_c0, source_r0 + src_h - 1, source_c0 + src_w - 1),
            "marker": source_marker,
            "is_source": True,
        },
        {
            "bbox": (distractor_r0, distractor_c0, distractor_r0 + dst_h - 1, distractor_c0 + dst_w - 1),
            "marker": distractor_marker,
            "is_source": False,
        },
    ]

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        if r0 < 1 or c0 < 1 or r1 >= grid_h - 1 or c1 >= grid_w - 1:
            return None

    for (sr0, sc0), sprite in zip(sprite_positions, sprites):
        sh, sw = sprite["shape"].shape
        if sr0 + sh >= grid_h - 1 or sc0 + sw >= grid_w - 1:
            return None
    if len(sprite_positions) >= 2:
        for (a_r, a_c), (b_r, b_c) in zip(sprite_positions, sprite_positions[1:]):
            if a_r == b_r and a_c == b_c:
                return None
    if distractor_overlap > 0:
        source_box = rects[0]["bbox"]
        distractor_box = rects[1]["bbox"]
        if not (source_box[3] < distractor_box[1] or distractor_box[3] < source_box[1]):
            return None

    grid = _render_case((grid_h, grid_w), frame_bbox, holes, sprites, sprite_positions, rects)
    expected, latent = _latent_output(grid, frame_bbox, holes, sprites, rects[0])
    if latent["applied"] == 0 or latent["ignored"] == 0:
        return None
    if not _patterns_distinguish_holes(rects[0], rects[1], holes):
        return None

    overlaps = [_row_overlap_len(rect["bbox"], frame_bbox) for rect in rects]
    if overlaps[0] <= overlaps[1]:
        return None

    solver_out, details = _solve(grid, return_details=True)
    if not np.array_equal(solver_out, expected):
        return None
    if details["source_bbox"] != rects[0]["bbox"]:
        return None
    if len({hole["base_color"] for hole in details["holes"]}) < 2:
        return None
    if sum(hole["ignored_count"] for hole in details["holes"]) == 0:
        return None
    if sum(int(hole["applied_marker"].sum()) for hole in details["holes"]) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
