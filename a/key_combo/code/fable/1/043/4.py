from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, paste, random_sprite


BG = int(Color.BLACK)
COLOR_POOL = [int(c) for c in Color.NOT_BLACK]
GEN_RETRIES = 500
SYM_TYPES_CYCLE = ("horizontal", "vertical", "diagonal", "rot180", "radial")
SYM_COUNT_CHOICES = (2, 2, 3)
DISTRACTOR_COUNT_CHOICES = (1, 2, 2, 3)
DISTANCE_CHOICES = (2, 3, 4)
RECT_SIZE_CHOICES = (3, 4, 5)
SQUARE_SIZE_CHOICES = (3, 4, 5)
_GEN_COUNTER = 0


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask.astype(int), b).astype(bool)


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_rectangular_frame_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _is_single_component(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=True)
    return len(comps) == 1


def _symmetry_kinds(mask: np.ndarray) -> set[str]:
    kinds: set[str] = set()
    if np.array_equal(mask, np.flipud(mask)):
        kinds.add("horizontal")
    if np.array_equal(mask, np.fliplr(mask)):
        kinds.add("vertical")
    if np.array_equal(mask, np.rot90(mask, 2)):
        kinds.add("rot180")
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, np.rot90(mask, 1)):
            kinds.add("rot90")
        if np.array_equal(mask, mask.T):
            kinds.add("diag_main")
        if np.array_equal(mask, np.rot90(mask, 2).T):
            kinds.add("diag_anti")
    return kinds


def _has_any_symmetry(mask: np.ndarray) -> bool:
    return bool(_symmetry_kinds(mask))


def _valid_object_mask(mask: np.ndarray) -> bool:
    mask = _crop_mask(mask)
    h, w = mask.shape
    area = int(mask.sum())
    if area < 4 or h < 2 or w < 2:
        return False
    if area == h * w:
        return False
    if _is_rectangular_frame_mask(mask):
        return False
    return _is_single_component(mask)


def _full_mask(comp: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    coords = comp["coords"]
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _rectangular_frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        border = np.zeros(grid.shape, dtype=bool)
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        if np.array_equal(_full_mask(comp, grid.shape), border):
            candidates.append(comp)
    return candidates


def _min_chebyshev(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    deltas = np.abs(coords_a[:, None, :] - coords_b[None, :, :])
    return int(np.min(np.max(deltas, axis=2)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather only the symmetric sprites into the hollow rectangle, encoding each selected area as a row-major color run.

    Input structure:
    - The grid has one dominant background color, one one-pixel-thick monochrome rectangular frame, and several separated monochrome sprites.
    - Some sprites are symmetric under a horizontal, vertical, diagonal, or rotational symmetry; the others are distractors.
    - The frame interior area equals the total pixel count of the symmetric sprites, and those selected sprites have distinct frame distances.

    Rule:
    - Detect the unique rectangular frame and inspect every other monochrome 8-connected component through its tight sprite mask.
    - Keep only the sprites whose shape has at least one horizontal, vertical, diagonal, or rotational symmetry, then sort them by minimum Chebyshev distance to the frame border from nearest to farthest.
    - Erase those selected sprites, and fill the frame interior in row-major order with one solid color run per selected sprite whose length equals that sprite's area; leave the frame and every distractor unchanged.

    Key insight:
    - Shape symmetry decides which sprites migrate, while the frame interior stores both the selected pixel counts and their near-to-far ordering as contiguous colored bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    frames = _rectangular_frame_candidates(g, bg)
    assert len(frames) == 1, "expected exactly one hollow rectangular frame"

    frame = frames[0]
    frame_mask = _full_mask(frame, g.shape)
    frame_coords = frame["coords"]
    r0, c0, r1, c1 = frame["bbox"]
    interior_coords = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]

    selected: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=True):
        if np.array_equal(_full_mask(comp, g.shape), frame_mask):
            continue
        sprite_mask = _component_mask(comp)
        if not _has_any_symmetry(sprite_mask):
            continue
        selected.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "area": int(comp["area"]),
                "distance": _min_chebyshev(comp["coords"], frame_coords),
                "bbox": comp["bbox"],
            }
        )

    selected.sort(key=lambda item: (item["distance"], item["bbox"][0], item["bbox"][1]))
    assert sum(item["area"] for item in selected) == len(interior_coords), "frame interior must match selected area"

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = bg
    for item in selected:
        coords = item["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    cursor = 0
    for item in selected:
        for rr, cc in interior_coords[cursor : cursor + item["area"]]:
            out[rr, cc] = int(item["color"])
        cursor += item["area"]
    return out


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _random_axis_or_diagonal_mask(symmetry: str) -> np.ndarray:
    for _ in range(200):
        density = random.uniform(0.4, 0.72)
        if symmetry in {"diagonal", "radial"}:
            side = random.choice(SQUARE_SIZE_CHOICES)
            sprite = random_sprite(
                side,
                side,
                density=density,
                symmetry=symmetry,
                color_palette=[1],
                connectivity=8,
                background=BG,
            )
        else:
            h = random.choice(RECT_SIZE_CHOICES)
            w = random.choice(RECT_SIZE_CHOICES)
            sprite = random_sprite(
                h,
                w,
                density=density,
                symmetry=symmetry,
                color_palette=[1],
                connectivity=8,
                background=BG,
            )
        mask = _crop_mask(sprite != BG)
        kinds = _symmetry_kinds(mask)
        if not _valid_object_mask(mask):
            continue
        if symmetry == "horizontal" and "horizontal" not in kinds:
            continue
        if symmetry == "vertical" and "vertical" not in kinds:
            continue
        if symmetry == "diagonal" and not ({"diag_main", "diag_anti"} & kinds):
            continue
        if symmetry == "radial" and "rot90" not in kinds:
            continue
        return mask
    raise RuntimeError(f"failed to generate a valid {symmetry} sprite")


def _random_rot180_mask() -> np.ndarray:
    for _ in range(400):
        h = random.choice(RECT_SIZE_CHOICES)
        w = random.choice(RECT_SIZE_CHOICES)
        density = random.uniform(0.35, 0.58)
        base = random_sprite(
            h,
            w,
            density=density,
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=8,
            background=BG,
        )
        mask = _crop_mask((base != BG) | np.rot90(base != BG, 2))
        kinds = _symmetry_kinds(mask)
        if not _valid_object_mask(mask):
            continue
        if "rot180" not in kinds:
            continue
        if kinds & {"horizontal", "vertical", "diag_main", "diag_anti", "rot90"}:
            continue
        return mask
    raise RuntimeError("failed to generate a rotation-only sprite")


def _random_symmetric_mask(symmetry: str) -> np.ndarray:
    if symmetry == "rot180":
        return _random_rot180_mask()
    return _random_axis_or_diagonal_mask(symmetry)


def _random_asymmetric_mask() -> np.ndarray:
    for _ in range(300):
        h = random.choice(RECT_SIZE_CHOICES)
        w = random.choice(RECT_SIZE_CHOICES)
        density = random.uniform(0.36, 0.68)
        sprite = random_sprite(
            h,
            w,
            density=density,
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=8,
            background=BG,
        )
        mask = _crop_mask(sprite != BG)
        if not _valid_object_mask(mask):
            continue
        if _has_any_symmetry(mask):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric distractor")


def _inner_size_choices(total_area: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for h in range(2, min(8, total_area) + 1):
        if total_area % h != 0:
            continue
        w = total_area // h
        if 2 <= w <= 10:
            choices.append((h, w))
    random.shuffle(choices)
    return choices


def _order_crosses_rows(areas: list[int], width: int, order: tuple[int, ...]) -> bool:
    cursor = 0
    crosses = False
    for idx in order:
        area = areas[idx]
        if cursor % width + area > width:
            crosses = True
        cursor += area
    return crosses


def _choose_inner_shape_and_order(sym_items: list[dict]) -> tuple[tuple[int, int], tuple[int, ...]] | None:
    areas = [int(item["area"]) for item in sym_items]
    all_orders = list(itertools.permutations(range(len(sym_items))))
    random.shuffle(all_orders)

    for inner_shape in _inner_size_choices(sum(areas)):
        width = inner_shape[1]
        crossing_orders = [order for order in all_orders if _order_crosses_rows(areas, width, order)]
        if crossing_orders:
            return inner_shape, random.choice(crossing_orders)

    choices = _inner_size_choices(sum(areas))
    if not choices:
        return None
    return random.choice(choices), tuple(range(len(sym_items)))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _coords_outside_bbox(coords: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    inside_r = (coords[:, 0] >= r0) & (coords[:, 0] <= r1)
    inside_c = (coords[:, 1] >= c0) & (coords[:, 1] <= c1)
    return not np.any(inside_r & inside_c)


def _candidate_positions(
    canvas: np.ndarray,
    mask: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    frame_coords: np.ndarray,
    *,
    exact_distance: int | None,
) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    fg_coords = np.argwhere(mask)
    blocked = _dilate8(canvas != BG)
    candidates: list[tuple[int, int, int]] = []

    for r0 in range(canvas.shape[0] - h + 1):
        for c0 in range(canvas.shape[1] - w + 1):
            if np.any(blocked[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            shifted = fg_coords + np.array([r0, c0])
            if not _coords_outside_bbox(shifted, frame_bbox):
                continue
            dist = _min_chebyshev(shifted, frame_coords)
            if exact_distance is not None and dist != exact_distance:
                continue
            candidates.append((r0, c0, dist))
    random.shuffle(candidates)
    return candidates


def _expected_output(
    inp: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    sym_records: list[dict],
) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = frame_bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    out[r0 + 1 : r1, c0 + 1 : c1] = BG

    ordered = sorted(sym_records, key=lambda item: item["distance"])
    for item in ordered:
        coords = item["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    cursor = 0
    for item in ordered:
        for rr, cc in interior[cursor : cursor + item["area"]]:
            out[rr, cc] = int(item["color"])
        cursor += item["area"]
    return out


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    sym_records: list[dict],
    distractor_records: list[dict],
) -> bool:
    if not np.array_equal(_expected_output(inp, frame_bbox, sym_records), expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    frames = _rectangular_frame_candidates(inp, BG)
    if len(frames) != 1:
        return False
    frame = frames[0]
    detected_distances: list[int] = []
    frame_mask = _full_mask(frame, inp.shape)
    for comp in find_components(inp, bg=BG, connectivity=8, monochromatic=True):
        if np.array_equal(_full_mask(comp, inp.shape), frame_mask):
            continue
        mask = _component_mask(comp)
        if _has_any_symmetry(mask):
            detected_distances.append(_min_chebyshev(comp["coords"], frame["coords"]))

    if sorted(detected_distances) != sorted(item["distance"] for item in sym_records):
        return False
    if len(set(detected_distances)) != len(sym_records):
        return False

    r0, c0, r1, c1 = frame_bbox
    inner_area = (r1 - r0 - 1) * (c1 - c0 - 1)
    if inner_area != sum(item["area"] for item in sym_records):
        return False

    width = c1 - c0 - 1
    order = tuple(np.argsort([item["distance"] for item in sym_records]).tolist())
    areas = [int(item["area"]) for item in sym_records]
    if not _order_crosses_rows(areas, width, order):
        return False

    for item in distractor_records:
        if _has_any_symmetry(item["mask"]):
            return False
    return True


def _generate_case(seed_index: int) -> tuple[np.ndarray, np.ndarray] | None:
    n_sym = random.choice(SYM_COUNT_CHOICES)
    n_distractors = random.choice(DISTRACTOR_COUNT_CHOICES)
    total_colors = 1 + n_sym + n_distractors
    colors = random.sample(COLOR_POOL, total_colors)
    frame_color = int(colors[0])

    forced_type = SYM_TYPES_CYCLE[seed_index % len(SYM_TYPES_CYCLE)]
    extra_types = [random.choice(SYM_TYPES_CYCLE) for _ in range(n_sym - 1)]
    sym_types = [forced_type, *extra_types]
    random.shuffle(sym_types)

    sym_items: list[dict] = []
    for color, symmetry in zip(colors[1 : 1 + n_sym], sym_types):
        mask = _random_symmetric_mask(symmetry)
        sym_items.append(
            {
                "mask": mask,
                "color": int(color),
                "area": int(mask.sum()),
                "symmetry": symmetry,
            }
        )

    choice = _choose_inner_shape_and_order(sym_items)
    if choice is None:
        return None
    (inner_h, inner_w), order = choice
    sampled_distances = sorted(random.sample(DISTANCE_CHOICES, n_sym))
    for rank, idx in enumerate(order):
        sym_items[idx]["distance"] = int(sampled_distances[rank])

    distractor_items: list[dict] = []
    for color in colors[1 + n_sym :]:
        mask = _random_asymmetric_mask()
        distractor_items.append({"mask": mask, "color": int(color)})

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    all_masks = [item["mask"] for item in [*sym_items, *distractor_items]]
    max_h = max(mask.shape[0] for mask in all_masks)
    max_w = max(mask.shape[1] for mask in all_masks)
    max_distance = max(int(item["distance"]) for item in sym_items)

    margin_h = max_h + max_distance + 1
    margin_w = max_w + max_distance + 1
    min_h = frame_h + 2 * margin_h
    min_w = frame_w + 2 * margin_w
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, min(30, min_h + 4))
    grid_w = random.randint(min_w, min(30, min_w + 4))
    frame_r0 = random.randint(margin_h, grid_h - margin_h - frame_h)
    frame_c0 = random.randint(margin_w, grid_w - margin_w - frame_w)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    canvas = np.full((grid_h, grid_w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    canvas[fr0 : fr1 + 1, fc0] = frame_color
    canvas[fr0 : fr1 + 1, fc1] = frame_color
    canvas[fr0, fc0 : fc1 + 1] = frame_color
    canvas[fr1, fc0 : fc1 + 1] = frame_color
    frame_coords = np.argwhere(canvas == frame_color)

    placed_sym: list[dict] = []
    for item in sorted(sym_items, key=lambda obj: (-obj["mask"].size, -obj["distance"])):
        candidates = _candidate_positions(
            canvas,
            item["mask"],
            frame_bbox,
            frame_coords,
            exact_distance=int(item["distance"]),
        )
        if not candidates:
            return None
        r0, c0, dist = candidates[0]
        patch = _mask_to_patch(item["mask"], item["color"])
        paste(canvas, patch, r0, c0, transparent=BG)
        coords = np.argwhere(item["mask"]) + np.array([r0, c0])
        placed_sym.append(
            {
                "mask": item["mask"],
                "color": int(item["color"]),
                "area": int(item["area"]),
                "distance": int(dist),
                "coords": coords,
            }
        )

    placed_distractors: list[dict] = []
    for item in sorted(distractor_items, key=lambda obj: -obj["mask"].size):
        candidates = _candidate_positions(canvas, item["mask"], frame_bbox, frame_coords, exact_distance=None)
        if not candidates:
            return None
        r0, c0, _ = candidates[0]
        patch = _mask_to_patch(item["mask"], item["color"])
        paste(canvas, patch, r0, c0, transparent=BG)
        placed_distractors.append({"mask": item["mask"], "color": int(item["color"])})

    expected = _expected_output(canvas, frame_bbox, placed_sym)
    if not _validate_case(canvas, expected, frame_bbox, placed_sym, placed_distractors):
        return None
    return canvas, expected


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GEN_RETRIES):
        case = _generate_case(_GEN_COUNTER)
        _GEN_COUNTER += 1
        if case is None:
            continue
        return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 043-4")
