from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
LEFT_WALL = 1
RIGHT_WALL = 2
GREY = 5
PAIR_COLORS = (3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (14, 19)
GRID_WIDTH_RANGE = (14, 19)
PAIR_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 80
MAX_OUTPUT_OVERLAP = 2

BASE_REFERENCE_SHAPES = {
    "square": np.array([[1, 1], [1, 1]], dtype=bool),
    "ell": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-color sprite pair toward the wall whose grey reference shape matches their emergent union.

    Input structure:
    - The grid has black background, a full blue wall in the leftmost column, and a full red wall in the rightmost column.
    - One grey 4-connected reference shape sits next to the left wall and one non-equivalent grey reference shape sits next to the right wall.
    - The interior contains several non-grey colors; each such color appears as exactly two separate 4-connected sprites.

    Rule:
    - Group the interior sprites by color to recover the pairs.
    - For each pair, build a union shape and compare it with the left and right grey reference shapes up to rotation and reflection.
    - If the pair matches the left reference, slide both sprites left; if it matches the right reference, slide both sprites right.
    - Within one pair, process the sprite nearer to its destination wall first; a sprite stops when another settled sprite from the same pair or a fixed wall/reference cell blocks the next horizontal step.

    Key insight:
    - A single sprite is insufficient: only the combined pair shape reveals which wall the pair should slide toward.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    ref_info = _extract_references(g)
    left_sig = _shape_signature(ref_info["left"]["mask"])
    right_sig = _shape_signature(ref_info["right"]["mask"])
    static_mask = _static_mask(g)

    color_to_components = _extract_pair_components(g)
    ordered_colors = sorted(
        color_to_components,
        key=lambda color: min(_component_anchor(comp) for comp in color_to_components[color]),
    )

    out = np.full_like(g, BACKGROUND)
    out[:, 0] = LEFT_WALL
    out[:, -1] = RIGHT_WALL
    for side in ("left", "right"):
        ref = ref_info[side]
        out[ref["coords"][:, 0], ref["coords"][:, 1]] = GREY

    for color in ordered_colors:
        comps = color_to_components[color]
        direction = _pair_direction(comps, left_sig, right_sig)
        placed = _settle_pair(comps, direction, static_mask, g.shape)
        for coords in placed:
            out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 2 input")


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((H, W), BACKGROUND, dtype=int)
    grid[:, 0] = LEFT_WALL
    grid[:, -1] = RIGHT_WALL

    left_kind = random.choice(tuple(BASE_REFERENCE_SHAPES))
    right_kind = "ell" if left_kind == "square" else "square"

    left_ref = _random_dihedral(BASE_REFERENCE_SHAPES[left_kind])
    right_ref = _random_dihedral(BASE_REFERENCE_SHAPES[right_kind])

    left_pos = _choose_reference_position(left_ref, H, left=True)
    right_pos = _choose_reference_position(right_ref, H, left=False)
    _paint_mask(grid, left_ref, left_pos[0], left_pos[1], GREY)
    _paint_mask(grid, right_ref, right_pos[0], right_pos[1], GREY)

    ref_signatures = {
        "left": _shape_signature(left_ref),
        "right": _shape_signature(right_ref),
    }
    forbidden_component_sigs = set(ref_signatures.values())

    pair_count = random.randint(*PAIR_COUNT_RANGE)
    side_labels = ["left"] * random.randint(1, pair_count - 1)
    side_labels += ["right"] * (pair_count - len(side_labels))
    random.shuffle(side_labels)
    colors = random.sample(PAIR_COLORS, pair_count)

    occupied = grid != BACKGROUND
    halo = _expand_mask(occupied, H, W)
    pair_records: list[dict] = []

    for color, side in zip(colors, side_labels):
        ref_mask = left_ref if side == "left" else right_ref
        component_pair = _choose_component_pair(ref_mask, forbidden_component_sigs)
        if component_pair is None:
            return None

        comp_a, comp_b = component_pair
        blocking = random.random() < 0.5
        placements = _place_pair_components(
            H,
            W,
            side,
            comp_a,
            comp_b,
            occupied,
            halo,
            blocking=blocking,
        )
        if placements is None:
            return None

        pair_comps = []
        for mask, (r0, c0) in zip((comp_a, comp_b), placements):
            _paint_mask(grid, mask, r0, c0, color)
            comp = _component_from_mask(mask, r0, c0, color)
            pair_comps.append(comp)
            occupied[comp["coords"][:, 0], comp["coords"][:, 1]] = True
            halo = _expand_mask(occupied, H, W)
        pair_records.append({"color": color, "components": pair_comps, "side": side})

    solver_dirs = {}
    left_sig = ref_signatures["left"]
    right_sig = ref_signatures["right"]
    for record in pair_records:
        solver_side = _pair_direction(record["components"], left_sig, right_sig)
        solver_dirs[record["color"]] = solver_side
        if solver_side != record["side"]:
            return None

    if not _output_is_clear(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    return grid


def _output_is_clear(grid: np.ndarray) -> bool:
    ref_info = _extract_references(grid)
    left_sig = _shape_signature(ref_info["left"]["mask"])
    right_sig = _shape_signature(ref_info["right"]["mask"])
    static_mask = _static_mask(grid)
    pair_map = _extract_pair_components(grid)

    layers: list[np.ndarray] = []
    for comps in pair_map.values():
        direction = _pair_direction(comps, left_sig, right_sig)
        settled = _settle_pair(comps, direction, static_mask, grid.shape)
        layer = np.zeros(grid.shape, dtype=bool)
        for coords in settled:
            layer[coords[:, 0], coords[:, 1]] = True
        if np.any(layer & static_mask):
            return False
        layers.append(layer)

    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            if int(np.sum(layers[i] & layers[j])) > MAX_OUTPUT_OVERLAP:
                return False
    return True


def _extract_references(grid: np.ndarray) -> dict[str, dict]:
    grey_components = _find_components_of_color(grid, GREY)
    if len(grey_components) != 2:
        raise AssertionError("expected exactly two grey reference shapes")

    left = min(grey_components, key=lambda comp: comp["bbox"][1])
    right = max(grey_components, key=lambda comp: comp["bbox"][3])
    if left["bbox"][1] != 1:
        raise AssertionError("left reference must touch column 1")
    if right["bbox"][3] != grid.shape[1] - 2:
        raise AssertionError("right reference must touch column W-2")
    return {"left": left, "right": right}


def _extract_pair_components(grid: np.ndarray) -> dict[int, list[dict]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in {BACKGROUND, LEFT_WALL, RIGHT_WALL, GREY})
    color_to_components: dict[int, list[dict]] = {}
    for color in colors:
        comps = _find_components_of_color(grid, color)
        if len(comps) != 2:
            raise AssertionError(f"expected exactly two components for color {color}")
        color_to_components[color] = comps
    if not color_to_components:
        raise AssertionError("expected at least one sprite pair")
    return color_to_components


def _pair_direction(comps: list[dict], left_sig, right_sig) -> str:
    joint_union = _pair_joint_bbox_union(comps[0], comps[1])
    overlay_union = _pair_overlay_union(comps[0], comps[1])

    joint_side = _match_signature(_shape_signature(joint_union), left_sig, right_sig)
    overlay_side = _match_signature(_shape_signature(overlay_union), left_sig, right_sig)

    if joint_side is not None and overlay_side is not None and joint_side == overlay_side:
        return joint_side
    if joint_side is not None and overlay_side is None:
        return joint_side
    if overlay_side is not None:
        return overlay_side
    raise AssertionError("pair union does not uniquely match either reference")


def _match_signature(sig, left_sig, right_sig) -> str | None:
    left = sig == left_sig
    right = sig == right_sig
    if left == right:
        return None
    return "left" if left else "right"


def _settle_pair(comps: list[dict], direction: str, static_mask: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    if direction not in {"left", "right"}:
        raise AssertionError("invalid direction")

    if direction == "left":
        ordered = sorted(comps, key=lambda comp: (comp["bbox"][1], comp["bbox"][0], comp["bbox"][3]))
        step = -1
    else:
        ordered = sorted(comps, key=lambda comp: (-comp["bbox"][3], comp["bbox"][0], comp["bbox"][1]))
        step = 1

    occupied = static_mask.copy()
    placed_coords: list[np.ndarray] = []
    for comp in ordered:
        r0, c0 = comp["bbox"][0], comp["bbox"][1]
        while _can_place_rel(comp["rel_coords"], r0, c0 + step, occupied, shape):
            c0 += step
        coords = comp["rel_coords"] + np.array([r0, c0], dtype=int)
        occupied[coords[:, 0], coords[:, 1]] = True
        placed_coords.append(coords)
    return placed_coords


def _static_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == LEFT_WALL) | (grid == RIGHT_WALL) | (grid == GREY)


def _component_anchor(comp: dict) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in comp["coords"])


def _pair_joint_bbox_union(comp_a: dict, comp_b: dict) -> np.ndarray:
    r0 = min(comp_a["bbox"][0], comp_b["bbox"][0])
    c0 = min(comp_a["bbox"][1], comp_b["bbox"][1])
    r1 = max(comp_a["bbox"][2], comp_b["bbox"][2])
    c1 = max(comp_a["bbox"][3], comp_b["bbox"][3])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for comp in (comp_a, comp_b):
        rel = comp["coords"] - np.array([r0, c0], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
    return _crop_mask(mask)


def _pair_overlay_union(comp_a: dict, comp_b: dict) -> np.ndarray:
    return _overlay_union_masks(comp_a["mask"], comp_b["mask"])


def _overlay_union_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = max(mask_a.shape[1], mask_b.shape[1])
    union = np.zeros((h, w), dtype=bool)
    union[: mask_a.shape[0], : mask_a.shape[1]] |= mask_a
    union[: mask_b.shape[0], : mask_b.shape[1]] |= mask_b
    return _crop_mask(union)


def _choose_reference_position(mask: np.ndarray, H: int, *, left: bool) -> tuple[int, int]:
    max_r = H - mask.shape[0]
    r0 = random.randint(1, max(1, max_r - 1))
    c0 = 1 if left else 0
    if left:
        c0 = 1
    else:
        c0 = -mask.shape[1] - 1
    return r0, c0


def _choose_component_pair(ref_mask: np.ndarray, forbidden_sigs: set) -> tuple[np.ndarray, np.ndarray] | None:
    ref_mask = _crop_mask(ref_mask)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(ref_mask)]
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    subset_masks = []
    for bits in range(1, 1 << len(coords)):
        area = bits.bit_count()
        if area < 2 or area > 4:
            continue
        canvas = np.zeros_like(ref_mask, dtype=bool)
        for idx, (r, c) in enumerate(coords):
            if bits & (1 << idx):
                canvas[r, c] = True
        if not _is_connected(canvas):
            continue
        tight = _crop_mask(canvas)
        sig = _shape_signature(tight)
        if sig in forbidden_sigs:
            continue
        subset_masks.append(tight)

    for mask_a in subset_masks:
        for mask_b in subset_masks:
            union = _overlay_union_masks(mask_a, mask_b)
            if not np.array_equal(union, ref_mask):
                continue
            if _shape_signature(mask_a) in forbidden_sigs or _shape_signature(mask_b) in forbidden_sigs:
                continue
            if not _both_contribute(mask_a, mask_b, ref_mask):
                continue
            key = tuple(sorted((_mask_key(mask_a), _mask_key(mask_b))))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((mask_a.copy(), mask_b.copy()))

    if not candidates:
        return None
    return random.choice(candidates)


def _both_contribute(mask_a: np.ndarray, mask_b: np.ndarray, ref_mask: np.ndarray) -> bool:
    h = max(mask_a.shape[0], mask_b.shape[0], ref_mask.shape[0])
    w = max(mask_a.shape[1], mask_b.shape[1], ref_mask.shape[1])
    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    a[: mask_a.shape[0], : mask_a.shape[1]] = mask_a
    b[: mask_b.shape[0], : mask_b.shape[1]] = mask_b
    return bool(np.any(a & ~b)) and bool(np.any(b & ~a))


def _place_pair_components(
    H: int,
    W: int,
    side: str,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    occupied: np.ndarray,
    halo: np.ndarray,
    *,
    blocking: bool,
) -> list[tuple[int, int]] | None:
    dims = [(mask_a.shape[0], mask_a.shape[1]), (mask_b.shape[0], mask_b.shape[1])]

    for _ in range(PLACEMENT_TRIES):
        rows = _sample_pair_rows(H, dims[0][0], dims[1][0], blocking)
        if rows is None:
            continue
        r_a, r_b = rows

        cols = _sample_pair_cols(W, dims[0][1], dims[1][1], side)
        if cols is None:
            continue
        c_a, c_b = cols

        if not _placement_ok(mask_a, r_a, c_a, occupied, halo):
            continue

        occ2 = occupied.copy()
        halo2 = halo.copy()
        coords_a = _absolute_coords(mask_a, r_a, c_a)
        occ2[coords_a[:, 0], coords_a[:, 1]] = True
        halo2 = _expand_mask(occ2, H, W)

        if not _placement_ok(mask_b, r_b, c_b, occ2, halo2):
            continue

        bbox_gap = _bbox_gap(_bbox_from_mask(mask_a, r_a, c_a), _bbox_from_mask(mask_b, r_b, c_b))
        if bbox_gap < 2:
            continue

        return [(r_a, c_a), (r_b, c_b)]

    return None


def _sample_pair_rows(H: int, h_a: int, h_b: int, blocking: bool) -> tuple[int, int] | None:
    max_a = H - h_a - 1
    max_b = H - h_b - 1
    if max_a < 1 or max_b < 1:
        return None

    for _ in range(40):
        r_a = random.randint(1, max_a)
        if blocking:
            lo = max(1, r_a - h_b + 1)
            hi = min(max_b, r_a + h_a - 1)
            if lo > hi:
                continue
            r_b = random.randint(lo, hi)
        else:
            choices = []
            if r_a - h_b - 1 >= 1:
                choices.extend(range(1, r_a - h_b))
            if r_a + h_a + 1 <= max_b:
                choices.extend(range(r_a + h_a + 1, max_b + 1))
            if not choices:
                continue
            r_b = random.choice(choices)
        return r_a, r_b
    return None


def _sample_pair_cols(W: int, w_a: int, w_b: int, side: str) -> tuple[int, int] | None:
    inner_min = 3
    inner_max = W - 4
    if inner_max <= inner_min:
        return None

    for _ in range(40):
        if side == "left":
            near_max = inner_max - w_a + 1
            far_max = inner_max - w_b + 1
            if near_max < inner_min or far_max < inner_min + 3:
                return None
            c_a = random.randint(inner_min, max(inner_min, near_max - 2))
            far_lo = c_a + w_a + 2
            if far_lo > far_max:
                continue
            c_b = random.randint(far_lo, far_max)
        else:
            near_min = inner_min
            far_min = inner_min
            near_lo = max(inner_min, W - 6 - w_a)
            near_hi = inner_max - w_a + 1
            if near_lo > near_hi:
                continue
            c_a = random.randint(near_lo, near_hi)
            far_hi = c_a - w_b - 2
            if far_hi < far_min:
                continue
            c_b = random.randint(far_min, far_hi)
        return c_a, c_b
    return None


def _placement_ok(mask: np.ndarray, r0: int, c0: int, occupied: np.ndarray, halo: np.ndarray) -> bool:
    H, W = occupied.shape
    coords = _absolute_coords(mask, r0, c0)
    if coords.size == 0:
        return False
    if coords[:, 0].min() < 1 or coords[:, 0].max() >= H - 1:
        return False
    if coords[:, 1].min() < 2 or coords[:, 1].max() >= W - 2:
        return False
    return not bool(np.any(halo[coords[:, 0], coords[:, 1]]))


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            r0 = int(coords_arr[:, 0].min())
            c0 = int(coords_arr[:, 1].min())
            r1 = int(coords_arr[:, 0].max())
            c1 = int(coords_arr[:, 1].max())
            rel = coords_arr - np.array([r0, c0], dtype=int)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            components.append(
                {
                    "color": int(color),
                    "coords": coords_arr,
                    "bbox": (r0, c0, r1, c1),
                    "rel_coords": rel,
                    "mask": mask,
                }
            )
    return components


def _component_from_mask(mask: np.ndarray, r0: int, c0: int, color: int) -> dict:
    rel = np.argwhere(mask).astype(int)
    coords = rel + np.array([r0, c0], dtype=int)
    return {
        "color": int(color),
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r0 + mask.shape[0] - 1), int(c0 + mask.shape[1] - 1)),
        "rel_coords": rel,
        "mask": mask.copy(),
    }


def _can_place_rel(rel_coords: np.ndarray, r0: int, c0: int, occupied: np.ndarray, shape: tuple[int, int]) -> bool:
    H, W = shape
    coords = rel_coords + np.array([r0, c0], dtype=int)
    if coords[:, 0].min() < 0 or coords[:, 0].max() >= H:
        return False
    if coords[:, 1].min() < 0 or coords[:, 1].max() >= W:
        return False
    return not bool(np.any(occupied[coords[:, 0], coords[:, 1]]))


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    tight = _crop_mask(mask)
    variants = []
    for k in range(4):
        rot = np.rot90(tight, k)
        variants.append(_mask_key(_crop_mask(rot)))
        variants.append(_mask_key(_crop_mask(np.fliplr(rot))))
    return min(variants)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    if c0 < 0:
        c0 = grid.shape[1] + c0
    coords = np.argwhere(mask)
    for dr, dc in coords:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _absolute_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask).astype(int) + np.array([r0, c0], dtype=int)


def _bbox_from_mask(mask: np.ndarray, r0: int, c0: int) -> tuple[int, int, int, int]:
    return (int(r0), int(c0), int(r0 + mask.shape[0] - 1), int(c0 + mask.shape[1] - 1))


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    row_gap = max(b2[0] - b1[2] - 1, b1[0] - b2[2] - 1, 0)
    col_gap = max(b2[1] - b1[3] - 1, b1[1] - b2[3] - 1, 0)
    if row_gap == 0:
        return int(col_gap)
    if col_gap == 0:
        return int(row_gap)
    return int(min(row_gap, col_gap))


def _expand_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    expanded = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(H, rr + 2)):
            for nc in range(max(0, cc - 1), min(W, cc + 2)):
                expanded[nr, nc] = True
    return expanded


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _crop_mask(out)


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
