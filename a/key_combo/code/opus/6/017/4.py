import random
from collections import Counter, deque

import numpy as np


BG = 0
CONTROL = 8
RED = 2
PALETTE = [1, 3, 4, 6, 7]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_TO_STEP = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
CORNER_TO_INDEX = {"tl": 0, "tr": 1, "bl": 2, "br": 3}

# Canonical full 3x3 object masks. Value 1 = body pixel, 2 = control pixel.
# Each template is defined with its control in the top-left corner.
TEMPLATES = [
    np.array([[2, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[2, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[2, 1, 0], [0, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[2, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _tight_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((rot.shape, tuple(rot.flatten().astype(int))))
        flip = np.fliplr(rot)
        variants.append((flip.shape, tuple(flip.flatten().astype(int))))
    return min(variants)


def _touch_or_overlap(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in bset:
                    return True
    return False


def _transform_template(template: np.ndarray, variant_id: int) -> np.ndarray:
    arr = template.copy()
    if variant_id >= 4:
        arr = np.fliplr(arr)
    return np.rot90(arr, variant_id % 4)


def _all_variants_by_corner() -> dict[str, list[dict]]:
    variants = {"tl": [], "tr": [], "bl": [], "br": []}
    seen = set()
    for tid, template in enumerate(TEMPLATES):
        for variant_id in range(8):
            arr = _transform_template(template, variant_id)
            ctrl = tuple(np.argwhere(arr == 2)[0])
            if ctrl == (0, 0):
                corner = "tl"
            elif ctrl == (0, 2):
                corner = "tr"
            elif ctrl == (2, 0):
                corner = "bl"
            elif ctrl == (2, 2):
                corner = "br"
            else:
                continue
            key = (tid, corner, tuple(arr.flatten()))
            if key in seen:
                continue
            seen.add(key)
            body = np.argwhere(arr == 1)
            signature = _shape_signature(arr == 1)
            variants[corner].append(
                {
                    "template_id": tid,
                    "corner": corner,
                    "array": arr,
                    "body_coords": [(int(r), int(c)) for r, c in body],
                    "signature": signature,
                }
            )
    return variants


VARIANTS_BY_CORNER = _all_variants_by_corner()
SIGNATURES = sorted({v["signature"] for items in VARIANTS_BY_CORNER.values() for v in items})


def _object_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = {int(grid[r, c]) for r, c in coords}
    assert CONTROL in colors, "each object must contain a control pixel"
    body_colors = sorted(color for color in colors if color != CONTROL)
    assert len(body_colors) == 1, "each object must otherwise be monochrome"
    base_color = body_colors[0]
    control = next((r, c) for r, c in coords if int(grid[r, c]) == CONTROL)
    local_control = (control[0] - r0, control[1] - c0)
    corner_by_pos = {(0, 0): "tl", (0, patch.shape[1] - 1): "tr", (patch.shape[0] - 1, 0): "bl", (patch.shape[0] - 1, patch.shape[1] - 1): "br"}
    assert local_control in corner_by_pos, "control pixel must sit at a bbox corner"
    body_coords = [(r - r0, c - c0) for r, c in coords if (r, c) != control]
    return {
        "base_color": base_color,
        "bbox": (r0, c0, r1, c1),
        "corner": corner_by_pos[local_control],
        "rel_coords": [(r - r0, c - c0) for r, c in coords],
        "body_signature": _shape_signature(_tight_mask(body_coords)),
    }


def _max_steps(shape: tuple[int, int], bbox: tuple[int, int, int, int], corner: str) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if corner == "tl":
        return min(r0, c0)
    if corner == "tr":
        return min(r0, w - 1 - c1)
    if corner == "bl":
        return min(h - 1 - r1, c0)
    return min(h - 1 - r1, w - 1 - c1)


def _translated_coords(obj: dict, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in obj["rel_coords"]]


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _extract_components(g, bg)
    objects = [_object_from_component(g, comp) for comp in components]
    order = []
    for obj in objects:
        steps = _max_steps(g.shape, obj["bbox"], obj["corner"])
        r0, c0, _, _ = obj["bbox"]
        order.append((steps, r0, c0, obj))
    order.sort(key=lambda item: (item[0], item[1], item[2]))

    occupied = np.zeros(g.shape, dtype=bool)
    placed = []
    for max_step, _r0, _c0, obj in order:
        sr0, sc0, _, _ = obj["bbox"]
        dr, dc = CORNER_TO_STEP[obj["corner"]]
        last_ok = 0
        for step in range(max_step + 1):
            tr0 = sr0 + dr * step
            tc0 = sc0 + dc * step
            coords = _translated_coords(obj, tr0, tc0)
            if any(occupied[r, c] for r, c in coords):
                break
            last_ok = step
        fr0 = sr0 + dr * last_ok
        fc0 = sc0 + dc * last_ok
        final_coords = _translated_coords(obj, fr0, fc0)
        for r, c in final_coords:
            occupied[r, c] = True
        placed.append(
            {
                **obj,
                "final_top_left": (fr0, fc0),
                "final_coords": final_coords,
            }
        )

    red_indices = set()
    adjacent_pairs = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if not _touch_or_overlap(placed[i]["final_coords"], placed[j]["final_coords"]):
                continue
            same_shape = placed[i]["body_signature"] == placed[j]["body_signature"]
            adjacent_pairs.append((i, j, same_shape))
            if same_shape:
                red_indices.add(i)
                red_indices.add(j)

    out = np.full_like(g, bg)
    for idx, obj in enumerate(placed):
        color = RED if idx in red_indices else obj["base_color"]
        for r, c in obj["final_coords"]:
            out[r, c] = color

    return {"output": out, "placed": placed, "adjacent_pairs": adjacent_pairs, "red_indices": red_indices}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each teal-cornered sprite diagonally to the boundary, then recolor shape-equivalent adjacent landings red.

    Input structure:
    - The background is black.
    - The grid contains 4-connected single-color sprites, each confined to a 3x3 bbox and carrying exactly one teal control pixel.
    - The teal control pixel replaces one sprite pixel at a bbox corner and indicates one of the four diagonal slide directions.

    Rule:
    - Treat each teal pixel as part of its sprite, erase all original sprite locations, and slide every sprite along the diagonal named by its control corner.
    - Order placements by boundary distance from near to far, breaking ties by bbox top-left position; later sprites stop one step before overlapping an earlier landing.
    - After all sprites land, compare every 8-neighboring pair after removing the control pixel from each normalized shape.
    - Any sprite that has at least one adjacent partner equivalent under rotation or reflection becomes red; all other sprites keep their original color, and control pixels are absorbed into the final sprite color.

    Key insight:
    The control corner decides only the slide direction and landing order, while the recoloring decision happens afterward from landed adjacency plus dihedral shape equivalence with the control pixel ignored.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _paint_object(grid: np.ndarray, top: int, left: int, variant: dict, color: int) -> None:
    for r in range(3):
        for c in range(3):
            val = int(variant["array"][r, c])
            if val == 0:
                continue
            grid[top + r, left + c] = CONTROL if val == 2 else color


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    row_gap = max(0, max(a[0] - b[2] - 1, b[0] - a[2] - 1))
    col_gap = max(0, max(a[1] - b[3] - 1, b[1] - a[3] - 1))
    return row_gap, col_gap


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], min_cheb_gap: int) -> bool:
    row_gap, col_gap = _bbox_gap(a, b)
    return max(row_gap, col_gap) >= min_cheb_gap


def _candidate_final_positions(h: int, w: int, corner: str) -> list[tuple[int, int]]:
    positions = []
    if corner in ("tl", "tr"):
        row = 0
        for col in range(0, w - 2):
            positions.append((row, col))
    if corner in ("bl", "br"):
        row = h - 3
        for col in range(0, w - 2):
            positions.append((row, col))
    if corner in ("tl", "bl"):
        col = 0
        for row in range(0, h - 2):
            positions.append((row, col))
    if corner in ("tr", "br"):
        col = w - 3
        for row in range(0, h - 2):
            positions.append((row, col))
    random.shuffle(positions)
    return positions


def _final_to_start(fr0: int, fc0: int, corner: str, steps: int) -> tuple[int, int]:
    dr, dc = CORNER_TO_STEP[corner]
    return fr0 - dr * steps, fc0 - dc * steps


def _build_attempt() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    n_objects = random.choice([4, 5])
    colors = random.sample(PALETTE, n_objects)
    top_corner = random.choice(["tl", "tr"])
    bottom_corner = random.choice(["bl", "br"])
    eq_on_top = random.choice([True, False])
    eq_corner = top_corner if eq_on_top else bottom_corner
    neq_corner = bottom_corner if eq_on_top else top_corner

    def make_pair(corner: str, color_a: int, color_b: int, same_shape: bool) -> list[dict] | None:
        pool = VARIANTS_BY_CORNER[corner]
        row = 0 if corner in ("tl", "tr") else h - 3
        if corner in ("tl", "bl"):
            base_min = 0
            base_max = w - 12
            steps = [3, 6]
        else:
            base_min = 6
            base_max = w - 6
            steps = [6, 3]
        if base_min > base_max:
            return None
        base_col = random.randint(base_min, base_max)
        finals = [(row, base_col), (row, base_col + 3)]

        if same_shape:
            sig = random.choice(SIGNATURES)
            variants = [v for v in pool if v["signature"] == sig]
            if not variants:
                return None
            chosen = [random.choice(variants), random.choice(variants)]
        else:
            sigs = random.sample(SIGNATURES, 2)
            variants_a = [v for v in pool if v["signature"] == sigs[0]]
            variants_b = [v for v in pool if v["signature"] == sigs[1]]
            if not variants_a or not variants_b:
                return None
            chosen = [random.choice(variants_a), random.choice(variants_b)]

        pair_specs = []
        for idx in range(2):
            fr0, fc0 = finals[idx]
            sr0, sc0 = _final_to_start(fr0, fc0, corner, steps[idx])
            if not (0 <= sr0 <= h - 3 and 0 <= sc0 <= w - 3):
                return None
            pair_specs.append(
                {
                    "variant": chosen[idx],
                    "color": color_a if idx == 0 else color_b,
                    "corner": corner,
                    "final_bbox": (fr0, fc0, fr0 + 2, fc0 + 2),
                    "start_bbox": (sr0, sc0, sr0 + 2, sc0 + 2),
                }
            )
        if not _boxes_far_enough(pair_specs[0]["start_bbox"], pair_specs[1]["start_bbox"], 2):
            return None
        return pair_specs

    eq_pair = make_pair(eq_corner, colors[0], colors[1], True)
    neq_pair = make_pair(neq_corner, colors[2], colors[3], False)
    if eq_pair is None or neq_pair is None:
        return None

    specs = [*eq_pair, *neq_pair]
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if not _boxes_far_enough(specs[i]["start_bbox"], specs[j]["start_bbox"], 2):
                return None

    remaining_colors = colors[4:]
    used_finals = [spec["final_bbox"] for spec in specs]
    used_starts = [spec["start_bbox"] for spec in specs]
    extra_corners = ["tl", "tr", "bl", "br"]

    for color in remaining_colors:
        placed = False
        random.shuffle(extra_corners)
        for corner in extra_corners:
            pool = VARIANTS_BY_CORNER[corner]
            candidates = _candidate_final_positions(h, w, corner)
            random.shuffle(pool)
            for variant in pool:
                for fr0, fc0 in candidates[:20]:
                    max_steps = 0
                    if corner == "tl":
                        max_steps = min(h - 3 - fr0, w - 3 - fc0)
                    elif corner == "tr":
                        max_steps = min(h - 3 - fr0, fc0)
                    elif corner == "bl":
                        max_steps = min(fr0, w - 3 - fc0)
                    else:
                        max_steps = min(fr0, fc0)
                    if max_steps < 3:
                        continue
                    steps = random.randint(3, min(7, max_steps))
                    sr0, sc0 = _final_to_start(fr0, fc0, corner, steps)
                    final_bbox = (fr0, fc0, fr0 + 2, fc0 + 2)
                    start_bbox = (sr0, sc0, sr0 + 2, sc0 + 2)
                    if any(not _boxes_far_enough(final_bbox, box, 2) for box in used_finals):
                        continue
                    if any(not _boxes_far_enough(start_bbox, box, 2) for box in used_starts):
                        continue
                    specs.append(
                        {
                            "variant": variant,
                            "color": color,
                            "corner": corner,
                            "final_bbox": final_bbox,
                            "start_bbox": start_bbox,
                        }
                    )
                    used_finals.append(final_bbox)
                    used_starts.append(start_bbox)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        sr0, sc0, _, _ = spec["start_bbox"]
        _paint_object(grid, sr0, sc0, spec["variant"], spec["color"])

    details = _solve_details(grid)
    pair_flags = [same for _, _, same in details["adjacent_pairs"]]
    if not pair_flags or not any(pair_flags) or not any(not same for same in pair_flags):
        return None

    components = _extract_components(grid, BG)
    if len(components) != n_objects:
        return None
    start_boxes = [_bbox_from_coords(comp) for comp in components]
    for i in range(len(start_boxes)):
        for j in range(i + 1, len(start_boxes)):
            if not _boxes_far_enough(start_boxes[i], start_boxes[j], 2):
                return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
