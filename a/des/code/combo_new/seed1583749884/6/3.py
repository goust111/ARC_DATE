from __future__ import annotations

import random

import numpy as np


BG = 0
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": _bbox(arr),
                }
            )
    return comps


def _line_orientation(coords: list[tuple[int, int]]) -> str:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        ordered = sorted(c for _, c in coords)
        if len(ordered) > 1:
            assert all(b - a == 2 for a, b in zip(ordered, ordered[1:])), "horizontal line dots must be spaced by one gap"
        return "H"
    if len(cols) == 1:
        ordered = sorted(r for r, _ in coords)
        if len(ordered) > 1:
            assert all(b - a == 2 for a, b in zip(ordered, ordered[1:])), "vertical line dots must be spaced by one gap"
        return "V"
    raise AssertionError("line dots must be collinear")


def _touches_line(obj: dict[str, object], line_cells: set[tuple[int, int]]) -> bool:
    for r, c in obj["coords"]:
        rr = int(r)
        cc = int(c)
        for dr, dc in ORTHO_DIRS:
            if (rr + dr, cc + dc) in line_cells:
                return True
    return False


def _translation_to_midpoint(a: dict[str, object], b: dict[str, object]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    dr_num = (br0 + br1) - (ar0 + ar1)
    dc_num = (bc0 + bc1) - (ac0 + ac1)
    assert dr_num % 4 == 0 and dc_num % 4 == 0, "pair midpoint must land on grid-aligned object centers"
    return dr_num // 4, dc_num // 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair line-linked objects by connector color and redraw their merged masks at the midpoint.

    Input structure:
    - The background is black.
    - Foreground objects are 4-connected monochrome shapes with 3-5 pixels each.
    - Every pair of objects is linked by a unique-color dashed line of singleton pixels,
      arranged horizontally or vertically with one background gap between consecutive dots.
    - Object colors can differ from the connector color and from each other.

    Rule:
    - Split non-background cells into 4-connected monochrome components.
    - Treat area-1 components grouped by color as one dashed connector, and area-3-to-5
      components as the endpoint objects.
    - For each connector color, find the two objects touching that connector, compute the
      midpoint between their bbox centers, translate both object masks so their bbox centers
      land on that midpoint, and paint the union in the connector color.
    - Remove the original objects and dashed connectors; the output keeps only the merged shapes.

    Key insight:
    - The connector color is the pairing key, while the output location is determined by bbox-center
      midpoint geometry rather than by either original object position alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    comps = _find_components(g, bg=BG)
    objects = [comp for comp in comps if 3 <= int(comp["area"]) <= 5]
    singleton_groups: dict[int, list[tuple[int, int]]] = {}

    for comp in comps:
        area = int(comp["area"])
        if area == 1:
            r, c = comp["coords"][0]
            singleton_groups.setdefault(int(comp["color"]), []).append((int(r), int(c)))
        else:
            assert 3 <= area <= 5, "foreground components must be either singleton connector dots or 3-5 pixel objects"

    assert objects, "expected paired objects"
    assert singleton_groups, "expected connector dots"

    used_objects: set[int] = set()
    out = np.zeros_like(g)
    occupied = np.zeros_like(g, dtype=bool)

    for line_color, coords in singleton_groups.items():
        _line_orientation(coords)
        line_cells = set(coords)
        touching = [i for i, obj in enumerate(objects) if _touches_line(obj, line_cells)]
        assert len(touching) == 2, "each connector must touch exactly two objects"
        a_idx, b_idx = touching
        assert a_idx not in used_objects and b_idx not in used_objects, "objects must belong to a single pair"
        used_objects.update(touching)

        obj_a = objects[a_idx]
        obj_b = objects[b_idx]
        dr, dc = _translation_to_midpoint(obj_a, obj_b)

        for obj, shift_r, shift_c in ((obj_a, dr, dc), (obj_b, -dr, -dc)):
            for r, c in obj["coords"]:
                nr = int(r) + shift_r
                nc = int(c) + shift_c
                assert 0 <= nr < h and 0 <= nc < w, "merged object must stay in bounds"
                if occupied[nr, nc]:
                    assert int(out[nr, nc]) == line_color, "different merged pairs must not overlap"
                out[nr, nc] = line_color
                occupied[nr, nc] = True

    assert len(used_objects) == len(objects), "every object must be matched to a connector"
    return out


# --------------- generator ---------------

PAIR_COUNT_CHOICES = (2, 3)
GRID_SIDE_RANGE = (20, 29)
DOT_COUNT_RANGE = (2, 5)
PLACEMENT_MARGIN = 1
GENERATION_RETRIES = 400


def _bool_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in ORTHO_DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _enumerate_masks() -> list[dict[str, object]]:
    masks: list[dict[str, object]] = []
    for bits in range(1 << 9):
        area = bits.bit_count()
        if area < 3 or area > 5:
            continue
        flat = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool)
        mask = flat.reshape(3, 3)
        if not _bool_connected(mask):
            continue
        coords = np.argwhere(mask)
        if tuple(_bbox(coords)) != (0, 0, 2, 2):
            continue

        anchors: set[str] = set()
        if mask[1, 0]:
            anchors.add("L")
        if mask[1, 2]:
            anchors.add("R")
        if mask[0, 1]:
            anchors.add("U")
        if mask[2, 1]:
            anchors.add("D")
        if not anchors:
            continue
        masks.append({"mask": mask, "anchors": anchors, "area": area})
    return masks


OBJECT_MASKS = _enumerate_masks()
LEFT_ANCHORED = [item for item in OBJECT_MASKS if "R" in item["anchors"]]
RIGHT_ANCHORED = [item for item in OBJECT_MASKS if "L" in item["anchors"]]
TOP_ANCHORED = [item for item in OBJECT_MASKS if "D" in item["anchors"]]
BOTTOM_ANCHORED = [item for item in OBJECT_MASKS if "U" in item["anchors"]]


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    canvas[top + rows, left + cols] = color


def _choose_pair_masks(orientation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pool_a = LEFT_ANCHORED if orientation == "H" else TOP_ANCHORED
    pool_b = RIGHT_ANCHORED if orientation == "H" else BOTTOM_ANCHORED
    for _ in range(80):
        a = random.choice(pool_a)
        b = random.choice(pool_b)
        mask_a = np.array(a["mask"], copy=True)
        mask_b = np.array(b["mask"], copy=True)
        union = mask_a | mask_b
        if np.array_equal(mask_a, mask_b):
            continue
        if int(union.sum()) <= max(int(a["area"]), int(b["area"])):
            continue
        return mask_a, mask_b, union
    return None


def _build_pair_patch(
    orientation: str,
    line_color: int,
    color_a: int,
    color_b: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    picked = _choose_pair_masks(orientation)
    if picked is None:
        return None
    mask_a, mask_b, union = picked
    dots = random.randint(*DOT_COUNT_RANGE)

    if orientation == "H":
        patch_h = 3
        patch_w = 2 * dots + 5
        inp = np.zeros((patch_h, patch_w), dtype=int)
        out = np.zeros((patch_h, patch_w), dtype=int)
        _paint_mask(inp, mask_a, 0, 0, color_a)
        _paint_mask(inp, mask_b, 0, patch_w - 3, color_b)
        for i in range(dots):
            inp[1, 3 + 2 * i] = line_color
        _paint_mask(out, union, 0, dots + 1, line_color)
        return inp, out

    patch_h = 2 * dots + 5
    patch_w = 3
    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)
    _paint_mask(inp, mask_a, 0, 0, color_a)
    _paint_mask(inp, mask_b, patch_h - 3, 0, color_b)
    for i in range(dots):
        inp[3 + 2 * i, 1] = line_color
    _paint_mask(out, union, dots + 1, 0, line_color)
    return inp, out


def _sample_orientations(pair_count: int) -> list[str]:
    if pair_count == 2:
        out = ["H", "V"]
        random.shuffle(out)
        return out

    majority = random.choice(["H", "V"])
    minority = "V" if majority == "H" else "H"
    out = [majority, majority, minority]
    random.shuffle(out)
    return out


def _place_boxes(
    grid_shape: tuple[int, int],
    box_shapes: list[tuple[int, int]],
    margin: int = PLACEMENT_MARGIN,
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    blocked = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(box_shapes)
    order = sorted(range(len(box_shapes)), key=lambda i: box_shapes[i][0] * box_shapes[i][1], reverse=True)

    for idx in order:
        box_h, box_w = box_shapes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - box_h + 1):
            rr0 = max(0, r0 - margin)
            rr1 = min(h, r0 + box_h + margin)
            for c0 in range(w - box_w + 1):
                cc0 = max(0, c0 - margin)
                cc1 = min(w, c0 + box_w + margin)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        rr0 = max(0, r0 - margin)
        rr1 = min(h, r0 + box_h + margin)
        cc0 = max(0, c0 - margin)
        cc1 = min(w, c0 + box_w + margin)
        blocked[rr0:rr1, cc0:cc1] = True

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != 0
    sub = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    assert not np.any(mask & (sub != 0))
    sub[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    orientations = _sample_orientations(pair_count)
    colors = random.sample(list(range(1, 10)), 3 * pair_count)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i, orientation in enumerate(orientations):
        line_color, color_a, color_b = colors[3 * i : 3 * i + 3]
        built = _build_pair_patch(orientation, line_color, color_a, color_b)
        if built is None:
            return None
        patches.append(built)

    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    positions = _place_boxes((grid_h, grid_w), [inp.shape for inp, _ in patches])
    if positions is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    out = np.zeros((grid_h, grid_w), dtype=int)
    for (inp_patch, out_patch), (r0, c0) in zip(patches, positions):
        _paste_nonzero(inp, inp_patch, r0, c0)
        _paste_nonzero(out, out_patch, r0, c0)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
