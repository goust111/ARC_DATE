from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    utils_path = str(_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.append(utils_path)

from arc_utils import Color, crop_bbox, find_components, paste, random_free_location


BG = Color.BLACK
GEN_MAX_ATTEMPTS = 400
GEN_GROUP_COUNT_CHOICES = (2, 3)
GEN_GROUP_SIZE_WEIGHTS = ((2, 0.4), (3, 0.45), (4, 0.15))
GEN_PROTO_SIDE_RANGE = (5, 8)
GEN_CANVAS_RANGE = (22, 30)
GEN_PADDING = 1
GEN_GROUP_TRIES = 200
GEN_COPY_TRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    stack = [start]
    seen = {start}

    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))

    return len(seen) == int(mask.sum())


def _prototype_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _interior_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if bool(mask[r, c])
    ]


def _choose_cells(
    pool: list[tuple[int, int]],
    count: int,
    *,
    avoid: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    avoid = set() if avoid is None else set(avoid)
    candidates = [cell for cell in pool if cell not in avoid]
    if len(candidates) < count:
        return None

    for min_gap in (1, 0):
        ordered = candidates[:]
        random.shuffle(ordered)
        chosen: list[tuple[int, int]] = []
        for cell in ordered:
            if any(max(abs(cell[0] - r), abs(cell[1] - c)) <= min_gap for r, c in chosen):
                continue
            chosen.append(cell)
            if len(chosen) == count:
                return chosen

    return random.sample(candidates, count)


def _carve_corner(mask: np.ndarray) -> None:
    h, w = mask.shape
    notch_h = random.randint(1, max(1, h - 3))
    notch_w = random.randint(1, max(1, w - 3))
    corner = random.choice(("tl", "tr", "bl", "br"))

    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False


def _carve_side(mask: np.ndarray) -> None:
    h, w = mask.shape
    side = random.choice(("top", "bottom", "left", "right"))

    if side in ("top", "bottom"):
        depth = random.randint(1, min(2, h - 3))
        length = random.randint(1, w - 2)
        start = random.randint(1, w - length - 1)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        mask[rows, start : start + length] = False
    else:
        depth = random.randint(1, min(2, w - 3))
        length = random.randint(1, h - 2)
        start = random.randint(1, h - length - 1)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        mask[start : start + length, cols] = False


def _make_prototype() -> np.ndarray:
    for _ in range(GEN_GROUP_TRIES):
        h = random.randint(*GEN_PROTO_SIDE_RANGE)
        w = random.randint(*GEN_PROTO_SIDE_RANGE)
        mask = np.ones((h, w), dtype=bool)

        for _ in range(random.randint(2, 4)):
            if random.random() < 0.6:
                _carve_corner(mask)
            else:
                _carve_side(mask)

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = np.rot90(mask, random.randrange(4))

        if mask.all():
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _mask_connected(mask):
            continue

        interior = _interior_coords(mask)
        if len(interior) < 5:
            continue
        if int(mask.sum()) < max(12, (mask.shape[0] * mask.shape[1]) // 2):
            continue
        if int((~mask).sum()) < 2:
            continue
        return mask

    raise RuntimeError("failed to generate prototype")


def _copies_from_missing_sets(
    prototype: np.ndarray,
    missing_sets: list[set[tuple[int, int]]],
) -> list[np.ndarray] | None:
    copies: list[np.ndarray] = []
    for missing in missing_sets:
        copy_mask = prototype.copy()
        for r, c in missing:
            copy_mask[r, c] = False
        if not _mask_connected(copy_mask):
            return None
        copies.append(copy_mask)

    union = np.any(np.stack(copies, axis=0), axis=0)
    if not np.array_equal(union, prototype):
        return None
    if any(np.array_equal(copy_mask, prototype) for copy_mask in copies):
        return None
    if len({_prototype_signature(copy_mask) for copy_mask in copies}) != len(copies):
        return None
    return copies


def _build_hard_group_copies(prototype: np.ndarray) -> list[np.ndarray] | None:
    interior = _interior_coords(prototype)
    for _ in range(GEN_COPY_TRIES):
        cells = _choose_cells(interior, 3)
        if cells is None:
            return None
        a, b, x = cells

        missing_sets = [
            {a, b},
            {b, x},
            {a, x},
        ]

        remaining = [cell for cell in interior if cell not in {a, b, x}]
        for idx in (1, 2):
            if remaining and random.random() < 0.45:
                extra = _choose_cells(remaining, 1, avoid=missing_sets[idx])
                if extra is not None:
                    missing_sets[idx].update(extra)

        copies = _copies_from_missing_sets(prototype, missing_sets)
        if copies is None:
            continue

        if any(np.array_equal(np.logical_or(copies[i], copies[j]), prototype) for i in range(3) for j in range(i + 1, 3)):
            continue

        random.shuffle(copies)
        return copies
    return None


def _build_regular_group_copies(prototype: np.ndarray, n_copies: int) -> list[np.ndarray] | None:
    interior = _interior_coords(prototype)
    for _ in range(GEN_COPY_TRIES):
        if n_copies == 2:
            if random.random() < 0.5:
                cells = _choose_cells(interior, 2)
                if cells is None:
                    return None
                a, x = cells
                missing_sets = [{a}, {x}]
                special = {a, x}
            else:
                cells = _choose_cells(interior, 3)
                if cells is None:
                    return None
                a, b, x = cells
                missing_sets = [{a, b}, {x}]
                special = {a, b, x}
        elif n_copies == 3:
            cells = _choose_cells(interior, 3)
            if cells is None:
                return None
            a, b, x = cells
            missing_sets = [{a, b}, {b, x}, {a, x}]
            special = {a, b, x}
        else:
            cells = _choose_cells(interior, 4)
            if cells is None:
                return None
            a, b, x, y = cells
            missing_sets = [{a, b}, {b, x}, {a, y}, {x, y}]
            special = {a, b, x, y}

        remaining = [cell for cell in interior if cell not in special]
        for idx in range(1, n_copies):
            if remaining and random.random() < 0.55:
                extra_count = 1 if random.random() < 0.7 else min(2, len(remaining))
                extra = _choose_cells(remaining, extra_count, avoid=missing_sets[idx])
                if extra is not None:
                    missing_sets[idx].update(extra)

        copies = _copies_from_missing_sets(prototype, missing_sets)
        if copies is None:
            continue

        random.shuffle(copies)
        return copies
    return None


def _make_group_specs() -> list[dict] | None:
    n_groups = random.choice(GEN_GROUP_COUNT_CHOICES)
    sizes = [3]
    while len(sizes) < n_groups:
        sizes.append(random.choices(
            [size for size, _weight in GEN_GROUP_SIZE_WEIGHTS],
            weights=[weight for _size, weight in GEN_GROUP_SIZE_WEIGHTS],
            k=1,
        )[0])

    colors = random.sample(list(Color.NOT_BLACK), n_groups)
    specs: list[dict] = []
    seen_signatures: set[tuple[tuple[int, int], bytes]] = set()

    for idx, (size, color) in enumerate(zip(sizes, colors)):
        for _ in range(GEN_GROUP_TRIES):
            prototype = _make_prototype()
            sig = _prototype_signature(prototype)
            if sig in seen_signatures:
                continue

            if idx == 0:
                copies = _build_hard_group_copies(prototype)
            else:
                copies = _build_regular_group_copies(prototype, size)

            if copies is None:
                continue

            seen_signatures.add(sig)
            specs.append(
                {
                    "color": int(color),
                    "prototype": prototype,
                    "copies": copies,
                }
            )
            break
        else:
            return None

    random.shuffle(specs)
    return specs


def _choose_canvas_shape(specs: list[dict]) -> tuple[int, int]:
    total_bbox_area = sum(int(spec["prototype"].size) * len(spec["copies"]) for spec in specs)
    base = max(GEN_CANVAS_RANGE[0], int(np.ceil(np.sqrt(total_bbox_area * 1.9))))
    low = min(GEN_CANVAS_RANGE[1], max(GEN_CANVAS_RANGE[0], base - 2))
    high = min(GEN_CANVAS_RANGE[1], max(low, base + 3))
    rows = random.randint(low, high)
    cols = random.randint(low, high)
    return rows, cols


def _render_scene(specs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = _choose_canvas_shape(specs)
    inp = np.full((rows, cols), BG, dtype=int)
    expected = np.full((rows, cols), BG, dtype=int)
    occupied = np.zeros((rows, cols), dtype=int)

    placements = []
    for spec in specs:
        for copy_mask in spec["copies"]:
            placements.append((spec["color"], spec["prototype"], copy_mask))
    random.shuffle(placements)

    for color, prototype, copy_mask in placements:
        h, w = copy_mask.shape
        bbox_patch = np.ones((h, w), dtype=int)
        try:
            r0, c0 = random_free_location(
                occupied,
                bbox_patch,
                transparent=0,
                padding=GEN_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(occupied, bbox_patch, r0, c0, transparent=0)
        paste(inp, np.where(copy_mask, color, BG), r0, c0, transparent=BG)
        paste(expected, np.where(prototype, color, BG), r0, c0, transparent=BG)

    return inp, expected


def _recover_groups(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    grouped: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        grouped[int(comp["color"])].append(comp)

    groups = []
    for color, color_comps in sorted(grouped.items()):
        color_comps = sorted(color_comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        crops = [crop_bbox(grid == color, comp["bbox"]) for comp in color_comps]
        prototype = np.any(np.stack(crops, axis=0), axis=0)
        groups.append(
            {
                "color": color,
                "comps": color_comps,
                "crops": crops,
                "prototype": prototype,
            }
        )
    return groups


def _shortcut_biggest_copy(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for group in _recover_groups(out):
        biggest = max(group["crops"], key=lambda crop: int(crop.sum()))
        patch = np.where(biggest, group["color"], BG)
        for comp in group["comps"]:
            r0, c0, _r1, _c1 = comp["bbox"]
            paste(out, patch, r0, c0, transparent=BG)
    return out


def _shortcut_fill_bbox(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for group in _recover_groups(out):
        for comp in group["comps"]:
            r0, c0, r1, c1 = comp["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = group["color"]
    return out


def _validate_scene(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    groups = _recover_groups(inp)
    if not (2 <= len(groups) <= 3):
        return False
    if len({_prototype_signature(group["prototype"]) for group in groups}) != len(groups):
        return False

    found_multi_copy_requirement = False
    for group in groups:
        crops = group["crops"]
        prototype = group["prototype"]
        color = group["color"]
        h, w = prototype.shape

        if not (2 <= len(crops) <= 4):
            return False
        if prototype.all():
            return False
        if not _mask_connected(prototype):
            return False
        if not any(np.any(prototype & ~crop) for crop in crops):
            return False
        if len({_prototype_signature(crop) for crop in crops}) != len(crops):
            return False

        near_complete = 0
        for crop in crops:
            if crop.shape != prototype.shape:
                return False
            if not _mask_connected(crop):
                return False
            if np.array_equal(crop, prototype):
                return False
            missing = np.argwhere(prototype & ~crop)
            if len(missing) in (1, 2) and all(0 < int(r) < h - 1 and 0 < int(c) < w - 1 for r, c in missing):
                near_complete += 1
        if near_complete == 0:
            return False

        pair_complete = any(
            np.array_equal(np.logical_or(crops[i], crops[j]), prototype)
            for i in range(len(crops))
            for j in range(i + 1, len(crops))
        )
        if len(crops) >= 3 and not pair_complete:
            found_multi_copy_requirement = True

        full_patch = np.where(prototype, color, BG)
        for comp in group["comps"]:
            if not np.array_equal(crop_bbox(expected, comp["bbox"]), full_patch):
                return False

    if not found_multi_copy_requirement:
        return False
    if np.array_equal(_shortcut_biggest_copy(inp), expected):
        return False
    if np.array_equal(_shortcut_fill_bbox(inp), expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each sprite from the union of its same-color corrupted copies.

    Input structure:
    - A black background contains 2 to 3 color groups of separated monochromatic sprites.
    - Each color group has 2 to 4 copies of one prototype, all with the same bounding-box size and orientation.
    - Every visible copy is corrupted: some prototype pixels are missing, but the bbox extremes remain so the cropped boxes still align.

    Rule:
    - Partition sprites by color.
    - Within each color group, crop every copy to its bounding box, align those crops by their top-left bbox corner, and take the pixelwise union to recover the prototype mask.
    - Replace every corrupted copy in place by that recovered full prototype while leaving the rest of the grid unchanged.

    Key insight:
    - No complete template is present anywhere; the only full prototype is the overlay union of all same-color incomplete copies.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for group in _recover_groups(g):
        prototype = group["prototype"]
        patch = np.where(prototype, group["color"], BG)
        for comp in group["comps"]:
            r0, c0, _r1, _c1 = comp["bbox"]
            paste(out, patch, r0, c0, transparent=BG)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        specs = _make_group_specs()
        if specs is None:
            continue

        rendered = _render_scene(specs)
        if rendered is None:
            continue

        inp, expected = rendered
        if _validate_scene(inp, expected):
            return inp

    raise RuntimeError("failed to generate valid input")
