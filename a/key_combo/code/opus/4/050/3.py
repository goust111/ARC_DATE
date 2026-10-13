from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils_for_050_3",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BG = 0
MARKER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
BOX_OPTIONS = [(4, 4), (4, 5), (5, 4), (5, 5)]
AREA_MIN = 8
AREA_MAX = 15
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                cells.append((r, c))
    return cells


def _vertical_groups(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    groups: list[list[tuple[int, int]]] = []
    seen = set()
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or r not in {0, h - 1} and c not in {0, w - 1}:
                continue
            if (r, c) in seen:
                continue
            mc = w - 1 - c
            group = sorted({(r, c), (r, mc)})
            for cell in group:
                seen.add(cell)
            groups.append(group)
    return groups


def _border_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and (r in {0, h - 1} or c in {0, w - 1}):
                out.append((r, c))
    return out


def _generate_mask(symmetric: bool, min_markers: int) -> np.ndarray:
    for _ in range(500):
        h, w = random.choice(BOX_OPTIONS)
        area = random.randint(AREA_MIN, min(AREA_MAX, h * w - 1))
        mask = np.ones((h, w), dtype=bool)
        removed = 0

        if symmetric:
            groups = _vertical_groups(mask)
            random.shuffle(groups)
            while int(mask.sum()) > area and groups:
                group = groups.pop()
                if int(mask.sum()) - len(group) < area:
                    continue
                test = mask.copy()
                for r, c in group:
                    test[r, c] = False
                if not _touches_all_sides(test):
                    continue
                if not _mask_connected(test):
                    continue
                if len(_interior_cells(test)) < min_markers:
                    continue
                mask = test
                removed += len(group)
            if int(mask.sum()) != area or removed == 0:
                continue
            if not np.array_equal(mask, np.fliplr(mask)):
                continue
        else:
            candidates = _border_cells(mask)
            random.shuffle(candidates)
            while int(mask.sum()) > area and candidates:
                r, c = candidates.pop()
                test = mask.copy()
                test[r, c] = False
                if not _touches_all_sides(test):
                    continue
                if not _mask_connected(test):
                    continue
                if len(_interior_cells(test)) < min_markers:
                    continue
                mask = test
                removed += 1
                candidates = _border_cells(mask)
                random.shuffle(candidates)
            if int(mask.sum()) != area or removed == 0:
                continue
            if np.array_equal(mask, np.fliplr(mask)):
                continue

        if len(_interior_cells(mask)) >= min_markers:
            return mask
    raise RuntimeError("failed to generate object mask")


def _build_object(color: int, symmetric: bool, marker_count: int) -> dict:
    mask = _generate_mask(symmetric=symmetric, min_markers=marker_count)
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    marker_cells = random.sample(_interior_cells(mask), marker_count)
    for r, c in marker_cells:
        patch[r, c] = MARKER
    return {
        "color": int(color),
        "symmetric": bool(symmetric),
        "marker_count": int(marker_count),
        "patch": patch,
        "mask": mask,
        "shape": patch.shape,
        "area": int(mask.sum()),
    }


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    coords = np.argwhere(patch != BG)
    for dr, dc in coords:
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - padding)
        r2 = min(H, r + padding + 1)
        c1 = max(0, c - padding)
        c2 = min(W, c + padding + 1)
        if np.any(occupied[r1:r2, c1:c2]):
            return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, int(r0), int(c0), transparent=BG)
    occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch != BG


def _center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + r1), int(c0 + c1)


def _analyze_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects = []
    for obj in find_components(g, bg=bg, connectivity=4, monochromatic=False):
        coords = np.asarray(obj["coords"], dtype=int)
        colors = sorted({int(g[r, c]) for r, c in coords if int(g[r, c]) != MARKER})
        assert len(colors) == 1, "each object must have one body color"
        color = int(colors[0])
        bbox = tuple(map(int, obj["bbox"]))
        patch = crop_bbox(g, bbox)
        filled = np.where(patch == MARKER, color, patch)
        assert set(np.unique(patch)).issubset({bg, MARKER, color}), "unexpected colors inside object bbox"
        is_symmetric = bool(np.array_equal(filled, np.fliplr(filled)))
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "color": color,
                "marker_count": int(np.count_nonzero(g[coords[:, 0], coords[:, 1]] == MARKER)),
                "is_symmetric": is_symmetric,
                "center2": _center2(bbox),
                "top_left": (bbox[0], bbox[1]),
                "area": int(coords.shape[0]),
            }
        )
    return bg, objects


def _nearest_reference(candidate: dict, references: list[dict]) -> dict:
    dists = []
    cr, cc = candidate["center2"]
    for ref in references:
        rr, rc = ref["center2"]
        dists.append((abs(cr - rr) + abs(cc - rc), ref))
    dists.sort(key=lambda item: item[0])
    assert len(dists) >= 1
    if len(dists) >= 2:
        assert dists[0][0] != dists[1][0], "nearest reference must be unique"
    return dists[0][1]


def _outline_cells(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    own = {tuple(map(int, cell)) for cell in coords.tolist()}
    ring: set[tuple[int, int]] = set()
    for r, c in own:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in own:
                ring.add((nr, nc))
    return ring


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition objects by vertical symmetry, erase the least-marked asymmetric object, and outline the survivors with colors borrowed from nearby symmetric references.

    Input structure:
    - The background is black, with 4-6 separated foreground objects that are 4-connected and each use one body color plus optional grey marker pixels.
    - Grey (5) marker pixels lie inside an object's bounding box and belong to the same connected object as its body color.
    - Treating grey markers as body-colored cells, some object bbox crops are vertically symmetric while others are not.

    Rule:
    - For each foreground object, crop its bbox and replace grey markers by the object's body color only for the symmetry test.
    - Keep vertically symmetric objects as the reference group; among non-symmetric candidate objects, erase the unique object with the fewest grey markers.
    - For every remaining candidate, find the unique nearest reference by Manhattan distance between bbox centers, then draw a 1-pixel 8-neighborhood outline around that candidate using the matched reference color.
    - Process surviving candidates in row-major order of their bbox top-left corners; later outlines overwrite earlier outline pixels, but original non-background object pixels stay unchanged.

    Key insight:
    - Symmetry decides role rather than direct selection: symmetric objects donate outline colors, while asymmetric objects compete by internal marker counts and then inherit color through cross-group nearest-reference matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, objects = _analyze_objects(g)
    references = [obj for obj in objects if obj["is_symmetric"]]
    candidates = [obj for obj in objects if not obj["is_symmetric"]]
    assert references, "expected at least one symmetric reference object"
    assert len(candidates) >= 2, "expected at least two non-symmetric candidate objects"

    marker_counts = [obj["marker_count"] for obj in candidates]
    fewest = min(marker_counts)
    doomed = [obj for obj in candidates if obj["marker_count"] == fewest]
    assert len(doomed) == 1, "least marker count must be unique"
    erased = doomed[0]

    out = g.copy()
    erased_coords = erased["coords"]
    out[erased_coords[:, 0], erased_coords[:, 1]] = bg

    protected = out != bg
    survivors = sorted(
        [obj for obj in candidates if obj is not erased],
        key=lambda obj: (obj["top_left"][0], obj["top_left"][1]),
    )

    for candidate in survivors:
        ref = _nearest_reference(candidate, references)
        for r, c in _outline_cells(candidate["coords"], out.shape):
            if not protected[r, c]:
                out[r, c] = ref["color"]

    return out


def _pair_overlap_ok(
    pair_left: dict,
    pair_left_pos: tuple[int, int],
    pair_right: dict,
    pair_right_pos: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    left_coords = np.argwhere(pair_left["patch"] != BG) + np.array(pair_left_pos, dtype=int)
    right_coords = np.argwhere(pair_right["patch"] != BG) + np.array(pair_right_pos, dtype=int)
    left_ring = _outline_cells(left_coords, shape)
    right_ring = _outline_cells(right_coords, shape)
    return len(left_ring & right_ring) >= 1


def _build_layout() -> tuple[np.ndarray, dict] | None:
    H = random.randint(17, 21)
    W = random.randint(17, 21)
    mode_six = random.random() < 0.6

    colors = random.sample(PALETTE, 6 if mode_six else 5)
    ref_left = _build_object(colors[0], symmetric=True, marker_count=random.randint(1, 4))
    ref_right = _build_object(colors[1], symmetric=True, marker_count=random.randint(1, 4))
    pair_left = _build_object(colors[2], symmetric=False, marker_count=random.choice([2, 3, 4]))
    pair_right = _build_object(colors[3], symmetric=False, marker_count=random.choice([2, 3, 4]))

    extras = []
    extra_roles = ["extra_left", "extra_right"] if mode_six else ["extra_single"]
    for i, role in enumerate(extra_roles, start=4):
        extras.append((role, _build_object(colors[i], symmetric=False, marker_count=1)))

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    placed: dict[str, tuple[dict, tuple[int, int]]] = {}

    left_ref_positions = [
        (r, c)
        for r in range(1, 4)
        for c in range(1, 4)
    ]
    right_ref_positions = [
        (r, c)
        for r in range(1, 4)
        for c in range(W - ref_right["shape"][1] - 3, W - ref_right["shape"][1])
    ]
    random.shuffle(left_ref_positions)
    random.shuffle(right_ref_positions)

    for r, c in left_ref_positions:
        if _can_place(occupied, ref_left["patch"], r, c):
            _place_patch(grid, occupied, ref_left["patch"], r, c)
            placed["ref_left"] = (ref_left, (r, c))
            break
    else:
        return None

    for r, c in right_ref_positions:
        if _can_place(occupied, ref_right["patch"], r, c):
            _place_patch(grid, occupied, ref_right["patch"], r, c)
            placed["ref_right"] = (ref_right, (r, c))
            break
    else:
        return None

    pair_band = list(range(max(6, H // 2 - 1), min(H - 6, H // 2 + 2) + 1))
    random.shuffle(pair_band)
    pair_success = False
    for row_left in pair_band:
        row_right_choices = pair_band[:]
        random.shuffle(row_right_choices)
        for row_right in row_right_choices:
            min_left = max(2, W // 2 - pair_left["shape"][1] - 2)
            max_left = min(W // 2 - 1, W - pair_left["shape"][1] - pair_right["shape"][1] - 2)
            if min_left > max_left:
                continue
            c_left_choices = list(range(min_left, max_left + 1))
            random.shuffle(c_left_choices)
            for c_left in c_left_choices:
                c_right = c_left + pair_left["shape"][1] + 1
                if not _can_place(occupied, pair_left["patch"], row_left, c_left):
                    continue
                if not _can_place(occupied, pair_right["patch"], row_right, c_right):
                    continue
                if not _pair_overlap_ok(pair_left, (row_left, c_left), pair_right, (row_right, c_right), (H, W)):
                    continue
                _place_patch(grid, occupied, pair_left["patch"], row_left, c_left)
                _place_patch(grid, occupied, pair_right["patch"], row_right, c_right)
                placed["pair_left"] = (pair_left, (row_left, c_left))
                placed["pair_right"] = (pair_right, (row_right, c_right))
                pair_success = True
                break
            if pair_success:
                break
        if pair_success:
            break
    if not pair_success:
        return None

    for role, obj in extras:
        if role == "extra_left":
            candidates = [
                (r, c)
                for r in range(H - obj["shape"][0] - 4, H - obj["shape"][0] - 1)
                for c in range(1, 5)
            ]
        elif role == "extra_right":
            candidates = [
                (r, c)
                for r in range(H - obj["shape"][0] - 4, H - obj["shape"][0] - 1)
                for c in range(W - obj["shape"][1] - 5, W - obj["shape"][1] - 1)
            ]
        else:
            left_side = random.random() < 0.5
            if left_side:
                candidates = [
                    (r, c)
                    for r in range(H - obj["shape"][0] - 4, H - obj["shape"][0] - 1)
                    for c in range(1, 5)
                ]
            else:
                candidates = [
                    (r, c)
                    for r in range(H - obj["shape"][0] - 4, H - obj["shape"][0] - 1)
                    for c in range(W - obj["shape"][1] - 5, W - obj["shape"][1] - 1)
                ]
        random.shuffle(candidates)
        for r, c in candidates:
            if _can_place(occupied, obj["patch"], r, c):
                _place_patch(grid, occupied, obj["patch"], r, c)
                placed[role] = (obj, (r, c))
                break
        else:
            return None

    return grid, placed


def _validate_generated(grid: np.ndarray, placed: dict) -> bool:
    try:
        bg, objects = _analyze_objects(grid)
    except AssertionError:
        return False
    if bg != BG:
        return False
    if len(objects) not in {5, 6}:
        return False

    refs = [obj for obj in objects if obj["is_symmetric"]]
    cands = [obj for obj in objects if not obj["is_symmetric"]]
    if len(refs) != 2 or len(cands) not in {3, 4}:
        return False
    counts = [obj["marker_count"] for obj in cands]
    if sorted(counts) != list(range(1, len(cands) + 1)):
        return False

    fewest = min(counts)
    erased = [obj for obj in cands if obj["marker_count"] == fewest]
    if len(erased) != 1:
        return False
    erased = erased[0]

    areas = [obj["area"] for obj in cands]
    if erased["area"] == min(areas) and areas.count(min(areas)) == 1:
        return False

    left_cols = [obj["bbox"][1] for obj in cands]
    if erased["bbox"][1] == min(left_cols) and left_cols.count(min(left_cols)) == 1:
        return False

    survivors = sorted(
        [obj for obj in cands if obj is not erased],
        key=lambda obj: (obj["bbox"][0], obj["bbox"][1]),
    )
    matched_colors = []
    for obj in survivors:
        try:
            ref = _nearest_reference(obj, refs)
        except AssertionError:
            return False
        matched_colors.append(ref["color"])
    if len(set(matched_colors)) < 2:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False

    pair_left_obj, pair_left_pos = placed["pair_left"]
    pair_right_obj, pair_right_pos = placed["pair_right"]
    left_coords = np.argwhere(pair_left_obj["patch"] != BG) + np.array(pair_left_pos, dtype=int)
    right_coords = np.argwhere(pair_right_obj["patch"] != BG) + np.array(pair_right_pos, dtype=int)
    pair_left_ring = _outline_cells(left_coords, grid.shape)
    pair_right_ring = _outline_cells(right_coords, grid.shape)
    if len(pair_left_ring & pair_right_ring) < 1:
        return False

    if np.array_equal(out, grid):
        return False
    return True


def _reassign_candidate_markers(grid: np.ndarray) -> np.ndarray:
    bg, objects = _analyze_objects(grid)
    refs = [obj for obj in objects if obj["is_symmetric"]]
    cands = [obj for obj in objects if not obj["is_symmetric"]]
    assert bg == BG and len(refs) == 2 and len(cands) in {3, 4}

    counts = list(range(1, len(cands) + 1))
    heavy_roles = cands[:]
    random.shuffle(heavy_roles)
    out = grid.copy()
    for obj in cands:
        coords = obj["coords"]
        body_color = obj["color"]
        out[coords[:, 0], coords[:, 1]] = body_color

    for obj, count in zip(heavy_roles, counts):
        bbox = obj["bbox"]
        patch = crop_bbox(out, bbox)
        mask = patch == obj["color"]
        interior = _interior_cells(mask)
        if len(interior) < count:
            raise AssertionError("candidate lost required interior cells")
        for r, c in random.sample(interior, count):
            patch[r, c] = MARKER
        paste(out, patch, bbox[0], bbox[1], transparent=None)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_layout()
        if built is None:
            continue
        grid, placed = built
        try:
            grid = _reassign_candidate_markers(grid)
        except AssertionError:
            continue
        if _validate_generated(grid, placed):
            return grid
    raise RuntimeError("failed to generate valid symmetry-group outline puzzle input")
