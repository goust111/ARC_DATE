from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import (
    bbox,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


BG = 0
NON_BG = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _orbit_positions(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (int(r), int(c)),
        (int(h - 1 - r), int(c)),
        (int(r), int(w - 1 - c)),
        (int(h - 1 - r), int(w - 1 - c)),
    }
    return tuple(sorted(pts))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask.astype(int), b).astype(bool)


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            key = candidate.shape.__repr__().encode() + candidate.astype(np.uint8).tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append(candidate.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> bytes:
    return min(
        v.shape.__repr__().encode() + v.astype(np.uint8).tobytes()
        for v in _dihedral_masks(mask)
    )


def _connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    comps = components_mask(mask.astype(bool), connectivity=connectivity)
    return len(comps) == 1


def _find_pattern_component(grid: np.ndarray, bg: int) -> dict:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert comps, "expected at least one foreground component"
    return max(comps, key=lambda comp: int(comp["area"]))


def _find_defect_mask(patch: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, patch.shape)
    defect = np.zeros((h, w), dtype=bool)
    seen: set[tuple[tuple[int, int], ...]] = set()

    for r in range(h):
        for c in range(w):
            orbit = _orbit_positions(r, c, h, w)
            if orbit in seen:
                continue
            seen.add(orbit)

            vals = [int(patch[rr, cc]) for rr, cc in orbit]
            fg_colors = {v for v in vals if v != bg}
            if len(fg_colors) != 1:
                continue
            if all(v == bg for v in vals) or all(v != bg for v in vals):
                continue
            for (rr, cc), val in zip(orbit, vals):
                if val == bg:
                    defect[rr, cc] = True

    return defect


def _candidate_components(grid: np.ndarray, bg: int, pattern_bbox: tuple[int, int, int, int]) -> list[dict]:
    pr0, pc0, pr1, pc1 = map(int, pattern_bbox)
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    out = []
    for comp in comps:
        if tuple(map(int, comp["bbox"])) == (pr0, pc0, pr1, pc1):
            continue
        out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the symmetry-broken hidden shape and output the uniquely matching candidate piece.

    Input structure:
    - The dominant background surrounds one large multicolor pattern and several smaller candidate pieces.
    - The large pattern occupies a full bounding box and was originally mirror-symmetric across both its
      central row and central column, while still containing many natural background holes.
    - A small set of foreground cells inside that pattern was changed to background; each candidate piece is
      a separate connected monochrome sprite elsewhere on the canvas.

    Rule:
    - Isolate the largest foreground component as the damaged main pattern and crop it to its bounding box.
    - Group each cell with its horizontal, vertical, and double-mirrored partners inside that box; any orbit
      containing both background and one consistent foreground color marks its background members as defects.
    - Crop the full defect mask to its minimal box, compare that shape against every candidate under the 8
      dihedral transforms, and output the uniquely matching candidate in its original color.

    Key insight:
    - The answer is not the repaired pattern: symmetry only exposes a hidden negative-space shape, and the
      real task is to identify which external candidate sprite has that same shape up to rotation or flip.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    pattern = _find_pattern_component(g, bg)
    pattern_bbox = tuple(map(int, pattern["bbox"]))
    patch = crop_bbox(g, pattern_bbox)

    defect_mask = _find_defect_mask(patch, bg)
    defect_box = bbox(defect_mask)
    assert defect_box is not None, "expected a non-empty hidden defect"
    defect_crop = crop_bbox(defect_mask.astype(int), defect_box).astype(bool)

    matches: list[np.ndarray] = []
    for comp in _candidate_components(g, bg, pattern_bbox):
        candidate_patch = crop_bbox(g, tuple(map(int, comp["bbox"])))
        candidate_mask = candidate_patch != bg
        if any(np.array_equal(candidate_mask, variant) for variant in _dihedral_masks(defect_crop)):
            matches.append(candidate_patch)

    assert len(matches) == 1, "expected exactly one candidate to match the defect shape"
    return matches[0]


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_shape(
    *,
    area: int | None = None,
    bbox_shape: tuple[int, int] | None = None,
    banned_keys: set[bytes] | None = None,
) -> np.ndarray | None:
    banned_keys = banned_keys or set()
    for _ in range(300):
        if bbox_shape is None:
            h = random.randint(2, 5)
            w = random.randint(2, 5)
        else:
            h, w = map(int, bbox_shape)
        max_area = min(8, h * w - 1)
        min_area = 4
        if max_area < min_area:
            continue
        cell_count = int(area if area is not None else random.randint(min_area, max_area))
        if cell_count > h * w - 1:
            continue
        grown = _random_connected_mask(h, w, cell_count)
        if grown is None:
            continue
        cropped = _crop_mask(grown)
        if min(cropped.shape) < 2:
            continue
        if cropped.all():
            continue
        if bbox_shape is not None and tuple(map(int, cropped.shape)) != tuple(map(int, bbox_shape)):
            continue
        if not _connected(cropped, connectivity=4):
            continue
        key = _canonical_key(cropped)
        if key in banned_keys:
            continue
        return cropped
    return None


def _build_symmetric_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    target_holes = random.randint(max(10, area // 6), max(14, area // 3))
    mask = np.ones((h, w), dtype=bool)

    eligible = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            orbit = _orbit_positions(r, c, h, w)
            if orbit in seen:
                continue
            seen.add(orbit)
            if any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in orbit):
                continue
            eligible.append(orbit)
    random.shuffle(eligible)

    holes = 0
    min_fg = int(area * 0.58)
    for orbit in eligible:
        if holes >= target_holes:
            break
        test = mask.copy()
        for rr, cc in orbit:
            test[rr, cc] = False
        if int(test.sum()) < min_fg:
            continue
        if test.sum(axis=0).min() < 2 or test.sum(axis=1).min() < 2:
            continue
        if not _connected(test, connectivity=8):
            continue
        mask = test
        holes += len(orbit)

    tl = mask[: h // 2, : w // 2]
    if holes < 10 or np.count_nonzero(~tl) < 3:
        return None
    return mask


def _colorize_pattern(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = np.zeros((h, w), dtype=int)
    seen: set[tuple[tuple[int, int], ...]] = set()
    used = set()
    fg_orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orbit = _orbit_positions(r, c, h, w)
            if orbit in seen:
                continue
            seen.add(orbit)
            if not all(mask[rr, cc] for rr, cc in orbit):
                continue
            color = int(random.choice(palette))
            used.add(color)
            fg_orbits.append(orbit)
            for rr, cc in orbit:
                out[rr, cc] = color

    if len(used) < 2 and len(fg_orbits) >= 2:
        for idx, orbit in enumerate(fg_orbits[:2]):
            color = int(palette[idx % len(palette)])
            for rr, cc in orbit:
                out[rr, cc] = color
    return out


def _place_defect(mask: np.ndarray, defect_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = map(int, mask.shape)
    sh, sw = map(int, defect_shape.shape)
    max_r0 = h // 2 - sh
    max_c0 = w // 2 - sw
    if max_r0 < 1 or max_c0 < 1:
        return None

    positions = [(r0, c0) for r0 in range(1, max_r0 + 1) for c0 in range(1, max_c0 + 1)]
    random.shuffle(positions)
    flip_r = random.choice([False, True])
    flip_c = random.choice([False, True])

    for r0, c0 in positions:
        coords = [(r0 + r, c0 + c) for r, c in np.argwhere(defect_shape)]
        if not all(mask[r, c] for r, c in coords):
            continue
        mapped = []
        for r, c in coords:
            rr = h - 1 - r if flip_r else r
            cc = w - 1 - c if flip_c else c
            mapped.append((int(rr), int(cc)))

        test = mask.copy()
        for rr, cc in mapped:
            test[rr, cc] = False
        if not _connected(test, connectivity=8):
            continue

        local_bg = 0
        for rr, cc in mapped:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mapped and not mask[nr, nc]:
                        local_bg += 1
        if local_bg == 0:
            continue

        defect_mask = np.zeros((h, w), dtype=bool)
        for rr, cc in mapped:
            defect_mask[rr, cc] = True
        return test, defect_mask
    return None


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _build_candidate_shapes(target_shape: np.ndarray) -> list[np.ndarray] | None:
    target_key = _canonical_key(target_shape)
    banned = {target_key}

    same_area = _sample_shape(area=int(target_shape.sum()), banned_keys=banned)
    if same_area is None:
        return None
    banned.add(_canonical_key(same_area))

    same_box = _sample_shape(bbox_shape=tuple(map(int, target_shape.shape)), banned_keys=banned)
    if same_box is None:
        return None
    banned.add(_canonical_key(same_box))

    shapes = [target_shape, same_area, same_box]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        extra = _sample_shape(banned_keys=banned)
        if extra is None:
            return None
        banned.add(_canonical_key(extra))
        shapes.append(extra)

    random.shuffle(shapes[1:])
    return shapes


def _match_count(defect_crop: np.ndarray, candidate_masks: list[np.ndarray]) -> int:
    variants = _dihedral_masks(defect_crop)
    return sum(any(np.array_equal(mask, v) for v in variants) for mask in candidate_masks)


def _gen_attempt() -> np.ndarray | None:
    pattern_h = random.randrange(11, 20, 2)
    pattern_w = random.randrange(11, 20, 2)

    base_mask = _build_symmetric_mask(pattern_h, pattern_w)
    if base_mask is None:
        return None

    target_base = _sample_shape()
    if target_base is None:
        return None
    defect_shape = random.choice(_dihedral_masks(target_base))

    placed = _place_defect(base_mask, defect_shape)
    if placed is None:
        return None
    damaged_mask, true_defect = placed

    pattern_colors = random.sample(NON_BG, 3)
    candidate_palette = [c for c in NON_BG if c not in pattern_colors]

    full_pattern = _colorize_pattern(base_mask, pattern_colors)
    damaged_pattern = full_pattern.copy()
    damaged_pattern[true_defect] = BG
    if len(find_components(damaged_pattern, bg=BG, connectivity=8, monochromatic=False, return_format="dict")) != 1:
        return None

    target_candidate = random.choice(_dihedral_masks(target_base))
    candidate_shapes = _build_candidate_shapes(target_candidate)
    if candidate_shapes is None:
        return None
    candidate_shapes[0] = target_candidate
    random.shuffle(candidate_shapes)

    candidate_keys = [_canonical_key(shape) for shape in candidate_shapes]
    true_index = candidate_keys.index(_canonical_key(target_candidate))
    candidate_masks = [shape.astype(bool) for shape in candidate_shapes]
    if _match_count(_crop_mask(true_defect), candidate_masks) != 1:
        return None

    candidate_patches = []
    for shape, color in zip(candidate_shapes, random.sample(candidate_palette, len(candidate_shapes))):
        oriented = random.choice(_dihedral_masks(shape))
        candidate_patches.append(_shape_patch(oriented, color))

    expected_output = candidate_patches[true_index].copy()

    layout = random.choice(["below", "right"])
    gap = random.randint(2, 4)
    margin = random.randint(1, 3)
    cand_max_h = max(p.shape[0] for p in candidate_patches)
    cand_max_w = max(p.shape[1] for p in candidate_patches)
    cand_sum_h = sum(p.shape[0] for p in candidate_patches) + gap * (len(candidate_patches) - 1)
    cand_sum_w = sum(p.shape[1] for p in candidate_patches) + gap * (len(candidate_patches) - 1)

    if layout == "below":
        H = min(30, margin * 2 + pattern_h + gap + max(cand_max_h + 4, 8))
        W = min(30, max(margin * 2 + pattern_w, margin * 2 + cand_sum_w + 4))
    else:
        H = min(30, max(margin * 2 + pattern_h, margin * 2 + cand_sum_h + 4))
        W = min(30, margin * 2 + pattern_w + gap + max(cand_max_w + 4, 8))

    if H < pattern_h + 6 or W < pattern_w + 6:
        return None

    canvas = np.full((H, W), BG, dtype=int)
    pattern_r = random.randint(1, H - pattern_h - (cand_max_h + 2 if layout == "below" else 1) - 1)
    pattern_c = random.randint(1, W - pattern_w - (cand_max_w + 2 if layout == "right" else 1) - 1)
    paste(canvas, damaged_pattern, pattern_r, pattern_c, transparent=BG)

    band = np.full_like(canvas, BG)
    if layout == "below":
        band[: pattern_r + pattern_h + gap, :] = 9
    else:
        band[:, : pattern_c + pattern_w + gap] = 9

    for patch in candidate_patches:
        tries = 0
        placed_ok = False
        while tries < 200:
            tries += 1
            probe = canvas.copy()
            probe[band != BG] = 9
            try:
                rr, cc = random_free_location(probe, patch, transparent=BG, border=1, padding=1, padding_connectivity=8)
            except ValueError:
                return None
            if band[rr : rr + patch.shape[0], cc : cc + patch.shape[1]].any():
                continue
            if layout == "below" and rr <= pattern_r + pattern_h + 1:
                continue
            if layout == "right" and cc <= pattern_c + pattern_w + 1:
                continue
            paste(canvas, patch, rr, cc, transparent=BG)
            placed_ok = True
            break
        if not placed_ok:
            return None

    bg = int(get_bg(canvas))
    if bg != BG:
        return None

    try:
        solved = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected_output):
        return None

    pattern = _find_pattern_component(canvas, BG)
    patch = crop_bbox(canvas, tuple(map(int, pattern["bbox"])))
    defect_mask = _find_defect_mask(patch, BG)
    defect_box = bbox(defect_mask)
    if defect_box is None:
        return None
    defect_crop = crop_bbox(defect_mask.astype(int), defect_box).astype(bool)
    if _canonical_key(defect_crop) != _canonical_key(_crop_mask(true_defect)):
        return None

    candidate_comps = _candidate_components(canvas, BG, tuple(map(int, pattern["bbox"])))
    candidate_masks_now = [crop_bbox(canvas, tuple(map(int, comp["bbox"]))) != BG for comp in candidate_comps]
    if _match_count(defect_crop, candidate_masks_now) != 1:
        return None

    natural_holes = np.count_nonzero((base_mask == 0) & (~true_defect))
    if natural_holes < int(true_defect.sum()) + 6:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
