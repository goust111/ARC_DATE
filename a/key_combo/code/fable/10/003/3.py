from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
TEMPLATE_DIMS = ((2, 3), (2, 4), (3, 4))
MIN_TEMPLATE_AREA = 4
MAX_TEMPLATE_AREA = 8
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cur: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cur.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cur, dtype=int))
    return out


def _find_foreground_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = sorted({int(grid[int(r), int(c)]) for r, c in coords})
        comps.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "colors": colors,
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _transform_patch(patch: np.ndarray, rot: int, mirrored: bool) -> np.ndarray:
    out = np.rot90(np.asarray(patch, dtype=int), int(rot))
    if mirrored:
        out = np.fliplr(out)
    return out.copy()


def _dihedral_variants(patch: np.ndarray, bg: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rot in range(4):
        base = np.rot90(patch, rot)
        for mirrored in (False, True):
            cur = np.fliplr(base) if mirrored else base
            mask = cur != bg
            key = (tuple(map(int, mask.shape)), mask.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "patch": cur.copy(),
                    "mask": mask.copy(),
                    "rot": int(rot),
                    "mirrored": bool(mirrored),
                    "shape_key": key,
                }
            )
    return out


def _scale_patch(patch: np.ndarray, k: int) -> np.ndarray:
    return np.kron(np.asarray(patch, dtype=int), np.ones((int(k), int(k)), dtype=int))


def _downsample_mask(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None
    block_view = mask.reshape(hh // k, k, ww // k, k).transpose(0, 2, 1, 3)
    all_on = block_view.all(axis=(2, 3))
    any_on = block_view.any(axis=(2, 3))
    if not np.array_equal(all_on, any_on):
        return None
    return all_on


def _match_placeholder(
    patch: np.ndarray,
    color: int,
    variants: list[dict[str, object]],
    bg: int,
) -> dict[str, object] | None:
    mask = patch != bg
    hh, ww = map(int, mask.shape)
    hits: list[dict[str, object]] = []
    for variant in variants:
        vmask = np.asarray(variant["mask"], dtype=bool)
        vh, vw = map(int, vmask.shape)
        if hh % vh != 0 or ww % vw != 0:
            continue
        k_h = hh // vh
        k_w = ww // vw
        if k_h != k_w or k_h < 2:
            continue
        reduced = _downsample_mask(mask, int(k_h))
        if reduced is None:
            continue
        if not np.array_equal(reduced, vmask):
            continue
        transformed = _scale_patch(np.asarray(variant["patch"], dtype=int), int(k_h))
        transformed[transformed == int(color)] = int(bg)
        hits.append(
            {
                "scale": int(k_h),
                "color": int(color),
                "variant": variant,
                "output_patch": transformed,
            }
        )
    if not hits:
        return None
    assert len(hits) == 1, "placeholder match must be unique"
    return hits[0]


def _analyze_scene(grid: np.ndarray) -> tuple[int, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    comps = _find_foreground_components(g, bg)
    assert comps, "expected at least one foreground component"

    template_candidates = [comp for comp in comps if len(comp["colors"]) >= 2]
    assert len(template_candidates) == 1, "expected exactly one multicolor template"
    template = template_candidates[0]
    template_patch = np.asarray(template["patch"], dtype=int)
    template_mask = template_patch != bg
    assert len(set(map(int, template_patch[template_mask].tolist()))) >= 2
    variants = _dihedral_variants(template_patch, bg)
    assert len(variants) >= 2, "template mask must admit at least two distinct poses"

    matches: list[dict[str, object]] = []
    distractors: list[dict[str, object]] = []
    for comp in comps:
        if comp is template:
            continue
        colors = comp["colors"]
        assert len(colors) == 1, "non-template objects must be monochrome"
        match = _match_placeholder(np.asarray(comp["patch"], dtype=int), int(colors[0]), variants, bg)
        if match is None:
            distractors.append(comp)
            continue
        record = dict(comp)
        record.update(match)
        matches.append(record)

    assert matches, "expected at least one valid placeholder"
    assert len(distractors) <= 1, "expected at most one distractor"
    return bg, template, matches, distractors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the multicolor template onto each scaled placeholder pose while carving out template cells that match the placeholder color.

    Input structure:
    - A black background contains one small multicolor template sprite.
    - Separate large monochrome placeholder objects are block-scaled copies of the template mask under rotation or reflection.
    - There may also be one monochrome distractor whose size is an integer multiple of the template but whose downsampled mask does not match.

    Rule:
    - Find the unique multicolor component and use its foreground mask as the template mask.
    - For each monochrome object whose block-downsampled mask matches some rotated or reflected template mask, infer its scale factor and pose.
    - Replace that object by the correspondingly transformed colored template scaled into k×k blocks, but turn every scaled template cell whose color equals the placeholder's own color into background.
    - Erase the original small template and leave any non-matching distractor unchanged.

    Key insight:
    - The placeholder color is not merely filler: it is the deletion key that punches holes out of the transferred template, so matching the mask alone is insufficient without also applying the color-based carve-out.
    """
    g = np.asarray(grid, dtype=int)
    bg, template, matches, _distractors = _analyze_scene(g)
    out = g.copy()

    for r, c in np.asarray(template["coords"], dtype=int):
        out[int(r), int(c)] = bg

    for match in matches:
        r0, c0, r1, c1 = map(int, match["bbox"])
        patch = np.asarray(match["output_patch"], dtype=int)
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _enumerate_masks(height: int, width: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_masks: list[np.ndarray] = []
    asym_masks: list[np.ndarray] = []
    for bits in range(1, 1 << (height * width)):
        mask = np.array([(bits >> i) & 1 for i in range(height * width)], dtype=bool).reshape(height, width)
        area = int(mask.sum())
        if area < MIN_TEMPLATE_AREA or area > min(MAX_TEMPLATE_AREA, height * width):
            continue
        if _bbox(mask) != (0, 0, height - 1, width - 1):
            continue
        if len(_components(mask)) != 1:
            continue
        all_masks.append(mask.copy())
        if len(_dihedral_variants(mask.astype(int), BG)) == 8:
            asym_masks.append(mask.copy())
    return all_masks, asym_masks


MASK_LIBRARY: dict[tuple[int, int], list[np.ndarray]] = {}
ASYM_MASK_LIBRARY: dict[tuple[int, int], list[np.ndarray]] = {}
for _dims in {dims for base in TEMPLATE_DIMS for dims in (base, (base[1], base[0]))}:
    _all, _asym = _enumerate_masks(*_dims)
    MASK_LIBRARY[_dims] = _all
    ASYM_MASK_LIBRARY[_dims] = _asym


def _color_template(mask: np.ndarray, rng: random.Random, n_colors: int) -> tuple[np.ndarray, list[int]]:
    coords = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    rng.shuffle(coords)
    palette = rng.sample(PALETTE, n_colors)
    patch = np.zeros(mask.shape, dtype=int)
    for idx, (r, c) in enumerate(coords):
        if idx < n_colors:
            patch[r, c] = int(palette[idx])
        else:
            patch[r, c] = int(rng.choice(palette))
    return patch, [int(color) for color in palette]


def _solid_patch_from_mask(mask: np.ndarray, color: int, k: int) -> np.ndarray:
    small = np.where(mask, int(color), BG).astype(int)
    return _scale_patch(small, int(k))


def _place_patches(
    rng: random.Random,
    patches: list[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    heights = [int(p.shape[0]) for p in patches]
    widths = [int(p.shape[1]) for p in patches]
    min_h = max(max(heights) + 2, 18)
    min_w = max(max(widths) + 2, 18)

    for _ in range(120):
        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        blocked = np.zeros((h, w), dtype=bool)
        positions: list[tuple[int, int]] = [(-1, -1)] * len(patches)
        order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = map(int, patch.shape)
            candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            rng.shuffle(candidates)
            chosen: tuple[int, int] | None = None
            for r0, c0 in candidates:
                if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                    continue
                chosen = (int(r0), int(c0))
                break
            if chosen is None:
                ok = False
                break
            r0, c0 = chosen
            positions[idx] = chosen
            blocked[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
        if ok:
            return np.zeros((h, w), dtype=int), positions
    raise RuntimeError("failed to place non-overlapping objects")


def _build_expected_output(
    shape: tuple[int, int],
    template_patch: np.ndarray,
    template_pos: tuple[int, int],
    placeholders: list[dict[str, object]],
    distractor: dict[str, object] | None,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    if distractor is not None:
        dr, dc = map(int, distractor["pos"])
        dpatch = np.asarray(distractor["patch"], dtype=int)
        out[dr : dr + dpatch.shape[0], dc : dc + dpatch.shape[1]] = dpatch

    for spec in placeholders:
        r0, c0 = map(int, spec["pos"])
        patch = np.asarray(spec["output_patch"], dtype=int)
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    tr, tc = map(int, template_pos)
    template_mask = template_patch != BG
    out[tr : tr + template_patch.shape[0], tc : tc + template_patch.shape[1]][template_mask] = BG
    return out


def _semantic_validation(
    input_grid: np.ndarray,
    expected_output: np.ndarray,
    template_patch: np.ndarray,
    placeholders: list[dict[str, object]],
    distractor: dict[str, object] | None,
) -> bool:
    try:
        actual = main(input_grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected_output):
        return False

    try:
        _bg, _template, matches, distractors = _analyze_scene(input_grid)
    except AssertionError:
        return False

    if len(matches) != len(placeholders):
        return False
    if len(distractors) != (1 if distractor is not None else 0):
        return False

    identity = _scale_patch(template_patch, 1)
    if identity.shape != template_patch.shape:
        return False

    patch_keys = set()
    seen_scales = {int(spec["scale"]) for spec in placeholders}
    if seen_scales != {2, 3}:
        return False
    if not any(bool(spec["variant"]["mirrored"]) for spec in placeholders):
        return False
    if not any(int(spec["variant"]["rot"]) != 0 for spec in placeholders):
        return False

    for spec in placeholders:
        scaled_source = _scale_patch(np.asarray(spec["variant"]["patch"], dtype=int), int(spec["scale"]))
        carved = np.asarray(spec["output_patch"], dtype=int)
        if not np.any((scaled_source == int(spec["color"])) & (carved == BG)):
            return False
        key = (tuple(map(int, carved.shape)), carved.tobytes())
        if key in patch_keys:
            return False
        patch_keys.add(key)

    if distractor is not None:
        dr, dc = map(int, distractor["pos"])
        dpatch = np.asarray(distractor["patch"], dtype=int)
        if not np.array_equal(actual[dr : dr + dpatch.shape[0], dc : dc + dpatch.shape[1]], dpatch):
            return False

    return True


def _generate_case(rng: random.Random) -> np.ndarray | None:
    dims = rng.choice(TEMPLATE_DIMS)
    mask = rng.choice(ASYM_MASK_LIBRARY[dims]).copy()
    area = int(mask.sum())

    n_placeholders = rng.randint(2, min(4, area, 5))
    n_colors = rng.randint(max(3, n_placeholders), min(5, area))
    template_patch, palette = _color_template(mask, rng, n_colors)

    variants = _dihedral_variants(template_patch, BG)
    identity_variant = next(v for v in variants if int(v["rot"]) == 0 and not bool(v["mirrored"]))
    mirrored_pool = [v for v in variants if bool(v["mirrored"])]
    rotated_pool = [v for v in variants if int(v["rot"]) != 0 and not bool(v["mirrored"])]
    if not mirrored_pool or not rotated_pool:
        return None

    used_shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    chosen_variants = [rng.choice(mirrored_pool), rng.choice(rotated_pool)]
    used_shape_keys.update(tuple(v["shape_key"]) for v in chosen_variants)
    remaining = [v for v in variants if tuple(v["shape_key"]) not in used_shape_keys]
    rng.shuffle(remaining)
    while len(chosen_variants) < n_placeholders and remaining:
        chosen_variants.append(remaining.pop())
    while len(chosen_variants) < n_placeholders:
        extra = rng.choice([v for v in variants if v is not identity_variant])
        chosen_variants.append(extra)

    delete_colors = rng.sample(palette, n_placeholders)
    scales = [2, 3]
    while len(scales) < n_placeholders:
        scales.append(rng.choice((2, 3)))
    tail = scales[2:]
    rng.shuffle(tail)
    scales[2:] = tail

    placeholders: list[dict[str, object]] = []
    for variant, color, scale in zip(chosen_variants, delete_colors, scales):
        solid_patch = _solid_patch_from_mask(np.asarray(variant["mask"], dtype=bool), int(color), int(scale))
        colored = _scale_patch(np.asarray(variant["patch"], dtype=int), int(scale))
        colored[colored == int(color)] = BG
        placeholders.append(
            {
                "variant": variant,
                "color": int(color),
                "scale": int(scale),
                "input_patch": solid_patch,
                "output_patch": colored,
            }
        )

    distractor: dict[str, object] | None = None
    if rng.random() < 0.75:
        target_variant = rng.choice(variants)
        dh, dw = map(int, np.asarray(target_variant["mask"], dtype=bool).shape)
        template_shape_keys = {tuple(v["shape_key"]) for v in _dihedral_variants(mask.astype(int), BG)}
        candidates = [
            fake.copy()
            for fake in MASK_LIBRARY[(dh, dw)]
            if all(tuple(v["shape_key"]) not in template_shape_keys for v in _dihedral_variants(fake.astype(int), BG))
        ]
        if not candidates:
            return None
        fake_mask = rng.choice(candidates)
        fake_scale = rng.choice((2, 3))
        fake_color = int(rng.choice(palette + [rng.choice(PALETTE)]))
        distractor = {
            "patch": _solid_patch_from_mask(fake_mask, fake_color, fake_scale),
        }

    patches = [template_patch] + [np.asarray(spec["input_patch"], dtype=int) for spec in placeholders]
    if distractor is not None:
        patches.append(np.asarray(distractor["patch"], dtype=int))

    try:
        grid, positions = _place_patches(rng, patches)
    except RuntimeError:
        return None

    template_pos = positions[0]
    for patch, (r0, c0) in zip(patches, positions):
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    for spec, pos in zip(placeholders, positions[1 : 1 + len(placeholders)]):
        spec["pos"] = pos
    if distractor is not None:
        distractor["pos"] = positions[-1]

    expected = _build_expected_output(grid.shape, template_patch, template_pos, placeholders, distractor)
    if not _semantic_validation(grid, expected, template_patch, placeholders, distractor):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scaled-stamping placeholder input")
