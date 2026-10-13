from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, is_contiguous, paste, random_free_location


Variant = tuple[str, np.ndarray]
Placement = tuple[str, int, int]

GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 28
TEMPLATE_H_RANGE = (4, 7)
TEMPLATE_W_RANGE = (4, 7)
TEMPLATE_AREA_RANGE = (9, 16)
COPY_COUNT_CHOICES = (3, 4)
PADDING = 2


def _mask_coords(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _coords_to_patch(coords: set[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _variant_masks(mask: np.ndarray) -> list[Variant]:
    candidates = [
        ("identity", mask.copy()),
        ("flip_lr", np.fliplr(mask)),
        ("flip_ud", np.flipud(mask)),
    ]
    unique: list[Variant] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for name, item in candidates:
        key = tuple(sorted(_mask_coords(item)))
        if key not in seen:
            seen.add(key)
            unique.append((name, item))
    return unique


def _normalized_patch(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _find_embeddings(piece_mask: np.ndarray, variants: list[Variant]) -> list[Placement]:
    piece_coords = _mask_coords(piece_mask)
    first_piece = next(iter(piece_coords))
    matches: set[Placement] = set()
    for name, variant in variants:
        variant_coords = _mask_coords(variant)
        for vr, vc in variant_coords:
            dr = int(vr - first_piece[0])
            dc = int(vc - first_piece[1])
            shifted = {(r + dr, c + dc) for r, c in piece_coords}
            if shifted <= variant_coords:
                matches.add((name, dr, dc))
    return sorted(matches)


def _recover_component(
    comp: dict,
    variants: list[Variant],
) -> tuple[Placement, set[tuple[int, int]]]:
    r0, c0, _r1, _c1 = map(int, comp["bbox"])
    sprite = np.zeros((comp["bbox"][2] - comp["bbox"][0] + 1, comp["bbox"][3] - comp["bbox"][1] + 1), dtype=bool)
    for r, c in comp["coords"]:
        sprite[int(r) - r0, int(c) - c0] = True
    placements = _find_embeddings(sprite, variants)
    assert len(placements) == 1, "damaged copy must have a unique mirror+translation embedding"
    name, dr, dc = placements[0]
    variant_mask = dict(variants)[name]
    full_abs = {
        (r0 - dr + vr, c0 - dc + vc)
        for vr, vc in _mask_coords(variant_mask)
    }
    comp_abs = {tuple(map(int, rc)) for rc in comp["coords"]}
    return placements[0], full_abs - comp_abs


def _scene_analysis(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert len(comps) >= 2, "expected one complete mold plus damaged copies"

    candidates = []
    for idx, template in enumerate(comps):
        tr0, tc0, tr1, tc1 = map(int, template["bbox"])
        template_mask = np.zeros((tr1 - tr0 + 1, tc1 - tc0 + 1), dtype=bool)
        for r, c in template["coords"]:
            template_mask[int(r) - tr0, int(c) - tc0] = True
        variants = _variant_masks(template_mask)

        placements: list[tuple[int, Placement]] = []
        ok = True
        for j, comp in enumerate(comps):
            if j == idx:
                continue
            if int(comp["area"]) >= int(template["area"]):
                ok = False
                break
            r0, c0, r1, c1 = map(int, comp["bbox"])
            piece = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp["coords"]:
                piece[int(r) - r0, int(c) - c0] = True
            matches = _find_embeddings(piece, variants)
            if len(matches) != 1:
                ok = False
                break
            placements.append((j, matches[0]))
        if ok:
            candidates.append((idx, variants, placements))

    assert len(candidates) == 1, "mold must be uniquely identifiable"
    return bg, comps, candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each damaged monochrome copy by uniquely embedding it into the complete mold or one of its mirrors.

    Input structure:
    - The background contains several monochrome 8-connected objects in distinct colors.
    - Exactly one object is the complete mold; every other object is a damaged copy of that mold after either no mirror, a left-right mirror, or an up-down mirror, plus translation.
    - Each damaged copy has exactly one mirror+translation placement that makes all of its visible cells lie inside a mold variant.

    Rule:
    - Identify the mold as the unique object whose mirror variants can uniquely contain every other object while those others are all strictly smaller.
    - For each damaged copy, enumerate the mold's allowed mirror variants and find its unique embedding by translation.
    - Fill every missing cell of that embedded variant with the damaged copy's own color, leaving the mold and the background unchanged.

    Key insight:
    - The copy's orientation and offset are not read directly from its bbox; they are forced by the only way its remaining pixels can sit inside a mirrored mold silhouette.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, comps, (template_idx, variants, placements) = _scene_analysis(g)

    by_index = {idx: placement for idx, placement in placements}
    for idx, comp in enumerate(comps):
        if idx == template_idx:
            continue
        placement, missing = _recover_component(comp, variants)
        assert placement == by_index[idx]
        out[tuple(r for r, _ in missing), tuple(c for _, c in missing)] = int(comp["color"])
    return out


def _random_template_mask() -> np.ndarray | None:
    h = random.randint(*TEMPLATE_H_RANGE)
    w = random.randint(*TEMPLATE_W_RANGE)
    area = random.randint(max(TEMPLATE_AREA_RANGE[0], h + w - 1), min(TEMPLATE_AREA_RANGE[1], h * w - 1))

    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    if not is_contiguous(mask, background=False, connectivity=8):
        return None
    if not _touches_all_sides(mask):
        return None
    if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
        return None
    if len(_variant_masks(mask)) != 3:
        return None
    return mask


def _sample_damaged_piece(
    target_name: str,
    target_mask: np.ndarray,
    all_variants: list[Variant],
    require_shift: bool,
) -> tuple[np.ndarray, Placement] | None:
    coords = list(_mask_coords(target_mask))
    max_remove = min(5, len(coords) - 4)
    if max_remove < 1:
        return None

    for _ in range(200):
        remove_count = random.randint(1, max_remove)
        removed = set(random.sample(coords, remove_count))
        remain_coords = set(coords) - removed
        piece = _coords_to_patch(remain_coords, 1, 0).astype(bool)
        if not is_contiguous(piece, background=False, connectivity=8):
            continue
        piece = _normalized_patch(piece)
        placements = _find_embeddings(piece, all_variants)
        if len(placements) != 1:
            continue
        placement = placements[0]
        if placement[0] != target_name:
            continue
        if require_shift and (placement[1], placement[2]) == (0, 0):
            continue
        return piece, placement
    return None


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randrange(10)
    copy_count = random.choice(COPY_COUNT_CHOICES)
    colors = random.sample([c for c in range(10) if c != bg], copy_count + 1)
    template_color = colors[0]

    template_mask = _random_template_mask()
    if template_mask is None:
        return None
    variants = _variant_masks(template_mask)
    variant_by_name = {name: mask for name, mask in variants}

    chosen_names = ["identity", "flip_lr", "flip_ud"]
    if copy_count == 4:
        chosen_names.append(random.choice(chosen_names))
    random.shuffle(chosen_names)

    copy_specs = []
    saw_shift = False
    for idx, name in enumerate(chosen_names):
        piece = _sample_damaged_piece(
            name,
            variant_by_name[name],
            variants,
            require_shift=(idx == 0),
        )
        if piece is None:
            return None
        piece_mask, placement = piece
        saw_shift |= (placement[1], placement[2]) != (0, 0)
        copy_specs.append((name, piece_mask, placement, colors[idx + 1]))
    if not saw_shift:
        return None

    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)

    objects = [("template", template_mask, template_mask, ("identity", 0, 0), template_color)]
    for name, piece_mask, placement, color in copy_specs:
        full_mask = variant_by_name[name]
        objects.append((name, piece_mask, full_mask, placement, color))
    random.shuffle(objects)

    for _name, piece_mask, full_mask, placement, color in objects:
        full_patch = np.where(full_mask, color, bg)
        try:
            r0, c0 = random_free_location(
                out,
                full_patch,
                transparent=bg,
                border=1,
                padding=PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(out, full_patch, r0, c0, transparent=bg)
        piece_patch = np.where(piece_mask, color, bg)
        paste(inp, piece_patch, r0 + placement[1], c0 + placement[2], transparent=bg)

    if np.array_equal(inp, out):
        return None

    try:
        _, comps, (template_idx, variants_found, placements) = _scene_analysis(inp)
    except AssertionError:
        return None
    if len(comps) != copy_count + 1:
        return None
    recovered_names = {placement[0] for _, placement in placements}
    if recovered_names != {"identity", "flip_lr", "flip_ud"}:
        return None
    if not any((placement[1], placement[2]) != (0, 0) for _, placement in placements):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid mirror-repair scene")
