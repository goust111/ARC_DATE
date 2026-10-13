from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_opus_13_031_4", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()

bbox = _ARC_UTILS.bbox
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, int(r0), int(c0)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    h, w = mask.shape
    if h % k != 0 or w % k != 0:
        return None
    out = np.zeros((h // k, w // k), dtype=bool)
    for r in range(h // k):
        for c in range(w // k):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if block.all():
                out[r, c] = True
            elif block.any():
                return None
    return out


def _scale_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((k, k), dtype=bool))


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c] and mask[r - 1 : r + 2, c - 1 : c + 2].all():
                interior[r, c] = True
    return interior


def _apply_transform(arr: np.ndarray, rot: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(arr, int(rot))
    if flip_lr:
        out = np.fliplr(out)
    return out


def _variant_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _dihedral_variants(
    patch: np.ndarray, mask: np.ndarray
) -> list[tuple[tuple[int, bool], np.ndarray, np.ndarray]]:
    out: list[tuple[tuple[int, bool], np.ndarray, np.ndarray]] = []
    for rot in range(4):
        for flip_lr in (False, True):
            out.append(
                (
                    (int(rot), bool(flip_lr)),
                    _apply_transform(patch, rot, flip_lr),
                    _apply_transform(mask, rot, flip_lr),
                )
            )
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = [_variant_key(v) for _, _, v in _dihedral_variants(mask.astype(int), mask)]
    return min(keys)


def _orientation_count(mask: np.ndarray) -> int:
    return len({_variant_key(v) for _, _, v in _dihedral_variants(mask.astype(int), mask)})


def _extract_layout_and_slots(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, list[dict], dict, list[dict]]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict"
    )
    assert comps, "expected at least one foreground component"
    frame = max(comps, key=lambda comp: int(comp["area"]))
    frame_vals = [int(grid[int(r), int(c)]) for r, c in frame["coords"]]
    frame_color = int(Counter(frame_vals).most_common(1)[0][0])

    sub = crop_bbox(grid, frame["bbox"])
    hole_mask = sub == bg
    hole_box = bbox(hole_mask)
    assert hole_box is not None, "frame must enclose at least one hole"
    r0, c0, r1, c1 = map(int, hole_box)
    layout = hole_mask[r0 : r1 + 1, c0 : c1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        mask, sr0, sc0 = _tight_mask(coords)
        slots.append(
            {
                "r0": int(sr0),
                "c0": int(sc0),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "expected at least one hole slot"
    return frame_color, layout, slots, frame, comps


def _extract_pieces(grid: np.ndarray, bg: int, frame: dict, comps: list[dict]) -> list[dict]:
    pieces: list[dict] = []
    for comp in comps:
        if comp is frame:
            continue
        coords = comp["coords"]
        if int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(grid, (pr0, pc0, pr1, pc1))
        mask = np.zeros(patch.shape, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        vals = patch[mask]
        counts = Counter(map(int, vals))
        base_color = int(counts.most_common(1)[0][0])
        marker_colors = [col for col in counts if col != base_color]
        assert len(marker_colors) == 1, "each piece must have one marker color"
        marker_color = int(marker_colors[0])
        assert counts[marker_color] == 1, "marker color must appear once"
        pieces.append(
            {
                "patch": patch,
                "mask": mask,
                "area": int(mask.sum()),
                "base": base_color,
                "marker": marker_color,
            }
        )
    assert pieces, "expected outside pieces"
    return pieces


def _infer_scale(pieces: list[dict]) -> int:
    valid = []
    for k in (2, 3):
        if all(_downsample_blocks(piece["mask"], k) is not None for piece in pieces):
            valid.append(int(k))
    assert len(valid) == 1, "scale factor must be uniquely inferable"
    return int(valid[0])


def _match_slots_to_pieces(slots: list[dict], pieces: list[dict], k: int) -> list[dict]:
    slot_options: list[list[dict]] = []
    for slot in slots:
        options: list[dict] = []
        for piece_idx, piece in enumerate(pieces):
            for transform, patch_v, mask_v in _dihedral_variants(piece["patch"], piece["mask"]):
                shape_v = _downsample_blocks(mask_v, k)
                if shape_v is None:
                    continue
                if shape_v.shape != slot["mask"].shape:
                    continue
                if np.array_equal(shape_v, slot["mask"]):
                    options.append(
                        {
                            "piece_idx": int(piece_idx),
                            "transform": transform,
                            "patch": patch_v,
                            "mask": mask_v,
                        }
                    )
        assert options, "every slot must have a matching piece"
        slot_options.append(options)

    order = sorted(range(len(slots)), key=lambda idx: len(slot_options[idx]))
    solutions: list[list[dict | None]] = []

    def dfs(depth: int, used: set[int], chosen: list[dict | None]) -> None:
        if len(solutions) > 1:
            return
        if depth == len(order):
            solutions.append(chosen.copy())
            return
        slot_idx = order[depth]
        for option in slot_options[slot_idx]:
            piece_idx = int(option["piece_idx"])
            if piece_idx in used:
                continue
            used.add(piece_idx)
            chosen[slot_idx] = option
            dfs(depth + 1, used, chosen)
            chosen[slot_idx] = None
            used.remove(piece_idx)

    dfs(0, set(), [None] * len(slots))
    assert len(solutions) == 1, "piece-to-slot matching must be unique"
    return [opt for opt in solutions[0] if opt is not None]


def _render_output(
    frame_color: int,
    layout: np.ndarray,
    slots: list[dict],
    pieces: list[dict],
    placements: list[dict],
    k: int,
    *,
    recolor: bool,
) -> np.ndarray:
    out = np.full(
        ((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), int(frame_color), dtype=int
    )
    for slot, placement in zip(slots, placements):
        patch = placement["patch"].copy()
        mask = placement["mask"]
        marker_color = int(pieces[int(placement["piece_idx"])]["marker"])
        if recolor:
            patch[_interior_mask(mask)] = marker_color
        paste(out, patch, (int(slot["r0"]) + 1) * k, (int(slot["c0"]) + 1) * k, mask=mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each enlarged outside piece to its frame hole, then recolor the placed piece's solid interior by its marker.

    Input structure:
    - The background is black, and the largest 4-connected foreground component is a monochrome frame enclosing 2-3 black hole slots.
    - Outside the frame sit 2-3 multicolor pieces whose binary masks are k-times scaled versions of the hole shapes under rotation/flip.
    - Each piece has one singleton marker pixel in a unique non-base color.

    Rule:
    - Extract the hole slots from inside the frame and the outside pieces as multicolor 4-connected components.
    - Infer k by finding the unique block size that lets every piece mask downsample cleanly into whole kxk cells.
    - Match each piece to exactly one hole by comparing the downsampled piece mask against every hole under all 8 rotation/flip orientations.
    - Output the scaled frame layout, place each matched piece into its hole, then recolor every placed pixel whose full 3x3 neighborhood stays inside that same piece to the piece's marker color.

    Key insight:
    - The shape match determines both which hole a piece occupies and which orientation its marker lands in; only after that placement is the 3x3-supported interior well defined for recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_color, layout, slots, frame, comps = _extract_layout_and_slots(g)
    pieces = _extract_pieces(g, bg, frame, comps)
    k = _infer_scale(pieces)
    placements = _match_slots_to_pieces(slots, pieces, k)
    return _render_output(frame_color, layout, slots, pieces, placements, k, recolor=True)


FRAME_COLOR = 5
BASE_COLORS = (1, 3, 4)
MARKER_COLORS = (2, 6, 7, 8, 9)
K_CHOICES = (2, 3)
SLOT_COUNT_CHOICES = (2, 3)
LAYOUT_SIZE_BY_K = {2: (6, 9), 3: (6, 8)}
SLOT_AREA_RANGE = (6, 15)
CANVAS_RANGE = (22, 30)
MAX_RETRIES = 400


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _grow_slot(forbidden: np.ndarray, target_area: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free = np.argwhere(~forbidden)
    if free.size == 0:
        return None
    random.shuffle(free)
    for start in free[:10]:
        seed = (int(start[0]), int(start[1]))
        chosen = {seed}
        frontier = set()
        frontier.update({(seed[0] - 1, seed[1]), (seed[0] + 1, seed[1]), (seed[0], seed[1] - 1), (seed[0], seed[1] + 1)})
        while len(chosen) < target_area and frontier:
            candidates = [
                (r, c)
                for r, c in frontier
                if 0 <= r < h and 0 <= c < w and not forbidden[r, c] and (r, c) not in chosen
            ]
            if not candidates:
                break
            nr, nc = random.choice(candidates)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_area:
            return np.array(sorted(chosen), dtype=int)
    return None


def _slot_is_valid(mask: np.ndarray, k: int, seen: set[tuple[tuple[int, int], bytes]]) -> bool:
    if not (SLOT_AREA_RANGE[0] <= int(mask.sum()) <= SLOT_AREA_RANGE[1]):
        return False
    if int(mask.sum()) == mask.shape[0] * mask.shape[1]:
        return False
    if _orientation_count(mask) != 8:
        return False
    max_bbox = 5 if k == 2 else 4
    if mask.shape[0] > max_bbox or mask.shape[1] > max_bbox:
        return False
    canon = _canonical_shape_key(mask)
    if canon in seen:
        return False
    scaled = _scale_mask(mask, k)
    interior = _interior_mask(scaled)
    if not interior.any():
        return False
    if int(interior.sum()) >= int(scaled.sum()):
        return False
    return True


def _has_counterexample_coverage(slots: list[dict]) -> bool:
    same_area = False
    same_bbox = False
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            if int(slots[i]["area"]) == int(slots[j]["area"]):
                same_area = True
            if tuple(slots[i]["mask"].shape) == tuple(slots[j]["mask"].shape):
                same_bbox = True
    return same_area or same_bbox


def _gen_layout(k: int) -> tuple[np.ndarray, list[dict]] | None:
    lo, hi = LAYOUT_SIZE_BY_K[k]
    layout_h = random.randint(lo, hi)
    layout_w = random.randint(lo, hi)
    n_slots = random.choice(SLOT_COUNT_CHOICES)

    forbidden = np.zeros((layout_h, layout_w), dtype=bool)
    full_mask = np.zeros((layout_h, layout_w), dtype=bool)
    seen: set[tuple[tuple[int, int], bytes]] = set()
    slots: list[dict] = []

    duplicate_area = random.randint(*SLOT_AREA_RANGE)
    planned_areas = []
    if n_slots >= 2 and random.random() < 0.8:
        planned_areas = [duplicate_area, duplicate_area]
    while len(planned_areas) < n_slots:
        planned_areas.append(random.randint(*SLOT_AREA_RANGE))
    random.shuffle(planned_areas)

    for target_area in planned_areas:
        placed = False
        for _ in range(80):
            coords = _grow_slot(forbidden, int(target_area))
            if coords is None:
                continue
            mask, r0, c0 = _tight_mask(coords)
            if not _slot_is_valid(mask, k, seen):
                continue
            full_mask[coords[:, 0], coords[:, 1]] = True
            forbid = np.zeros_like(forbidden)
            forbid[coords[:, 0], coords[:, 1]] = True
            forbidden |= _dilate8(forbid)
            slots.append({"mask": mask, "r0": int(r0), "c0": int(c0), "area": int(mask.sum())})
            seen.add(_canonical_shape_key(mask))
            placed = True
            break
        if not placed:
            return None

    union_box = bbox(full_mask)
    if union_box is None:
        return None
    ur0, uc0, ur1, uc1 = map(int, union_box)
    layout = full_mask[ur0 : ur1 + 1, uc0 : uc1 + 1]
    cropped_slots: list[dict] = []
    for slot in slots:
        cropped_slots.append(
            {
                "mask": slot["mask"],
                "r0": int(slot["r0"]) - ur0,
                "c0": int(slot["c0"]) - uc0,
                "area": int(slot["area"]),
            }
        )
    cropped_slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    if not _has_counterexample_coverage(cropped_slots):
        return None
    return layout, cropped_slots


def _piece_output_patch(slot_mask: np.ndarray, k: int, base_color: int, marker_color: int) -> np.ndarray:
    mask = _scale_mask(slot_mask, k)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(base_color)
    interior_coords = np.argwhere(_interior_mask(mask))
    assert interior_coords.size > 0
    mr, mc = map(int, random.choice(interior_coords.tolist()))
    patch[mr, mc] = int(marker_color)
    return patch


def _bbox_gap_ok(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_frame_and_pieces(frame_patch: np.ndarray, pieces: list[np.ndarray]) -> np.ndarray | None:
    fh, fw = frame_patch.shape
    dims_h = random.randint(max(CANVAS_RANGE[0], fh + 2), CANVAS_RANGE[1])
    dims_w = random.randint(max(CANVAS_RANGE[0], fw + 2), CANVAS_RANGE[1])
    canvas = np.zeros((dims_h, dims_w), dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    frame_positions = [
        (r, c)
        for r in range(dims_h - fh + 1)
        for c in range(dims_w - fw + 1)
    ]
    random.shuffle(frame_positions)
    if not frame_positions:
        return None
    fr, fc = frame_positions[0]
    paste(canvas, frame_patch, fr, fc)
    boxes.append((fr, fc, fr + fh - 1, fc + fw - 1))

    order = sorted(range(len(pieces)), key=lambda idx: pieces[idx].shape[0] * pieces[idx].shape[1], reverse=True)
    for idx in order:
        piece = pieces[idx]
        ph, pw = piece.shape
        candidates = []
        for r in range(dims_h - ph + 1):
            for c in range(dims_w - pw + 1):
                box = (r, c, r + ph - 1, c + pw - 1)
                if _bbox_gap_ok(box, boxes, gap=1):
                    candidates.append((r, c, box))
        if not candidates:
            return None
        r, c, box = random.choice(candidates)
        paste(canvas, piece, r, c, transparent=0)
        boxes.append(box)
    return canvas


def _build_expected(layout: np.ndarray, slots: list[dict], out_patches: list[np.ndarray], k: int) -> np.ndarray:
    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), FRAME_COLOR, dtype=int)
    for slot, patch in zip(slots, out_patches):
        mask = patch != 0
        recolored = patch.copy()
        marker_vals = [int(v) for v in np.unique(recolored[mask]) if int(v) not in BASE_COLORS]
        assert len(marker_vals) == 1
        recolored[_interior_mask(mask)] = int(marker_vals[0])
        paste(out, recolored, (int(slot["r0"]) + 1) * k, (int(slot["c0"]) + 1) * k, mask=mask)
    return out


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    layout: np.ndarray,
    slots: list[dict],
    out_patches: list[np.ndarray],
    k: int,
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if expected.shape != ((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k):
        return False
    if len(slots) not in (2, 3):
        return False
    if not _has_counterexample_coverage(slots):
        return False

    no_recolor = np.full_like(expected, FRAME_COLOR)
    for slot, patch in zip(slots, out_patches):
        paste(no_recolor, patch, (int(slot["r0"]) + 1) * k, (int(slot["c0"]) + 1) * k, mask=patch != 0)
        if not _interior_mask(patch != 0).any():
            return False
    if np.array_equal(no_recolor, expected):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    frame_color, solved_layout, solved_slots, frame, comps = _extract_layout_and_slots(inp)
    if frame_color != FRAME_COLOR or not np.array_equal(solved_layout, layout):
        return False
    pieces = _extract_pieces(inp, 0, frame, comps)
    try:
        inferred_k = _infer_scale(pieces)
        placements = _match_slots_to_pieces(solved_slots, pieces, inferred_k)
    except Exception:
        return False
    if inferred_k != k or len(placements) != len(slots):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_result = _gen_layout(k)
    if layout_result is None:
        return None
    layout, slots = layout_result

    base_colors = random.sample(BASE_COLORS, len(slots))
    marker_colors = random.sample(MARKER_COLORS, len(slots))
    out_patches: list[np.ndarray] = []
    in_patches: list[np.ndarray] = []
    transforms = [(rot, flip) for rot in range(4) for flip in (False, True)]
    chosen_transforms = [random.choice(transforms) for _ in slots]
    if all(rot == 0 and not flip for rot, flip in chosen_transforms):
        chosen_transforms[0] = random.choice(transforms[1:])

    for slot, base_color, marker_color, transform in zip(slots, base_colors, marker_colors, chosen_transforms):
        out_patch = _piece_output_patch(slot["mask"], k, base_color, marker_color)
        out_patches.append(out_patch)
        in_patches.append(_apply_transform(out_patch, transform[0], transform[1]))

    frame_patch = np.full((layout.shape[0] + 2, layout.shape[1] + 2), FRAME_COLOR, dtype=int)
    frame_patch[1:-1, 1:-1][layout] = 0
    inp = _place_frame_and_pieces(frame_patch, in_patches)
    if inp is None:
        return None

    expected = _build_expected(layout, slots, out_patches, k)
    if not _validate_generated(inp, expected, layout, slots, out_patches, k):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/13/031/4")
