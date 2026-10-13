from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, components_mask, crop_bbox, find_components, get_bg, paste, scale_sprite


BG = int(Color.BLACK)
PALETTE = [c for c in Color.NOT_BLACK]
GEN_RETRIES = 500
TEMPLATE_RETRIES = 300


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), tuple(arr.astype(int).ravel())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    box = bbox(arr)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    return crop_bbox(arr.astype(int), box).astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    arr = np.asarray(mask, dtype=bool)
    return bool(arr.any()) and len(components_mask(arr, connectivity=4)) == 1


def _rotation_orbit(mask: np.ndarray) -> list[np.ndarray]:
    return [_crop_mask(np.rot90(np.asarray(mask, dtype=bool), k)) for k in range(4)]


def _reflection_orbit(mask: np.ndarray) -> list[np.ndarray]:
    mirrored = np.fliplr(np.asarray(mask, dtype=bool))
    return _rotation_orbit(mirrored)


def _is_chiral_template(mask: np.ndarray) -> bool:
    rot_sigs = {_mask_signature(rot) for rot in _rotation_orbit(mask)}
    if len(rot_sigs) < 2:
        return False
    return all(_mask_signature(ref) not in rot_sigs for ref in _reflection_orbit(mask))


def _shape_in_orbit(mask: np.ndarray, orbit: list[np.ndarray]) -> bool:
    sig = _mask_signature(mask)
    return any(sig == _mask_signature(other) for other in orbit)


def _connected_growth_mask(h: int, w: int, n_cells: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < n_cells:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_template() -> np.ndarray:
    for _ in range(TEMPLATE_RETRIES):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_cells = min(h * w - 1, 6)
        min_cells = min(max(4, h + w - 1), max_cells)
        if min_cells > max_cells:
            continue
        n_cells = random.randint(min_cells, max_cells)
        mask = _crop_mask(_connected_growth_mask(h, w, n_cells))
        if min(mask.shape) < 2:
            continue
        if not _is_connected(mask):
            continue
        if not _is_chiral_template(mask):
            continue
        return mask
    raise RuntimeError("failed to create a chiral template")


def _scaled_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return scale_sprite(np.asarray(mask, dtype=int), int(scale)).astype(bool)


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _match_scaled_rotation(comp_mask: np.ndarray, template_mask: np.ndarray) -> tuple[int, int] | None:
    candidate = np.asarray(comp_mask, dtype=bool)
    for rot_idx, rotated in enumerate(_rotation_orbit(template_mask)):
        th, tw = rotated.shape
        h, w = candidate.shape
        if h % th != 0 or w % tw != 0:
            continue
        scale_h = h // th
        scale_w = w // tw
        if scale_h != scale_w or scale_h < 1:
            continue
        if np.array_equal(candidate, _scaled_mask(rotated, scale_h)):
            return int(scale_h), int(rot_idx)
    return None


def _match_scaled_reflection(comp_mask: np.ndarray, template_mask: np.ndarray) -> tuple[int, int] | None:
    candidate = np.asarray(comp_mask, dtype=bool)
    for rot_idx, reflected in enumerate(_reflection_orbit(template_mask)):
        th, tw = reflected.shape
        h, w = candidate.shape
        if h % th != 0 or w % tw != 0:
            continue
        scale_h = h // th
        scale_w = w // tw
        if scale_h != scale_w or scale_h < 1:
            continue
        if np.array_equal(candidate, _scaled_mask(reflected, scale_h)):
            return int(scale_h), int(rot_idx)
    return None


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            border_color = int(g[r, c])
            if border_color == int(bg):
                continue
            if not bool(np.all(g[r, : c + 1] == border_color)):
                continue
            if not bool(np.all(g[: r + 1, c] == border_color)):
                continue
            if c + 1 < w and int(g[r, c + 1]) == border_color:
                continue
            if r + 1 < h and int(g[r + 1, c]) == border_color:
                continue

            interior = g[:r, :c]
            if np.any(interior == border_color):
                continue
            inner_colors = set(map(int, np.unique(interior))) - {int(bg)}
            if len(inner_colors) != 1:
                continue
            template_color = inner_colors.pop()
            template_mask = interior == template_color
            if len(components_mask(template_mask, connectivity=4)) != 1:
                continue
            cropped = _crop_mask(template_mask)
            if min(cropped.shape) < 2 or not _is_connected(cropped):
                continue
            candidates.append((r, c, border_color, cropped))

    assert len(candidates) == 1, "expected exactly one top-left legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the components that are scaled-and-rotated copies of the legend miniature.

    Input structure:
    - The upper-left corner contains a legend region whose bottom edge and right edge form a single-color L-shaped border.
    - Inside that legend is one monochrome template shape on the background.
    - Elsewhere in the grid are several monochrome 4-connected components of different colors, sizes, and orientations.

    Rule:
    - Detect the legend, extract the template shape, and clear the entire legend rectangle including its border.
    - For every remaining component, ignore its color and test whether its cropped shape equals an integer pixel-scale copy of the template after a rotation by 0/90/180/270 degrees.
    - Recolor every matching component to the legend border color, and erase every non-matching component, including mirror-only decoys.

    Key insight:
    - The legend is a scale-independent shape key: only exact integer scale-ups of the template survive, and reflections must be rejected even if they look very similar.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legend_r, legend_c, border_color, template_mask = _find_legend(g, bg)

    out = g.copy()
    out[: legend_r + 1, : legend_c + 1] = bg

    work = g.copy()
    work[: legend_r + 1, : legend_c + 1] = bg
    comps = find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    for comp in comps:
        comp_mask = _component_mask(comp)
        target_color = border_color if _match_scaled_rotation(comp_mask, template_mask) is not None else bg
        for r, c in comp["coords"]:
            out[int(r), int(c)] = int(target_color)

    return out


def _random_color(*, prefer: list[int] | None = None, exclude: set[int] | None = None) -> int:
    exclude = set() if exclude is None else set(exclude)
    prefer = [] if prefer is None else [int(c) for c in prefer if int(c) not in exclude]
    if prefer and random.random() < 0.35:
        return int(random.choice(prefer))
    pool = [int(c) for c in PALETTE if int(c) not in exclude]
    return int(random.choice(pool))


def _random_near_miss(base_mask: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    base = np.asarray(base_mask, dtype=bool)
    rotations = _rotation_orbit(template_mask)
    reflections = _reflection_orbit(template_mask)
    occupied = [tuple(map(int, rc)) for rc in np.argwhere(base)]
    empty = [tuple(map(int, rc)) for rc in np.argwhere(~base)]
    random.shuffle(occupied)
    random.shuffle(empty)

    for rr, cc in occupied:
        removed = base.copy()
        removed[rr, cc] = False
        if not _is_connected(removed):
            continue
        add_candidates = []
        for ar, ac in empty:
            if removed[ar, ac]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < removed.shape[0] and 0 <= nc < removed.shape[1] and removed[nr, nc]:
                    add_candidates.append((ar, ac))
                    break
        random.shuffle(add_candidates)
        for ar, ac in add_candidates:
            trial = removed.copy()
            trial[ar, ac] = True
            if not (trial[0].any() and trial[-1].any() and trial[:, 0].any() and trial[:, -1].any()):
                continue
            if not _is_connected(trial):
                continue
            if _shape_in_orbit(trial, rotations) or _shape_in_orbit(trial, reflections):
                continue
            return trial
    raise RuntimeError("failed to create a near-miss decoy")


def _random_other_negative(template_mask: np.ndarray) -> np.ndarray:
    rotation_orbit = _rotation_orbit(template_mask)
    reflection_orbit = _reflection_orbit(template_mask)
    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_cells = min(h * w - 1, 6)
        min_cells = min(max(3, h + w - 2), max_cells)
        if min_cells > max_cells:
            continue
        n_cells = random.randint(min_cells, max_cells)
        mask = _crop_mask(_connected_growth_mask(h, w, n_cells))
        if min(mask.shape) < 2 or not _is_connected(mask):
            continue
        if _shape_in_orbit(mask, rotation_orbit) or _shape_in_orbit(mask, reflection_orbit):
            continue
        return mask
    raise RuntimeError("failed to create an unrelated negative")


def _random_place(blocked: np.ndarray, patch_shape: tuple[int, int], *, pad: int = 1) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = map(int, patch_shape)
    choices = []
    for r in range(0, h - ph + 1):
        for c in range(0, w - pw + 1):
            r0 = max(0, r - pad)
            c0 = max(0, c - pad)
            r1 = min(h, r + ph + pad)
            c1 = min(w, c + pw + pad)
            if not bool(blocked[r0:r1, c0:c1].any()):
                choices.append((r, c))
    if not choices:
        return None
    return random.choice(choices)


def _mark_blocked(blocked: np.ndarray, r: int, c: int, shape: tuple[int, int], *, pad: int = 1) -> None:
    ph, pw = map(int, shape)
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(blocked.shape[0], r + ph + pad)
    c1 = min(blocked.shape[1], c + pw + pad)
    blocked[r0:r1, c0:c1] = True


def _build_instance() -> np.ndarray:
    template_mask = _random_template()
    legend_h = template_mask.shape[0] + 1
    legend_w = template_mask.shape[1] + 1

    border_color, template_color = random.sample(PALETTE, 2)
    big_scale = random.choice([2, 3])
    big_rot = random.choice([1, 2, 3])
    small_rot = random.randint(0, 3)
    mirror_rot = random.randint(0, 3)
    near_scale = 1 if big_scale == 3 or random.random() < 0.6 else 2

    big_positive = _scaled_mask(np.rot90(template_mask, big_rot), big_scale)
    small_positive = _scaled_mask(np.rot90(template_mask, small_rot), 1)
    mirror_negative = _scaled_mask(np.rot90(np.fliplr(template_mask), mirror_rot), big_scale)
    near_negative = _scaled_mask(_random_near_miss(np.rot90(template_mask, random.randint(0, 3)), template_mask), near_scale)

    objects = [
        {"kind": "positive", "mask": small_positive},
        {"kind": "positive", "mask": big_positive},
        {"kind": "mirror", "mask": mirror_negative},
        {"kind": "near", "mask": near_negative},
    ]

    if random.random() < 0.55:
        extra_kind = random.choice(["positive", "mirror", "other"])
        extra_scale = random.choice([1, 2])
        extra_rot = random.randint(0, 3)
        if extra_kind == "positive":
            extra_mask = _scaled_mask(np.rot90(template_mask, extra_rot), extra_scale)
        elif extra_kind == "mirror":
            extra_mask = _scaled_mask(np.rot90(np.fliplr(template_mask), extra_rot), extra_scale)
        else:
            extra_mask = _scaled_mask(_random_other_negative(template_mask), extra_scale)
        objects.append({"kind": extra_kind, "mask": extra_mask})

    random.shuffle(objects)

    max_obj_h = max(obj["mask"].shape[0] for obj in objects)
    max_obj_w = max(obj["mask"].shape[1] for obj in objects)
    min_h = max(legend_h + 3, max_obj_h + 2, 22)
    min_w = max(legend_w + 3, max_obj_w + 2, 22)
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend = np.full((legend_h, legend_w), BG, dtype=int)
    legend[: template_mask.shape[0], : template_mask.shape[1]][template_mask] = template_color
    legend[-1, :] = border_color
    legend[:, -1] = border_color
    paste(grid, legend, 0, 0)
    _mark_blocked(blocked, 0, 0, legend.shape, pad=1)

    border_and_template_used = False
    for idx, obj in enumerate(objects):
        patch_shape = obj["mask"].shape
        loc = _random_place(blocked, patch_shape, pad=1)
        if loc is None:
            raise RuntimeError("packing failed")
        r, c = loc
        if not border_and_template_used and idx == len(objects) - 1:
            color = random.choice([border_color, template_color])
        elif obj["kind"] == "positive" and random.random() < 0.35:
            color = border_color
            border_and_template_used = True
        elif obj["kind"] != "positive" and random.random() < 0.35:
            color = template_color
            border_and_template_used = True
        else:
            color = _random_color(prefer=[border_color, template_color], exclude={BG})
            border_and_template_used = border_and_template_used or color in {border_color, template_color}
        paste(grid, _shape_patch(obj["mask"], color), r, c, transparent=BG)
        _mark_blocked(blocked, r, c, patch_shape, pad=1)

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legend_r, legend_c, border_color, template_mask = _find_legend(g, bg)
    assert _is_chiral_template(template_mask)

    scene = g.copy()
    scene[: legend_r + 1, : legend_c + 1] = bg
    comps = find_components(scene, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(comps) >= 4

    matched_scales = set()
    matched_rots = set()
    positives = 0
    negatives = 0
    mirror_negatives = 0

    out = main(g)
    assert np.all(out[: legend_r + 1, : legend_c + 1] == bg)

    for comp in comps:
        comp_mask = _component_mask(comp)
        match = _match_scaled_rotation(comp_mask, template_mask)
        reflection = _match_scaled_reflection(comp_mask, template_mask)
        values = {int(out[int(r), int(c)]) for r, c in comp["coords"]}
        if match is not None:
            positives += 1
            matched_scales.add(int(match[0]))
            matched_rots.add(int(match[1]))
            assert values == {border_color}
        else:
            negatives += 1
            if reflection is not None:
                mirror_negatives += 1
            assert values == {bg}

    assert positives >= 2
    assert negatives >= 2
    assert 1 in matched_scales
    assert any(scale > 1 for scale in matched_scales)
    assert any(rot != 0 for rot in matched_rots)
    assert mirror_negatives >= 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _build_instance()
            _validate_instance(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid instance")
