from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BG = int(Color.BLACK)
GREEN = int(Color.GREEN)
COPY_COLORS = [int(c) for c in Color.NOT_BLACK if int(c) != GREEN]
SIDE_LR = ("left", "right")
SIDE_TB = ("top", "bottom")
ALL_SIDES = SIDE_LR + SIDE_TB

GRID_MIN = 24
GRID_MAX = 30
FRAME_MIN = 3
FRAME_MAX = 6
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 5
GAP_VALUES = (1, 2, 3)
MIN_CLIPPED_VISIBLE_CELLS = 4
MIN_CLIPPED_VISIBLE_FRAC = 0.45
GEN_ATTEMPTS = 500

_RELATION_BAG: list[bool] = []
_CLIP_BAG: list[bool] = []
_GAP_BAG: list[int] = []
_COUNT_BAG: list[int] = []


def _next_from_bag(bag: list, values: list):
    if not bag:
        bag.extend(values)
        random.shuffle(bag)
    return bag.pop()


def _next_relation() -> bool:
    return bool(_next_from_bag(_RELATION_BAG, [False, True]))


def _next_clip_mode() -> bool:
    return bool(_next_from_bag(_CLIP_BAG, [False, True, True]))


def _next_gap() -> int:
    return int(_next_from_bag(_GAP_BAG, list(GAP_VALUES)))


def _next_object_count() -> int:
    return int(_next_from_bag(_COUNT_BAG, list(range(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1))))


def _tab_side(mask_or_patch: np.ndarray) -> str:
    mask = np.asarray(mask_or_patch, dtype=bool)
    h, w = map(int, mask.shape)
    boundary_counts = {
        "top": int(mask[0, :].sum()),
        "bottom": int(mask[h - 1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, w - 1].sum()),
    }
    sides = [side for side, count in boundary_counts.items() if count == 1]
    assert len(sides) == 1, "expected exactly one one-pixel tab side"
    return sides[0]


def _mirror_mask(mask: np.ndarray, side: str) -> np.ndarray:
    return np.fliplr(mask) if side in SIDE_LR else np.flipud(mask)


def _copy_top_left(
    bbox: tuple[int, int, int, int],
    patch_shape: tuple[int, int],
    side: str,
    gap: int,
    same_side: bool,
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = map(int, patch_shape)
    gap = int(gap)
    if side == "left":
        return r0, c0 - gap - w if same_side else c1 + gap + 1
    if side == "right":
        return r0, c1 + gap + 1 if same_side else c0 - gap - w
    if side == "top":
        return r0 - gap - h if same_side else r1 + gap + 1, c0
    if side == "bottom":
        return r1 + gap + 1 if same_side else r0 - gap - h, c0
    raise ValueError(f"invalid side {side}")


def _match_exemplar(
    src_mask: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    copy_mask: np.ndarray,
    copy_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, bool] | None:
    if tuple(map(int, src_mask.shape)) != tuple(map(int, copy_mask.shape)):
        return None
    if not np.array_equal(copy_mask, _mirror_mask(src_mask, side)):
        return None

    r0, c0, r1, c1 = map(int, src_bbox)
    rr0, cc0, rr1, cc1 = map(int, copy_bbox)
    if side in SIDE_LR:
        if rr0 != r0 or rr1 != r1:
            return None
        if cc1 < c0:
            gap = c0 - cc1 - 1
            copy_side = "left"
        elif cc0 > c1:
            gap = cc0 - c1 - 1
            copy_side = "right"
        else:
            return None
        same_side = (side == "left" and copy_side == "left") or (side == "right" and copy_side == "right")
    else:
        if cc0 != c0 or cc1 != c1:
            return None
        if rr1 < r0:
            gap = r0 - rr1 - 1
            copy_side = "top"
        elif rr0 > r1:
            gap = rr0 - r1 - 1
            copy_side = "bottom"
        else:
            return None
        same_side = (side == "top" and copy_side == "top") or (side == "bottom" and copy_side == "bottom")

    if gap < 1:
        return None
    return int(gap), bool(same_side)


def _infer_exemplar(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    green_components = find_components(
        g,
        bg=bg,
        colors=[GREEN],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(green_components) >= 2, "expected multiple green outlines"

    all_components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    colored_components = [comp for comp in all_components if int(comp["color"]) not in {bg, GREEN}]
    assert len(colored_components) == 1, "expected exactly one pre-drawn colored mirror"
    copy_component = colored_components[0]
    copy_color = int(copy_component["color"])
    copy_bbox = tuple(map(int, copy_component["bbox"]))
    copy_patch = crop_bbox(g, copy_bbox)
    copy_mask = copy_patch == copy_color

    matches = []
    for component in green_components:
        bbox = tuple(map(int, component["bbox"]))
        patch = crop_bbox(g, bbox)
        mask = patch == GREEN
        side = _tab_side(mask)
        matched = _match_exemplar(mask, bbox, copy_mask, copy_bbox, side)
        if matched is not None:
            gap, same_side = matched
            matches.append(
                {
                    "bbox": bbox,
                    "gap": int(gap),
                    "same_side": bool(same_side),
                    "side": side,
                }
            )

    assert len(matches) == 1, "expected exactly one green object to explain the exemplar mirror"
    match = matches[0]
    return {
        "bg": bg,
        "copy_color": copy_color,
        "gap": int(match["gap"]),
        "same_side": bool(match["same_side"]),
        "exemplar_bbox": tuple(match["bbox"]),
        "green_components": green_components,
    }


def _stamp_mask(
    canvas: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
    *,
    overwrite: bool,
    clip: bool,
) -> None:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    paste(canvas, patch, int(top), int(left), transparent=BG, overwrite=overwrite, clip=clip)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each green tabbed outline using the color, gap, and side relation demonstrated by the lone completed exemplar pair.

    Input structure
    - The background is black.
    - Several separated green hollow outlines appear, and each outline's tight bbox has exactly one side with a single-pixel tab.
    - Exactly one of those outlines already has a non-green mirrored copy nearby, separated by a puzzle-specific gap and placed either on the tab side or the opposite side.

    Rule
    - Use the tab side to choose the mirror axis: left/right tabs mean a horizontal mirror, top/bottom tabs mean a vertical mirror.
    - Read three parameters from the unique completed exemplar pair: the copy color, the bbox-to-bbox gap, and whether the copy sits on the tab side or the opposite side.
    - For every other green outline, add its mirrored copy with those same three parameters, clipping off-grid parts and painting only background cells.

    Key insight
    - The puzzle does not fix the copy color or spacing in advance; the only colored mirror already present inside the grid is the instruction sheet that determines how every remaining outline must be completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    inferred = _infer_exemplar(g)
    copy_color = int(inferred["copy_color"])
    gap = int(inferred["gap"])
    same_side = bool(inferred["same_side"])
    exemplar_bbox = tuple(inferred["exemplar_bbox"])

    out = g.copy()
    green_components = sorted(
        inferred["green_components"],
        key=lambda comp: tuple(map(int, comp["bbox"])),
    )
    for component in green_components:
        bbox = tuple(map(int, component["bbox"]))
        if bbox == exemplar_bbox:
            continue
        patch = crop_bbox(g, bbox)
        mask = patch == GREEN
        side = _tab_side(mask)
        reflected = _mirror_mask(mask, side)
        top, left = _copy_top_left(bbox, mask.shape, side, gap, same_side)
        _stamp_mask(out, reflected, top, left, copy_color, overwrite=False, clip=True)

    return out


def _make_outline_mask(side: str, frame_h: int, frame_w: int, tab_offset: int) -> np.ndarray:
    extra_top = int(side == "top")
    extra_bottom = int(side == "bottom")
    extra_left = int(side == "left")
    extra_right = int(side == "right")

    mask = np.zeros((frame_h + extra_top + extra_bottom, frame_w + extra_left + extra_right), dtype=bool)
    top = extra_top
    left = extra_left
    bottom = top + frame_h - 1
    right = left + frame_w - 1

    mask[top, left : right + 1] = True
    mask[bottom, left : right + 1] = True
    mask[top : bottom + 1, left] = True
    mask[top : bottom + 1, right] = True

    if side == "left":
        mask[top + tab_offset, 0] = True
    elif side == "right":
        mask[top + tab_offset, mask.shape[1] - 1] = True
    elif side == "top":
        mask[0, left + tab_offset] = True
    else:
        mask[mask.shape[0] - 1, left + tab_offset] = True

    assert _tab_side(mask) == side
    return mask


def _sample_specs(n_objects: int) -> list[dict]:
    base_sides = [
        random.choice(SIDE_LR),
        random.choice(SIDE_TB),
    ]
    while len(base_sides) < n_objects:
        base_sides.append(random.choice(ALL_SIDES))
    random.shuffle(base_sides)

    specs = []
    seen_signatures = set()
    for side in base_sides:
        for _ in range(100):
            frame_h = random.randint(FRAME_MIN, FRAME_MAX)
            frame_w = random.randint(FRAME_MIN, FRAME_MAX)
            tab_limit = frame_h if side in SIDE_LR else frame_w
            tab_offset = random.randrange(tab_limit)
            mask = _make_outline_mask(side, frame_h, frame_w, tab_offset)
            signature = (tuple(map(int, mask.shape)), tuple(int(v) for v in mask.ravel()))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            specs.append(
                {
                    "side": side,
                    "mask": mask,
                    "mirror_mask": _mirror_mask(mask, side),
                    "area": int(mask.sum()),
                }
            )
            break
        else:
            raise RuntimeError("failed to sample a unique outline shape")

    return specs


def _visible_cells(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for rr, cc in np.argwhere(mask):
        r = int(top) + int(rr)
        c = int(left) + int(cc)
        if 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
    return cells


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _candidate_positions(
    spec: dict,
    shape: tuple[int, int],
    gap: int,
    same_side: bool,
    requirement: str,
) -> list[dict]:
    h, w = map(int, shape)
    mask = spec["mask"]
    mirror_mask = spec["mirror_mask"]
    patch_h, patch_w = map(int, mask.shape)
    area = int(mask.sum())

    candidates = []
    for top in range(h - patch_h + 1):
        for left in range(w - patch_w + 1):
            bbox = (top, left, top + patch_h - 1, left + patch_w - 1)
            copy_top, copy_left = _copy_top_left(bbox, mask.shape, spec["side"], gap, same_side)
            full_copy = 0 <= copy_top <= h - patch_h and 0 <= copy_left <= w - patch_w
            if requirement == "full" and not full_copy:
                continue
            if requirement == "clipped" and full_copy:
                continue

            orig_cells = _visible_cells(mask, top, left, shape)
            copy_cells = _visible_cells(mirror_mask, copy_top, copy_left, shape)
            if not copy_cells:
                continue
            if requirement == "clipped":
                min_visible = max(MIN_CLIPPED_VISIBLE_CELLS, int(np.ceil(area * MIN_CLIPPED_VISIBLE_FRAC)))
                if len(copy_cells) < min_visible:
                    continue

            candidates.append(
                {
                    "bbox": bbox,
                    "top": int(top),
                    "left": int(left),
                    "copy_top": int(copy_top),
                    "copy_left": int(copy_left),
                    "orig_cells": orig_cells,
                    "copy_cells": copy_cells,
                    "final_buffer": _dilate_cells(orig_cells | copy_cells, shape),
                    "orig_buffer": _dilate_cells(orig_cells, shape),
                    "input_pair_buffer": _dilate_cells(orig_cells | copy_cells, shape),
                }
            )

    random.shuffle(candidates)
    return candidates


def _place_specs(
    specs: list[dict],
    shape: tuple[int, int],
    gap: int,
    same_side: bool,
    exemplar_idx: int,
    clip_idx: int | None,
) -> list[dict] | None:
    requirements = []
    for idx in range(len(specs)):
        if idx == exemplar_idx:
            requirements.append("full")
        elif idx == clip_idx:
            requirements.append("clipped")
        else:
            requirements.append("full")

    candidates_by_idx = []
    for spec, requirement in zip(specs, requirements):
        candidates = _candidate_positions(spec, shape, gap, same_side, requirement)
        if not candidates:
            return None
        candidates_by_idx.append(candidates)

    order = sorted(
        range(len(specs)),
        key=lambda idx: (
            0 if idx == exemplar_idx else 1 if idx == clip_idx else 2,
            len(candidates_by_idx[idx]),
            -specs[idx]["area"],
        ),
    )

    placements: list[dict | None] = [None] * len(specs)
    final_occ: set[tuple[int, int]] = set()
    input_occ: set[tuple[int, int]] = set()

    for idx in order:
        placed = False
        for candidate in candidates_by_idx[idx]:
            if candidate["final_buffer"] & final_occ:
                continue
            input_buffer = candidate["input_pair_buffer"] if idx == exemplar_idx else candidate["orig_buffer"]
            if input_buffer & input_occ:
                continue

            placements[idx] = candidate
            final_added = candidate["orig_cells"] | candidate["copy_cells"]
            input_added = candidate["orig_cells"] | (candidate["copy_cells"] if idx == exemplar_idx else set())
            final_occ.update(final_added)
            input_occ.update(input_added)
            placed = True
            break
        if not placed:
            return None

    return [placement for placement in placements if placement is not None]


def _render_scene(
    specs: list[dict],
    placements: list[dict],
    shape: tuple[int, int],
    exemplar_idx: int,
    copy_color: int,
    gap: int,
    same_side: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for idx, (spec, placement) in enumerate(zip(specs, placements)):
        _stamp_mask(inp, spec["mask"], placement["top"], placement["left"], GREEN, overwrite=True, clip=False)
        _stamp_mask(out, spec["mask"], placement["top"], placement["left"], GREEN, overwrite=True, clip=False)

        if idx == exemplar_idx:
            _stamp_mask(
                inp,
                spec["mirror_mask"],
                placement["copy_top"],
                placement["copy_left"],
                copy_color,
                overwrite=True,
                clip=False,
            )
            _stamp_mask(
                out,
                spec["mirror_mask"],
                placement["copy_top"],
                placement["copy_left"],
                copy_color,
                overwrite=True,
                clip=False,
            )

        if idx != exemplar_idx:
            _stamp_mask(
                out,
                spec["mirror_mask"],
                placement["copy_top"],
                placement["copy_left"],
                copy_color,
                overwrite=False,
                clip=True,
            )

    assert np.array_equal(main(inp), out)

    recovered = _infer_exemplar(inp)
    exemplar_bbox = tuple(placements[exemplar_idx]["bbox"])
    assert int(recovered["copy_color"]) == int(copy_color)
    assert int(recovered["gap"]) == int(gap)
    assert bool(recovered["same_side"]) == bool(same_side)
    assert tuple(recovered["exemplar_bbox"]) == exemplar_bbox
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_objects = _next_object_count()
    gap = _next_gap()
    same_side = _next_relation()
    clip_mode = _next_clip_mode()
    copy_color = random.choice(COPY_COLORS)

    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    specs = _sample_specs(n_objects)
    exemplar_idx = random.randrange(n_objects)
    clip_candidates = [idx for idx in range(n_objects) if idx != exemplar_idx]
    clip_idx = random.choice(clip_candidates) if clip_mode and clip_candidates else None

    placements = _place_specs(specs, (grid_h, grid_w), gap, same_side, exemplar_idx, clip_idx)
    if placements is None:
        return None

    inp, _out = _render_scene(specs, placements, (grid_h, grid_w), exemplar_idx, copy_color, gap, same_side)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-exemplar input")
