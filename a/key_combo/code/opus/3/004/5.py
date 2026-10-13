from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import (  # type: ignore
    bbox_coords,
    colors_present,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_sprite,
)

BG = 0
GLYPH = 2
PLACEHOLDER = 5
SOURCE_PALETTE = (1, 3, 4, 6, 7, 8, 9)
AREA_TO_OFFSET = {
    3: (0, 1),
    4: (2, 0),
    5: (0, -1),
    6: (-2, 0),
    7: (0, 3),
}
GLYPH_TEMPLATES: dict[int, list[np.ndarray]] = {
    3: [
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=int),
    ],
    4: [
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    ],
    5: [
        np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
    ],
    6: [
        np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=int),
    ],
    7: [
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=int),
    ],
}
AREA_PLANS = (
    {"areas": (3, 4, 7), "force_overlap": False, "add_distractor": False},
    {"areas": (5, 6, 7), "force_overlap": True, "add_distractor": False},
    {"areas": (3, 4, 5, 6), "force_overlap": False, "add_distractor": True},
    {"areas": (3, 5), "force_overlap": False, "add_distractor": False},
    {"areas": (4, 6, 7), "force_overlap": True, "add_distractor": True},
)
MAX_GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _normalized_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _component_signature(comp: dict) -> tuple[tuple[int, int], bytes]:
    return _mask_signature(_normalized_mask(comp["coords"]))


def _absolute_mask_coords(mask: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    rel = np.argwhere(mask)
    return rel + np.array(top_left)


def _cells_key(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _glyph_region(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, _c1 = map(int, bbox)
    return r1 + 2, c0, r1 + 4, c0 + 2


def _glyph_info(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int | None, np.ndarray]:
    gr0, gc0, gr1, gc1 = _glyph_region(bbox)
    if gr1 >= grid.shape[0] or gc1 >= grid.shape[1]:
        return None, np.zeros((0, 2), dtype=int)
    region = grid[gr0 : gr1 + 1, gc0 : gc1 + 1]
    if region.shape != (3, 3) or not np.all(np.isin(region, [BG, GLYPH])):
        return None, np.zeros((0, 2), dtype=int)
    rel = np.argwhere(region == GLYPH)
    if rel.size == 0:
        return None, np.zeros((0, 2), dtype=int)
    return int(rel.shape[0]), rel + np.array([gr0, gc0])


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _has_vertical_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _has_horizontal_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)))


def _generate_source_patch() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mono = random_sprite(
            h,
            w,
            density=random.uniform(0.45, 0.7),
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=4,
            background=BG,
        )
        mask = mono != BG
        area = int(mask.sum())
        if not (6 <= area <= 12):
            continue
        if mask.all() or mask.shape[1] < 3:
            continue
        if _has_vertical_symmetry(mask) and _has_horizontal_symmetry(mask):
            continue
        colors = random.sample(SOURCE_PALETTE, random.randint(2, min(4, area)))
        patch = np.zeros(mask.shape, dtype=int)
        coords = np.argwhere(mask).tolist()
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            patch[int(r), int(c)] = colors[idx % len(colors)]
        if len(set(map(int, patch[mask]))) < 2:
            continue
        return patch, mask
    raise RuntimeError("failed to generate a valid source sprite")


def _generate_distractor_mask(shape: tuple[int, int], source_sig: tuple[tuple[int, int], bytes]) -> np.ndarray:
    h, w = map(int, shape)
    for _ in range(300):
        mono = random_sprite(
            h,
            w,
            density=random.uniform(0.45, 0.75),
            symmetry="not_symmetric",
            color_palette=[PLACEHOLDER],
            connectivity=4,
            background=BG,
        )
        mask = mono == PLACEHOLDER
        if int(mask.sum()) < 5 or mask.all():
            continue
        sig = _mask_signature(mask)
        if sig == source_sig:
            continue
        return mask
    raise RuntimeError("failed to generate distractor mask")


def _glyph_patch(area: int) -> np.ndarray:
    template = random.choice(GLYPH_TEMPLATES[int(area)])
    return template.astype(int) * GLYPH


def _target_anchor(anchor: tuple[int, int], area: int) -> tuple[int, int]:
    dr, dc = AREA_TO_OFFSET[int(area)]
    return int(anchor[0] + dr), int(anchor[1] + dc)


def _target_bbox(anchor: tuple[int, int], area: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    tr, tc = _target_anchor(anchor, area)
    sh, sw = map(int, shape)
    return tr, tc, tr + sh - 1, tc + sw - 1


def _sample_placeholder_layout(
    grid_shape: tuple[int, int],
    source_mask: np.ndarray,
    areas: tuple[int, ...],
    force_overlap: bool,
) -> list[dict] | None:
    H, W = map(int, grid_shape)
    sh, sw = map(int, source_mask.shape)
    panel_h = sh + 4
    panel_w = max(sw, 3)
    rel_mask_coords = np.argwhere(source_mask)

    by_area: dict[int, list[tuple[int, int]]] = {}
    for area in areas:
        dr, dc = AREA_TO_OFFSET[int(area)]
        cands: list[tuple[int, int]] = []
        for r in range(0, H - panel_h + 1):
            tr = r + dr
            if not (0 <= tr <= H - sh):
                continue
            for c in range(0, W - panel_w + 1):
                tc = c + dc
                if 0 <= tc <= W - sw:
                    cands.append((int(r), int(c)))
        random.shuffle(cands)
        by_area[int(area)] = cands

    for _ in range(250):
        chosen: list[dict] = []
        used_rects: list[tuple[int, int, int, int]] = []
        ok = True
        for area in random.sample(list(areas), len(areas)):
            anchor = None
            for r, c in by_area[int(area)]:
                rect = (r, c, r + panel_h - 1, c + panel_w - 1)
                if any(_rects_overlap(rect, other, gap=1) for other in used_rects):
                    continue
                anchor = (r, c)
                used_rects.append(rect)
                chosen.append(
                    {
                        "area": int(area),
                        "anchor": anchor,
                        "panel_rect": rect,
                        "target_bbox": _target_bbox(anchor, area, source_mask.shape),
                    }
                )
                break
            if anchor is None:
                ok = False
                break
        if not ok:
            continue

        target_sets = []
        for item in chosen:
            tr, tc = _target_anchor(item["anchor"], item["area"])
            coords = rel_mask_coords + np.array([tr, tc])
            target_sets.append(_cells_key(coords))
        has_overlap = any(target_sets[i] & target_sets[j] for i in range(len(target_sets)) for j in range(i + 1, len(target_sets)))
        if has_overlap != bool(force_overlap):
            continue
        return chosen
    return None


def _sample_source_anchor(
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    reserved_rects: list[tuple[int, int, int, int]],
    blocked_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    H, W = map(int, grid_shape)
    sh, sw = map(int, sprite_shape)
    cands: list[tuple[int, int]] = []
    for r in range(0, H - sh + 1):
        for c in range(0, W - sw + 1):
            rect = (r, c, r + sh - 1, c + sw - 1)
            if any(_rects_overlap(rect, other, gap=1) for other in reserved_rects):
                continue
            if any(_rects_overlap(rect, other, gap=0) for other in blocked_bboxes):
                continue
            cands.append((int(r), int(c)))
    return random.choice(cands) if cands else None


def _sample_distractor_anchor(
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    reserved_rects: list[tuple[int, int, int, int]],
    blocked_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    H, W = map(int, grid_shape)
    sh, sw = map(int, sprite_shape)
    panel_h = sh + 4
    panel_w = max(sw, 3)
    cands: list[tuple[int, int]] = []
    for r in range(0, H - panel_h + 1):
        for c in range(0, W - panel_w + 1):
            rect = (r, c, r + panel_h - 1, c + panel_w - 1)
            if any(_rects_overlap(rect, other, gap=1) for other in reserved_rects):
                continue
            if any(_rects_overlap(rect, other, gap=0) for other in blocked_bboxes):
                continue
            cands.append((int(r), int(c)))
    return random.choice(cands) if cands else None


def _find_source_component(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    candidates = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        cols = colors_present(crop_bbox(grid, comp["bbox"]), exclude={bg})
        if len(cols) >= 2:
            candidates.append(comp)
    assert len(candidates) == 1, "expected exactly one multicolor source component"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each placeholder's glyph area and stamp the source sprite at that shifted anchor.

    Input structure:
    - Black background, one 4-connected multicolor source sprite, and 2-4 gray 4-connected placeholder sprites.
    - Every true placeholder has the same normalized mask as the source and a red 3x3 glyph one blank row below its bbox.
    - Additional gray distractors may exist, but only mask-matching gray components count as placeholders.

    Rule:
    - Find the unique multicolor source component and keep both its normalized mask and its per-cell colors.
    - Among gray components, keep only those whose normalized mask matches the source; read the red-pixel count in the 3x3 glyph below each one.
    - Decode glyph area 3/4/5/6/7 into offsets right1/down2/left1/up2/right3, then stamp the source colors at placeholder top-left plus that offset.
    - Process placeholders in bbox top-left order so later stamps overwrite earlier ones, erase the original source, erase matched gray placeholders, and erase all detected glyph pixels.

    Key insight:
    - The gray shapes are only anchor templates; the actual destination is recovered from the attached glyph area, while exact mask matching separates true placeholders from gray lookalikes.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    source = _find_source_component(g)
    source_bbox = tuple(map(int, source["bbox"]))
    source_patch = crop_bbox(g, source_bbox)
    source_mask = _normalized_mask(source["coords"])
    source_sig = _mask_signature(source_mask)

    placeholder_infos: list[dict] = []
    glyph_cells_to_clear: set[tuple[int, int]] = set()
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[PLACEHOLDER], return_format="dict"):
        area, glyph_coords = _glyph_info(g, tuple(map(int, comp["bbox"])))
        glyph_cells_to_clear |= _cells_key(glyph_coords)
        if area not in AREA_TO_OFFSET:
            continue
        if _component_signature(comp) != source_sig:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        placeholder_infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": comp["coords"],
                "target": _target_anchor((r0, c0), int(area)),
            }
        )

    assert 2 <= len(placeholder_infos) <= 4, "expected 2-4 true placeholders"

    out = g.copy()
    out[source["coords"][:, 0], source["coords"][:, 1]] = bg
    for info in placeholder_infos:
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for r, c in glyph_cells_to_clear:
        out[int(r), int(c)] = bg

    for info in sorted(placeholder_infos, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        tr, tc = info["target"]
        paste(out, source_patch, int(tr), int(tc), mask=source_mask)

    return out


def _render_from_record(record: dict) -> np.ndarray:
    out = np.array(record["grid"], copy=True)
    out[record["source_coords"][:, 0], record["source_coords"][:, 1]] = BG
    for info in record["placeholders"]:
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for r, c in record["glyph_cells"]:
        out[int(r), int(c)] = BG
    for info in sorted(record["placeholders"], key=lambda item: (item["bbox"][0], item["bbox"][1])):
        tr, tc = info["target_anchor"]
        paste(out, record["source_patch"], int(tr), int(tc), mask=record["source_mask"])
    return out


def _validate_record(record: dict) -> None:
    inp = record["grid"]
    expected = _render_from_record(record)
    assert not np.array_equal(inp, expected), "output must differ from input"

    source = _find_source_component(inp)
    found_mask = _normalized_mask(source["coords"])
    assert _mask_signature(found_mask) == _mask_signature(record["source_mask"])
    assert len(colors_present(crop_bbox(inp, source["bbox"]), exclude={BG})) >= 2

    gray_comps = find_components(inp, bg=BG, connectivity=4, monochromatic=True, colors=[PLACEHOLDER], return_format="dict")
    matched = [comp for comp in gray_comps if _component_signature(comp) == _mask_signature(record["source_mask"])]
    assert len(matched) == len(record["placeholders"])
    if record["distractor"] is not None:
        assert len(gray_comps) == len(record["placeholders"]) + 1

    areas = []
    for info in record["placeholders"]:
        area, _glyph_coords = _glyph_info(inp, info["bbox"])
        assert area == info["area"]
        areas.append(int(area))
    assert len(set(areas)) == len(areas)

    target_sets = [_cells_key(info["target_coords"]) for info in record["placeholders"]]
    has_overlap = any(target_sets[i] & target_sets[j] for i in range(len(target_sets)) for j in range(i + 1, len(target_sets)))
    assert has_overlap == bool(record["force_overlap"])

    solved = main(inp)
    assert np.array_equal(solved, expected)


def _build_case(plan: dict) -> dict | None:
    source_patch, source_mask = _generate_source_patch()
    sh, sw = map(int, source_mask.shape)
    areas = tuple(int(a) for a in plan["areas"])
    panel_h = sh + 4
    panel_w = max(sw, 3)

    min_h = max(18, panel_h * 2 + sh + 2)
    min_w = max(18, panel_w * 3 + sw + 2)
    H = random.randint(min_h, min(29, min_h + 3))
    W = random.randint(min_w, min(29, min_w + 4))

    placeholders = _sample_placeholder_layout((H, W), source_mask, areas, bool(plan["force_overlap"]))
    if placeholders is None:
        return None

    target_bboxes = [item["target_bbox"] for item in placeholders]
    source_anchor = _sample_source_anchor(
        (H, W),
        (sh, sw),
        [item["panel_rect"] for item in placeholders],
        target_bboxes,
    )
    if source_anchor is None:
        return None

    source_sig = _mask_signature(source_mask)
    distractor = None
    reserved_rects = [item["panel_rect"] for item in placeholders]
    reserved_rects.append((source_anchor[0], source_anchor[1], source_anchor[0] + sh - 1, source_anchor[1] + sw - 1))
    if bool(plan["add_distractor"]):
        unused_areas = [a for a in AREA_TO_OFFSET if a not in areas]
        distractor_area = random.choice(unused_areas if unused_areas else list(AREA_TO_OFFSET))
        distractor_mask = _generate_distractor_mask((sh, sw), source_sig)
        distractor_anchor = _sample_distractor_anchor((H, W), (sh, sw), reserved_rects, target_bboxes)
        if distractor_anchor is None:
            return None
        distractor = {
            "anchor": distractor_anchor,
            "area": int(distractor_area),
            "mask": distractor_mask,
            "panel_rect": (
                distractor_anchor[0],
                distractor_anchor[1],
                distractor_anchor[0] + panel_h - 1,
                distractor_anchor[1] + panel_w - 1,
            ),
        }
        reserved_rects.append(distractor["panel_rect"])

    grid = np.zeros((H, W), dtype=int)
    paste(grid, source_patch, int(source_anchor[0]), int(source_anchor[1]), mask=source_mask)
    source_coords = _absolute_mask_coords(source_mask, source_anchor)

    placeholders_record: list[dict] = []
    glyph_cells: set[tuple[int, int]] = set()
    gray_patch = np.full(source_mask.shape, PLACEHOLDER, dtype=int)
    for item in placeholders:
        anchor = item["anchor"]
        r0, c0 = map(int, anchor)
        bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        paste(grid, gray_patch, r0, c0, mask=source_mask)
        glyph = _glyph_patch(item["area"])
        paste(grid, glyph, r0 + sh + 1, c0, transparent=BG)
        gray_coords = _absolute_mask_coords(source_mask, anchor)
        area, gcoords = _glyph_info(grid, bbox)
        if area != item["area"]:
            return None
        target_anchor = _target_anchor(anchor, item["area"])
        placeholders_record.append(
            {
                "area": int(item["area"]),
                "bbox": bbox,
                "coords": gray_coords,
                "glyph_coords": gcoords,
                "target_anchor": target_anchor,
                "target_coords": _absolute_mask_coords(source_mask, target_anchor),
            }
        )
        glyph_cells |= _cells_key(gcoords)

    if distractor is not None:
        r0, c0 = map(int, distractor["anchor"])
        glyph = _glyph_patch(int(distractor["area"]))
        gray_like = np.full(distractor["mask"].shape, PLACEHOLDER, dtype=int)
        paste(grid, gray_like, r0, c0, mask=distractor["mask"])
        paste(grid, glyph, r0 + sh + 1, c0, transparent=BG)
        bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        _area, gcoords = _glyph_info(grid, bbox)
        glyph_cells |= _cells_key(gcoords)
        distractor["bbox"] = bbox
        distractor["coords"] = _absolute_mask_coords(distractor["mask"], distractor["anchor"])

    record = {
        "grid": grid,
        "source_patch": source_patch,
        "source_mask": source_mask,
        "source_coords": source_coords,
        "placeholders": placeholders_record,
        "glyph_cells": glyph_cells,
        "force_overlap": bool(plan["force_overlap"]),
        "distractor": distractor,
    }

    _validate_record(record)
    return record


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(MAX_GEN_ATTEMPTS):
        base_plan = AREA_PLANS[_GEN_CALL_COUNT % len(AREA_PLANS)]
        plan = {
            "areas": tuple(base_plan["areas"]),
            "force_overlap": bool(base_plan["force_overlap"]),
            "add_distractor": bool(base_plan["add_distractor"]),
        }
        record = _build_case(plan)
        if record is None:
            continue
        out = _render_from_record(record)
        sig = (tuple(map(int, out.shape)), out.tobytes())
        if sig in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(sig)
        _GEN_CALL_COUNT += 1
        return record["grid"]
    raise RuntimeError(f"failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
