from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


GREY = 5
SOURCE_BG = 0
TARGET_BG = 8
TARGET_PARTIAL = 2
USABLE_COLORS = (1, 3, 4, 6, 7, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(sorted(cells), dtype=int))
    out.sort(key=lambda x: _bbox(x)[:2])
    return out


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _outline_mask(silhouette: np.ndarray) -> np.ndarray:
    h, w = silhouette.shape
    outline = np.zeros_like(silhouette, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not silhouette[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not silhouette[nr, nc]:
                    outline[r, c] = True
                    break
    return outline


def _parse_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert grey_cols, "expected a full-height grey separator"
    runs: list[tuple[int, int]] = []
    start = prev = grey_cols[0]
    for c in grey_cols[1:]:
        if c == prev + 1:
            prev = c
        else:
            runs.append((start, prev))
            start = prev = c
    runs.append((start, prev))
    assert len(runs) == 1, "expected exactly one grey separator run"
    c0, c1 = runs[0]
    assert c0 > 0 and c1 < grid.shape[1] - 1, "separator must split the grid"
    return grid[:, :c0], grid[:, c1 + 1 :], c0


def _extract_source_sprites(source_panel: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for coords in _components(source_panel != SOURCE_BG):
        bbox = _bbox(coords)
        patch = _crop(source_panel, bbox)
        silhouette = patch != SOURCE_BG
        colors = sorted(int(x) for x in np.unique(patch[silhouette]))
        assert len(colors) == 2, "each source sprite must use exactly two visible colors"
        outline = _outline_mask(silhouette)
        boundary_colors = {int(patch[r, c]) for r, c in np.argwhere(outline)}
        assert len(boundary_colors) == 1, "outline must be monochrome"
        outline_color = next(iter(boundary_colors))
        fill_color = next(col for col in colors if col != outline_color)
        interior = silhouette & ~outline
        assert interior.any(), "source sprite must contain an interior fill"
        assert np.all(patch[interior] == fill_color), "interior fill must be monochrome"
        outline_coords = np.argwhere(outline)
        sprites.append(
            {
                "bbox": bbox,
                "patch": patch,
                "outline_sig": _normalize(outline_coords),
                "outline_size": outline.shape,
            }
        )
    assert 2 <= len(sprites) <= 3, "expected 2-3 source sprites"
    sigs = [sprite["outline_sig"] for sprite in sprites]
    assert len(set(sigs)) == len(sigs), "source outline shapes must be unique"
    return sprites


def _completion_candidates(coords: np.ndarray, target_shape: tuple[int, int]) -> list[dict]:
    r0, c0, r1, c1 = _bbox(coords)
    local = coords - np.array([r0, c0], dtype=int)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    panel_h, panel_w = target_shape

    candidates: list[dict] = []

    def add_candidate(kind: str, full_r0: int, full_c0: int, full_h: int, full_w: int, mapper) -> None:
        if full_r0 < 0 or full_c0 < 0 or full_r0 + full_h > panel_h or full_c0 + full_w > panel_w:
            return
        full = np.zeros((full_h, full_w), dtype=bool)
        for rr, cc in local:
            for nr, nc in mapper(int(rr), int(cc), h, w):
                full[nr, nc] = True
        full_coords = np.argwhere(full)
        candidates.append(
            {
                "kind": kind,
                "top": (int(full_r0), int(full_c0)),
                "mask": full,
                "sig": _normalize(full_coords),
                "occupied": {(int(full_r0 + rr), int(full_c0 + cc)) for rr, cc in full_coords},
            }
        )

    add_candidate(
        "keep_left",
        r0,
        c0,
        h,
        2 * w - 1,
        lambda rr, cc, _h, width: (
            ((rr, cc),) if cc == width - 1 else ((rr, cc), (rr, 2 * width - 2 - cc))
        ),
    )
    add_candidate(
        "keep_right",
        r0,
        c0 - (w - 1),
        h,
        2 * w - 1,
        lambda rr, cc, _h, width: (
            ((rr, width - 1 + cc),)
            if cc == 0
            else ((rr, width - 1 + cc), (rr, width - 1 - cc))
        ),
    )
    add_candidate(
        "keep_top",
        r0,
        c0,
        2 * h - 1,
        w,
        lambda rr, cc, height, _w: (
            ((rr, cc),) if rr == height - 1 else ((rr, cc), (2 * height - 2 - rr, cc))
        ),
    )
    add_candidate(
        "keep_bottom",
        r0 - (h - 1),
        c0,
        2 * h - 1,
        w,
        lambda rr, cc, height, _w: (
            ((height - 1 + rr, cc),)
            if rr == 0
            else ((height - 1 + rr, cc), (height - 1 - rr, cc))
        ),
    )
    return candidates


def _find_assignments(source_sprites: list[dict], target_components: list[np.ndarray], target_shape: tuple[int, int]) -> list[dict]:
    all_candidates: list[list[dict]] = []
    for target_idx, coords in enumerate(target_components):
        target_candidates: list[dict] = []
        for completion in _completion_candidates(coords, target_shape):
            for source_idx, sprite in enumerate(source_sprites):
                if completion["sig"] != sprite["outline_sig"]:
                    continue
                target_candidates.append(
                    {
                        "target_idx": target_idx,
                        "source_idx": source_idx,
                        "top": completion["top"],
                        "occupied": completion["occupied"],
                        "kind": completion["kind"],
                    }
                )
        assert target_candidates, "every target remnant must have at least one valid completion"
        all_candidates.append(target_candidates)

    order = sorted(range(len(target_components)), key=lambda idx: len(all_candidates[idx]))
    solutions: list[list[dict]] = []

    def search(pos: int, used_sources: set[int], used_cells: set[tuple[int, int]], chosen: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(list(chosen))
            return
        idx = order[pos]
        for candidate in all_candidates[idx]:
            if candidate["source_idx"] in used_sources:
                continue
            if candidate["occupied"] & used_cells:
                continue
            search(
                pos + 1,
                used_sources | {candidate["source_idx"]},
                used_cells | candidate["occupied"],
                chosen + [candidate],
            )

    search(0, set(), set(), [])
    assert len(solutions) == 1, "expected a unique source-target assignment"
    solution = sorted(solutions[0], key=lambda item: item["target_idx"])
    assert len(solution) == len(target_components)
    return solution


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each partial symmetric outline, match it to a source sprite, and stamp the matched sprite into the target panel.

    Input structure
    - A full-height grey divider splits the grid into a left black source panel and a right teal target panel.
    - The source panel contains 2-3 complete 4-connected two-color sprites; one color forms each sprite's outline and the other fills its interior.
    - The target panel contains 2-3 red 4-connected remnants, each showing only one side of some source sprite's symmetric outline.

    Rule
    - For every red target remnant, consider the four possible single-axis completions obtained by mirroring it across the left, right, top, or bottom edge of its own bbox.
    - Keep only the completion whose normalized full outline matches exactly one source sprite outline, with all target remnants assigned to different source sprites and their completed placements remaining non-overlapping.
    - Stamp the matched full source sprite at that completed bbox location on the teal panel, treating black source-background pixels as transparent; leave the source panel and grey divider unchanged.

    Key insight
    - The red target objects are not directly matchable sprites: they must first be symmetrically completed into full outlines, and only then can the corresponding full-color source stamp be identified and overlaid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    source_panel, target_panel, sep_col = _parse_panels(g)
    source_sprites = _extract_source_sprites(source_panel)
    target_components = _components(target_panel == TARGET_PARTIAL)
    assert len(target_components) == len(source_sprites), "source and target panels must contain the same number of objects"

    assignments = _find_assignments(source_sprites, target_components, target_panel.shape)
    out = g.copy()
    target_origin = sep_col + 1
    for assignment in assignments:
        sprite = source_sprites[assignment["source_idx"]]
        top_r, top_c = assignment["top"]
        patch = sprite["patch"]
        h, w = patch.shape
        for r in range(h):
            for c in range(w):
                color = int(patch[r, c])
                if color == SOURCE_BG:
                    continue
                out[top_r + r, target_origin + top_c + c] = color
    return out


def _profile_shape(axis: str) -> np.ndarray:
    for _ in range(200):
        if axis == "vertical":
            h = random.randint(5, 7)
            w = random.choice((5, 7))
            max_indent = (w - 3) // 2
            indents = [random.randint(0, max_indent)]
            for _row in range(1, h):
                prev = indents[-1]
                lo = max(0, prev - 1)
                hi = min(max_indent, prev + 1)
                indents.append(random.randint(lo, hi))
            if indents == indents[::-1]:
                continue
            if len(set(indents)) < 2:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for r, indent in enumerate(indents):
                mask[r, indent : w - indent] = True
        else:
            w = random.randint(5, 7)
            h = random.choice((5, 7))
            max_indent = (h - 3) // 2
            indents = [random.randint(0, max_indent)]
            for _col in range(1, w):
                prev = indents[-1]
                lo = max(0, prev - 1)
                hi = min(max_indent, prev + 1)
                indents.append(random.randint(lo, hi))
            if indents == indents[::-1]:
                continue
            if len(set(indents)) < 2:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for c, indent in enumerate(indents):
                mask[indent : h - indent, c] = True

        outline = _outline_mask(mask)
        interior = mask & ~outline
        if not interior.any():
            continue
        if len(_components(mask)) != 1 or len(_components(outline)) != 1:
            continue
        if axis == "vertical" and np.array_equal(mask, np.flipud(mask)):
            continue
        if axis == "horizontal" and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate a valid symmetric silhouette")


def _sprite_from_mask(mask: np.ndarray, outline_color: int, fill_color: int) -> dict:
    outline = _outline_mask(mask)
    patch = np.zeros(mask.shape, dtype=int)
    patch[outline] = outline_color
    patch[mask & ~outline] = fill_color
    return {
        "patch": patch,
        "mask": mask,
        "outline": outline,
        "outline_sig": _normalize(np.argwhere(outline)),
    }


def _remnant_mask(outline: np.ndarray, side: str) -> np.ndarray:
    h, w = outline.shape
    if side == "left":
        return outline[:, : w // 2 + 1]
    if side == "right":
        return outline[:, w // 2 :]
    if side == "top":
        return outline[: h // 2 + 1, :]
    if side == "bottom":
        return outline[h // 2 :, :]
    raise ValueError("unknown remnant side")


def _full_bbox_for_remnant(side: str, rem_h: int, rem_w: int, r0: int, c0: int) -> tuple[int, int, int, int]:
    if side == "left":
        return r0, c0, r0 + rem_h - 1, c0 + 2 * rem_w - 2
    if side == "right":
        return r0, c0 - (rem_w - 1), r0 + rem_h - 1, c0 + rem_w - 1
    if side == "top":
        return r0, c0, r0 + 2 * rem_h - 2, c0 + rem_w - 1
    if side == "bottom":
        return r0 - (rem_h - 1), c0, r0 + rem_h - 1, c0 + rem_w - 1
    raise ValueError("unknown remnant side")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _place_source_boxes(panel_h: int, panel_w: int, sprites: list[dict]) -> list[tuple[int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = []
    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        patch = sprites[idx]["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(300):
            r0 = random.randint(0, panel_h - h)
            c0 = random.randint(0, panel_w - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_overlap(box, other, padding=1) for other in boxes):
                continue
            boxes.append(box)
            positions.append((idx, r0, c0))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place source sprite")
    out = [(-1, -1)] * len(sprites)
    for idx, r0, c0 in positions:
        out[idx] = (r0, c0)
    return out


def _target_anchor_ranges(panel_h: int, panel_w: int, patch_h: int, patch_w: int, side: str) -> tuple[range, range]:
    if side == "left":
        max_c = min(panel_w - patch_w, patch_w // 2 - 1)
        return range(0, panel_h - patch_h + 1), range(0, max_c + 1)
    if side == "right":
        min_c = max(0, panel_w - patch_w - (patch_w // 2 - 1))
        return range(0, panel_h - patch_h + 1), range(min_c, panel_w - patch_w + 1)
    if side == "top":
        max_r = min(panel_h - patch_h, patch_h // 2 - 1)
        return range(0, max_r + 1), range(0, panel_w - patch_w + 1)
    if side == "bottom":
        min_r = max(0, panel_h - patch_h - (patch_h // 2 - 1))
        return range(min_r, panel_h - patch_h + 1), range(0, panel_w - patch_w + 1)
    raise ValueError("unknown side")


def _place_target_boxes(panel_h: int, panel_w: int, sprites: list[dict], sides: list[str]) -> list[tuple[int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = [(-1, -1)] * len(sprites)
    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        patch = sprites[idx]["patch"]
        h, w = patch.shape
        row_range, col_range = _target_anchor_ranges(panel_h, panel_w, h, w, sides[idx])
        choices = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.shuffle(choices)
        placed = False
        for r0, c0 in choices:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_overlap(box, other, padding=1) for other in boxes):
                continue
            boxes.append(box)
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place target sprite")
    return positions


def _render_source_panel(panel_h: int, panel_w: int, sprites: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w), dtype=int)
    for sprite, (r0, c0) in zip(sprites, positions):
        patch = sprite["patch"]
        h, w = patch.shape
        region = panel[r0 : r0 + h, c0 : c0 + w]
        assert not np.any((region != SOURCE_BG) & (patch != SOURCE_BG))
        region[patch != SOURCE_BG] = patch[patch != SOURCE_BG]
    return panel


def _render_target_panels(panel_h: int, panel_w: int, sprites: list[dict], positions: list[tuple[int, int]], sides: list[str]) -> tuple[np.ndarray, np.ndarray]:
    input_panel = np.full((panel_h, panel_w), TARGET_BG, dtype=int)
    output_panel = np.full((panel_h, panel_w), TARGET_BG, dtype=int)
    for sprite, (r0, c0), side in zip(sprites, positions, sides):
        patch = sprite["patch"]
        outline = sprite["outline"]
        h, w = patch.shape
        output_region = output_panel[r0 : r0 + h, c0 : c0 + w]
        output_region[patch != SOURCE_BG] = patch[patch != SOURCE_BG]

        rem = _remnant_mask(outline, side)
        rem_coords = np.argwhere(rem)
        assert len(_components(rem)) == 1, "partial remnant must stay connected"
        rr0, cc0, rr1, cc1 = _bbox(rem_coords)
        rem_crop = rem[rr0 : rr1 + 1, cc0 : cc1 + 1]
        global_r0 = r0 + rr0
        global_c0 = c0 + cc0
        full_box = _full_bbox_for_remnant(side, rem_crop.shape[0], rem_crop.shape[1], global_r0, global_c0)
        assert full_box == (r0, c0, r0 + h - 1, c0 + w - 1), "remnant bbox must recover the true full bbox"
        region = input_panel[global_r0 : global_r0 + rem_crop.shape[0], global_c0 : global_c0 + rem_crop.shape[1]]
        region[rem_crop] = TARGET_PARTIAL
    return input_panel, output_panel


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> None:
    source_panel, target_panel, _sep = _parse_panels(grid)
    source_sprites = _extract_source_sprites(source_panel)
    target_components = _components(target_panel == TARGET_PARTIAL)
    assignments = _find_assignments(source_sprites, target_components, target_panel.shape)
    assert len(assignments) == len(source_sprites)
    source_sigs = {sprite["outline_sig"] for sprite in source_sprites}
    for coords in target_components:
        assert _normalize(coords) not in source_sigs, "partial must not already equal a full source outline"
    solved = main(grid)
    assert np.array_equal(solved, expected), "solver must reproduce the intended output"


def generate_input() -> np.ndarray:
    for _ in range(400):
        count = random.choice((2, 3))
        panel_h = random.randint(17, 23)
        panel_w = random.randint(12, 14)

        axis_pool = ["vertical", "horizontal"]
        if count == 3:
            axis_pool.append(random.choice(("vertical", "horizontal")))
        random.shuffle(axis_pool)

        sides: list[str] = []
        sprites: list[dict] = []
        used_sigs: set[tuple[tuple[int, int], ...]] = set()
        try:
            color_pool = list(USABLE_COLORS)
            random.shuffle(color_pool)
            for idx in range(count):
                axis = axis_pool[idx]
                side = random.choice(("left", "right")) if axis == "vertical" else random.choice(("top", "bottom"))
                mask = _profile_shape(axis)
                outline_sig = _normalize(np.argwhere(_outline_mask(mask)))
                if outline_sig in used_sigs:
                    raise RuntimeError("duplicate outline shape")
                used_sigs.add(outline_sig)
                outline_color = color_pool.pop()
                fill_color = color_pool.pop()
                sprite = _sprite_from_mask(mask, outline_color, fill_color)
                rem = _remnant_mask(sprite["outline"], side)
                if len(_components(rem)) != 1:
                    raise RuntimeError("disconnected remnant")
                sprites.append(sprite)
                sides.append(side)

            source_positions = _place_source_boxes(panel_h, panel_w, sprites)
            target_positions = _place_target_boxes(panel_h, panel_w, sprites, sides)
            source_panel = _render_source_panel(panel_h, panel_w, sprites, source_positions)
            target_input, target_output = _render_target_panels(panel_h, panel_w, sprites, target_positions, sides)

            grid = np.concatenate(
                [source_panel, np.full((panel_h, 1), GREY, dtype=int), target_input],
                axis=1,
            )
            expected = np.concatenate(
                [source_panel, np.full((panel_h, 1), GREY, dtype=int), target_output],
                axis=1,
            )
            _semantic_checks(grid, expected)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
