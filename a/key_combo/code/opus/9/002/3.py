from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, paste, random_free_location


CATALOG_BG = Color.GREY
WORK_BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 250
SCENE_RETRIES = 30


SHAPE_LIBRARY = [
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=int,
    ),
]


def _rotation_variants(sprite: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(sprite, k).copy() for k in range(4)]


def _catalog_data(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    grey_components = [coords for coords in components_mask(grid == CATALOG_BG, connectivity=4) if np.any(coords[:, 1] == 0)]
    assert grey_components, "expected a grey catalog strip touching the left edge"
    grey_components.sort(key=lambda coords: int(coords.shape[0]), reverse=True)
    if len(grey_components) > 1:
        assert int(grey_components[0].shape[0]) > int(grey_components[1].shape[0]), "catalog strip must be unique"

    coords = grey_components[0]
    box = bbox_coords(coords)
    right = int(box[3])
    mask = np.zeros_like(grid, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask, box, right


def _extract_templates(grid: np.ndarray, catalog_right: int) -> list[dict]:
    catalog = grid[:, : catalog_right + 1]
    raw_mask = (catalog != CATALOG_BG) & (catalog != WORK_BG)
    templates = []
    for idx, coords in enumerate(sorted(components_mask(raw_mask, connectivity=4), key=lambda arr: tuple(arr.min(axis=0)))):
        box = bbox_coords(coords)
        raw = crop_bbox(catalog, box)
        sprite = np.where(raw == CATALOG_BG, WORK_BG, raw).astype(int)
        assert np.count_nonzero(sprite) >= 5, "templates must be nontrivial"
        assert len(set(map(int, sprite[sprite != 0].flat))) >= 3, "templates must be multi-color"
        templates.append(
            {
                "id": idx,
                "grid": sprite,
                "mask": sprite != 0,
                "area": int(np.count_nonzero(sprite)),
                "box": box,
            }
        )

    assert 2 <= len(templates) <= 3, "expected 2 or 3 catalog templates"
    return templates


def _component_color_map(grid: np.ndarray, coords: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(r), int(c)): int(grid[int(r), int(c)]) for r, c in coords}


def _dedupe_matches(matches: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for match in matches:
        key = (match["template_id"], match["rotation"], match["anchor"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(match)
    return unique


def _candidate_match(
    grid: np.ndarray,
    comp_map: dict[tuple[int, int], int],
    sprite: np.ndarray,
    anchor: tuple[int, int],
    workspace_left: int,
) -> bool:
    anchor_r, anchor_c = map(int, anchor)
    height, width = map(int, sprite.shape)
    if anchor_r < 0 or anchor_c < workspace_left or anchor_r + height > grid.shape[0] or anchor_c + width > grid.shape[1]:
        return False

    comp_positions = set(comp_map)
    sprite_mask = sprite != 0

    for (gr, gc), color in comp_map.items():
        rr = gr - anchor_r
        cc = gc - anchor_c
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if int(sprite[rr, cc]) != color:
            return False

    for rr, cc in np.argwhere(sprite_mask):
        gr = anchor_r + int(rr)
        gc = anchor_c + int(cc)
        if int(grid[gr, gc]) == WORK_BG:
            return False
        if (gr, gc) in comp_positions and comp_map[(gr, gc)] != int(sprite[rr, cc]):
            return False

    return True


def _match_component(grid: np.ndarray, coords: np.ndarray, templates: list[dict], workspace_left: int) -> dict:
    comp_map = _component_color_map(grid, coords)
    color_counts = Counter(comp_map.values())
    pivot = min(comp_map, key=lambda pos: (color_counts[comp_map[pos]], pos[0], pos[1]))
    pivot_color = comp_map[pivot]

    matches = []
    for template in templates:
        for rotation, sprite in enumerate(_rotation_variants(template["grid"])):
            if color_counts[pivot_color] > int(np.sum(sprite == pivot_color)):
                continue
            spots = np.argwhere(sprite == pivot_color)
            for rr, cc in spots:
                anchor = (int(pivot[0] - rr), int(pivot[1] - cc))
                if not _candidate_match(grid, comp_map, sprite, anchor, workspace_left):
                    continue
                matches.append(
                    {
                        "template_id": int(template["id"]),
                        "rotation": int(rotation),
                        "anchor": anchor,
                        "sprite": sprite,
                        "area": int(np.count_nonzero(sprite)),
                        "component_box": bbox_coords(coords),
                    }
                )

    unique = _dedupe_matches(matches)
    assert len(unique) == 1, "each workspace component must match exactly one rotated template"
    return unique[0]


def _workspace_candidates(grid: np.ndarray, templates: list[dict], workspace_left: int) -> list[dict]:
    workspace = grid[:, workspace_left:]
    visible_cells = [(int(r), int(c + workspace_left), int(workspace[r, c])) for r, c in np.argwhere(workspace != WORK_BG)]
    by_anchor: dict[tuple[int, int, int], dict] = {}

    for template in templates:
        for rotation, sprite in enumerate(_rotation_variants(template["grid"])):
            area = int(np.count_nonzero(sprite))
            positions_by_color: dict[int, list[tuple[int, int]]] = {}
            for rr, cc in np.argwhere(sprite != 0):
                positions_by_color.setdefault(int(sprite[rr, cc]), []).append((int(rr), int(cc)))

            for gr, gc, color in visible_cells:
                for rr, cc in positions_by_color.get(color, []):
                    anchor = (int(gr - rr), int(gc - cc))
                    key = (int(template["id"]), int(rotation), anchor[0], anchor[1])
                    if key in by_anchor:
                        continue
                    if anchor[0] < 0 or anchor[1] < workspace_left:
                        continue
                    if anchor[0] + sprite.shape[0] > grid.shape[0] or anchor[1] + sprite.shape[1] > grid.shape[1]:
                        continue

                    sprite_mask = sprite != 0
                    absolute_visible = set()
                    valid = True
                    for sr, sc in np.argwhere(sprite_mask):
                        ar = anchor[0] + int(sr)
                        ac = anchor[1] + int(sc)
                        cell = int(grid[ar, ac])
                        if cell == WORK_BG:
                            valid = False
                            break
                        if cell == int(sprite[sr, sc]):
                            absolute_visible.add((ar, ac))
                    if not valid or len(absolute_visible) * 5 < area * 3:
                        continue

                    by_anchor[key] = {
                        "template_id": int(template["id"]),
                        "rotation": int(rotation),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "sprite": sprite,
                        "area": area,
                        "visible_cells": absolute_visible,
                    }

    return sorted(by_anchor.values(), key=lambda cand: (-len(cand["visible_cells"]), -cand["area"], cand["anchor"]))


def _render_workspace(work_shape: tuple[int, int], matches: list[dict], workspace_left: int) -> np.ndarray:
    out = np.zeros(work_shape, dtype=int)
    ordered = sorted(matches, key=lambda match: (-int(match["area"]), int(match["anchor"][0]), int(match["anchor"][1])))
    for match in ordered:
        paste(
            out,
            match["sprite"],
            int(match["anchor"][0]),
            int(match["anchor"][1] - workspace_left),
            transparent=WORK_BG,
            overwrite=True,
        )
    return out


def _solve_workspace(grid: np.ndarray, templates: list[dict], workspace_left: int) -> list[dict]:
    workspace = grid[:, workspace_left:]
    universe = {(int(r), int(c + workspace_left)) for r, c in np.argwhere(workspace != WORK_BG)}
    candidates = _workspace_candidates(grid, templates, workspace_left)
    assert 2 <= len(candidates) <= 20, "workspace candidate set must stay small enough to remain unambiguous"
    coverers: dict[tuple[int, int], list[int]] = {cell: [] for cell in universe}
    for idx, cand in enumerate(candidates):
        for cell in cand["visible_cells"]:
            if cell in coverers:
                coverers[cell].append(idx)

    solutions: list[list[int]] = []
    selected: list[int] = []
    template_use: Counter[int] = Counter()

    def _dfs(uncovered: set[tuple[int, int]]) -> None:
        if len(solutions) >= 2:
            return
        if not uncovered:
            if len(selected) < 2:
                return
            rendered = _render_workspace(workspace.shape, [candidates[idx] for idx in selected], workspace_left)
            if np.array_equal(rendered, workspace):
                solutions.append(selected.copy())
            return
        if len(selected) == 4:
            return

        target = min(
            uncovered,
            key=lambda cell: len([idx for idx in coverers.get(cell, []) if template_use[candidates[idx]["template_id"]] < 2 and idx not in selected]),
        )
        options = [
            idx
            for idx in coverers.get(target, [])
            if template_use[candidates[idx]["template_id"]] < 2 and idx not in selected
        ]
        options.sort(key=lambda idx: (-len(candidates[idx]["visible_cells"] & uncovered), -candidates[idx]["area"]))

        for idx in options:
            covered = candidates[idx]["visible_cells"] & uncovered
            if not covered:
                continue
            selected.append(idx)
            template_use[candidates[idx]["template_id"]] += 1
            _dfs(uncovered - candidates[idx]["visible_cells"])
            template_use[candidates[idx]["template_id"]] -= 1
            if template_use[candidates[idx]["template_id"]] == 0:
                del template_use[candidates[idx]["template_id"]]
            selected.pop()

    _dfs(set(universe))
    assert len(solutions) == 1, "workspace must admit exactly one valid set of rotated template placements"
    return [candidates[idx] for idx in solutions[0]]


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _catalog_mask, _catalog_box, catalog_right = _catalog_data(g)
    templates = _extract_templates(g, catalog_right)
    workspace_left = catalog_right + 1
    matches = _solve_workspace(g, templates, workspace_left)
    return {
        "catalog_right": catalog_right,
        "workspace_left": workspace_left,
        "templates": templates,
        "matches": matches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the partially hidden workspace sprites by matching each visible fragment to a rotated catalog template.

    Input structure:
    - A grey strip on the left is a catalog region whose non-grey 4-connected sprites are the full templates.
    - The right workspace is black except for 2 to 4 visible foreground components, each coming from one rotated template.
    - Some workspace sprites are partially occluded by smaller sprites drawn above them, so only a connected visible subset remains.

    Rule:
    - Extract every catalog sprite as a tight crop with grey treated as transparent.
    - For each visible workspace component, find the unique template and rotation whose colored pixels contain that component at one translation and whose remaining template pixels fall only on already-nonblack workspace cells.
    - Order the matched workspace objects by full template area from large to small, breaking equal areas by smaller top row first, then redraw the complete rotated templates from bottom to top while keeping the catalog strip unchanged.

    Key insight:
    - The missing pixels in the workspace cannot be inferred from geometry alone; they are determined by the only rotated catalog template that can explain the visible fragment and its occluded cells.
    """
    analysis = _analyze_grid(grid)
    out = np.asarray(grid, dtype=int).copy()
    workspace_left = int(analysis["workspace_left"])
    out[:, workspace_left:] = WORK_BG

    ordered = sorted(
        analysis["matches"],
        key=lambda match: (-int(match["area"]), int(match["anchor"][0]), int(match["anchor"][1])),
    )
    for match in ordered:
        paste(out, match["sprite"], int(match["anchor"][0]), int(match["anchor"][1]), transparent=WORK_BG, overwrite=True)
    return out


def _colorize_mask(mask: np.ndarray) -> np.ndarray | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    palette_size = random.randint(3, min(5, len(PALETTE)))
    colors = random.sample(PALETTE, palette_size)
    random.shuffle(cells)
    sprite = np.zeros(mask.shape, dtype=int)
    for idx, (r, c) in enumerate(cells):
        color = colors[(2 * idx + r + 3 * c) % palette_size]
        sprite[r, c] = int(color)

    pivot_cells = sorted(cells)
    sprite[pivot_cells[0]] = int(colors[0])
    sprite[pivot_cells[-1]] = int(colors[1 % palette_size])
    sprite[pivot_cells[len(pivot_cells) // 2]] = int(colors[2 % palette_size])
    if len(set(map(int, sprite[sprite != 0].flat))) < 3:
        return None

    rotations = _rotation_variants(sprite)
    hashes = {rot.tobytes() for rot in rotations}
    if len(hashes) < 4:
        return None
    return sprite


def _sample_templates() -> list[np.ndarray] | None:
    n_templates = random.randint(2, 3)
    masks = random.sample(SHAPE_LIBRARY, n_templates)
    sprites = []
    for mask in masks:
        sprite = _colorize_mask(mask)
        if sprite is None:
            return None
        sprites.append(sprite)
    return sprites


def _render_catalog(sprites: list[np.ndarray]) -> tuple[np.ndarray, int]:
    gap = 1
    margin = 1
    width = max(sprite.shape[1] for sprite in sprites) + 2
    height_needed = sum(sprite.shape[0] for sprite in sprites) + gap * (len(sprites) - 1) + 2
    height = random.randint(max(18, height_needed), 24)
    catalog = np.full((height, width), CATALOG_BG, dtype=int)
    row = margin
    for sprite in sprites:
        col = random.randint(1, width - sprite.shape[1] - 1)
        paste(catalog, sprite, row, col, transparent=WORK_BG, overwrite=True)
        row += sprite.shape[0] + gap
    return catalog, width


def _make_object_specs(templates: list[np.ndarray]) -> list[dict] | None:
    n_objects = random.choice([2, 3, 3, 4])
    counts = {idx: 0 for idx in range(len(templates))}
    specs = []
    areas = [int(np.count_nonzero(template)) for template in templates]
    if len(set(areas)) < 2:
        return None
    for _ in range(n_objects):
        choices = [idx for idx, count in counts.items() if count < 2]
        if not choices:
            return None
        if not specs:
            template_id = int(max(choices, key=lambda idx: areas[idx]))
        elif len(specs) == 1:
            smaller = [idx for idx in choices if areas[idx] < specs[0]["area"]]
            template_id = random.choice(smaller or choices)
        else:
            template_id = random.choice(choices)
        counts[template_id] += 1
        rotation = random.randint(0, 3)
        sprite = np.rot90(templates[template_id], rotation).copy()
        specs.append(
            {
                "template_id": int(template_id),
                "rotation": int(rotation),
                "sprite": sprite,
                "area": int(np.count_nonzero(sprite)),
            }
        )

    if len({spec["area"] for spec in specs}) < 2:
        return None
    if all(spec["rotation"] == 0 for spec in specs):
        pick = random.randrange(len(specs))
        specs[pick]["rotation"] = 1
        specs[pick]["sprite"] = np.rot90(templates[specs[pick]["template_id"]], 1).copy()
    return specs


def _place_non_overlapping(
    canvas: np.ndarray,
    objects: list[dict],
    skip_indices: set[int],
) -> bool:
    order = [idx for idx in range(len(objects)) if idx not in skip_indices]
    order.sort(key=lambda idx: (-objects[idx]["area"], idx))
    for idx in order:
        sprite = objects[idx]["sprite"]
        try:
            loc = random_free_location(canvas, sprite, transparent=WORK_BG, border=1, padding=1, padding_connectivity=4)
        except ValueError:
            return False
        objects[idx]["work_anchor"] = tuple(map(int, loc))
        paste(canvas, sprite, int(loc[0]), int(loc[1]), transparent=WORK_BG, overwrite=True)
    return True


def _full_object_mask(work_shape: tuple[int, int], obj: dict) -> np.ndarray:
    mask = np.zeros(work_shape, dtype=bool)
    r0, c0 = map(int, obj["work_anchor"])
    sprite_mask = obj["sprite"] != 0
    h, w = sprite_mask.shape
    mask[r0 : r0 + h, c0 : c0 + w] = sprite_mask
    return mask


def _layer_order(objects: list[dict], workspace_left: int) -> list[int]:
    return sorted(
        range(len(objects)),
        key=lambda idx: (
            -int(objects[idx]["area"]),
            int(objects[idx]["work_anchor"][0]),
            int(workspace_left + objects[idx]["work_anchor"][1]),
        ),
    )


def _scene_valid(
    catalog: np.ndarray,
    workspace_left: int,
    work_shape: tuple[int, int],
    objects: list[dict],
    require_hidden: bool,
) -> tuple[bool, np.ndarray | None]:
    if workspace_left + work_shape[1] > 30:
        return False, None

    object_masks = [_full_object_mask(work_shape, obj) for obj in objects]
    order = _layer_order(objects, workspace_left)
    upper = np.zeros(work_shape, dtype=bool)
    visible_masks = {}
    hidden_count = 0
    for idx in reversed(order):
        visible = object_masks[idx] & (~upper)
        visible_masks[idx] = visible
        hidden_count += int(object_masks[idx].sum() - visible.sum())
        upper |= object_masks[idx]

    if require_hidden and hidden_count == 0:
        return False, None

    for idx, obj in enumerate(objects):
        area = int(obj["area"])
        visible_area = int(visible_masks[idx].sum())
        if visible_area * 5 < area * 3:
            return False, None

    work = np.zeros(work_shape, dtype=int)
    for idx in order:
        r0, c0 = map(int, objects[idx]["work_anchor"])
        paste(work, objects[idx]["sprite"], r0, c0, transparent=WORK_BG, overwrite=True)

    full = np.zeros((catalog.shape[0], workspace_left + work_shape[1]), dtype=int)
    full[:, :workspace_left] = catalog
    full[:, workspace_left:] = work

    try:
        analysis = _analyze_grid(full)
    except AssertionError:
        return False, None

    found = sorted((match["template_id"], match["rotation"], match["anchor"]) for match in analysis["matches"])
    expected = sorted(
        (
            int(obj["template_id"]),
            int(obj["rotation"]),
            (int(obj["work_anchor"][0]), int(workspace_left + obj["work_anchor"][1])),
        )
        for obj in objects
    )
    if found != expected:
        return False, None

    return True, full


def _overlap_candidates_simple(work_shape: tuple[int, int], objects: list[dict], top_idx: int, bottom_idx: int) -> list[tuple[int, int]]:
    bottom_mask = _full_object_mask(work_shape, objects[bottom_idx])
    occupied_by_others = np.zeros(work_shape, dtype=bool)
    for idx, obj in enumerate(objects):
        if idx in {top_idx, bottom_idx}:
            continue
        occupied_by_others |= _full_object_mask(work_shape, obj)

    sprite = objects[top_idx]["sprite"]
    sprite_mask = sprite != 0
    h, w = sprite.shape
    candidates = []
    for r0 in range(1, work_shape[0] - h):
        for c0 in range(1, work_shape[1] - w):
            placed = np.zeros(work_shape, dtype=bool)
            placed[r0 : r0 + h, c0 : c0 + w] = sprite_mask
            if np.any(placed & occupied_by_others):
                continue
            overlap = int(np.sum(placed & bottom_mask))
            if overlap == 0:
                continue
            if overlap * 5 > int(objects[bottom_idx]["area"]) * 2:
                continue
            candidates.append((int(r0), int(c0)))
    random.shuffle(candidates)
    return candidates


def _pair_layouts(bottom_sprite: np.ndarray, top_sprite: np.ndarray) -> list[dict]:
    bh, bw = bottom_sprite.shape
    th, tw = top_sprite.shape
    canvas_shape = (bh + th + 4, bw + tw + 4)
    bottom_anchor = (th + 1, tw + 1)
    bottom_mask = np.zeros(canvas_shape, dtype=bool)
    bottom_mask[
        bottom_anchor[0] : bottom_anchor[0] + bh,
        bottom_anchor[1] : bottom_anchor[1] + bw,
    ] = bottom_sprite != 0

    layouts = []
    for dr in range(-th + 1, bh):
        for dc in range(-tw + 1, bw):
            top_anchor = (bottom_anchor[0] + dr, bottom_anchor[1] + dc)
            if top_anchor[0] < 0 or top_anchor[1] < 0 or top_anchor[0] + th > canvas_shape[0] or top_anchor[1] + tw > canvas_shape[1]:
                continue
            top_mask = np.zeros(canvas_shape, dtype=bool)
            top_mask[
                top_anchor[0] : top_anchor[0] + th,
                top_anchor[1] : top_anchor[1] + tw,
            ] = top_sprite != 0
            overlap = int(np.sum(bottom_mask & top_mask))
            if overlap == 0:
                continue
            if overlap * 5 > int(np.count_nonzero(bottom_sprite)) * 2:
                continue

            bottom_visible = bottom_mask & (~top_mask)
            if int(bottom_visible.sum()) * 5 < int(np.count_nonzero(bottom_sprite)) * 3:
                continue
            if len(components_mask(bottom_visible, connectivity=4)) != 1:
                continue

            pair = np.zeros(canvas_shape, dtype=int)
            paste(pair, bottom_sprite, int(bottom_anchor[0]), int(bottom_anchor[1]), transparent=WORK_BG, overwrite=True)
            paste(pair, top_sprite, int(top_anchor[0]), int(top_anchor[1]), transparent=WORK_BG, overwrite=True)
            if len(components_mask(pair != 0, connectivity=4)) != 2:
                continue

            union_coords = np.argwhere(bottom_mask | top_mask)
            r0, c0, r1, c1 = bbox_coords(union_coords)
            layouts.append(
                {
                    "pair_patch": crop_bbox(pair, (r0, c0, r1, c1)),
                    "bottom_rel": (int(bottom_anchor[0] - r0), int(bottom_anchor[1] - c0)),
                    "top_rel": (int(top_anchor[0] - r0), int(top_anchor[1] - c0)),
                }
            )
    random.shuffle(layouts)
    return layouts


def _place_overlap_pair(
    work_shape: tuple[int, int],
    objects: list[dict],
    bottom_idx: int,
    top_idx: int,
) -> np.ndarray | None:
    layouts = _pair_layouts(objects[bottom_idx]["sprite"], objects[top_idx]["sprite"])
    for layout in layouts[:80]:
        canvas = np.zeros(work_shape, dtype=int)
        patch = layout["pair_patch"]
        try:
            pair_anchor = random_free_location(canvas, patch, transparent=WORK_BG, border=1, padding=1, padding_connectivity=4)
        except ValueError:
            continue
        pr, pc = map(int, pair_anchor)
        br, bc = layout["bottom_rel"]
        tr, tc = layout["top_rel"]
        objects[bottom_idx]["work_anchor"] = (pr + int(br), pc + int(bc))
        objects[top_idx]["work_anchor"] = (pr + int(tr), pc + int(tc))
        paste(canvas, patch, pr, pc, transparent=WORK_BG, overwrite=True)
        return canvas
    return None


def _build_scene() -> np.ndarray | None:
    start_time = time.time()
    templates = _sample_templates()
    if templates is None:
        return None

    catalog, workspace_left = _render_catalog(templates)
    objects = _make_object_specs(templates)
    if objects is None:
        return None

    max_work_width = 30 - workspace_left
    min_work_width = min(max_work_width, 15)
    work_width = random.randint(min_work_width, max_work_width)
    work_shape = (catalog.shape[0], work_width)

    top_idx = min(range(len(objects)), key=lambda idx: (objects[idx]["area"], idx))
    bottom_choices = [idx for idx in range(len(objects)) if objects[idx]["area"] > objects[top_idx]["area"]]
    if not bottom_choices:
        return None
    bottom_idx = max(bottom_choices, key=lambda idx: (objects[idx]["area"], -idx))

    for _ in range(SCENE_RETRIES):
        if time.time() - start_time > 0.5:
            return None
        canvas = np.zeros(work_shape, dtype=int)
        if not _place_non_overlapping(canvas, objects, {top_idx}):
            continue
        candidates = _overlap_candidates_simple(work_shape, objects, top_idx, bottom_idx)
        if not candidates:
            continue
        for r0, c0 in candidates[:30]:
            if time.time() - start_time > 0.5:
                return None
            objects[top_idx]["work_anchor"] = (int(r0), int(c0))
            ok, full = _scene_valid(catalog, workspace_left, work_shape, objects, require_hidden=True)
            if ok:
                return full
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_scene()
        if grid is None:
            continue
        out = main(grid)
        if out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid rotated-template occlusion scene")
