from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
LABEL_COLORS = (1, 2, 3, 4, 7)
SPRITE_COLORS = (6, 8, 9)

GRID_HEIGHT_RANGE = (15, 19)
GRID_WIDTH_RANGE = (15, 19)
TARGET_COUNT_RANGE = (3, 5)
SPRITE_AREA_RANGE = (5, 8)
SPRITE_HEIGHT_RANGE = (3, 5)
SPRITE_WIDTH_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 500

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

LABEL_TEMPLATES = {
    1: [
        ((-1, -1),),
        ((-1, 0),),
        ((-1, 1),),
        ((0, -1),),
        ((0, 1),),
        ((1, -1),),
        ((1, 0),),
        ((1, 1),),
    ],
    2: [
        ((-1, -1), (-1, 0)),
        ((-1, 0), (-1, 1)),
        ((-1, 1), (0, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 0)),
        ((1, 0), (1, -1)),
        ((1, -1), (0, -1)),
        ((0, -1), (-1, -1)),
    ],
    3: [
        ((-1, -1), (-1, 0), (-1, 1)),
        ((1, -1), (1, 0), (1, 1)),
        ((-1, -1), (0, -1), (1, -1)),
        ((-1, 1), (0, 1), (1, 1)),
        ((-1, -1), (-1, 0), (0, -1)),
        ((-1, 0), (-1, 1), (0, 1)),
        ((0, 1), (1, 1), (1, 0)),
        ((0, -1), (1, -1), (1, 0)),
    ],
    4: [
        ((-1, -1), (-1, 0), (-1, 1), (0, 1)),
        ((-1, -1), (-1, 0), (-1, 1), (0, -1)),
        ((1, -1), (1, 0), (1, 1), (0, 1)),
        ((1, -1), (1, 0), (1, 1), (0, -1)),
        ((-1, -1), (0, -1), (1, -1), (-1, 0)),
        ((-1, -1), (0, -1), (1, -1), (1, 0)),
        ((-1, 1), (0, 1), (1, 1), (-1, 0)),
        ((-1, 1), (0, 1), (1, 1), (1, 0)),
    ],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    steps = N4 if connectivity == 4 else N8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    if coords.size == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _expand_cells(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _anchor_neighborhood(shape: tuple[int, int], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {
        (r, c)
        for r in range(ar - 1, ar + 2)
        for c in range(ac - 1, ac + 2)
        if _in_bounds(shape, r, c)
    }


def _touches_anchor(cells: set[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    ar, ac = anchor
    return any(max(abs(r - ar), abs(c - ac)) <= 1 for r, c in cells)


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: tuple[int, int]) -> None:
    r0, c0 = top
    h, w = sprite.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    mask = sprite != BG
    patch[mask] = sprite[mask]


def _generate_sprite(rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        h = int(rng.integers(SPRITE_HEIGHT_RANGE[0], SPRITE_HEIGHT_RANGE[1] + 1))
        w = int(rng.integers(SPRITE_WIDTH_RANGE[0], SPRITE_WIDTH_RANGE[1] + 1))
        area = int(rng.integers(SPRITE_AREA_RANGE[0], min(SPRITE_AREA_RANGE[1], h * w) + 1))

        seed = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in N4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != area:
            continue

        cells = _normalize_cells(cells)
        bh = max(r for r, _ in cells) + 1
        bw = max(c for _, c in cells) + 1
        if bh < 2 or bw < 2:
            continue

        anchor = tuple(sorted(cells)[int(rng.integers(len(cells)))])
        others = sorted(cells - {anchor})
        n_colors = int(rng.integers(2, min(3, len(others)) + 1))
        palette = list(map(int, rng.choice(np.array(SPRITE_COLORS, dtype=int), size=n_colors, replace=False)))
        body_colors = palette[:]
        while len(body_colors) < len(others):
            body_colors.append(palette[int(rng.integers(len(palette)))])
        rng.shuffle(body_colors)

        sprite = np.zeros((bh, bw), dtype=int)
        for (r, c), color in zip(others, body_colors):
            sprite[r, c] = int(color)
        sprite[anchor] = GRAY
        return sprite, anchor
    raise RuntimeError("failed to generate source sprite")


def _sprite_offsets(sprite: np.ndarray, anchor: tuple[int, int]) -> dict[tuple[int, int], int]:
    ar, ac = anchor
    offsets: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(sprite != BG):
        offsets[(int(r) - ar, int(c) - ac)] = int(sprite[int(r), int(c)])
    return offsets


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    gray_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == GRAY)]
    assert len(gray_cells) >= 2, "expected one source anchor and at least one target anchor"

    source_info = None
    for comp in _components_from_mask(g != BG, connectivity=4):
        cells = [tuple(map(int, rc)) for rc in comp]
        gray_inside = [cell for cell in cells if int(g[cell]) == GRAY]
        if not gray_inside:
            continue
        colors = {int(g[cell]) for cell in cells}
        if len(colors) >= 3:
            assert source_info is None, "expected a unique multicolor source sprite"
            assert len(gray_inside) == 1, "source sprite must contain one gray anchor"
            source_info = {
                "coords": comp,
                "cells": set(cells),
                "anchor": gray_inside[0],
                "colors": colors,
                "body_colors": colors - {BG, GRAY},
            }
    assert source_info is not None, "could not locate the multicolor source sprite"

    label_components = []
    for color in sorted(set(map(int, np.unique(g))) - {BG, GRAY}):
        for comp in _components_from_mask(g == color, connectivity=4):
            cells = {tuple(map(int, rc)) for rc in comp}
            if cells & source_info["cells"]:
                continue
            label_components.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "cells": cells,
                    "area": int(comp.shape[0]),
                }
            )

    targets = []
    used_components: set[int] = set()
    for anchor in gray_cells:
        if anchor == source_info["anchor"]:
            continue
        nearby = [idx for idx, comp in enumerate(label_components) if _touches_anchor(comp["cells"], anchor)]
        assert len(nearby) == 1, "each target anchor must have exactly one nearby label object"
        idx = nearby[0]
        assert idx not in used_components, "label objects must pair with only one target anchor"
        used_components.add(idx)
        comp = label_components[idx]
        neighborhood = _anchor_neighborhood(g.shape, anchor)
        visible = {(r, c) for (r, c) in neighborhood if int(g[r, c]) != BG}
        visible.discard(anchor)
        assert visible == comp["cells"], "target 8-neighborhood must contain only its own label object"
        assert all(max(abs(r - anchor[0]), abs(c - anchor[1])) <= 1 for r, c in comp["cells"])
        targets.append(
            {
                "anchor": anchor,
                "label_cells": comp["cells"],
                "label_color": comp["color"],
                "label_area": comp["area"],
                "active": comp["area"] > 1,
            }
        )

    assert len(used_components) == len(label_components), "unexpected extra non-source foreground objects"
    assert len(targets) >= 3, "expected 3-5 targets"
    return {"source": source_info, "targets": targets}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the multicolor source sprite onto each area-qualified target anchor, recolor by label color, then crop tightly.

    Input structure:
    - A black background contains one 4-connected multicolor source sprite with exactly one gray anchor pixel inside it.
    - Elsewhere there are several gray target-anchor pixels, each paired with exactly one nearby monochrome label object inside that anchor's 8-neighborhood.
    - Every label object is 4-connected and its area is either 1 pixel or greater than 1 pixel.

    Rule:
    - Find the unique source sprite as the only gray-containing foreground component with at least two non-gray body colors.
    - For each other gray anchor, inspect its nearby label object: if the label area is 1, erase the gray anchor and keep that single-pixel label; if the label area is greater than 1, erase both the gray anchor and the whole label object, then place a translated copy of the source sprite so the source anchor aligns to that target anchor.
    - In every activated copy, recolor every copied source pixel, including the copied gray anchor pixel, to the label object's color; keep the original source sprite unchanged.
    - Finally crop the result to the tight bounding box of all non-black pixels, preserving black holes inside that crop.

    Key insight:
    - Each label object has two visible roles at once: its area decides whether the nearby target anchor is activated, and its color decides the paint color of the copied sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    analysis = _analyze_grid(g)
    source = analysis["source"]
    targets = analysis["targets"]

    sr, sc = source["anchor"]
    source_offsets = [(int(r) - sr, int(c) - sc) for r, c in source["cells"]]

    out = g.copy()
    for target in targets:
        tr, tc = target["anchor"]
        out[tr, tc] = BG
        if target["active"]:
            for r, c in target["label_cells"]:
                out[r, c] = BG

    for target in targets:
        if not target["active"]:
            continue
        tr, tc = target["anchor"]
        color = target["label_color"]
        for dr, dc in source_offsets:
            rr, cc = tr + dr, tc + dc
            assert _in_bounds(g.shape, rr, cc), "translated copy must stay in bounds"
            out[rr, cc] = int(color)

    return _crop_nonblack(out)


def _build_target_specs(
    rng: np.random.Generator, source_rel_cells: set[tuple[int, int]]
) -> list[dict] | None:
    n_targets = int(rng.integers(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    n_active = int(rng.integers(2, n_targets))
    n_inactive = n_targets - n_active

    active_palette = list(map(int, rng.choice(np.array(LABEL_COLORS, dtype=int), size=2, replace=False)))
    active_colors = active_palette[:]
    while len(active_colors) < n_active:
        active_colors.append(int(rng.choice(np.array(LABEL_COLORS, dtype=int))))
    rng.shuffle(active_colors)

    inactive_colors = []
    if n_inactive > 0:
        inactive_colors.append(active_colors[0])
        while len(inactive_colors) < n_inactive:
            inactive_colors.append(int(rng.choice(np.array(LABEL_COLORS, dtype=int))))
        rng.shuffle(inactive_colors)

    exposed_areas = [
        area
        for area in (2, 3, 4)
        if any(set(template) - source_rel_cells for template in LABEL_TEMPLATES[area])
    ]
    if not exposed_areas:
        return None

    active_specs = [
        {
            "active": True,
            "color": active_colors[0],
            "area": int(exposed_areas[int(rng.integers(len(exposed_areas)))]),
            "require_exposed_erase": True,
        }
    ]
    for color in active_colors[1:]:
        active_specs.append(
            {
                "active": True,
                "color": int(color),
                "area": int(rng.integers(2, 5)),
                "require_exposed_erase": False,
            }
        )

    inactive_specs = [
        {"active": False, "color": int(color), "area": 1, "require_exposed_erase": False}
        for color in inactive_colors
    ]
    rng.shuffle(active_specs)
    active_specs.sort(key=lambda spec: not spec["require_exposed_erase"])
    rng.shuffle(inactive_specs)
    return active_specs + inactive_specs


def _choose_source_top(
    rng: np.random.Generator, grid_shape: tuple[int, int], sprite: np.ndarray
) -> tuple[int, int]:
    h, w = sprite.shape
    gh, gw = grid_shape
    r0 = int(rng.integers(1, gh - h))
    c0 = int(rng.integers(1, gw - w))
    return r0, c0


def _place_target(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    spec: dict,
    source_offsets: set[tuple[int, int]],
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> dict | None:
    gh, gw = grid_shape
    anchors = [(r, c) for r in range(1, gh - 1) for c in range(1, gw - 1)]
    rng.shuffle(anchors)

    templates = list(LABEL_TEMPLATES[spec["area"]])
    rng.shuffle(templates)

    for anchor in anchors:
        ar, ac = anchor
        for template in templates:
            label_cells = {(ar + dr, ac + dc) for dr, dc in template}
            planned_input = set(label_cells) | {anchor}
            if not all(_in_bounds(grid_shape, r, c) for r, c in planned_input):
                continue
            if _expand_cells(planned_input, 1) & input_occupied:
                continue
            if planned_input & output_occupied:
                continue

            if spec["active"]:
                copy_cells = {(ar + dr, ac + dc) for dr, dc in source_offsets}
                if not all(_in_bounds(grid_shape, r, c) for r, c in copy_cells):
                    continue
                if copy_cells & input_occupied:
                    continue
                if _expand_cells(copy_cells, 1) & output_occupied:
                    continue
                if spec["require_exposed_erase"] and not (label_cells - copy_cells):
                    continue
                return {
                    "anchor": anchor,
                    "label_cells": label_cells,
                    "label_color": spec["color"],
                    "label_area": spec["area"],
                    "active": True,
                    "copy_cells": copy_cells,
                    "exposed_erased_cells": label_cells - copy_cells,
                }

            output_cells = set(label_cells)
            if _expand_cells(output_cells, 1) & output_occupied:
                continue
            return {
                "anchor": anchor,
                "label_cells": label_cells,
                "label_color": spec["color"],
                "label_area": 1,
                "active": False,
                "copy_cells": set(),
                "exposed_erased_cells": set(),
            }
    return None


def _render_input(
    grid_shape: tuple[int, int], sprite: np.ndarray, sprite_top: tuple[int, int], targets: list[dict]
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    _paste_sprite(grid, sprite, sprite_top)
    for target in targets:
        ar, ac = target["anchor"]
        grid[ar, ac] = GRAY
        _paint_cells(grid, target["label_cells"], target["label_color"])
    return grid


def _render_expected(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    sprite_top: tuple[int, int],
    source_offsets_with_colors: dict[tuple[int, int], int],
    source_anchor_abs: tuple[int, int],
    targets: list[dict],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    _paste_sprite(out, sprite, sprite_top)
    sr, sc = source_anchor_abs
    for target in targets:
        if not target["active"]:
            _paint_cells(out, target["label_cells"], target["label_color"])
            continue
        ar, ac = target["anchor"]
        for (dr, dc), _src_color in source_offsets_with_colors.items():
            out[ar + dr, ac + dc] = int(target["label_color"])
    return _crop_nonblack(out)


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    sprite: np.ndarray,
    sprite_top: tuple[int, int],
    source_anchor: tuple[int, int],
    targets: list[dict],
) -> bool:
    try:
        analysis = _analyze_grid(inp)
        parsed_targets = {
            target["anchor"]: (target["label_color"], target["label_area"], target["active"])
            for target in analysis["targets"]
        }
        planned_targets = {
            target["anchor"]: (target["label_color"], target["label_area"], target["active"])
            for target in targets
        }
        if parsed_targets != planned_targets:
            return False
        if len(analysis["source"]["body_colors"]) < 2:
            return False
        if not (1 <= sum(not t["active"] for t in targets) <= len(targets) - 2):
            return False
        if sum(t["active"] for t in targets) < 2:
            return False
        if len({t["label_color"] for t in targets if t["active"]}) < 2:
            return False
        active_colors = {t["label_color"] for t in targets if t["active"]}
        inactive_colors = {t["label_color"] for t in targets if not t["active"]}
        if not (active_colors & inactive_colors):
            return False
        if not any(t["active"] and t["exposed_erased_cells"] for t in targets):
            return False

        sprite_offsets = {(int(r) - source_anchor[0], int(c) - source_anchor[1]) for r, c in np.argwhere(sprite != BG)}
        for target in targets:
            anchor = target["anchor"]
            neighborhood = _anchor_neighborhood(inp.shape, anchor)
            nonblack = {(r, c) for r, c in neighborhood if int(inp[r, c]) != BG}
            if nonblack != target["label_cells"] | {anchor}:
                return False
            if target["active"]:
                copy_cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in sprite_offsets}
                if any(int(inp[r, c]) != BG for r, c in copy_cells - target["label_cells"] - {anchor}):
                    return False

        if not np.array_equal(main(inp), expected):
            return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        gh = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
        gw = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
        grid_shape = (gh, gw)

        sprite, source_anchor_local = _generate_sprite(rng)
        sprite_top = _choose_source_top(rng, grid_shape, sprite)
        source_anchor_abs = (sprite_top[0] + source_anchor_local[0], sprite_top[1] + source_anchor_local[1])
        source_offsets_with_colors = _sprite_offsets(sprite, source_anchor_local)
        source_offsets = set(source_offsets_with_colors)

        source_cells_abs = {
            (sprite_top[0] + int(r), sprite_top[1] + int(c))
            for r, c in np.argwhere(sprite != BG)
        }

        specs = _build_target_specs(rng, source_offsets)
        if specs is None:
            continue

        input_occupied = set(source_cells_abs)
        output_occupied = set(source_cells_abs)
        targets: list[dict] = []
        ok = True
        for spec in specs:
            placed = _place_target(
                rng,
                grid_shape,
                spec,
                source_offsets,
                input_occupied,
                output_occupied,
            )
            if placed is None:
                ok = False
                break
            targets.append(placed)
            input_occupied |= {placed["anchor"]} | placed["label_cells"]
            if placed["active"]:
                output_occupied |= placed["copy_cells"]
            else:
                output_occupied |= placed["label_cells"]

        if not ok:
            continue

        inp = _render_input(grid_shape, sprite, sprite_top, targets)
        expected = _render_expected(grid_shape, sprite, sprite_top, source_offsets_with_colors, source_anchor_abs, targets)
        if _validate_example(inp, expected, sprite, sprite_top, source_anchor_local, targets):
            return inp

    raise RuntimeError("failed to generate a valid puzzle-9 input")
