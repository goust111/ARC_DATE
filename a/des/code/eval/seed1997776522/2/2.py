from __future__ import annotations

import random
from collections import Counter

import numpy as np


GEN_BG = 0
GEN_TEMPLATE = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500
GEN_PLACE_ATTEMPTS = 120
GEN_SHAPE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = _neighbors4 if connectivity == 4 else _neighbors8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _rotation_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _has_unique_rotation(mask: np.ndarray) -> bool:
    return len({_mask_key(np.rot90(mask, k)) for k in range(4)}) == 4


def _rotation_to_match(piece_mask: np.ndarray, hole_mask: np.ndarray) -> int | None:
    hits: list[int] = []
    for k in range(4):
        rotated = np.rot90(piece_mask, k)
        if rotated.shape == hole_mask.shape and np.array_equal(rotated, hole_mask):
            hits.append(k)
    if len(hits) == 1:
        return hits[0]
    return None


def _rotate_point(
    r: int,
    c: int,
    hole_bbox: tuple[int, int, int, int],
    k: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = hole_bbox
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    dr2 = 2 * r - center_r2
    dc2 = 2 * c - center_c2
    for _ in range(k % 4):
        dr2, dc2 = -dc2, dr2
    nr2 = center_r2 + dr2
    nc2 = center_c2 + dc2
    assert nr2 % 2 == 0 and nc2 % 2 == 0
    return nr2 // 2, nc2 // 2


def _rotate_cells(
    cells: list[tuple[int, int]],
    hole_bbox: tuple[int, int, int, int],
    k: int,
) -> list[tuple[int, int]]:
    return [_rotate_point(r, c, hole_bbox, k) for r, c in cells]


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey template hole with its matching rotated piece, and rotate
    the template's colored decoration markers by the same angle around the
    hole's center.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    non_bg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert non_bg_colors, "expected foreground colors"
    template_color = max(non_bg_colors, key=lambda color: int(np.sum(g == color)))

    template_components = _components(g == template_color, connectivity=4)
    assert template_components, "expected grey template components"

    templates: list[dict] = []
    template_boxes: np.ndarray = np.zeros(g.shape, dtype=bool)

    for comp in sorted(template_components, key=lambda cells: _bbox(cells)):
        r0, c0, r1, c1 = _bbox(comp)
        template_boxes[r0 : r1 + 1, c0 : c1 + 1] = True

        hole_cells = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(g[r, c]) == bg
        ]
        assert hole_cells, "each template must contain one hole"

        markers: dict[int, list[tuple[int, int]]] = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                value = int(g[r, c])
                if value in {bg, template_color}:
                    continue
                markers.setdefault(value, []).append((r, c))

        hole_bbox = _bbox(hole_cells)
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "hole_cells": hole_cells,
                "hole_mask": _mask_from_cells(hole_cells),
                "hole_bbox": hole_bbox,
                "markers": markers,
            }
        )

    piece_mask = (g != bg) & (g != template_color) & (~template_boxes)
    piece_components = _components(piece_mask, connectivity=4)
    assert len(piece_components) == len(templates), "piece count must match template count"

    assignments: dict[int, dict] = {}
    used_templates: set[int] = set()

    for comp in piece_components:
        piece_colors = {int(g[r, c]) for r, c in comp}
        assert len(piece_colors) == 1, "each piece must be monochrome"
        piece_color = next(iter(piece_colors))
        piece_shape = _mask_from_cells(comp)

        options: list[tuple[int, int]] = []
        for idx, template in enumerate(templates):
            rot = _rotation_to_match(piece_shape, template["hole_mask"])
            if rot is not None:
                options.append((idx, rot))

        assert len(options) == 1, "each piece must match exactly one template"
        template_idx, rot = options[0]
        assert template_idx not in used_templates, "templates must match uniquely"
        used_templates.add(template_idx)
        assignments[template_idx] = {
            "piece_cells": comp,
            "piece_color": piece_color,
            "rotation": rot,
        }

    assert len(assignments) == len(templates), "every template must receive one piece"

    out = g.copy()

    for info in assignments.values():
        for r, c in info["piece_cells"]:
            out[r, c] = bg

    for idx, template in enumerate(templates):
        info = assignments[idx]
        rotation = int(info["rotation"])
        piece_color = int(info["piece_color"])

        for cells in template["markers"].values():
            for r, c in cells:
                out[r, c] = template_color

        for r, c in template["hole_cells"]:
            out[r, c] = piece_color

        for marker_color, cells in template["markers"].items():
            rotated = _rotate_cells(cells, template["hole_bbox"], rotation)
            for rr, cc in rotated:
                assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
                out[rr, cc] = int(marker_color)

    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = random.randint(max(5, min(h * w - 1, 5)), min(h * w - 1, 8))

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(
                        0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        for nr, nc in _neighbors4(r, c)
                    ):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        mask = _mask_from_cells(cells)
        if int(mask.sum()) < 5:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if (mask.shape[0] % 2) != (mask.shape[1] % 2):
            continue
        if int(mask.sum()) == mask.shape[0] * mask.shape[1]:
            continue
        if not _has_unique_rotation(mask):
            continue
        return mask
    return None


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    padding: int,
) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _find_candidate_marker_cells(
    patch_shape: tuple[int, int],
    hole_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = patch_shape
    candidates: set[tuple[int, int]] = set()
    for r, c in hole_cells:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in hole_cells:
                candidates.add((nr, nc))
    return sorted(candidates)


def _valid_rotated_group(
    group: list[tuple[int, int]],
    hole_bbox: tuple[int, int, int, int],
    rotation: int,
    patch_shape: tuple[int, int],
    hole_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    rotated = _rotate_cells(group, hole_bbox, rotation)
    if len(set(rotated)) != len(rotated):
        return None
    h, w = patch_shape
    for r, c in rotated:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if (r, c) in hole_cells:
            return None
    return rotated


def _choose_marker_groups(
    patch_shape: tuple[int, int],
    hole_bbox: tuple[int, int, int, int],
    hole_cells: set[tuple[int, int]],
    marker_count: int,
    rotation: int,
) -> list[list[tuple[int, int]]] | None:
    candidates = _find_candidate_marker_cells(patch_shape, hole_cells)
    single_groups = []
    for cell in candidates:
        rotated = _valid_rotated_group([cell], hole_bbox, rotation, patch_shape, hole_cells)
        if rotated is not None:
            single_groups.append(([cell], rotated))

    pair_groups = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            dr = abs(a[0] - b[0])
            dc = abs(a[1] - b[1])
            if (dr, dc) not in {(1, 1), (0, 1), (1, 0)}:
                continue
            rotated = _valid_rotated_group([a, b], hole_bbox, rotation, patch_shape, hole_cells)
            if rotated is not None:
                pair_groups.append(([a, b], rotated))

    desired_sizes = [random.choice([1, 2]) for _ in range(marker_count)]
    random.shuffle(desired_sizes)

    for _ in range(80):
        used_input: set[tuple[int, int]] = set()
        used_output: set[tuple[int, int]] = set()
        groups: list[list[tuple[int, int]]] = []
        ok = True

        for desired in desired_sizes:
            pools = [pair_groups, single_groups] if desired == 2 else [single_groups, pair_groups]
            chosen_group = None
            for pool in pools:
                options = [
                    group
                    for group, rotated in pool
                    if set(group).isdisjoint(used_input) and set(rotated).isdisjoint(used_output)
                ]
                if options:
                    chosen_group = random.choice(options)
                    break
            if chosen_group is None:
                ok = False
                break

            rotated_group = _valid_rotated_group(
                chosen_group,
                hole_bbox,
                rotation,
                patch_shape,
                hole_cells,
            )
            assert rotated_group is not None
            groups.append(chosen_group)
            used_input.update(chosen_group)
            used_output.update(rotated_group)

        if ok and len(groups) == marker_count:
            return groups

    return None


def _build_template(
    hole_mask: np.ndarray,
    piece_color: int,
    marker_colors: list[int],
    rotation: int,
) -> dict | None:
    hh, hw = hole_mask.shape
    top = random.randint(2, 4)
    bottom = random.randint(2, 4)
    left = random.randint(2, 4)
    right = random.randint(2, 4)
    th = hh + top + bottom
    tw = hw + left + right

    patch = np.full((th, tw), GEN_TEMPLATE, dtype=int)
    hole_cells_local: set[tuple[int, int]] = set()
    for r in range(hh):
        for c in range(hw):
            if hole_mask[r, c]:
                rr = top + r
                cc = left + c
                patch[rr, cc] = GEN_BG
                hole_cells_local.add((rr, cc))

    hole_bbox = _bbox(hole_cells_local)
    groups = _choose_marker_groups(
        patch.shape,
        hole_bbox,
        hole_cells_local,
        len(marker_colors),
        rotation,
    )
    if groups is None:
        return None

    markers: dict[int, list[tuple[int, int]]] = {}
    for color, group in zip(marker_colors, groups, strict=True):
        markers[int(color)] = [(int(r), int(c)) for r, c in group]
        for r, c in group:
            patch[r, c] = int(color)

    piece_mask = np.rot90(hole_mask, -rotation)
    piece_patch = np.full(piece_mask.shape, GEN_BG, dtype=int)
    piece_patch[piece_mask] = int(piece_color)

    return {
        "template_patch": patch,
        "piece_patch": piece_patch,
        "piece_color": int(piece_color),
        "hole_cells_local": sorted(hole_cells_local),
        "hole_bbox_local": hole_bbox,
        "markers_local": markers,
        "rotation": int(rotation),
    }


def _place_patch(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    padding: int,
    border: int,
) -> tuple[int, int] | None:
    h, w = grid_shape
    ph, pw = patch_shape
    if ph + 2 * border > h or pw + 2 * border > w:
        return None

    for _ in range(GEN_PLACE_ATTEMPTS):
        r0 = random.randint(border, h - ph - border)
        c0 = random.randint(border, w - pw - border)
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if any(_boxes_overlap(box, other, padding) for other in existing_boxes):
            continue
        return r0, c0
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _expected_output(inp: np.ndarray, placed_templates: list[dict], placed_pieces: list[dict]) -> np.ndarray:
    out = inp.copy()

    for piece in placed_pieces:
        r0, c0 = piece["origin"]
        patch = piece["patch"]
        for rr, cc in np.argwhere(patch != GEN_BG):
            out[r0 + int(rr), c0 + int(cc)] = GEN_BG

    for template, piece in zip(placed_templates, placed_pieces, strict=True):
        tr, tc = template["origin"]

        for cells in template["markers_local"].values():
            for r, c in cells:
                out[tr + r, tc + c] = GEN_TEMPLATE

        for r, c in template["hole_cells_local"]:
            out[tr + r, tc + c] = int(piece["color"])

        abs_hole_bbox = (
            tr + template["hole_bbox_local"][0],
            tc + template["hole_bbox_local"][1],
            tr + template["hole_bbox_local"][2],
            tc + template["hole_bbox_local"][3],
        )
        for marker_color, cells in template["markers_local"].items():
            abs_cells = [(tr + r, tc + c) for r, c in cells]
            rotated = _rotate_cells(abs_cells, abs_hole_bbox, template["rotation"])
            for rr, cc in rotated:
                out[rr, cc] = int(marker_color)

    return out


def _marker_counts(template_count: int) -> list[int]:
    if template_count == 2:
        return [random.choice([1, 2]), random.choice([1, 2])]
    pattern = random.choice([(1, 1, 1), (2, 1, 1), (2, 2, 1)])
    counts = list(pattern)
    random.shuffle(counts)
    return counts


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    template_count = random.choice([2, 3])
    marker_counts = _marker_counts(template_count)
    total_markers = sum(marker_counts)

    palette = GEN_COLORS[:]
    random.shuffle(palette)
    piece_colors = palette[:template_count]
    marker_palette = palette[template_count : template_count + total_markers]

    hole_masks: list[np.ndarray] = []
    seen_shapes: set[tuple[str, ...]] = set()
    for _ in range(template_count):
        mask = None
        for _ in range(GEN_SHAPE_ATTEMPTS):
            candidate = _random_shape()
            if candidate is None:
                continue
            signature = _rotation_signature(candidate)
            if signature in seen_shapes:
                continue
            seen_shapes.add(signature)
            mask = candidate
            break
        if mask is None:
            return None
        hole_masks.append(mask)

    templates_local: list[dict] = []
    marker_cursor = 0
    for idx in range(template_count):
        count = marker_counts[idx]
        marker_colors = marker_palette[marker_cursor : marker_cursor + count]
        marker_cursor += count
        rotation = random.randint(0, 3)
        template = _build_template(hole_masks[idx], piece_colors[idx], marker_colors, rotation)
        if template is None:
            return None
        templates_local.append(template)

    if template_count == 2:
        h = random.randint(24, 30)
        w = random.randint(24, 32)
    else:
        h = random.randint(28, 34)
        w = random.randint(28, 34)
    grid = np.full((h, w), GEN_BG, dtype=int)

    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_templates: list[dict] = []
    template_order = list(range(template_count))
    random.shuffle(template_order)
    template_origins: dict[int, tuple[int, int]] = {}

    for idx in template_order:
        patch = templates_local[idx]["template_patch"]
        origin = _place_patch(grid.shape, patch.shape, placed_boxes, padding=2, border=1)
        if origin is None:
            return None
        r0, c0 = origin
        _paste_patch(grid, patch, r0, c0)
        placed_boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))
        template_origins[idx] = origin

    placed_pieces_by_idx: dict[int, dict] = {}
    piece_order = list(range(template_count))
    random.shuffle(piece_order)

    for idx in piece_order:
        patch = templates_local[idx]["piece_patch"]
        origin = _place_patch(grid.shape, patch.shape, placed_boxes, padding=2, border=1)
        if origin is None:
            return None
        r0, c0 = origin
        for rr, cc in np.argwhere(patch != GEN_BG):
            grid[r0 + int(rr), c0 + int(cc)] = int(templates_local[idx]["piece_color"])
        placed_boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))
        placed_pieces_by_idx[idx] = {
            "origin": origin,
            "patch": patch,
            "color": int(templates_local[idx]["piece_color"]),
        }

    for idx in range(template_count):
        local = templates_local[idx]
        placed_templates.append(
            {
                "origin": template_origins[idx],
                "hole_cells_local": local["hole_cells_local"],
                "hole_bbox_local": local["hole_bbox_local"],
                "markers_local": local["markers_local"],
                "rotation": local["rotation"],
            }
        )

    placed_pieces = [placed_pieces_by_idx[idx] for idx in range(template_count)]
    expected = _expected_output(grid, placed_templates, placed_pieces)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
