from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, in_bounds, neighbors4


BG = 0
MARKER = 5
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 600


def _coverage_from_markers(markers: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    cov = np.zeros(shape, dtype=int)
    h, w = shape
    for r, c in markers:
        cov[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] += 1
    return cov


def _template_mask_from_markers(markers: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    return _coverage_from_markers(markers, shape) >= 2


def _normalize_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "template mask must be non-empty"
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(sorted(cells), dtype=int)
    return bbox_coords(coords)


def _is_connected_offsets(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if not cell_set:
        return False
    stack = [next(iter(cell_set))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cell_set)


def _is_hole_free(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    padded_bg = np.pad(~crop, 1, constant_values=True)
    return len(components_mask(padded_bg, connectivity=4)) == 1


def _subset_translations(
    template_offsets: tuple[tuple[int, int], ...], subset_offsets: tuple[tuple[int, int], ...]
) -> set[tuple[int, int]]:
    template_set = set(template_offsets)
    first = subset_offsets[0]
    shifts: set[tuple[int, int]] = set()
    for tr, tc in template_offsets:
        dr = first[0] - tr
        dc = first[1] - tc
        if all((sr - dr, sc - dc) in template_set for sr, sc in subset_offsets):
            shifts.add((dr, dc))
    return shifts


def _unique_anchor_subsets(template_offsets: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    max_size = min(4, len(template_offsets) - 1)
    for size in range(2, max_size + 1):
        for combo in combinations(template_offsets, size):
            subset = tuple(sorted(combo))
            if not _is_connected_offsets(subset):
                continue
            if _subset_translations(template_offsets, subset) == {(0, 0)}:
                out.append(subset)
    return out


def _template_dims(template_offsets: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in template_offsets) + 1, max(c for _r, c in template_offsets) + 1


def _placement_cells(
    template_offsets: tuple[tuple[int, int], ...], anchor_r: int, anchor_c: int
) -> set[tuple[int, int]]:
    return {(anchor_r + r, anchor_c + c) for r, c in template_offsets}


def _find_fragment_placements(
    grid: np.ndarray, template_offsets: tuple[tuple[int, int], ...], fragment_cells: set[tuple[int, int]]
) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    th, tw = _template_dims(template_offsets)
    bg = int(get_bg(grid))
    out: list[set[tuple[int, int]]] = []

    for ar in range(h - th + 1):
        for ac in range(w - tw + 1):
            cells = _placement_cells(template_offsets, ar, ac)
            if not fragment_cells.issubset(cells):
                continue
            ok = True
            for r, c in cells:
                v = int(grid[r, c])
                if v != bg and (r, c) not in fragment_cells:
                    ok = False
                    break
            if ok:
                out.append(cells)
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden overlap template from grey 3x3 halos, then complete every colored fragment into a translated copy of that template.

    Input structure:
    - A black background contains 4 to 8 grey marker pixels clustered in one area.
    - Elsewhere the grid contains 1 or 2 small colored fragments, each a 4-connected subset of the same unseen template.
    - The template itself is not drawn directly; it is defined only by the markers' halo overlaps.

    Rule:
    - Dilate every grey marker to its 3x3 neighborhood, clipped by the grid border.
    - Count how many halos cover each cell; the cells covered by 2 or more halos form the template mask.
    - For each colored fragment, find the unique translation of that template whose cell set contains the fragment and does not overlap other non-background cells.
    - Fill the missing cells of that translated template with the fragment's color, while keeping the grey markers and all existing pixels.

    Key insight:
    - The needed shape is an emergent object: it only appears after thresholding the overlap counts of multiple marker halos, so fragment completion depends on first reconstructing that hidden mask.
    """
    g = np.asarray(grid, dtype=int)
    markers = [tuple(map(int, cell)) for cell in np.argwhere(g == MARKER)]
    assert markers, "expected grey markers"

    template_mask = _template_mask_from_markers(markers, g.shape)
    assert template_mask.any(), "marker overlaps must produce a template"
    template_offsets = _normalize_mask(template_mask)

    out = g.copy()
    bg = int(get_bg(g))
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, MARKER)):
        for comp in components_mask(g == color, connectivity=4):
            fragment_cells = {tuple(map(int, cell)) for cell in comp}
            placements = _find_fragment_placements(g, template_offsets, fragment_cells)
            assert len(placements) == 1, f"fragment color {color} must have a unique template placement"
            for r, c in placements[0]:
                out[r, c] = color

    return out


def _sample_marker_scene(
    h: int, w: int, min_unique_subsets: int
) -> tuple[list[tuple[int, int]], tuple[tuple[int, int], ...], np.ndarray] | None:
    for _ in range(300):
        box_h = random.randint(3, 5)
        box_w = random.randint(3, 5)
        top = random.randint(2, h - box_h - 3)
        left = random.randint(2, w - box_w - 3)
        box_cells = [(r, c) for r in range(top, top + box_h) for c in range(left, left + box_w)]
        marker_count = random.randint(4, min(8, len(box_cells)))
        markers = random.sample(box_cells, marker_count)

        # Keep the marker cloud locally dense so halo overlaps matter.
        if any(
            min(max(abs(r - r2), abs(c - c2)) for r2, c2 in markers if (r2, c2) != (r, c)) > 3
            for r, c in markers
        ):
            continue

        template_mask = _template_mask_from_markers(markers, (h, w))
        area = int(template_mask.sum())
        if not 4 <= area <= 8:
            continue
        if len(components_mask(template_mask, connectivity=4)) != 1:
            continue
        if not _is_hole_free(template_mask):
            continue

        coords = np.argwhere(template_mask)
        r0, c0, r1, c1 = bbox_coords(coords)
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area == bbox_area:
            continue

        template_cells = {tuple(map(int, cell)) for cell in coords}
        if any(
            template_cells
            == {
                (rr, cc)
                for rr in range(max(0, mr - 1), min(h, mr + 2))
                for cc in range(max(0, mc - 1), min(w, mc + 2))
            }
            for mr, mc in markers
        ):
            continue

        # Every marker must be semantically relevant.
        necessary = True
        for idx in range(len(markers)):
            reduced = markers[:idx] + markers[idx + 1 :]
            if np.array_equal(_template_mask_from_markers(reduced, (h, w)), template_mask):
                necessary = False
                break
        if not necessary:
            continue

        template_offsets = _normalize_mask(template_mask)
        unique_subsets = _unique_anchor_subsets(template_offsets)
        if len(unique_subsets) < min_unique_subsets:
            continue

        return markers, template_offsets, template_mask
    return None


def _anchor_candidates(
    h: int,
    w: int,
    template_offsets: tuple[tuple[int, int], ...],
    forbidden_boxes: list[tuple[int, int, int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    th, tw = _template_dims(template_offsets)
    candidates: list[tuple[int, int]] = []
    for ar in range(h - th + 1):
        for ac in range(w - tw + 1):
            cells = _placement_cells(template_offsets, ar, ac)
            if cells & forbidden_cells:
                continue
            box = _bbox_from_cells(cells)
            if any(_boxes_too_close(box, other, gap=2) for other in forbidden_boxes):
                continue
            candidates.append((ar, ac))
    random.shuffle(candidates)
    return candidates


def _expected_output(
    grid: np.ndarray, template_offsets: tuple[tuple[int, int], ...], fragments: list[dict]
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frag in fragments:
        for r, c in _placement_cells(template_offsets, frag["anchor"][0], frag["anchor"][1]):
            out[r, c] = frag["color"]
    return out


def _validate_scene(
    grid: np.ndarray, markers: list[tuple[int, int]], template_offsets: tuple[tuple[int, int], ...], fragments: list[dict]
) -> bool:
    if int(np.count_nonzero(grid == MARKER)) != len(markers):
        return False

    derived_mask = _template_mask_from_markers(markers, grid.shape)
    if _normalize_mask(derived_mask) != template_offsets:
        return False

    if len(components_mask(derived_mask, connectivity=4)) != 1 or not _is_hole_free(derived_mask):
        return False

    for idx in range(len(markers)):
        reduced = markers[:idx] + markers[idx + 1 :]
        if np.array_equal(_template_mask_from_markers(reduced, grid.shape), derived_mask):
            return False

    seen_colors: set[int] = set()
    for frag in fragments:
        if frag["color"] in seen_colors:
            return False
        seen_colors.add(frag["color"])
        if _subset_translations(template_offsets, frag["subset"]) != {(0, 0)}:
            return False
        fragment_cells = _placement_cells(frag["subset"], frag["anchor"][0], frag["anchor"][1])
        if len(fragment_cells) >= len(template_offsets):
            return False
        placements = _find_fragment_placements(grid, template_offsets, fragment_cells)
        if len(placements) != 1:
            return False
        expected_cells = _placement_cells(template_offsets, frag["anchor"][0], frag["anchor"][1])
        if placements[0] != expected_cells:
            return False

    return np.array_equal(main(grid), _expected_output(grid, template_offsets, fragments))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(13, 18)
        w = random.randint(13, 18)
        num_fragments = 2 if random.random() < 0.45 else 1

        marker_scene = _sample_marker_scene(h, w, min_unique_subsets=num_fragments)
        if marker_scene is None:
            continue
        markers, template_offsets, template_mask = marker_scene
        unique_subsets = _unique_anchor_subsets(template_offsets)
        random.shuffle(unique_subsets)
        random.shuffle(markers)

        grid = np.full((h, w), BG, dtype=int)
        for r, c in markers:
            grid[r, c] = MARKER

        marker_box = _bbox_from_cells(set(markers))
        template_box_coords = np.argwhere(template_mask)
        template_box = bbox_coords(template_box_coords)
        forbidden_boxes = [marker_box, template_box]
        forbidden_cells = set(markers)

        anchors = _anchor_candidates(h, w, template_offsets, forbidden_boxes, forbidden_cells)
        if len(anchors) < num_fragments:
            continue

        colors = random.sample(FRAGMENT_COLORS, num_fragments)
        fragments: list[dict] = []
        ok = True
        for color, subset in zip(colors, unique_subsets[:num_fragments]):
            placed = False
            for anchor in anchors:
                cells = _placement_cells(template_offsets, anchor[0], anchor[1])
                box = _bbox_from_cells(cells)
                if any(_boxes_too_close(box, other, gap=2) for other in forbidden_boxes):
                    continue
                if cells & forbidden_cells:
                    continue
                frag_cells = _placement_cells(subset, anchor[0], anchor[1])
                for r, c in frag_cells:
                    grid[r, c] = color
                forbidden_boxes.append(box)
                forbidden_cells |= cells
                fragments.append({"color": color, "subset": subset, "anchor": anchor})
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if _validate_scene(grid, markers, template_offsets, fragments):
            return grid

    raise RuntimeError("failed to generate a valid halo-intersection completion scene")
