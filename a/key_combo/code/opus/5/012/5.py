from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np


BG = 0
ANCHOR = 5
PALETTE = [1, 2, 3, 4]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
ANCHOR_REL = {(0, 0), (0, 1), (1, 0), (1, 1)}
GEN_ATTEMPTS = 600


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _anchor_components(grid: np.ndarray) -> list[dict]:
    anchors = []
    for coords in _components_mask(grid == ANCHOR):
        if int(coords.shape[0]) != 4:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR):
            continue
        anchors.append(
            {
                "coords": coords,
                "top_left": (int(r0), int(c0)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    anchors.sort(key=lambda item: item["top_left"])
    return anchors


def _component_info(color: int, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return {
        "color": int(color),
        "coords": np.array(coords, dtype=int),
        "area": int(coords.shape[0]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _foreground_components_by_color(grid: np.ndarray) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, ANCHOR)):
        comps = [_component_info(color, coords) for coords in _components_mask(grid == color)]
        comps.sort(key=lambda comp: (-comp["area"], comp["bbox"], comp["color"]))
        out[color] = comps
    return out


def _rounded_centroid(coords: np.ndarray) -> tuple[int, int]:
    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    return _round_half_up(rows.mean()), _round_half_up(cols.mean())


def _chebyshev_distance(a: np.ndarray, b: np.ndarray) -> int:
    diffs = np.abs(a[:, None, :] - b[None, :, :])
    return int(np.max(diffs, axis=2).min())


def _nearest_anchor_index(coords: np.ndarray, anchors: list[dict]) -> tuple[int, list[int]]:
    dists = [_chebyshev_distance(coords, anchor["coords"]) for anchor in anchors]
    best = min(dists)
    winners = [idx for idx, dist in enumerate(dists) if dist == best]
    return winners[0], dists


def _translated_coords(coords: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array(shift, dtype=int)


def _kept_components(grid: np.ndarray) -> tuple[list[dict], dict[int, list[dict]], list[dict]]:
    anchors = _anchor_components(grid)
    by_color = _foreground_components_by_color(grid)
    kept: list[dict] = []
    for color, comps in by_color.items():
        if not comps:
            continue
        comp = dict(comps[0])
        centroid = _rounded_centroid(comp["coords"])
        anchor_idx, dists = _nearest_anchor_index(comp["coords"], anchors)
        comp["centroid"] = centroid
        comp["anchor_idx"] = int(anchor_idx)
        comp["anchor_dists"] = list(map(int, dists))
        comp["shift"] = (
            int(anchors[anchor_idx]["top_left"][0] - centroid[0]),
            int(anchors[anchor_idx]["top_left"][1] - centroid[1]),
        )
        comp["translated"] = _translated_coords(comp["coords"], comp["shift"])
        kept.append(comp)
    kept.sort(key=lambda comp: (comp["anchor_idx"], -comp["area"], comp["color"]))
    return anchors, by_color, kept


def _render_components(
    grid: np.ndarray,
    anchors: list[dict],
    components: list[dict],
    *,
    reverse_overlay: bool = False,
) -> np.ndarray:
    out = np.full_like(grid, BG)
    anchor_mask = np.zeros_like(grid, dtype=bool)
    for anchor in anchors:
        coords = anchor["coords"]
        out[coords[:, 0], coords[:, 1]] = ANCHOR
        anchor_mask[coords[:, 0], coords[:, 1]] = True

    groups: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        groups[int(comp["anchor_idx"])].append(comp)

    for anchor_idx in range(len(anchors)):
        comps = groups.get(anchor_idx, [])
        if reverse_overlay:
            comps.sort(key=lambda comp: (comp["area"], comp["color"], comp["bbox"]))
        else:
            comps.sort(key=lambda comp: (-comp["area"], comp["color"], comp["bbox"]))
        for comp in comps:
            translated = comp.get("translated")
            if translated is None:
                centroid = _rounded_centroid(comp["coords"])
                shift = (
                    int(anchors[anchor_idx]["top_left"][0] - centroid[0]),
                    int(anchors[anchor_idx]["top_left"][1] - centroid[1]),
                )
                translated = _translated_coords(comp["coords"], shift)
            for r, c in translated:
                rr = int(r)
                cc = int(c)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and not anchor_mask[rr, cc]:
                    out[rr, cc] = int(comp["color"])
    return out


def _solve_variant(grid: np.ndarray, *, keep_all: bool, reverse_overlay: bool) -> np.ndarray:
    anchors = _anchor_components(grid)
    by_color = _foreground_components_by_color(grid)
    components: list[dict] = []
    for _color, comps in by_color.items():
        selected = comps if keep_all else comps[:1]
        for comp0 in selected:
            comp = dict(comp0)
            centroid = _rounded_centroid(comp["coords"])
            anchor_idx, dists = _nearest_anchor_index(comp["coords"], anchors)
            comp["centroid"] = centroid
            comp["anchor_idx"] = int(anchor_idx)
            comp["anchor_dists"] = list(map(int, dists))
            comp["shift"] = (
                int(anchors[anchor_idx]["top_left"][0] - centroid[0]),
                int(anchors[anchor_idx]["top_left"][1] - centroid[1]),
            )
            comp["translated"] = _translated_coords(comp["coords"], comp["shift"])
            components.append(comp)
    return _render_components(grid, anchors, components, reverse_overlay=reverse_overlay)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove same-color distractors, snap the surviving shapes onto the nearest gray anchors, then overlay by area.

    Input structure:
    - The grid is mostly black background.
    - Two or three gray 2x2 solid anchor blocks appear apart from each other.
    - Three or four non-gray foreground colors each appear in exactly two 4-connected components:
      one larger irregular component and one tiny distractor component.

    Rule:
    - For each non-black, non-gray color, keep only its unique largest 4-connected component and erase the smaller component.
    - For each kept component, measure Chebyshev distance from any of its pixels to each gray anchor block and assign it to the unique nearest anchor.
    - Clear the kept components from their original locations, then translate each one so its rounded centroid lands on the anchor block's top-left cell.
    - For every anchor, draw its assigned components from largest area to smallest area so later smaller ones overwrite earlier larger ones, while gray anchor cells stay gray.

    Key insight:
    - The tiny same-color pieces are deliberate distractors: they must be removed before nearest-anchor assignment, and shared anchors reveal the required large-first / small-on-top overlay order.
    """
    g = np.asarray(grid, dtype=int)
    anchors, _by_color, kept = _kept_components(g)
    return _render_components(g, anchors, kept, reverse_overlay=False)


def _normalize_shape(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    center = np.array(_rounded_centroid(arr), dtype=int)
    arr = arr - center
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
    return arr


def _random_polyomino(area: int) -> np.ndarray:
    for _ in range(300):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in DIRS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < area and frontier:
            nxt = random.choice(list(frontier))
            frontier.remove(nxt)
            cells.add(nxt)
            for dr, dc in DIRS4:
                nr, nc = nxt[0] + dr, nxt[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        norm = _normalize_shape(list(cells))
        r0, c0, r1, c1 = _bbox_coords(norm)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height == 1 or width == 1:
            continue
        if height * width == area:
            continue
        if max(abs(int(v)) for v in norm[:, 0]) > 3 or max(abs(int(v)) for v in norm[:, 1]) > 3:
            continue
        return norm
    raise RuntimeError("failed to create polyomino")


def _anchor_distance_between(top_left_a: tuple[int, int], top_left_b: tuple[int, int]) -> int:
    a = np.array([(top_left_a[0] + dr, top_left_a[1] + dc) for dr in (0, 1) for dc in (0, 1)], dtype=int)
    b = np.array([(top_left_b[0] + dr, top_left_b[1] + dc) for dr in (0, 1) for dc in (0, 1)], dtype=int)
    return _chebyshev_distance(a, b)


def _sample_anchors(h: int, w: int, n_anchors: int) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(3, h - 4) for c in range(3, w - 4)]
    for _ in range(200):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            if all(_anchor_distance_between((r, c), other) >= 5 for other in chosen):
                chosen.append((r, c))
                if len(chosen) == n_anchors:
                    return chosen
    raise RuntimeError("failed to place anchors")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _output_shift_for_anchor(coords: np.ndarray, anchor_top_left: tuple[int, int]) -> tuple[int, int]:
    centroid = _rounded_centroid(coords)
    return int(anchor_top_left[0] - centroid[0]), int(anchor_top_left[1] - centroid[1])


def _candidate_origins(
    norm_coords: np.ndarray,
    blocked: np.ndarray,
    anchors: list[dict],
    anchor_idx: int,
    *,
    min_movement: int = 3,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    candidates = []
    target = anchors[anchor_idx]["top_left"]
    for r0 in range(h):
        for c0 in range(w):
            coords = norm_coords + np.array([r0, c0], dtype=int)
            rows = coords[:, 0]
            cols = coords[:, 1]
            if int(rows.min()) < 0 or int(cols.min()) < 0 or int(rows.max()) >= h or int(cols.max()) >= w:
                continue
            if bool(blocked[rows, cols].any()):
                continue
            best_idx, dists = _nearest_anchor_index(coords, anchors)
            best = min(dists)
            if best_idx != anchor_idx or dists.count(best) != 1:
                continue
            if abs(int(r0) - target[0]) + abs(int(c0) - target[1]) < min_movement:
                continue
            candidates.append((int(r0), int(c0)))
    random.shuffle(candidates)
    return candidates


def _place_cells(
    grid: np.ndarray,
    mask_occupied: np.ndarray,
    color: int,
    coords: np.ndarray,
) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)
    mask_occupied[coords[:, 0], coords[:, 1]] = True


def _translated_overlap_visible(shape_a: np.ndarray, shape_b: np.ndarray) -> bool:
    shared = {tuple(map(int, rc)) for rc in shape_a} & {tuple(map(int, rc)) for rc in shape_b}
    return any(cell not in ANCHOR_REL for cell in shared)


def _sample_shapes_for_assignments(
    areas: list[int],
    assignments: list[int],
) -> list[np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, anchor_idx in enumerate(assignments):
        groups[int(anchor_idx)].append(idx)

    shapes: list[np.ndarray | None] = [None] * len(areas)
    for anchor_idx, indices in groups.items():
        if len(indices) >= 2:
            success = False
            for _ in range(200):
                local = [_random_polyomino(areas[idx]) for idx in indices]
                overlap_found = False
                for i in range(len(local)):
                    for j in range(i + 1, len(local)):
                        if _translated_overlap_visible(local[i], local[j]):
                            overlap_found = True
                if overlap_found:
                    for idx, shape in zip(indices, local):
                        shapes[idx] = shape
                    success = True
                    break
            if not success:
                raise RuntimeError(f"failed to create overlapping shapes for anchor {anchor_idx}")
        else:
            shapes[indices[0]] = _random_polyomino(areas[indices[0]])

    return [np.array(shape, dtype=int) for shape in shapes]


def _assignment_plan(n_anchors: int, n_colors: int) -> list[int]:
    if n_anchors == 2:
        if n_colors == 3:
            counts = [2, 1]
        else:
            counts = [2, 2]
    else:
        if n_colors == 3:
            counts = [2, 1, 0]
        else:
            counts = [2, 1, 1]
    random.shuffle(counts)
    assignments = []
    for anchor_idx, count in enumerate(counts):
        assignments.extend([anchor_idx] * count)
    random.shuffle(assignments)
    return assignments


def _union_best_anchor(coords_list: list[np.ndarray], anchors: list[dict]) -> int:
    merged = np.concatenate(coords_list, axis=0)
    idx, dists = _nearest_anchor_index(merged, anchors)
    best = min(dists)
    if dists.count(best) != 1:
        return -1
    return int(idx)


def _analyze_generated_grid(grid: np.ndarray) -> dict:
    anchors, by_color, kept = _kept_components(grid)
    analysis = {
        "anchors": anchors,
        "by_color": by_color,
        "kept": kept,
        "correct": main(grid),
        "keep_all": _solve_variant(grid, keep_all=True, reverse_overlay=False),
        "reverse_overlay": _solve_variant(grid, keep_all=False, reverse_overlay=True),
    }
    return analysis


def _validate_instance(grid: np.ndarray) -> None:
    analysis = _analyze_generated_grid(grid)
    anchors = analysis["anchors"]
    by_color = analysis["by_color"]
    kept = analysis["kept"]
    correct = analysis["correct"]
    keep_all = analysis["keep_all"]
    reverse_overlay = analysis["reverse_overlay"]

    assert 2 <= len(anchors) <= 3
    assert 3 <= len(by_color) <= 4

    misleading_color_found = False
    anchor_groups: dict[int, list[dict]] = defaultdict(list)
    for color, comps in by_color.items():
        assert len(comps) == 2
        areas = sorted(comp["area"] for comp in comps)
        assert 1 <= areas[0] <= 2
        assert 4 <= areas[1] <= 8
        assert areas[0] < areas[1]

        real = comps[0]
        small = comps[1]
        real_best, real_dists = _nearest_anchor_index(real["coords"], anchors)
        small_best, small_dists = _nearest_anchor_index(small["coords"], anchors)
        assert real_dists.count(min(real_dists)) == 1
        assert small_dists.count(min(small_dists)) == 1
        union_best = _union_best_anchor([real["coords"], small["coords"]], anchors)
        if union_best != -1 and union_best != real_best and small_best != real_best:
            misleading_color_found = True
        anchor_groups[int(real_best)].append(real)

    assert misleading_color_found
    assert not np.array_equal(correct, keep_all)
    assert not np.array_equal(correct, reverse_overlay)

    overlap_found = False
    anchor_mask = np.zeros_like(grid, dtype=bool)
    for anchor in anchors:
        anchor_mask[anchor["coords"][:, 0], anchor["coords"][:, 1]] = True
    for anchor_idx, comps in anchor_groups.items():
        if len(comps) < 2:
            continue
        translated_sets = []
        for comp in comps:
            shift = _output_shift_for_anchor(comp["coords"], anchors[anchor_idx]["top_left"])
            translated_sets.append(_translated_coords(comp["coords"], shift))
        for i in range(len(translated_sets)):
            for j in range(i + 1, len(translated_sets)):
                sa = {tuple(map(int, rc)) for rc in translated_sets[i]}
                sb = {tuple(map(int, rc)) for rc in translated_sets[j]}
                visible = [cell for cell in sa & sb if not anchor_mask[cell[0], cell[1]]]
                if visible:
                    overlap_found = True
    assert overlap_found
    assert not np.array_equal(grid, correct)
    assert int(np.count_nonzero(correct)) > int(np.count_nonzero(anchor_mask))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_anchors = random.choice([2, 3])
        if n_anchors == 3:
            h = random.randint(17, 19)
            w = random.randint(17, 19)
        else:
            h = random.randint(15, 19)
            w = random.randint(15, 19)
        n_colors = random.choice([3, 4])
        colors = random.sample(PALETTE, n_colors)
        assignments = _assignment_plan(n_anchors, n_colors)
        anchors_top_left = _sample_anchors(h, w, n_anchors)

        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        for r0, c0 in anchors_top_left:
            grid[r0 : r0 + 2, c0 : c0 + 2] = ANCHOR
            occupied[r0 : r0 + 2, c0 : c0 + 2] = True
        blocked = _dilate8(occupied)

        anchor_infos = _anchor_components(grid)
        areas = random.sample([4, 5, 6, 7, 8], n_colors)
        shapes = _sample_shapes_for_assignments(areas, assignments)

        real_coords_by_color: dict[int, np.ndarray] = {}
        place_order = sorted(range(n_colors), key=lambda idx: (-areas[idx], assignments[idx], colors[idx]))
        failed = False
        for idx in place_order:
            candidates = _candidate_origins(shapes[idx], blocked, anchor_infos, assignments[idx], min_movement=4)
            if not candidates:
                failed = True
                break
            origin = random.choice(candidates[: min(20, len(candidates))])
            coords = shapes[idx] + np.array(origin, dtype=int)
            _place_cells(grid, occupied, colors[idx], coords)
            blocked = _dilate8(occupied)
            real_coords_by_color[colors[idx]] = coords
        if failed:
            continue

        misleading_color = random.choice(colors)
        color_to_anchor = {colors[idx]: assignments[idx] for idx in range(n_colors)}
        for color in colors:
            decoy_choices = [idx for idx in range(n_anchors) if idx != color_to_anchor[color]]
            random.shuffle(decoy_choices)
            if not decoy_choices:
                failed = True
                break
            size = random.choice([1, 2])
            distractor_shapes = [np.array([[0, 0]], dtype=int)]
            if size == 2:
                distractor_shapes = [
                    np.array([[0, 0], [1, 0]], dtype=int),
                    np.array([[0, 0], [0, 1]], dtype=int),
                ]
            placed = False
            for decoy_anchor in decoy_choices:
                shape_options = distractor_shapes[:]
                random.shuffle(shape_options)
                for shape in shape_options:
                    hshape, wshape = shape[:, 0].max() + 1, shape[:, 1].max() + 1
                    origins = []
                    for r0 in range(h - hshape + 1):
                        for c0 in range(w - wshape + 1):
                            coords = shape + np.array([r0, c0], dtype=int)
                            rows = coords[:, 0]
                            cols = coords[:, 1]
                            if bool(blocked[rows, cols].any()):
                                continue
                            best_idx, dists = _nearest_anchor_index(coords, anchor_infos)
                            best = min(dists)
                            if best_idx != decoy_anchor or dists.count(best) != 1:
                                continue
                            if color == misleading_color:
                                union_best = _union_best_anchor([real_coords_by_color[color], coords], anchor_infos)
                                if union_best != decoy_anchor:
                                    continue
                            origins.append((int(r0), int(c0)))
                    random.shuffle(origins)
                    if origins:
                        coords = shape + np.array(origins[0], dtype=int)
                        _place_cells(grid, occupied, color, coords)
                        blocked = _dilate8(occupied)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                failed = True
                break
        if failed:
            continue

        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
