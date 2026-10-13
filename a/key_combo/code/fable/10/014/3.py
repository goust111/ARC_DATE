from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
SUPPORT_COLORS = (1, 3, 5, 6, 7, 8, 9)

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_ATTEMPTS = 400


def _find_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _object_data(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for coords in _find_components(grid != BLACK, connectivity=8):
        colors = grid[coords[:, 0], coords[:, 1]]
        anchor_coords = coords[colors == RED]
        assert len(anchor_coords) == 1, "each copy must contain exactly one red anchor"
        anchor = tuple(map(int, anchor_coords[0]))
        rel: dict[tuple[int, int], int] = {}
        for rr, cc in coords:
            rel[(int(rr - anchor[0]), int(cc - anchor[1]))] = int(grid[rr, cc])
        objects.append({"anchor": anchor, "coords": coords, "rel": rel})
    assert len(objects) >= 2, "expected multiple partial copies"
    return objects


def _reconstruct_template(objects: list[dict]) -> dict[tuple[int, int], int]:
    template: dict[tuple[int, int], int] = {}
    for obj in objects:
        for offset, color in obj["rel"].items():
            prev = template.get(offset)
            if prev is None:
                template[offset] = color
            else:
                assert prev == color, "aligned copies disagree on color"
    assert template.get((0, 0)) == RED, "anchor offset must be red"
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every anchored partial copy by crowd-reconstructing the unseen full template.

    Input structure:
    - A black background contains several partial copies of the same multicolor sprite.
    - Each copy keeps exactly one red anchor pixel at the same relative template position.
    - No copy is complete, but all copies share one orientation and their aligned colors agree.

    Rule:
    - Extract each non-black copy and use its red anchor as the local origin.
    - Align all copies at the anchor and take the union of their colored relative pixels to rebuild the full template.
    - For every copy, keep its existing pixels unchanged and fill every missing non-anchor template pixel with yellow.

    Key insight:
    - The full sprite never appears directly; it is only recoverable by aligning all fragments on the red anchor and unioning their evidence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _object_data(g)
    template = _reconstruct_template(objects)
    template_offsets = set(template)

    rel_sets = [set(obj["rel"]) for obj in objects]
    assert any(rel_set != template_offsets for rel_set in rel_sets), "expected no complete copy"
    assert all(rel_set != template_offsets for rel_set in rel_sets), "no copy may be complete"
    assert len({frozenset(rel_set) for rel_set in rel_sets}) == len(rel_sets), "missing patterns must differ"

    out = g.copy()
    h, w = g.shape
    for obj, rel_set in zip(objects, rel_sets):
        ar, ac = obj["anchor"]
        for (dr, dc), color in template.items():
            rr, cc = ar + dr, ac + dc
            assert 0 <= rr < h and 0 <= cc < w, "template placement must stay in bounds"
            if (dr, dc) in rel_set:
                assert int(out[rr, cc]) == color, "existing pixels must already match the template"
            elif color != RED:
                assert int(out[rr, cc]) == BLACK, "missing template cells must be background before filling"
                out[rr, cc] = YELLOW
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_connected_shape(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(NEIGHBORS4)
        cells.add((r + dr, c + dc))
    return _normalize(cells)


def _frontier(template: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = set()
    for r, c in template:
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt not in template:
                out.add(nxt)
    return sorted(out)


def _connected_offsets(offsets: set[tuple[int, int]]) -> bool:
    if not offsets:
        return False
    cells = set(offsets)
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == cells


def _template_bounds(template: dict[tuple[int, int], int]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in template]
    cols = [c for _, c in template]
    return min(rows), min(cols), max(rows), max(cols)


def _placeable(anchor: tuple[int, int], template: dict[tuple[int, int], int], shape: tuple[int, int]) -> bool:
    h, w = shape
    ar, ac = anchor
    for dr, dc in template:
        rr, cc = ar + dr, ac + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
    return True


def _choose_anchor_positions(
    template: dict[tuple[int, int], int], shape: tuple[int, int], count: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    min_r, min_c, max_r, max_c = _template_bounds(template)
    candidates = [
        (r, c)
        for r in range(-min_r, h - max_r)
        for c in range(-min_c, w - max_c)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []
    for ar, ac in candidates:
        r0, c0 = ar + min_r - 2, ac + min_c - 2
        r1, c1 = ar + max_r + 2, ac + max_c + 2
        if any(not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0) for rr0, cc0, rr1, cc1 in reserved):
            continue
        chosen.append((ar, ac))
        reserved.append((r0, c0, r1, c1))
        if len(chosen) == count:
            return chosen
    return None


def _build_template() -> tuple[dict[tuple[int, int], int], list[set[tuple[int, int]]]]:
    copy_count = random.randint(3, 5)
    for _ in range(200):
        backbone_size = random.randint(4, 7)
        backbone = _random_connected_shape(backbone_size)
        frontier = _frontier(backbone)
        if len(frontier) < copy_count + 1:
            continue

        unique_cells = random.sample(frontier, copy_count)
        template_cells = set(backbone) | set(unique_cells)

        # Add 0-3 extra cells that remain connected and appear in a nontrivial subset of copies.
        extras: list[tuple[tuple[int, int], set[int]]] = []
        extra_budget = random.randint(0, min(3, len(frontier) - copy_count))
        extra_candidates = [cell for cell in frontier if cell not in unique_cells]
        random.shuffle(extra_candidates)
        for cell in extra_candidates:
            if len(extras) == extra_budget:
                break
            template_cells.add(cell)
            if not _connected_offsets(template_cells):
                template_cells.remove(cell)
                continue
            subset_size = random.randint(2, copy_count - 1)
            subset = set(random.sample(range(copy_count), subset_size))
            extras.append((cell, subset))

        if not _connected_offsets(template_cells):
            continue

        anchor = random.choice(sorted(backbone))
        rel_cells = {(r - anchor[0], c - anchor[1]) for r, c in template_cells}
        if (0, 0) not in rel_cells:
            continue

        color_pool = random.sample(SUPPORT_COLORS, random.randint(2, min(4, len(SUPPORT_COLORS))))
        template: dict[tuple[int, int], int] = {(0, 0): RED}
        for cell in sorted(rel_cells):
            if cell == (0, 0):
                continue
            template[cell] = random.choice(color_pool)
        if len({template[cell] for cell in template if cell != (0, 0)}) < 2:
            continue

        rel_unique = [(r - anchor[0], c - anchor[1]) for r, c in unique_cells]
        copies: list[set[tuple[int, int]]] = []
        for idx in range(copy_count):
            offsets = {(0, 0)}
            offsets |= {(r - anchor[0], c - anchor[1]) for r, c in backbone if (r, c) != anchor}
            offsets.add(rel_unique[idx])
            for cell, subset in extras:
                if idx in subset:
                    offsets.add((cell[0] - anchor[0], cell[1] - anchor[1]))
            if offsets == set(template):
                break
            if not _connected_offsets(offsets):
                break
            copies.append(offsets)
        if len(copies) != copy_count:
            continue

        # Every copy must be necessary and every template pixel must appear at least once.
        union = set().union(*copies)
        if union != set(template):
            continue
        if len({frozenset(c) for c in copies}) != copy_count:
            continue
        if any(c == set(template) for c in copies):
            continue
        if any(sum(cell in other for j, other in enumerate(copies) if j != i) != 0 for i, cell in enumerate(rel_unique)):
            continue
        return template, copies
    raise RuntimeError("failed to build a nondegenerate template")


def _render_input(template: dict[tuple[int, int], int], copies: list[set[tuple[int, int]]]) -> np.ndarray | None:
    copy_count = len(copies)
    min_r, min_c, max_r, max_c = _template_bounds(template)
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1
    min_h = max(14, copy_count * 3 + box_h + 4)
    min_w = max(14, copy_count * 3 + box_w + 4)
    h = random.randint(min_h, min(30, max(min_h, min_h + 8)))
    w = random.randint(min_w, min(30, max(min_w, min_w + 8)))
    if h > 30 or w > 30:
        return None

    anchors = _choose_anchor_positions(template, (h, w), copy_count)
    if anchors is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    for anchor, offsets in zip(anchors, copies):
        ar, ac = anchor
        for dr, dc in offsets:
            inp[ar + dr, ac + dc] = template[(dr, dc)]
    return inp


def _expected_output(inp: np.ndarray, template: dict[tuple[int, int], int]) -> np.ndarray:
    out = inp.copy()
    objects = _object_data(inp)
    expected_offsets = set(template)
    for obj in objects:
        ar, ac = obj["anchor"]
        for (dr, dc), color in template.items():
            if (dr, dc) in obj["rel"] or color == RED:
                continue
            out[ar + dr, ac + dc] = YELLOW
        assert set(obj["rel"]) < expected_offsets, "generator produced a complete copy"
    return out


def _validate_instance(inp: np.ndarray, template: dict[tuple[int, int], int], copies: list[set[tuple[int, int]]]) -> bool:
    try:
        objects = _object_data(inp)
    except AssertionError:
        return False
    if len(objects) != len(copies):
        return False

    rel_sets = [set(obj["rel"]) for obj in objects]
    if set(map(frozenset, rel_sets)) != set(map(frozenset, copies)):
        return False

    rebuilt = _reconstruct_template(objects)
    if rebuilt != template:
        return False

    union = set().union(*rel_sets)
    if union != set(template):
        return False
    for idx, rel_set in enumerate(rel_sets):
        if rel_set == set(template):
            return False
        if not _connected_offsets(rel_set):
            return False
        unique_contrib = rel_set - set().union(*(rel_sets[j] for j in range(len(rel_sets)) if j != idx))
        if not unique_contrib:
            return False

    expected = _expected_output(inp, template)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        template, copies = _build_template()
        inp = _render_input(template, copies)
        if inp is None:
            continue
        if _validate_instance(inp, template, copies):
            return inp
    raise RuntimeError("failed to generate a valid crowd-template completion input")
