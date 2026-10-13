from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


Coord = tuple[int, int]

MAX_SIDE = 30
GEN_RETRIES = 500
TEMPLATE_RETRIES = 200
FRAGMENT_RETRIES = 200
PLACEMENT_RETRIES = 200


def get_bg(grid: np.ndarray) -> int:
    """Most common color, matching arc_utils' background convention."""
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components as coord arrays, following arc_utils' (r,c) convention."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_from_offsets(offsets: Iterable[Coord]) -> tuple[int, int, int, int]:
    pts = list(offsets)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _offset_component_count(offsets: Iterable[Coord]) -> int:
    cells = set(offsets)
    if not cells:
        return 0
    seen: set[Coord] = set()
    count = 0
    for cell in cells:
        if cell in seen:
            continue
        count += 1
        stack = [cell]
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            for nxt in neighbors4(r, c):
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return count


def _shape_degrees(cells: set[Coord]) -> dict[Coord, int]:
    return {cell: sum((nbr in cells) for nbr in neighbors4(*cell)) for cell in cells}


def _find_anchor_candidates(grid: np.ndarray, bg: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    candidates: list[int] = []
    for color in colors:
        comps = components_mask(g == int(color))
        if not comps or any(comp.shape[0] != 1 for comp in comps):
            continue
        attached_colors: list[int] = []
        ok = True
        for rr, cc in np.argwhere(g == int(color)):
            neighbor_colors = {
                int(g[nr, nc])
                for nr, nc in neighbors4(int(rr), int(cc))
                if 0 <= nr < g.shape[0]
                and 0 <= nc < g.shape[1]
                and int(g[nr, nc]) not in (int(bg), int(color))
            }
            if len(neighbor_colors) != 1:
                ok = False
                break
            attached_colors.append(next(iter(neighbor_colors)))
        if not ok:
            continue
        other_colors = set(colors) - {int(color)}
        if set(attached_colors) == other_colors and len(attached_colors) == len(other_colors):
            candidates.append(int(color))
    return sorted(candidates)


def _extract_fragments(grid: np.ndarray, bg: int, anchor_color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fg_colors = {int(c) for c in np.unique(g) if int(c) not in (int(bg), int(anchor_color))}
    anchors_by_color: dict[int, Coord] = {}
    for rr, cc in np.argwhere(g == int(anchor_color)):
        rr = int(rr)
        cc = int(cc)
        neighbor_colors = {
            int(g[nr, nc])
            for nr, nc in neighbors4(rr, cc)
            if 0 <= nr < g.shape[0]
            and 0 <= nc < g.shape[1]
            and int(g[nr, nc]) not in (int(bg), int(anchor_color))
        }
        assert len(neighbor_colors) == 1, "each anchor pixel must attach to exactly one fragment color"
        color = next(iter(neighbor_colors))
        assert color not in anchors_by_color, "each fragment color must have exactly one anchor"
        anchors_by_color[color] = (rr, cc)
    assert set(anchors_by_color) == fg_colors, "every fragment color must have exactly one anchor"

    fragments: list[dict] = []
    for color in sorted(fg_colors):
        anchor = anchors_by_color[color]
        coords = [tuple(map(int, xy)) for xy in np.argwhere(g == int(color))]
        offsets = {(0, 0)}
        for rr, cc in coords:
            offsets.add((int(rr - anchor[0]), int(cc - anchor[1])))
        fragments.append(
            {
                "color": int(color),
                "anchor": anchor,
                "coords": frozenset(coords),
                "visible_offsets": frozenset(offsets),
            }
        )
    return fragments


def _template_from_fragments(fragments: Iterable[dict]) -> frozenset[Coord]:
    template: set[Coord] = set()
    for fragment in fragments:
        template.update(fragment["visible_offsets"])
    assert (0, 0) in template, "anchor position must belong to the template"
    return frozenset(template)


def _render_completed(shape: tuple[int, int], bg: int, anchor_color: int, fragments: Iterable[dict], template: Iterable[Coord]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    for fragment in fragments:
        ar, ac = fragment["anchor"]
        color = int(fragment["color"])
        for dr, dc in template:
            rr = int(ar + dr)
            cc = int(ac + dc)
            assert 0 <= rr < h and 0 <= cc < w
            paint = int(anchor_color) if (dr, dc) == (0, 0) else color
            prev = int(out[rr, cc])
            assert prev in (int(bg), paint), "completed fragments must not overlap with conflicting colors"
            out[rr, cc] = paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden template by aligning fragments on their shared anchor pixel, then complete every fragment with that template.

    Input structure:
    - A dominant background contains several fragment colors, one fragment per color, and one shared anchor color.
    - Every anchor-colored pixel is isolated and 4-adjacent to exactly one fragment color.
    - Each fragment color may be split into multiple pieces, but together with its anchor it is a translated partial copy of one common template.

    Rule:
    - Identify the unique anchor color whose pixels are all single-cell markers, each attached to a different fragment color.
    - For every fragment, treat its anchor pixel as the same template position, translate all fragments to that common origin, and take the union of their occupied template cells.
    - Translate that hidden union mask back onto every fragment; paint all non-anchor template cells in the fragment's own color and keep each anchor cell in the shared anchor color.

    Key insight:
    - The complete shape never appears in the input; it exists only as the union of all fragments after aligning them by their anchor pixels rather than by their bboxes or absolute positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    anchor_candidates = _find_anchor_candidates(g, bg)
    assert len(anchor_candidates) == 1, "anchor color must be uniquely identifiable"
    anchor_color = int(anchor_candidates[0])

    fragments = _extract_fragments(g, bg, anchor_color)
    assert len(fragments) >= 2, "expected multiple fragments"
    template = _template_from_fragments(fragments)

    for fragment in fragments:
        assert fragment["visible_offsets"] < template, "no input fragment may already be complete"

    return _render_completed(g.shape, bg, anchor_color, fragments, template)


def _random_template(rng: np.random.Generator) -> frozenset[Coord] | None:
    for _ in range(TEMPLATE_RETRIES):
        h = int(rng.integers(4, 8))
        w = int(rng.integers(4, 8))
        max_area = min(h * w - 1, 16)
        min_area = max(8, (h * w + 1) // 2)
        if min_area > max_area:
            continue
        target = int(rng.integers(min_area, max_area + 1))

        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < target:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in neighbors4(r, c)
                    if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in cells
                }
            )
            if not frontier:
                break
            pick = frontier[int(rng.integers(0, len(frontier)))]
            cells.add(pick)
            if len(cells) < target and rng.random() < 0.18:
                extras = [
                    nbr
                    for nbr in frontier
                    if nbr not in cells and sum((nn in cells) for nn in neighbors4(*nbr)) >= 2
                ]
                if extras:
                    cells.add(extras[int(rng.integers(0, len(extras)))])
            if len(cells) > target:
                cells = set(sorted(cells)[:target])

        if len(cells) != target:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) < 3 or len(cols) < 3:
            continue
        r0, c0, r1, c1 = _bbox_from_offsets(cells)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue

        norm = {(r - r0, c - c0) for r, c in cells}
        degrees = _shape_degrees(norm)
        anchor_candidates = [cell for cell, deg in degrees.items() if 2 <= deg <= 3]
        if not anchor_candidates:
            anchor_candidates = [cell for cell, deg in degrees.items() if deg >= 1]
        if not anchor_candidates:
            continue

        anchor = anchor_candidates[int(rng.integers(0, len(anchor_candidates)))]
        offsets = frozenset((r - anchor[0], c - anchor[1]) for r, c in norm)
        if len(offsets) < 8:
            continue
        temp_bbox = _bbox_from_offsets(offsets)
        area = (temp_bbox[2] - temp_bbox[0] + 1) * (temp_bbox[3] - temp_bbox[1] + 1)
        if len(offsets) >= area:
            continue
        return offsets
    return None


def _random_fragments(
    template: frozenset[Coord],
    rng: np.random.Generator,
    n_fragments: int,
    require_disconnected: bool,
) -> list[frozenset[Coord]] | None:
    non_anchor = [off for off in sorted(template) if off != (0, 0)]
    anchor_neighbors = [off for off in non_anchor if abs(off[0]) + abs(off[1]) == 1]
    if len(non_anchor) < n_fragments + 2 or not anchor_neighbors:
        return None

    template_bbox = _bbox_from_offsets(template)
    for _ in range(FRAGMENT_RETRIES):
        perm = list(rng.permutation(len(non_anchor)))
        unique_offsets = [non_anchor[idx] for idx in perm[:n_fragments]]
        unique_set = set(unique_offsets)
        shared_pool = [off for off in non_anchor if off not in unique_set]
        if not shared_pool:
            continue

        fragments = [set([unique_offsets[i]]) for i in range(n_fragments)]
        for frag in fragments:
            if not any(off in anchor_neighbors for off in frag):
                frag.add(anchor_neighbors[int(rng.integers(0, len(anchor_neighbors)))])
            while len(frag) < 2:
                frag.add(shared_pool[int(rng.integers(0, len(shared_pool)))])

        for off in shared_pool:
            owners_n = int(rng.integers(1, min(3, n_fragments) + 1))
            owners = rng.choice(n_fragments, size=owners_n, replace=False)
            for idx in owners:
                fragments[int(idx)].add(off)

        for idx, frag in enumerate(fragments):
            extra_budget = int(rng.integers(0, min(3, len(shared_pool)) + 1))
            if extra_budget == 0:
                continue
            picks = rng.choice(len(shared_pool), size=extra_budget, replace=False)
            for pick in np.atleast_1d(picks):
                frag.add(shared_pool[int(pick)])

        coverage = {off: 0 for off in non_anchor}
        for frag in fragments:
            for off in frag:
                coverage[off] += 1
        if any(count == 0 for count in coverage.values()):
            continue
        if any(len(frag) >= len(non_anchor) for frag in fragments):
            continue
        if any(not any(coverage[off] == 1 for off in frag) for frag in fragments):
            continue

        bbox_signatures = {_bbox_from_offsets(frag | {(0, 0)}) for frag in fragments}
        if len(bbox_signatures) < 2:
            continue
        shrunken_sides: set[str] = set()
        for bbox in bbox_signatures:
            if bbox[0] > template_bbox[0]:
                shrunken_sides.add("top")
            if bbox[1] > template_bbox[1]:
                shrunken_sides.add("left")
            if bbox[2] < template_bbox[2]:
                shrunken_sides.add("bottom")
            if bbox[3] < template_bbox[3]:
                shrunken_sides.add("right")
        if len(shrunken_sides) < 2:
            continue

        if require_disconnected and not any(_offset_component_count(frag) >= 2 for frag in fragments):
            continue

        return [frozenset(frag | {(0, 0)}) for frag in fragments]
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_fragments(
    template: frozenset[Coord],
    n_fragments: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[Coord]] | None:
    min_r, min_c, max_r, max_c = _bbox_from_offsets(template)
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1
    min_h = max(14, box_h + 4)
    min_w = max(14, box_w + 4)
    if min_h > MAX_SIDE or min_w > MAX_SIDE:
        return None

    for _ in range(PLACEMENT_RETRIES):
        h = int(rng.integers(min_h, MAX_SIDE + 1))
        w = int(rng.integers(min_w, MAX_SIDE + 1))
        outer_margin = int(rng.integers(0, 2))
        gap = int(rng.integers(1, 3))

        r_low = outer_margin - min_r
        r_high = h - 1 - outer_margin - max_r
        c_low = outer_margin - min_c
        c_high = w - 1 - outer_margin - max_c
        if r_low > r_high or c_low > c_high:
            continue

        anchors: list[Coord] = []
        blocked: list[tuple[int, int, int, int]] = []
        ok = True
        for _frag in range(n_fragments):
            placed = False
            for _ in range(200):
                ar = int(rng.integers(r_low, r_high + 1))
                ac = int(rng.integers(c_low, c_high + 1))
                bbox = (ar + min_r, ac + min_c, ar + max_r, ac + max_c)
                padded = (
                    max(0, bbox[0] - gap),
                    max(0, bbox[1] - gap),
                    min(h - 1, bbox[2] + gap),
                    min(w - 1, bbox[3] + gap),
                )
                if any(_boxes_overlap(padded, other) for other in blocked):
                    continue
                anchors.append((ar, ac))
                blocked.append(padded)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return (h, w), anchors
    return None


def _build_case(rng: np.random.Generator) -> dict | None:
    template = _random_template(rng)
    if template is None:
        return None

    non_anchor_count = len(template) - 1
    max_fragments = min(5, non_anchor_count - 2)
    if max_fragments < 3:
        return None
    n_fragments = int(rng.integers(3, max_fragments + 1))
    require_disconnected = bool(rng.integers(0, 2))

    visible_sets = _random_fragments(template, rng, n_fragments, require_disconnected)
    if visible_sets is None:
        return None

    placement = _place_fragments(template, n_fragments, rng)
    if placement is None:
        return None
    shape, anchors = placement

    palette = list(range(10))
    bg = int(rng.choice(palette))
    remaining = [color for color in palette if color != bg]
    anchor_color = int(rng.choice(remaining))
    remaining = [color for color in remaining if color != anchor_color]
    frag_colors = [int(color) for color in rng.choice(remaining, size=n_fragments, replace=False)]

    inp = np.full(shape, bg, dtype=int)
    fragments: list[dict] = []
    for color, anchor, visible in zip(frag_colors, anchors, visible_sets):
        ar, ac = anchor
        inp[ar, ac] = anchor_color
        coords = {(int(ar + dr), int(ac + dc)) for dr, dc in visible if (dr, dc) != (0, 0)}
        for rr, cc in coords:
            inp[rr, cc] = color
        fragments.append(
            {
                "color": int(color),
                "anchor": (int(ar), int(ac)),
                "coords": frozenset(coords),
                "visible_offsets": frozenset(visible),
            }
        )

    out = _render_completed(shape, bg, anchor_color, fragments, template)
    case = {
        "input": inp,
        "output": out,
        "bg": int(bg),
        "anchor_color": int(anchor_color),
        "template": frozenset(template),
        "fragments": fragments,
        "require_disconnected": require_disconnected,
    }
    if _validate_case(case):
        return case
    return None


def _validate_case(case: dict) -> bool:
    inp = np.asarray(case["input"], dtype=int)
    out = np.asarray(case["output"], dtype=int)
    bg = int(case["bg"])
    anchor_color = int(case["anchor_color"])
    template = frozenset(case["template"])
    fragments = case["fragments"]

    assert inp.ndim == 2 and out.ndim == 2
    assert inp.shape == out.shape
    assert max(inp.shape) <= MAX_SIDE
    assert get_bg(inp) == bg
    assert not np.array_equal(inp, out)
    assert np.count_nonzero(inp == anchor_color) == len(fragments)

    anchor_candidates = _find_anchor_candidates(inp, bg)
    assert anchor_candidates == [anchor_color]

    solved_fragments = _extract_fragments(inp, bg, anchor_color)
    by_color = {frag["color"]: frag for frag in solved_fragments}
    assert set(by_color) == {frag["color"] for frag in fragments}
    for frag in fragments:
        got = by_color[int(frag["color"])]
        assert got["anchor"] == frag["anchor"]
        assert got["visible_offsets"] == frag["visible_offsets"]

    recovered_template = _template_from_fragments(solved_fragments)
    assert recovered_template == template

    coverage = {off: 0 for off in template if off != (0, 0)}
    for frag in solved_fragments:
        assert frag["visible_offsets"] < template
        for off in frag["visible_offsets"]:
            if off != (0, 0):
                coverage[off] += 1
    assert all(count >= 1 for count in coverage.values())
    assert all(any(coverage[off] == 1 for off in frag["visible_offsets"] if off != (0, 0)) for frag in solved_fragments)

    largest = max(solved_fragments, key=lambda frag: len(frag["visible_offsets"]))
    assert largest["visible_offsets"] != template

    template_bbox = _bbox_from_offsets(template)
    bbox_signatures = {_bbox_from_offsets(frag["visible_offsets"]) for frag in solved_fragments}
    assert len(bbox_signatures) >= 2
    side_shrink: set[str] = set()
    for bbox in bbox_signatures:
        if bbox[0] > template_bbox[0]:
            side_shrink.add("top")
        if bbox[1] > template_bbox[1]:
            side_shrink.add("left")
        if bbox[2] < template_bbox[2]:
            side_shrink.add("bottom")
        if bbox[3] < template_bbox[3]:
            side_shrink.add("right")
    assert len(side_shrink) >= 2

    if case["require_disconnected"]:
        assert any(_offset_component_count(frag["coords"]) >= 2 for frag in fragments)

    rebuilt = main(inp)
    assert np.array_equal(rebuilt, out)
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        case = _build_case(rng)
        if case is not None:
            return np.asarray(case["input"], dtype=int)
    raise RuntimeError("failed to generate a valid hidden-template-union input")
