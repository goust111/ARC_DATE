from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND_FALLBACK = 0
ANCHOR_COLOR = 2
ALL_COLORS = tuple(range(10))


Coord = tuple[int, int]
Fragment = dict[str, object]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors(r: int, c: int, connectivity: int = 8):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or not bool(mask[r0, c0]):
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[Coord] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors(r, c, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _offset_bbox(offsets: set[Coord] | frozenset[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in offsets]
    cols = [c for _r, c in offsets]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_fragments(g: np.ndarray, bg: int) -> list[Fragment]:
    assert bg != ANCHOR_COLOR, "red is reserved for fragment anchors"
    comps = _components(g != bg, connectivity=8)
    assert len(comps) >= 2, "expected at least two anchored fragments"

    fragments: list[Fragment] = []
    seen_colors: set[int] = set()
    for coords in comps:
        vals = g[coords[:, 0], coords[:, 1]]
        red_mask = vals == ANCHOR_COLOR
        assert int(np.count_nonzero(red_mask)) == 1, "each fragment must contain exactly one red anchor"

        own_colors = set(map(int, vals.tolist())) - {ANCHOR_COLOR}
        assert len(own_colors) == 1, "each fragment must have one non-red body color"
        color = own_colors.pop()
        assert color not in seen_colors, "fragment body colors must be distinct"
        seen_colors.add(color)

        anchor_r, anchor_c = map(int, coords[red_mask][0])
        offsets = frozenset((int(r) - anchor_r, int(c) - anchor_c) for r, c in coords)
        assert (0, 0) in offsets
        assert len(offsets) >= 2, "a fragment must include body cells in addition to its anchor"
        fragments.append(
            {
                "coords": coords,
                "anchor": (anchor_r, anchor_c),
                "color": color,
                "offsets": offsets,
            }
        )

    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden template from anchored fragments and complete every copy.

    Input structure:
    - A dominant background contains several separated multicolor fragments.
    - Each fragment has one red anchor pixel and one body color not used by any other fragment.
    - Every fragment is a proper subset of the same hidden template, aligned by its red anchor.

    Rule:
    - For each fragment, record all occupied offsets relative to its red anchor.
    - Union those offset sets to obtain the complete template mask.
    - At every anchor, paint that complete mask back in the fragment's own color while keeping the anchor red.

    Key insight:
    - The template is never shown whole; it exists only as the transparent overlay union of all red-aligned fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    fragments = _parse_fragments(g, bg)

    template: set[Coord] = set()
    for fragment in fragments:
        template.update(fragment["offsets"])  # type: ignore[arg-type]
    assert (0, 0) in template and len(template) >= 3, "template must contain the anchor and body cells"
    template_fs = frozenset(template)
    for fragment in fragments:
        offsets = fragment["offsets"]  # type: ignore[assignment]
        assert offsets < template_fs, "no input fragment may already be the complete template"

    h, w = g.shape
    desired: dict[Coord, int] = {}
    for fragment in fragments:
        anchor_r, anchor_c = fragment["anchor"]  # type: ignore[misc]
        color = int(fragment["color"])
        for dr, dc in template_fs:
            r = int(anchor_r) + dr
            c = int(anchor_c) + dc
            assert 0 <= r < h and 0 <= c < w, "completed template must fit inside the grid"
            value = ANCHOR_COLOR if (dr, dc) == (0, 0) else color
            old = desired.get((r, c))
            assert old is None or old == value, "completed template copies must not conflict"
            desired[(r, c)] = value

    for r, c in np.argwhere(g != bg):
        rc = (int(r), int(c))
        assert rc in desired, "all input foreground must belong to a completed template copy"
        assert desired[rc] == int(g[r, c]), "completion must preserve existing fragment pixels"

    out = g.copy()
    for (r, c), value in desired.items():
        out[r, c] = value
    return out


# ------------------------------ generator ------------------------------

GRID_SIDE_RANGE = (16, 30)
TEMPLATE_H_RANGE = (3, 5)
TEMPLATE_W_RANGE = (3, 6)
TEMPLATE_AREA_RANGE = (7, 11)
FRAGMENT_COUNT_RANGE = (3, 5)
GENERATION_ATTEMPTS = 500
TEMPLATE_ATTEMPTS = 200
SUBSET_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 400
PLACEMENT_PADDING = 1


def _template_mask_attempt() -> tuple[frozenset[Coord], Coord] | None:
    h = random.randint(*TEMPLATE_H_RANGE)
    w = random.randint(*TEMPLATE_W_RANGE)
    max_area = min(TEMPLATE_AREA_RANGE[1], h * w - 1)
    min_area = min(max(TEMPLATE_AREA_RANGE[0], 6), max_area)
    if min_area > max_area:
        return None

    anchor = (random.randint(1, h - 2), random.randint(1, w - 2))
    target_area = random.randint(min_area, max_area)
    cells: set[Coord] = {anchor}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors(anchor[0], anchor[1], 8)
        if 0 <= nr < h and 0 <= nc < w
    }
    while len(cells) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors(cell[0], cell[1], 8):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != target_area:
        return None
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    if not (0 in rows and h - 1 in rows and 0 in cols and w - 1 in cols):
        return None
    if not (any(r < anchor[0] for r, _c in cells) and any(r > anchor[0] for r, _c in cells)):
        return None
    if not (any(c < anchor[1] for _r, c in cells) and any(c > anchor[1] for _r, c in cells)):
        return None

    offsets = frozenset((r - anchor[0], c - anchor[1]) for r, c in cells)
    if len(offsets) == h * w:
        return None
    return offsets, (0, 0)


def _make_template() -> frozenset[Coord] | None:
    for _ in range(TEMPLATE_ATTEMPTS):
        made = _template_mask_attempt()
        if made is not None:
            offsets, _anchor = made
            return offsets
    return None


def _connected_subset(template: frozenset[Coord], size: int) -> frozenset[Coord] | None:
    subset: set[Coord] = {(0, 0)}
    frontier = {
        nb
        for nb in template
        if nb != (0, 0) and any(max(abs(nb[0] - r), abs(nb[1] - c)) == 1 for r, c in subset)
    }
    while len(subset) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        subset.add(cell)
        for nb in template:
            if nb not in subset and any(max(abs(nb[0] - r), abs(nb[1] - c)) == 1 for r, c in subset):
                frontier.add(nb)
    return frozenset(subset)


def _anchor_bbox_position(offsets: frozenset[Coord]) -> Coord:
    r0, c0, _r1, _c1 = _offset_bbox(offsets)
    return -r0, -c0


def _subsets_are_semantic(template: frozenset[Coord], subsets: list[frozenset[Coord]]) -> bool:
    if len(set(subsets)) != len(subsets):
        return False
    if set().union(*map(set, subsets)) != set(template):
        return False
    if any(not (subset < template) for subset in subsets):
        return False
    counts = Counter(offset for subset in subsets for offset in subset)
    body_offsets = [offset for offset in template if offset != (0, 0)]
    if not any(counts[offset] == 1 for offset in body_offsets):
        return False
    if not any(counts[offset] >= 2 for offset in body_offsets):
        return False
    if not any(len(template) - len(subset) >= 2 for subset in subsets):
        return False
    if len({_anchor_bbox_position(subset) for subset in subsets}) < 2:
        return False
    return True


def _make_fragment_subsets(template: frozenset[Coord], count: int) -> list[frozenset[Coord]] | None:
    area = len(template)
    if area < 4:
        return None
    for _ in range(SUBSET_ATTEMPTS):
        subsets: list[frozenset[Coord]] = []
        for i in range(count):
            if i == 0:
                lo = max(3, area - 3)
                hi = area - 1
            elif i == 1:
                lo = 3
                hi = max(3, min(area - 2, area // 2 + 1))
            else:
                lo = 3
                hi = area - 1
            size = random.randint(min(lo, area - 1), min(hi, area - 1))
            subset = _connected_subset(template, size)
            if subset is None:
                break
            subsets.append(subset)
        if len(subsets) == count and _subsets_are_semantic(template, subsets):
            return subsets
    return None


def _dilate_cells(cells: set[Coord], shape: tuple[int, int], padding: int) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_anchors(shape: tuple[int, int], template: frozenset[Coord], count: int) -> list[Coord] | None:
    h, w = shape
    min_r, min_c, max_r, max_c = _offset_bbox(template)
    r_low, r_high = -min_r, h - 1 - max_r
    c_low, c_high = -min_c, w - 1 - max_c
    if r_low > r_high or c_low > c_high:
        return None

    occupied: set[Coord] = set()
    anchors: list[Coord] = []
    for _ in range(count):
        found: Coord | None = None
        for _attempt in range(PLACEMENT_ATTEMPTS):
            anchor = (random.randint(r_low, r_high), random.randint(c_low, c_high))
            footprint = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in template}
            if footprint & occupied:
                continue
            found = anchor
            occupied |= _dilate_cells(footprint, shape, PLACEMENT_PADDING)
            break
        if found is None:
            return None
        anchors.append(found)
    return anchors


def _render(
    shape: tuple[int, int],
    bg: int,
    template: frozenset[Coord],
    subsets: list[frozenset[Coord]],
    anchors: list[Coord],
    colors: list[int],
    *,
    complete: bool,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for subset, anchor, color in zip(subsets, anchors, colors):
        offsets = template if complete else subset
        for dr, dc in offsets:
            r, c = anchor[0] + dr, anchor[1] + dc
            out[r, c] = ANCHOR_COLOR if (dr, dc) == (0, 0) else int(color)
    return out


def _semantic_generation_check(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    template: frozenset[Coord],
    subsets: list[frozenset[Coord]],
    anchors: list[Coord],
    colors: list[int],
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if _most_common_color(inp) != bg:
        return False
    if int(np.count_nonzero(inp == ANCHOR_COLOR)) != len(subsets):
        return False
    if int(np.count_nonzero(expected != inp)) < len(subsets):
        return False
    if not _subsets_are_semantic(template, subsets):
        return False
    if len(set(colors)) != len(colors):
        return False
    if len(set(anchors)) != len(anchors):
        return False
    if not np.array_equal(_render(inp.shape, bg, template, subsets, anchors, colors, complete=True), expected):
        return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    template = _make_template()
    if template is None:
        return None

    color_pool = [c for c in ALL_COLORS if c != ANCHOR_COLOR]
    bg = random.choice(color_pool)
    body_pool = [c for c in ALL_COLORS if c not in (bg, ANCHOR_COLOR)]
    fragment_count = random.randint(*FRAGMENT_COUNT_RANGE)
    if len(body_pool) < fragment_count:
        return None
    colors = random.sample(body_pool, fragment_count)

    subsets = _make_fragment_subsets(template, fragment_count)
    if subsets is None:
        return None

    min_r, min_c, max_r, max_c = _offset_bbox(template)
    template_h = max_r - min_r + 1
    template_w = max_c - min_c + 1
    min_h = max(GRID_SIDE_RANGE[0], template_h + 2 * PLACEMENT_PADDING + 4)
    min_w = max(GRID_SIDE_RANGE[0], template_w + 2 * PLACEMENT_PADDING + 4)
    if min_h > GRID_SIDE_RANGE[1] or min_w > GRID_SIDE_RANGE[1]:
        return None
    shape = (
        random.randint(min_h, GRID_SIDE_RANGE[1]),
        random.randint(min_w, GRID_SIDE_RANGE[1]),
    )
    anchors = _place_anchors(shape, template, fragment_count)
    if anchors is None:
        return None

    order = list(range(fragment_count))
    random.shuffle(order)
    subsets = [subsets[i] for i in order]
    anchors = [anchors[i] for i in order]
    colors = [colors[i] for i in order]

    inp = _render(shape, bg, template, subsets, anchors, colors, complete=False)
    expected = _render(shape, bg, template, subsets, anchors, colors, complete=True)
    if not _semantic_generation_check(inp, expected, bg, template, subsets, anchors, colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-template input")
