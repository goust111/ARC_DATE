from __future__ import annotations

from collections import deque
import random

import numpy as np

BBox = tuple[int, int, int, int]

_COLOR_COUNT = 10
_MAX_ATTEMPTS = 300
_LAYOUTS = {
    1: [(1, 1), (1, 2), (2, 1), (2, 2)],
    2: [(1, 2), (2, 1), (2, 2)],
    3: [(1, 3), (3, 1), (2, 2)],
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _strictly_contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _centered_bbox(outer_h: int, outer_w: int, inner_h: int, inner_w: int) -> BBox:
    r0 = (outer_h - inner_h) // 2
    c0 = (outer_w - inner_w) // 2
    return int(r0), int(c0), int(r0 + inner_h - 1), int(c0 + inner_w - 1)


def _draw_ring(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _fill_gap(out: np.ndarray, ring_bbox: BBox, inner_bbox: BBox, core_color: int, ring_color: int) -> None:
    r0, c0, r1, c1 = ring_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    top = int(r0 + 1)
    bottom = int(r1 - 1)
    left = int(c0 + 1)
    right = int(c1 - 1)
    for r in range(top, bottom + 1):
        fill_color = int(core_color if (r - top) % 2 == 0 else ring_color)
        for c in range(left, right + 1):
            if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                continue
            out[r, c] = fill_color


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            min_r = max_r = int(r)
            min_c = max_c = int(c)

            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                min_r = min(min_r, int(cr))
                max_r = max(max_r, int(cr))
                min_c = min(min_c, int(cc))
                max_c = max(max_c, int(cc))

                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            bbox = (int(min_r), int(min_c), int(max_r), int(max_c))
            local = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
            for rr, cc in coords:
                local[rr - min_r, cc - min_c] = True
            components.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "coords": np.asarray(coords, dtype=int),
                    "area": int(len(coords)),
                    "mask": local,
                }
            )

    return components


def _is_rect_ring(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return int(component["area"]) == int(expected.sum()) and np.array_equal(component["mask"], expected)


def _is_solid_rect(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return int(component["area"]) == h * w and bool(np.all(component["mask"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each nested rectangular gap with horizontal stripes that alternate the core color and the enclosing ring color.

    Input structure:
    - The background is the dominant color.
    - The foreground contains one or more disjoint groups of concentric one-cell-thick rectangular rings.
    - Each group has 2 or 3 differently colored rings and one centered solid core block inside the innermost ring.
    - Rings and cores are axis-aligned rectangles; different groups do not overlap.

    Rule:
    - Identify every rectangular ring and the unique smaller object directly nested inside it.
    - For each ring, keep its border unchanged and fill the cells inside that ring but outside its direct child with horizontal stripes.
    - The stripes alternate from top to bottom between the group's core color and the current ring's color, starting with the core color on the top interior row.
    - Leave the core block and the background unchanged.

    Key insight:
    - The centered solid core is the signal source: its color propagates outward through each nested gap, but each layer mixes that signal with its own border color as row-wise stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)

    objects: list[dict] = []
    for component in components:
        if _is_rect_ring(component):
            objects.append({"kind": "ring", "bbox": component["bbox"], "color": int(component["color"])})
        elif _is_solid_rect(component):
            objects.append({"kind": "core", "bbox": component["bbox"], "color": int(component["color"])})

    rings = [i for i, obj in enumerate(objects) if obj["kind"] == "ring"]
    assert rings, "expected at least one rectangular ring"

    child_of: dict[int, int] = {}
    for ring_idx in rings:
        outer_bbox = objects[ring_idx]["bbox"]
        contained = [i for i, obj in enumerate(objects) if i != ring_idx and _strictly_contains(outer_bbox, obj["bbox"])]
        assert contained, "each ring must contain a smaller nested object"
        max_area = max(_bbox_area(objects[i]["bbox"]) for i in contained)
        candidates = [i for i in contained if _bbox_area(objects[i]["bbox"]) == max_area]
        assert len(candidates) == 1, "expected a unique direct child inside each ring"
        child_of[ring_idx] = int(candidates[0])

    def source_color(ring_idx: int) -> int:
        seen: set[int] = set()
        current = int(ring_idx)
        while True:
            assert current not in seen, "nested structure must be acyclic"
            seen.add(current)
            child_idx = child_of[current]
            child = objects[child_idx]
            if child["kind"] == "core":
                return int(child["color"])
            current = int(child_idx)

    out = g.copy()
    for ring_idx in sorted(rings, key=lambda idx: _bbox_area(objects[idx]["bbox"])):
        ring = objects[ring_idx]
        child = objects[child_of[ring_idx]]
        _fill_gap(
            out,
            ring_bbox=ring["bbox"],
            inner_bbox=child["bbox"],
            core_color=source_color(ring_idx),
            ring_color=int(ring["color"]),
        )

    return out


def _build_group_patch(bg: int, max_outer_h: int, max_outer_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    layer_count = random.choice((2, 3))
    core_size = random.choice((1, 2))
    colors = random.sample([c for c in range(_COLOR_COUNT) if c != bg], layer_count + 1)
    core_color = int(colors[0])
    ring_colors = [int(c) for c in colors[1:]]

    child_h = core_size
    child_w = core_size
    ring_sizes: list[tuple[int, int]] = []
    gap_choices = (1, 2, 3) if layer_count == 2 else (1, 2)
    for _ in range(layer_count):
        gap_v = random.choice(gap_choices)
        gap_h = random.choice(gap_choices)
        child_h += 2 * (gap_v + 1)
        child_w += 2 * (gap_h + 1)
        ring_sizes.append((int(child_h), int(child_w)))

    outer_h, outer_w = ring_sizes[-1]
    if outer_h > max_outer_h or outer_w > max_outer_w:
        return None

    inp = np.full((outer_h, outer_w), int(bg), dtype=int)
    out = inp.copy()

    core_bbox = _centered_bbox(outer_h, outer_w, core_size, core_size)
    cr0, cc0, cr1, cc1 = core_bbox
    inp[cr0 : cr1 + 1, cc0 : cc1 + 1] = core_color
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = core_color

    ring_bboxes: list[BBox] = []
    for (ring_h, ring_w), ring_color in zip(ring_sizes, ring_colors, strict=False):
        bbox = _centered_bbox(outer_h, outer_w, ring_h, ring_w)
        ring_bboxes.append(bbox)
        _draw_ring(inp, bbox, ring_color)
        _draw_ring(out, bbox, ring_color)

    for idx, (ring_bbox, ring_color) in enumerate(zip(ring_bboxes, ring_colors, strict=False)):
        inner_bbox = core_bbox if idx == 0 else ring_bboxes[idx - 1]
        _fill_gap(out, ring_bbox, inner_bbox, core_color, ring_color)

    return inp, out


def _place_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _pack_groups(groups: list[tuple[np.ndarray, np.ndarray]], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    count = len(groups)
    rows, cols = random.choice(_LAYOUTS[count])
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen_slots = random.sample(slots, count)

    row_heights = [0] * rows
    col_widths = [0] * cols
    for (inp_patch, _out_patch), (slot_r, slot_c) in zip(groups, chosen_slots, strict=False):
        ph, pw = inp_patch.shape
        row_heights[slot_r] = max(row_heights[slot_r], int(ph))
        col_widths[slot_c] = max(col_widths[slot_c], int(pw))

    row_heights = [h + (random.randint(0, 2) if h else 0) for h in row_heights]
    col_widths = [w + (random.randint(0, 2) if w else 0) for w in col_widths]
    row_gaps = [random.randint(1, 3) for _ in range(max(0, rows - 1))]
    col_gaps = [random.randint(1, 3) for _ in range(max(0, cols - 1))]
    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)

    total_h = margin_top + margin_bottom + sum(row_heights) + sum(row_gaps)
    total_w = margin_left + margin_right + sum(col_widths) + sum(col_gaps)
    if total_h > 30 or total_w > 30:
        return None

    row_starts = [margin_top]
    for i in range(1, rows):
        row_starts.append(row_starts[-1] + row_heights[i - 1] + row_gaps[i - 1])
    col_starts = [margin_left]
    for i in range(1, cols):
        col_starts.append(col_starts[-1] + col_widths[i - 1] + col_gaps[i - 1])

    inp = np.full((total_h, total_w), int(bg), dtype=int)
    out = inp.copy()

    for (inp_patch, out_patch), (slot_r, slot_c) in zip(groups, chosen_slots, strict=False):
        ph, pw = inp_patch.shape
        slot_h = row_heights[slot_r]
        slot_w = col_widths[slot_c]
        top = row_starts[slot_r] + random.randint(0, slot_h - ph)
        left = col_starts[slot_c] + random.randint(0, slot_w - pw)
        _place_patch(inp, inp_patch, int(top), int(left))
        _place_patch(out, out_patch, int(top), int(left))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice((1, 2, 3))
    bg = random.randint(0, _COLOR_COUNT - 1)
    max_dim = {1: 22, 2: 18, 3: 15}[group_count]

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(group_count):
        group = None
        for _ in range(60):
            group = _build_group_patch(bg, max_dim, max_dim)
            if group is not None:
                break
        if group is None:
            return None
        groups.append(group)

    packed = _pack_groups(groups, bg)
    if packed is None:
        return None
    inp, expected = packed

    if _get_bg(inp) != bg:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
