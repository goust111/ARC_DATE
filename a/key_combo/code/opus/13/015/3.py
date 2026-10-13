import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (RED, GREEN, YELLOW)
TEMPLATE_MARKER = 9

_TEMPLATE_ROWS = (
    (
        "1110",
        "1911",
        "1101",
        "0111",
    ),
    (
        "01110",
        "11191",
        "10011",
        "11110",
    ),
    (
        "11111",
        "19101",
        "11001",
        "11110",
    ),
    (
        "1110",
        "1911",
        "1101",
        "1101",
        "0111",
    ),
)

GEN_ATTEMPTS = 400
RIGHT_PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, f"expected exactly one full-height separator, got {cols}"
    return int(cols[0])


def _hole_components(occupied: np.ndarray) -> list[np.ndarray]:
    holes = []
    h, w = occupied.shape
    for comp in _components_from_mask(~occupied):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        holes.append(comp)
    return holes


def _extract_single_hole_shape(occupied_coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _bbox(occupied_coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[occupied_coords[:, 0] - r0, occupied_coords[:, 1] - c0] = True
    holes = _hole_components(local)
    assert len(holes) == 1, f"expected exactly one enclosed hole, got {len(holes)}"
    return _normalize_coords(holes[0])


def _parse_template(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _template_variants() -> list[np.ndarray]:
    return [_parse_template(rows) for rows in _TEMPLATE_ROWS]


def _validate_template(template: np.ndarray) -> tuple[tuple[int, int], ...]:
    marker_pos = np.argwhere(template == TEMPLATE_MARKER)
    assert marker_pos.shape == (1, 2)
    mr, mc = map(int, marker_pos[0])
    for nr, nc in _neighbors4(mr, mc):
        assert 0 <= nr < template.shape[0] and 0 <= nc < template.shape[1]
        assert int(template[nr, nc]) == BLUE
    blue = template == BLUE
    assert len(_components_from_mask(blue)) == 1
    occupied = template != BLACK
    shape = _extract_single_hole_shape(np.argwhere(occupied))
    assert len(shape) <= 4
    hole_h = max(r for r, _ in shape) + 1
    hole_w = max(c for _, c in shape) + 1
    assert hole_h <= 3 and hole_w <= 3
    return shape


_BASE_TEMPLATES = tuple(_template_variants())
_BASE_HOLE_SHAPES = tuple(_validate_template(t.copy()) for t in _BASE_TEMPLATES)


def _transform_template(template: np.ndarray) -> np.ndarray:
    out = template.copy()
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _label_components(mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=int)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            labels[r, c] = idx
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = idx
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps, labels


def _collect_left_objects(grid: np.ndarray, sep: int) -> list[dict]:
    left = grid[:, :sep]
    blue_mask = left == BLUE
    comps, labels = _label_components(blue_mask)
    assert comps, "expected blue components on the left side"

    objects: dict[int, dict] = {idx: {"blue": coords} for idx, coords in enumerate(comps)}
    marker_cells = []
    for color in MARKER_COLORS:
        for r, c in np.argwhere(left == color):
            marker_cells.append((int(r), int(c), int(color)))
    assert marker_cells, "expected colored markers inside left components"

    seen_labels = set()
    for r, c, color in marker_cells:
        neighbor_labels = {
            int(labels[nr, nc])
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < left.shape[0] and 0 <= nc < left.shape[1] and labels[nr, nc] != -1
        }
        assert len(neighbor_labels) == 1, f"left marker at {(r, c)} must belong to exactly one blue component"
        idx = next(iter(neighbor_labels))
        assert idx not in seen_labels, "each blue component must have exactly one marker"
        seen_labels.add(idx)
        objects[idx]["marker"] = (r, c)
        objects[idx]["color"] = color

    assert len(seen_labels) == len(comps), "every left blue component must have a marker"
    return [objects[idx] for idx in sorted(objects)]


def _right_markers(grid: np.ndarray, sep: int) -> dict[int, tuple[int, int]]:
    markers = {}
    for color in MARKER_COLORS:
        pts = np.argwhere(grid[:, sep + 1 :] == color)
        if pts.size == 0:
            continue
        assert pts.shape[0] == 1, f"expected one right-side marker for color {color}"
        r, c = map(int, pts[0])
        markers[color] = (r, c + sep + 1)
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each left blue object to the separator, extract its enclosed hole, and stamp that hole shape on the matching right marker.

    Input structure:
    - A black (0) background grid is split by a full-height grey (5) separator column.
    - On the left are 2-3 blue (1) connected components, each containing one embedded colored marker pixel from {red(2), green(3), yellow(4)}.
    - On the right are singleton colored markers using the same marker colors.

    Rule:
    - Move every left component horizontally right until its occupied cells are flush with the separator's left side; its embedded colored marker moves with it.
    - For each moved component, find the unique enclosed black hole inside that component and record the hole's shape relative to its own bounding box.
    - Match that hole to the right-side singleton with the same marker color and stamp the hole shape there in blue, aligning the hole bbox's top-left corner to the marker.
    - The output clears the entire left side and the separator to black, keeps the right-side markers, and overlays the blue stamps on the right.

    Key insight:
    - The colored pixel identifies which moved component belongs to which destination marker, but the actual stamp content is the enclosed hole created by analyzing the moved component's geometry.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g)
    objects = _collect_left_objects(g, sep)
    right_by_color = _right_markers(g, sep)

    out = np.zeros_like(g)
    for color, pos in right_by_color.items():
        out[pos] = color

    for obj in objects:
        blue = obj["blue"]
        mr, mc = obj["marker"]
        color = int(obj["color"])

        occupied = np.vstack([blue, np.array([[mr, mc]], dtype=int)])
        delta = (sep - 1) - int(occupied[:, 1].max())
        assert delta >= 0

        moved_occ = occupied.copy()
        moved_occ[:, 1] += delta
        hole_shape = _extract_single_hole_shape(moved_occ)

        anchor = right_by_color[color]
        for dr, dc in hole_shape:
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            out[rr, cc] = BLUE

    return out


def _choose_variants(count: int) -> list[dict] | None:
    for _ in range(80):
        chosen_bases = random.sample(list(_BASE_TEMPLATES), count)
        variants = []
        seen_shapes = set()
        ok = True
        for base in chosen_bases:
            variant = _transform_template(base)
            shape = _validate_template(variant)
            if shape in seen_shapes:
                ok = False
                break
            seen_shapes.add(shape)
            variants.append({"template": variant, "shape": shape})
        if ok and any((0, 0) not in v["shape"] for v in variants):
            return variants
    return None


def _distribute_rows(total: int, heights: list[int]) -> list[int] | None:
    mandatory = sum(heights) + max(0, len(heights) - 1)
    if mandatory > total:
        return None
    slots = [0] + [1] * (len(heights) - 1) + [0]
    extra = total - mandatory
    for _ in range(extra):
        slots[random.randrange(len(slots))] += 1
    tops = []
    cur = slots[0]
    for idx, h in enumerate(heights):
        tops.append(cur)
        cur += h
        if idx < len(heights) - 1:
            cur += slots[idx + 1]
    return tops


def _assign_deltas(widths: list[int], sep: int) -> list[int] | None:
    order = sorted(range(len(widths)), key=lambda i: sep - widths[i])
    used = set()
    deltas = [0] * len(widths)
    for idx in order:
        max_delta = sep - widths[idx]
        choices = [d for d in range(1, max_delta + 1) if d not in used]
        if not choices:
            return None
        delta = random.choice(choices)
        used.add(delta)
        deltas[idx] = delta
    return deltas


def _place_right_markers(h: int, w: int, sep: int, variants: list[dict], colors: list[int]) -> dict[int, tuple[int, int]] | None:
    order = sorted(range(len(variants)), key=lambda i: (-max(r for r, _ in variants[i]["shape"]), -max(c for _, c in variants[i]["shape"])))
    occupied_stamp_cells: set[tuple[int, int]] = set()
    marker_cells: dict[int, tuple[int, int]] = {}

    for idx in order:
        shape = variants[idx]["shape"]
        hole_h = max(r for r, _ in shape) + 1
        hole_w = max(c for _, c in shape) + 1
        placed = False
        for _ in range(RIGHT_PLACEMENT_ATTEMPTS):
            anchor_r = random.randrange(0, h - hole_h + 1)
            anchor_c = random.randrange(sep + 1, w - hole_w + 1)
            stamp = {(anchor_r + dr, anchor_c + dc) for dr, dc in shape}
            if stamp & occupied_stamp_cells:
                continue
            if (anchor_r, anchor_c) in marker_cells.values():
                continue
            if any((anchor_r, anchor_c) == cell for cell in occupied_stamp_cells):
                continue
            if any(cell in marker_cells.values() for cell in stamp):
                continue
            marker_cells[colors[idx]] = (anchor_r, anchor_c)
            occupied_stamp_cells |= stamp
            placed = True
            break
        if not placed:
            return None
    return marker_cells


def _expected_output(shape: tuple[int, int], markers: dict[int, tuple[int, int]], hole_shapes: dict[int, tuple[tuple[int, int], ...]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, pos in markers.items():
        out[pos] = color
    for color, hole_shape in hole_shapes.items():
        ar, ac = markers[color]
        for dr, dc in hole_shape:
            out[ar + dr, ac + dc] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 3))
    variants = _choose_variants(count)
    if variants is None:
        return None

    heights = [int(v["template"].shape[0]) for v in variants]
    widths = [int(v["template"].shape[1]) for v in variants]
    max_w = max(widths)
    sep_min = max_w + 4
    sep_max = min(12, max_w + 7)
    if sep_min > sep_max:
        return None
    sep = random.randint(sep_min, sep_max)

    deltas = _assign_deltas(widths, sep)
    if deltas is None:
        return None

    right_w_min = max(6, max(max(c for _, c in v["shape"]) + 1 for v in variants) + 3)
    right_w = random.randint(right_w_min, min(9, 29 - sep))
    w = sep + 1 + right_w
    if w > 30:
        return None

    min_h = max(18, sum(heights) + (count - 1))
    max_h = min(24, min_h + 5)
    h = random.randint(min_h, max_h)
    tops = _distribute_rows(h, heights)
    if tops is None:
        return None

    colors = random.sample(list(MARKER_COLORS), count)
    right_markers = _place_right_markers(h, w, sep, variants, colors)
    if right_markers is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[:, sep] = GREY
    expected_holes: dict[int, tuple[tuple[int, int], ...]] = {}

    for idx, variant in enumerate(variants):
        tpl = variant["template"]
        color = colors[idx]
        top = tops[idx]
        final_left = sep - tpl.shape[1]
        left = final_left - deltas[idx]
        if left < 0:
            return None
        for r in range(tpl.shape[0]):
            for c in range(tpl.shape[1]):
                val = int(tpl[r, c])
                if val == BLACK:
                    continue
                rr = top + r
                cc = left + c
                if cc >= sep:
                    return None
                if val == BLUE:
                    grid[rr, cc] = BLUE
                else:
                    grid[rr, cc] = color
        expected_holes[color] = variant["shape"]

    for color, (r, c) in right_markers.items():
        grid[r, c] = color

    blue_left = len(_components_from_mask(grid[:, :sep] == BLUE))
    if blue_left != count:
        return None

    produced = main(grid)
    expected = _expected_output(grid.shape, right_markers, expected_holes)
    if not np.array_equal(produced, expected):
        return None
    if max(grid.shape) > 30 or max(produced.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
