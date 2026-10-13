from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR = 5
MARKER_COLORS = (1, 2, 3, 4)  # blue, red, green, yellow
AREA_VALUES = (2, 3, 4, 5, 6)
MAX_HALF = 14
GEN_RETRIES = 400
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_BASE_SHAPES = {
    2: [((0, 0), (0, 1))],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ],
    6: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
    ],
}


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _normalize_shape(cells: tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    height = max(r for r, _c in cells) + 1
    rotated = {(c, height - 1 - r) for r, c in cells}
    return _normalize_shape(rotated)


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    width = max(c for _r, c in cells) + 1
    flipped = {(r, width - 1 - c) for r, c in cells}
    return _normalize_shape(flipped)


def _shape_variants(base_shapes: list[tuple[tuple[int, int], ...]]) -> list[tuple[tuple[int, int], ...]]:
    out: set[tuple[tuple[int, int], ...]] = set()
    for base in base_shapes:
        cur = _normalize_shape(base)
        for _ in range(4):
            out.add(cur)
            out.add(_flip_shape(cur))
            cur = _rotate_shape(cur)
    return sorted(out)


AREA_TO_SHAPES = {area: _shape_variants(list(shapes)) for area, shapes in _BASE_SHAPES.items()}


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = []
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            row_candidates.append((int(grid[r, 0]), int(r)))
    col_candidates = []
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            col_candidates.append((int(grid[0, c]), int(c)))

    matches = [
        (color_r, r, c)
        for color_r, r in row_candidates
        for color_c, c in col_candidates
        if color_r == color_c and 0 < r < h - 1 and 0 < c < w - 1
    ]
    assert len(matches) == 1, "expected exactly one separator cross"
    color, cross_r, cross_c = matches[0]
    return int(cross_r), int(cross_c), int(color)


def _partition_boxes(shape: tuple[int, int], cross_r: int, cross_c: int) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    top_h = int(cross_r)
    bottom_h = int(h - cross_r - 1)
    left_w = int(cross_c)
    right_w = int(w - cross_c - 1)
    assert top_h == bottom_h and left_w == right_w
    assert top_h > 0 and left_w > 0
    return [
        (0, cross_r - 1, 0, cross_c - 1),
        (0, cross_r - 1, cross_c + 1, w - 1),
        (cross_r + 1, h - 1, 0, cross_c - 1),
        (cross_r + 1, h - 1, cross_c + 1, w - 1),
    ]


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _fragment_from_coords(color: int, coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    shape = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    return {
        "color": int(color),
        "area": int(len(coords)),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "shape": shape,
        "height": height,
        "width": width,
    }


def _extract_fragments(
    grid: np.ndarray,
    marker_positions: list[tuple[int, int]],
    marker_colors: list[int],
) -> dict[int, list[dict[str, object]]]:
    allowed = set(marker_colors) | {int(_dominant_color(grid))}
    marker_set = {tuple(map(int, pos)) for pos in marker_positions}
    fragments_by_color: dict[int, list[dict[str, object]]] = {color: [] for color in marker_colors}

    for color in marker_colors:
        mask = grid == int(color)
        for r, c in marker_set:
            if int(grid[r, c]) == int(color):
                mask[r, c] = False
        for coords in _connected_components(mask):
            fragments_by_color[int(color)].append(_fragment_from_coords(int(color), coords))

    for value in map(int, np.unique(grid)):
        assert value in allowed or value == SEPARATOR
    return fragments_by_color


def _render_output(
    grid_shape: tuple[int, int],
    bg: int,
    separator_color: int,
    cross_r: int,
    cross_c: int,
    boxes: list[tuple[int, int, int, int]],
    marker_colors_by_box: list[int],
    fragments_by_color: dict[int, list[dict[str, object]]],
) -> np.ndarray:
    out = np.full(grid_shape, int(bg), dtype=int)
    out[cross_r, :] = int(separator_color)
    out[:, cross_c] = int(separator_color)

    color_to_box: dict[int, tuple[int, int, int, int]] = {}
    for box, marker_color in zip(boxes, marker_colors_by_box):
        r0, _r1, c0, _c1 = box
        out[r0, c0] = int(marker_color)
        color_to_box[int(marker_color)] = box

    for color, box in color_to_box.items():
        cursor = 1
        fragments = sorted(
            fragments_by_color.get(int(color), []),
            key=lambda frag: (
                -int(frag["area"]),
                int(frag["bbox"][0]),
                int(frag["bbox"][1]),
            ),
        )
        for frag in fragments:
            r0, r1, c0, c1 = box
            top = int(r0 + cursor)
            left = int(c0)
            height = int(frag["height"])
            width = int(frag["width"])
            assert top + height - 1 <= r1
            assert left + width - 1 <= c1
            for dr, dc in frag["shape"]:
                rr = int(top + dr)
                cc = int(left + dc)
                assert out[rr, cc] == int(bg)
                out[rr, cc] = int(color)
            cursor += height + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered monochrome fragments into the quadrant whose corner marker has the same color.

    Input structure:
    - The background is black, with one full grey row and one full grey column forming a single-pixel cross.
    - The cross splits the grid into four equal rectangular quadrants.
    - The top-left cell inside each quadrant is a one-pixel color marker; the four marker colors are all different.
    - Additional 4-connected monochrome fragments of those marker colors are scattered across the quadrants.

    Rule:
    - Keep the grey cross fixed and keep the four corner markers fixed.
    - Collect every non-marker fragment and group fragments by their color.
    - Send each color group to the quadrant whose corner marker has that color.
    - Inside the destination quadrant, stack that color's fragments from largest area to smallest: the first fragment starts one row below the marker, all fragment bounding boxes are left-aligned to the quadrant interior, and consecutive bounding boxes are separated by one background row.

    Key insight:
    - The corner markers define four color-coded inboxes, so fragments must be regrouped by color before being reordered by area inside the matching quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    cross_r, cross_c, separator_color = _find_separator_cross(g)
    boxes = _partition_boxes(g.shape, cross_r, cross_c)

    marker_positions: list[tuple[int, int]] = []
    marker_colors_by_box: list[int] = []
    for r0, _r1, c0, _c1 in boxes:
        marker_positions.append((int(r0), int(c0)))
        marker_colors_by_box.append(int(g[r0, c0]))

    assert len(set(marker_colors_by_box)) == 4

    fragments_by_color = _extract_fragments(g, marker_positions, marker_colors_by_box)
    return _render_output(
        g.shape,
        int(bg),
        int(separator_color),
        int(cross_r),
        int(cross_c),
        boxes,
        marker_colors_by_box,
        fragments_by_color,
    )


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in DIR8:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _sample_color_counts() -> dict[int, int]:
    for _ in range(200):
        counts = {color: random.randint(0, 3) for color in MARKER_COLORS}
        total = sum(counts.values())
        if not (5 <= total <= 8):
            continue
        if max(counts.values()) < 2:
            continue
        if sum(count > 0 for count in counts.values()) < 3:
            continue
        return counts
    raise RuntimeError("failed to sample fragment counts")


def _sample_fragments(counts: dict[int, int]) -> list[dict[str, object]]:
    fragments: list[dict[str, object]] = []
    for color in MARKER_COLORS:
        count = int(counts[int(color)])
        if count == 0:
            continue
        areas = random.sample(AREA_VALUES, count)
        for area in areas:
            shape = random.choice(AREA_TO_SHAPES[int(area)])
            height, width = _shape_dims(shape)
            fragments.append(
                {
                    "color": int(color),
                    "area": int(area),
                    "shape": shape,
                    "height": int(height),
                    "width": int(width),
                }
            )
    random.shuffle(fragments)
    return fragments


def _stack_requirement(fragments: list[dict[str, object]], color: int) -> int:
    color_frags = [frag for frag in fragments if int(frag["color"]) == int(color)]
    if not color_frags:
        return 1
    return int(sum(int(frag["height"]) for frag in color_frags) + len(color_frags))


def _choose_half_sizes(fragments: list[dict[str, object]]) -> tuple[int, int]:
    max_stack = max(_stack_requirement(fragments, color) for color in MARKER_COLORS)
    max_width = max(int(frag["width"]) for frag in fragments)
    ph_min = max(7, int(max_stack))
    pw_min = max(7, int(max_width + 2))
    if ph_min > MAX_HALF or pw_min > MAX_HALF:
        raise RuntimeError("fragments do not fit within ARC bounds")
    ph = random.randint(ph_min, MAX_HALF)
    pw = random.randint(pw_min, MAX_HALF)
    return int(ph), int(pw)


def _assign_source_boxes(
    fragments: list[dict[str, object]],
    target_box_by_color: dict[int, int],
) -> list[int]:
    total = len(fragments)
    for _ in range(200):
        order = list(range(total))
        random.shuffle(order)
        assignments = [0] * total
        for box_index, frag_index in enumerate(order[:4]):
            assignments[frag_index] = int(box_index)
        for frag_index in order[4:]:
            assignments[frag_index] = random.randint(0, 3)

        if any(assignments.count(box_index) == 0 for box_index in range(4)):
            continue
        if all(
            int(assignments[idx]) == int(target_box_by_color[int(fragments[idx]["color"])])
            for idx in range(total)
        ):
            continue
        return assignments
    raise RuntimeError("failed to assign source boxes")


def _place_fragment_in_box(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    blocked: np.ndarray,
    fragment: dict[str, object],
) -> tuple[int, int, int, int] | None:
    _r0, _r1, _c0, _c1 = box
    ph, pw = blocked.shape
    height = int(fragment["height"])
    width = int(fragment["width"])
    top_candidates = list(range(1, ph - height))
    left_candidates = list(range(1, pw - width))
    random.shuffle(top_candidates)
    random.shuffle(left_candidates)

    for local_top in top_candidates:
        for local_left in left_candidates:
            cells = [(int(local_top + dr), int(local_left + dc)) for dr, dc in fragment["shape"]]
            if any(blocked[r, c] for r, c in cells):
                continue

            global_top = int(box[0] + local_top)
            global_left = int(box[2] + local_left)
            color = int(fragment["color"])
            for dr, dc in fragment["shape"]:
                grid[global_top + dr, global_left + dc] = color
            _mark_blocked(blocked, cells)
            return (
                int(global_top),
                int(global_left),
                int(global_top + height - 1),
                int(global_left + width - 1),
            )

    return None


def _build_expected_output(
    inp: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    box_colors: list[int],
    fragments: list[dict[str, object]],
    ph: int,
    pw: int,
) -> np.ndarray:
    fragments_by_color: dict[int, list[dict[str, object]]] = {color: [] for color in box_colors}
    for frag in fragments:
        fragments_by_color[int(frag["color"])].append(frag)
    return _render_output(
        inp.shape,
        BG,
        SEPARATOR,
        ph,
        pw,
        boxes,
        box_colors,
        fragments_by_color,
    )


def _generate_one() -> np.ndarray | None:
    counts = _sample_color_counts()
    fragments = _sample_fragments(counts)
    ph, pw = _choose_half_sizes(fragments)
    h = 2 * ph + 1
    w = 2 * pw + 1

    inp = np.full((h, w), BG, dtype=int)
    inp[ph, :] = SEPARATOR
    inp[:, pw] = SEPARATOR
    boxes = _partition_boxes(inp.shape, ph, pw)

    box_colors = random.sample(list(MARKER_COLORS), 4)
    target_box_by_color = {int(color): int(box_index) for box_index, color in enumerate(box_colors)}
    for box, color in zip(boxes, box_colors):
        inp[box[0], box[2]] = int(color)

    assignments = _assign_source_boxes(fragments, target_box_by_color)
    blocked_by_box = {}
    for box in boxes:
        local_blocked = np.zeros((ph, pw), dtype=bool)
        _mark_blocked(local_blocked, [(0, 0)])
        blocked_by_box[box] = local_blocked

    for fragment, box_index in zip(fragments, assignments):
        box = boxes[int(box_index)]
        bbox = _place_fragment_in_box(inp, box, blocked_by_box[box], fragment)
        if bbox is None:
            return None
        fragment["bbox"] = bbox

    expected = _build_expected_output(inp, boxes, box_colors, fragments, ph, pw)
    if not np.array_equal(main(inp), expected):
        return None

    bg_count = int(np.sum(inp == BG))
    non_bg_counts = Counter(map(int, inp.reshape(-1)))
    if any(color != BG and count >= bg_count for color, count in non_bg_counts.items()):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
