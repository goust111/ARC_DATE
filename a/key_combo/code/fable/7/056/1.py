from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, is_contiguous, paste


GRAY = Color.GREY
BG = Color.BLACK
LINE_COLORS = [c for c in Color.NOT_BLACK if c != GRAY]

GRID_RANGE = (18, 30)
COLOR_COUNT_RANGE = (2, 4)
TEMPLATE_SIZES = (3, 5)
TEMPLATE_ATTEMPTS = 200
GEN_ATTEMPTS = 400
ANCHOR_ATTEMPTS = 300
SEGMENT_LEN_RANGE = (2, 5)
INTERSECTION_GAP = 1
PLACEMENT_GAP = 1


def _diag_signature(coords: np.ndarray) -> tuple[int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, c0 = rows.min(), cols.min()
    r1, c1 = rows.max(), cols.max()
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w >= 2, "diagonal segment must have a square bbox with length >= 2"
    area = int(coords.shape[0])
    assert area == h, "diagonal segment must be one pixel thick with no gaps"

    diffs = rows - cols
    sums = rows + cols
    if np.all(diffs == diffs[0]):
        expected = {(int(r0 + k), int(c0 + k)) for k in range(h)}
        actual = {tuple(map(int, rc)) for rc in coords.tolist()}
        assert actual == expected, "main-diagonal segment must be contiguous"
        return 1, int(diffs[0])
    if np.all(sums == sums[0]):
        expected = {(int(r0 + k), int(c1 - k)) for k in range(h)}
        actual = {tuple(map(int, rc)) for rc in coords.tolist()}
        assert actual == expected, "anti-diagonal segment must be contiguous"
        return -1, int(sums[0])
    raise AssertionError("foreground segment must lie on one diagonal orientation")


def _parse_template(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid == GRAY
    assert mask.any(), "expected one gray stencil pattern"
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and side_h % 2 == 1, "gray stencil bbox must be an odd square"
    patch = crop_bbox(grid, b)
    assert set(map(int, np.unique(patch))) <= {bg, GRAY}, "gray stencil bbox may only contain gray and background"
    assert np.any(patch == GRAY), "gray stencil must contain visible pixels"
    return patch


def _parse_line_pairs(grid: np.ndarray, bg: int) -> dict[int, dict[int, int]]:
    components = find_components(grid, bg=bg, connectivity=8, monochromatic=True)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        color = int(comp["color"])
        if color == GRAY:
            continue
        by_color[color].append(comp)

    assert by_color, "expected at least one colored diagonal pair"
    line_pairs: dict[int, dict[int, int]] = {}
    for color, comps in by_color.items():
        assert len(comps) == 2, "each non-gray color must appear as exactly two diagonal segments"
        seen: dict[int, int] = {}
        for comp in comps:
            orientation, invariant = _diag_signature(comp["coords"])
            assert orientation not in seen, "each color must have one segment per diagonal orientation"
            seen[orientation] = invariant
        assert set(seen) == {1, -1}, "each color needs one main-diagonal and one anti-diagonal segment"
        line_pairs[color] = seen
    return line_pairs


def _anchor_from_invariants(diag_diff: int, anti_sum: int) -> tuple[int, int]:
    assert (diag_diff + anti_sum) % 2 == 0, "line intersection must land on a grid cell"
    r = (diag_diff + anti_sum) // 2
    c = (anti_sum - diag_diff) // 2
    return int(r), int(c)


def _stamp_mask(template: np.ndarray) -> np.ndarray:
    return template == GRAY


def _render_output(shape: tuple[int, int], template: np.ndarray, anchors: dict[int, tuple[int, int]], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    side = template.shape[0]
    radius = side // 2
    mask = _stamp_mask(template)
    for color, (r, c) in anchors.items():
        patch = np.where(mask, int(color), int(bg))
        r0 = int(r) - radius
        c0 = int(c) - radius
        region = out[r0 : r0 + side, c0 : c0 + side]
        assert region.shape == patch.shape, "stamp must fit in bounds"
        assert not np.any((region != bg) & mask), "stamps must not overlap"
        paste(out, patch, r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the hidden same-color diagonal intersections with recolored copies of the gray stencil.

    Input structure:
    - A black-background grid contains one gray stencil sprite whose bounding box is an odd square.
    - Every other foreground color appears exactly twice as separated diagonal segments.
    - For each such color, one segment follows the main diagonal direction and the other follows the anti-diagonal direction.

    Rule:
    - Extend every visible diagonal segment into its full infinite grid line while preserving its own orientation.
    - For each color, intersect only that color's two extended lines.
    - Center a recolored copy of the gray stencil on the resulting intersection, then erase every original segment and the gray stencil.

    Key insight:
    - The visible diagonals are only generators for invisible anchors: cross-color line crossings are decoys, while the gray stencil contributes shape but not output color.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG
    template = _parse_template(g, bg)
    line_pairs = _parse_line_pairs(g, bg)

    anchors: dict[int, tuple[int, int]] = {}
    side = template.shape[0]
    radius = side // 2
    for color, invariants in line_pairs.items():
        anchor = _anchor_from_invariants(invariants[1], invariants[-1])
        r, c = anchor
        assert radius <= r < g.shape[0] - radius and radius <= c < g.shape[1] - radius, "stamp anchor must fit the stencil bbox"
        anchors[color] = anchor

    return _render_output(g.shape, template, anchors, bg)


def _line_coords(shape: tuple[int, int], orientation: int, invariant: int) -> list[tuple[int, int]]:
    h, w = shape
    coords: list[tuple[int, int]] = []
    if orientation == 1:
        for r in range(h):
            c = r - int(invariant)
            if 0 <= c < w:
                coords.append((r, c))
    else:
        for r in range(h):
            c = int(invariant) - r
            if 0 <= c < w:
                coords.append((r, c))
    return coords


def _random_template(size: int) -> np.ndarray:
    center = size // 2
    coords = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(TEMPLATE_ATTEMPTS):
        count = random.randint(size + 1, size * size - 1)
        chosen = {(center, center)}
        remaining = [rc for rc in coords if rc != (center, center)]
        random.shuffle(remaining)
        for rc in remaining[: count - 1]:
            chosen.add(rc)
        mask = np.zeros((size, size), dtype=int)
        for r, c in chosen:
            mask[r, c] = 1
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not is_contiguous(mask, background=0, connectivity=8):
            continue
        if np.all(mask == 1):
            continue
        return np.where(mask == 1, GRAY, BG)
    raise RuntimeError("failed to sample a valid stencil pattern")


def _stamp_box(anchor: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    radius = side // 2
    r, c = anchor
    return r - radius, c - radius, r + radius, c + radius


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _sample_anchors(shape: tuple[int, int], count: int, side: int) -> list[tuple[int, int]] | None:
    h, w = shape
    radius = side // 2
    margin = radius + 2
    parity = random.randint(0, 1)
    candidates = [
        (r, c)
        for r in range(margin, h - margin)
        for c in range(margin, w - margin)
        if (r + c) % 2 == parity
    ]
    random.shuffle(candidates)

    for _ in range(ANCHOR_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        used_diff: set[int] = set()
        used_sum: set[int] = set()
        for r, c in candidates:
            box = _stamp_box((r, c), side)
            if r - c in used_diff or r + c in used_sum:
                continue
            if any(not _boxes_separated(box, _stamp_box(other, side), gap=1) for other in chosen):
                continue
            chosen.append((r, c))
            used_diff.add(r - c)
            used_sum.add(r + c)
            if len(chosen) == count:
                cross = _cross_color_decoys(shape, chosen)
                if len(cross) >= 1:
                    return chosen
                break
    return None


def _cross_color_decoys(shape: tuple[int, int], anchors: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    diffs = [r - c for r, c in anchors]
    sums = [r + c for r, c in anchors]
    own = set(anchors)
    decoys: set[tuple[int, int]] = set()
    for i, diff in enumerate(diffs):
        for j, total in enumerate(sums):
            if i == j or (diff + total) % 2 != 0:
                continue
            r = (diff + total) // 2
            c = (total - diff) // 2
            if 0 <= r < h and 0 <= c < w and (r, c) not in own:
                decoys.add((int(r), int(c)))
    return decoys


def _template_candidates(
    shape: tuple[int, int],
    side: int,
    forbidden_boxes: list[tuple[int, int, int, int]],
    forbidden_points: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    choices: list[tuple[int, int]] = []
    for r0 in range(0, h - side + 1):
        for c0 in range(0, w - side + 1):
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            if not all(_boxes_separated(box, other, gap=PLACEMENT_GAP) for other in forbidden_boxes):
                continue
            if any(r0 <= r <= r0 + side - 1 and c0 <= c <= c0 + side - 1 for r, c in forbidden_points):
                continue
            if all(_boxes_separated(box, other, gap=PLACEMENT_GAP) for other in forbidden_boxes):
                choices.append((r0, c0))
    return choices


def _run_lengths(indexes: list[int]) -> list[list[int]]:
    if not indexes:
        return []
    runs = [[indexes[0]]]
    for idx in indexes[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def _select_segment(
    line: list[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    allowed = [i for i, cell in enumerate(line) if cell not in forbidden_cells and not blocked[cell]]
    runs = [run for run in _run_lengths(allowed) if len(run) >= SEGMENT_LEN_RANGE[0]]
    random.shuffle(runs)
    for run in runs:
        max_len = min(SEGMENT_LEN_RANGE[1], len(run))
        seg_len = random.randint(SEGMENT_LEN_RANGE[0], max_len)
        start_positions = list(range(0, len(run) - seg_len + 1))
        random.shuffle(start_positions)
        for start in start_positions:
            idxs = run[start : start + seg_len]
            cells = [line[i] for i in idxs]
            if all(not blocked[r, c] for r, c in cells):
                return cells
    return None


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - PLACEMENT_GAP), min(h, r + PLACEMENT_GAP + 1)):
            for cc in range(max(0, c - PLACEMENT_GAP), min(w, c + PLACEMENT_GAP + 1)):
                blocked[rr, cc] = True


def _intersection_forbidden_indexes(
    line: list[tuple[int, int]],
    intersection_points: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    point_to_idx = {cell: i for i, cell in enumerate(line)}
    forbidden: set[tuple[int, int]] = set()
    for point in intersection_points:
        idx = point_to_idx.get(point)
        if idx is None:
            continue
        lo = max(0, idx - INTERSECTION_GAP)
        hi = min(len(line), idx + INTERSECTION_GAP + 1)
        forbidden.update(line[lo:hi])
    return forbidden


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    side = random.choice(TEMPLATE_SIZES)
    template = _random_template(side)
    color_count = random.randint(*COLOR_COUNT_RANGE)
    anchors = _sample_anchors((h, w), color_count, side)
    if anchors is None:
        return None

    colors = random.sample(LINE_COLORS, color_count)
    anchor_by_color = dict(zip(colors, anchors))
    line_specs = {
        color: {1: r - c, -1: r + c}
        for color, (r, c) in anchor_by_color.items()
    }

    decoys = _cross_color_decoys((h, w), anchors)
    if not decoys:
        return None

    stamp_boxes = [_stamp_box(anchor, side) for anchor in anchors]
    template_spots = _template_candidates((h, w), side, stamp_boxes, set(anchors) | decoys)
    if not template_spots:
        return None
    template_r0, template_c0 = random.choice(template_spots)

    inp = np.full((h, w), BG, dtype=int)
    paste(inp, template, template_r0, template_c0, transparent=BG)

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, [(r, c) for r in range(template_r0, template_r0 + side) for c in range(template_c0, template_c0 + side)])

    intersections_by_line: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for color, invs in line_specs.items():
        diff = invs[1]
        total = invs[-1]
        intersections_by_line[(color, 1, diff)] = set()
        intersections_by_line[(color, -1, total)] = set()
        for other_color, other_invs in line_specs.items():
            point_same = _anchor_from_invariants(diff, other_invs[-1])
            if 0 <= point_same[0] < h and 0 <= point_same[1] < w:
                intersections_by_line[(color, 1, diff)].add(point_same)
            point_cross = _anchor_from_invariants(other_invs[1], total)
            if 0 <= point_cross[0] < h and 0 <= point_cross[1] < w:
                intersections_by_line[(color, -1, total)].add(point_cross)

    line_order = [(color, orientation) for color in colors for orientation in (1, -1)]
    random.shuffle(line_order)
    for color, orientation in line_order:
        invariant = line_specs[color][orientation]
        line = _line_coords((h, w), orientation, invariant)
        forbidden = _intersection_forbidden_indexes(
            line, intersections_by_line[(color, orientation, invariant)]
        )
        segment = _select_segment(line, forbidden, blocked)
        if segment is None:
            return None
        for r, c in segment:
            inp[r, c] = int(color)
        _mark_blocked(blocked, segment)

    expected = _render_output(inp.shape, template, anchor_by_color, BG)
    if not _validate_case(inp, expected, anchor_by_color, template):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    anchors: dict[int, tuple[int, int]],
    template: np.ndarray,
) -> bool:
    try:
        parsed_template = _parse_template(inp, BG)
        if not np.array_equal(parsed_template, template):
            return False

        line_pairs = _parse_line_pairs(inp, BG)
        if set(line_pairs) != set(anchors):
            return False

        recovered_anchors = {
            color: _anchor_from_invariants(invariants[1], invariants[-1])
            for color, invariants in line_pairs.items()
        }
        if recovered_anchors != anchors:
            return False

        decoys = _cross_color_decoys(inp.shape, list(anchors.values()))
        if len(decoys) < 1:
            return False

        comps = find_components(inp, bg=BG, connectivity=8, monochromatic=True)
        for comp in comps:
            color = int(comp["color"])
            if color == GRAY:
                continue
            orientation, invariant = _diag_signature(comp["coords"])
            line = _line_coords(inp.shape, orientation, invariant)
            if orientation == 1:
                intersections = {
                    _anchor_from_invariants(invariant, other_invariants[-1])
                    for other_invariants in line_pairs.values()
                }
            else:
                intersections = {
                    _anchor_from_invariants(other_invariants[1], invariant)
                    for other_invariants in line_pairs.values()
                }
            forbidden = _intersection_forbidden_indexes(line, intersections)
            for r, c in map(tuple, comp["coords"].tolist()):
                cell = (int(r), int(c))
                if cell in intersections or cell in forbidden:
                    return False
                if expected[cell] != BG:
                    return False

        independent = _render_output(inp.shape, parsed_template, recovered_anchors, BG)
        if not np.array_equal(independent, expected):
            return False

        template_box = bbox(inp == GRAY)
        if template_box is None:
            return False
        tr0, tc0, tr1, tc1 = template_box
        for r, c in set(anchors.values()) | decoys:
            if tr0 <= r <= tr1 and tc0 <= c <= tc1:
                return False

        if np.any((inp == GRAY) & (expected != BG)):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            inp, _expected = built
            return inp
    raise RuntimeError("failed to generate valid hidden-intersection stamp input")
