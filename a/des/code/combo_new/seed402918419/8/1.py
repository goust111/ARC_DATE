from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_ALL_COLORS = tuple(range(10))
_CORNERS = ("TL", "TR", "BL", "BR")
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] - pref[r1 + 1, c0] + pref[r0, c0])


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    non_bg = _integral(grid != bg)
    best: tuple[int, tuple[int, int, int, int]] | None = None

    for r0 in range(h - 4):
        for c0 in range(w - 4):
            for r1 in range(r0 + 4, h):
                height = r1 - r0 + 1
                for c1 in range(c0 + 4, w):
                    width = c1 - c0 + 1
                    area = height * width
                    if best is not None and area <= best[0]:
                        continue
                    if _rect_sum(non_bg, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    perimeter = 2 * (height + width) - 4
                    if _rect_sum(non_bg, r0, c0, r1, c1) != perimeter:
                        continue
                    best = (area, (r0, c0, r1, c1))

    assert best is not None, "expected one hollow frame"
    return best[1]


def _frame_boundary_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r, c1) for r in range(r0 + 1, r1)]
    cells += [(r1, c) for c in range(c1, c0 - 1, -1)]
    cells += [(r, c0) for r in range(r1 - 1, r0, -1)]
    return cells


def _side_of_frame_cell(pos: tuple[int, int], rect: tuple[int, int, int, int]) -> str:
    r, c = pos
    r0, c0, r1, c1 = rect
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("marker must sit on a non-corner frame edge cell")


def _corner_region_from_pair(
    rect: tuple[int, int, int, int], points: list[tuple[int, int]]
) -> tuple[str, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect
    side_to_point = {_side_of_frame_cell(point, rect): point for point in points}
    sides = set(side_to_point)

    if sides == {"top", "left"}:
        top_col = side_to_point["top"][1]
        left_row = side_to_point["left"][0]
        return "TL", (r0 + 1, c0 + 1, left_row - 1, top_col - 1)
    if sides == {"top", "right"}:
        top_col = side_to_point["top"][1]
        right_row = side_to_point["right"][0]
        return "TR", (r0 + 1, top_col + 1, right_row - 1, c1 - 1)
    if sides == {"bottom", "left"}:
        bottom_col = side_to_point["bottom"][1]
        left_row = side_to_point["left"][0]
        return "BL", (left_row + 1, c0 + 1, r1 - 1, bottom_col - 1)
    if sides == {"bottom", "right"}:
        bottom_col = side_to_point["bottom"][1]
        right_row = side_to_point["right"][0]
        return "BR", (right_row + 1, bottom_col + 1, r1 - 1, c1 - 1)
    raise AssertionError("marker pair must lie on adjacent frame sides")


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _largest_color_component_patch(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _components4(grid == color)
    assert comps, "expected at least one same-color component"
    comp = max(comps, key=len)
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    return patch


def _color_first_occurrences(grid: np.ndarray, colors: list[int]) -> dict[int, tuple[int, int]]:
    target = set(colors)
    found: dict[int, tuple[int, int]] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val in target and val not in found:
                found[val] = (r, c)
                if len(found) == len(target):
                    return found
    return found


def _tile_region(out: np.ndarray, region: tuple[int, int, int, int], patch: np.ndarray) -> None:
    rr0, cc0, rr1, cc1 = region
    ph, pw = patch.shape
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            out[r, c] = int(patch[(r - rr0) % ph, (c - cc0) % pw])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition a framed interior by paired edge markers and fill each corner region with its matching exterior template.

    Input structure:
    - The background is the dominant color.
    - One hollow rectangular frame sits near the center; its border is mostly one color and its interior is pure background.
    - Extra marker colors appear on the frame border in same-color pairs, with each pair split across two adjacent sides.
    - Outside the frame, each marker color also appears as one small solid rectangular template.

    Rule:
    - Find the hollow frame, then for each marker color read its two border positions and infer which corner they share.
    - Treat the marker column and marker row as virtual cut lines into the frame interior; the shared corner determines the rectangular corner region bounded by those cuts and the frame walls.
    - Find the same-color exterior template and tile it from that corner region's top-left cell, clipping at the region boundary.
    - If corner regions overlap, process colors by their first row-major appearance in the whole grid; later colors overwrite earlier ones.

    Key insight: A same-color marker pair encodes both the cut coordinates and the target corner, so the border geometry determines the partition while the matching exterior template determines the fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame = _find_frame(g, bg)

    border_vals = [int(g[r, c]) for r, c in _frame_boundary_cells(frame)]
    border_color = int(Counter(border_vals).most_common(1)[0][0])

    markers: dict[int, list[tuple[int, int]]] = {}
    for r, c in _frame_boundary_cells(frame):
        val = int(g[r, c])
        if val != border_color:
            markers.setdefault(val, []).append((r, c))

    marker_colors = sorted(markers)
    assert marker_colors, "expected colored marker pairs on the frame"

    regions: dict[int, tuple[int, int, int, int]] = {}
    templates: dict[int, np.ndarray] = {}
    for color, points in markers.items():
        assert len(points) == 2, "each marker color must appear twice on the frame"
        _corner, region = _corner_region_from_pair(frame, points)
        assert region[0] <= region[2] and region[1] <= region[3], "marker pair must define a non-empty region"
        regions[color] = region
        templates[color] = _largest_color_component_patch(g, color)

    first = _color_first_occurrences(g, list(marker_colors))
    order = sorted(marker_colors, key=lambda color: first[color])

    out = g.copy()
    for color in order:
        _tile_region(out, regions[color], templates[color])
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _boxes_overlap_with_pad(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int
) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _region_from_corner(
    rect: tuple[int, int, int, int], corner: str, row: int, col: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    if corner == "TL":
        return (r0 + 1, c0 + 1, row - 1, col - 1)
    if corner == "TR":
        return (r0 + 1, col + 1, row - 1, c1 - 1)
    if corner == "BL":
        return (row + 1, c0 + 1, r1 - 1, col - 1)
    if corner == "BR":
        return (row + 1, col + 1, r1 - 1, c1 - 1)
    raise AssertionError("unknown corner")


def _sample_marker_specs(
    rect: tuple[int, int, int, int], corners: list[str], want_overlap: bool
) -> dict[str, dict[str, object]] | None:
    r0, c0, r1, c1 = rect
    side_groups = {
        "top": [corner for corner in corners if corner in {"TL", "TR"}],
        "bottom": [corner for corner in corners if corner in {"BL", "BR"}],
        "left": [corner for corner in corners if corner in {"TL", "BL"}],
        "right": [corner for corner in corners if corner in {"TR", "BR"}],
    }
    side_candidates = {
        "top": list(range(c0 + 3, c1 - 2)),
        "bottom": list(range(c0 + 3, c1 - 2)),
        "left": list(range(r0 + 3, r1 - 2)),
        "right": list(range(r0 + 3, r1 - 2)),
    }
    if any(len(side_candidates[side]) < len(group) for side, group in side_groups.items()):
        return None

    for _ in range(300):
        side_values: dict[str, dict[str, int]] = {}
        for side, group in side_groups.items():
            picks = random.sample(side_candidates[side], len(group))
            random.shuffle(picks)
            side_values[side] = {corner: pick for corner, pick in zip(group, picks)}

        specs: dict[str, dict[str, object]] = {}
        regions: list[tuple[int, int, int, int]] = []
        for corner in corners:
            if corner == "TL":
                row = side_values["left"]["TL"]
                col = side_values["top"]["TL"]
                markers = [(r0, col), (row, c0)]
            elif corner == "TR":
                row = side_values["right"]["TR"]
                col = side_values["top"]["TR"]
                markers = [(r0, col), (row, c1)]
            elif corner == "BL":
                row = side_values["left"]["BL"]
                col = side_values["bottom"]["BL"]
                markers = [(r1, col), (row, c0)]
            else:
                row = side_values["right"]["BR"]
                col = side_values["bottom"]["BR"]
                markers = [(r1, col), (row, c1)]

            region = _region_from_corner(rect, corner, row, col)
            specs[corner] = {"markers": markers, "region": region}
            regions.append(region)

        has_overlap = any(
            _boxes_overlap(regions[i], regions[j]) for i in range(len(regions)) for j in range(i + 1, len(regions))
        )
        if has_overlap == want_overlap:
            return specs
    return None


def _place_templates(
    shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    colors: list[int],
) -> dict[int, tuple[int, int, int, int]] | None:
    h, w = shape
    boxes: dict[int, tuple[int, int, int, int]] = {}
    taken: list[tuple[int, int, int, int]] = [frame]

    for color in random.sample(colors, len(colors)):
        placed = False
        for _ in range(300):
            th = random.randint(1, 4)
            tw = random.randint(1, 4)
            if th * tw == 1:
                continue
            r0 = random.randint(0, h - th)
            c0 = random.randint(0, w - tw)
            box = (r0, c0, r0 + th - 1, c0 + tw - 1)
            if any(_boxes_overlap_with_pad(box, other, pad=1) for other in taken):
                continue
            boxes[color] = box
            taken.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _first_occurrence_order(grid: np.ndarray, colors: list[int]) -> list[int]:
    first = _color_first_occurrences(grid, colors)
    return sorted(colors, key=lambda color: first[color])


def _expected_output(
    grid: np.ndarray,
    color_specs: dict[int, dict[str, object]],
    template_boxes: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    order = _first_occurrence_order(grid, list(color_specs))
    for color in order:
        region = color_specs[color]["region"]
        box = template_boxes[color]
        patch = grid[box[0] : box[2] + 1, box[1] : box[3] + 1]
        _tile_region(out, region, patch)
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    bg = random.choice(_ALL_COLORS)
    n_regions = random.randint(2, 4)
    want_overlap = bool(random.getrandbits(1))

    available = [color for color in _ALL_COLORS if color != bg]
    border_color = random.choice(available)
    marker_colors = random.sample([color for color in available if color != border_color], n_regions)
    corners = random.sample(list(_CORNERS), n_regions)

    max_frame_h = min(18, h - 6)
    max_frame_w = min(18, w - 6)
    if max_frame_h < 8 or max_frame_w < 8:
        return None
    frame_h = random.randint(8, max_frame_h)
    frame_w = random.randint(8, max_frame_w)
    r0 = random.randint(2, h - frame_h - 3)
    c0 = random.randint(2, w - frame_w - 3)
    frame = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    corner_specs = _sample_marker_specs(frame, corners, want_overlap)
    if corner_specs is None:
        return None

    template_boxes = _place_templates((h, w), frame, marker_colors)
    if template_boxes is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[r0, c0 : c0 + frame_w] = border_color
    grid[r0 + frame_h - 1, c0 : c0 + frame_w] = border_color
    grid[r0 : r0 + frame_h, c0] = border_color
    grid[r0 : r0 + frame_h, c0 + frame_w - 1] = border_color

    color_specs: dict[int, dict[str, object]] = {}
    for color, corner in zip(marker_colors, corners):
        spec = corner_specs[corner]
        markers = spec["markers"]
        for mr, mc in markers:
            grid[mr, mc] = color
        color_specs[color] = {"corner": corner, "region": spec["region"], "markers": markers}

    for color, box in template_boxes.items():
        tr0, tc0, tr1, tc1 = box
        grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = color

    expected = _expected_output(grid, color_specs, template_boxes)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the corner-template frame task")
