from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
MARKER = 6
RECT_COLORS = (1, 2, 3, 4, 7)
DIM_OPTIONS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (4, 2),
    (2, 4),
    (4, 3),
    (3, 4),
)
PLACEMENT_KINDS = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")
GEN_RETRIES = 1000


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_color_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def bbox_contains(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def rect_shift(rect: dict, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = rect["size"]
    r0, c0, r1, c1 = rect["input_bbox"]
    grid_h, grid_w = shape
    dr = 0
    dc = 0
    if r0 == 0:
        dr += h
    if r1 == grid_h - 1:
        dr -= h
    if c0 == 0:
        dc += w
    if c1 == grid_w - 1:
        dc -= w
    return dr, dc


def shifted_bbox(rect: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    dr, dc = rect_shift(rect, shape)
    r0, c0, r1, c1 = rect["input_bbox"]
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def marker_count_in_bbox(markers: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> int:
    return sum(1 for cell in markers if bbox_contains(bbox, cell))


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    comps = find_color_components(grid, set(RECT_COLORS))
    assert 4 <= len(comps) <= 6
    rects: list[dict] = []
    h, w = grid.shape
    for comp in comps:
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        rh = r1 - r0 + 1
        rw = c1 - c0 + 1
        assert comp["area"] == rh * rw
        assert r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        rects.append(
            {
                "color": int(comp["color"]),
                "input_bbox": bbox,
                "size": (rh, rw),
            }
        )
    return rects


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    rects = _parse_rectangles(g)
    by_color: dict[int, list[dict]] = {}
    for rect in rects:
        by_color.setdefault(int(rect["color"]), []).append(rect)

    moving_colors = [color for color, items in by_color.items() if len(items) > 1]
    assert len(moving_colors) == 1
    moving_color = int(moving_colors[0])

    markers = {(int(r), int(c)) for r, c in np.argwhere(g == MARKER)}
    assert 3 <= len(markers) <= 5

    final_rects: list[dict] = []
    occupied = np.zeros((h, w), dtype=int)
    for rect in rects:
        final_bbox = shifted_bbox(rect, g.shape) if rect["color"] == moving_color else rect["input_bbox"]
        fr0, fc0, fr1, fc1 = final_bbox
        assert 0 <= fr0 <= fr1 < h and 0 <= fc0 <= fc1 < w
        overlap = occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] != 0
        assert not bool(overlap.any())
        occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = int(rect["color"])
        current_markers = marker_count_in_bbox(markers, final_bbox)
        kept = current_markers <= 1
        final_rect = {
            "color": rect["color"],
            "input_bbox": rect["input_bbox"],
            "final_bbox": final_bbox,
            "size": rect["size"],
            "marker_count": current_markers,
            "kept": kept,
            "moved": rect["color"] == moving_color,
        }
        final_rects.append(final_rect)

    out = np.zeros_like(g)
    for rect in final_rects:
        if not rect["kept"]:
            continue
        r0, c0, r1, c1 = rect["final_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])

    for r, c in markers:
        out[r, c] = MARKER

    moved = [rect for rect in final_rects if rect["moved"]]
    assert any(rect["marker_count"] >= 2 for rect in moved)
    assert any(rect["marker_count"] <= 1 for rect in moved)
    return out, final_rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the repeated border-touching rectangles inward, then erase any rectangle that contains too many markers.

    Input structure:
    - Black background with 4-6 solid monochrome rectangles in colors 1,2,3,4,7, each touching at least one grid edge.
    - Exactly one rectangle color appears in 2-3 disconnected components; all other rectangle colors appear once.
    - Several isolated magenta marker pixels lie in the interior and are outside every rectangle in the input.

    Rule:
    - Identify the repeated color and translate each of its rectangles inward by its own touched-edge dimensions: top/bottom by height and left/right by width, combining both directions for corner rectangles.
    - Leave all non-repeated rectangles fixed, then count how many magenta markers lie inside each rectangle's post-translation area.
    - Erase every rectangle whose post-translation marker count is at least 2, keep rectangles with 0 or 1 markers, and always preserve every magenta marker pixel.

    Key insight:
    - The markers are judged against the rectangles after the repeated-color translation, so movement changes marker ownership before the erase threshold is tested.
    """
    out, _ = _solve(grid)
    return out


def _random_bbox(kind: str, size: tuple[int, int], shape: tuple[int, int], rng: random.Random):
    h, w = size
    grid_h, grid_w = shape
    if kind == "top":
        if grid_w - w - 2 < 1:
            return None
        c0 = rng.randint(1, grid_w - w - 2)
        return 0, c0, h - 1, c0 + w - 1
    if kind == "bottom":
        if grid_w - w - 2 < 1:
            return None
        c0 = rng.randint(1, grid_w - w - 2)
        return grid_h - h, c0, grid_h - 1, c0 + w - 1
    if kind == "left":
        if grid_h - h - 2 < 1:
            return None
        r0 = rng.randint(1, grid_h - h - 2)
        return r0, 0, r0 + h - 1, w - 1
    if kind == "right":
        if grid_h - h - 2 < 1:
            return None
        r0 = rng.randint(1, grid_h - h - 2)
        return r0, grid_w - w, r0 + h - 1, grid_w - 1
    if kind == "tl":
        return 0, 0, h - 1, w - 1
    if kind == "tr":
        return 0, grid_w - w, h - 1, grid_w - 1
    if kind == "bl":
        return grid_h - h, 0, grid_h - 1, w - 1
    if kind == "br":
        return grid_h - h, grid_w - w, grid_h - 1, grid_w - 1
    return None


def _candidate_bbox(kind: str, size: tuple[int, int], shape: tuple[int, int], rng: random.Random):
    if kind in {"top", "bottom", "left", "right"}:
        for _ in range(20):
            bbox = _random_bbox(kind, size, shape, rng)
            if bbox is not None:
                return bbox
        return None
    return _random_bbox(kind, size, shape, rng)


def _cell_options_with_gap(bbox: tuple[int, int, int, int], blocked: set[tuple[int, int]], margin: int) -> list[tuple[int, int]]:
    choices = []
    for cell in bbox_cells(bbox):
        if cell in blocked:
            continue
        ok = True
        for br, bc in blocked:
            if abs(cell[0] - br) + abs(cell[1] - bc) <= margin:
                ok = False
                break
        if ok:
            choices.append(cell)
    return choices


def _pick_cells_in_bbox(
    bbox: tuple[int, int, int, int],
    count: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    cells = [cell for cell in bbox_cells(bbox) if cell not in blocked]
    valid = []
    for combo in itertools.combinations(cells, count):
        ok = True
        for a, b in itertools.combinations(combo, 2):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1:
                ok = False
                break
        if not ok:
            continue
        for cell in combo:
            if any(abs(cell[0] - br) + abs(cell[1] - bc) <= 1 for br, bc in blocked):
                ok = False
                break
        if ok:
            valid.append(combo)
    if not valid:
        return None
    return list(rng.choice(valid))


def _free_interior_cells(
    shape: tuple[int, int],
    blocked_bboxes: list[tuple[int, int, int, int]],
    blocked_markers: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cell = (r, c)
            if any(bbox_contains(bbox, cell) for bbox in blocked_bboxes):
                continue
            if any(abs(r - mr) + abs(c - mc) <= 1 for mr, mc in blocked_markers):
                continue
            out.append(cell)
    return out


def _spec_to_grid(shape: tuple[int, int], rects: list[dict], markers: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["input_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    for r, c in markers:
        assert grid[r, c] == BG
        grid[r, c] = MARKER
    return grid


def _all_input_cells(rects: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rect in rects:
        cells.update(bbox_cells(rect["input_bbox"]))
    return cells


def _validate_generated(shape: tuple[int, int], rects: list[dict], markers: set[tuple[int, int]], grid: np.ndarray) -> None:
    repeated = [color for color, count in Counter(rect["color"] for rect in rects).items() if count > 1]
    assert len(repeated) == 1
    moving_color = repeated[0]

    assert 4 <= len(rects) <= 6
    assert 3 <= len(markers) <= 5
    assert all(0 < r < shape[0] - 1 and 0 < c < shape[1] - 1 for r, c in markers)

    final_bboxes = []
    moving_rects = []
    pre_counts = []
    post_counts = []
    for rect in rects:
        in_bbox = rect["input_bbox"]
        out_bbox = shifted_bbox(rect, shape) if rect["color"] == moving_color else in_bbox
        final_bboxes.append(out_bbox)
        if rect["color"] == moving_color:
            moving_rects.append(out_bbox)
        pre_counts.append(marker_count_in_bbox(markers, in_bbox))
        post_counts.append(marker_count_in_bbox(markers, out_bbox))
        assert marker_count_in_bbox(markers, in_bbox) == 0

    for i, a in enumerate(final_bboxes):
        for b in final_bboxes[i + 1 :]:
            assert not bboxes_overlap(a, b, pad=0)

    changed = any(pre != post for pre, post in zip(pre_counts, post_counts))
    assert changed
    assert any(count >= 2 for count in post_counts)
    assert any(count <= 1 for count in post_counts)
    assert any(any(bbox_contains(bbox, marker) for bbox in moving_rects) for marker in markers)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.count_nonzero(out == MARKER) == len(markers)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 20)
    w = rng.randint(14, 20)
    shape = (h, w)

    repeated_count = rng.choice((2, 3))
    rect_count = rng.randint(max(4, repeated_count + 1), 6)
    single_count = rect_count - repeated_count

    repeated_color = rng.choice(RECT_COLORS)
    singleton_colors = rng.sample([color for color in RECT_COLORS if color != repeated_color], single_count)
    colors = [repeated_color] * repeated_count + singleton_colors
    rng.shuffle(colors)

    rects: list[dict] = []
    repeated_kinds = rng.sample(PLACEMENT_KINDS, repeated_count)
    remaining_pool = list(PLACEMENT_KINDS)
    rng.shuffle(remaining_pool)
    kinds = repeated_kinds + remaining_pool[:single_count]
    rng.shuffle(kinds)

    for color, kind in zip(colors, kinds):
        placed = None
        for _ in range(80):
            size = rng.choice(DIM_OPTIONS)
            bbox = _candidate_bbox(kind, size, shape, rng)
            if bbox is None:
                continue
            if any(bboxes_overlap(bbox, other["input_bbox"], pad=1) for other in rects):
                continue
            placed = {"color": color, "input_bbox": bbox, "size": size}
            rects.append(placed)
            break
        if placed is None:
            return None

    counts = Counter(rect["color"] for rect in rects)
    if sorted(counts.values(), reverse=True)[0] != repeated_count:
        return None
    if len([color for color, cnt in counts.items() if cnt > 1]) != 1:
        return None

    moving_color = repeated_color
    final_bboxes = []
    for rect in rects:
        final_bbox = shifted_bbox(rect, shape) if rect["color"] == moving_color else rect["input_bbox"]
        r0, c0, r1, c1 = final_bbox
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return None
        final_bboxes.append(final_bbox)

    for i, a in enumerate(final_bboxes):
        for b in final_bboxes[i + 1 :]:
            if bboxes_overlap(a, b, pad=0):
                return None

    moved_rects = [rect for rect in rects if rect["color"] == moving_color]
    rng.shuffle(moved_rects)
    victim = moved_rects[0]
    support = moved_rects[1]

    markers: set[tuple[int, int]] = set()
    input_cells = _all_input_cells(rects)

    victim_final = shifted_bbox(victim, shape)
    picked = _pick_cells_in_bbox(victim_final, 2, input_cells | markers, rng)
    if picked is None:
        return None
    markers.update(picked)

    if repeated_count == 3 or rng.random() < 0.6:
        support_final = shifted_bbox(support, shape)
        support_pick = _pick_cells_in_bbox(support_final, 1, input_cells | markers, rng)
        if support_pick is None:
            return None
        markers.update(support_pick)

    target_marker_total = rng.randint(3, 5)
    blocked_bboxes = [rect["input_bbox"] for rect in rects] + final_bboxes
    free_cells = _free_interior_cells(shape, blocked_bboxes, markers)
    rng.shuffle(free_cells)
    for cell in free_cells:
        if len(markers) >= target_marker_total:
            break
        markers.add(cell)

    if not (3 <= len(markers) <= 5):
        return None

    grid = _spec_to_grid(shape, rects, markers)

    try:
        _validate_generated(shape, rects, markers, grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
