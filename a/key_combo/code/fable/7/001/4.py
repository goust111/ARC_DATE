import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg, in_bounds


BG_COLOR = 0
RAY_COLOR = 3
PORTAL_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
MAX_SIDE = 30
GEN_RETRIES = 400


def _normalize_bbox(bbox):
    return tuple(map(int, bbox))


def _infer_marker(grid: np.ndarray, bg: int, coords: np.ndarray):
    cells = [tuple(map(int, rc)) for rc in coords]
    assert len(cells) == 2, "green marker must contain exactly two cells"
    (r0, c0), (r1, c1) = cells
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "green marker must be a 2-cell segment"

    candidates = []
    for tail, front in ((cells[0], cells[1]), (cells[1], cells[0])):
        dr = int(front[0] - tail[0])
        dc = int(front[1] - tail[1])
        nr, nc = front[0] + dr, front[1] + dc
        br, bc = tail[0] - dr, tail[1] - dc
        if not in_bounds(grid.shape, nr, nc):
            continue
        if int(grid[nr, nc]) != bg:
            continue
        if in_bounds(grid.shape, br, bc) and int(grid[br, bc]) == bg:
            continue
        candidates.append((front, (dr, dc)))

    assert len(candidates) == 1, "green marker direction must be uniquely inferable"
    return candidates[0]


def _parse_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= MAX_SIDE, f"grid exceeds {MAX_SIDE}x{MAX_SIDE}"

    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    marker = None
    by_color = {}
    for comp in comps:
        color = int(comp["color"])
        bbox = _normalize_bbox(comp["bbox"])
        h = bbox_h(bbox)
        w = bbox_w(bbox)
        area = int(comp["area"])

        if color == RAY_COLOR:
            assert marker is None, "expected exactly one green marker component"
            assert area == 2, "green marker must contain exactly two cells"
            marker = comp
            continue

        assert area == h * w, "each portal must be a solid rectangle"
        assert h >= 2 and w >= 2, "portal rectangles must be at least 2x2"
        by_color.setdefault(color, []).append(bbox)

    assert marker is not None, "missing green marker"
    assert by_color, "expected at least one portal pair"

    front, direction = _infer_marker(g, bg, marker["coords"])

    rects = []
    portal_idx = np.full(g.shape, -1, dtype=int)
    for color in sorted(by_color):
        pair = sorted(by_color[color])
        assert len(pair) == 2, "each portal color must appear on exactly two rectangles"
        assert bbox_h(pair[0]) == bbox_h(pair[1]) and bbox_w(pair[0]) == bbox_w(pair[1]), (
            "paired portals must have the same size"
        )
        start = len(rects)
        for bbox in pair:
            r0, c0, r1, c1 = bbox
            assert np.all(portal_idx[r0 : r1 + 1, c0 : c1 + 1] == -1), "portal rectangles may not overlap"
            portal_idx[r0 : r1 + 1, c0 : c1 + 1] = len(rects)
            rects.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "mate": None,
                }
            )
        rects[start]["mate"] = start + 1
        rects[start + 1]["mate"] = start

    return {
        "grid": g,
        "bg": int(bg),
        "front": tuple(map(int, front)),
        "direction": tuple(map(int, direction)),
        "rects": rects,
        "portal_idx": portal_idx,
    }


def _teleport(parsed, rect_id: int, hit):
    rect = parsed["rects"][int(rect_id)]
    mate = parsed["rects"][int(rect["mate"])]
    (dr, dc) = parsed["direction"]
    r, c = map(int, hit)
    r0, c0, r1, c1 = rect["bbox"]
    mr0, mc0, mr1, mc1 = mate["bbox"]

    if dr == 1:
        assert r == r0, "downward ray must enter the top edge"
        offset = c - c0
        exit_edge = (mr1, mc0 + offset)
        offset_size = bbox_w(rect["bbox"])
    elif dr == -1:
        assert r == r1, "upward ray must enter the bottom edge"
        offset = c - c0
        exit_edge = (mr0, mc0 + offset)
        offset_size = bbox_w(rect["bbox"])
    elif dc == 1:
        assert c == c0, "rightward ray must enter the left edge"
        offset = r - r0
        exit_edge = (mr0 + offset, mc1)
        offset_size = bbox_h(rect["bbox"])
    else:
        assert dc == -1, "ray direction must be cardinal"
        assert c == c1, "leftward ray must enter the right edge"
        offset = r - r0
        exit_edge = (mr0 + offset, mc0)
        offset_size = bbox_h(rect["bbox"])

    return tuple(map(int, exit_edge)), {
        "color": int(rect["color"]),
        "entry_hit": (int(r), int(c)),
        "entry_bbox": rect["bbox"],
        "exit_bbox": mate["bbox"],
        "exit_edge": tuple(map(int, exit_edge)),
        "offset": int(offset),
        "offset_size": int(offset_size),
    }


def _trace_stepwise(parsed, *, collect_hits: bool = False):
    g = parsed["grid"]
    out = g.copy()
    portal_idx = parsed["portal_idx"]
    dr, dc = parsed["direction"]
    r, c = parsed["front"]
    hits = []
    seen = set()

    while True:
        state = (int(r), int(c))
        assert state not in seen, "portal layout must not create an infinite loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        rect_id = int(portal_idx[nr, nc])
        if rect_id >= 0:
            (r, c), hit = _teleport(parsed, rect_id, (nr, nc))
            hits.append(hit)
            continue

        out[nr, nc] = RAY_COLOR
        r, c = nr, nc

    if collect_hits:
        return out, hits
    return out


def _trace_segment_oracle(parsed):
    g = parsed["grid"]
    out = g.copy()
    portal_idx = parsed["portal_idx"]
    dr, dc = parsed["direction"]
    r, c = parsed["front"]
    hits = []
    seen = set()
    height, width = g.shape

    while True:
        state = (int(r), int(c))
        assert state not in seen, "portal layout must not create an infinite loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if dr == 0:
            while 0 <= nc < width and int(portal_idx[nr, nc]) < 0:
                out[nr, nc] = RAY_COLOR
                nc += dc
            if not (0 <= nc < width):
                break
        else:
            while 0 <= nr < height and int(portal_idx[nr, nc]) < 0:
                out[nr, nc] = RAY_COLOR
                nr += dr
            if not (0 <= nr < height):
                break

        rect_id = int(portal_idx[nr, nc])
        (r, c), hit = _teleport(parsed, rect_id, (nr, nc))
        hits.append(hit)

    return out, hits


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Trace the green ray across matched portal rectangles and add every visible trajectory segment.

    Input structure:
    - The background contains several solid rectangles; each non-green portal color appears on exactly two rectangles of the same size.
    - A green two-cell short segment gives the start line, and its uniquely open end indicates the launch direction.
    - The ray always travels in a cardinal direction and every collision is with one portal rectangle edge or with the canvas boundary.

    Rule:
    - Extend the green ray straight ahead, painting only the background cells it traverses.
    - When the next step would enter a portal rectangle, locate the same-color partner rectangle.
    - Keep the entrance offset along the struck edge, emerge from the partner's opposite edge with the same direction, and continue; portal cells themselves stay unchanged.
    - Stop when the ray would leave the grid.

    Key insight:
    - The rectangles are not blockers but paired teleporters, and the missing links between visible line segments are determined jointly by color pairing, preserved edge offset, and opposite-side exit.
    """
    parsed = _parse_grid(np.asarray(input_grid, dtype=int))
    return _trace_stepwise(parsed)


def _can_place(occ: np.ndarray, top: int, left: int, height: int, width: int, *, pad: int = 1) -> bool:
    h, w = occ.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(np.any(occ[r0:r1, c0:c1]))


def _mark_rect(occ: np.ndarray, top: int, left: int, height: int, width: int):
    occ[top : top + height, left : left + width] = True


def _sample_height():
    return random.choice([2, 3, 3, 4, 4, 5])


def _sample_width():
    return random.choice([2, 2, 3, 3, 4])


def _place_distractor_pair(occ: np.ndarray, width_limit: int, path_rows, color: int):
    grid_h = occ.shape[0]
    for _ in range(160):
        height = _sample_height()
        width = _sample_width()
        tops = [top for top in range(0, grid_h - height + 1) if all(not (top <= row < top + height) for row in path_rows)]
        if not tops:
            continue
        left_max = width_limit - width - 1
        if left_max < 1:
            continue
        for _ in range(80):
            top_a = random.choice(tops)
            top_b = random.choice(tops)
            left_a = random.randint(1, left_max)
            left_b = random.randint(1, left_max)
            if not _can_place(occ, top_a, left_a, height, width):
                continue
            if not _can_place(occ, top_b, left_b, height, width):
                continue
            _mark_rect(occ, top_a, left_a, height, width)
            _mark_rect(occ, top_b, left_b, height, width)
            return [
                (top_a, left_a, height, width, color),
                (top_b, left_b, height, width, color),
            ]
    return None


def _build_canonical():
    grid_h = random.randint(11, 18)
    occ = np.zeros((grid_h, MAX_SIDE), dtype=bool)

    used_pair_count = random.choice([2, 2, 3])
    distractor_count = random.choice([0, 1, 1, 2])
    colors = random.sample(PORTAL_COLORS, used_pair_count + distractor_count)
    used_colors = colors[:used_pair_count]
    distractor_colors = colors[used_pair_count:]

    current_row = random.randint(1, grid_h - 2)
    current_col = 1
    path_rows = [current_row]
    rect_specs = []
    saw_shift = False
    saw_inner_offset = False

    for pair_index, color in enumerate(used_colors):
        remaining = used_pair_count - pair_index - 1
        placed = False
        for _ in range(160):
            height = _sample_height()
            width = _sample_width()
            offsets = [
                offset
                for offset in range(height)
                if 0 <= current_row - offset and current_row - offset + height <= grid_h
            ]
            random.shuffle(offsets)
            for offset in offsets:
                top_a = current_row - offset
                left_a = current_col + random.randint(2, 4)
                left_b = left_a + width + random.randint(1, 4)
                next_col = left_b + width - 1
                reserve = remaining * 6 + 2
                if next_col + reserve > MAX_SIDE - 1:
                    continue
                if not _can_place(occ, top_a, left_a, height, width):
                    continue

                candidate_tops = list(range(0, grid_h - height + 1))
                random.shuffle(candidate_tops)
                for top_b in candidate_tops:
                    next_row = top_b + offset
                    if remaining > 0 and not (1 <= next_row <= grid_h - 2):
                        continue
                    if not _can_place(occ, top_b, left_b, height, width):
                        continue

                    _mark_rect(occ, top_a, left_a, height, width)
                    _mark_rect(occ, top_b, left_b, height, width)
                    rect_specs.append((top_a, left_a, height, width, color))
                    rect_specs.append((top_b, left_b, height, width, color))
                    current_row = next_row
                    current_col = next_col
                    path_rows.append(current_row)
                    saw_shift |= top_a != top_b
                    saw_inner_offset |= 0 < offset < height - 1
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if not saw_shift or not saw_inner_offset:
        return None

    grid_w = current_col + random.randint(2, 4) + 1
    if not (12 <= grid_w <= MAX_SIDE):
        return None

    for color in distractor_colors:
        placed = _place_distractor_pair(occ, grid_w, path_rows, color)
        if placed is None:
            return None
        rect_specs.extend(placed)

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for top, left, height, width, color in rect_specs:
        grid[top : top + height, left : left + width] = color
    grid[path_rows[0], 0:2] = RAY_COLOR
    return grid


def _validate_generated_instance(grid: np.ndarray):
    parsed = _parse_grid(grid)
    out_main, hits_main = _trace_stepwise(parsed, collect_hits=True)
    out_oracle, hits_oracle = _trace_segment_oracle(parsed)
    assert np.array_equal(out_main, out_oracle), "oracle mismatch"
    assert len(hits_main) == len(hits_oracle) >= 2, "expected multiple portal traversals"

    dr, dc = parsed["direction"]
    orth_shift = any(
        (hit["entry_hit"][0] != hit["exit_edge"][0]) if dc != 0 else (hit["entry_hit"][1] != hit["exit_edge"][1])
        for hit in hits_main
    )
    assert orth_shift, "generator must include a row/column shift through a portal pair"
    assert any(0 < hit["offset"] < hit["offset_size"] - 1 for hit in hits_main), (
        "generator must include a non-edge offset case"
    )

    added = (out_main == RAY_COLOR) & (grid != RAY_COLOR)
    assert int(np.count_nonzero(added)) >= 4, "trajectory should add visible background segments"
    assert np.array_equal(main(grid), out_main), "main should match the validated trace"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_canonical()
        if grid is None:
            continue
        rotated = np.rot90(grid, k=random.randint(0, 3)).copy()
        try:
            _validate_generated_instance(rotated)
        except AssertionError:
            continue
        return rotated
    raise RuntimeError("failed to generate a valid portal-ray ARC input")
