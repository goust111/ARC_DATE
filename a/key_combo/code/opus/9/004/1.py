from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
RECT_COLORS = (1, 2, 3, 4)
TAIL_DIRECTIONS = ("up", "down", "left", "right")
DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_TAIL = 4


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR_STEP.values():
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
) -> list[dict]:
    assert connectivity == 4
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR_STEP.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _full_rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _touches_all_sides(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(
        mask[r0, c0 : c1 + 1].any()
        and mask[r1, c0 : c1 + 1].any()
        and mask[r0 : r1 + 1, c0].any()
        and mask[r0 : r1 + 1, c1].any()
    )


def _decode_component(component: dict, shape: tuple[int, int]) -> dict:
    mask = np.zeros(shape, dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0], coords[:, 1]] = True
    r0, c0, r1, c1 = map(int, component["bbox"])

    def build_candidate(direction: str, k: int) -> dict | None:
        if not (1 <= int(k) <= MAX_TAIL):
            return None
        body_mask = mask.copy()
        if direction == "up":
            body_mask[r0 : r0 + k, :] = False
            expected = (r0 + k, c0, r1, c1)
        elif direction == "down":
            body_mask[r1 - k + 1 : r1 + 1, :] = False
            expected = (r0, c0, r1 - k, c1)
        elif direction == "left":
            body_mask[:, c0 : c0 + k] = False
            expected = (r0, c0 + k, r1, c1)
        else:
            body_mask[:, c1 - k + 1 : c1 + 1] = False
            expected = (r0, c0, r1, c1 - k)

        body_coords = np.argwhere(body_mask)
        if body_coords.size == 0:
            return None
        body_box = bbox_coords(body_coords)
        if body_box != expected:
            return None
        if bbox_h(body_box) < 4 or bbox_w(body_box) < 4:
            return None
        if not _touches_all_sides(body_mask, body_box):
            return None
        body_crop = body_mask[body_box[0] : body_box[2] + 1, body_box[1] : body_box[3] + 1]
        if not _mask_connected(body_crop):
            return None

        if direction == "up":
            col = None
            for rr in range(r0, r0 + k):
                cols = np.flatnonzero(mask[rr])
                if len(cols) != 1:
                    return None
                cc = int(cols[0])
                col = cc if col is None else col
                if cc != col or not body_mask[body_box[0], cc]:
                    return None
        elif direction == "down":
            col = None
            for rr in range(r1 - k + 1, r1 + 1):
                cols = np.flatnonzero(mask[rr])
                if len(cols) != 1:
                    return None
                cc = int(cols[0])
                col = cc if col is None else col
                if cc != col or not body_mask[body_box[2], cc]:
                    return None
        elif direction == "left":
            row = None
            for cc in range(c0, c0 + k):
                rows = np.flatnonzero(mask[:, cc])
                if len(rows) != 1:
                    return None
                rr = int(rows[0])
                row = rr if row is None else row
                if rr != row or not body_mask[rr, body_box[1]]:
                    return None
        else:
            row = None
            for cc in range(c1 - k + 1, c1 + 1):
                rows = np.flatnonzero(mask[:, cc])
                if len(rows) != 1:
                    return None
                rr = int(rows[0])
                row = rr if row is None else row
                if rr != row or not body_mask[rr, body_box[3]]:
                    return None

        return {
            "color": int(component["color"]),
            "bbox": body_box,
            "tail_len": int(k),
            "tail_dir": direction,
        }

    candidates: list[dict] = []

    run_col = None
    up_len = 0
    for rr in range(r0, r1 + 1):
        cols = np.flatnonzero(mask[rr])
        if len(cols) != 1:
            break
        cc = int(cols[0])
        if run_col is None:
            run_col = cc
        if cc != run_col:
            break
        up_len += 1
    cand = build_candidate("up", up_len)
    if cand is not None:
        candidates.append(cand)

    run_col = None
    down_len = 0
    for rr in range(r1, r0 - 1, -1):
        cols = np.flatnonzero(mask[rr])
        if len(cols) != 1:
            break
        cc = int(cols[0])
        if run_col is None:
            run_col = cc
        if cc != run_col:
            break
        down_len += 1
    cand = build_candidate("down", down_len)
    if cand is not None:
        candidates.append(cand)

    run_row = None
    left_len = 0
    for cc in range(c0, c1 + 1):
        rows = np.flatnonzero(mask[:, cc])
        if len(rows) != 1:
            break
        rr = int(rows[0])
        if run_row is None:
            run_row = rr
        if rr != run_row:
            break
        left_len += 1
    cand = build_candidate("left", left_len)
    if cand is not None:
        candidates.append(cand)

    run_row = None
    right_len = 0
    for cc in range(c1, c0 - 1, -1):
        rows = np.flatnonzero(mask[:, cc])
        if len(rows) != 1:
            break
        rr = int(rows[0])
        if run_row is None:
            run_row = rr
        if rr != run_row:
            break
        right_len += 1
    cand = build_candidate("right", right_len)
    if cand is not None:
        candidates.append(cand)

    assert len(candidates) == 1, "each color component must decode to one rectangle+tail"
    return candidates[0]


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4)
    rects = [_decode_component(comp, g.shape) for comp in comps]
    rects.sort(key=lambda rect: rect["color"])
    tail_lengths = [rect["tail_len"] for rect in rects]
    assert len(rects) in {3, 4}, "expected three or four colored rectangles"
    assert len(set(tail_lengths)) == len(tail_lengths), "tail lengths must be unique"
    return bg, rects


def _draw_output(rects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for rect in sorted(rects, key=lambda item: item["tail_len"], reverse=True):
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the overlapped rectangles, erase their tails, and redraw them by tail-length priority with grey interiors.

    Input structure:
    - A black background contains three or four differently colored axis-aligned rectangles.
    - Each rectangle has a same-color one-pixel-wide tail extending straight outward from one edge.
    - Rectangle bodies may overlap, so lower-priority rectangles can be partially hidden, but each color still forms one 4-connected component with its tail.

    Rule:
    - For each color component, identify the unique straight tail and remove it.
    - Recover the rectangle's full axis-aligned body bbox from the remaining visible body span, even where other rectangles occluded some cells.
    - Redraw all recovered rectangles from longest tail to shortest tail, using the rectangle color on the one-pixel border and grey on the strict interior.
    - Later rectangles overwrite earlier ones in overlap regions.

    Key insight:
    - The tail is not a displacement code here: its length encodes draw priority, while its shape isolates the rectangle's true bbox despite occlusion.
    """
    g = np.asarray(grid, dtype=int)
    bg, rects = _extract_rectangles(g)
    return _draw_output(rects, g.shape, bg)


def _rect_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _place_corner_overlap(
    anchor: tuple[int, int, int, int],
    height: int,
    width: int,
    corner: str,
    overlap_h: int,
    overlap_w: int,
) -> tuple[int, int, int, int]:
    ar0, ac0, ar1, ac1 = anchor
    if corner == "tl":
        nr1 = ar0 + overlap_h - 1
        nc1 = ac0 + overlap_w - 1
        nr0 = nr1 - height + 1
        nc0 = nc1 - width + 1
    elif corner == "tr":
        nr1 = ar0 + overlap_h - 1
        nc0 = ac1 - overlap_w + 1
        nr0 = nr1 - height + 1
        nc1 = nc0 + width - 1
    elif corner == "bl":
        nr0 = ar1 - overlap_h + 1
        nc1 = ac0 + overlap_w - 1
        nr1 = nr0 + height - 1
        nc0 = nc1 - width + 1
    else:
        nr0 = ar1 - overlap_h + 1
        nc0 = ac1 - overlap_w + 1
        nr1 = nr0 + height - 1
        nc1 = nc0 + width - 1
    return int(nr0), int(nc0), int(nr1), int(nc1)


def _random_chain_boxes(count: int) -> list[tuple[int, int, int, int]]:
    first_h = random.randint(5, 6)
    first_w = random.randint(5, 6)
    boxes = [(0, 0, first_h - 1, first_w - 1)]
    for idx in range(1, count):
        anchor = boxes[idx - 1]
        height = random.randint(4, 6)
        width = random.randint(4, 6)
        overlap_h = random.randint(2, min(height, bbox_h(anchor)) - 2)
        overlap_w = random.randint(2, min(width, bbox_w(anchor)) - 2)
        corner = random.choice(("tl", "tr", "bl", "br"))
        boxes.append(_place_corner_overlap(anchor, height, width, corner, overlap_h, overlap_w))

    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    shifted = []
    for r0, c0, r1, c1 in boxes:
        shifted.append((r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c))

    if random.random() < 0.5:
        max_c = max(box[3] for box in shifted)
        shifted = [(r0, max_c - c1, r1, max_c - c0) for r0, c0, r1, c1 in shifted]
    if random.random() < 0.5:
        max_r = max(box[2] for box in shifted)
        shifted = [(max_r - r1, c0, max_r - r0, c1) for r0, c0, r1, c1 in shifted]

    return shifted


def _shift_boxes_into_grid(boxes: list[tuple[int, int, int, int]]) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    union_h = max(box[2] for box in boxes) + 1
    union_w = max(box[3] for box in boxes) + 1
    pad = MAX_TAIL
    min_h = union_h + 2 * pad
    min_w = union_w + 2 * pad
    max_h = min(26, max(min_h, union_h + 2 * pad + 2))
    max_w = min(26, max(min_w, union_w + 2 * pad + 2))
    if min_h > 26 or min_w > 26:
        return None
    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    dr = random.randint(pad, height - union_h - pad)
    dc = random.randint(pad, width - union_w - pad)
    shifted = [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]
    return (height, width), shifted


def _render_bodies(rects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for rect in sorted(rects, key=lambda item: item["tail_len"], reverse=True):
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return grid


def _visible_body_mask(body_grid: np.ndarray, rect: dict) -> np.ndarray:
    r0, c0, r1, c1 = rect["bbox"]
    patch = body_grid[r0 : r1 + 1, c0 : c1 + 1] == int(rect["color"])
    return patch


def _edge_visibility_ok(patch: np.ndarray) -> bool:
    return bool(
        int(patch[0].sum()) >= 2
        and int(patch[-1].sum()) >= 2
        and int(patch[:, 0].sum()) >= 2
        and int(patch[:, -1].sum()) >= 2
    )


def _visible_body_global(body_grid: np.ndarray, rect: dict) -> np.ndarray:
    mask = np.zeros(body_grid.shape, dtype=bool)
    r0, c0, r1, c1 = rect["bbox"]
    patch = _visible_body_mask(body_grid, rect)
    mask[r0 : r1 + 1, c0 : c1 + 1] = patch
    return mask


def _other_body_touch_mask(shape: tuple[int, int], rects: list[dict], index: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for j, rect in enumerate(rects):
        if j == index:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _candidate_tail_cells(rect: dict, direction: str, anchor: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    length = int(rect["tail_len"])
    if direction == "up":
        return [(r0 - step, anchor) for step in range(1, length + 1)]
    if direction == "down":
        return [(r1 + step, anchor) for step in range(1, length + 1)]
    if direction == "left":
        return [(anchor, c0 - step) for step in range(1, length + 1)]
    return [(anchor, c1 + step) for step in range(1, length + 1)]


def _cells_touch_mask(cells: list[tuple[int, int]], mask: np.ndarray) -> bool:
    height, width = mask.shape
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return True
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc]:
                    return True
    return False


def _tail_options(rects: list[dict], index: int, body_grid: np.ndarray, bg: int) -> list[dict]:
    rect = rects[index]
    visible_global = _visible_body_global(body_grid, rect)
    visible_patch = _visible_body_mask(body_grid, rect)
    r0, c0, r1, c1 = rect["bbox"]
    other_bodies = _other_body_touch_mask(body_grid.shape, rects, index)
    options: list[dict] = []

    if int(visible_patch[0].sum()) >= 2:
        for cc in range(c0 + 1, c1):
            if not visible_global[r0, cc]:
                continue
            cells = _candidate_tail_cells(rect, "up", cc)
            if any(int(body_grid[r, c]) != int(bg) for r, c in cells):
                continue
            if _cells_touch_mask(cells, other_bodies):
                continue
            options.append({"tail_dir": "up", "tail_cells": cells})

    if int(visible_patch[-1].sum()) >= 2:
        for cc in range(c0 + 1, c1):
            if not visible_global[r1, cc]:
                continue
            cells = _candidate_tail_cells(rect, "down", cc)
            if any(int(body_grid[r, c]) != int(bg) for r, c in cells):
                continue
            if _cells_touch_mask(cells, other_bodies):
                continue
            options.append({"tail_dir": "down", "tail_cells": cells})

    if int(visible_patch[:, 0].sum()) >= 2:
        for rr in range(r0 + 1, r1):
            if not visible_global[rr, c0]:
                continue
            cells = _candidate_tail_cells(rect, "left", rr)
            if any(int(body_grid[r, c]) != int(bg) for r, c in cells):
                continue
            if _cells_touch_mask(cells, other_bodies):
                continue
            options.append({"tail_dir": "left", "tail_cells": cells})

    if int(visible_patch[:, -1].sum()) >= 2:
        for rr in range(r0 + 1, r1):
            if not visible_global[rr, c1]:
                continue
            cells = _candidate_tail_cells(rect, "right", rr)
            if any(int(body_grid[r, c]) != int(bg) for r, c in cells):
                continue
            if _cells_touch_mask(cells, other_bodies):
                continue
            options.append({"tail_dir": "right", "tail_cells": cells})

    random.shuffle(options)
    return options


def _assign_tails(rects: list[dict], body_grid: np.ndarray, bg: int) -> bool:
    options_by_rect = [_tail_options(rects, idx, body_grid, bg) for idx in range(len(rects))]
    if any(not opts for opts in options_by_rect):
        return False

    order = sorted(range(len(rects)), key=lambda idx: len(options_by_rect[idx]))
    occupied = np.zeros(body_grid.shape, dtype=bool)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for option in options_by_rect[idx]:
            cells = option["tail_cells"]
            if _cells_touch_mask(cells, occupied):
                continue
            for r, c in cells:
                occupied[r, c] = True
            rects[idx]["tail_dir"] = option["tail_dir"]
            rects[idx]["tail_cells"] = tuple(cells)
            if backtrack(pos + 1):
                return True
            for r, c in cells:
                occupied[r, c] = False
            rects[idx].pop("tail_dir", None)
            rects[idx].pop("tail_cells", None)
        return False

    return backtrack(0)


def _render_input(rects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = _render_bodies(rects, shape, bg)
    for rect in rects:
        for r, c in rect["tail_cells"]:
            grid[r, c] = int(rect["color"])
    return grid


def _render_output_for_order(rects: list[dict], order: tuple[int, ...], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for idx in order:
        rect = rects[idx]
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY
        out[r0, c0 : c1 + 1] = int(rect["color"])
        out[r1, c0 : c1 + 1] = int(rect["color"])
        out[r0 : r1 + 1, c0] = int(rect["color"])
        out[r0 : r1 + 1, c1] = int(rect["color"])
    return out


def _scene_semantics_ok(rects: list[dict], body_grid: np.ndarray, inp: np.ndarray, out: np.ndarray, bg: int) -> bool:
    hidden_counts = []
    for rect in rects:
        full_area = bbox_h(rect["bbox"]) * bbox_w(rect["bbox"])
        visible_area = int(_visible_body_mask(body_grid, rect).sum())
        hidden_counts.append(full_area - visible_area)
    if sum(count > 0 for count in hidden_counts) < 2:
        return False

    decoded_bg, decoded = _extract_rectangles(inp)
    if decoded_bg != int(bg):
        return False
    expected = sorted(
        [{"color": r["color"], "bbox": r["bbox"], "tail_len": r["tail_len"], "tail_dir": r["tail_dir"]} for r in rects],
        key=lambda item: item["color"],
    )
    if decoded != expected:
        return False
    if not np.array_equal(main(inp), out):
        return False

    correct_order = tuple(idx for idx, _ in sorted(enumerate(rects), key=lambda item: item[1]["tail_len"], reverse=True))
    distinct_outputs = set()
    for order in itertools.permutations(range(len(rects))):
        rendered = _render_output_for_order(rects, order, inp.shape, bg)
        distinct_outputs.add(rendered.tobytes())
    if len(distinct_outputs) < 2:
        return False

    pair_changes = 0
    correct = _render_output_for_order(rects, correct_order, inp.shape, bg)
    for i, j in itertools.combinations(range(len(rects)), 2):
        if _rect_intersection(rects[i]["bbox"], rects[j]["bbox"]) is None:
            continue
        swapped = list(correct_order)
        pi = swapped.index(i)
        pj = swapped.index(j)
        swapped[pi], swapped[pj] = swapped[pj], swapped[pi]
        alt = _render_output_for_order(rects, tuple(swapped), inp.shape, bg)
        if not np.array_equal(correct, alt):
            pair_changes += 1
    return pair_changes >= 1


def _generate_scene_attempt() -> np.ndarray | None:
    count = random.choice((3, 3, 4))
    relative_boxes = _random_chain_boxes(count)
    shifted = _shift_boxes_into_grid(relative_boxes)
    if shifted is None:
        return None
    shape, boxes = shifted

    colors = random.sample(list(RECT_COLORS), count)
    lengths = random.sample((1, 2, 3, 4), count)
    rects = [
        {
            "color": int(color),
            "bbox": box,
            "tail_len": int(length),
        }
        for color, box, length in zip(colors, boxes, lengths)
    ]

    body_grid = _render_bodies(rects, shape, BLACK)
    for rect in rects:
        visible = _visible_body_mask(body_grid, rect)
        if bbox_h(rect["bbox"]) < 4 or bbox_w(rect["bbox"]) < 4:
            return None
        if not _mask_connected(visible):
            return None
        if not _edge_visibility_ok(visible):
            return None
        if not _touches_all_sides(_visible_body_global(body_grid, rect), rect["bbox"]):
            return None

    if not _assign_tails(rects, body_grid, BLACK):
        return None

    inp = _render_input(rects, shape, BLACK)
    out = _draw_output(rects, shape, BLACK)
    if not _scene_semantics_ok(rects, body_grid, inp, out, BLACK):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-tail priority scene")
