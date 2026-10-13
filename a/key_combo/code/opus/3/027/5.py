from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _filled_rect(mask: np.ndarray) -> bool:
    return bool(mask.size) and bool(np.all(mask))


def _tail_candidates(mask: np.ndarray) -> list[tuple[BBox, tuple[int, int], int]]:
    h, w = mask.shape
    candidates: list[tuple[BBox, tuple[int, int], int]] = []

    for tail_len in range(1, h):
        body = mask[tail_len:, :]
        tail = mask[:tail_len, :]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and _filled_rect(body) and np.all(tail.sum(axis=1) == 1):
            cols = np.argmax(tail, axis=1)
            if np.all(cols == cols[0]):
                candidates.append(((tail_len, 0, h - 1, w - 1), (-1, 0), tail_len))

        body = mask[: h - tail_len, :]
        tail = mask[h - tail_len :, :]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and _filled_rect(body) and np.all(tail.sum(axis=1) == 1):
            cols = np.argmax(tail, axis=1)
            if np.all(cols == cols[0]):
                candidates.append(((0, 0, h - tail_len - 1, w - 1), (1, 0), tail_len))

    for tail_len in range(1, w):
        body = mask[:, tail_len:]
        tail = mask[:, :tail_len]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and _filled_rect(body) and np.all(tail.sum(axis=0) == 1):
            rows = np.argmax(tail, axis=0)
            if np.all(rows == rows[0]):
                candidates.append(((0, tail_len, h - 1, w - 1), (0, -1), tail_len))

        body = mask[:, : w - tail_len]
        tail = mask[:, w - tail_len :]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and _filled_rect(body) and np.all(tail.sum(axis=0) == 1):
            rows = np.argmax(tail, axis=0)
            if np.all(rows == rows[0]):
                candidates.append(((0, 0, h - 1, w - tail_len - 1), (0, 1), tail_len))

    return candidates


def _parse_rectangle(component: dict) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    mask = _component_mask(component)
    if _filled_rect(mask):
        body_bbox = (r0, c0, r1, c1)
        return {
            "color": int(component["color"]),
            "body_bbox": body_bbox,
            "final_bbox": body_bbox,
            "tail_dir": None,
            "tail_len": 0,
        }

    candidates = _tail_candidates(mask)
    assert len(candidates) == 1, "component must be a solid rectangle with at most one straight tail"
    (br0, bc0, br1, bc1), direction, tail_len = candidates[0]
    body_bbox = (r0 + br0, c0 + bc0, r0 + br1, c0 + bc1)
    dr, dc = direction
    final_bbox = (
        body_bbox[0] + dr * tail_len,
        body_bbox[1] + dc * tail_len,
        body_bbox[2] + dr * tail_len,
        body_bbox[3] + dc * tail_len,
    )
    return {
        "color": int(component["color"]),
        "body_bbox": body_bbox,
        "final_bbox": final_bbox,
        "tail_dir": direction,
        "tail_len": tail_len,
    }


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _pair_relation(a: BBox, b: BBox) -> str:
    row_overlap = _intervals_overlap(a[0], a[2], b[0], b[2])
    col_overlap = _intervals_overlap(a[1], a[3], b[1], b[3])
    if not row_overlap and not col_overlap:
        return "diagonal"
    if row_overlap and not col_overlap:
        return "horizontal"
    if not row_overlap and col_overlap:
        return "vertical"
    return "overlap"


def _facing_corners(a: BBox, b: BBox) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if _pair_relation(a, b) != "diagonal":
        return None
    if a[2] < b[0] and a[3] < b[1]:
        return (a[2], a[3]), (b[0], b[1])
    if a[2] < b[0] and b[3] < a[1]:
        return (a[2], a[1]), (b[0], b[3])
    if b[2] < a[0] and a[3] < b[1]:
        return (a[0], a[3]), (b[2], b[1])
    if b[2] < a[0] and b[3] < a[1]:
        return (a[0], a[1]), (b[2], b[3])
    return None


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    dr = 1 if end[0] > start[0] else -1
    dc = 1 if end[1] > start[1] else -1
    assert abs(end[0] - start[0]) == abs(end[1] - start[1]), "facing corners must lie on a 45-degree line"
    cells: list[tuple[int, int]] = []
    r, c = start[0] + dr, start[1] + dc
    while (r, c) != end:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _paint_rect(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = [_parse_rectangle(component) for component in _find_components(g, bg)]
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_color[int(obj["color"])].append(obj)
    assert by_color and all(len(group) == 2 for group in by_color.values()), "each foreground color must form exactly one pair"

    out = np.full_like(g, bg)
    occupied = np.zeros_like(g, dtype=bool)
    for obj in objects:
        r0, c0, r1, c1 = obj["final_bbox"]
        assert 0 <= r0 <= r1 < g.shape[0] and 0 <= c0 <= c1 < g.shape[1], "slid rectangle moved out of bounds"
        assert not np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]), "slid rectangles must not overlap"
        _paint_rect(out, obj["final_bbox"], int(obj["color"]))
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    pair_infos: list[dict] = []
    for color, group in by_color.items():
        first, second = group
        input_relation = _pair_relation(first["body_bbox"], second["body_bbox"])
        final_relation = _pair_relation(first["final_bbox"], second["final_bbox"])
        assert input_relation != "overlap" and final_relation != "overlap", "same-color rectangles must remain distinct"
        info = {
            "color": int(color),
            "input_relation": input_relation,
            "final_relation": final_relation,
            "tail_count": int(sum(obj["tail_len"] > 0 for obj in group)),
        }
        pair_infos.append(info)

        corners = _facing_corners(first["final_bbox"], second["final_bbox"])
        if corners is None:
            continue
        start, end = corners
        for r, c in _line_cells(start, end):
            assert int(out[r, c]) == bg, "diagonal connector may only paint background"
            out[r, c] = int(color)

    return out, objects, pair_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide tailed rectangles by their tail vectors, then connect only the same-color pairs that become diagonally offset.

    Input structure:
    - The background is black.
    - Each foreground color appears as exactly two solid rectangles, and some rectangles may have a same-color 1-pixel-wide straight tail extending from one side.
    - Rectangles are separated so that each same-color rectangle-plus-tail shape is its own 4-connected component.

    Rule:
    - For every component, identify the solid rectangular body and any single straight tail.
    - Remove the tail, then translate the rectangle body along the tail direction by exactly the tail length; rectangles without tails stay fixed.
    - After all moves, inspect each same-color pair: if their new bounding boxes are separated in both rows and columns, connect the nearest facing corners with a 45-degree diagonal drawn only through background cells.

    Key insight:
    - The decisive geometry is the post-slide layout: tails act only as displacement vectors, and the diagonal connectors are determined from the rectangles' new positions rather than their original ones.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _rect_and_tail_cells(rect: dict) -> list[tuple[int, int]]:
    r0, c0, h, w = rect["r"], rect["c"], rect["h"], rect["w"]
    cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    direction = rect.get("tail_dir")
    tail_len = int(rect.get("tail_len", 0))
    if direction is None or tail_len == 0:
        return cells

    anchor = int(rect["anchor"])
    if direction == (-1, 0):
        col = c0 + anchor
        cells.extend((r0 - k, col) for k in range(1, tail_len + 1))
    elif direction == (1, 0):
        col = c0 + anchor
        cells.extend((r0 + h - 1 + k, col) for k in range(1, tail_len + 1))
    elif direction == (0, -1):
        row = r0 + anchor
        cells.extend((row, c0 - k) for k in range(1, tail_len + 1))
    else:
        row = r0 + anchor
        cells.extend((row, c0 + w - 1 + k) for k in range(1, tail_len + 1))
    return cells


def _build_patch(color: int, rects: list[dict]) -> np.ndarray:
    parts = [_rect_and_tail_cells(rect) for rect in rects]
    all_cells = [cell for part in parts for cell in part]
    r_min = min(r for r, _ in all_cells)
    c_min = min(c for _, c in all_cells)
    r_max = max(r for r, _ in all_cells)
    c_max = max(c for _, c in all_cells)
    patch = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=int)
    for part in parts:
        for r, c in part:
            patch[r - r_min, c - c_min] = int(color)
    return patch


def _rand_hw() -> tuple[int, int]:
    return random.randint(2, 3), random.randint(2, 4)


def _rotate_patch(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, random.randrange(4)).copy()


def _make_a2d_patch(color: int) -> np.ndarray:
    h1, w1 = _rand_hw()
    h2, w2 = _rand_hw()
    shift = random.randint(2, 3)
    gap = shift - 1
    rects = [
        {"r": shift, "c": 0, "h": h1, "w": w1, "tail_dir": (-1, 0), "tail_len": shift, "anchor": random.randrange(w1)},
        {"r": h1 + gap, "c": w1 + gap, "h": h2, "w": w2},
    ]
    return _rotate_patch(_build_patch(color, rects))


def _make_d2a_patch(color: int) -> np.ndarray:
    h1, w1 = _rand_hw()
    h2, w2 = _rand_hw()
    shift = random.randint(2, 3)
    gap = shift - 1
    rects = [
        {"r": 0, "c": 0, "h": h1, "w": w1, "tail_dir": (1, 0), "tail_len": shift, "anchor": random.randrange(w1)},
        {"r": h1 + gap, "c": w1 + gap, "h": h2, "w": w2},
    ]
    return _rotate_patch(_build_patch(color, rects))


def _make_diag_static_patch(color: int) -> np.ndarray:
    h1, w1 = _rand_hw()
    h2, w2 = _rand_hw()
    gap = random.randint(1, 3)
    rects = [
        {"r": 0, "c": 0, "h": h1, "w": w1},
        {"r": h1 + gap, "c": w1 + gap, "h": h2, "w": w2},
    ]
    return _rotate_patch(_build_patch(color, rects))


def _make_align_static_patch(color: int) -> np.ndarray:
    h1, w1 = _rand_hw()
    h2, w2 = _rand_hw()
    gap = random.randint(1, 3)
    rects = [
        {"r": 0, "c": 0, "h": h1, "w": w1},
        {"r": h1 - 1, "c": w1 + gap, "h": h2, "w": w2},
    ]
    return _rotate_patch(_build_patch(color, rects))


def _make_diag_both_tail_patch(color: int) -> np.ndarray:
    h1, w1 = _rand_hw()
    h2, w2 = _rand_hw()
    shift = random.randint(1, 3)
    gap = random.randint(1, 3)
    rects = [
        {"r": shift, "c": 0, "h": h1, "w": w1, "tail_dir": (-1, 0), "tail_len": shift, "anchor": random.randrange(w1)},
        {
            "r": shift + h1 + gap,
            "c": w1 + gap,
            "h": h2,
            "w": w2,
            "tail_dir": (-1, 0),
            "tail_len": shift,
            "anchor": random.randrange(w2),
        },
    ]
    return _rotate_patch(_build_patch(color, rects))


def _compose_grid(patches: list[np.ndarray]) -> np.ndarray | None:
    random.shuffle(patches)
    if len(patches) == 3:
        row_sizes = random.choice(([2, 1], [1, 2]))
    else:
        row_sizes = [2, 2]

    rows: list[list[np.ndarray]] = []
    idx = 0
    for size in row_sizes:
        rows.append(patches[idx : idx + size])
        idx += size

    for _ in range(40):
        h_gap = random.randint(2, 4)
        v_gap = random.randint(2, 4)
        margins = [random.randint(0, 2) for _ in range(4)]
        top, right, bottom, left = margins

        row_heights = [max(p.shape[0] for p in row) for row in rows]
        row_widths = [sum(p.shape[1] for p in row) + h_gap * (len(row) - 1) for row in rows]
        height = top + bottom + sum(row_heights) + v_gap * (len(rows) - 1)
        width = left + right + max(row_widths)
        if not (14 <= height <= 30 and 14 <= width <= 30):
            continue

        grid = np.zeros((height, width), dtype=int)
        r = top
        for row, row_h, row_w in zip(rows, row_heights, row_widths):
            c = random.randint(left, width - right - row_w)
            for patch in row:
                offset = row_h - patch.shape[0]
                rr = r + random.choice([0, offset // 2, offset])
                mask = patch != 0
                grid[rr : rr + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]
                c += patch.shape[1] + h_gap
            r += row_h + v_gap
        return grid
    return None


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        _out, objects, pair_infos = _solve(grid)
    except AssertionError:
        return False

    if len(objects) not in (6, 8):
        return False

    if not any(info["input_relation"] in {"horizontal", "vertical"} and info["final_relation"] == "diagonal" for info in pair_infos):
        return False
    if not any(info["input_relation"] == "diagonal" and info["final_relation"] in {"horizontal", "vertical"} for info in pair_infos):
        return False
    if not any(info["tail_count"] == 0 and info["final_relation"] == "diagonal" for info in pair_infos):
        return False
    if not any(info["tail_count"] > 0 and info["final_relation"] in {"horizontal", "vertical"} for info in pair_infos):
        return False
    if any(info["input_relation"] == "overlap" or info["final_relation"] == "overlap" for info in pair_infos):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        color_count = random.choice([3, 4])
        colors = random.sample(list(range(1, 10)), color_count)
        patches = [
            _make_a2d_patch(colors[0]),
            _make_d2a_patch(colors[1]),
            _make_diag_static_patch(colors[2]),
        ]
        if color_count == 4:
            extra_builder = random.choice([_make_align_static_patch, _make_diag_both_tail_patch])
            patches.append(extra_builder(colors[3]))

        grid = _compose_grid(patches)
        if grid is not None and _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
