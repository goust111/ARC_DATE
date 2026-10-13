from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SEP = 4
ARROW = 5
PALETTE = (1, 2, 3, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_ROT = {(-1, 0): 0, (0, 1): 3, (1, 0): 2, (0, -1): 1}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _separator_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    if not cols:
        return []
    runs: list[tuple[int, int]] = []
    start = end = cols[0]
    for c in cols[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = end = c
    runs.append((start, end))
    return runs


def _extract_stamps(panel: np.ndarray) -> list[dict]:
    stamps = []
    for coords in _components(panel != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = [int(v) for v in np.unique(patch) if int(v) != BG]
        assert len(colors) == 1, "stamp must use exactly one visible color"
        stamps.append(
            {
                "color": colors[0],
                "patch": patch,
                "row": r0,
                "col": c0,
            }
        )
    stamps.sort(key=lambda item: (int(item["row"]), int(item["col"])))
    return stamps


def _infer_arrow_direction(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    row_counts = Counter(r for r, _c in pts)
    col_counts = Counter(c for _r, c in pts)

    line_col = next((c for c, n in col_counts.items() if n == 3), None)
    if line_col is not None:
        line_rows = sorted(r for r, c in pts if c == line_col)
        extra = [(r, c) for r, c in pts if c != line_col]
        assert len(line_rows) == 3 and len(extra) == 1
        hr, hc = extra[0]
        if hr == line_rows[0] and abs(hc - line_col) == 1:
            return (-1, 0), (line_rows[-1], line_col)
        if hr == line_rows[-1] and abs(hc - line_col) == 1:
            return (1, 0), (line_rows[0], line_col)

    line_row = next((r for r, n in row_counts.items() if n == 3), None)
    assert line_row is not None, "grey component must contain a 3-pixel shaft"
    line_cols = sorted(c for r, c in pts if r == line_row)
    extra = [(r, c) for r, c in pts if r != line_row]
    assert len(line_cols) == 3 and len(extra) == 1
    hr, hc = extra[0]
    if hc == line_cols[0] and abs(hr - line_row) == 1:
        return (0, -1), (line_row, line_cols[-1])
    if hc == line_cols[-1] and abs(hr - line_row) == 1:
        return (0, 1), (line_row, line_cols[0])
    raise AssertionError("failed to infer arrow direction")


def _extract_instructions(panel: np.ndarray) -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    for coords in _components(panel == ARROW):
        direction, tail = _infer_arrow_direction(coords)
        shaft = {(int(r), int(c)) for r, c in coords}
        candidates = []
        for nr, nc in _neighbors4(*tail):
            if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1] and (nr, nc) not in shaft:
                value = int(panel[nr, nc])
                if value not in (BG, ARROW):
                    candidates.append(value)
        if not candidates:
            for r, c in shaft:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1] and (nr, nc) not in shaft:
                        value = int(panel[nr, nc])
                        if value not in (BG, ARROW):
                            candidates.append(value)
        assert candidates, "each arrow must have a colored label pixel"
        color = candidates[0]
        mapping[color] = direction
    return mapping


def _extract_anchors(canvas: np.ndarray) -> dict[int, tuple[int, int]]:
    anchors: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(canvas != BG):
        anchors[int(canvas[r, c])] = (int(r), int(c))
    return anchors


def _stamp_onto(canvas: np.ndarray, patch: np.ndarray, anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    top = ar - patch.shape[0] // 2
    left = ac - patch.shape[1] // 2
    for r in range(patch.shape[0]):
        rr = top + r
        if not (0 <= rr < canvas.shape[0]):
            continue
        for c in range(patch.shape[1]):
            cc = left + c
            if 0 <= cc < canvas.shape[1] and int(patch[r, c]) != BG:
                canvas[rr, cc] = int(patch[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored stamp according to its matched grey arrow, then stamp it onto the right canvas at the matching colored anchor.
    The output is the canvas region alone.
    """
    g = np.asarray(grid, dtype=int)
    runs = _separator_runs(g)
    assert len(runs) >= 2, "expected two yellow separator runs"
    left_sep, right_sep = runs[0], runs[1]

    stamp_panel = g[:, : left_sep[0]]
    instruction_panel = g[:, left_sep[1] + 1 : right_sep[0]]
    canvas_panel = g[:, right_sep[1] + 1 :]

    stamps = _extract_stamps(stamp_panel)
    directions = _extract_instructions(instruction_panel)
    anchors = _extract_anchors(canvas_panel)

    out = np.full(canvas_panel.shape, BG, dtype=int)
    for item in stamps:
        color = int(item["color"])
        assert color in directions and color in anchors, "every stamp color must have one arrow and one anchor"
        rot = DIR_TO_ROT[directions[color]]
        patch = np.rot90(item["patch"], rot)
        _stamp_onto(out, patch, anchors[color])
    return out


def _connected_cells(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _build_shape_pool() -> list[np.ndarray]:
    shapes = []
    seen = set()
    for mask in range(1 << 9):
        if not (mask & (1 << 4)):
            continue
        cells = {(idx // 3, idx % 3) for idx in range(9) if (mask >> idx) & 1}
        if not 4 <= len(cells) <= 6:
            continue
        if not _connected_cells(cells):
            continue
        if not any(r == 0 for r, _c in cells):
            continue
        if not any(r == 2 for r, _c in cells):
            continue
        if not any(c == 0 for _r, c in cells):
            continue
        if not any(c == 2 for _r, c in cells):
            continue
        arr = np.zeros((3, 3), dtype=int)
        for r, c in cells:
            arr[r, c] = 1
        if any(np.array_equal(arr, np.rot90(arr, k)) for k in (1, 2, 3)):
            continue
        key = tuple(int(x) for x in arr.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        shapes.append(arr)
    assert shapes, "shape pool must not be empty"
    return shapes


SHAPE_POOL = _build_shape_pool()


def _make_arrow_patch(direction: tuple[int, int], color: int) -> np.ndarray:
    dr, dc = direction
    pr, pc = (dc, -dr) if random.random() < 0.5 else (-dc, dr)
    grey = [(0, 0), (dr, dc), (2 * dr, 2 * dc), (2 * dr + pr, 2 * dc + pc)]
    tag = (-dr, -dc)
    all_pts = grey + [tag]
    min_r = min(r for r, _c in all_pts)
    min_c = min(c for _r, c in all_pts)
    max_r = max(r for r, _c in all_pts)
    max_c = max(c for _r, c in all_pts)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for r, c in grey:
        patch[r - min_r, c - min_c] = ARROW
    patch[tag[0] - min_r, tag[1] - min_c] = int(color)
    return patch


def _overlay(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    assert region.shape == patch.shape
    assert not np.any((region != BG) & (patch != BG))
    region[patch != BG] = patch[patch != BG]


def _placed_pixels(patch: np.ndarray, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    top = anchor[0] - patch.shape[0] // 2
    left = anchor[1] - patch.shape[1] // 2
    pts = set()
    for r, c in np.argwhere(patch != BG):
        pts.add((int(top + r), int(left + c)))
    return pts


def _choose_overlap_pair(
    patches: list[np.ndarray], anchors_h: int, anchors_w: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
    for _ in range(80):
        delta = random.choice(deltas)
        ar = random.randint(1, anchors_h - 2)
        ac = random.randint(1, anchors_w - 2)
        br = ar + delta[0]
        bc = ac + delta[1]
        if not (1 <= br < anchors_h - 1 and 1 <= bc < anchors_w - 1):
            continue
        pts0 = _placed_pixels(patches[0], (ar, ac))
        pts1 = _placed_pixels(patches[1], (br, bc))
        if pts0 & pts1:
            return (ar, ac), (br, bc)
    return None


def _pick_far_anchor(
    used: list[tuple[int, int]], anchors_h: int, anchors_w: int
) -> tuple[int, int] | None:
    choices = [(r, c) for r in range(1, anchors_h - 1) for c in range(1, anchors_w - 1)]
    random.shuffle(choices)
    for r, c in choices:
        if (r, c) in used:
            continue
        if all(max(abs(r - rr), abs(c - cc)) >= 3 for rr, cc in used):
            return (r, c)
    for r, c in choices:
        if (r, c) not in used:
            return (r, c)
    return None


def _attempt_generate() -> np.ndarray | None:
    count = random.randint(2, 4)
    height = random.randint(12, 18)
    stamp_cols = 2 if count >= 3 or random.random() < 0.55 else 1
    instr_cols = 2 if count >= 3 or random.random() < 0.55 else 1

    stamp_slots = [(r, c) for r in range(1, height - 2, 4) for c in ([1] if stamp_cols == 1 else [1, 5])]
    instr_slots = [(r, c) for r in range(1, height - 3, 5) for c in ([1] if instr_cols == 1 else [1, 6])]
    if len(stamp_slots) < count or len(instr_slots) < count:
        return None

    stamp_w = 5 if stamp_cols == 1 else 9
    instr_w = 6 if instr_cols == 1 else 11
    canvas_w = random.randint(9, 15)
    total_w = stamp_w + 1 + instr_w + 1 + canvas_w
    grid = np.full((height, total_w), BG, dtype=int)
    grid[:, stamp_w] = SEP
    grid[:, stamp_w + 1 + instr_w] = SEP

    colors = random.sample(PALETTE, count)
    directions = random.sample(DIRS, count)
    shapes = random.sample(SHAPE_POOL, count)

    stamp_positions = random.sample(stamp_slots, count)
    random.shuffle(colors)
    random.shuffle(directions)
    random.shuffle(shapes)
    stamp_items = []
    for color, direction, shape, (r0, c0) in zip(colors, directions, shapes, stamp_positions):
        patch = np.where(shape, int(color), BG)
        _overlay(grid[:, :stamp_w], patch, r0, c0)
        stamp_items.append(
            {
                "color": int(color),
                "direction": direction,
                "patch": patch,
                "row": r0,
                "col": c0,
            }
        )
    stamp_items.sort(key=lambda item: (int(item["row"]), int(item["col"])))

    direction_by_color = {int(item["color"]): item["direction"] for item in stamp_items}
    rotated_by_color = {
        int(item["color"]): np.rot90(item["patch"], DIR_TO_ROT[item["direction"]])
        for item in stamp_items
    }

    instr_positions = random.sample(instr_slots, count)
    random.shuffle(instr_positions)
    instruction_panel = grid[:, stamp_w + 1 : stamp_w + 1 + instr_w]
    for color, (r0, c0) in zip([int(item["color"]) for item in stamp_items], instr_positions):
        patch = _make_arrow_patch(direction_by_color[color], color)
        max_r_off = 4 - patch.shape[0]
        max_c_off = 4 - patch.shape[1]
        rr = r0 + random.randint(0, max_r_off)
        cc = c0 + random.randint(0, max_c_off)
        _overlay(instruction_panel, patch, rr, cc)

    canvas_panel = grid[:, stamp_w + 2 + instr_w :]
    overlap = count >= 2 and random.random() < 0.5
    anchors: dict[int, tuple[int, int]] = {}
    used: list[tuple[int, int]] = []

    if overlap:
        first_two = [stamp_items[0]["color"], stamp_items[1]["color"]]
        pair = _choose_overlap_pair(
            [rotated_by_color[int(first_two[0])], rotated_by_color[int(first_two[1])]],
            height,
            canvas_w,
        )
        if pair is None:
            return None
        anchors[int(first_two[0])] = pair[0]
        anchors[int(first_two[1])] = pair[1]
        used.extend(pair)

    for item in stamp_items:
        color = int(item["color"])
        if color in anchors:
            continue
        anchor = _pick_far_anchor(used, height, canvas_w)
        if anchor is None:
            return None
        anchors[color] = anchor
        used.append(anchor)

    for color, (r, c) in anchors.items():
        canvas_panel[r, c] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != canvas_panel.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
