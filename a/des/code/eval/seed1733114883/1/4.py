from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            arr = np.asarray(coords, dtype=int)
            comps.append({"coords": arr, "bbox": _bbox(arr)})
    return comps


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    color, count = counts.most_common(1)[0]
    assert sum(1 for v in counts.values() if int(v) == int(count)) == 1
    return int(color)


def _cell_signature(patch: np.ndarray) -> tuple:
    h, w = patch.shape
    top = tuple((int(c), int(patch[0, c])) for c in range(w) if int(patch[0, c]) != BLACK)
    bottom = tuple((int(c), int(patch[h - 1, c])) for c in range(w) if int(patch[h - 1, c]) != BLACK)
    left = tuple((int(r), int(patch[r, 0])) for r in range(h) if int(patch[r, 0]) != BLACK)
    right = tuple((int(r), int(patch[r, w - 1])) for r in range(h) if int(patch[r, w - 1]) != BLACK)
    return (int(h), int(w), top, bottom, left, right)


def _fragment_body_and_signature(patch: np.ndarray) -> tuple[int, tuple]:
    body = _unique_mode(patch)
    h, w = patch.shape
    top = tuple((int(c), int(patch[0, c])) for c in range(w) if int(patch[0, c]) != int(body))
    bottom = tuple((int(c), int(patch[h - 1, c])) for c in range(w) if int(patch[h - 1, c]) != int(body))
    left = tuple((int(r), int(patch[r, 0])) for r in range(h) if int(patch[r, 0]) != int(body))
    right = tuple((int(r), int(patch[r, w - 1])) for r in range(h) if int(patch[r, w - 1]) != int(body))
    return int(body), (int(h), int(w), top, bottom, left, right)


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0
    return _bbox(coords)


def _extract_cells(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    fr0, fc0, fr1, fc1 = _frame_bbox(grid)
    sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    comps = _components(sub != GREY)
    cells: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = sub[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch != GREY))
        cells.append(
            {
                "bbox": (int(fr0 + r0), int(fc0 + c0), int(fr0 + r1), int(fc0 + c1)),
                "signature": _cell_signature(patch),
            }
        )
    cells.sort(key=lambda item: item["bbox"])
    return (int(fr0), int(fc0), int(fr1), int(fc1)), cells


def _extract_fragments(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_box
    mask = grid != BLACK
    mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    fragments: list[dict] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch != BLACK))
        body, signature = _fragment_body_and_signature(patch)
        fragments.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "body": int(body), "signature": signature})
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered rectangular fragments into the grey frame cells by exact edge-fingerprint matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_box, cells = _extract_cells(g)
    fragments = _extract_fragments(g, frame_box)
    assert len(cells) == len(fragments)

    by_signature: dict[tuple, dict] = {}
    for cell in cells:
        signature = cell["signature"]
        assert signature not in by_signature
        by_signature[signature] = cell

    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY

    for fragment in fragments:
        assert fragment["signature"] in by_signature
        cell = by_signature[fragment["signature"]]
        r0, c0, r1, c1 = cell["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(fragment["body"])

    return out


def _sample_sizes(count: int) -> list[int]:
    return [random.randint(3, 6) for _ in range(count)]


def _partition_sizes(n_rows: int, n_cols: int) -> tuple[list[int], list[int]]:
    rows = _sample_sizes(n_rows)
    cols = _sample_sizes(n_cols)
    if n_rows >= 2 and n_cols >= 1 and random.random() < 0.5:
        i, j = random.sample(range(n_rows), 2)
        dup = random.randint(3, 6)
        rows[i] = dup
        rows[j] = dup
    elif n_cols >= 2:
        i, j = random.sample(range(n_cols), 2)
        dup = random.randint(3, 6)
        cols[i] = dup
        cols[j] = dup
    return rows, cols


def _cell_boxes(frame_r: int, frame_c: int, row_heights: list[int], col_widths: list[int]) -> list[dict]:
    boxes: list[dict] = []
    r = int(frame_r + 1)
    for i, h in enumerate(row_heights):
        c = int(frame_c + 1)
        for j, w in enumerate(col_widths):
            boxes.append(
                {
                    "row": int(i),
                    "col": int(j),
                    "bbox": (int(r), int(c), int(r + h - 1), int(c + w - 1)),
                    "shape": (int(h), int(w)),
                }
            )
            c += int(w + 1)
        r += int(h + 1)
    return boxes


def _draw_frame(canvas: np.ndarray, frame_r: int, frame_c: int, row_heights: list[int], col_widths: list[int]) -> None:
    frame_h = int(sum(row_heights) + len(row_heights) + 1)
    frame_w = int(sum(col_widths) + len(col_widths) + 1)
    canvas[frame_r, frame_c : frame_c + frame_w] = GREY
    canvas[frame_r + frame_h - 1, frame_c : frame_c + frame_w] = GREY
    canvas[frame_r : frame_r + frame_h, frame_c] = GREY
    canvas[frame_r : frame_r + frame_h, frame_c + frame_w - 1] = GREY

    cursor = int(frame_r)
    for h in row_heights[:-1]:
        cursor += int(h + 1)
        canvas[cursor, frame_c : frame_c + frame_w] = GREY

    cursor = int(frame_c)
    for w in col_widths[:-1]:
        cursor += int(w + 1)
        canvas[frame_r : frame_r + frame_h, cursor] = GREY


def _random_signature(h: int, w: int) -> tuple:
    slots: list[tuple[str, int]] = []
    slots.extend(("top", c) for c in range(1, w - 1))
    slots.extend(("bottom", c) for c in range(1, w - 1))
    slots.extend(("left", r) for r in range(1, h - 1))
    slots.extend(("right", r) for r in range(1, h - 1))
    max_markers = min(4, len(slots))
    target = random.randint(1, max_markers)
    for _ in range(200):
        chosen = random.sample(slots, target)
        side_counts = Counter(side for side, _ in chosen)
        if max(side_counts.values()) > 2:
            continue
        parts = {"top": [], "bottom": [], "left": [], "right": []}
        for side, offset in chosen:
            parts[side].append((int(offset), int(random.choice(PALETTE))))
        for side in parts:
            parts[side].sort()
        return (int(h), int(w), tuple(parts["top"]), tuple(parts["bottom"]), tuple(parts["left"]), tuple(parts["right"]))
    raise RuntimeError("failed to sample a valid fingerprint")


def _signature_colors(signature: tuple) -> set[int]:
    used: set[int] = set()
    for edge in signature[2:]:
        for _offset, color in edge:
            used.add(int(color))
    return used


def _apply_cell_markers(canvas: np.ndarray, box: tuple[int, int, int, int], signature: tuple) -> None:
    r0, c0, r1, c1 = box
    _h, _w, top, bottom, left, right = signature
    for offset, color in top:
        canvas[r0, c0 + int(offset)] = int(color)
    for offset, color in bottom:
        canvas[r1, c0 + int(offset)] = int(color)
    for offset, color in left:
        canvas[r0 + int(offset), c0] = int(color)
    for offset, color in right:
        canvas[r0 + int(offset), c1] = int(color)


def _fragment_patch(h: int, w: int, body: int, signature: tuple) -> np.ndarray:
    patch = np.full((h, w), int(body), dtype=int)
    _h, _w, top, bottom, left, right = signature
    for offset, color in top:
        patch[0, int(offset)] = int(color)
    for offset, color in bottom:
        patch[h - 1, int(offset)] = int(color)
    for offset, color in left:
        patch[int(offset), 0] = int(color)
    for offset, color in right:
        patch[int(offset), w - 1] = int(color)
    return patch


def _candidate_positions(occupied: np.ndarray, h: int, w: int, pad: int) -> list[tuple[int, int]]:
    height, width = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(height - h + 1):
        r0 = max(0, int(r - pad))
        r1 = min(height, int(r + h + pad))
        for c in range(width - w + 1):
            c0 = max(0, int(c - pad))
            c1 = min(width, int(c + w + pad))
            if bool(occupied[r0:r1, c0:c1].any()):
                continue
            positions.append((int(r), int(c)))
    return positions


def _body_color(signature: tuple, used_neighbors: set[int]) -> int:
    banned = _signature_colors(signature)
    candidates = [color for color in PALETTE if color not in banned]
    preferred = [color for color in candidates if color not in used_neighbors]
    pool = preferred if preferred else candidates
    assert pool
    return int(random.choice(pool))


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 3)
    n_cols = random.randint(2, 3)
    row_heights, col_widths = _partition_sizes(n_rows, n_cols)

    frame_h = int(sum(row_heights) + n_rows + 1)
    frame_w = int(sum(col_widths) + n_cols + 1)
    grid_h = int(frame_h + random.randint(10, 16))
    grid_w = int(frame_w + random.randint(10, 16))
    if grid_h > 36 or grid_w > 36:
        return None

    frame_r = max(2, (grid_h - frame_h) // 2 + random.randint(-1, 1))
    frame_c = max(2, (grid_w - frame_w) // 2 + random.randint(-1, 1))
    frame_r = min(frame_r, grid_h - frame_h - 2)
    frame_c = min(frame_c, grid_w - frame_w - 2)
    if frame_r < 1 or frame_c < 1:
        return None

    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    out = np.full((grid_h, grid_w), BLACK, dtype=int)
    _draw_frame(inp, frame_r, frame_c, row_heights, col_widths)
    _draw_frame(out, frame_r, frame_c, row_heights, col_widths)

    cells = _cell_boxes(frame_r, frame_c, row_heights, col_widths)

    used_signatures: set[tuple] = set()
    shape_counter = Counter(cell["shape"] for cell in cells)
    saw_duplicate_shape = any(int(v) >= 2 for v in shape_counter.values())
    if not saw_duplicate_shape:
        return None

    for cell in cells:
        h, w = cell["shape"]
        signature = None
        for _ in range(200):
            candidate = _random_signature(int(h), int(w))
            if candidate in used_signatures:
                continue
            signature = candidate
            break
        if signature is None:
            return None
        cell["signature"] = signature
        used_signatures.add(signature)

    grid_cells = {(cell["row"], cell["col"]): cell for cell in cells}
    for i in range(n_rows):
        for j in range(n_cols):
            cell = grid_cells[(i, j)]
            neighbors = set()
            if i > 0:
                neighbors.add(int(grid_cells[(i - 1, j)]["body"]))
            if j > 0:
                neighbors.add(int(grid_cells[(i, j - 1)]["body"]))
            cell["body"] = _body_color(cell["signature"], neighbors)

    for cell in cells:
        _apply_cell_markers(inp, cell["bbox"], cell["signature"])
        r0, c0, r1, c1 = cell["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(cell["body"])

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    fr0 = int(frame_r)
    fc0 = int(frame_c)
    fr1 = int(frame_r + frame_h - 1)
    fc1 = int(frame_c + frame_w - 1)
    occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    fragments = []
    for cell in cells:
        h, w = cell["shape"]
        patch = _fragment_patch(int(h), int(w), int(cell["body"]), cell["signature"])
        fragments.append({"patch": patch, "shape": (int(h), int(w))})
    random.shuffle(fragments)
    fragments.sort(key=lambda item: int(item["shape"][0] * item["shape"][1]), reverse=True)

    for item in fragments:
        h, w = item["shape"]
        pad = 1
        positions = _candidate_positions(occupied, int(h), int(w), pad)
        if not positions:
            return None
        r, c = random.choice(positions)
        inp[r : r + h, c : c + w] = item["patch"]
        occupied[r : r + h, c : c + w] = True

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
