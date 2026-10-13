import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """ARC-style background inference: the most common color in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Minimal 4-connected component helper in the same (r, c) convention as arc_utils."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_from_component(grid: np.ndarray, frame_color: int, coords: np.ndarray):
    r0, c0, r1, c1 = bbox_coords(coords)
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    if side_h != side_w or side_h < 5:
        return None
    if int(coords.shape[0]) != 4 * side_h - 4:
        return None

    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(box[0, :] == frame_color):
        return None
    if not np.all(box[-1, :] == frame_color):
        return None
    if not np.all(box[:, 0] == frame_color):
        return None
    if not np.all(box[:, -1] == frame_color):
        return None
    if np.any(box[1:-1, 1:-1] == frame_color):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "side": side_h,
    }


def _detect_squares(grid: np.ndarray) -> tuple[int, list[dict]]:
    colors = sorted(int(v) for v in np.unique(grid))
    best_frame = None
    best_squares = None
    best_score = None

    for color in colors:
        raw = []
        for coords in components_mask(grid == color):
            square = _frame_from_component(grid, color, coords)
            if square is not None:
                raw.append(square)
        if len(raw) < 2:
            continue
        side = Counter(square["side"] for square in raw).most_common(1)[0][0]
        squares = [square for square in raw if int(square["side"]) == int(side)]
        score = (len(squares), int(side))
        if best_score is None or score > best_score:
            best_score = score
            best_frame = color
            best_squares = squares

    assert best_frame is not None and best_squares is not None and len(best_squares) >= 3
    return int(best_frame), best_squares


def _annotate_squares(grid: np.ndarray, squares: list[dict]) -> list[dict]:
    color_presence = Counter()
    palettes: list[set[int]] = []
    for square in squares:
        r0, c0, r1, c1 = square["inner_bbox"]
        inner = grid[r0 : r1 + 1, c0 : c1 + 1]
        palette = set(map(int, np.unique(inner)))
        assert len(palette) >= 2
        palettes.append(palette)
        for color in palette:
            color_presence[int(color)] += 1

    marker_candidates = [color for color, freq in color_presence.items() if int(freq) == len(squares)]
    if len(marker_candidates) != 1:
        marker_candidates = [color for color, freq in color_presence.items() if int(freq) > 1]
    assert len(marker_candidates) == 1
    marker = int(marker_candidates[0])

    annotated = []
    for square, palette in zip(squares, palettes):
        fill_candidates = sorted(color for color in palette if int(color) != marker)
        assert len(fill_candidates) == 1
        fill = int(fill_candidates[0])
        r0, c0, r1, c1 = square["inner_bbox"]
        inner = grid[r0 : r1 + 1, c0 : c1 + 1]
        count = int(np.count_nonzero(inner == marker))
        assert count > 0
        enriched = dict(square)
        enriched["fill"] = fill
        enriched["count"] = count
        annotated.append(enriched)

    counts = [int(square["count"]) for square in annotated]
    assert len(set(counts)) == len(counts)
    return sorted(annotated, key=lambda square: int(square["count"]))


def _draw_bridge(out: np.ndarray, first: dict, second: dict, color: int) -> None:
    ar0, ac0, ar1, ac1 = first["bbox"]
    br0, bc0, br1, bc1 = second["bbox"]
    ai0, aj0, ai1, aj1 = first["inner_bbox"]
    bi0, bj0, bi1, bj1 = second["inner_bbox"]

    if ar0 == br0 and ar1 == br1:
        left, right = (first, second) if ac0 < bc0 else (second, first)
        li0, _, li1, _ = left["inner_bbox"]
        gap_start = int(left["bbox"][3] + 1)
        gap_stop = int(right["bbox"][1])
        assert gap_start < gap_stop
        out[li0 : li1 + 1, gap_start:gap_stop] = int(color)
        return

    if ac0 == bc0 and ac1 == bc1:
        top, bottom = (first, second) if ar0 < br0 else (second, first)
        _, tj0, _, tj1 = top["inner_bbox"]
        gap_start = int(top["bbox"][2] + 1)
        gap_stop = int(bottom["bbox"][0])
        assert gap_start < gap_stop
        out[gap_start:gap_stop, tj0 : tj1 + 1] = int(color)
        return

    raise AssertionError("consecutive squares must share a row band or column band")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares by the hidden order encoded in their internal marker counts.

    Input structure:
    - A background contains several equal-sized framed squares with a shared frame color.
    - Each square has its own fill color, plus some grey marker pixels scattered inside.
    - Marker counts are all distinct, and count-sorted neighboring squares are aligned in one row band or one column band.

    Rule:
    - Detect the framed squares and count the marker pixels inside each one.
    - Sort the squares from the fewest markers to the most markers.
    - For every consecutive pair in that order, draw a bridge through the gap between their frames, using the inner-square thickness and the lower-count square's fill color.

    Key insight:
    - The visitation order is not drawn explicitly anywhere; it is hidden in the per-square marker counts, and the bridge color always flows downstream from the lower-count square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _frame_color, squares = _detect_squares(g)
    ordered = _annotate_squares(g, squares)

    out = g.copy()
    for idx in range(len(ordered) - 1):
        first = ordered[idx]
        second = ordered[idx + 1]
        _draw_bridge(out, first, second, int(first["fill"]))
    return out


GEN_BG = 0
GEN_MARKER = 5
GEN_MAX_ATTEMPTS = 400
GEN_FAMILIES = ("mixed", "horizontal", "vertical")
_GEN_CALL_COUNT = 0


def _path_has_both_axes(path: list[tuple[int, int]]) -> bool:
    has_horizontal = any(path[i][0] == path[i + 1][0] for i in range(len(path) - 1))
    has_vertical = any(path[i][1] == path[i + 1][1] for i in range(len(path) - 1))
    return bool(has_horizontal and has_vertical)


def _dfs_path(path: list[tuple[int, int]], used: set[tuple[int, int]], target_len: int) -> bool:
    if len(path) == target_len:
        return True
    r, c = path[-1]
    nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    random.shuffle(nbrs)
    for nr, nc in nbrs:
        if not (0 <= nr < 3 and 0 <= nc < 3):
            continue
        if (nr, nc) in used:
            continue
        used.add((nr, nc))
        path.append((nr, nc))
        if _dfs_path(path, used, target_len):
            return True
        path.pop()
        used.remove((nr, nc))
    return False


def _generate_path(family: str) -> list[tuple[int, int]]:
    if family == "horizontal":
        length = random.randint(3, 4)
        path = [(0, c) for c in range(length)]
    elif family == "vertical":
        length = random.randint(3, 4)
        path = [(r, 0) for r in range(length)]
    else:
        length = random.randint(4, 6)
        for _ in range(200):
            start = (random.randrange(3), random.randrange(3))
            path = [start]
            used = {start}
            if _dfs_path(path, used, length) and _path_has_both_axes(path):
                break
        else:
            raise RuntimeError("failed to build a mixed path")

    if random.random() < 0.5:
        path = list(reversed(path))
    return path


def _choose_geometry(path: list[tuple[int, int]]):
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    row_span = max(rows) - min(rows) + 1
    col_span = max(cols) - min(cols) + 1

    candidates = []
    for inner in (3, 4, 5):
        square_side = inner + 2
        for gap in (1, 2, 3):
            base_h = row_span * square_side + (row_span - 1) * gap
            base_w = col_span * square_side + (col_span - 1) * gap
            if base_h <= 30 and base_w <= 30:
                candidates.append((inner, gap, base_h, base_w))
    if not candidates:
        return None

    inner, gap, base_h, base_w = random.choice(candidates)
    extra_h = random.randint(0, min(6, 30 - base_h))
    extra_w = random.randint(0, min(6, 30 - base_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    return inner, gap, top, left, base_h + extra_h, base_w + extra_w


def _draw_square(grid: np.ndarray, r0: int, c0: int, side: int, frame: int, fill: int, marker: int, count: int) -> None:
    r1 = int(r0 + side - 1)
    c1 = int(c0 + side - 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(fill)
    grid[r0, c0 : c1 + 1] = int(frame)
    grid[r1, c0 : c1 + 1] = int(frame)
    grid[r0 : r1 + 1, c0] = int(frame)
    grid[r0 : r1 + 1, c1] = int(frame)

    inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    assert 0 < count < len(inner_cells)
    for r, c in random.sample(inner_cells, count):
        grid[r, c] = int(marker)


def _planned_output(grid: np.ndarray, squares: list[dict]) -> np.ndarray:
    out = grid.copy()
    for idx in range(len(squares) - 1):
        _draw_bridge(out, squares[idx], squares[idx + 1], int(squares[idx]["fill"]))
    return out


def _gen_attempt(family: str) -> np.ndarray | None:
    path = _generate_path(family)
    geometry = _choose_geometry(path)
    if geometry is None:
        return None
    inner, gap, top, left, height, width = geometry

    square_side = inner + 2
    max_count = inner * inner - 1
    if len(path) > max_count:
        return None

    path_rows = [r for r, _ in path]
    path_cols = [c for _, c in path]
    row0 = min(path_rows)
    col0 = min(path_cols)
    norm_path = [(r - row0, c - col0) for r, c in path]

    available = [1, 2, 3, 4, 6, 7, 8, 9]
    frame = random.choice(available)
    fills = random.sample([color for color in available if int(color) != frame], len(path))
    counts = sorted(random.sample(list(range(1, inner * inner)), len(path)))

    grid = np.full((height, width), GEN_BG, dtype=int)
    placed: list[dict] = []
    for idx, (gr, gc) in enumerate(norm_path):
        r0 = int(top + gr * (square_side + gap))
        c0 = int(left + gc * (square_side + gap))
        _draw_square(grid, r0, c0, square_side, frame, fills[idx], GEN_MARKER, counts[idx])
        placed.append(
            {
                "bbox": (r0, c0, r0 + square_side - 1, c0 + square_side - 1),
                "inner_bbox": (r0 + 1, c0 + 1, r0 + square_side - 2, c0 + square_side - 2),
                "fill": int(fills[idx]),
                "count": int(counts[idx]),
            }
        )

    expected = _planned_output(grid, placed)
    actual = main(grid)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(actual, expected):
        return None

    for idx in range(len(placed) - 1):
        a = placed[idx]
        b = placed[idx + 1]
        same_row = int(a["bbox"][0]) == int(b["bbox"][0]) and int(a["bbox"][2]) == int(b["bbox"][2])
        same_col = int(a["bbox"][1]) == int(b["bbox"][1]) and int(a["bbox"][3]) == int(b["bbox"][3])
        if not (same_row or same_col):
            return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    family = GEN_FAMILIES[_GEN_CALL_COUNT % len(GEN_FAMILIES)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(family)
        if grid is not None:
            return grid

    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(random.choice(GEN_FAMILIES))
        if grid is not None:
            return grid

    raise RuntimeError("failed to generate a valid hidden-count bridge puzzle")
