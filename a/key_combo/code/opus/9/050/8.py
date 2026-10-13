from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
FRAME = 5
PAIR_COLORS = (2, 3, 7)
_GEN_COUNTER = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _gray_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != FRAME or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == FRAME:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _gray_components(grid):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        border = {
            (r0, c) for c in range(c0, c1 + 1)
        } | {
            (r1, c) for c in range(c0, c1 + 1)
        } | {
            (r, c0) for r in range(r0, r1 + 1)
        } | {
            (r, c1) for r in range(r0, r1 + 1)
        }
        comp_set = set(comp)
        if comp_set != border:
            continue
        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if not np.all(interior == BG):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
            }
        )
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _find_edge_pairs(grid: np.ndarray) -> list[dict]:
    H, W = grid.shape
    edge_positions: set[tuple[int, int]] = set()
    for c in range(W):
        edge_positions.add((0, c))
        edge_positions.add((H - 1, c))
    for r in range(H):
        edge_positions.add((r, 0))
        edge_positions.add((r, W - 1))

    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in sorted(edge_positions):
        color = int(grid[r, c])
        if color not in PAIR_COLORS:
            continue
        by_color[color].append((r, c))

    pairs: list[dict] = []
    for color, coords in sorted(by_color.items()):
        if len(coords) != 2:
            raise ValueError(f"color {color} does not form exactly one edge pair")
        (r0, c0), (r1, c1) = coords
        if r0 == r1 and {c0, c1} == {0, W - 1}:
            pairs.append({"color": color, "orientation": "h", "a": (r0, min(c0, c1)), "b": (r1, max(c0, c1))})
        elif c0 == c1 and {r0, r1} == {0, H - 1}:
            pairs.append({"color": color, "orientation": "v", "a": (min(r0, r1), c0), "b": (max(r0, r1), c1)})
        else:
            raise ValueError(f"color {color} is not placed on opposite edges")
    return pairs


def _meeting_point(pair: dict, shape: tuple[int, int]) -> tuple[int, int]:
    H, W = shape
    center = (H // 2, W // 2)
    (r0, c0), (r1, c1) = pair["a"], pair["b"]
    if pair["orientation"] == "h":
        gap = c1 - c0 - 1
        if gap % 2 == 1:
            return r0, (c0 + c1) // 2
        candidates = [(r0, c0 + gap // 2), (r0, c0 + gap // 2 + 1)]
    else:
        gap = r1 - r0 - 1
        if gap % 2 == 1:
            return (r0 + r1) // 2, c0
        candidates = [(r0 + gap // 2, c0), (r0 + gap // 2 + 1, c0)]
    return min(candidates, key=lambda rc: ((rc[0] - center[0]) ** 2 + (rc[1] - center[1]) ** 2, rc))


def _segment_cells(pair: dict) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = pair["a"], pair["b"]
    if pair["orientation"] == "h":
        return [(r0, c) for c in range(c0, c1 + 1)]
    return [(r, c0) for r in range(r0, r1 + 1)]


def _trail_crosses_frame(frame_bbox: tuple[int, int, int, int], pair: dict) -> bool:
    r0, c0, r1, c1 = frame_bbox
    for r, c in _segment_cells(pair):
        if r0 < r < r1 and c0 < c < c1:
            return True
    return False


def _choose_frame_color(frame: dict, meetings: list[dict]) -> int:
    cr, cc = frame["center"]
    winner = min(
        meetings,
        key=lambda m: (((m["pos"][0] - cr) ** 2 + (m["pos"][1] - cc) ** 2), m["pos"], m["color"]),
    )
    return int(winner["color"])


def _apply_transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pairs = _find_edge_pairs(g)
    meetings: list[dict] = []

    for pair in pairs:
        for r, c in _segment_cells(pair):
            if int(out[r, c]) != FRAME:
                out[r, c] = int(pair["color"])
        meetings.append({"pos": _meeting_point(pair, g.shape), "color": int(pair["color"])})

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        inside = [m for m in meetings if r0 < m["pos"][0] < r1 and c0 < m["pos"][1] < c1]
        if not inside:
            continue
        out[r0 + 1 : r1, c0 + 1 : c1] = _choose_frame_color(frame, inside)

    out[g == FRAME] = FRAME
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw inward trails from opposite edge pairs and fill any gray frame that contains a meeting point.

    Input structure:
    - The grid is mostly black, with 2-3 same-color pixel pairs placed on opposite edges in a shared row or column.
    - Several gray one-pixel-thick hollow rectangular frames sit strictly inside the grid and do not overlap.
    - Pair colors are distinct from one another, and frame interiors are black in the input.

    Rule:
    - Match each colored edge pixel to the same-color pixel on the opposite edge.
    - Extend both endpoints straight inward toward each other, coloring the whole route in that pair's color while keeping gray frame outlines unchanged.
    - Compute the pair's meeting point: use the single center cell when one exists; otherwise choose the central candidate closer to the grid center cell.
    - For every gray frame, if one or more meeting points lie strictly inside its interior, fill the entire interior with the color of the meeting point nearest the frame center.

    Key insight:
    - The visible trails are not the final decision signal; the decisive object is the derived meeting point, whose position relative to each frame determines whether and how that frame is filled.
    """
    return _apply_transform(grid)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _all_frame_bboxes(H: int, W: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for h in range(4, min(8, H - 2) + 1):
        for w in range(4, min(8, W - 2) + 1):
            for r0 in range(1, H - h):
                for c0 in range(1, W - w):
                    boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
    random.shuffle(boxes)
    return boxes


def _pick_rows(H: int, n_pairs: int) -> list[int] | None:
    candidates = list(range(2, H - 2))
    for _ in range(200):
        if n_pairs == 3:
            r1 = random.randint(2, H - 8)
            r2 = r1 + random.randint(2, 3)
            far = [r for r in candidates if min(abs(r - r1), abs(r - r2)) >= 4]
            if not far:
                continue
            r3 = random.choice(far)
            rows = sorted([r1, r2, r3])
        else:
            rows = sorted(random.sample(candidates, 2))
            if abs(rows[1] - rows[0]) < 3:
                continue
        return rows
    return None


def _meeting_column(W: int) -> int:
    gap = W - 2
    if gap % 2 == 1:
        return W // 2
    return W // 2


def _candidate_single_fill(
    boxes: list[tuple[int, int, int, int]], rows: list[int], target_row: int, mc: int
) -> list[tuple[int, int, int, int]]:
    other_rows = {r for r in rows if r != target_row}
    out = []
    for r0, c0, r1, c1 in boxes:
        inner_rows = range(r0 + 1, r1)
        inner_cols = range(c0 + 1, c1)
        if target_row not in inner_rows or mc not in inner_cols:
            continue
        if any(r in inner_rows for r in other_rows):
            continue
        out.append((r0, c0, r1, c1))
    return out


def _candidate_multi_fill(
    boxes: list[tuple[int, int, int, int]], rows: list[int], target_rows: tuple[int, int], mc: int
) -> list[tuple[int, int, int, int]]:
    target = set(target_rows)
    other_rows = {r for r in rows if r not in target}
    out = []
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        inner_rows = range(r0 + 1, r1)
        inner_cols = range(c0 + 1, c1)
        if mc not in inner_cols:
            continue
        if not all(r in inner_rows for r in target_rows):
            continue
        if any(r in inner_rows for r in other_rows):
            continue
        frame = {"bbox": bbox, "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0)}
        meetings = [{"pos": (r, mc), "color": 1} for r in target_rows]
        dists = sorted(((m["pos"][0] - frame["center"][0]) ** 2 + (m["pos"][1] - frame["center"][1]) ** 2) for m in meetings)
        if len(dists) == 2 and abs(dists[0] - dists[1]) > 1e-9:
            out.append(bbox)
    return out


def _candidate_decoy(
    boxes: list[tuple[int, int, int, int]], rows: list[int], target_row: int, mc: int
) -> list[tuple[int, int, int, int]]:
    other_rows = {r for r in rows if r != target_row}
    out = []
    for r0, c0, r1, c1 in boxes:
        inner_rows = range(r0 + 1, r1)
        inner_cols = range(c0 + 1, c1)
        if target_row not in inner_rows or mc in inner_cols:
            continue
        if any(r in inner_rows for r in other_rows):
            continue
        out.append((r0, c0, r1, c1))
    return out


def _candidate_empty(boxes: list[tuple[int, int, int, int]], rows: list[int]) -> list[tuple[int, int, int, int]]:
    out = []
    for r0, c0, r1, c1 in boxes:
        inner_rows = range(r0 + 1, r1)
        if any(r in inner_rows for r in rows):
            continue
        out.append((r0, c0, r1, c1))
    return out


def _pick_non_overlapping(
    candidates: list[tuple[int, int, int, int]], existing: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int] | None:
    valid = [bbox for bbox in candidates if all(_disjoint(bbox, other, gap=1) for other in existing)]
    return random.choice(valid) if valid else None


def _build_horizontal_case(H: int, W: int, want_multi: bool) -> np.ndarray | None:
    n_pairs = 3 if want_multi else random.choice([2, 3])
    rows = _pick_rows(H, n_pairs)
    if rows is None:
        return None
    colors = random.sample(list(PAIR_COLORS), n_pairs)
    mc = _meeting_column(W)
    boxes = _all_frame_bboxes(H, W)
    frames: list[tuple[int, int, int, int]] = []

    if want_multi:
        close_pair = min(((rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))), key=lambda ab: abs(ab[1] - ab[0]))
        multi_box = _pick_non_overlapping(_candidate_multi_fill(boxes, rows, close_pair, mc), frames)
        if multi_box is None:
            return None
        frames.append(multi_box)
        far_row = [r for r in rows if r not in close_pair][0]
        decoy_row = far_row
    else:
        fill_row = random.choice(rows)
        single_box = _pick_non_overlapping(_candidate_single_fill(boxes, rows, fill_row, mc), frames)
        if single_box is None:
            return None
        frames.append(single_box)
        decoy_row = random.choice(rows)

    decoy_box = _pick_non_overlapping(_candidate_decoy(boxes, rows, decoy_row, mc), frames)
    if decoy_box is None:
        return None
    frames.append(decoy_box)

    empty_box = _pick_non_overlapping(_candidate_empty(boxes, rows), frames)
    if empty_box is None:
        return None
    frames.append(empty_box)

    if want_multi:
        extra_empty = _pick_non_overlapping(_candidate_empty(boxes, rows), frames)
        if extra_empty is None:
            return None
        frames.append(extra_empty)

    grid = np.zeros((H, W), dtype=int)
    for row, color in zip(rows, colors):
        grid[row, 0] = color
        grid[row, W - 1] = color
    for bbox in frames:
        _draw_frame(grid, bbox)
    return grid


def _rand_with_parity(lo: int, hi: int, parity: int) -> int:
    choices = [n for n in range(lo, hi + 1) if n % 2 == parity]
    return random.choice(choices)


def _generate_case() -> np.ndarray:
    global _GEN_COUNTER
    mode = _GEN_COUNTER % 4
    want_vertical = mode >= 2
    want_unique = mode in (0, 2)
    want_multi = ((_GEN_COUNTER // 4) % 2) == 0
    _GEN_COUNTER += 1

    for _ in range(400):
        if want_vertical:
            H = _rand_with_parity(17, 24, 1 if want_unique else 0)
            W = random.randint(17, 24)
            base = _build_horizontal_case(W, H, want_multi)
            if base is None:
                continue
            grid = base.T.copy()
        else:
            H = random.randint(17, 24)
            W = _rand_with_parity(17, 24, 1 if want_unique else 0)
            grid = _build_horizontal_case(H, W, want_multi)
            if grid is None:
                continue
        if _validate_generated_grid(grid, require_multi=want_multi):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")


def _oracle_transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pairs = _find_edge_pairs(g)
    meetings = []
    for pair in pairs:
        meeting = _meeting_point(pair, g.shape)
        meetings.append((meeting, int(pair["color"])))
        for rr, cc in _segment_cells(pair):
            if int(g[rr, cc]) != FRAME:
                out[rr, cc] = int(pair["color"])
    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        inside = [
            {"pos": pos, "color": color}
            for pos, color in meetings
            if r0 < pos[0] < r1 and c0 < pos[1] < c1
        ]
        if inside:
            fill = _choose_frame_color(frame, inside)
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr, cc] = fill
    out[g == FRAME] = FRAME
    return out


def _validate_generated_grid(grid: np.ndarray, require_multi: bool) -> bool:
    try:
        pairs = _find_edge_pairs(grid)
        frames = _find_frames(grid)
    except ValueError:
        return False
    if not (2 <= len(pairs) <= 3 and 3 <= len(frames) <= 4):
        return False
    if len({p["color"] for p in pairs}) != len(pairs):
        return False
    if len({p["orientation"] for p in pairs}) != 1:
        return False

    out = main(grid)
    oracle = _oracle_transform(grid)
    if not np.array_equal(out, oracle):
        return False

    meetings = [{"pos": _meeting_point(pair, grid.shape), "color": int(pair["color"])} for pair in pairs]
    inside_any = False
    outside_any = False
    empty_any = False
    decoy_any = False
    multi_any = False

    for m in meetings:
        if not any(r0 < m["pos"][0] < r1 and c0 < m["pos"][1] < c1 for r0, c0, r1, c1 in [f["bbox"] for f in frames]):
            outside_any = True

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        border_ok = (
            np.all(out[r0, c0 : c1 + 1] == FRAME)
            and np.all(out[r1, c0 : c1 + 1] == FRAME)
            and np.all(out[r0 : r1 + 1, c0] == FRAME)
            and np.all(out[r0 : r1 + 1, c1] == FRAME)
        )
        if not border_ok:
            return False
        inside = [m for m in meetings if r0 < m["pos"][0] < r1 and c0 < m["pos"][1] < c1]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if inside:
            inside_any = True
            fill = _choose_frame_color(frame, inside)
            if not np.all(interior == fill):
                return False
            if len(inside) >= 2:
                multi_any = True
        else:
            empty_any = True
            crossed = any(_trail_crosses_frame(frame["bbox"], pair) for pair in pairs)
            if crossed and not np.any(interior != BG):
                return False
            if crossed:
                decoy_any = True

    return inside_any and outside_any and empty_any and decoy_any and (multi_any or not require_multi)


def generate_input() -> np.ndarray:
    return _generate_case()
