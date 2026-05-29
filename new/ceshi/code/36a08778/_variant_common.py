from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARC2_DIR = ROOT / "arc2_solver_select"
if str(ARC2_DIR) not in sys.path:
    sys.path.insert(0, str(ARC2_DIR))

from arc_utils import components_mask

BG = 7
WATER = 6
SHELF = 2
DRAIN = 1
DEPTH_COLORS = (6, 1, 8)
STOP_LIMIT = 3


def _detect_roles(g: np.ndarray, variant_id: str) -> tuple[int, int, int, int | None]:
    cnt = Counter(map(int, g.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non_bg = [int(c) for c in cnt if int(c) != int(bg)]
    assert non_bg

    water = None
    for col in sorted(non_bg):
        if bool((g[0] == int(col)).any()):
            water = int(col)
            break
    if water is None:
        water = int(min(non_bg, key=lambda c: (int(cnt[int(c)]), int(c))))

    rest = [int(c) for c in non_bg if int(c) != int(water)]
    assert rest
    shelf = int(max(rest, key=lambda c: (int(cnt[int(c)]), -int(c))))
    drain = None
    if variant_id == "D7_0":
        for col in rest:
            if int(col) != int(shelf):
                drain = int(col)
                break
    return bg, water, shelf, drain


def _get_sources(g: np.ndarray, water: int) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for coords in components_mask(g == int(water), connectivity=8):
        if coords.size == 0 or not bool((coords[:, 0] == 0).any()):
            continue
        rmax = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == int(rmax)})
        for c in cols:
            sources.append((int(rmax), int(c)))
    return sources


def _build_shelf_segments(g: np.ndarray, shelf: int) -> list[list[tuple[int, int]]]:
    H, W = g.shape
    shelf_segs: list[list[tuple[int, int]]] = [[] for _ in range(H)]
    for r in range(H):
        c = 0
        while c < W:
            if int(g[r, c]) != int(shelf):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < W and int(g[r, c + 1]) == int(shelf):
                c += 1
            shelf_segs[r].append((int(c0), int(c)))
            c += 1
    return shelf_segs


def _find_shelf_seg(row_segs: list[tuple[int, int]], c: int) -> tuple[int, int] | None:
    for a, b in row_segs:
        if int(a) <= int(c) <= int(b):
            return (int(a), int(b))
    return None


def _find_hole_structure(row_segs: list[tuple[int, int]], c: int) -> dict[str, int] | None:
    for idx, (a, b) in enumerate(row_segs):
        if not (int(a) <= int(c) <= int(b)):
            continue
        if idx + 1 < len(row_segs):
            na, nb = row_segs[idx + 1]
            if int(na) == int(b) + 2:
                return {
                    "a": int(a),
                    "b": int(nb),
                    "gap": int(b + 1),
                }
        if idx - 1 >= 0:
            pa, pb = row_segs[idx - 1]
            if int(a) == int(pb) + 2:
                return {
                    "a": int(pa),
                    "b": int(b),
                    "gap": int(pb + 1),
                }
    return None


def _paint_vertical(out: np.ndarray, rr: int, cc: int, color: int) -> None:
    out[int(rr), int(cc)] = int(color)


def _paint_run(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    rr: int,
    start: int,
    stop: int,
    color: int,
) -> bool:
    step = 1 if int(stop) >= int(start) else -1
    x = int(start)
    while True:
        if x != int(start) and int(g[int(rr), int(x)]) != int(bg):
            return False
        if int(g[int(rr), int(x)]) == int(bg):
            out[int(rr), int(x)] = int(color)
        if int(x) == int(stop):
            return True
        x += step


def _depth_color(depth: int) -> int:
    return int(DEPTH_COLORS[min(int(depth), len(DEPTH_COLORS) - 1)])


def _overflow_side_from_edge(c: int, w: int) -> str:
    return "left" if int(c) * 2 <= int(w - 1) else "right"


def _entry_side(c: int, a: int, b: int) -> str:
    length = int(b - a + 1)
    offset = int(c - a)
    if length % 2 == 1 and offset == length // 2:
        return "center"
    if offset < length / 2:
        return "left"
    return "right"


def run_main(grid: np.ndarray, variant_id: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, water, shelf, drain = _detect_roles(g, variant_id)
    out = g.copy()

    sources = _get_sources(g, water)
    shelf_segs = _build_shelf_segments(g, shelf)

    if variant_id in {"D4_1", "D7_3", "D14_2"}:
        q = deque((int(r), int(c), 0) for r, c in sources)
        seen: set[tuple[int, int, int]] = set()
    else:
        q = deque((int(r), int(c), None) for r, c in sources)
        seen = set()

    while q:
        r, c, state = q.popleft()
        key = (int(r), int(c), int(state)) if state is not None else (int(r), int(c))
        if key in seen:
            continue
        seen.add(key)

        depth = int(state) if variant_id == "D4_1" and state is not None else 0
        hits = int(state) if variant_id in {"D7_3", "D14_2"} and state is not None else 0
        cur_color = _depth_color(depth) if variant_id == "D4_1" else int(water)

        rr, cc = int(r), int(c)
        _paint_vertical(out, rr, cc, cur_color)

        while True:
            nr = int(rr + 1)
            if nr >= H:
                break
            if variant_id == "D7_0" and drain is not None and int(g[nr, cc]) == int(drain):
                break

            below = int(g[nr, cc])
            if below == int(shelf):
                if variant_id == "D2_1":
                    hole = _find_hole_structure(shelf_segs[nr], cc)
                    if hole is not None:
                        gap_col = int(hole["gap"])
                        ok = _paint_run(g, out, bg, rr, cc, gap_col, cur_color)
                        if ok and gap_col != int(cc):
                            q.append((int(rr), gap_col, None))
                        break

                seg = _find_shelf_seg(shelf_segs[nr], cc)
                if seg is None:
                    break
                a, b = seg
                left = int(max(0, a - 1))
                right = int(min(W - 1, b + 1))

                if variant_id == "D2_0":
                    side = _overflow_side_from_edge(cc, W)
                    target = int(left if side == "left" else right)
                    ok = _paint_run(g, out, bg, rr, cc, target, cur_color)
                    if ok and target != int(cc):
                        q.append((int(rr), target, None))
                    break

                if variant_id == "D14_2":
                    side = "left" if hits % 2 == 0 else "right"
                    target = int(left if side == "left" else right)
                    ok = _paint_run(g, out, bg, rr, cc, target, cur_color)
                    if ok and target != int(cc):
                        q.append((int(rr), target, int(hits + 1)))
                    break

                if variant_id == "D14_3":
                    side = _entry_side(cc, a, b)
                    targets = []
                    if side in {"left", "center"}:
                        targets.append(left)
                    if side in {"right", "center"}:
                        targets.append(right)
                    for target in targets:
                        ok = _paint_run(g, out, bg, rr, cc, int(target), cur_color)
                        if ok and int(target) != int(cc):
                            q.append((int(rr), int(target), None))
                    break

                if variant_id == "D7_3":
                    _paint_run(g, out, bg, rr, cc, left, cur_color)
                    _paint_run(g, out, bg, rr, cc, right, cur_color)
                    next_hits = int(hits + 1)
                    if next_hits < STOP_LIMIT:
                        if left != int(cc):
                            q.append((int(rr), left, next_hits))
                        if right != int(cc):
                            q.append((int(rr), right, next_hits))
                    break

                _paint_run(g, out, bg, rr, cc, left, cur_color)
                _paint_run(g, out, bg, rr, cc, right, cur_color)
                next_depth = int(depth + 1)
                if left != int(cc):
                    q.append((int(rr), left, next_depth if variant_id == "D4_1" else None))
                if right != int(cc):
                    q.append((int(rr), right, next_depth if variant_id == "D4_1" else None))
                break

            rr = nr
            _paint_vertical(out, rr, cc, cur_color)

    return out


def _sample_source_cols(variant_id: str, W: int) -> list[int]:
    if variant_id == "D2_0" and W >= 12:
        left = random.randint(1, max(1, W // 3))
        right = random.randint(max(left + 6, (2 * W) // 3), W - 2)
        return [int(left), int(right)]
    if variant_id in {"D7_3", "D14_2"}:
        return [random.randint(1, W - 2)]
    if W >= 16 and random.random() < 0.5:
        left = random.randint(1, W // 2 - 2)
        right = random.randint(W // 2 + 1, W - 2)
        return [int(left), int(right)]
    return [random.randint(1, W - 2)]


def _choose_solid_shelf(W: int, col: int, variant_id: str) -> tuple[int, int]:
    for _ in range(80):
        if variant_id == "D14_3":
            mode = random.choice(["left", "right", "center"])
            if mode == "center":
                choices = [n for n in (3, 5, 7, 9) if n <= W]
                if not choices:
                    continue
                length = random.choice(choices)
                a = int(col - length // 2)
                b = int(a + length - 1)
                if 0 <= a and b < W:
                    return int(a), int(b)
                continue

            length = random.randint(4, min(8, W))
            if mode == "left":
                max_off = max(0, int(np.ceil(length / 2)) - 1)
                offset = random.randint(0, max_off)
            else:
                min_off = max(1, int(length // 2))
                offset = random.randint(min_off, length - 1)
            a = int(col - offset)
            b = int(a + length - 1)
            if 0 <= a and b < W:
                return int(a), int(b)
            continue

        length = random.randint(3, min(8, W))
        offset = random.randint(0, length - 1)
        a = int(col - offset)
        b = int(a + length - 1)
        if 0 <= a and b < W:
            return int(a), int(b)
    raise RuntimeError("failed to choose solid shelf")


def _choose_hole_shelf(W: int, col: int) -> tuple[list[tuple[int, int]], int]:
    for _ in range(80):
        gap_side = random.choice(["left", "right"])
        left_len = random.randint(2, 4)
        right_len = random.randint(2, 4)
        if gap_side == "right":
            offset = random.randint(0, left_len - 1)
            a1 = int(col - offset)
            b1 = int(a1 + left_len - 1)
            gap = int(b1 + 1)
            a2 = int(gap + 1)
            b2 = int(a2 + right_len - 1)
            if 0 <= a1 and b2 < W:
                return [(int(a1), int(b1)), (int(a2), int(b2))], int(gap)
        else:
            offset = random.randint(0, right_len - 1)
            a2 = int(col - offset)
            b2 = int(a2 + right_len - 1)
            gap = int(a2 - 1)
            b1 = int(gap - 1)
            a1 = int(b1 - left_len + 1)
            if 0 <= a1 and b2 < W:
                return [(int(a1), int(b1)), (int(a2), int(b2))], int(gap)
    raise RuntimeError("failed to choose hole shelf")


def _reserve_clear(cells: set[tuple[int, int]], row: int, col0: int, col1: int) -> None:
    lo, hi = sorted((int(col0), int(col1)))
    for c in range(lo, hi + 1):
        cells.add((int(row), int(c)))


def _place_attempt(variant_id: str) -> np.ndarray | None:
    H = random.randint(12, 28)
    W = random.randint(10, 28)
    grid = np.full((H, W), BG, dtype=int)

    source_cols = _sample_source_cols(variant_id, W)
    reserved_clear: set[tuple[int, int]] = set()
    shelf_cells: set[tuple[int, int]] = set()
    stream_specs: list[dict[str, int]] = []

    for sc in source_cols:
        grid[0, sc] = WATER
        grid[1, sc] = WATER
        reserved_clear.add((0, int(sc)))
        reserved_clear.add((1, int(sc)))
        stream_specs.append({"row": 1, "col": int(sc), "depth": 0})

    min_depth = 4 if variant_id in {"D4_1", "D7_3", "D14_2"} else 3
    max_depth = 5 if variant_id in {"D2_0", "D2_1", "D14_2"} else 4
    if variant_id == "D7_3":
        min_depth = STOP_LIMIT + 1
        max_depth = STOP_LIMIT + 1

    shelves_placed = 0
    hole_used = False
    center_used = False

    while stream_specs:
        spec = stream_specs.pop(0)
        row = int(spec["row"])
        col = int(spec["col"])
        depth = int(spec["depth"])

        if depth >= max_depth or row + 2 >= H - 1:
            continue
        if depth >= min_depth and random.random() < 0.25:
            continue

        placed = False
        for _ in range(80):
            shelf_row = random.randint(row + 2, min(H - 2, row + 6))
            spread_row = int(shelf_row - 1)

            blocked = False
            for rr in range(row + 1, shelf_row):
                if (int(rr), int(col)) in shelf_cells:
                    blocked = True
                    break
            if blocked:
                continue

            use_hole = variant_id == "D2_1" and (not hole_used or random.random() < 0.5)
            if use_hole:
                segs, child_col = _choose_hole_shelf(W, col)
                spread_targets = [int(child_col)]
                shelf_parts = segs
            else:
                a, b = _choose_solid_shelf(W, col, variant_id)
                child_left = int(max(0, a - 1))
                child_right = int(min(W - 1, b + 1))
                shelf_parts = [(int(a), int(b))]
                if variant_id in {"D2_0", "D14_2"}:
                    side = _overflow_side_from_edge(col, W) if variant_id == "D2_0" else ("left" if depth % 2 == 0 else "right")
                    spread_targets = [int(child_left if side == "left" else child_right)]
                elif variant_id == "D14_3":
                    side = _entry_side(col, a, b)
                    if side == "center":
                        center_used = True
                        spread_targets = [int(child_left), int(child_right)]
                    elif side == "left":
                        spread_targets = [int(child_left)]
                    else:
                        spread_targets = [int(child_right)]
                elif variant_id == "D7_3" and depth + 1 >= STOP_LIMIT:
                    spread_targets = []
                else:
                    spread_targets = [int(child_left), int(child_right)]

            conflict = False
            for a0, b0 in shelf_parts:
                for cc in range(a0, b0 + 1):
                    cell = (int(shelf_row), int(cc))
                    if cell in shelf_cells or cell in reserved_clear:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue

            spread_cells = {(int(spread_row), int(col))}
            for target in spread_targets:
                lo, hi = sorted((int(col), int(target)))
                for cc in range(lo, hi + 1):
                    spread_cells.add((int(spread_row), int(cc)))
            if any(cell in shelf_cells for cell in spread_cells):
                continue

            for rr in range(row + 1, shelf_row):
                reserved_clear.add((int(rr), int(col)))
            for sr, sc in spread_cells:
                reserved_clear.add((int(sr), int(sc)))
            for a0, b0 in shelf_parts:
                for cc in range(a0, b0 + 1):
                    shelf_cells.add((int(shelf_row), int(cc)))
                    grid[int(shelf_row), int(cc)] = SHELF

            if use_hole:
                hole_used = True

            next_depth = int(depth + 1)
            if variant_id == "D7_3" and next_depth >= STOP_LIMIT:
                pass
            else:
                for target in spread_targets:
                    if int(target) != int(col):
                        stream_specs.append({"row": int(spread_row), "col": int(target), "depth": next_depth})

            shelves_placed += 1
            placed = True
            break

        if not placed and depth < min_depth - 1:
            return None

    if shelves_placed < 3:
        return None
    if variant_id == "D2_1" and not hole_used:
        return None
    if variant_id == "D14_3" and not center_used and random.random() < 0.7:
        return None

    if variant_id == "D7_0":
        base_out = run_main(grid.copy(), "BASE")
        candidates: list[tuple[int, int]] = []
        for r in range(2, H - 1):
            for c in range(1, W - 1):
                if int(grid[r, c]) != int(BG) or int(base_out[r, c]) != int(WATER):
                    continue
                if int(base_out[r - 1, c]) != int(WATER):
                    continue
                candidates.append((int(r), int(c)))
        if len(candidates) < 2:
            return None
        random.shuffle(candidates)
        used_cols: set[int] = set()
        drains = []
        for r, c in candidates:
            if c in used_cols:
                continue
            drains.append((int(r), int(c)))
            used_cols.add(int(c))
            if len(drains) >= random.randint(2, 3):
                break
        if len(drains) < 2:
            return None
        for r, c in drains:
            grid[int(r), int(c)] = DRAIN

    out = run_main(grid.copy(), variant_id)
    if np.array_equal(out, grid):
        return None
    if variant_id == "D7_0" and int((grid == DRAIN).sum()) < 2:
        return None
    return grid


def generate_input_for_variant(variant_id: str) -> np.ndarray:
    for _ in range(300):
        result = _place_attempt(variant_id)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input for {variant_id} after 300 attempts")


def build_variant_functions(variant_id: str):
    def main(grid: np.ndarray) -> np.ndarray:
        return run_main(grid, variant_id)

    def generate_input() -> np.ndarray:
        return generate_input_for_variant(variant_id)

    return main, generate_input
