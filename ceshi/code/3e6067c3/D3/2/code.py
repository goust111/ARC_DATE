# Variant of 3e6067c3
# Dimension: D3 — 源对象：控制像素的编码形式 (src)
# Change: 底部区域的单行控制像素序列，每个像素代表一种颜色，从左到右编码访问顺序... → 散布的标记像素
"""
Design analysis for the D3 variant

a) Concrete behavior:
The framed squares and bridge-drawing rule stay unchanged. Only the control source changes.
Instead of one bottom row of colored pixels, the input now has several control markers scattered on
the outer border. Each marker is a single edge pixel whose color is the next square color to visit.
Next to that edge pixel, extending inward orthogonally, there is a short uniform bar. The bar length
encodes the order number: length 1 means first, length 2 means second, etc. The solver must collect
markers from all four edges, recover their inward bar lengths, sort by that length, and then reuse
the original bridge logic on the resulting color sequence.

Small trace:
Original input logic:
- bottom row pixels: [red, blue, yellow, green]
- solver reads left-to-right and gets sequence red -> blue -> yellow -> green
Variant input logic:
- top edge red with inward bar length 1
- right edge blue with inward bar length 2
- left edge yellow with inward bar length 3
- bottom edge green with inward bar length 4
- solver gathers all edge markers, sorts by bar length, and recovers the same sequence.
Difference:
- old code depends on one dedicated row and spatial left-to-right ordering
- new code depends on edge collection plus local length decoding.

b) Complexity calibration:
The control source is visibly different in every sample because the markers occupy at least two
different sides. The order bars use only lengths 1..N with N in {4,5}, so the rule is learnable
from a few examples and never hides behind large numeric freedom. The marker colors still directly
name the squares, which preserves the original problem's core matching behavior.

c) Hidden assumptions in the original main():
- "bottom-most row below frames with >=2 non-bg pixels" no longer holds, so that detection must go.
- "control order equals left-to-right order" no longer holds, so sequence extraction must sort by
  decoded bar length instead.
- frame detection, start-frame disambiguation, row/column alignment, and bridge painting still hold,
  so they are kept unchanged.

d) Placement legality:
The generator reserves clear top/bottom/left/right margins around the frame lattice. Order bars stay
entirely inside those margins, so they never overlap frames or future bridges. Top/bottom markers use
columns over the central frame span; left/right markers use rows over the central frame span. This
prevents perpendicular control bars from crossing near the corners.

e) Visual observability:
Humans can see that the only input/output difference in the main area is still the added bridges, and
that the "instruction area" is no longer a row but a set of numbered edge tags. Because the number is
shown by a visible count of adjacent pixels, the new mechanism is inferable from 2-3 examples.

f) Visual cleanliness:
Controls are limited to simple 1-cell markers plus straight bars in the outer margin. No control
element overlaps a frame, another control bar, or a bridge corridor.

g) Color inference:
No hidden color ordering is used. The marker color itself is the square color to visit, exactly as in
the source task. The bar color is only a neutral order-indicator color and is excluded from the frame
palette by construction.

h) New solver assumptions and generator guarantees:
- every control marker lies on one of the four edges
- each marker has one straight inward bar of positive length
- bar lengths are unique and equal to 1..N
- at least two sides are used
The generator enforces all four properties, and validation runs `main(input)` against the synthesized
target output before accepting a sample.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


def _find_frames(g: np.ndarray, border: int) -> list[dict]:
    lab, n = label_mask(g == int(border), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        t = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0 or bool((inner == int(border)).any()):
                continue
            u = set(map(int, inner.flatten()))
            if len(u) != 1:
                continue
            ok = True
            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue
            t = int(tt)
            inner_col = int(next(iter(u)))
            break

        if t is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            }
        )
    return frames


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _extract_scattered_controls(g: np.ndarray, bg: int, border: int) -> list[int]:
    H, W = g.shape
    markers: list[tuple[int, int]] = []

    def read_marker(r: int, c: int, dr: int, dc: int) -> None:
        marker_col = int(g[int(r), int(c)])
        if marker_col in (int(bg), int(border)):
            return
        rr = int(r + dr)
        cc = int(c + dc)
        if not in_bounds((int(H), int(W)), int(rr), int(cc)):
            return
        bar_col = int(g[int(rr), int(cc)])
        if bar_col == int(bg):
            return
        length = 0
        while in_bounds((int(H), int(W)), int(rr), int(cc)) and int(g[int(rr), int(cc)]) == int(bar_col):
            length += 1
            rr += int(dr)
            cc += int(dc)
        if length >= 1:
            markers.append((int(length), int(marker_col)))

    for c in range(1, int(W) - 1):
        read_marker(0, int(c), 1, 0)
        read_marker(int(H - 1), int(c), -1, 0)
    for r in range(1, int(H) - 1):
        read_marker(int(r), 0, 0, 1)
        read_marker(int(r), int(W - 1), 0, -1)

    assert len(markers) >= 2
    orders = [int(k) for k, _col in markers]
    assert len(set(orders)) == len(orders)
    markers.sort(key=lambda item: int(item[0]))
    return [int(col) for _k, col in markers]


def _choose_start_frame(ctrl: list[int], by: dict[int, list[dict]]) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

    nxt_color = int(ctrl[1])
    nxt_cands = by.get(nxt_color, [])
    best = None
    bestf = None
    for frame in start_cands:
        for g2 in (nxt_cands if nxt_cands else [frame]):
            sr, sc = _aligned(frame, g2)
            aligned_score = 1 if (sr or sc) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(g2)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                bestf = frame
    assert bestf is not None
    return bestf


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following scattered edge markers.

    Variant rule:
    - Control colors are no longer on one bottom row.
    - Each control is an edge marker pixel whose inward bar length encodes its sequence index.
    - After sorting markers by bar length, the original bridge logic is reused unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _find_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)

    ctrl = _extract_scattered_controls(g, int(bg), int(border))
    assert len(ctrl) >= 2

    cur = _choose_start_frame(ctrl, by)
    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        r0a, c0a, r1a, c1a = map(int, fa["bbox"])
        r0b, c0b, r1b, c1b = map(int, fb["bbox"])
        ia = fa["inner_bbox"]
        sr, sc = _aligned(fa, fb)
        if sr:
            rr0, rr1 = int(ia[0]), int(ia[2])
            if int(c1a) < int(c0b):
                cc0, cc1 = int(c1a + 1), int(c0b - 1)
            else:
                cc0, cc1 = int(c1b + 1), int(c0a - 1)
            for r in range(int(rr0), int(rr1) + 1):
                for c in range(int(cc0), int(cc1) + 1):
                    if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                        out[int(r), int(c)] = int(color)
            return
        if sc:
            cc0, cc1 = int(ia[1]), int(ia[3])
            if int(r1a) < int(r0b):
                rr0, rr1 = int(r1a + 1), int(r0b - 1)
            else:
                rr0, rr1 = int(r1b + 1), int(r0a - 1)
            for r in range(int(rr0), int(rr1) + 1):
                for c in range(int(cc0), int(cc1) + 1):
                    if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                        out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        best = None
        bestf = None
        for frame in by[b]:
            sr, sc = _aligned(cur, frame)
            aligned_score = 1 if (sr or sc) else 0
            r1, c1 = _center(cur)
            r2, c2 = _center(frame)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                bestf = frame
        assert bestf is not None
        draw_bridge(cur, bestf, int(a))
        cur = bestf

    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    return set()


def _sample_path(rows: int, cols: int, n_frames: int) -> list[tuple[int, int]] | None:
    for _ in range(400):
        path = [(0, 0)]
        visited = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = path[-1]
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(rows), int(cols)), int(rr), int(cc)) and (int(rr), int(cc)) not in visited
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            path.append((int(nxt[0]), int(nxt[1])))
            visited.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in path}) < 2 or len({c for _r, c in path}) < 2:
            continue
        return path
    return None


def _sample_side_assignment(n_frames: int) -> list[str]:
    all_sides = ["top", "bottom", "left", "right"]
    n_sides = int(np.random.randint(2, min(4, int(n_frames)) + 1))
    chosen = list(np.random.choice(np.array(all_sides, dtype=object), size=int(n_sides), replace=False))
    assigned = [str(side) for side in chosen]
    for _ in range(int(n_frames) - int(n_sides)):
        assigned.append(str(chosen[int(np.random.randint(0, len(chosen)))]))
    np.random.shuffle(assigned)
    return assigned


def _paint_controls(
    out: np.ndarray,
    *,
    bg: int,
    ctrl_colors: list[int],
    order_color: int,
    top_pad: int,
    left_pad: int,
    max_r1: int,
    max_c1: int,
) -> bool:
    H, W = out.shape
    assigned_sides = _sample_side_assignment(len(ctrl_colors))

    pools = {
        "top": [int(c) for c in range(int(left_pad), int(max_c1) + 1)],
        "bottom": [int(c) for c in range(int(left_pad), int(max_c1) + 1)],
        "left": [int(r) for r in range(int(top_pad), int(max_r1) + 1)],
        "right": [int(r) for r in range(int(top_pad), int(max_r1) + 1)],
    }
    side_counts = Counter(assigned_sides)
    chosen_positions: dict[str, list[int]] = {}
    for side, count in side_counts.items():
        pool = pools[str(side)]
        if len(pool) < int(count):
            return False
        pos = list(np.random.choice(np.array(pool, dtype=int), size=int(count), replace=False))
        np.random.shuffle(pos)
        chosen_positions[str(side)] = [int(v) for v in pos]

    def paint_one(side: str, order: int, color: int) -> bool:
        pos = int(chosen_positions[str(side)].pop())
        if side == "top":
            cells = [(0, int(pos))] + [(int(d), int(pos)) for d in range(1, int(order) + 1)]
        elif side == "bottom":
            cells = [(int(H - 1), int(pos))] + [(int(H - 1 - d), int(pos)) for d in range(1, int(order) + 1)]
        elif side == "left":
            cells = [(int(pos), 0)] + [(int(pos), int(d)) for d in range(1, int(order) + 1)]
        else:
            cells = [(int(pos), int(W - 1))] + [(int(pos), int(W - 1 - d)) for d in range(1, int(order) + 1)]

        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return False
            if int(out[int(r), int(c)]) != int(bg):
                return False

        marker_r, marker_c = cells[0]
        out[int(marker_r), int(marker_c)] = int(color)
        for r, c in cells[1:]:
            out[int(r), int(c)] = int(order_color)
        return True

    for order, (side, color) in enumerate(zip(assigned_sides, ctrl_colors), start=1):
        if not paint_one(str(side), int(order), int(color)):
            return False
    return True


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    remaining = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(remaining) < 4:
        return None
    order_color = int(np.random.choice(np.array(remaining, dtype=int)))
    inner_palette = [int(d) for d in remaining if int(d) != int(order_color)]
    if len(inner_palette) < 4:
        return None

    t, inner = [(1, 1), (1, 2), (1, 3), (2, 1)][int(np.random.randint(0, 4))]
    outer = int(inner + 2 * t)

    rows = int(np.random.randint(2, 5))
    cols = int(np.random.randint(2, 5))
    max_frames = int(rows * cols)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(5, max_frames)) + 1))

    path = _sample_path(int(rows), int(cols), int(n_frames))
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(rows) - 1)]
    col_gap = [int(np.random.randint(1, 3)) for _ in range(int(cols) - 1)]
    top_pad = int(n_frames + 1)
    left_pad = int(n_frames + 1)
    bottom_pad = int(n_frames + 1)
    right_pad = int(n_frames + 1)

    r0s = [int(top_pad)]
    for k in range(int(rows) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_pad)]
    for k in range(int(cols) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    H = int(max_r1 + 1 + bottom_pad)
    W = int(max_c1 + 1 + right_pad)
    if not (12 <= int(H) <= 30 and 12 <= int(W) <= 30):
        return None

    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)
    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_color)}
        choices = [int(x) for x in inner_palette if int(x) not in forbid]
        if not choices:
            return None
        color = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(color)
        used_in_row[int(rr)].add(int(color))
        used_in_col[int(cc)].add(int(color))

    if len(colors_by_slot) != len(path):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells or cells & bridge_cells_all:
            ok = False
            break
        for r, c in cells:
            if int(out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    if not _paint_controls(
        out,
        bg=int(bg),
        ctrl_colors=ctrl_colors,
        order_color=int(order_color),
        top_pad=int(top_pad),
        left_pad=int(left_pad),
        max_r1=int(max_r1),
        max_c1=int(max_c1),
    ):
        return None

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape or not bool(np.array_equal(pred, out)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
