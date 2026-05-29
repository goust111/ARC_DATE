# Variant of 3e6067c3
# Dimension: D14 — 控制像素的空间编码 (meta)
# Change: 控制像素等间距排列在一行中（间隔2像素），位于所有方块下方... → 控制像素嵌入方块内部

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

# Design analysis for the D14 variant
# a) Concrete behavior:
#    Original inputs carry one detached control row whose colored pixels list the visitation order.
#    In this variant, the order is distributed across the framed squares themselves.
#    The unique start square has a special marker on its top-left corner. Every non-terminal square
#    has one extra border pixel on the side that faces the next square; the color of that pixel is
#    exactly the inner color of the next square. Example:
#      old: bottom row [2, 5, 7] means 2 -> 5 -> 7
#      new: square(2) gets a border pixel colored 5 on the side that points toward square(5),
#           square(5) gets a border pixel colored 7 that points toward square(7),
#           square(7) has no outgoing marker.
#    The solver now follows a spatial chain: find start marker -> read outgoing marker side/color ->
#    find the aligned target square -> draw the bridge colored by the current square.
#
# b) Complexity calibration:
#    A single global control row would no longer make D14 visually distinct, so it is removed entirely.
#    The per-square markers keep the scene readable because there are only two marker roles:
#      - one fixed start corner marker
#      - zero or one outgoing marker on an edge midpoint
#    This avoids an over-simplified "just point to nearest neighbor" rule by retaining target-color
#    information, while also avoiding excessive freedom that would make examples ambiguous.
#
# c) Implicit assumption audit against the original main():
#    - The original border connected-component scan assumes every border cell has the border color.
#      That is no longer true, because markers overwrite up to two border cells. The new solver
#      therefore detects each square as a non-background rectangular component and tolerates up to
#      two non-border cells on the ring.
#    - The original ctrl_r search assumes there is a non-background row below all squares.
#      This is invalid now and must be removed.
#    - The original ctrl sequence directly names the next color. In the variant, the next color is
#      read from the current square's outgoing marker, and the side of that marker constrains the
#      direction of the next square.
#
# d) Placement legality:
#    The generator keeps the original lattice/path construction so squares stay axis-aligned and
#    separated by at least one background cell. Markers are placed only on the outer perimeter of
#    a square, never in bridge cells, so input objects remain disjoint and solver-visible.
#
# e) Visual observability:
#    Across 2-4 examples, a human can notice that there is no bottom control row anymore, that one
#    square has a unique corner marker, and that small edge markers "point" toward the next square
#    while matching the next square's fill color.
#
# f) Visual cleanliness:
#    Only one corner marker and one side marker are allowed per square, so markings do not clutter
#    the border. Because bridges are still drawn only in the gaps between aligned squares, no extra
#    overlap is introduced inside the framed objects.
#
# g) Color inference:
#    The outgoing marker color is not an arbitrary remapping: it literally repeats the inner color
#    of the next square. This is easy to see from examples and avoids hidden ordering by numeric ids.
#
# New main() assumptions introduced by the variant and enforced by generate_input():
# - There is exactly one start marker, and it is always the top-left corner cell of the start frame.
# - Every frame has at most one outgoing marker, and that marker is always on an outer edge midpoint.
# - The visitation graph is a simple path, not a branch or a cycle.
# - If a frame points in a direction with color X, there is exactly one nearest aligned frame in that
#   direction whose inner color is X. The generator preserves the original row/column uniqueness rule
#   precisely so this matching step stays deterministic.
# - No detached non-background control pixels exist below the frame field; all metadata is attached
#   to frames themselves, which is why the original bottom-row scan is intentionally removed.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges, but read the control information from markers
    embedded in the square borders instead of a detached control row.

    Input structure:
    - Each framed square is a solid non-background rectangle component.
    - The start square has a special top-left corner marker.
    - Every non-terminal square has one outgoing marker on an edge midpoint.
      Its side encodes the direction of the next square, and its color matches that next square's inner color.

    Rule:
    - Start from the unique square with the corner marker.
    - Repeatedly follow the outgoing marker to choose the next aligned square of that color in that direction.
    - Draw a bridge colored by the current square's inner color, using the same bridge geometry as the original task.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert border is not None

    frames = _detect_frames(g, int(bg), int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    starts = [f for f in frames if bool(f["is_start"])]
    assert len(starts) == 1
    cur = starts[0]
    seen = {tuple(map(int, cur["bbox"]))}

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
        (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
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
                    if 0 <= r < H and 0 <= c < W:
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
                    if 0 <= r < H and 0 <= c < W:
                        out[int(r), int(c)] = int(color)

    for _ in range(int(len(frames)) - 1):
        next_dir = cur["next_dir"]
        next_color = cur["next_color"]
        if next_dir is None or next_color is None:
            break
        cands = by.get(int(next_color), [])
        best = None
        bestf = None
        for f in cands:
            if f is cur:
                continue
            if not _fits_direction(cur, f, str(next_dir)):
                continue
            (r1_, c1_) = _center(cur)
            (r2_, c2_) = _center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = -int(d)
            if best is None or cand > best:
                best = cand
                bestf = f
        assert bestf is not None
        draw_bridge(cur, bestf, int(cur["inner"]))
        cur = bestf
        key = tuple(map(int, cur["bbox"]))
        assert key not in seen
        seen.add(key)

    return out


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _fits_direction(fa: dict, fb: dict, direction: str) -> bool:
    sr, sc = _aligned(fa, fb)
    if direction == "E":
        return bool(sr and int(fb["bbox"][1]) > int(fa["bbox"][3]))
    if direction == "W":
        return bool(sr and int(fb["bbox"][3]) < int(fa["bbox"][1]))
    if direction == "S":
        return bool(sc and int(fb["bbox"][0]) > int(fa["bbox"][2]))
    if direction == "N":
        return bool(sc and int(fb["bbox"][2]) < int(fa["bbox"][0]))
    return False


def _outer_perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    coords: list[tuple[int, int]] = []
    for c in range(int(c0), int(c1) + 1):
        coords.append((int(r0), int(c)))
    for r in range(int(r0) + 1, int(r1) + 1):
        coords.append((int(r), int(c1)))
    for c in range(int(c1) - 1, int(c0) - 1, -1):
        coords.append((int(r1), int(c)))
    for r in range(int(r1) - 1, int(r0), -1):
        coords.append((int(r), int(c0)))
    return coords


def _detect_frames(g: np.ndarray, bg: int, border: int) -> list[dict]:
    lab, n = label_mask(g != int(bg), connectivity=4)
    frames: list[dict] = []
    H, W = g.shape
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 9:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if not (0 <= int(r0) <= int(r1) < int(H) and 0 <= int(c0) <= int(c1) < int(W)):
            continue
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        block = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if bool((block == int(bg)).any()):
            continue

        t = None
        inner_col = None
        outlier_cells: list[tuple[int, int, int]] | None = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(bg)).any()) or bool((inner == int(border)).any()):
                continue
            u = set(map(int, inner.flatten()))
            if len(u) != 1:
                continue

            ring_outliers: list[tuple[int, int, int]] = []
            for rr in range(int(r0), int(r1) + 1):
                for cc in range(int(c0), int(c1) + 1):
                    if int(r0 + tt) <= int(rr) <= int(r1 - tt) and int(c0 + tt) <= int(cc) <= int(c1 - tt):
                        continue
                    if int(g[int(rr), int(cc)]) != int(border):
                        ring_outliers.append((int(rr), int(cc), int(g[int(rr), int(cc)])))
            if len(ring_outliers) > 2:
                continue
            perimeter_set = set(_outer_perimeter_coords((int(r0), int(c0), int(r1), int(c1))))
            if any((int(rr), int(cc)) not in perimeter_set for rr, cc, _val in ring_outliers):
                continue

            t = int(tt)
            inner_col = int(next(iter(u)))
            outlier_cells = ring_outliers
            break

        if t is None or inner_col is None or outlier_cells is None:
            continue

        is_start = False
        next_dir = None
        next_color = None
        for rr, cc, val in outlier_cells:
            if int(rr) == int(r0) and int(cc) == int(c0):
                is_start = True
                continue
            if int(rr) == int(r0) and int(c0) < int(cc) < int(c1):
                next_dir = "N"
                next_color = int(val)
                continue
            if int(rr) == int(r1) and int(c0) < int(cc) < int(c1):
                next_dir = "S"
                next_color = int(val)
                continue
            if int(cc) == int(c0) and int(r0) < int(rr) < int(r1):
                next_dir = "W"
                next_color = int(val)
                continue
            if int(cc) == int(c1) and int(r0) < int(rr) < int(r1):
                next_dir = "E"
                next_color = int(val)
                continue
            raise AssertionError("Unexpected marker position")

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
                "is_start": bool(is_start),
                "next_dir": next_dir,
                "next_color": next_color,
            }
        )

    return frames


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
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
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


def _edge_marker_cell(frame: dict, direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    rm = int((r0 + r1) // 2)
    cm = int((c0 + c1) // 2)
    if direction == "N":
        return int(r0), int(cm)
    if direction == "S":
        return int(r1), int(cm)
    if direction == "W":
        return int(rm), int(c0)
    if direction == "E":
        return int(rm), int(c1)
    raise ValueError(f"Unknown direction: {direction}")


def _step_direction(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    dr = int(b[0]) - int(a[0])
    dc = int(b[1]) - int(a[1])
    if dr == -1 and dc == 0:
        return "N"
    if dr == 1 and dc == 0:
        return "S"
    if dr == 0 and dc == -1:
        return "W"
    if dr == 0 and dc == 1:
        return "E"
    return None


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    marker_choices = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(marker_choices) < 4:
        return None
    start_marker = int(np.random.choice(np.array(marker_choices, dtype=int)))
    inner_palette = [int(d) for d in marker_choices if int(d) != int(start_marker)]
    if len(inner_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(10, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            vis.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    bottom_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 5))
    H = int(max_r1 + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)

    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        rr = int(rr)
        cc = int(cc)
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_marker)}
        choices = [int(x) for x in inner_palette if int(x) not in forbid]
        if not choices:
            return None
        col = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(col)
        used_in_row[int(rr)].add(int(col))
        used_in_col[int(cc)].add(int(col))

    if len(colors_by_slot) != len(path):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
        )

    start_frame = slot_to_frame[(0, 0)]
    sr0, sc0, _sr1, _sc1 = map(int, start_frame["bbox"])
    out[int(sr0), int(sc0)] = int(start_marker)

    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        direction = _step_direction(a_slot, b_slot)
        if direction is None:
            return None
        mr, mc = _edge_marker_cell(fa, str(direction))
        out[int(mr), int(mc)] = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])

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
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
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
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
