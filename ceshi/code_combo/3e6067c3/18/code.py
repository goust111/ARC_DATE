# Combo variant of 3e6067c3 (index 18)
# combo_spec: D9/0+D5/1
# Name: 桥梁色控制语义+专用第三色桥梁

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


"""
Design analysis for combo D9/0 + D5/1

a) Concrete fused behavior
   - Scene structure stays visually close to the source task: equal-sized framed squares on a row/column
     lattice, with one sparse control row below them.
   - Original task:
     control pixels name the visited square colors, so path decoding and bridge coloring are entangled.
   - Fused task:
     the path is no longer read from colors at all. Start from the top-left-most frame, then repeatedly move
     to the unique nearest unvisited aligned frame (same row band or column band). The single control row
     gives one bridge color per step. Each such color is a dedicated third color, different from both endpoint
     square colors. So geometry decides "where to go", while the control row decides only "what color to use".
   - Small trace:
     suppose geometry forces frame order A -> B -> C -> D with square colors [2, 5, 7, 3]. The control row
     may be [9, 4, 9]. Then bridges are A-B in 9, B-C in 4, C-D in 9. Under the original rule the same three
     bridges would have been colored [2, 5, 7] and the row would have named the squares instead.

b) Complexity calibration
   - To keep the fusion visible, bridge colors come from a small palette that is disjoint from the frame-color
     palette. This makes it obvious that control pixels are paint instructions, not object identifiers.
   - The bridge palette is small (usually 2-3 colors) and reused across edges, so the control row has repeated
     support instead of becoming arbitrary noise.
   - Path geometry is validated so each next move is the unique nearest aligned unvisited frame; otherwise the
     generator rejects the sample.

c) Implicit-assumption audit against the source main()
   - `by[inner_color]` lookups are invalid: control pixels no longer refer to square identities.
   - `start_color = ctrl[0]` is invalid: the start square is geometric, not color-addressed.
   - `assert cur["inner"] == a` is invalid: bridge color need not match the current square color.
   - `if b not in by` is invalid: the next control pixel is not a destination-square color.
   - `draw_bridge(cur, nxt, a)` is invalid: paint comes from the control row directly and must be allowed to be
     a third color unrelated to either endpoint.
   - Frame extraction, row/column alignment, and rectangular bridge rasterization still remain valid.

d) Placement legality
   - Frames are placed on a lattice with uniform outer size and at least one background cell between neighbors.
   - The generator builds only bridges returned by `_bridge_cells()`, rejects empty/overlapping/out-of-bounds
     corridors, and draws bridges only on background.
   - The control row is placed strictly below all frames and widened as needed to fit columns 1,3,5,...

e) Visual observability
   - Across examples, the top-left frame is consistently the geometric start and the path follows nearest
     aligned neighbors. The control row colors often do not appear inside any square, making the semantic split
     legible from a few input/output pairs.
   - Because every bridge color differs from both endpoint fills, the D5 part is visible on every edge rather
     than only on rare mismatches.

f) Visual cleanliness
   - Only three layers remain: frames, empty bridge corridors, and one sparse control row. No symbols are
     stacked on top of frames, and the disjoint bridge palette avoids confusing bridge paint with square fills.

g) Inferability of the changed rule
   - Frame colors come from a small frame palette; bridge colors come from a separate small bridge palette.
   - Repetition within the bridge row provides structure, while palette disjointness and per-edge endpoint
     mismatch make the "control row = bridge paint recipe" rule recoverable by humans.

New solver assumptions introduced by the fusion
   - The intended path starts at the unique top-left-most frame.
   - At each step there is a unique nearest aligned unvisited frame under Manhattan distance.
   - The control row length does not exceed the number of drawable path steps.
   The generator enforces all three properties explicitly.
"""


_DIGITS = [int(i) for i in range(10)]


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _extract_frames(g: np.ndarray, border: int) -> list[dict]:
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
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            uniq = set(map(int, inner.flatten()))
            if len(uniq) != 1:
                continue
            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                continue
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                continue
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                continue
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                continue
            t = int(tt)
            inner_col = int(next(iter(uniq)))
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


def _find_control_row(g: np.ndarray, bg: int, frames: list[dict]) -> list[int]:
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    rows = [int(r) for r in range(int(max_frame_r) + 1, int(g.shape[0])) if bool((g[int(r)] != int(bg)).any())]
    assert rows
    ctrl_r = int(rows[-1])
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert ctrl
    return ctrl


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _top_left_key(f: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = map(int, f["bbox"])
    return int(r0), int(c0)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _aligned_distance(f1: dict, f2: dict) -> int | None:
    sr, sc = _aligned(f1, f2)
    if not (sr or sc):
        return None
    r1, c1 = _center(f1)
    r2, c2 = _center(f2)
    return int(abs(int(r1) - int(r2)) + abs(int(c1) - int(c2)))


def _best_unvisited_neighbor(cur: dict, remaining: list[dict]) -> dict | None:
    scored: list[tuple[int, int, int, dict]] = []
    for cand in remaining:
        dist = _aligned_distance(cur, cand)
        if dist is None:
            continue
        r0, c0, _r1, _c1 = map(int, cand["bbox"])
        scored.append((int(dist), int(r0), int(c0), cand))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]


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


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    for r, c in _bridge_cells(fa, fb):
        if 0 <= int(r) < int(out.shape[0]) and 0 <= int(c) < int(out.shape[1]):
            out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _extract_frames(g, int(border))
    assert frames

    ctrl = _find_control_row(g, int(bg), frames)

    out = g.copy()
    cur = min(frames, key=_top_left_key)
    remaining = [f for f in frames if f is not cur]

    for color in ctrl:
        nxt = _best_unvisited_neighbor(cur, remaining)
        if nxt is None:
            break
        _draw_bridge(out, cur, nxt, int(color))
        remaining.remove(nxt)
        cur = nxt

    return out


def _path_respects_nearest_rule(path_frames: list[dict]) -> bool:
    cur = path_frames[0]
    remaining = list(path_frames[1:])
    for expected in path_frames[1:]:
        dists = []
        for cand in remaining:
            dist = _aligned_distance(cur, cand)
            if dist is not None:
                dists.append((int(dist), cand))
        if not dists:
            return False
        best_dist = min(dist for dist, _cand in dists)
        if sum(1 for dist, _cand in dists if int(dist) == int(best_dist)) != 1:
            return False
        actual = _best_unvisited_neighbor(cur, remaining)
        if actual is not expected:
            return False
        remaining.remove(expected)
        cur = expected
    return True


def _sample_frame_colors(frame_palette: list[int], n_frames: int) -> list[int] | None:
    colors = [int(np.random.choice(np.array(frame_palette, dtype=int)))]
    for _ in range(1, int(n_frames)):
        prev = int(colors[-1])
        choices = [int(x) for x in frame_palette if int(x) != int(prev)]
        if not choices:
            return None
        colors.append(int(np.random.choice(np.array(choices, dtype=int))))
    if len(set(colors)) < min(3, int(len(colors))):
        return None
    return colors


def _sample_bridge_colors(bridge_palette: list[int], n_bridges: int) -> list[int] | None:
    for _ in range(80):
        seq = [int(np.random.choice(np.array(bridge_palette, dtype=int))) for _ in range(int(n_bridges))]
        if len(set(seq)) < min(2, int(n_bridges)):
            continue
        if int(n_bridges) >= 4 and max(Counter(seq).values()) < 2:
            continue
        return seq
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    usable = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(usable) < 5:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    gap = int(np.random.randint(1, 4))
    step = int(outer + gap)
    max_r = min(4, int(22 // step))
    max_c = min(4, int(26 // step))
    if max_r < 2 or max_c < 2:
        return None
    R = int(np.random.randint(2, int(max_r) + 1))
    C = int(np.random.randint(2, int(max_c) + 1))

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
        if len({int(r) for r, _c in p}) < 2 or len({int(c) for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    frame_k = int(np.random.randint(3, 5))
    bridge_k = int(np.random.randint(2, 4))
    if int(frame_k + bridge_k) > len(usable):
        return None
    palette = [int(x) for x in np.random.choice(np.array(usable, dtype=int), size=int(frame_k + bridge_k), replace=False)]
    frame_palette = palette[: int(frame_k)]
    bridge_palette = palette[int(frame_k) :]
    frame_colors = _sample_frame_colors(frame_palette, int(n_frames))
    if frame_colors is None:
        return None
    bridge_colors = _sample_bridge_colors(bridge_palette, int(n_frames - 1))
    if bridge_colors is None:
        return None

    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    r0s = [int(top_margin + step * r) for r in range(int(R))]
    c0s = [int(left_margin + step * c) for c in range(int(C))]

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames_in_path: list[dict] = []
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
        slot_to_frame[(int(rr), int(cc))] = frame
        frames_in_path.append(frame)

    if not _path_respects_nearest_rule(frames_in_path):
        return None

    max_r1 = max(int(f["bbox"][2]) for f in frames_in_path)
    max_c1 = max(int(f["bbox"][3]) for f in frames_in_path)
    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1

    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(bridge_colors)))]
    right_margin = int(np.random.randint(1, 4))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, ctrl_cols[-1] + 2))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    colors_by_slot: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames_in_path):
        rr, cc = map(int, frame["slot"])
        colors_by_slot[(int(rr), int(cc))] = int(frame_colors[int(idx)])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(frame_colors[int(idx)]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    for i, (a_slot, b_slot) in enumerate(zip(path, path[1:])):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        a_color = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        b_color = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        paint = int(bridge_colors[int(i)])
        if int(paint) in (int(a_color), int(b_color)):
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    for color, col in zip(bridge_colors, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)

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
