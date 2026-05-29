# Variant of 3e6067c3
# Dimension: D9 — 符号指令：控制像素的语义 (symbol_instructions)
# Change: 控制像素表示按序访问的方块颜色，隐含定义连接路径（每对相邻像素之间画桥）... → 控制像素编码桥梁颜色而非方块颜色

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


"""
Design analysis for the D9 variant

a) Concretization and behavior trace
   - Visual scene stays close to the source task: a background field contains several equal-sized framed
     squares arranged on a row/column lattice, plus one control row underneath.
   - Original behavior:
     the control row is a sequence of square colors. Reading it left-to-right gives a walk through the
     squares. Each bridge is painted with the color of the square we are leaving.
   - New D9 behavior:
     the control row no longer identifies squares. Instead, the squares are visited by geometry alone:
     start from the top-left-most framed square, then repeatedly move to the nearest unvisited square that
     shares the same row band or column band. The i-th control pixel directly gives the color of the i-th
     bridge. So the path comes from space; the colors come from the control row.
   - Small trace:
     suppose the path of squares in space is TL -> TR -> BR -> BL, with square colors [2, 3, 6, 9].
     Original control row might be [2, 3, 6, 9], producing bridge colors [2, 3, 6].
     Variant control row might be [7, 4, 7]. The visited squares are still TL -> TR -> BR -> BL,
     but the bridge colors become [7, 4, 7]. This is the exact locus of change: path selection is
     detached from control-row colors.

b) Complexity calibration
   - If bridge colors always match the source-square colors, the task visually collapses back toward the
     original. The generator therefore forces many bridge/source mismatches and often uses bridge colors
     that do not appear inside any square.
   - If too many independent colors are used, the task becomes noisy. The generator keeps bridge colors on
     a small repeated palette (usually 2-4 colors) so the human can see that the row is a paint recipe.
   - Path geometry must be unambiguous. The generator uses a simple lattice with uniform spacing and
     validates that every next step is the unique nearest aligned unvisited square under the solver rule.

c) Review of original assumptions that break under D9
   - `by[inner_color]` lookup assumed control pixels name destination squares. This is invalid now.
   - `start_color = ctrl[0]` assumed the first control pixel names the start square. Invalid now.
   - `assert cur["inner"] == a` assumed each bridge color equals the current square color. Invalid now.
   - `if b not in by: continue` assumed the next control color names a square color. Invalid now.
   - `draw_bridge(cur, nxt, a)` assumed the paint color is the current square color. Invalid now.
   - The frame detector, control-row detector, and bridge rasterization logic do not depend on the old
     semantics and can remain structurally unchanged.

d) Placement legality
   - Frames are still equal-sized, axis-aligned, and separated by at least one background cell.
   - The generator constructs bridges only between consecutive lattice neighbors on the intended path, then
     checks that bridge cells are non-empty, non-overlapping, within bounds, and drawn only over background.
   - The control row is placed strictly below all frames. Width is enlarged when needed so every control
     pixel fits at columns 1, 3, 5, ...

e) Visual observability
   - The human-visible clue is that control-row colors track bridge colors directly, not square colors.
     This is especially apparent because bridge colors often repeat while square colors do not follow that
     same sequence, and some bridge colors may not appear inside any square at all.
   - The start square is fixed geometrically as the top-left-most square. Across examples, this makes the
     nearest-neighbor chain inferable from the outputs.

f) Visual neatness
   - Only three visual layers coexist: framed squares, empty bridge corridors, and the control row.
     Bridges are drawn only in the gaps between frames, so no symbols overlap.
   - Uniform lattice spacing keeps the nearest-neighbor rule easy to see and avoids accidental crowding.

g) Color-rule inferability
   - Bridge colors come from a small repeated bridge palette.
   - Square inner colors come from a separate palette that may overlap only partially.
   - The generator enforces at least one substantial mismatch between "bridge color" and "current square
     color", so the solver cannot silently regress to the source semantics.

9) New implicit assumptions introduced by the variant
   - `main()` now assumes the path starts at the top-left-most frame. The generator always places the
     intended path start at lattice slot (0, 0), which is uniquely top-left-most.
   - `main()` assumes that from each current frame there exists a unique nearest aligned unvisited frame
     until the needed number of bridges has been drawn. The generator validates this exact property.
   - `main()` assumes the control row length does not exceed the number of drawable path steps. The
     generator sets `len(control) == len(path) - 1`.
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


def _find_control_row(g: np.ndarray, bg: int, frames: list[dict]) -> int:
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            return int(r)
    raise AssertionError("control row not found")


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

    ctrl_r = _find_control_row(g, int(bg), frames)
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 1

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


def _sample_frame_colors(inner_palette: list[int], n_frames: int) -> list[int] | None:
    if len(inner_palette) < 4:
        return None
    max_k = min(5, len(inner_palette) - 1)
    if max_k < 3:
        return None
    frame_k = int(np.random.randint(3, int(max_k) + 1))
    frame_palette = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=frame_k, replace=False)]
    colors = [int(np.random.choice(np.array(frame_palette, dtype=int)))]
    for _ in range(1, int(n_frames)):
        prev = int(colors[-1])
        choices = [int(x) for x in frame_palette if int(x) != int(prev)]
        if not choices:
            return None
        colors.append(int(np.random.choice(np.array(choices, dtype=int))))
    if len(set(colors)) < 3:
        return None
    return colors


def _sample_bridge_colors(inner_palette: list[int], frame_colors: list[int]) -> list[int] | None:
    n_bridges = int(len(frame_colors) - 1)
    outside = [int(c) for c in inner_palette if int(c) not in set(map(int, frame_colors))]
    available = list(map(int, inner_palette))
    if len(available) < 2:
        return None

    for _ in range(80):
        base: list[int] = []
        if outside:
            base.append(int(np.random.choice(np.array(outside, dtype=int))))
        need = int(np.random.randint(2, int(min(4, len(available))) + 1))
        while len(base) < need:
            pick = int(np.random.choice(np.array(available, dtype=int)))
            if int(pick) not in base:
                base.append(int(pick))
        seq = [int(np.random.choice(np.array(base, dtype=int))) for _ in range(int(n_bridges))]
        if len(set(seq)) < 2:
            continue
        mismatch = sum(int(seq[i]) != int(frame_colors[i]) for i in range(int(n_bridges)))
        if mismatch < max(1, int(n_bridges) // 2):
            continue
        if outside and not any(int(x) in set(outside) for x in seq):
            continue
        return seq
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
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

    frame_colors = _sample_frame_colors(inner_palette, int(n_frames))
    if frame_colors is None:
        return None
    bridge_colors = _sample_bridge_colors(inner_palette, frame_colors)
    if bridge_colors is None:
        return None

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
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(bridge_colors[int(i)])
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
