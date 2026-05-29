# Variant of 3e6067c3
# Dimension: D6 — 轨迹：源对象保留策略 (trajectory)
# Change: 保留控制像素不变 (keep src)，方块和控制行在输出中都保持原样... → 已访问方块变色
#
# Design analysis for this variant
# a) Concrete behavior:
#    The original task keeps every framed square unchanged and only paints bridges in the gaps.
#    This variant keeps the same control row and the same bridge colors, but every square that
#    participates in the traversed control path becomes "visited": its outer frame is recolored
#    to the square's inner color, so the whole square turns into a solid block. Unvisited squares
#    remain framed. Example: if a red-inside square with blue border is the start square and is
#    connected to a green-inside square, the gap is still painted red, but both endpoint squares
#    become solid red / solid green blocks. The processing change happens after frame matching and
#    path selection; the solver must keep using the input-time border structure for detection, and
#    only alter the output rendering.
# b) Complexity calibration:
#    If every square in every generated scene were on the path, "visited squares recolor" would
#    collapse visually into "all squares lose their borders", which is weaker and less obviously
#    trajectory-related. To keep the D6 effect observable, the generator now includes at least one
#    extra framed square that is not mentioned in the control row. This gives direct contrast
#    between visited solid blocks and untouched framed blocks without changing the bridge logic.
# c) Hidden-assumption audit of the original main():
#    - Border-color detection assumes the input still has one dominant shared frame color. This
#      remains true because recoloring happens only in the output, not the input.
#    - Frame detection via connected components assumes every square is framed in the input. Still
#      true for generated inputs.
#    - The `by[inner_color]` map assumes inner colors seen in the control row can locate candidate
#      squares. Still true; the generator uses unique path colors and keeps decoy colors out of the
#      control row so the visited-state change does not interfere with matching.
#    - `draw_bridge()` writes only in the gap between aligned squares. That logic remains correct.
#    - Any new visited-state painting must not feed back into frame detection. Therefore main()
#      detects all frames from the input copy first, stores their geometry, and only then paints
#      visited squares on the output copy.
# d) Placement legality:
#    Frames still live on a lattice with at least one background cell between neighboring slots.
#    The path uses 4-neighbor-adjacent lattice steps, so each bridge occupies exactly the open gap
#    between two neighboring framed squares. Extra non-path squares are placed only on unused
#    lattice slots, so they never overlap a path square or its bridge gap.
# e) Visual observability:
#    Across a few examples a human can see three stable layers: unchanged control row, colored
#    bridges, and a subset of squares whose borders disappear because they were visited. This is
#    visibly different from the original task, where all squares keep the frame border.
# f) Visual cleanliness:
#    The visited mark is localized to a square's own bbox, so there is no extra clutter outside
#    the existing objects. Unvisited squares retain the original border signal, avoiding ambiguity.
# g) Color-change inferability:
#    The recolor rule is structural, not palette-dependent: the border becomes the same color as
#    the square's own interior. This can be inferred from examples because the whole square becomes
#    a solid version of itself, and the control row is unchanged.
# Source-task note:
#    The prompt asks to read the solver of source task 8b9c3697 first. A direct code search in the
#    current workspace found the official JSON examples but no readable 8b9c3697 solver file, so
#    this implementation only uses the visible "continuous trail leaves a persistent mark" idea as
#    the baseline and does not reuse any other variant code.
# Post-change solver assumptions that the generator guarantees:
#    - At least one successful bridge exists, so at least two squares become visited.
#    - Every control color corresponds to exactly one path square in generated inputs.
#    - At least one non-path square exists, so D6's visited/unvisited contrast is usually present.

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


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
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
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


def _choose_candidate(cur: dict, cands: list[dict]) -> dict | None:
    best = None
    bestf = None
    for frame in cands:
        sr, sc = _aligned(cur, frame)
        al = 1 if (sr or sc) else 0
        (r1_, c1_) = _center(cur)
        (r2_, c2_) = _center(frame)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = frame
    return bestf


def _mark_visited(out: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(frame["inner"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Variant change:
    - The bridge routing and control row stay unchanged.
    - Every square that is actually visited by the traversed path is recolored into a solid block
      of its own inner color, leaving a visible visitation trail on the nodes.
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
    assert not (border is None)

    frames = _find_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

    start_color = int(ctrl[0])
    assert not (start_color not in by)
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
        if nxt_cands:
            best = None
            bestf = None
            for frame in start_cands:
                for g2 in nxt_cands:
                    sr, sc = _aligned(frame, g2)
                    al = 1 if (sr or sc) else 0
                    (r1_, c1_) = _center(frame)
                    (r2_, c2_) = _center(g2)
                    d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                    cand = (int(al), -int(d))
                    if best is None or cand > best:
                        best = cand
                        bestf = frame
            cur = bestf
        else:
            cur = _choose_candidate(start_cands[0], start_cands) or start_cands[0]

    assert not (cur is None)
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

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        nxt = _choose_candidate(cur, by[b])
        if nxt is None:
            continue
        draw_bridge(cur, nxt, int(a))
        _mark_visited(out, cur)
        _mark_visited(out, nxt)
        cur = nxt

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


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 5:
        return None

    max_path = int(min(7, max_frames - 1, len(inner_palette) - 1))
    if max_path < 4:
        return None
    n_path = int(np.random.randint(4, max_path + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_path) - 1):
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

    spare_slots = [(int(r), int(c)) for r in range(int(R)) for c in range(int(C)) if (int(r), int(c)) not in set(path)]
    spare_colors = int(len(inner_palette) - n_path)
    max_decoys = int(min(3, len(spare_slots), spare_colors))
    if max_decoys < 1:
        return None
    n_decoys = int(np.random.randint(1, max_decoys + 1))
    decoy_idx = np.random.choice(len(spare_slots), size=int(n_decoys), replace=False)
    decoy_slots = [spare_slots[int(i)] for i in np.array(decoy_idx).tolist()]

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
    all_slots = list(path) + list(decoy_slots)
    frames: list[dict] = []
    for rr, cc in all_slots:
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

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, 2 * n_path + 1))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(n_path), replace=False).tolist()]
    colors_by_slot: dict[tuple[int, int], int] = {}
    for slot, color in zip(path, path_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)

    remaining_colors = [int(c) for c in inner_palette if int(c) not in set(path_colors)]
    if len(remaining_colors) < len(decoy_slots):
        return None
    decoy_colors = [int(x) for x in np.random.choice(np.array(remaining_colors, dtype=int), size=int(len(decoy_slots)), replace=False).tolist()]
    for slot, color in zip(decoy_slots, decoy_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)

    inp = np.full((int(H), int(W)), int(bg), dtype=int)
    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        inner_col = int(colors_by_slot[(int(rr), int(cc))])
        _draw_frame(inp, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells:
            ok = False
            break
        if cells & bridge_cells_all:
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

    for slot in path:
        frame = slot_to_frame[(int(slot[0]), int(slot[1]))]
        r0, c0, r1, c1 = map(int, frame["bbox"])
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(colors_by_slot[(int(slot[0]), int(slot[1]))])

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(ctrl_colors, ctrl_cols):
        inp[int(ctrl_r), int(col)] = int(color)
        out[int(ctrl_r), int(col)] = int(color)

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
