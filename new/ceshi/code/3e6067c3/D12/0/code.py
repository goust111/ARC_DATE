# Variant of 3e6067c3
# Dimension: D12 — 数量关系：源-目标基数 (meta)
# Change: 控制序列定义唯一路径，每段桥梁是1:1的方块对连接... → 一对多连接：星形桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


"""
Design analysis for the D12 variant

a) Concrete behavior trace
   - The visual vocabulary stays the same: equal-sized framed squares sit on a row/column lattice,
     and a control row below them lists inner colors left-to-right.
   - Original behavior on a local example:
     control = [2, 4, 7], current square has inner color 2, and color 4 appears once.
     The solver picks that single 4-square and paints one bridge colored 2.
   - Variant behavior on the same kind of scene:
     control = [2, 4, 7], current 2-square has one nearby 4-square on the trunk and one extra
     4-square leaf elsewhere on the same row/column lattice. The output paints both 2-colored
     bridges simultaneously. The nearer 4-square remains the continuation point for the later
     4 -> 7 trunk step, so the topology becomes a chain with a visible star burst at one step.
   - Flow impact:
     input construction must now allow repeated inner colors for one target step; solver matching
     must draw to all bridgeable same-color targets; output generation must validate non-overlap.

b) Complexity calibration
   - Only one control color is repeated per sample, and it repeats on 2-3 squares total.
     That is enough to make the one-to-many rule visible without turning the scene into a dense tree.
   - All other control colors remain unique. This keeps the trunk interpretable and prevents
     uncontrolled ambiguity from multiple repeated colors interacting at once.

c) Implicit assumptions audited from the original main()
   - `by[color]` originally could contain repeats, but the loop always collapsed them to one `bestf`.
     That was harmless because the original generator never used duplicates. In this variant that
     would silently erase the star branch, so the draw step must iterate over all bridgeable targets.
   - `cur = nxt` assumes the process continues from exactly one square. The variant keeps that
     assumption by designating the nearest same-color target as the trunk continuation while the
     other same-color squares are leaves only.
   - `draw_bridge()` assumes the two frames share an inner row-band or column-band. The generator
     therefore places every repeated-color leaf on a clear horizontal or vertical ray from the
     source so each star spoke is drawable by the unchanged bridge geometry.

d) Placement legality
   - The generator first builds the desired output and only then erases bridges to make the input.
   - A self-avoiding trunk path uses adjacent lattice cells. The extra star leaves are placed on
     longer empty rays from the star source, so no leaf occupies the trunk's immediate next cell.
   - Every bridge cell set is checked against frame pixels and previously reserved bridge pixels.
     Any overlap, empty bridge, or out-of-bounds placement rejects the sample.

e) Visual observability
   - The repeated color is directly visible because multiple framed squares share it while the
     control row still shows that color only once.
   - In the output, one source-colored bridge fans out to multiple equal-color targets in a single
     step. Comparing input/output pairs makes the new rule inferable from a few examples.

f) Visual cleanliness
   - Star leaves are limited in number and forced onto distinct clear rays, so bridges do not stack
     on top of each other or run through unrelated frames.
   - The control row format, frame border color, and bridge thickness are unchanged, preserving the
     original task's readable layout.

g) Color-change inferability
   - No hidden ordering uses numeric color ids. Colors only act as visible matching labels between
     framed squares and control pixels.
   - Because only one target color repeats and the bridge color remains the source color as before,
     the human-facing rule change is localized and learnable.
"""


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


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _score_candidate(src: dict, dst: dict) -> tuple[int, int]:
    sr, sc = _aligned(src, dst)
    al = 1 if (sr or sc) else 0
    r1, c1 = _center(src)
    r2, c2 = _center(dst)
    d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(al), -int(d)


def _best_candidate(src: dict, cands: list[dict]) -> dict | None:
    best = None
    bestf = None
    for f in cands:
        cand = _score_candidate(src, f)
        if best is None or cand > best:
            best = cand
            bestf = f
    return bestf


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    Variant rule:
    - Keep the original trunk traversal logic.
    - When the next control color appears on multiple bridgeable squares, paint bridges from the
      current square to all such same-color targets at once.
    - Continue the trunk from the nearest of those targets, preserving a single continuation path.
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

    frames = _extract_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
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
        best = None
        bestf = None
        for f in start_cands:
            anchors = nxt_cands if nxt_cands else [f]
            for g2 in anchors:
                cand = _score_candidate(f, g2)
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf
    assert not (cur is None)

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        for r, c in _bridge_cells(fa, fb):
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue

        all_cands = by[b]
        nxt = _best_candidate(cur, all_cands)
        if nxt is None:
            continue

        bridgeable = [f for f in all_cands if _bridge_cells(cur, f)]
        targets = bridgeable if bridgeable else [nxt]
        for target in targets:
            draw_bridge(cur, target, int(a))
        cur = nxt

    return out


_DIGITS = [int(i) for i in range(10)]


def _sample_path(R: int, C: int, n_nodes: int) -> list[tuple[int, int]] | None:
    for _ in range(500):
        start = (int(np.random.randint(0, R)), int(np.random.randint(0, C)))
        p = [start]
        vis = {start}
        ok = True
        for _step in range(int(n_nodes) - 1):
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
        return p
    return None


def _find_star_choices(path: list[tuple[int, int]], R: int, C: int) -> list[tuple[int, list[tuple[int, int]]]]:
    occupied = set(path)
    choices: list[tuple[int, list[tuple[int, int]]]] = []
    for idx in range(int(len(path)) - 1):
        src = path[idx]
        trunk_next = path[idx + 1]
        rays: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            blocked = False
            for dist in range(1, 4):
                rr = int(src[0] + dr * dist)
                cc = int(src[1] + dc * dist)
                if not in_bounds((int(R), int(C)), int(rr), int(cc)):
                    break
                pos = (int(rr), int(cc))
                if pos in occupied:
                    blocked = True
                if blocked:
                    break
                if dist >= 2 and pos != trunk_next:
                    rays.append(pos)
        if rays:
            choices.append((int(idx), rays))
    return choices


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 6:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(4, 7))
    C = int(np.random.randint(4, 7))
    n_path = int(np.random.randint(4, 7))
    if n_path > len(inner_palette):
        return None

    path = _sample_path(int(R), int(C), int(n_path))
    if path is None:
        return None

    star_choices = _find_star_choices(path, int(R), int(C))
    if not star_choices:
        return None

    star_step, star_rays = star_choices[int(np.random.randint(0, len(star_choices)))]
    extra_count = int(np.random.randint(1, min(2, len(star_rays)) + 1))
    pick_idx = np.random.choice(np.arange(len(star_rays), dtype=int), size=int(extra_count), replace=False)
    star_leaves = [star_rays[int(i)] for i in np.array(pick_idx).tolist()]

    used_slots = set(path) | set(star_leaves)

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
    for rr, cc in used_slots:
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
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    trunk_colors = np.random.choice(np.array(inner_palette, dtype=int), size=int(n_path), replace=False)
    colors_by_slot: dict[tuple[int, int], int] = {}
    for idx, slot in enumerate(path):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(trunk_colors[int(idx)])
    star_color = int(colors_by_slot[path[int(star_step + 1)]])
    for slot in star_leaves:
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(star_color)

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

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True

    def reserve_bridge(fa: dict, fb: dict, color: int) -> bool:
        nonlocal bridge_cells_all
        cells = _bridge_cells(fa, fb)
        if not cells:
            return False
        if cells & bridge_cells_all:
            return False
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return False
            if int(out[int(r), int(c)]) != int(bg):
                return False
        for r, c in cells:
            out[int(r), int(c)] = int(color)
        bridge_cells_all |= set(cells)
        return True

    for idx, (a_slot, b_slot) in enumerate(zip(path, path[1:])):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        if not reserve_bridge(fa, fb, int(paint)):
            ok = False
            break
        if int(idx) == int(star_step):
            for leaf in star_leaves:
                lf = slot_to_frame[(int(leaf[0]), int(leaf[1]))]
                if not reserve_bridge(fa, lf, int(paint)):
                    ok = False
                    break
        if not ok:
            break

    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if len(ctrl_colors) < 2 or int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

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
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
