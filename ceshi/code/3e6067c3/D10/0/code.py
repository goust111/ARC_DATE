# Variant of 3e6067c3
# Dimension: D10 — 对象属性：边框颜色的角色 (meta)
# Change: 所有方块共享同一种边框颜色（最常见的非背景色），边框仅作结构分隔用... → 多色边框标识分组

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

# Design analysis for the D10 variant
# a) Concrete behavior:
#    The scene still contains equal-sized framed squares plus a control row. The new visible cue is that
#    squares use at least two border colors. Squares with the same border color belong to the same group.
#    The control row still lists inner colors left-to-right, but when the solver moves from color ci to
#    color ci+1 it may only choose a ci+1 square whose border color matches the current square's border.
#    Small trace: suppose the current square is red-bordered with inner 2, the next control color is 4,
#    the true target is another red-bordered 4-square farther away in the same row, and a blue-bordered
#    4-square sits closer on the opposite side. The old logic would choose the closer blue square; the
#    new logic must first filter to red-bordered candidates, then choose the farther red one.
# b) Complexity calibration:
#    Only one new idea is introduced: border-color grouping. To make it visible, the generator creates
#    2 border colors and duplicates 2-3 inner colors across groups. That is enough repeated evidence to
#    force group-aware matching without exploding the number of cases.
# c) Hidden-assumption review of the original solver:
#    - "most common non-bg color is the border" no longer holds once borders are multi-colored.
#    - "label only one border-color mask" would miss every frame from other groups.
#    - candidate search by inner color alone would cross groups and pick wrong targets.
#    - start-frame disambiguation also needs border-aware lookahead, otherwise a decoy group can win.
#    The rewritten solver removes those assumptions by extracting valid frames from every non-bg color
#    and keeping each frame's border color as metadata used during matching.
# d) Placement legality:
#    All frames keep the same size/thickness. The generator builds an output first, reserves the bridge
#    corridors of the active path, and forbids distractor frames from occupying those corridors. That
#    prevents overlap, bridge collisions, and accidental blockage of the intended route.
# e) Visual observability:
#    Humans can see that some inner colors appear in several squares, but only the subset sharing one
#    border color gets connected. Across examples the untouched distractor group makes the border-color
#    role inferable from pixels alone.
# f) Visual tidiness:
#    Inner colors never reuse any border color, the control row remains isolated below the objects, and
#    bridges are the only added marks in the output. This keeps the scene readable.
# g) Color-rule inferability:
#    No numeric color identity is used for sorting or logic. The only color-dependent rule is visible:
#    same border color means same group. The control sequence still uses the inner fills exactly as in
#    the original task family.
# h) Generator-side anti-degeneration target:
#    At least one duplicated target color is placed in a different border group at a geometrically
#    better location for the old solver. That guarantees the D10 change affects actual decisions.
# i) New solver assumptions and how the generator guarantees them:
#    - Every consecutive control color has at least one aligned target in the current border group.
#    - The active path uses one border group throughout, so every bridge is legal.
#    - Duplicated colors never include the first control color, so start selection stays stable.
#    - Distractor frames never occupy reserved active bridge corridors.

_DIGITS = [int(i) for i in range(10)]


def _detect_frame_component(g: np.ndarray, pts: np.ndarray, border: int) -> dict | None:
    if int(pts.shape[0]) < 8:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    t = None
    inner_col = None
    for tt in range(1, int(min(h, w) // 2) + 1):
        inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
        if inner.size == 0:
            continue
        if bool((inner == int(border)).any()):
            continue
        inner_vals = set(map(int, inner.flatten()))
        if len(inner_vals) != 1:
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
        inner_col = int(next(iter(inner_vals)))
        break

    if t is None or inner_col is None:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "t": int(t),
        "inner": int(inner_col),
        "border": int(border),
        "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
    }


def _extract_frames(g: np.ndarray) -> tuple[list[dict], int]:
    bg = int(get_bg(g))
    colors = [int(c) for c in sorted(set(map(int, g.flatten()))) if int(c) != int(bg)]
    frames: list[dict] = []
    for border in colors:
        lab, n = label_mask(g == int(border), connectivity=4)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            frame = _detect_frame_component(g, pts, int(border))
            if frame is not None:
                frames.append(frame)
    return frames, int(bg)


def _control_sequence(g: np.ndarray, frames: list[dict], bg: int) -> list[int]:
    H, _W = g.shape
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(v) for v in g[int(ctrl_r)] if int(v) != int(bg)]
    assert len(ctrl) >= 2
    return ctrl


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _transition_score(src: dict, dst: dict) -> tuple[int, int]:
    sr, sc = _aligned(src, dst)
    al = 1 if (sr or sc) else 0
    (r1, c1) = _center(src)
    (r2, c2) = _center(dst)
    dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(al), -int(dist)


def _best_target(src: dict, target_color: int, by_inner: dict[int, list[dict]], *, use_group: bool) -> dict | None:
    cands = by_inner.get(int(target_color), [])
    best = None
    bestf = None
    for frame in cands:
        if bool(use_group) and int(frame["border"]) != int(src["border"]):
            continue
        cand = _transition_score(src, frame)
        if best is None or cand > best:
            best = cand
            bestf = frame
    return bestf


def _choose_start(ctrl: list[int], by_inner: dict[int, list[dict]], *, use_group: bool) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by_inner
    start_cands = by_inner[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

    next_color = int(ctrl[1])
    best = None
    bestf = None
    for frame in start_cands:
        nxt = _best_target(frame, int(next_color), by_inner, use_group=bool(use_group))
        if nxt is None:
            cand = (0, -1, -10**9)
        else:
            al, neg_dist = _transition_score(frame, nxt)
            cand = (1, int(al), int(neg_dist))
        if best is None or cand > best:
            best = cand
            bestf = frame
    assert bestf is not None
    return bestf


def _trace_route(grid: np.ndarray, *, use_group: bool) -> tuple[list[dict], list[int], list[dict], int]:
    g = np.array(grid, dtype=int)
    frames, bg = _extract_frames(g)
    assert frames
    ctrl = _control_sequence(g, frames, int(bg))
    by_inner: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_inner[int(frame["inner"])].append(frame)

    cur = _choose_start(ctrl, by_inner, use_group=bool(use_group))
    route = [cur]
    for idx in range(int(len(ctrl)) - 1):
        target_color = int(ctrl[idx + 1])
        nxt = _best_target(cur, int(target_color), by_inner, use_group=bool(use_group))
        assert nxt is not None
        route.append(nxt)
        cur = nxt
    return route, ctrl, frames, int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    route, ctrl, _frames, _bg = _trace_route(g, use_group=True)

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

    for idx in range(int(len(ctrl)) - 1):
        draw_bridge(route[int(idx)], route[int(idx + 1)], int(ctrl[int(idx)]))

    return out


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


def _slot_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r1, c1) = map(int, a)
    (r2, c2) = map(int, b)
    if int(r1) == int(r2):
        step = 1 if int(c2) > int(c1) else -1
        return [(int(r1), int(c)) for c in range(int(c1 + step), int(c2), int(step))]
    if int(c1) == int(c2):
        step = 1 if int(r2) > int(r1) else -1
        return [(int(r), int(c1)) for r in range(int(r1 + step), int(r2), int(step))]
    return []


def _slot_aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return bool(int(a[0]) == int(b[0]) or int(a[1]) == int(b[1]))


def _slot_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _build_rook_path(R: int, C: int, n_path: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    all_slots = [(int(r), int(c)) for r in range(int(R)) for c in range(int(C))]
    for _attempt in range(400):
        start = all_slots[int(np.random.randint(0, len(all_slots)))]
        path = [start]
        used = {start}
        reserved: set[tuple[int, int]] = set()
        long_hops = 0

        ok = True
        for _ in range(int(n_path) - 1):
            cur = path[-1]
            cands: list[tuple[int, int]] = []
            for nxt in all_slots:
                if nxt == cur or nxt in used or nxt in reserved:
                    continue
                if not _slot_aligned(cur, nxt):
                    continue
                between = _slot_between(cur, nxt)
                if not between:
                    continue
                if any(cell in used or cell in reserved for cell in between):
                    continue
                cands.append(nxt)
            if not cands:
                ok = False
                break
            order = list(np.random.permutation(len(cands)))
            chosen = None
            for idx in order:
                nxt = cands[int(idx)]
                span = _slot_dist(cur, nxt)
                if int(long_hops) == 0 and int(n_path - len(path)) <= 2 and int(span) == 1:
                    continue
                chosen = nxt
                break
            if chosen is None:
                chosen = cands[int(np.random.randint(0, len(cands)))]
            path.append(chosen)
            used.add(chosen)
            between = _slot_between(cur, chosen)
            reserved |= set(between)
            if int(_slot_dist(cur, chosen)) > 1:
                long_hops += 1

        if not ok:
            continue
        if len({int(r) for r, _c in path}) < 2 or len({int(c) for _r, c in path}) < 2:
            continue
        if int(long_hops) < 1:
            continue
        return path, reserved
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    borders = [int(x) for x in np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int), size=2, replace=False)]
    active_border, decoy_border = map(int, borders)
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(active_border), int(decoy_border))]

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(3, 6))
    C = int(np.random.randint(3, 6))
    n_path = int(np.random.randint(4, min(7, int(R * C - 2)) + 1))
    if int(n_path) > len(inner_palette):
        return None

    built = _build_rook_path(int(R), int(C), int(n_path))
    if built is None:
        return None
    path, reserved_slots = built
    path_set = set(path)

    all_slots = {(int(r), int(c)) for r in range(int(R)) for c in range(int(C))}
    free_slots = [slot for slot in all_slots if slot not in path_set and slot not in reserved_slots]
    if len(free_slots) < 2:
        return None

    critical_options: list[tuple[int, tuple[int, int], int]] = []
    for idx, (cur, nxt) in enumerate(zip(path, path[1:])):
        true_dist = int(_slot_dist(cur, nxt))
        if true_dist <= 1:
            continue
        for decoy_slot in free_slots:
            if not _slot_aligned(cur, decoy_slot):
                continue
            decoy_dist = int(_slot_dist(cur, decoy_slot))
            if 1 <= int(decoy_dist) < int(true_dist):
                critical_options.append((int(idx + 1), decoy_slot, int(decoy_dist)))

    if not critical_options:
        return None

    np.random.shuffle(critical_options)
    chosen_decoys: dict[int, tuple[int, int]] = {}
    used_decoy_slots: set[tuple[int, int]] = set()
    for target_pos, slot, _dist in critical_options:
        if target_pos in chosen_decoys or slot in used_decoy_slots:
            continue
        chosen_decoys[int(target_pos)] = (int(slot[0]), int(slot[1]))
        used_decoy_slots.add((int(slot[0]), int(slot[1])))
        if len(chosen_decoys) >= 2:
            break
    if not chosen_decoys:
        return None

    while len(chosen_decoys) < 2:
        extras = [slot for slot in free_slots if slot not in used_decoy_slots]
        if not extras:
            return None
        extra_slot = extras[int(np.random.randint(0, len(extras)))]
        for target_pos in range(1, int(len(path))):
            if target_pos not in chosen_decoys:
                chosen_decoys[int(target_pos)] = (int(extra_slot[0]), int(extra_slot[1]))
                used_decoy_slots.add((int(extra_slot[0]), int(extra_slot[1])))
                break

    active_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(n_path), replace=False)]
    colors_by_slot: dict[tuple[int, int], int] = {}
    for slot, color in zip(path, active_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)

    for target_idx, decoy_slot in chosen_decoys.items():
        dup_color = int(active_colors[int(target_idx)])
        if int(target_idx) == 0:
            return None
        colors_by_slot[(int(decoy_slot[0]), int(decoy_slot[1]))] = int(dup_color)

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

    frame_slots = list(path) + [slot for slot in used_decoy_slots]
    frame_slots = list(dict.fromkeys((int(r), int(c)) for r, c in frame_slots))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    for rr, cc in frame_slots:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        border = int(active_border if (int(rr), int(cc)) in path_set else decoy_border)
        slot_to_frame[(int(rr), int(cc))] = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "border": int(border),
            "inner": int(colors_by_slot[(int(rr), int(cc))]),
            "t": int(t),
        }

    max_r1 = max(int(f["bbox"][2]) for f in slot_to_frame.values())
    max_c1 = max(int(f["bbox"][3]) for f in slot_to_frame.values())
    ctrl_r = int(max_r1 + int(np.random.randint(2, 6)))
    H = int(ctrl_r + 2)
    W = int(max_c1 + 1 + int(np.random.randint(1, 5)))
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(path)))]
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None
    if int(ctrl_cols[-1]) >= int(W):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in slot_to_frame.values():
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(frame["border"]),
            inner_col=int(frame["inner"]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    for a_slot, b_slot in zip(path, path[1:]):
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
        paint = int(fa["inner"])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    for color, col in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    try:
        pred = main(inp)
        if pred.shape != out.shape or not bool(np.array_equal(pred, out)):
            return None
        route_grouped, ctrl, _frames, _bg = _trace_route(inp, use_group=True)
        route_plain, _ctrl2, _frames2, _bg2 = _trace_route(inp, use_group=False)
        if len(route_grouped) != len(route_plain):
            return None
        decisions_differ = False
        for g_frame, p_frame in zip(route_grouped[1:], route_plain[1:]):
            if int(g_frame["bbox"][0]) != int(p_frame["bbox"][0]) or int(g_frame["bbox"][1]) != int(p_frame["bbox"][1]):
                decisions_differ = True
                break
        if not decisions_differ:
            return None
        border_count = len({int(frame["border"]) for frame in _frames})
        if int(border_count) < 2:
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
