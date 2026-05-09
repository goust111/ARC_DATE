# Combo variant of 3e6067c3 (index 20)
# combo_spec: D10/0+D2/3
# Name: 多色边框分组+位置消歧

"""
Design analysis for the combo variant

a) Concrete behavior and trace
- The scene still consists of equal-sized framed squares and a control row below them.
- D10 changes the role of borders: several border colors appear, and equal border colors define a
  valid routing group. Bridges may only connect squares inside the current border-color group.
- D2 changes tie breaking: when the target inner color appears in multiple eligible squares, the
  solver must pick the first unvisited square in global position order (top-to-bottom, left-to-right),
  not the nearest one.
- The generator forces a visible three-way fork on the first move:
  start A(border X, color c0) has three c1 candidates:
  1) the true target T with border X, farther away, but earlier in global order;
  2) a same-group decoy S with border X, closer, but later in global order;
  3) a cross-group decoy U with border Y, earlier than T in global order.
  Old nearest-based logic picks S. Order-only logic without grouping picks U. The fused rule must
  first filter to border X, then pick T because T comes before S among same-group c1 squares.

b) Complexity calibration
- Only two border colors are used. One active group carries the intended path; one decoy group
  carries distractors. This keeps grouping readable instead of turning the scene into many groups.
- Path colors are unique, while 2 target colors are duplicated across decoys. That creates repeated
  evidence for both grouping and ordering without making the control row hard to parse.
- Geometry stays on a sparse lattice with one forced critical fork. The task remains close to the
  original family rather than becoming a generic search puzzle.

c) Hidden assumptions broken from the original solver
- "Most common non-bg color is the border" fails because borders are multi-colored.
- "Label one border-color mask to find all frames" fails because frames now live in multiple colors.
- "Choose by local alignment plus nearest distance" fails because ordering now overrides distance.
- "No visited state is needed" fails because D2 explicitly talks about the first unvisited match.
- In the fused setting these failures interact: even after filtering by group, there can still be
  multiple same-group matches, so grouping alone is insufficient; ordering must resolve that tie.

d) Placement legality
- All frames share the same size and thickness, and slot gaps are at least one background cell so
  border components never merge.
- The active route occupies a sparse rook-like lattice path, and bridges are painted only in empty
  corridors between aligned path squares.
- Decoy squares are placed on unused lattice slots chosen so they do not overlap the active squares
  or the reserved bridge corridors.

e) Visual observability
- The fused rule is visible from pixels alone: border color grouping is explicit, and position order
  uses only top/down and left/right, which are human-visible.
- The first move is intentionally diagnostic: the nearest same-group square is *not* chosen, and the
  earlier cross-group square is *also* not chosen. A human can infer both constraints from examples.

f) Visual tidiness
- Only framed squares, the isolated control row, and the added bridges are present.
- Inner colors never reuse the two border colors, so fills and borders remain easy to separate.
- The few distractors are kept off the bridge corridors, preventing clutter or ambiguity.

g) Inferability of changes
- No rule depends on numeric color ids. The only color semantics are visible border-group equality
  and the original "bridge color equals source inner color" rule.
- Ordering is global row-major order on square positions, again visible from geometry alone.

h) Anti-degeneration target
- The generator rejects any sample where the fused route matches either ablation:
  1) grouping + nearest-distance, or
  2) global-order without grouping.
  This guarantees that both dimensions materially affect the chosen route.

i) New solver assumptions and generator guarantees
- Assumption: the first control color has a stable earliest square. The generator makes the start
  color unique, so start selection is unambiguous.
- Assumption: every fused transition has an eligible same-group target. The active path uses one
  border group throughout, so each control step has a legal next square.
- Assumption: when ordering is applied inside a group, the chosen square is still bridgeable. The
  generator constructs the active path so each intended consecutive pair is row- or column-aligned.
- Assumption: decoys do not accidentally block intended bridges. The generator reserves active
  bridge corridors before placing decoy squares.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, label_mask


_DIGITS = [int(i) for i in range(10)]

_BASE_TEMPLATE = {
    "shape": (4, 4),
    "path": [(1, 1), (1, 3), (3, 3), (3, 1)],
    "same_group_decoy": (2, 1),
    "cross_group_decoy": (0, 1),
    "extra_cross_decoy": (0, 3),
}


def _frame_order_key(frame: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int(r0), int(c0), int(r1), int(c1)


def _transform_slot(slot: tuple[int, int], shape: tuple[int, int], mirror_h: bool) -> tuple[int, int]:
    r, c = map(int, slot)
    rows, cols = map(int, shape)
    if bool(mirror_h):
        c = int(cols - 1 - c)
    assert 0 <= int(r) < int(rows) and 0 <= int(c) < int(cols)
    return int(r), int(c)


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

    thickness = None
    inner_col = None
    for tt in range(1, int(min(h, w) // 2) + 1):
        inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
        if inner.size == 0:
            continue
        if bool((inner == int(border)).any()):
            continue
        vals = set(map(int, inner.flatten()))
        if len(vals) != 1:
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

        thickness = int(tt)
        inner_col = int(next(iter(vals)))
        break

    if thickness is None or inner_col is None:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "t": int(thickness),
        "inner": int(inner_col),
        "border": int(border),
        "inner_bbox": (int(r0 + thickness), int(c0 + thickness), int(r1 - thickness), int(c1 - thickness)),
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
    frames.sort(key=_frame_order_key)
    for idx, frame in enumerate(frames):
        frame["order"] = int(idx)
    return frames, int(bg)


def _control_sequence(g: np.ndarray, frames: list[dict], bg: int) -> list[int]:
    H, W = g.shape
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    return ctrl


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


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


def _paint_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    for r, c in _bridge_cells(fa, fb):
        out[int(r), int(c)] = int(color)


def _transition_score(src: dict, dst: dict) -> tuple[int, int, int]:
    sr, sc = _aligned(src, dst)
    aligned_score = 1 if (sr or sc) else 0
    r1, c1 = _center(src)
    r2, c2 = _center(dst)
    dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(aligned_score), -int(dist), -int(dst["order"])


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[int, list[dict]], list[int], int]:
    g = np.array(grid, dtype=int)
    frames, bg = _extract_frames(g)
    assert frames
    by_inner: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_inner[int(frame["inner"])].append(frame)
    for color in by_inner:
        by_inner[int(color)].sort(key=lambda frame: int(frame["order"]))
    ctrl = _control_sequence(g, frames, int(bg))
    return g, frames, by_inner, ctrl, int(bg)


def _trace_route(grid: np.ndarray, *, use_group: bool, use_order: bool) -> tuple[list[dict], list[int], list[dict], int]:
    g, frames, by_inner, ctrl, bg = _parse_scene(grid)

    start_color = int(ctrl[0])
    assert start_color in by_inner
    start_cands = by_inner[int(start_color)]
    cur = start_cands[0]
    visited = {int(cur["order"])}
    route = [cur]

    for idx in range(int(len(ctrl)) - 1):
        target_color = int(ctrl[idx + 1])
        cands = by_inner.get(int(target_color), [])
        if bool(use_group):
            cands = [frame for frame in cands if int(frame["border"]) == int(cur["border"])]
        nxt = None
        if bool(use_order):
            for frame in cands:
                if int(frame["order"]) not in visited:
                    nxt = frame
                    break
        else:
            best = None
            for frame in cands:
                cand = _transition_score(cur, frame)
                if best is None or cand > best[0]:
                    best = (cand, frame)
            if best is not None:
                nxt = best[1]
        if nxt is None:
            continue
        route.append(nxt)
        visited.add(int(nxt["order"]))
        cur = nxt

    return route, ctrl, frames, int(bg)


def _solve_with_rules(grid: np.ndarray, *, use_group: bool, use_order: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    route, ctrl, _frames, _bg = _trace_route(g, use_group=bool(use_group), use_order=bool(use_order))
    out = g.copy()
    for idx in range(min(len(route) - 1, len(ctrl) - 1)):
        _paint_bridge(out, route[int(idx)], route[int(idx + 1)], int(ctrl[int(idx)]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_rules(grid, use_group=True, use_order=True)


def _draw_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    t: int,
    border: int,
    inner_col: int,
) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _slot_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = map(int, a)
    r2, c2 = map(int, b)
    if int(r1) == int(r2):
        step = 1 if int(c2) > int(c1) else -1
        return [(int(r1), int(c)) for c in range(int(c1 + step), int(c2), int(step))]
    if int(c1) == int(c2):
        step = 1 if int(r2) > int(r1) else -1
        return [(int(r), int(c1)) for r in range(int(r1 + step), int(r2), int(step))]
    return []


def _route_signature(route: list[dict]) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(frame["bbox"][0]), int(frame["bbox"][1]), int(frame["border"])) for frame in route)


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border_choices = [int(d) for d in _DIGITS if int(d) != int(bg)]
    active_border, decoy_border = [int(x) for x in np.random.choice(np.array(border_choices, dtype=int), size=2, replace=False)]
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(active_border), int(decoy_border))]
    if len(inner_palette) < 4:
        return None

    t = 1
    inner_side = int(np.random.choice(np.array([1, 2], dtype=int)))
    outer = int(inner_side + 2 * t)

    mirror_h = bool(int(np.random.randint(0, 2)))
    shape = tuple(map(int, _BASE_TEMPLATE["shape"]))
    path = [_transform_slot(slot, shape, mirror_h) for slot in _BASE_TEMPLATE["path"]]
    same_group_decoy = _transform_slot(_BASE_TEMPLATE["same_group_decoy"], shape, mirror_h)
    cross_group_decoy = _transform_slot(_BASE_TEMPLATE["cross_group_decoy"], shape, mirror_h)
    extra_cross_decoy = _transform_slot(_BASE_TEMPLATE["extra_cross_decoy"], shape, mirror_h)

    rows, cols = shape
    row_gap = [int(np.random.randint(1, 3)) for _ in range(int(rows) - 1)]
    col_gap = [int(np.random.randint(1, 3)) for _ in range(int(cols) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    right_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(rows) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(cols) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    frame_specs: list[tuple[tuple[int, int], int, int]] = []
    path_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=len(path), replace=False)]
    for slot, color in zip(path, path_colors):
        frame_specs.append(((int(slot[0]), int(slot[1])), int(active_border), int(color)))
    frame_specs.append(((int(same_group_decoy[0]), int(same_group_decoy[1])), int(active_border), int(path_colors[1])))
    frame_specs.append(((int(cross_group_decoy[0]), int(cross_group_decoy[1])), int(decoy_border), int(path_colors[1])))
    frame_specs.append(((int(extra_cross_decoy[0]), int(extra_cross_decoy[1])), int(decoy_border), int(path_colors[2])))

    seen_slots: set[tuple[int, int]] = set()
    deduped_specs: list[tuple[tuple[int, int], int, int]] = []
    for slot, border, inner_col in frame_specs:
        if (int(slot[0]), int(slot[1])) in seen_slots:
            return None
        seen_slots.add((int(slot[0]), int(slot[1])))
        deduped_specs.append(((int(slot[0]), int(slot[1])), int(border), int(inner_col)))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for (rr, cc), border, inner_col in deduped_specs:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "border": int(border),
            "inner": int(inner_col),
            "t": int(t),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_r = int(max_r1 + int(np.random.randint(2, 5)))
    H = int(ctrl_r + 2)
    min_ctrl_w = int(2 * len(path) + 1)
    W = int(max(max_c1 + 1 + right_margin, min_ctrl_w + 1))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
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

    active_bridge_cells: set[tuple[int, int]] = set()
    for a_slot, b_slot in zip(path, path[1:]):
        between = _slot_between(a_slot, b_slot)
        if not between:
            return None
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
        if not cells or cells & active_bridge_cells:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(fa["inner"])
        active_bridge_cells |= set(cells)

    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(path)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(path_colors, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)

    inp = out.copy()
    for r, c in active_bridge_cells:
        inp[int(r), int(c)] = int(bg)

    try:
        pred = main(inp)
        route_fused, ctrl, _frames, _bg = _trace_route(inp, use_group=True, use_order=True)
        route_group_only, _ctrl2, _frames2, _bg2 = _trace_route(inp, use_group=True, use_order=False)
        route_order_only, _ctrl3, _frames3, _bg3 = _trace_route(inp, use_group=False, use_order=True)
        route_legacy, _ctrl4, _frames4, _bg4 = _trace_route(inp, use_group=False, use_order=False)
    except Exception:
        return None

    if pred.shape != out.shape or not bool(np.array_equal(pred, out)):
        return None
    if len(route_fused) != len(ctrl):
        return None
    if _route_signature(route_fused) == _route_signature(route_group_only):
        return None
    if _route_signature(route_fused) == _route_signature(route_order_only):
        return None
    if _route_signature(route_fused) == _route_signature(route_legacy):
        return None
    start_frame = route_fused[0]
    fused_first = route_fused[1]
    group_first = route_group_only[1]
    order_first = route_order_only[1]
    if int(fused_first["border"]) != int(start_frame["border"]):
        return None
    if int(group_first["border"]) != int(start_frame["border"]) or _route_signature(route_fused[:2]) == _route_signature(route_group_only[:2]):
        return None
    if int(order_first["border"]) == int(start_frame["border"]) or _route_signature(route_fused[:2]) == _route_signature(route_order_only[:2]):
        return None
    if bool(np.array_equal(_solve_with_rules(inp, use_group=True, use_order=False), out)):
        return None
    if bool(np.array_equal(_solve_with_rules(inp, use_group=False, use_order=True), out)):
        return None

    borders_seen = {int(frame["border"]) for frame in _frames}
    if len(borders_seen) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
