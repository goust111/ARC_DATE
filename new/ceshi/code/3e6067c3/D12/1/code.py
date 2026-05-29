# Variant of 3e6067c3
# Dimension: D12 — 数量关系：源-目标基数 (meta)
# Change: 控制序列定义唯一路径，每段桥梁是1:1的方块对连接... → 多对一汇聚

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from arc_utils import get_bg

"""
Design analysis for the D12 variant
===================================

a) Concrete behavior and trace
------------------------------
Original task:
- The bottom control region is a single row of colored pixels.
- Reading that row left-to-right yields a unique walk c0 -> c1 -> c2 -> ...
- The solver keeps one moving cursor `cur`, and each consecutive pair draws one 1:1 bridge.

Variant:
- The control region is now two rows tall.
- Each group is encoded as a U-shaped bracket made from the border color:
  top row:    border, source colors inside, border, gap, target color
  bottom row: a solid border underline from the left border post to the right border post
- Example group: `( A B C ) T` means "draw bridges A->T, B->T, C->T".
- Thus one target square receives several bridges of different colors in parallel.

Concrete trace:
- Old logic on a control row `[2, 4, 6]` draws `2->4`, then `4->6`.
- New logic on a control bracket containing `[2, 4]` and target `6` draws `2->6` and `4->6`.
- The output differs structurally: instead of a chain, several bridge segments terminate at the same target frame.

b) Complexity calibration
-------------------------
- The visual change must be obvious, so every generated sample has at least one group with >= 2 sources.
- To avoid an underconstrained "anything can connect to anything" puzzle, each source and target color is globally unique.
- The bracket syntax is low-entropy: only one marker shape exists, built from the already-salient border color.
- Source multiplicity is limited to 2..4 and the number of groups to 1..2, which is enough to make D12 visible without clutter.

c) Hidden assumptions in the original main() that break here
------------------------------------------------------------
- `ctrl = [non-bg pixels in one row]`: no longer valid because border-colored bracket markers would be misread as path colors.
- `cur = start frame` plus `for consecutive pairs`: invalid because there is no single cursor anymore.
- `assert cur["inner"] == a`: invalid because multiple independent source frames can be active inside one group.
- Start-frame disambiguation using the next color is no longer the right primitive; now we resolve source frames relative to a known target.

d) Placement legality
---------------------
- Every convergence group uses one target and sources placed on distinct cardinal sides (L/R/U/D) of that target.
- This guarantees each bridge is axis-aligned and non-overlapping with other bridges in the same group.
- Groups are packed using their full occupied bbox (frames plus bridge corridors), so different groups do not intersect.
- The control area is kept strictly below the frame region, preserving the original coarse layout.

e) Visual observability
-----------------------
- A human can see that several colors are literally enclosed by one bracket and one target color sits outside that bracket.
- In the output, those same bracketed colors all emit bridges into that single target frame.
- This is inferable from 2-3 examples because the marker syntax is repeated and the output always mirrors it.

f) Visual cleanliness
---------------------
- Only two control rows are used, so the control syntax stays compact.
- Bridge overlap is prevented by construction.
- Group count is capped, and groups are separated by padding so the scene remains legible.

g) Color-change inferability
----------------------------
- No hidden color arithmetic is introduced.
- A bridge keeps the color of its source square, exactly like the original task.
- The new rule is therefore about grouping/counting, not about opaque recoloring.

Post-change solver assumptions to audit
---------------------------------------
- The control area contains exactly two informative rows below the lowest frame.
- Each bracket group has at least two source colors and exactly one target color two columns to the right of the bracket.
- Every control color corresponds to an existing frame.
- For each source-target pair, at least one candidate source frame shares a row band or column band with the target frame.
- The generator enforces all of these constraints and validates them by re-running `main()` on the produced input.
"""


def _detect_frames(g: np.ndarray, border: int) -> list[dict]:
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    frames: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) != int(border):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == int(border):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            if len(pts) < 8:
                continue
            arr = np.array(pts, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
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
                cols = set(map(int, inner.flatten()))
                if len(cols) != 1:
                    continue
                top_ok = bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all())
                bot_ok = bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all())
                left_ok = bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all())
                right_ok = bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all())
                if top_ok and bot_ok and left_ok and right_ok:
                    t = int(tt)
                    inner_col = int(next(iter(cols)))
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


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    for r, c in _bridge_cells(fa, fb):
        out[int(r), int(c)] = int(color)


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _pick_best_target(target_cands: list[dict], source_colors: list[int], by_color: dict[int, list[dict]]) -> dict:
    if len(target_cands) == 1:
        return target_cands[0]
    best_score = None
    best_frame = None
    for cand in target_cands:
        aligned_count = 0
        total_dist = 0
        ct = _center(cand)
        for src_col in source_colors:
            src_cands = by_color.get(int(src_col), [])
            local_best = None
            for src in src_cands:
                sr, sc = _aligned(src, cand)
                al = 1 if (sr or sc) else 0
                cs = _center(src)
                dist = abs(int(cs[0]) - int(ct[0])) + abs(int(cs[1]) - int(ct[1]))
                key = (int(al), -int(dist))
                if local_best is None or key > local_best:
                    local_best = key
            if local_best is not None:
                aligned_count += int(local_best[0])
                total_dist += int(-local_best[1])
        score = (int(aligned_count), -int(total_dist))
        if best_score is None or score > best_score:
            best_score = score
            best_frame = cand
    assert best_frame is not None
    return best_frame


def _pick_best_source(source_cands: list[dict], target: dict) -> dict:
    best_score = None
    best_frame = None
    ct = _center(target)
    for src in source_cands:
        sr, sc = _aligned(src, target)
        al = 1 if (sr or sc) else 0
        cs = _center(src)
        dist = abs(int(cs[0]) - int(ct[0])) + abs(int(cs[1]) - int(ct[1]))
        score = (int(al), -int(dist))
        if best_score is None or score > best_score:
            best_score = score
            best_frame = src
    assert best_frame is not None
    return best_frame


def _parse_control_groups(g: np.ndarray, bg: int, border: int, max_frame_r: int) -> list[dict]:
    H, W = g.shape
    rows = [int(r) for r in range(int(max_frame_r) + 1, int(H)) if bool((g[int(r)] != int(bg)).any())]
    assert len(rows) >= 2
    ctrl_bottom = int(rows[-1])
    ctrl_top = int(ctrl_bottom - 1)
    assert ctrl_top in rows

    groups: list[dict] = []
    c = 0
    while c < W:
        if int(g[int(ctrl_bottom), int(c)]) != int(border):
            c += 1
            continue
        l = int(c)
        while c + 1 < W and int(g[int(ctrl_bottom), int(c + 1)]) == int(border):
            c += 1
        r = int(c)
        if int(r - l + 1) >= 5 and int(g[int(ctrl_top), int(l)]) == int(border) and int(g[int(ctrl_top), int(r)]) == int(border):
            sources = [
                int(g[int(ctrl_top), int(cc)])
                for cc in range(int(l + 1), int(r))
                if int(g[int(ctrl_top), int(cc)]) not in (int(bg), int(border))
            ]
            target_c = int(r + 2)
            assert target_c < W
            target = int(g[int(ctrl_top), int(target_c)])
            assert target not in (int(bg), int(border))
            assert len(sources) >= 2
            groups.append({"sources": sources, "target": int(target)})
        c += 1
    assert groups
    return groups


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

    frames = _detect_frames(g, int(border))
    assert frames
    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    groups = _parse_control_groups(g, int(bg), int(border), int(max_frame_r))

    out = g.copy()
    for group in groups:
        target_color = int(group["target"])
        assert target_color in by_color
        target = _pick_best_target(by_color[int(target_color)], list(group["sources"]), by_color)
        for source_color in group["sources"]:
            assert int(source_color) in by_color
            source = _pick_best_source(by_color[int(source_color)], target)
            sr, sc = _aligned(source, target)
            assert sr or sc
            _draw_bridge(out, source, target, int(source_color))
    return out


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _shift_frame(frame: dict, dr: int, dc: int) -> dict:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    ir0, ic0, ir1, ic1 = map(int, frame["inner_bbox"])
    return {
        **frame,
        "bbox": (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)),
        "inner_bbox": (int(ir0 + dr), int(ic0 + dc), int(ir1 + dr), int(ic1 + dc)),
    }


def _build_group_layout(outer: int, t: int, directions: list[str], order: list[str]) -> dict:
    gaps = {d: int(np.random.randint(1, 4)) for d in directions}
    boxes: dict[str, tuple[int, int, int, int]] = {"T": (0, 0, int(outer - 1), int(outer - 1))}
    for d in directions:
        gap = int(gaps[d])
        if d == "L":
            boxes[d] = (0, int(-outer - gap), int(outer - 1), int(-gap - 1))
        elif d == "R":
            boxes[d] = (0, int(outer + gap), int(outer - 1), int(2 * outer + gap - 1))
        elif d == "U":
            boxes[d] = (int(-outer - gap), 0, int(-gap - 1), int(outer - 1))
        elif d == "D":
            boxes[d] = (int(outer + gap), 0, int(2 * outer + gap - 1), int(outer - 1))

    min_r = min(int(b[0]) for b in boxes.values())
    min_c = min(int(b[1]) for b in boxes.values())
    max_r = max(int(b[2]) for b in boxes.values())
    max_c = max(int(b[3]) for b in boxes.values())
    shift_r = int(-min_r)
    shift_c = int(-min_c)

    frames: dict[str, dict] = {}
    for role, (r0, c0, r1, c1) in boxes.items():
        rr0 = int(r0 + shift_r)
        cc0 = int(c0 + shift_c)
        rr1 = int(r1 + shift_r)
        cc1 = int(c1 + shift_c)
        frames[role] = {
            "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
            "inner_bbox": (int(rr0 + t), int(cc0 + t), int(rr1 - t), int(cc1 - t)),
        }

    control_order = [d for d in order if d in directions]
    return {
        "frames": frames,
        "directions": list(directions),
        "source_roles": control_order,
        "H": int(max_r - min_r + 1),
        "W": int(max_c - min_c + 1),
    }


def _pack_group_layouts(layouts: list[dict], pad: int) -> tuple[list[tuple[int, int]], int, int] | None:
    if len(layouts) == 1:
        return [(0, 0)], int(layouts[0]["H"]), int(layouts[0]["W"])

    horizontal = []
    cur_c = 0
    max_h = 0
    for layout in layouts:
        horizontal.append((0, int(cur_c)))
        cur_c += int(layout["W"]) + int(pad)
        max_h = max(int(max_h), int(layout["H"]))
    scene_h_h = int(max_h)
    scene_w_h = int(cur_c - pad)

    vertical = []
    cur_r = 0
    max_w = 0
    for layout in layouts:
        vertical.append((int(cur_r), 0))
        cur_r += int(layout["H"]) + int(pad)
        max_w = max(int(max_w), int(layout["W"]))
    scene_h_v = int(cur_r - pad)
    scene_w_v = int(max_w)

    fits_h = scene_h_h <= 24 and scene_w_h <= 26
    fits_v = scene_h_v <= 22 and scene_w_v <= 26
    if fits_h and fits_v:
        if bool(np.random.randint(0, 2)):
            return horizontal, scene_h_h, scene_w_h
        return vertical, scene_h_v, scene_w_v
    if fits_h:
        return horizontal, scene_h_h, scene_w_h
    if fits_v:
        return vertical, scene_h_v, scene_w_v
    return None


def _encode_control(out: np.ndarray, groups: list[dict], ctrl_top: int, ctrl_bottom: int, border: int, bg: int) -> None:
    c = 1
    W = out.shape[1]
    for group in groups:
        k = len(group["source_colors"])
        l = int(c)
        r = int(l + 2 * k)
        target_c = int(r + 2)
        assert target_c < W
        out[int(ctrl_top), int(l)] = int(border)
        out[int(ctrl_top), int(r)] = int(border)
        out[int(ctrl_bottom), int(l) : int(r) + 1] = int(border)
        for i, col in enumerate(group["source_colors"]):
            out[int(ctrl_top), int(l + 1 + 2 * i)] = int(col)
        out[int(ctrl_top), int(target_c)] = int(group["target_color"])
        for fill_c in range(int(l + 1), int(r)):
            if int(out[int(ctrl_top), int(fill_c)]) == int(bg):
                out[int(ctrl_top), int(fill_c)] = int(bg)
        c = int(target_c + 3)


_DIGITS = [int(i) for i in range(10)]
_DIRECTION_ORDER = ["L", "U", "D", "R"]
_GROUP_CHOICES = [
    [["L", "R", "U"]],
    [["L", "R", "D"]],
    [["U", "D", "L"]],
    [["U", "D", "R"]],
    [["L", "R", "U", "D"]],
    [["L", "R"], ["U", "D"]],
    [["L", "U"], ["R", "D"]],
    [["L", "R", "U"], ["U", "D"]],
    [["U", "D", "L"], ["L", "R"]],
]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 5:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 6):
        return None

    pattern_combo = _GROUP_CHOICES[int(np.random.randint(0, len(_GROUP_CHOICES)))]
    total_frames = sum(len(group_dirs) + 1 for group_dirs in pattern_combo)
    if total_frames > len(inner_palette):
        return None

    layouts = [_build_group_layout(int(outer), int(t), list(group_dirs), _DIRECTION_ORDER) for group_dirs in pattern_combo]
    packed = _pack_group_layouts(layouts, pad=int(np.random.randint(2, 5)))
    if packed is None:
        return None
    placements, scene_h, scene_w = packed

    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 4))
    ctrl_gap = int(np.random.randint(2, 5))
    bottom_margin = 1

    ctrl_width = 2
    for group_dirs in pattern_combo:
        ctrl_width += int(2 * len(group_dirs) + 3)
    ctrl_width += int(3 * (len(pattern_combo) - 1))

    H = int(top_margin + scene_h + ctrl_gap + 2 + bottom_margin)
    W = int(max(left_margin + scene_w + right_margin, ctrl_width + right_margin))
    if not (9 <= H <= 30 and 9 <= W <= 30):
        return None

    colors = list(np.random.permutation(np.array(inner_palette, dtype=int))[:total_frames])
    color_idx = 0
    groups: list[dict] = []
    all_frames: list[dict] = []
    for layout, (dr, dc) in zip(layouts, placements):
        target_color = int(colors[color_idx])
        color_idx += 1
        target_frame = _shift_frame(layout["frames"]["T"], int(top_margin + dr), int(left_margin + dc))
        group = {
            "target_color": int(target_color),
            "target_frame": target_frame,
            "source_colors": [],
            "source_frames": [],
        }
        all_frames.append({**target_frame, "inner": int(target_color)})
        for role in layout["source_roles"]:
            source_color = int(colors[color_idx])
            color_idx += 1
            source_frame = _shift_frame(layout["frames"][role], int(top_margin + dr), int(left_margin + dc))
            group["source_colors"].append(int(source_color))
            group["source_frames"].append(source_frame)
            all_frames.append({**source_frame, "inner": int(source_color)})
        groups.append(group)

    max_frame_r = max(int(frame["bbox"][2]) for frame in all_frames)
    ctrl_top = int(max_frame_r + ctrl_gap)
    ctrl_bottom = int(ctrl_top + 1)
    if ctrl_bottom >= H:
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in all_frames:
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(frame["inner"]))

    bridge_cells_all: set[tuple[int, int]] = set()
    for group in groups:
        target_frame = group["target_frame"]
        for source_color, source_frame in zip(group["source_colors"], group["source_frames"]):
            cells = _bridge_cells(source_frame, target_frame)
            if not cells:
                return None
            if cells & bridge_cells_all:
                return None
            for r, c in cells:
                if not (0 <= int(r) < H and 0 <= int(c) < W):
                    return None
                if int(out[int(r), int(c)]) != int(bg):
                    return None
            for r, c in cells:
                out[int(r), int(c)] = int(source_color)
            bridge_cells_all |= set(cells)

    _encode_control(out, groups, int(ctrl_top), int(ctrl_bottom), int(border), int(bg))

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
