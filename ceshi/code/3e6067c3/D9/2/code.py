# Variant of 3e6067c3
# Dimension: D9 — 符号指令：控制像素的语义 (symbol_instructions)
# Change: 控制像素表示按序访问的方块颜色，隐含定义连接路径（每对相邻像素之间画桥）... → 控制像素同时编码方向

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


"""
Design analysis for the D9 variant

a) Concrete behavior and trace
- The framed-square geometry stays the same: a background field, many equal-sized border-color frames,
  and a single control row below them.
- The visible change is only in the control row. Instead of a pure color sequence, the non-background
  tokens now alternate as:
  [start square color, direction color, next square color, direction color, ..., final square color]
- Direction colors are fixed and globally meaningful:
  red(2)=right, blue(1)=down, green(3)=left, yellow(4)=up.
- Example trace:
  old logic: [6, 5, 7] means "start at inner color 6, connect to a 5-square, then to a 7-square".
  new logic: [6, 2, 5, 1, 7] means "start at a 6-square, go right to a 5-square, then go down to a 7-square".
  The bridge colors are still the source-square colors, so the first bridge is colored 6 and the second 5.

b) Complexity calibration
- To make direction decoding necessary, generator uses a small frame-color palette and allows repeated
  inner colors. It also places extra non-path frames so a requested target color can appear in multiple
  directions relative to the current square.
- Direction colors are reserved and never used as frame inner colors. This keeps the visual code legible:
  control-row tokens in {1,2,3,4} always mean directions, while frame fills come from a disjoint palette.

c) Old assumptions that break
- Original `ctrl` parsing assumed every non-bg control token is a frame color. That fails here because
  every odd token is now a direction symbol.
- Original target selection only used "same row/col band + nearest". That is insufficient once repeated
  frame colors are allowed; we must first restrict by the decoded direction and only then choose nearest.
- Original start disambiguation looked ahead only one color. Here the correct start frame may also depend
  on the first direction token.

d) Placement legality
- All framed squares remain on a rectangular lattice with at least one background cell between adjacent
  frames, so bridges can occupy only the gap corridors and never overwrite borders.
- The path still moves between lattice-adjacent squares. This preserves the original bridge shape and
  guarantees the desired target is the nearest frame in the instructed direction.
- Extra decoy frames are real frames, not marks on top of existing frames, so no overlap is introduced.

e) Visual observability
- Humans can see the new rule because the control row visibly alternates between "ordinary frame colors"
  and a repeated four-color direction alphabet.
- Repeated frame colors plus extra decoy squares make those direction symbols functional: in some samples
  the same next color exists in multiple directions, and only the direction token explains the chosen bridge.

f) Visual cleanliness
- The main region still contains only framed squares and bridges; no extra glyphs are placed inside the
  object area. The control row remains a single sparse row below the scene, so the image stays uncluttered.

g) Color inferability
- Direction semantics use only four fixed colors, repeated across samples.
- Frame fills come from a separate, smaller palette with deliberate repetitions, giving enough evidence
  that repeated object colors are normal while the control-row direction colors have a different role.

Main() assumptions introduced by the variant and generator guarantees
- Control tokens must form an odd-length alternating sequence color/dir/color/.../color.
  Generator always emits that shape.
- Every direction token must be one of {1,2,3,4}. Generator reserves those colors for directions only.
- For each step, there is at least one frame of the requested next color in the requested direction and
  row/column band. Generator validates this by running `main()` on each candidate input.
- The start color is generated as globally unique, but `main()` still supports repeated start colors by
  picking the candidate that admits the first directed move.
"""


_DIGITS = [int(i) for i in range(10)]
_DIR_COLOR_TO_NAME = {2: "right", 1: "down", 3: "left", 4: "up"}
_DIR_NAME_TO_DELTA = {"right": (0, 1), "down": (1, 0), "left": (0, -1), "up": (-1, 0)}
_RESERVED_DIR_COLORS = set(_DIR_COLOR_TO_NAME)
_NONDIR_POOL = [0, 5, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    border = _most_common_nonbg(g, int(bg))
    assert border is not None
    frames = _find_frames(g, int(border))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 3:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    tokens = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(tokens) >= 3 and len(tokens) % 2 == 1
    path_colors = [int(x) for x in tokens[::2]]
    dir_tokens = [int(x) for x in tokens[1::2]]
    assert len(path_colors) == len(dir_tokens) + 1
    assert all(int(d) in _DIR_COLOR_TO_NAME for d in dir_tokens)

    start_color = int(path_colors[0])
    next_color = int(path_colors[1])
    first_dir = int(dir_tokens[0])
    assert start_color in by

    cur = _choose_start_frame(by[int(start_color)], by.get(int(next_color), []), int(first_dir))
    assert cur is not None

    out = g.copy()
    for i, dir_col in enumerate(dir_tokens):
        src_color = int(path_colors[i])
        dst_color = int(path_colors[i + 1])
        assert int(cur["inner"]) == int(src_color)
        nxt = _choose_directed_target(cur, by.get(int(dst_color), []), int(dir_col))
        assert nxt is not None
        _draw_bridge(out, cur, nxt, int(src_color))
        cur = nxt

    return out


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
            colors = set(map(int, inner.flatten()))
            if len(colors) != 1:
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
            inner_col = int(next(iter(colors)))
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


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _relative_direction(f_from: dict, f_to: dict) -> str | None:
    same_row, same_col = _aligned(f_from, f_to)
    r0a, c0a, r1a, c1a = map(int, f_from["bbox"])
    r0b, c0b, r1b, c1b = map(int, f_to["bbox"])
    if same_row:
        if int(c1a) < int(c0b):
            return "right"
        if int(c1b) < int(c0a):
            return "left"
    if same_col:
        if int(r1a) < int(r0b):
            return "down"
        if int(r1b) < int(r0a):
            return "up"
    return None


def _frame_distance(f1: dict, f2: dict) -> int:
    r1, c1 = _center(f1)
    r2, c2 = _center(f2)
    return int(abs(int(r1) - int(r2)) + abs(int(c1) - int(c2)))


def _choose_directed_target(cur: dict, cands: list[dict], dir_col: int) -> dict | None:
    want = _DIR_COLOR_TO_NAME[int(dir_col)]
    best = None
    bestf = None
    for f in cands:
        if f is cur:
            continue
        if _relative_direction(cur, f) != want:
            continue
        cand = (int(_frame_distance(cur, f)),) + _center(f)
        if best is None or cand < best:
            best = cand
            bestf = f
    return bestf


def _choose_start_frame(start_cands: list[dict], next_cands: list[dict], first_dir: int) -> dict | None:
    best = None
    bestf = None
    for f in start_cands:
        nxt = _choose_directed_target(f, next_cands, int(first_dir))
        if nxt is None:
            continue
        cand = (int(_frame_distance(f, nxt)),) + _center(f)
        if best is None or cand < best:
            best = cand
            bestf = f
    if bestf is not None:
        return bestf
    if len(start_cands) == 1:
        return start_cands[0]
    return None


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    H, W = out.shape
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    same_row, same_col = _aligned(fa, fb)
    if same_row:
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
    if same_col:
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        for r in range(int(rr0), int(rr1) + 1):
            for c in range(int(cc0), int(cc1) + 1):
                if 0 <= r < H and 0 <= c < W:
                    out[int(r), int(c)] = int(color)


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


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    dr = int(b[0]) - int(a[0])
    dc = int(b[1]) - int(a[1])
    for dir_name, delta in _DIR_NAME_TO_DELTA.items():
        if (int(dr), int(dc)) == (int(delta[0]), int(delta[1])):
            for col, name in _DIR_COLOR_TO_NAME.items():
                if name == dir_name:
                    return int(col)
    return None


def _row_major(slot: tuple[int, int]) -> tuple[int, int]:
    return int(slot[0]), int(slot[1])


def _critical_decoys(
    current: tuple[int, int],
    target: tuple[int, int],
    path_set: set[tuple[int, int]],
    R: int,
    C: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for rr, cc in neighbors4(int(current[0]), int(current[1])):
        cand = (int(rr), int(cc))
        if not in_bounds((int(R), int(C)), int(rr), int(cc)):
            continue
        if cand == target or cand in path_set:
            continue
        if _row_major(cand) < _row_major(target):
            out.append(cand)
    return out


def _old_style_choice(
    current: tuple[int, int],
    target_color: int,
    frames_by_slot: dict[tuple[int, int], dict],
    colors_by_slot: dict[tuple[int, int], int],
    ordered_slots: list[tuple[int, int]],
) -> tuple[int, int] | None:
    cur_frame = frames_by_slot[current]
    best = None
    best_slot = None
    for slot in ordered_slots:
        if slot == current or int(colors_by_slot[slot]) != int(target_color):
            continue
        f = frames_by_slot[slot]
        same_row, same_col = _aligned(cur_frame, f)
        al = 1 if (same_row or same_col) else 0
        cand = (int(al), -int(_frame_distance(cur_frame, f)))
        if best is None or cand > best:
            best = cand
            best_slot = slot
    return best_slot


def _gen_attempt() -> np.ndarray | None:
    colors = list(np.random.permutation(np.array(_NONDIR_POOL, dtype=int)))
    bg = int(colors[0])
    border = int(colors[1])
    inner_palette = [int(x) for x in colors[2:]]
    if len(inner_palette) < 4:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 3))
    outer = int(inner + 2 * t)

    dims = [(3, 4), (4, 3), (4, 4)]
    R, C = dims[int(np.random.randint(0, len(dims)))]
    n_slots = int(R * C)
    n_path = int(np.random.randint(4, min(7, n_slots - 1) + 1))

    path: list[tuple[int, int]] | None = None
    critical_steps: list[tuple[int, list[tuple[int, int]]]] | None = None
    for _ in range(500):
        start = (int(np.random.randint(0, R)), int(np.random.randint(0, C)))
        p = [start]
        seen = {start}
        ok = True
        for _step in range(int(n_path) - 1):
            r, c = p[-1]
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({int(r) for r, _c in p}) < 2 or len({int(c) for _r, c in p}) < 2:
            continue

        pset = set(p)
        crit: list[tuple[int, list[tuple[int, int]]]] = []
        for i, (a, b) in enumerate(zip(p, p[1:])):
            opts = _critical_decoys(a, b, pset, int(R), int(C))
            if opts:
                crit.append((int(i), opts))
        if not crit:
            continue
        path = p
        critical_steps = crit
        break
    if path is None or critical_steps is None:
        return None

    row_gap = [int(np.random.randint(1, 5)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 5)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    frames_by_slot: dict[tuple[int, int], dict] = {}
    ordered_slots: list[tuple[int, int]] = []
    for rr in range(int(R)):
        for cc in range(int(C)):
            slot = (int(rr), int(cc))
            r0 = int(r0s[int(rr)])
            c0 = int(c0s[int(cc)])
            r1 = int(r0 + outer - 1)
            c1 = int(c0 + outer - 1)
            frames_by_slot[slot] = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
                "slot": slot,
            }
            ordered_slots.append(slot)

    max_r1 = max(int(f["bbox"][2]) for f in frames_by_slot.values())
    max_c1 = max(int(f["bbox"][3]) for f in frames_by_slot.values())
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    H = int(ctrl_r + 2)
    W = int(max_c1 + 1 + np.random.randint(1, 5))

    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    start_slot = path[0]
    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    other_palette = [int(x) for x in inner_palette if int(x) != int(start_color)]
    if len(other_palette) < 2:
        return None

    path_colors_by_slot: dict[tuple[int, int], int] = {start_slot: int(start_color)}
    for slot in path[1:]:
        path_colors_by_slot[slot] = int(np.random.choice(np.array(other_palette, dtype=int)))

    non_path_slots = [slot for slot in ordered_slots if slot not in set(path)]
    if not non_path_slots:
        return None

    chosen_critical: list[tuple[int, tuple[int, int]]] = []
    used_decoys: set[tuple[int, int]] = set()
    shuffled = list(np.random.permutation(np.array([i for i, _opts in critical_steps], dtype=int)))
    for step_idx in shuffled:
        opts_lookup = None
        for idx, opts in critical_steps:
            if int(idx) == int(step_idx):
                opts_lookup = list(opts)
                break
        if opts_lookup is None:
            continue
        np.random.shuffle(opts_lookup)
        for slot in opts_lookup:
            if slot in used_decoys:
                continue
            used_decoys.add(slot)
            chosen_critical.append((int(step_idx), slot))
            break
        if len(chosen_critical) >= min(3, len(critical_steps)):
            break
    if not chosen_critical:
        return None

    colors_by_slot: dict[tuple[int, int], int] = {}
    for slot in ordered_slots:
        if slot == start_slot:
            colors_by_slot[slot] = int(start_color)
        elif slot in path_colors_by_slot:
            colors_by_slot[slot] = int(path_colors_by_slot[slot])
        else:
            colors_by_slot[slot] = int(np.random.choice(np.array(other_palette, dtype=int)))

    for step_idx, decoy_slot in chosen_critical:
        target_slot = path[int(step_idx) + 1]
        colors_by_slot[decoy_slot] = int(path_colors_by_slot[target_slot])

    if sum(1 for slot, col in colors_by_slot.items() if int(col) == int(start_color) and slot != start_slot) != 0:
        return None

    tokens: list[int] = [int(path_colors_by_slot[path[0]])]
    dir_tokens_only: list[int] = []
    for a, b in zip(path, path[1:]):
        dir_col = _step_dir(a, b)
        if dir_col is None:
            return None
        dir_tokens_only.append(int(dir_col))
        tokens.append(int(dir_col))
        tokens.append(int(path_colors_by_slot[b]))

    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(tokens)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for slot in ordered_slots:
        frame = frames_by_slot[slot]
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[slot]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = frames_by_slot[a_slot]
        fb = frames_by_slot[b_slot]
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
        paint = int(colors_by_slot[a_slot])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    for col, tok in zip(ctrl_cols, tokens):
        out[int(ctrl_r), int(col)] = int(tok)

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

    # Generator-side anti-degeneration check:
    # at least one chosen critical step must make the old "ignore direction and use color only"
    # policy select a different square than the directed policy.
    diff_steps = 0
    for a_slot, dir_col, b_slot in zip(path, dir_tokens_only, path[1:]):
        target_color = int(colors_by_slot[b_slot])
        old_choice = _old_style_choice(a_slot, int(target_color), frames_by_slot, colors_by_slot, ordered_slots)
        if old_choice is not None and old_choice != b_slot:
            diff_steps += 1
    if diff_steps < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
