# Combo variant of 3e6067c3 (index 26)
# combo_spec: D3/1+D6/1+D7/0
# Name: 左列控制+擦除控制+紧凑裁剪

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

"""
Design analysis for combo D3/1 + D6/1 + D7/0

a) Concrete behavior and trace
- Source task: framed squares occupy the main field, a sparse bottom control row lists inner colors
  left-to-right, and the solver draws bridges between consecutive matching squares while leaving
  both the squares and the control row visible.
- Combo input: the same kind of framed squares are present, but the control sequence is moved into
  one dedicated column strictly left of all frames and must be read top-to-bottom.
- Combo output: first draw the same bridges, then recolor every actually visited square into a
  solid block of its own inner color, then erase the whole control column back to background, and
  finally tight-crop to the minimal bbox of the remaining non-background scene.
- Small trace:
  input has left control column [2, 5, 7], framed square A has inner 2, B has inner 5, C has
  inner 7, and one extra decoy square D is not listed.
  Original behavior would keep the full canvas and leave all frames intact.
  Combo behavior reads 2->5->7 vertically, paints a 2-colored bridge A->B and a 5-colored bridge
  B->C, turns A/B/C into solid 2/5/7 blocks, clears the left control column, and crops away every
  outer margin. The final output contains only the solved playfield; it has no visible control
  markers anywhere.

b) Complexity calibration
- D3 alone could be too close to the source task if the only change were guide placement, so the
  generator always reserves one dedicated left control column and never creates a usable bottom
  control row.
- D6 alone can visually collapse if every square is on the path. To keep the visited/unvisited
  contrast observable, the generator always includes at least one non-path decoy frame that stays
  framed in the output.
- D7 alone could look like a cosmetic resize if margins are zero. The generator keeps positive
  top/left/right/bottom margins and separates the control column from the playfield by background.
- The output therefore has three coupled visible effects at once: bridges appear, visited nodes
  lose their border, and all instruction pixels disappear because the erased left control column is
  outside the cropped bbox.

c) Implicit-assumption audit of the original main()
- Border detection by "most common non-background color" still works, but only on the input. The
  combo therefore performs all frame detection before any visited-node recoloring happens.
- Original control-row search below the frames is invalid after D3. It must be replaced by a scan
  for a non-background column strictly left of the leftmost frame.
- Original `return out` assumed the output canvas matches the input canvas. D7 breaks that, so the
  combo must erase the control strip and then compute a fresh tight crop.
- D6 introduces a new ordering dependency: visited-square recoloring must happen after frame
  extraction and after each path step is chosen, otherwise frame borders needed for detection would
  be destroyed too early.
- D3 + D7 together introduce a crop hazard: if the control column were not erased before cropping,
  the left strip would contaminate the bbox and remain visible. The combo resolves this by erasing
  the control column first and only then calling tight crop.

d) Placement legality
- Frames are placed on a lattice with at least one background cell between neighboring outer
  borders, so every bridge occupies only gap cells.
- The path is a self-avoiding 4-neighbor walk on lattice slots, so consecutive path squares are
  always aligned in the same row or column band and `_bridge_cells` is non-empty.
- Decoy frames are chosen from unused lattice slots, so they never overlap path frames or bridges.
- The left control column is separated from the playfield by at least two background columns and is
  the only non-background column to the left of all frames.

e) Visual observability
- Across a few examples, a human sees that the input contains a vertical color legend on the left,
  but the output omits it completely and shows only a solved, tightly cropped object field.
- The visited/unvisited contrast is visible because at least one decoy frame keeps the common
  border while path frames become solid blocks.
- No hidden property such as numeric color ordering is used; the only order clue is the visible
  top-to-bottom control layout in the input.

f) Visual cleanliness
- The control strip is isolated from the main scene horizontally, so it never merges with borders
  or bridges.
- The visited-state mark reuses each square's own color instead of introducing a new annotation
  palette, which keeps the scene legible.
- Tight crop removes empty margins and the now-erased control region, so the output is denser but
  less cluttered than the input.

g) Change inferability
- Bridge colors still equal the source square's inner color, matching the source task.
- The D6 change is structural rather than arbitrary: a visited frame becomes a solid version of the
  same color it already contained.
- Because the output no longer contains the guide pixels, the path must be inferred from the joint
  pattern of solidified squares plus bridge colors. That inference remains feasible because the
  generator limits colors to the normal ARC palette, keeps the path contiguous, and preserves at
  least one untouched reference frame.

h) Anti-degeneration checks used in generation
- The generator rejects scenes with fewer than two successful bridges.
- It also rejects scenes without at least one decoy frame, without strict crop shrinkage in both
  dimensions, or without any visited-square recoloring beyond the bridge pixels.
- These checks prevent the combo from collapsing back to "just D3" or "just D7".

i) New solver assumptions introduced by the combo
- Exactly one usable control column exists strictly left of all frames and it contains at least two
  non-background pixels. The generator reserves that column explicitly.
- Every control color refers to a path frame present in the input. The generator assigns unique
  colors to all frames and uses only path-frame colors in the control list.
- At least one decoy frame exists so the D6 visited/unvisited contrast is observable.
- Tight crop assumes that after erasing the control column there is still foreground left; frame
  existence guarantees that.
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

        thick = None
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

            thick = int(tt)
            inner_col = int(next(iter(uniq)))
            break

        if thick is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(thick),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + thick), int(c0 + thick), int(r1 - thick), int(c1 - thick)),
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
    best_frame = None
    for frame in cands:
        same_row, same_col = _aligned(cur, frame)
        aligned_score = 1 if (same_row or same_col) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            best_frame = frame
    return best_frame


def _choose_start(ctrl: list[int], by_color: dict[int, list[dict]]) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by_color
    start_cands = by_color[start_color]
    if len(start_cands) == 1:
        return start_cands[0]

    nxt_color = int(ctrl[1]) if len(ctrl) >= 2 else int(ctrl[0])
    nxt_cands = by_color.get(nxt_color, [])
    best = None
    best_frame = None
    for frame in start_cands:
        for other in (nxt_cands if nxt_cands else [frame]):
            same_row, same_col = _aligned(frame, other)
            aligned_score = 1 if (same_row or same_col) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(other)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
    assert best_frame is not None
    return best_frame


def _read_left_control_sequence(g: np.ndarray, bg: int, frames: list[dict]) -> tuple[int, list[int]]:
    min_frame_c = min(int(frame["bbox"][1]) for frame in frames)
    ctrl_c = None
    for c in range(int(min_frame_c) - 1, -1, -1):
        if int((g[:, int(c)] != int(bg)).sum()) >= 2:
            ctrl_c = int(c)
            break
    assert ctrl_c is not None
    ctrl = [int(g[int(r), int(ctrl_c)]) for r in range(int(g.shape[0])) if int(g[int(r), int(ctrl_c)]) != int(bg)]
    assert len(ctrl) >= 2
    return int(ctrl_c), ctrl


def _mark_visited(out: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(frame["inner"])


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    h, w = map(int, out.shape)
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
                if 0 <= r < h and 0 <= c < w:
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
                if 0 <= r < h and 0 <= c < w:
                    out[int(r), int(c)] = int(color)


def _tight_crop_nonbg(g: np.ndarray, bg: int) -> np.ndarray:
    fg = np.argwhere(g != int(bg))
    assert fg.size > 0
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _render_combo(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _n in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _extract_frames(g, int(border))
    assert frames

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    ctrl_c, ctrl = _read_left_control_sequence(g, int(bg), frames)
    cur = _choose_start(ctrl, by_color)
    out = g.copy()

    for i in range(int(len(ctrl)) - 1):
        src_color = int(ctrl[i])
        dst_color = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(src_color)
        if dst_color not in by_color:
            continue
        nxt = _choose_candidate(cur, by_color[dst_color])
        if nxt is None:
            continue
        _draw_bridge(out, cur, nxt, int(src_color))
        _mark_visited(out, cur)
        _mark_visited(out, nxt)
        cur = nxt

    out[:, int(ctrl_c)] = int(bg)
    return _tight_crop_nonbg(out, int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a left-side control column, draw bridges, mark visited squares as solid, erase the
    control strip, and tight-crop the solved scene.
    """
    return _render_combo(grid)


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _n in cnt.most_common():
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

    r_slots = int(np.random.randint(2, 6))
    c_slots = int(np.random.randint(2, 6))
    max_slots = int(r_slots * c_slots)
    if max_slots < 5:
        return None

    max_path = int(min(7, max_slots - 1, len(inner_palette) - 1))
    if max_path < 4:
        return None
    n_path = int(np.random.randint(4, max_path + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        walk = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for _step in range(int(n_path) - 1):
            r, c = map(int, walk[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(r_slots), int(c_slots)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            walk.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in walk}) < 2 or len({c for _r, c in walk}) < 2:
            continue
        path = walk
        break
    if path is None:
        return None

    used_slots = set(path)
    spare_slots = [(int(r), int(c)) for r in range(int(r_slots)) for c in range(int(c_slots)) if (int(r), int(c)) not in used_slots]
    max_decoys = int(min(3, len(spare_slots), len(inner_palette) - n_path))
    if max_decoys < 1:
        return None
    n_decoys = int(np.random.randint(1, max_decoys + 1))
    decoy_idx = np.random.choice(len(spare_slots), size=int(n_decoys), replace=False)
    decoy_slots = [spare_slots[int(i)] for i in np.array(decoy_idx).tolist()]

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(r_slots) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(c_slots) - 1)]
    top_margin = int(np.random.randint(1, 4))
    bottom_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 5))
    ctrl_c = int(np.random.randint(0, 3))
    ctrl_gap = int(np.random.randint(2, 6))
    left_margin = int(ctrl_c + ctrl_gap + 1)

    r0s = [int(top_margin)]
    for k in range(int(r_slots) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(c_slots) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in list(path) + list(decoy_slots):
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
    ctrl_top = int(np.random.randint(0, 3))
    ctrl_rows = [int(ctrl_top + 2 * i) for i in range(int(n_path))]

    h = int(max(int(max_r1 + 1), int(ctrl_rows[-1] + 1)) + bottom_margin)
    w = int(max_c1 + 1 + right_margin)
    if not (7 <= int(h) <= 30 and 7 <= int(w) <= 30):
        return None

    all_colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(n_path + n_decoys), replace=False).tolist()]
    path_colors = all_colors[: int(n_path)]
    decoy_colors = all_colors[int(n_path) :]

    colors_by_slot: dict[tuple[int, int], int] = {}
    for slot, color in zip(path, path_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)
    for slot, color in zip(decoy_slots, decoy_colors):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(color)

    inp = np.full((int(h), int(w)), int(bg), dtype=int)
    full_out = np.full((int(h), int(w)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        inner_col = int(colors_by_slot[(int(rr), int(cc))])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(inp, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))
        _draw_frame(full_out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))

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
            if not (0 <= int(r) < int(h) and 0 <= int(c) < int(w)):
                ok = False
                break
            if int(full_out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            full_out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok or len(bridge_cells_all) == 0:
        return None

    for slot in path:
        frame = slot_to_frame[(int(slot[0]), int(slot[1]))]
        r0, c0, r1, c1 = map(int, frame["bbox"])
        full_out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(colors_by_slot[(int(slot[0]), int(slot[1]))])

    for row, color in zip(ctrl_rows, path_colors):
        inp[int(row), int(ctrl_c)] = int(color)
        full_out[int(row), int(ctrl_c)] = int(color)

    expected_canvas = full_out.copy()
    expected_canvas[:, int(ctrl_c)] = int(bg)
    expected = _tight_crop_nonbg(expected_canvas, int(bg))

    if not (expected.shape[0] < inp.shape[0] and expected.shape[1] < inp.shape[1]):
        return None
    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    visited_gain = 0
    for slot in path:
        frame = slot_to_frame[(int(slot[0]), int(slot[1]))]
        r0, c0, r1, c1 = map(int, frame["bbox"])
        before = inp[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        after = full_out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        visited_gain += int(np.count_nonzero(before != after))
    if visited_gain <= int(len(bridge_cells_all)):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != expected.shape:
        return None
    if not bool(np.array_equal(pred, expected)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo sample after 500 attempts")
