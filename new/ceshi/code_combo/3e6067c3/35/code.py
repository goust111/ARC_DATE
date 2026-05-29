# Combo variant of 3e6067c3 (index 35)
# combo_spec: D1/0+D3/1+D5/0+D7/0
# Name: 边框匹配+左列控制+终点色桥梁+紧凑裁剪

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

"""
Design analysis for the fused combo variant
===========================================

a) Concrete behavior and trace
- Original task:
  framed squares share one global border color; the control sequence sits on a
  bottom row and is read left-to-right; each control color matches a square's
  inner fill; bridge A->B is painted with A's inner color; the full canvas is
  returned.
- Fused variant:
  1. Each square is identified by its border color, not by its inner fill.
  2. The control sequence moves to a dedicated left column and is read
     top-to-bottom.
  3. Each bridge segment is painted with the destination square's border color.
  4. The final output is tight-cropped to the main scene, excluding the control
     column and all external margins.
- Small trace:
  suppose three path squares have border colors [2, 5, 7] and all of them share
  the same inner fill 6. The input shows three framed squares whose borders are
  2/5/7, plus a left control column [2, 5, 7] from top to bottom.
  Original logic would have looked for inner colors on a bottom row and painted
  bridge colors [2, 5] only because those were the source colors.
  Fused logic instead:
  - matches the first control pixel 2 to the square whose *border* is 2;
  - then connects it to the square whose *border* is 5 and paints that segment 5;
  - then connects 5 to 7 and paints that segment 7;
  - finally crops away the control column and outer padding.
  The visible differences all reinforce each other: controls align with borders,
  bridge colors point toward their destinations, and the output is just the
  completed playfield.

b) Complexity calibration
- Border colors are unique per square while the inner fill is intentionally
  repeated across all squares. This prevents degeneration back to the original
  "match inner color" rule and makes the border-matching change obvious.
- The control sequence remains explicit pixel-by-pixel, only its placement and
  reading direction change. No hidden encoding is introduced.
- Bridge geometry is unchanged from the source task, so the problem stays about
  recognizing the same row/column bands rather than about a new drawing shape.
- The generator caps the number of frames by the available distinct border
  colors, avoiding ambiguous or impossible color assignments.

c) Implicit-assumption audit
- Original assumption: one global border color can be recovered as the most
  common non-background color. This fails under D1/0, so the fused solver must
  enumerate every non-background color as a possible border color and detect
  framed rings color-by-color.
- Original assumption: controls live below the scene on one row. This fails
  under D3/1, so the solver must search for a dedicated non-background column
  strictly left of all frames and read it top-to-bottom.
- Original assumption: `by[color]` groups frames by inner fill. This fails under
  border matching; the fused solver groups by `frame["border"]`.
- Original assumption: bridge color equals the source square's matched color.
  Under the fusion, D5/0's "destination color" must be interpreted in the new
  matching vocabulary introduced by D1/0, so bridge color becomes the
  destination square's border color, not its repeated inner fill.
- Original assumption: output shape equals input shape. D7/0 invalidates that,
  so the fused solver paints on the full input canvas first, then crops only the
  main scene to the right of the control column.

d) Placement legality
- The generator builds the full solved canvas first, then erases bridges to form
  the input. This guarantees `main(input) == expected_output` for accepted
  samples.
- Frames lie on a lattice with at least one background cell between neighbors,
  ensuring bridges fit in the row/column gaps without touching frames.
- The control column is placed strictly left of the frame field with a positive
  gap, so it never merges with borders or bridges.
- Bridge cells are validated for overlap, bounds, and background occupancy before
  painting.

e) Visual observability
- A human can infer the border-matching rule because control colors appear on the
  square borders while the interiors repeat a common color and therefore cannot
  explain the sequence.
- The left-column placement makes the top-to-bottom reading direction visually
  explicit across several examples.
- Destination-colored bridges are observable because adjacent path squares always
  have different border colors, so each segment matches the square it leads into.
- Tight crop is obvious because the output loses the control column and margins.

f) Visual cleanliness
- The scene still contains only three object classes: framed squares, bridges,
  and control pixels. They occupy disjoint spatial regions.
- Reusing one inner color keeps the image tidy and foregrounds the informative
  border colors.
- Tight crop removes dead background rather than adding more annotation layers.

g) Color-change inferability
- The palette roles are constrained:
  background = one color; inner fill = one repeated color; border/control colors
  = unique path labels reused for the bridges.
- No arbitrary recoloring table is introduced. Bridge colors come directly from
  the visibly marked destination borders.

h) Quantitative anti-degeneration targets
- Every accepted sample should satisfy:
  1. all frame inner colors are identical;
  2. all control colors are frame border colors;
  3. at least one bridge segment changes if source-colored bridges are used
     instead of destination-colored bridges, and in practice every segment does;
  4. output shape is strictly smaller than input shape in both dimensions.

i) New solver assumptions and generator guarantees
- Assumption: there exists exactly one useful control column strictly left of the
  frames with at least two non-background cells. The generator reserves that
  column and leaves other columns on the left empty.
- Assumption: every control color matches at least one detected frame border.
  The generator writes the control sequence directly from the path frames'
  border colors.
- Assumption: path-adjacent frames share either the same row band or the same
  column band. The generator uses a 4-neighbor lattice path, which guarantees
  this geometry.
- Assumption: the main scene lies strictly to the right of the control column.
  The generator enforces a positive gap between the control strip and the frame
  field, so tight cropping can safely ignore the control strip.
"""


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = sorted(set(map(int, g.flatten())) - {int(bg)})
    for border_col in colors:
        lab, n = label_mask(g == int(border_col), connectivity=4)
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

            found = None
            for tt in range(1, int(min(h, w) // 2) + 1):
                inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
                if inner.size == 0:
                    continue
                if bool((inner == int(border_col)).any()):
                    continue
                inner_vals = set(map(int, inner.flatten()))
                if len(inner_vals) != 1:
                    continue
                inner_col = int(next(iter(inner_vals)))
                if int(inner_col) == int(bg):
                    continue

                ring_area = int(h * w - (h - 2 * tt) * (w - 2 * tt))
                if int(pts.shape[0]) != int(ring_area):
                    continue

                ok = True
                if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border_col)).all()):
                    ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border_col)).all()):
                    ok = False
                if not ok:
                    continue

                found = {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "t": int(tt),
                    "border": int(border_col),
                    "inner": int(inner_col),
                    "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
                }
                break

            if found is not None:
                frames.append(found)

    return frames


def _read_left_control_sequence(g: np.ndarray, bg: int, frames: list[dict]) -> tuple[int, list[int]]:
    min_frame_c = min(int(f["bbox"][1]) for f in frames)
    ctrl_c = None
    for c in range(int(min_frame_c) - 1, -1, -1):
        if int((g[:, int(c)] != int(bg)).sum()) >= 2:
            ctrl_c = int(c)
            break
    assert ctrl_c is not None
    ctrl = [int(g[int(r), int(ctrl_c)]) for r in range(int(g.shape[0])) if int(g[int(r), int(ctrl_c)]) != int(bg)]
    assert len(ctrl) >= 2
    return int(ctrl_c), ctrl


def _tight_crop_main_scene(g: np.ndarray, bg: int, ctrl_c: int) -> np.ndarray:
    fg = np.argwhere(g[:, int(ctrl_c) + 1 :] != int(bg))
    assert fg.size > 0
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    c0 += int(ctrl_c) + 1
    c1 += int(ctrl_c) + 1
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _best_candidate(cur: dict, cands: list[dict]) -> dict:
    best = None
    bestf = None
    for f in cands:
        sr, sc = _aligned(cur, f)
        al = 1 if (sr or sc) else 0
        r1_, c1_ = _center(cur)
        r2_, c2_ = _center(f)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        cand = (int(al), -int(d))
        if best is None or cand > best:
            best = cand
            bestf = f
    assert bestf is not None
    return bestf


def _choose_start(ctrl: list[int], by: dict[int, list[dict]]) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        return start_cands[0]
    nxt_color = int(ctrl[1])
    nxt_cands = by.get(nxt_color, [])
    best = None
    bestf = None
    for f in start_cands:
        for g2 in (nxt_cands if nxt_cands else [f]):
            sr, sc = _aligned(f, g2)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = _center(f)
            r2_, c2_ = _center(g2)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
    assert bestf is not None
    return bestf


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    h, w = map(int, out.shape)
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
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
                if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
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
                if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
                    out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares using a left-column control sequence, where controls
    match border colors, bridges use the destination border color, and the final
    result is tight-cropped to the main scene.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert frames

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["border"])].append(frame)

    ctrl_c, ctrl = _read_left_control_sequence(g, int(bg), frames)
    cur = _choose_start(ctrl, by)

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["border"]) == int(a)
        assert int(b) in by
        nxt = _best_candidate(cur, by[b])
        _draw_bridge(out, cur, nxt, int(b))
        cur = nxt

    return _tight_crop_main_scene(out, int(bg), int(ctrl_c))


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
    usable = [int(d) for d in _DIGITS if int(d) != int(bg)]
    if len(usable) < 5:
        return None

    inner_col = int(np.random.choice(np.array(usable, dtype=int)))
    border_palette = [int(d) for d in usable if int(d) != int(inner_col)]
    max_unique_borders = int(len(border_palette))

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(2, 4))
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 7):
        return None

    r_dim = int(np.random.randint(2, 6))
    c_dim = int(np.random.randint(2, 6))
    max_frames = int(min(r_dim * c_dim, max_unique_borders))
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(8, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(r_dim), int(c_dim)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
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

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(r_dim) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(c_dim) - 1)]
    top_margin = int(np.random.randint(1, 4))
    ctrl_c = int(np.random.randint(0, 3))
    ctrl_gap = int(np.random.randint(2, 6))
    left_margin = int(ctrl_c + ctrl_gap + 1)

    r0s = [int(top_margin)]
    for k in range(int(r_dim) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(c_dim) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    border_colors = [int(x) for x in np.random.choice(np.array(border_palette, dtype=int), size=int(n_frames), replace=False)]

    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    border_by_slot: dict[tuple[int, int], int] = {}
    for idx, (rr, cc) in enumerate(path):
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        border_col = int(border_colors[int(idx)])
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "border": int(border_col),
            "inner": int(inner_col),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame
        border_by_slot[(int(rr), int(cc))] = int(border_col)

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_top = int(np.random.randint(0, 3))
    ctrl_rows = [int(ctrl_top + 2 * i) for i in range(int(n_frames))]
    bottom_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 5))
    h = int(max(int(max_r1 + 1), int(ctrl_rows[-1] + 1)) + bottom_margin)
    w = int(max_c1 + 1 + right_margin)
    if not (8 <= int(h) <= 30 and 8 <= int(w) <= 30):
        return None

    full_out = np.full((int(h), int(w)), int(bg), dtype=int)
    for frame in frames:
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            full_out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(frame["border"]),
            inner_col=int(inner_col),
        )

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
            if not (0 <= int(r) < int(h) and 0 <= int(c) < int(w)):
                ok = False
                break
            if int(full_out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(border_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        for r, c in cells:
            full_out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(border_by_slot[(int(r), int(c))]) for r, c in path]
    for row, col in zip(ctrl_rows, ctrl_colors):
        full_out[int(row), int(ctrl_c)] = int(col)

    expected = _tight_crop_main_scene(full_out, int(bg), int(ctrl_c))
    if not (expected.shape[0] < full_out.shape[0] and expected.shape[1] < full_out.shape[1]):
        return None

    inp = full_out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None

    frames_found = _detect_frames(inp, int(bg))
    if int(len(frames_found)) != int(len(frames)):
        return None
    if len({int(f["border"]) for f in frames_found}) != int(len(frames)):
        return None
    if {int(f["inner"]) for f in frames_found} != {int(inner_col)}:
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
    raise RuntimeError("Failed to generate valid input after 500 attempts")
