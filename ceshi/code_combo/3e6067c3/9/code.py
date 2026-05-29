# Combo variant of 3e6067c3 (index 9)
# combo_spec: D3/1+D7/0
# Name: 左侧列控制+紧凑裁剪

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

"""
Design analysis for combo D3/1 + D7/0.

a) Concrete behavior and trace
- Source task: framed squares live in the main field, a sparse control row sits below
  them, and the solver returns the full canvas after drawing bridges.
- Combo task: the control sequence moves to one dedicated column strictly left of the
  square field, read top-to-bottom; after bridges are drawn, the output is cropped to
  the smallest rectangle containing only squares and bridges.
- Small trace:
  source input: bottom row colors [2, 5, 7] -> read left-to-right -> paint bridge 2
  from square 2 to square 5, then bridge 5 to square 7 -> keep full grid.
  combo input: left column colors [2, 5, 7] -> read top-to-bottom -> paint the same
  bridges -> crop to the frame-field bbox, which usually removes the entire control
  column and all outer margins.
- Important fusion point: D7 cannot be implemented by "crop rows above control row"
  anymore, because under D3 there may be control pixels above, beside, and below the
  square field. The crop must be defined from the square/bridge scene itself.

b) Complexity calibration
- The generator keeps the full visitation order visible as repeated isolated pixels in
  one column, so the new reading direction is obvious.
- It also keeps positive top/right/bottom margins plus a left gap between control
  column and squares, so tight cropping is visually obvious and not a no-op.
- Row/column-band disambiguation among framed squares is preserved from the source
  task; the combo changes layout and output shape, not the bridge-matching principle.

c) Implicit-assumption audit
- Frame extraction from border-color connected components remains valid.
- Original `scan below frames for a control row` is invalid and replaced by `scan left
  of all frames for a non-bg control column`.
- Original D7 crop helper relied on a control row to exclude instruction pixels by
  height. That assumption breaks under D3, so the combo crop is based on frame bboxes.
- Bridge drawing still uses original coordinates on the full input-sized canvas, then
  the crop is applied last.

d) Placement legality
- The generator reserves one control column strictly left of every frame and uses
  background-only spacer columns between control and frame field.
- Frames lie on a lattice with at least one bg cell between neighboring outer borders,
  so every bridge occupies a clean gap.
- Output-first validation rejects overlaps, out-of-bounds placements, or any sample
  where `main(inp)` fails to reproduce the intended cropped output.

e) Visual observability
- Across a few examples, a human can see a vertical color legend on the left and that
  it vanishes in the output because the output is cropped around the solved playfield.
- No hidden ordering by numeric color id is used; the visible top-to-bottom order of
  the control pixels is the only sequence encoding.

f) Visual cleanliness
- Control pixels are isolated in their own strip and never touch frames or bridges.
- The output contains only the completed playfield, so the crop removes clutter rather
  than introducing additional layered markings.

g) Color inferability
- Each control pixel color still refers directly to a framed square's inner color.
- The palette stays moderate and row/column uniqueness constraints provide structure,
  so repeated examples support the matching rule.

h) Anti-degeneration checks in the generator
- The control information is always in a left column and never in a usable bottom row.
- The cropped output is forced to be strictly smaller than the input in both height
  and width.
- The crop always excludes the control column because the crop bbox starts at the
  leftmost frame, not the leftmost non-background pixel of the whole canvas.

i) New solver assumptions introduced by the combo
- There is exactly one useful non-background column strictly left of the frame field,
  and it contains at least two control pixels.
- All bridge pixels stay within the bounding box spanned by the framed squares.
- The generator guarantees both assumptions and validates them by reconstruction.
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


def _read_left_control_sequence(g: np.ndarray, bg: int, frames: list[dict]) -> tuple[int, list[int]]:
    min_frame_c = min(int(f["bbox"][1]) for f in frames)
    ctrl_c = None
    for c in range(int(min_frame_c) - 1, -1, -1):
        if int((g[:, int(c)] != int(bg)).sum()) >= 2:
            ctrl_c = int(c)
            break
    assert not (ctrl_c is None)
    ctrl = [int(g[int(r), int(ctrl_c)]) for r in range(int(g.shape[0])) if int(g[int(r), int(ctrl_c)]) != int(bg)]
    assert not (len(ctrl) < 2)
    return int(ctrl_c), ctrl


def _tight_crop_playfield(g: np.ndarray, frames: list[dict]) -> np.ndarray:
    """Crop to the minimal rectangle containing all frames and any bridges between them."""
    r0 = min(int(f["bbox"][0]) for f in frames)
    c0 = min(int(f["bbox"][1]) for f in frames)
    r1 = max(int(f["bbox"][2]) for f in frames)
    c1 = max(int(f["bbox"][3]) for f in frames)
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


def _choose_target(cur: dict, cands: list[dict]) -> dict:
    best = None
    best_frame = None
    for cand_frame in cands:
        sr, sc = _aligned(cur, cand_frame)
        aligned_score = 1 if (sr or sc) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(cand_frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            best_frame = cand_frame
    assert best_frame is not None
    return best_frame


def _choose_start_frame(ctrl: list[int], by_color: dict[int, list[dict]]) -> dict:
    start_color = int(ctrl[0])
    assert not (start_color not in by_color)
    start_cands = by_color[start_color]
    if len(start_cands) == 1:
        return start_cands[0]
    next_color = int(ctrl[1])
    next_cands = by_color.get(next_color, [])
    best = None
    best_frame = None
    for frame in start_cands:
        for next_frame in (next_cands if next_cands else [frame]):
            sr, sc = _aligned(frame, next_frame)
            aligned_score = 1 if (sr or sc) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(next_frame)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
    assert best_frame is not None
    return best_frame


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
                if 0 <= r < h and 0 <= c < w:
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
                if 0 <= r < h and 0 <= c < w:
                    out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a top-to-bottom control sequence from the left side, draw bridges, then
    tight-crop to the square/bridge playfield so the control strip disappears.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    counts = Counter(map(int, g.flatten()))
    border = None
    for color, _k in counts.most_common():
        if int(color) != int(bg):
            border = int(color)
            break
    assert not (border is None)

    frames = _extract_frames(g, int(border))
    assert not (not frames)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    _ctrl_c, ctrl = _read_left_control_sequence(g, int(bg), frames)

    cur = _choose_start_frame(ctrl, by_color)
    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by_color:
            continue
        nxt = _choose_target(cur, by_color[b])
        _draw_bridge(out, cur, nxt, int(a))
        cur = nxt

    return _tight_crop_playfield(out, frames)


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    counts = Counter(map(int, g.flatten()))
    for color, _k in counts.most_common():
        if int(color) != int(bg):
            return int(color)
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
    if len(inner_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    r_slots = int(np.random.randint(2, 6))
    c_slots = int(np.random.randint(2, 6))
    max_frames = int(r_slots * c_slots)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(12, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        candidate = [(0, 0)]
        visited = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, candidate[-1])
            options = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(r_slots), int(c_slots)), int(rr), int(cc)) and (int(rr), int(cc)) not in visited
            ]
            if not options:
                ok = False
                break
            nxt = options[int(np.random.randint(0, len(options)))]
            candidate.append((int(nxt[0]), int(nxt[1])))
            visited.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in candidate}) < 2 or len({c for _r, c in candidate}) < 2:
            continue
        path = candidate
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(r_slots) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(c_slots) - 1)]
    top_margin = int(np.random.randint(1, 4))
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
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)

    ctrl_top = int(np.random.randint(0, 3))
    ctrl_rows = [int(ctrl_top + 2 * i) for i in range(int(n_frames))]
    bottom_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 5))
    h = int(max(int(max_r1 + 1), int(ctrl_rows[-1] + 1)) + bottom_margin)
    w = int(max_c1 + 1 + right_margin)
    if not (7 <= int(h) <= 30 and 7 <= int(w) <= 30):
        return None

    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)
    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        rr = int(rr)
        cc = int(cc)
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_color)}
        choices = [int(color) for color in inner_palette if int(color) not in forbid]
        if not choices:
            return None
        chosen = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(chosen)
        used_in_row[int(rr)].add(int(chosen))
        used_in_col[int(cc)].add(int(chosen))

    if len(colors_by_slot) != len(path):
        return None

    full_out = np.full((int(h), int(w)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            full_out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(rr, cc)]),
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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            full_out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    for row, color in zip(ctrl_rows, ctrl_colors):
        full_out[int(row), int(ctrl_c)] = int(color)

    expected = _tight_crop_playfield(full_out, frames)
    if not (expected.shape[0] < full_out.shape[0] and expected.shape[1] < full_out.shape[1]):
        return None
    min_frame_c = min(int(frame["bbox"][1]) for frame in frames)
    if not (int(ctrl_c) < int(min_frame_c)):
        return None

    inp = full_out.copy()
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
