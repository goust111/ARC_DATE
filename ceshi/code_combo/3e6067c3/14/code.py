# Combo variant of 3e6067c3 (index 14)
# combo_spec: D5/3+D8/1
# Name: 渐变桥梁+反向处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


"""
Design analysis for the combo variant
=====================================

This combo fuses two changes:
1. D5/3: every bridge becomes direction-aware and uses a two-zone gradient:
   the half adjacent to the start frame uses the start color, and the half
   adjacent to the end frame uses the end color. For odd lengths, the middle
   cell stays with the start-side half.
2. D8/1 at combo level: although the single-variant note describes an
   "outside-in alternating" order, the combo-level specification explicitly
   says the control sequence must be read from right to left. The combo-level
   specification is the binding behavior here, so this implementation treats
   D8 as reversed control decoding rather than alternating pair scheduling.

a) Concretization and trace
- Input still shows equal-sized framed squares plus one sparse control row
  underneath.
- Suppose the visible control row left-to-right is [green, yellow, red]. In the
  original task, that would mean green -> yellow -> red, and bridges would be
  solid green then solid yellow.
- In this combo, the row must be read right-to-left, so the true path is
  red -> yellow -> green.
- Therefore the first bridge is between the red and yellow frames, and its
  rectangle is split along the travel axis: the source-side half is red, the
  destination-side half is yellow. The next bridge is yellow -> green with the
  same split rule.
- If a solver reads the row in the visible left-to-right direction, it still
  connects the same undirected frame pairs but every bridge gradient points the
  wrong way. This makes the two dimensions cooperate: reversing the control row
  changes which endpoint is "start", which directly changes the gradient.

b) Complexity calibration
- The bridge is not a free-form color interpolation; it has exactly two color
  zones, so the visual rule is crisp.
- The generator forces every bridge gap to be at least 2 cells, so both colors
  are visible on each bridge.
- Samples use 4..8 frames and multiple bridges, giving repeated evidence for
  both the reversed reading direction and the source/destination color split.
- Inner colors are globally unique in generated samples. This removes accidental
  matching ambiguity so the human challenge stays focused on the combo rule
  rather than on disambiguating repeated colors.

c) Hidden assumption audit
- Original `ctrl = [...]` assumed visible left-to-right decoding. That is false
  now, so control extraction must reverse the visible sequence.
- Original `draw_bridge(..., color)` assumed one scalar color for the whole gap.
  That is false now; bridge drawing must preserve axis and direction.
- Original geometry helpers only needed the gap rectangle. The combo now also
  needs to know which side is the start side.
- Original generator could paint bridges with the source color only and still
  validate. That would collapse D5/3, so generator painting must use the same
  directional split as `main()`.
- Because this combo keeps a single linear walk after reversing the control
  sequence, the original stateful `cur -> nxt` path logic is still valid after
  the control reversal. No alternating-pair logic is needed.

d) Placement legality
- Frames stay axis-aligned, equal-sized, non-overlapping, and separated by at
  least one background cell.
- The generator uses a self-avoiding lattice path, so every consecutive pair is
  row- or column-adjacent in slot space and therefore bridgeable.
- Row and column slot gaps are at least 2, so every bridge rectangle is
  non-empty and long enough to show the gradient.
- The output-first build verifies that bridge cells stay in bounds, do not
  collide with frames, and do not overlap other bridges.

e) Visual observability
- Humans can see that the visible rightmost control pixel, not the leftmost one,
  corresponds to the first bridge source color.
- They can also see that each bridge has a color change at its midpoint, with
  each half touching the correspondingly colored frame.
- Across several samples, the consistent clue is: read control pixels from the
  right edge, then make each bridge inherit the two endpoint colors in order.

f) Visual cleanliness
- The scene stays close to the original ARC task: framed squares, empty gaps,
  one control row.
- Bridges remain plain rectangles rather than noisy patterns.
- The generator forbids bridge overlap, so color evidence is not obscured by
  later overpainting.

g) Change inferability
- No hidden numeric ordering of color ids is used.
- Color behavior is structurally anchored: the bridge half touching a frame
  copies that frame's inner color.
- Because displayed control colors are the reverse of the actual walk, sample
  pairs clearly show that the visible left-to-right order is not the execution
  order.

h) Anti-degeneracy metric
- A sample is accepted only if:
  1. the correct solver (`reverse_ctrl=True`, `gradient=True`) reproduces the
     intended output;
  2. a wrong left-to-right gradient solver produces a different output;
  3. a wrong right-to-left solid-color solver also produces a different output.
- This guarantees that both fused dimensions matter on generated samples.

i) New main() assumptions and generator guarantees
- `main()` assumes every consecutive true-path pair is bridgeable by shared row
  or column bands. The self-avoiding lattice path guarantees that.
- `main()` assumes the start side of each bridge is meaningful. The reversed
  control sequence and minimum gap length make that direction visible.
- `main()` assumes there are at least two control colors. The generator always
  uses 4..8 frames.
- `main()` assumes frame colors can be matched stably. The generator guarantees
  global uniqueness of inner colors.
"""


def _frame_center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _bridge_spec(frame_a: dict, frame_b: dict) -> dict | None:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]

    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            return {
                "axis": "h",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(c1a + 1),
                "cc1": int(c0b - 1),
                "start_positive": True,
            }
        if int(c1b) < int(c0a):
            return {
                "axis": "h",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(c1b + 1),
                "cc1": int(c0a - 1),
                "start_positive": False,
            }

    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            return {
                "axis": "v",
                "rr0": int(r1a + 1),
                "rr1": int(r0b - 1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "start_positive": True,
            }
        if int(r1b) < int(r0a):
            return {
                "axis": "v",
                "rr0": int(r1b + 1),
                "rr1": int(r0a - 1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "start_positive": False,
            }

    return None


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    spec = _bridge_spec(frame_a, frame_b)
    if spec is None:
        return set()
    if spec["axis"] == "h":
        if int(spec["cc0"]) > int(spec["cc1"]):
            return set()
        return {
            (int(r), int(c))
            for r in range(int(spec["rr0"]), int(spec["rr1"]) + 1)
            for c in range(int(spec["cc0"]), int(spec["cc1"]) + 1)
        }
    if int(spec["rr0"]) > int(spec["rr1"]):
        return set()
    return {
        (int(r), int(c))
        for r in range(int(spec["rr0"]), int(spec["rr1"]) + 1)
        for c in range(int(spec["cc0"]), int(spec["cc1"]) + 1)
    }


def _bridge_span_len(frame_a: dict, frame_b: dict) -> int:
    spec = _bridge_spec(frame_a, frame_b)
    if spec is None:
        return 0
    if spec["axis"] == "h":
        return max(0, int(spec["cc1"] - spec["cc0"] + 1))
    return max(0, int(spec["rr1"] - spec["rr0"] + 1))


def _paint_bridge_half(grid: np.ndarray, spec: dict, color: int, *, part: str) -> None:
    if spec["axis"] == "h":
        length = int(spec["cc1"] - spec["cc0"] + 1)
        start_len = int((length + 1) // 2)
        if part == "start":
            if spec["start_positive"]:
                cc0 = int(spec["cc0"])
                cc1 = int(spec["cc0"] + start_len - 1)
            else:
                cc0 = int(spec["cc1"] - start_len + 1)
                cc1 = int(spec["cc1"])
        else:
            if spec["start_positive"]:
                cc0 = int(spec["cc0"] + start_len)
                cc1 = int(spec["cc1"])
            else:
                cc0 = int(spec["cc0"])
                cc1 = int(spec["cc1"] - start_len)
        if int(cc0) <= int(cc1):
            grid[int(spec["rr0"]) : int(spec["rr1"]) + 1, int(cc0) : int(cc1) + 1] = int(color)
        return

    length = int(spec["rr1"] - spec["rr0"] + 1)
    start_len = int((length + 1) // 2)
    if part == "start":
        if spec["start_positive"]:
            rr0 = int(spec["rr0"])
            rr1 = int(spec["rr0"] + start_len - 1)
        else:
            rr0 = int(spec["rr1"] - start_len + 1)
            rr1 = int(spec["rr1"])
    else:
        if spec["start_positive"]:
            rr0 = int(spec["rr0"] + start_len)
            rr1 = int(spec["rr1"])
        else:
            rr0 = int(spec["rr0"])
            rr1 = int(spec["rr1"] - start_len)
    if int(rr0) <= int(rr1):
        grid[int(rr0) : int(rr1) + 1, int(spec["cc0"]) : int(spec["cc1"]) + 1] = int(color)


def _paint_gradient_bridge(
    grid: np.ndarray, frame_a: dict, frame_b: dict, start_color: int, end_color: int
) -> bool:
    spec = _bridge_spec(frame_a, frame_b)
    if spec is None:
        return False
    if spec["axis"] == "h" and int(spec["cc0"]) > int(spec["cc1"]):
        return False
    if spec["axis"] == "v" and int(spec["rr0"]) > int(spec["rr1"]):
        return False
    _paint_bridge_half(grid, spec, int(start_color), part="start")
    _paint_bridge_half(grid, spec, int(end_color), part="end")
    return True


def _paint_solid_bridge(grid: np.ndarray, frame_a: dict, frame_b: dict, color: int) -> bool:
    cells = _bridge_cells(frame_a, frame_b)
    if not cells:
        return False
    for r, c in cells:
        grid[int(r), int(c)] = int(color)
    return True


def _detect_frames(grid: np.ndarray, border: int) -> list[dict]:
    labels, n_labels = label_mask(grid == int(border), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n_labels) + 1):
        pts = np.argwhere(labels == int(cid))
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
            inner = grid[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            vals = set(map(int, inner.flatten()))
            if len(vals) != 1:
                continue

            ok = True
            if not bool((grid[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((grid[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((grid[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((grid[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue

            thick = int(tt)
            inner_col = int(next(iter(vals)))
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


def _control_sequence(grid: np.ndarray, bg: int, frames: list[dict], *, reverse_visible: bool) -> list[int]:
    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(grid.shape[0] - 1), int(max_frame_r), -1):
        if int((grid[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(grid[int(ctrl_r), int(c)]) for c in range(int(grid.shape[1])) if int(grid[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2
    if reverse_visible:
        ctrl = list(reversed(ctrl))
    return ctrl


def _choose_start_frame(by_color: dict[int, list[dict]], ctrl: list[int]) -> dict:
    start_color = int(ctrl[0])
    assert start_color in by_color
    start_cands = by_color[int(start_color)]
    if len(start_cands) == 1:
        return start_cands[0]

    next_color = int(ctrl[1])
    next_cands = by_color.get(int(next_color), [])
    best = None
    best_frame = None
    for frame in start_cands:
        for nxt in (next_cands if next_cands else [frame]):
            same_row, same_col = _aligned(frame, nxt)
            aligned = 1 if (same_row or same_col) else 0
            r1, c1 = _frame_center(frame)
            r2, c2 = _frame_center(nxt)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
    assert best_frame is not None
    return best_frame


def _choose_next_frame(cur: dict, cands: list[dict]) -> dict | None:
    best = None
    best_frame = None
    for frame in cands:
        same_row, same_col = _aligned(cur, frame)
        aligned = 1 if (same_row or same_col) else 0
        r1, c1 = _frame_center(cur)
        r2, c2 = _frame_center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned), -int(dist))
        if best is None or cand > best:
            best = cand
            best_frame = frame
    return best_frame


def _solve(grid: np.ndarray, *, reverse_ctrl: bool, gradient: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for color, _count in cnt.most_common():
        if int(color) != int(bg):
            border = int(color)
            break
    assert border is not None

    frames = _detect_frames(g, int(border))
    assert frames

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    ctrl = _control_sequence(g, int(bg), frames, reverse_visible=bool(reverse_ctrl))
    cur = _choose_start_frame(by_color, ctrl)
    out = g.copy()

    for idx in range(int(len(ctrl)) - 1):
        color_a = int(ctrl[int(idx)])
        color_b = int(ctrl[int(idx) + 1])
        assert int(cur["inner"]) == int(color_a)
        if color_b not in by_color:
            continue
        nxt = _choose_next_frame(cur, by_color[int(color_b)])
        if nxt is None:
            continue
        if gradient:
            ok = _paint_gradient_bridge(out, cur, nxt, int(color_a), int(color_b))
        else:
            ok = _paint_solid_bridge(out, cur, nxt, int(color_a))
        if ok:
            cur = nxt

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, reverse_ctrl=True, gradient=True)


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, grid.flatten()))
    for color, _count in cnt.most_common():
        if int(color) != int(bg):
            return int(color)
    return None


def _draw_frame(grid: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    grid[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    grid[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner_side = int(np.random.randint(1, 4))
    outer = int(inner_side + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(min(R * C, len(inner_palette)))
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, max_frames + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        walk = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, walk[-1])
            options = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not options:
                ok = False
                break
            nxt = options[int(np.random.randint(0, len(options)))]
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

    row_gap = [int(np.random.randint(2, 6)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(2, 6)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for gap in row_gap:
        r0s.append(int(r0s[-1] + outer + int(gap)))
    c0s = [int(left_margin)]
    for gap in col_gap:
        c0s.append(int(c0s[-1] + outer + int(gap)))

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
            "t": int(t),
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
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    colors = [int(x) for x in np.random.choice(np.array(inner_palette, dtype=int), size=int(n_frames), replace=False)]
    color_by_slot = {slot: int(color) for slot, color in zip(path, colors)}

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(r1 - r0 + 1),
            t=int(frame["t"]),
            border=int(border),
            inner_col=int(color_by_slot[(int(rr), int(cc))]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    saw_gradient = False
    for slot_a, slot_b in zip(path, path[1:]):
        frame_a = slot_to_frame[(int(slot_a[0]), int(slot_a[1]))]
        frame_b = slot_to_frame[(int(slot_b[0]), int(slot_b[1]))]
        cells = _bridge_cells(frame_a, frame_b)
        if not cells:
            return None
        if int(_bridge_span_len(frame_a, frame_b)) < 2:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        if not _paint_gradient_bridge(
            out,
            frame_a,
            frame_b,
            int(color_by_slot[(int(slot_a[0]), int(slot_a[1]))]),
            int(color_by_slot[(int(slot_b[0]), int(slot_b[1]))]),
        ):
            return None
        bridge_cells_all |= set(cells)
        saw_gradient = True

    if not saw_gradient:
        return None

    actual_ctrl = [int(color_by_slot[(int(r), int(c))]) for r, c in path]
    visible_ctrl = list(reversed(actual_ctrl))
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(visible_ctrl)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(visible_ctrl, ctrl_cols):
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
        wrong_forward = _solve(inp, reverse_ctrl=False, gradient=True)
        wrong_solid = _solve(inp, reverse_ctrl=True, gradient=False)
    except Exception:
        return None

    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    if bool(np.array_equal(wrong_forward, out)):
        return None
    if bool(np.array_equal(wrong_solid, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
