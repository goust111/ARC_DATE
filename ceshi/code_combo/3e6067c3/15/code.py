# Combo variant of 3e6067c3 (index 15)
# combo_spec: D6/1+D7/0
# Name: 擦除控制+紧凑裁剪

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

"""
Design analysis for combo D6/1 + D7/0 on task 3e6067c3.

a) Concrete behavior and trace
   - Source task:
     framed equal-sized squares appear in the upper scene; a sparse control row below them
     lists inner colors left-to-right; the solver connects consecutive control colors with a
     bridge whose color equals the source square's inner color; the original output keeps the
     control row and all margins.
   - Combo behavior:
     first solve the same bridge-routing problem on the full input canvas, then erase the
     entire control row back to background, then tight-crop to the minimal bbox of all
     remaining non-background pixels.
   - Tiny trace:
     input = [top/left margins][framed squares][gap][control row][bottom margin]
     source output = same canvas + bridges + unchanged control row
     combo output = same bridges, but control row is blanked before crop, so the returned grid
     contains only the completed square/bridge scene.
   - Important spec resolution:
     the prompt text for the per-dimension reference lists a conflicting D6/1 description
     ("visited squares recolor"), but this task's own dimension-analysis YAML says D6/1 is
     "连接后擦除控制像素". The combo name and fusion description match the YAML, so this file
     treats control-row erasure as the authoritative D6 behavior.

b) Complexity calibration
   - Tight crop alone could visually dominate the combo. To keep the bridge rule observable,
     the generator preserves at least one non-path decoy square, so the cropped output still
     contains both connected and unconnected framed objects.
   - To keep the D6 step from becoming an inert hidden implementation detail, generated control
     pixels are forced to extend horizontally outside the playfield bbox. Therefore cropping the
     solved canvas before erasing control would yield a wider rectangle than the true combo
     output; the erasure step genuinely affects the crop result.
   - Path length stays in a small range and uses unique control colors, avoiding ambiguous
     color-to-square matching while still requiring the bridge geometry to be inferred.

c) Hidden-assumption audit of main()
   - Border detection still assumes one dominant shared frame color in the input. The generator
     keeps every square framed in the input and validates the inferred border color.
   - Frame detection via connected components assumes bridges are absent in the input. The
     generator still builds input by erasing all bridge cells from the solved canvas.
   - Control-row detection assumes the bottom-most row below all frames with >=2 non-bg cells is
     the instruction row. The generator keeps exactly one such row below the scene.
   - Bridge drawing assumes every chosen consecutive pair is aligned in the same row-band or
     column-band. The generator constructs the path as 4-neighbor-adjacent lattice steps, so
     every consecutive pair is aligned and separated only by legal background gap cells.
   - Tight crop no longer depends on "rows above ctrl_r". Instead, after explicit D6 erasure,
     crop is a generic bbox over all non-background cells. This matters because D6 is now the
     mechanism that removes the control row from the crop domain.

d) Placement legality
   - Frames sit on a lattice with at least one background gap between neighboring slots.
   - Path steps are 4-neighbor-adjacent lattice moves, so each bridge occupies only the open gap
     between two neighboring frames.
   - Decoy squares are sampled from unused lattice slots, so they cannot overlap path frames or
     bridge corridors.
   - Margins and control-row placement are checked against the 30x30 ARC limit.

e) Visual observability
   - Across examples a human sees: input has a detached control row; output has the same framed
     squares plus bridges, but the bottom legend disappears and the whole scene is zoomed tightly.
   - Because at least one decoy square survives in the crop, the bridge topology itself explains
     which squares were on the traversed path; this avoids the output collapsing into a trivial
     "all squares connected" picture.

f) Visual cleanliness
   - No extra marks are added on top of the squares. The only output layers are framed squares
     and rectangular bridges in the original color semantics.
   - Erasing the control row before crop removes clutter rather than adding another annotation.

g) Change inferability and anti-degeneracy metric
   - No new color remapping is introduced: border stays border, inner fill stays inner fill, and
     each bridge still uses the source square's inner color.
   - For generator auditing we measure:
       naive_crop = tight crop of the solved canvas before erasing the control row
       combo_crop = tight crop after erasing the control row
       metric = (naive_h - combo_h, naive_w - combo_w)
     Valid combo samples require both deltas to be positive, proving that D6 erasure changes the
     crop result in both height and width instead of being redundant.

h) New solver assumptions that the generator guarantees
   - There is at least one decoy square, so not every square is necessarily bridged.
   - Every control color corresponds to exactly one path square.
   - The playfield has positive top/left/right margin and a detached control row, so the final
     crop is strictly smaller than the input in both dimensions.
"""


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

        thickness = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            inner_set = set(map(int, inner.flatten()))
            if len(inner_set) != 1:
                continue

            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                continue
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                continue
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                continue
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                continue

            thickness = int(tt)
            inner_col = int(next(iter(inner_set)))
            break

        if thickness is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(thickness),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + thickness), int(c0 + thickness), int(r1 - thickness), int(c1 - thickness)),
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
        same_row, same_col = _aligned(cur, frame)
        aligned_score = 1 if (same_row or same_col) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            bestf = frame
    return bestf


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    h, w = out.shape
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


def _solve_canvas(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for color, _count in cnt.most_common():
        if int(color) != int(bg):
            border = int(color)
            break
    assert border is not None

    frames = _find_frames(g, int(border))
    assert frames

    by_inner: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_inner[int(frame["inner"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(h - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(w)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert start_color in by_inner
    start_cands = by_inner[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by_inner.get(nxt_color, [])
        best = None
        bestf = None
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
                    bestf = frame
        cur = bestf

    assert cur is not None
    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        src_color = int(ctrl[i])
        dst_color = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(src_color)
        if dst_color not in by_inner:
            continue
        nxt = _choose_candidate(cur, by_inner[dst_color])
        if nxt is None:
            continue
        _draw_bridge(out, cur, nxt, int(src_color))
        cur = nxt

    return out, int(bg), int(ctrl_r)


def inspect_fusion_behavior(grid: np.ndarray) -> dict[str, int]:
    """
    Return a small audit summary showing whether control-row erasure changes the crop.
    """
    solved, bg, ctrl_r = _solve_canvas(np.array(grid, dtype=int))
    naive = _tight_crop_nonbg(solved, int(bg))
    erased = solved.copy()
    erased[int(ctrl_r), :] = int(bg)
    fused = _tight_crop_nonbg(erased, int(bg))
    return {
        "height_gain": int(naive.shape[0] - fused.shape[0]),
        "width_gain": int(naive.shape[1] - fused.shape[1]),
        "control_pixels": int((solved[int(ctrl_r)] != int(bg)).sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the original bridges, erase the control row, then tight-crop the remaining scene.
    """
    solved, bg, ctrl_r = _solve_canvas(np.array(grid, dtype=int))
    solved[int(ctrl_r), :] = int(bg)
    return _tight_crop_nonbg(solved, int(bg))


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for color, _count in cnt.most_common():
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
    if len(inner_palette) < 5:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    r_slots = int(np.random.randint(2, 6))
    c_slots = int(np.random.randint(2, 6))
    max_frames = int(r_slots * c_slots)
    if max_frames < 5:
        return None

    max_path = int(min(7, max_frames - 1, len(inner_palette) - 1))
    if max_path < 4:
        return None
    n_path = int(np.random.randint(4, max_path + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        cur_path = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for _step in range(int(n_path) - 1):
            r, c = map(int, cur_path[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(r_slots), int(c_slots)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            cur_path.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in cur_path}) < 2 or len({c for _r, c in cur_path}) < 2:
            continue
        path = cur_path
        break
    if path is None:
        return None

    spare_slots = [(int(r), int(c)) for r in range(int(r_slots)) for c in range(int(c_slots)) if (int(r), int(c)) not in set(path)]
    spare_colors = int(len(inner_palette) - n_path)
    max_decoys = int(min(3, len(spare_slots), spare_colors))
    if max_decoys < 1:
        return None
    n_decoys = int(np.random.randint(1, max_decoys + 1))
    decoy_idx = np.random.choice(len(spare_slots), size=int(n_decoys), replace=False)
    decoy_slots = [spare_slots[int(i)] for i in np.array(decoy_idx).tolist()]

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(r_slots) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(c_slots) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(2, 5))

    r0s = [int(top_margin)]
    for k in range(int(r_slots) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(c_slots) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    all_slots = list(path) + list(decoy_slots)
    slot_to_frame: dict[tuple[int, int], dict] = {}
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
    bottom_margin = int(np.random.randint(1, 3))
    right_margin = int(np.random.randint(1, 5))
    ctrl_cols = [int(1 + 2 * i) for i in range(int(n_path))]
    h = int(ctrl_r + 1 + bottom_margin)
    w = int(max(max_c1 + 1 + right_margin, ctrl_cols[-1] + 2))
    if not (7 <= int(h) <= 30 and 7 <= int(w) <= 30):
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

    inp = np.full((int(h), int(w)), int(bg), dtype=int)
    full_out = np.full((int(h), int(w)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        inner_col = int(colors_by_slot[(int(rr), int(cc))])
        _draw_frame(inp, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))
        _draw_frame(full_out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))

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

    for color, col in zip(path_colors, ctrl_cols):
        full_out[int(ctrl_r), int(col)] = int(color)
        inp[int(ctrl_r), int(col)] = int(color)

    erased = full_out.copy()
    erased[int(ctrl_r), :] = int(bg)
    expected = _tight_crop_nonbg(erased, int(bg))
    naive = _tight_crop_nonbg(full_out, int(bg))
    if not (naive.shape[0] > expected.shape[0] and naive.shape[1] > expected.shape[1]):
        return None
    if not (expected.shape[0] < inp.shape[0] and expected.shape[1] < inp.shape[1]):
        return None

    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
        audit = inspect_fusion_behavior(inp)
    except Exception:
        return None
    if pred.shape != expected.shape:
        return None
    if not bool(np.array_equal(pred, expected)):
        return None
    if not (int(audit["height_gain"]) > 0 and int(audit["width_gain"]) > 0):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
