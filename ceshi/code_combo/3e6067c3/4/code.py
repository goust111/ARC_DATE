# Combo variant of 3e6067c3 (index 4)
# combo_spec: D1/1+D9/1
# Name: 形状匹配+图例式映射
#
# Design analysis for this fused variant
# a) Concrete behavior / trace:
#    - Original task: framed squares have plain fill colors, and one sparse control row
#      lists a traversal sequence by color. Consecutive control colors imply bridges.
#    - D1 change: matching key becomes the small motif shape inside each square instead
#      of the fill color. Bridge color still comes from the start square's fill.
#    - D9 change: the control area stops being one implicit sequence and becomes an
#      explicit legend where each entry names one bridge directly.
#    - Fused rule here:
#        * Each framed square contains a solid fill color plus a monochrome 3x3 motif.
#        * The control area below the square field is a 2-symbol-wide legend. Each
#          legend row contains two motifs: left = start-square shape, right = end-square
#          shape. Each row independently requests one bridge.
#        * Example trace: suppose one square is blue+L, another is red+T. A legend row
#          [L][T] means “find the L square and the T square, then draw the bridge in
#          blue”, regardless of the order of other legend rows.
#    - Difference from the original is therefore twofold and coupled: the solver must
#      first parse shape identity, then use the legend's explicit pairing semantics.
#
# b) Complexity calibration:
#    - Motifs come from a small fixed library of six visually distinct 4-connected
#      shapes. This keeps the visual vocabulary learnable while still making exact
#      shape matching necessary.
#    - Shapes are allowed to repeat globally, but never within the same lattice row or
#      lattice column. This preserves visual repetition without making pair lookup
#      ambiguous by default.
#    - Legend rows are shuffled away from the path order when possible, so the control
#      region reads as a lookup table instead of an implicit walk sequence.
#
# c) Implicit-assumption audit against the original solver:
#    - Original `len(u) == 1` for frame interiors is invalid now; each interior must be
#      parsed as `fill + marker-shape`.
#    - Original `by[inner_color]` is invalid; matching must index by `shape_key`.
#    - Original control parsing assumes one row of colored pixels and a persistent
#      current frame. Both are invalid under explicit legend rows.
#    - Original bridge color `a` equals the control token. That is invalid now because
#      control symbols are shapes, not colors. Bridge color must come from the selected
#      start frame's fill color.
#    - New fusion-specific risk: once shapes may repeat and rows are explicit, a legend
#      row could become ambiguous among multiple matching frame pairs. The generator
#      therefore validates that every legend row has exactly one best aligned frame pair
#      under the same scoring rule used by the solver.
#
# d) Placement legality:
#    - All framed squares lie on a 2-row lattice with uniform size and positive gaps.
#    - Bridges are only drawn between consecutive path squares, so every requested pair
#      shares a row band or column band and produces a non-empty bridge strip.
#    - Legend motifs are placed strictly below the square field on separate 3x3 canvases
#      with spacing that prevents accidental component merging.
#
# e) Visual observability:
#    - A human can see that legend entries are shape pairs rather than color pixels.
#    - Repeated shapes with different fill colors make “shape -> unique color” inference
#      invalid, so matching really must use motif identity plus row-wise pairing.
#    - Shuffled legend rows make “flatten controls into one sequence” visibly wrong.
#
# f) Visual cleanliness:
#    - Motifs always live on clean 3x3 canvases inside frames and inside the legend.
#    - The legend is isolated from the main field by a vertical gap and uses a fixed
#      two-symbol row layout, so it reads as a compact table.
#    - Bridge cells are validated to remain disjoint and never overlap the legend.
#
# g) Inference of changes:
#    - Colors have only structural roles: border, square fills / bridge paints, and one
#      shared motif color. No numeric color ordering is used anywhere.
#    - Because the same motif can appear on multiple differently colored squares, bridge
#      color is inferable only after identifying the correct start square for a legend
#      row.
#
# h) Variant-specific anti-degeneration target:
#    - Every generated puzzle must contain at least one repeated motif used on squares
#      with different fill colors, and the legend row order must differ from the natural
#      path-edge order when a shuffled order is feasible.
#
# i) New solver assumptions and generator guarantees:
#    - Assumption: each frame interior has exactly two colors and the non-fill color is a
#      single 4-connected motif. Guarantee: generator renders exactly that.
#    - Assumption: the control area below the frames consists only of separated motif
#      components arranged as two motifs per legend row. Guarantee: generator places
#      motifs on non-touching 3x3 canvases and leaves a blank gap below frames.
#    - Assumption: for each legend row, the intended start/end shapes identify exactly
#      one best aligned frame pair under `(aligned, nearest)` scoring. Guarantee:
#      generator checks and rejects ambiguous assignments before validation.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

ShapeKey = tuple[tuple[int, int], ...]

_DIGITS = [int(i) for i in range(10)]
_SHAPES: list[ShapeKey] = [
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
    ((0, 1), (1, 1), (2, 0), (2, 1)),  # mirrored L
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # Z
    ((0, 1), (0, 2), (1, 0), (1, 1)),  # S
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # 2x2 block
]


def _shape_key_from_coords(coords: np.ndarray) -> ShapeKey | None:
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(rel)


def _shape_key_from_mask(mask: np.ndarray) -> ShapeKey | None:
    return _shape_key_from_coords(np.argwhere(mask))


def _extract_frame_payload(inner: np.ndarray, bg: int, border: int) -> tuple[int, int, ShapeKey] | None:
    cnt = Counter(map(int, inner.flatten()))
    if len(cnt) != 2:
        return None
    fill_color, fill_n = cnt.most_common(1)[0]
    fill_color = int(fill_color)
    fill_n = int(fill_n)
    if fill_color in (int(bg), int(border)):
        return None
    marker_colors = [int(col) for col in cnt if int(col) != int(fill_color)]
    if len(marker_colors) != 1:
        return None
    marker_color = int(marker_colors[0])
    if marker_color in (int(bg), int(border), int(fill_color)):
        return None
    marker_mask = inner == int(marker_color)
    marker_n = int(marker_mask.sum())
    if marker_n <= 0 or marker_n >= int(inner.size) or fill_n <= marker_n:
        return None
    lab, n = label_mask(marker_mask, connectivity=4)
    if int(n) != 1:
        return None
    shape = _shape_key_from_mask(marker_mask)
    if shape is None or shape not in _SHAPES:
        return None
    return int(fill_color), int(marker_color), shape


def _detect_frames(g: np.ndarray, bg: int, border: int) -> list[dict]:
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
        if h < 5 or w < 5:
            continue

        t = None
        fill_color = None
        marker_color = None
        shape = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.shape != (3, 3):
                continue
            if bool((inner == int(border)).any()):
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

            payload = _extract_frame_payload(inner, int(bg), int(border))
            if payload is None:
                continue
            fill_color, marker_color, shape = payload
            t = int(tt)
            break

        if t is None or fill_color is None or marker_color is None or shape is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "fill": int(fill_color),
                "marker": int(marker_color),
                "shape": shape,
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            }
        )
    return frames


def _find_legend_pairs(g: np.ndarray, bg: int, min_row: int) -> list[tuple[ShapeKey, ShapeKey]]:
    H, W = g.shape
    ctrl_mask = np.zeros((int(H), int(W)), dtype=bool)
    ctrl_mask[int(min_row) :, :] = g[int(min_row) :, :] != int(bg)
    lab, n = label_mask(ctrl_mask, connectivity=4)

    rows: dict[int, list[tuple[int, ShapeKey]]] = defaultdict(list)
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        key = _shape_key_from_coords(pts)
        if key is None:
            continue
        r0, c0 = pts.min(axis=0)
        rows[int(r0)].append((int(c0), key))

    legend_pairs: list[tuple[ShapeKey, ShapeKey]] = []
    for r0 in sorted(rows):
        comps = sorted(rows[int(r0)], key=lambda item: int(item[0]))
        assert len(comps) == 2
        legend_pairs.append((comps[0][1], comps[1][1]))
    assert len(legend_pairs) >= 2
    return legend_pairs


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    H, W = out.shape
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


def _best_pair(start_frames: list[dict], end_frames: list[dict]) -> tuple[tuple[int, int] | None, tuple[dict, dict] | None, bool]:
    best_score = None
    best_pair = None
    best_count = 0
    for fa in start_frames:
        for fb in end_frames:
            if fa is fb:
                continue
            sr, sc = _aligned(fa, fb)
            aligned = 1 if (sr or sc) else 0
            (r1_, c1_) = _center(fa)
            (r2_, c2_) = _center(fb)
            dist = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            score = (int(aligned), -int(dist))
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (fa, fb)
                best_count = 1
            elif score == best_score:
                best_count += 1
    return best_score, best_pair, bool(best_count == 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via explicit legend rows of motif pairs.

    Input structure:
    - Equal-sized border frames on a lattice.
    - Each frame interior is a fill color plus a small monochrome motif.
    - Below the square field is a 2-symbol-wide legend; each row is `(start_shape, end_shape)`.

    Rule:
    - Match legend motifs to frame motifs by exact shape (no rotation / flip).
    - For each legend row, select the uniquely best aligned start/end frame pair using
      the original puzzle's row/column-band + nearest heuristic.
    - Draw that bridge using the start frame's fill color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert border is not None

    frames = _detect_frames(g, int(bg), int(border))
    assert frames

    by_shape: dict[ShapeKey, list[dict]] = defaultdict(list)
    for f in frames:
        by_shape[f["shape"]].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    legend_pairs = _find_legend_pairs(g, int(bg), int(max_frame_r + 1))

    out = g.copy()
    for start_shape, end_shape in legend_pairs:
        assert start_shape in by_shape and end_shape in by_shape
        score, pair, unique = _best_pair(by_shape[start_shape], by_shape[end_shape])
        assert score is not None and pair is not None and unique and int(score[0]) == 1
        fa, fb = pair
        _draw_bridge(out, fa, fb, int(fa["fill"]))

    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _render_shape(g: np.ndarray, *, top: int, left: int, shape: ShapeKey, color: int) -> None:
    for dr, dc in shape:
        g[int(top + dr), int(left + dc)] = int(color)


def _draw_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    t: int,
    border: int,
    fill_color: int,
    marker_color: int,
    shape: ShapeKey,
) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    inner_r0 = int(r0 + t)
    inner_c0 = int(c0 + t)
    g[int(inner_r0) : int(r0 + outer - t), int(inner_c0) : int(c0 + outer - t)] = int(fill_color)
    _render_shape(g, top=int(inner_r0), left=int(inner_c0), shape=shape, color=int(marker_color))


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


def _assign_shapes(path: list[tuple[int, int]], frames: list[dict], pair_slots: list[tuple[tuple[int, int], tuple[int, int]]]) -> dict[tuple[int, int], ShapeKey] | None:
    slot_to_frame = {tuple(map(int, f["slot"])): f for f in frames}
    neighbors_on_path: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for a_slot, b_slot in pair_slots:
        neighbors_on_path[tuple(map(int, a_slot))].add(tuple(map(int, b_slot)))
        neighbors_on_path[tuple(map(int, b_slot))].add(tuple(map(int, a_slot)))

    repeat_pairs = [
        (tuple(map(int, path[i])), tuple(map(int, path[j])))
        for i in range(int(len(path)))
        for j in range(i + 1, int(len(path)))
        if int(path[i][0]) != int(path[j][0]) and int(path[i][1]) != int(path[j][1])
    ]
    if not repeat_pairs:
        return None

    for _ in range(120):
        used_in_row: dict[int, set[ShapeKey]] = defaultdict(set)
        used_in_col: dict[int, set[ShapeKey]] = defaultdict(set)
        shape_by_slot: dict[tuple[int, int], ShapeKey] = {}

        pair_a, pair_b = repeat_pairs[int(np.random.randint(0, len(repeat_pairs)))]
        repeat_shape = _SHAPES[int(np.random.randint(0, len(_SHAPES)))]
        for rr, cc in (pair_a, pair_b):
            rr = int(rr)
            cc = int(cc)
            shape_by_slot[(int(rr), int(cc))] = repeat_shape
            used_in_row[int(rr)].add(repeat_shape)
            used_in_col[int(cc)].add(repeat_shape)

        ok = True
        for rr, cc in path:
            rr = int(rr)
            cc = int(cc)
            slot = (int(rr), int(cc))
            if slot in shape_by_slot:
                continue
            forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)])
            forbid |= {shape_by_slot[nbr] for nbr in neighbors_on_path[slot] if nbr in shape_by_slot}
            choices = [shape for shape in _SHAPES if shape not in forbid]
            if not choices:
                ok = False
                break
            shape = choices[int(np.random.randint(0, len(choices)))]
            shape_by_slot[slot] = shape
            used_in_row[int(rr)].add(shape)
            used_in_col[int(cc)].add(shape)
        if not ok:
            continue

        counts = Counter(shape_by_slot.values())
        if int(len(counts)) < 3 or max(int(v) for v in counts.values()) < 2:
            continue
        if any(shape_by_slot[a_slot] == shape_by_slot[b_slot] for a_slot, b_slot in pair_slots):
            continue

        frames_with_shape = []
        by_shape: dict[ShapeKey, list[dict]] = defaultdict(list)
        for f in frames:
            rr, cc = map(int, f["slot"])
            f2 = dict(f)
            f2["shape"] = shape_by_slot[(int(rr), int(cc))]
            frames_with_shape.append(f2)
            by_shape[f2["shape"]].append(f2)
        by_slot = {tuple(map(int, f["slot"])): f for f in frames_with_shape}

        resolved = True
        for a_slot, b_slot in pair_slots:
            a_slot = tuple(map(int, a_slot))
            b_slot = tuple(map(int, b_slot))
            score, pair, unique = _best_pair(by_shape[shape_by_slot[a_slot]], by_shape[shape_by_slot[b_slot]])
            if not unique or score is None or int(score[0]) != 1 or pair is None:
                resolved = False
                break
            if tuple(map(int, pair[0]["slot"])) != a_slot or tuple(map(int, pair[1]["slot"])) != b_slot:
                resolved = False
                break
            if not _bridge_cells(by_slot[a_slot], by_slot[b_slot]):
                resolved = False
                break
        if not resolved:
            continue

        used_shape_slots = {
            slot
            for slot in path
            if int(counts[shape_by_slot[tuple(map(int, slot))]]) >= 2
        }
        if not used_shape_slots:
            continue

        return shape_by_slot
    return None


def _shuffle_pairs(pair_slots: list[tuple[tuple[int, int], tuple[int, int]]], shape_by_slot: dict[tuple[int, int], ShapeKey]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    original = [
        (shape_by_slot[tuple(map(int, a_slot))], shape_by_slot[tuple(map(int, b_slot))])
        for a_slot, b_slot in pair_slots
    ]
    idx = np.arange(int(len(pair_slots)), dtype=int)
    for _ in range(60):
        perm = np.random.permutation(idx)
        shuffled = [pair_slots[int(i)] for i in perm]
        shuffled_shapes = [
            (shape_by_slot[tuple(map(int, a_slot))], shape_by_slot[tuple(map(int, b_slot))])
            for a_slot, b_slot in shuffled
        ]
        if shuffled_shapes != original:
            return shuffled
    return pair_slots[:]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border_choices = [d for d in _DIGITS if int(d) != int(bg)]
    border = int(np.random.choice(np.array(border_choices, dtype=int)))
    marker_choices = [d for d in _DIGITS if int(d) not in (int(bg), int(border))]
    marker = int(np.random.choice(np.array(marker_choices, dtype=int)))
    fill_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(marker))]
    if len(fill_palette) < 5:
        return None

    t = 1
    inner = 3
    outer = int(inner + 2 * t)

    R = 2
    C = int(np.random.randint(3, 5))
    max_frames = int(R * C)
    n_frames = int(np.random.randint(4, int(min(5, max_frames, len(fill_palette))) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
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
        if len({int(r) for r, _c in p}) < 2 or len({int(c) for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    row_gap = [int(np.random.randint(1, 4))]
    col_gap = [int(np.random.randint(1, 4)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))

    r0s = [int(top_margin), int(top_margin + outer + row_gap[0])]
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
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

    pair_slots = list(zip(path, path[1:]))
    if len(pair_slots) < 2:
        return None

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    legend_gap = int(np.random.randint(2, 4))
    legend_step = 4
    legend_r0 = int(max_r1 + legend_gap)
    legend_c0 = int(np.random.randint(1, 4))
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 4))
    H = int(legend_r0 + 3 + legend_step * (len(pair_slots) - 1) + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(legend_c0 + 7 + right_margin)))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    shape_by_slot = _assign_shapes(path, frames, pair_slots)
    if shape_by_slot is None:
        return None

    fill_choices = np.random.choice(np.array(fill_palette, dtype=int), size=int(len(path)), replace=False)
    fill_by_slot = {tuple(map(int, slot)): int(fill_choices[i]) for i, slot in enumerate(path)}

    counts = Counter(shape_by_slot.values())
    repeated_shapes = [shape for shape, n in counts.items() if int(n) >= 2]
    if not repeated_shapes:
        return None
    if not any(
        len({int(fill_by_slot[slot]) for slot in path if shape_by_slot[tuple(map(int, slot))] == shape}) >= 2
        for shape in repeated_shapes
    ):
        return None

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
            fill_color=int(fill_by_slot[(int(rr), int(cc))]),
            marker_color=int(marker),
            shape=shape_by_slot[(int(rr), int(cc))],
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in pair_slots:
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
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
        paint = int(fill_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    legend_pairs = _shuffle_pairs(pair_slots, shape_by_slot)
    for i, (a_slot, b_slot) in enumerate(legend_pairs):
        top = int(legend_r0 + i * legend_step)
        _render_shape(out, top=int(top), left=int(legend_c0), shape=shape_by_slot[(int(a_slot[0]), int(a_slot[1]))], color=int(marker))
        _render_shape(out, top=int(top), left=int(legend_c0 + 4), shape=shape_by_slot[(int(b_slot[0]), int(b_slot[1]))], color=int(marker))

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
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
