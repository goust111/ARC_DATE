# Variant of 3e6067c3
# Dimension: D4 — 目标对象：带框方块的结构 (dst)
# Change: 等尺寸的带框方块（外层边框色 + 内层纯色填充），排列在网格状的格点上... → 自由排列的方块

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D4 variant
# a) Concrete behavior:
#    Original samples place equal-sized framed squares on a latent row/column lattice.
#    The solver can therefore treat "same row band" as exact equality of the inner
#    row interval and "same column band" as exact equality of the inner column interval.
#    In this variant the squares are still equal-sized framed blocks, but they are
#    placed freely: a square may be shifted up/down (or left/right) relative to the
#    previous one as long as their inner regions still overlap in a visible strip.
#    Example: with 5x5 squares and 3x3 interiors, two horizontally connected squares
#    might have inner row ranges 4..6 and 5..7. The bridge must then occupy only the
#    overlap rows 5..6. Original logic would reject this pair because 4..6 != 5..7;
#    the variant must instead detect the overlap and draw the thinner bridge.
# b) Complexity calibration:
#    The freedom is limited to orthogonal offsets smaller than the inner size, so the
#    bridge remains a clear rectangular strip. This keeps the puzzle legible while
#    making the "discover alignment from geometry" step visible. We also require at
#    least one partially-overlapping bridge per generated sample so the variant does
#    not collapse back to the original exact-band case.
# c) Old implicit assumptions that break:
#    - aligned(): exact equality of inner row/col intervals no longer holds.
#    - draw_bridge(): bridge thickness can no longer be copied from one endpoint alone;
#      it must use the actual overlap strip between the two interiors.
#    - target selection: choosing a non-aligned same-color square and then silently
#      drawing nothing was harmless before because generator guaranteed exact bands.
#      In the variant main() filters to geometrically bridgeable candidates only.
# d) Placement legality:
#    Generator places squares sequentially along the intended path. Each new square is
#    horizontally or vertically separated from the previous one, but shifted on the
#    perpendicular axis by less than the inner size so the interior strips overlap.
#    Every new frame and every new bridge is checked against all existing frames and
#    bridges to avoid overlap and out-of-bounds issues after normalization.
# e) Visual observability:
#    Humans can see that output bridges follow the bottom control colors, but some
#    bridges are thinner or vertically offset because the squares are not on a regular
#    grid. This difference is visible in 2-3 examples and does not depend on color ids.
# f) Visual cleanliness:
#    Only rectangular bridges are drawn, with at least one background cell gap between
#    squares. The generator rejects crowded layouts and keeps a bottom control row with
#    the same simple structure as the original task.
# g) Color inference:
#    Colors are kept simple: path squares use distinct inner colors so the only new
#    reasoning burden is geometric alignment, which is the intended D4 change.
# h) Variant-specific validation:
#    Generated samples are required to contain at least one bridge whose overlap width
#    is smaller than the full inner size. On such inputs the original exact-band solver
#    would produce a different result, proving the variant behavior is active.
# i) New assumptions introduced by main():
#    - A valid next target must have a non-empty interior overlap strip with the
#      current square and a positive gap on the travel axis.
#    - If multiple same-color targets are bridgeable, larger overlap and shorter
#      distance break ties. The generator guarantees the intended path satisfies these
#      assumptions unambiguously.


def _detect_frames(g: np.ndarray, border: int) -> list[dict]:
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
            u = set(map(int, inner.flatten()))
            if len(u) != 1:
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
            inner_col = int(next(iter(u)))
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


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    if int(lo) > int(hi):
        return None
    return int(lo), int(hi)


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _bridge_info(fa: dict, fb: dict) -> dict | None:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]

    row_overlap = _interval_overlap(int(ia[0]), int(ia[2]), int(ib[0]), int(ib[2]))
    if row_overlap is not None:
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        elif int(c1b) < int(c0a):
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        else:
            cc0, cc1 = 1, 0
        if int(cc0) <= int(cc1):
            rr0, rr1 = row_overlap
            return {
                "axis": "h",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "overlap": int(rr1 - rr0 + 1),
                "gap": int(cc1 - cc0 + 1),
            }

    col_overlap = _interval_overlap(int(ia[1]), int(ia[3]), int(ib[1]), int(ib[3]))
    if col_overlap is not None:
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        elif int(r1b) < int(r0a):
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        else:
            rr0, rr1 = 1, 0
        if int(rr0) <= int(rr1):
            cc0, cc1 = col_overlap
            return {
                "axis": "v",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "overlap": int(cc1 - cc0 + 1),
                "gap": int(rr1 - rr0 + 1),
            }

    return None


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    info = _bridge_info(fa, fb)
    if info is None:
        return set()
    return {
        (int(r), int(c))
        for r in range(int(info["rr0"]), int(info["rr1"]) + 1)
        for c in range(int(info["cc0"]), int(info["cc1"]) + 1)
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares via colored bridges following a control-pixel sequence.

    D4 variant:
    The framed squares are no longer locked to exact row/column lattice bands.
    A bridge is valid whenever two square interiors overlap on one axis and the
    squares are separated on the other axis, so the overlap strip defines the
    bridge thickness.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert not (border is None)

    frames = _detect_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

    def score_link(src: dict, dst: dict) -> tuple[int, int, int]:
        info = _bridge_info(src, dst)
        if info is None:
            return (0, 0, -10**6)
        r1_, c1_ = _center(src)
        r2_, c2_ = _center(dst)
        d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
        return (1, int(info["overlap"]), -int(d))

    start_color = int(ctrl[0])
    assert not (start_color not in by)
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
        best = None
        bestf = None
        for f in start_cands:
            if nxt_cands:
                local_best = max(score_link(f, g2) for g2 in nxt_cands)
            else:
                r, c = _center(f)
                local_best = (0, 0, -(int(r) + int(c)))
            if best is None or local_best > best:
                best = local_best
                bestf = f
        assert not (bestf is None)
        cur = bestf

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        info = _bridge_info(fa, fb)
        if info is None:
            return
        for r in range(int(info["rr0"]), int(info["rr1"]) + 1):
            for c in range(int(info["cc0"]), int(info["cc1"]) + 1):
                if 0 <= r < H and 0 <= c < W:
                    out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        cands = by[b]
        bridgeable = [f for f in cands if _bridge_info(cur, f) is not None]
        if not bridgeable:
            continue
        nxt = max(bridgeable, key=lambda f: score_link(cur, f))
        draw_bridge(cur, nxt, int(a))
        cur = nxt

    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _boxes_too_close(a: dict, b: dict, pad: int = 0) -> bool:
    r0a, c0a, r1a, c1a = map(int, a["bbox"])
    r0b, c0b, r1b, c1b = map(int, b["bbox"])
    return not (
        int(r1a + pad) < int(r0b - pad)
        or int(r1b + pad) < int(r0a - pad)
        or int(c1a + pad) < int(c0b - pad)
        or int(c1b + pad) < int(c0a - pad)
    )


def _make_frame(r0: int, c0: int, outer: int, t: int) -> dict:
    r1 = int(r0 + outer - 1)
    c1 = int(c0 + outer - 1)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
    }


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(2, 5))
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 8):
        return None

    n_frames = int(np.random.randint(4, min(7, len(inner_palette)) + 1))
    frames: list[dict] | None = None
    bridge_cells_all: set[tuple[int, int]] = set()
    partial_edges = 0

    for _layout_try in range(300):
        cur_frames = [_make_frame(0, 0, int(outer), int(t))]
        cur_bridges: set[tuple[int, int]] = set()
        used_horizontal = False
        used_vertical = False
        partial = 0
        ok = True

        for _step in range(int(n_frames) - 1):
            prev = cur_frames[-1]
            placed = False
            for _cand_try in range(200):
                axis = str(np.random.choice(np.array(["h", "v"], dtype=object)))
                side = int(np.random.choice(np.array([-1, 1], dtype=int)))
                gap = int(np.random.randint(1, 6))
                max_shift = int(inner - 1)
                shift = int(np.random.randint(-max_shift, max_shift + 1))
                if max_shift > 0 and np.random.rand() < 0.7:
                    nonzero = [d for d in range(-max_shift, max_shift + 1) if d != 0]
                    shift = int(np.random.choice(np.array(nonzero, dtype=int)))

                pr0, pc0, pr1, pc1 = map(int, prev["bbox"])
                if axis == "h":
                    nr0 = int(pr0 + shift)
                    nc0 = int(pc1 + gap + 1) if int(side) > 0 else int(pc0 - gap - outer)
                    used_horizontal = True
                else:
                    nr0 = int(pr1 + gap + 1) if int(side) > 0 else int(pr0 - gap - outer)
                    nc0 = int(pc0 + shift)
                    used_vertical = True

                cand = _make_frame(int(nr0), int(nc0), int(outer), int(t))
                if any(_boxes_too_close(cand, old, pad=1) for old in cur_frames):
                    continue

                cells = _bridge_cells(prev, cand)
                if not cells:
                    continue
                if cells & cur_bridges:
                    continue
                hit_frame = False
                for old in cur_frames:
                    r0, c0, r1, c1 = map(int, old["bbox"])
                    for r, c in cells:
                        if int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1):
                            hit_frame = True
                            break
                    if hit_frame:
                        break
                if hit_frame:
                    continue

                cur_frames.append(cand)
                cur_bridges |= set(cells)
                if axis == "h" and cand["inner_bbox"][0] != prev["inner_bbox"][0]:
                    partial += 1
                if axis == "v" and cand["inner_bbox"][1] != prev["inner_bbox"][1]:
                    partial += 1
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        has_h = any(_bridge_info(a, b)["axis"] == "h" for a, b in zip(cur_frames, cur_frames[1:]))
        has_v = any(_bridge_info(a, b)["axis"] == "v" for a, b in zip(cur_frames, cur_frames[1:]))
        if not has_h or not has_v or int(partial) < 1:
            continue

        frames = cur_frames
        bridge_cells_all = cur_bridges
        partial_edges = int(partial)
        break

    if frames is None:
        return None

    min_r0 = min(int(f["bbox"][0]) for f in frames)
    min_c0 = min(int(f["bbox"][1]) for f in frames)
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))
    dr = int(top_margin - min_r0)
    dc = int(left_margin - min_c0)

    shifted_frames: list[dict] = []
    for f in frames:
        r0, c0, r1, c1 = map(int, f["bbox"])
        sf = {
            "bbox": (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)),
            "inner_bbox": (
                int(f["inner_bbox"][0] + dr),
                int(f["inner_bbox"][1] + dc),
                int(f["inner_bbox"][2] + dr),
                int(f["inner_bbox"][3] + dc),
            ),
        }
        shifted_frames.append(sf)

    shifted_bridges = {(int(r + dr), int(c + dc)) for r, c in bridge_cells_all}

    max_r1 = max(int(f["bbox"][2]) for f in shifted_frames)
    max_c1 = max(int(f["bbox"][3]) for f in shifted_frames)

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    ctrl_cols = [int(1 + 2 * i) for i in range(int(n_frames))]
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, ctrl_cols[-1] + 2))
    if not (8 <= int(H) <= 30 and 8 <= int(W) <= 30):
        return None

    path_colors = list(map(int, np.random.choice(np.array(inner_palette, dtype=int), size=int(n_frames), replace=False)))
    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f, inner_col in zip(shifted_frames, path_colors):
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_col=int(inner_col))

    for (fa, fb), paint in zip(zip(shifted_frames, shifted_frames[1:]), path_colors):
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(paint)

    for col, cc in zip(path_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

    inp = out.copy()
    for r, c in shifted_bridges:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    partial_after_shift = 0
    for fa, fb in zip(shifted_frames, shifted_frames[1:]):
        info = _bridge_info(fa, fb)
        if info is None:
            return None
        inner_h = int(fa["inner_bbox"][2] - fa["inner_bbox"][0] + 1)
        inner_w = int(fa["inner_bbox"][3] - fa["inner_bbox"][1] + 1)
        if (info["axis"] == "h" and int(info["overlap"]) < int(inner_h)) or (info["axis"] == "v" and int(info["overlap"]) < int(inner_w)):
            partial_after_shift += 1
    if int(partial_after_shift) < 1 or int(partial_edges) < 1:
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
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
