# Combo variant of 4c7dc4dd (index 20)
# combo_spec: D4/0+D6/2+D11/0+D19/0
# Name: 缺口定向空白框填充
#
# Design analysis:
# a) Concrete behavior and trace
#    - The original task had 4 complete hollow frames; the solver cropped the transformed test
#      interior and returned only that small patch.
#    - Here every frame is a broken frame: exactly one side has a contiguous 1-2 cell gap, with
#      the corners preserved. The gap side is no longer incidental: all frames in one sample share
#      the same gap-facing direction, and that direction matches the source->destination flow.
#    - The scene is organized as two adjacent directed pairs on the same axis:
#      one training pair and one test->blank pair. Example: if the global direction is rightward,
#      then both training frames and both test/blank frames have right-edge gaps, and a thin arrow
#      in each corridor also points right. The top pair demonstrates A->B; the other pair says
#      "apply the same transform from the test frame into the blank frame."
#    - So the original pipeline "find complete frames -> guess A/B direction both ways -> return
#      transformed crop" becomes "find gapped frames -> decode direction from both gap side and
#      corridor arrow -> transform the test sprite -> write it into the blank frame -> return the
#      whole 30x30 grid."
# b) Complexity calibration
#    - I keep the six original inner transformation families unchanged so the core task remains an
#      analogy over sprites rather than becoming a new symbolic language.
#    - I avoid excessive freedom in the directional encoding: per sample there is one global axis
#      (horizontal or vertical) and one global direction on that axis. This makes the new behavior
#      visible across all four frames instead of hiding it inside a single object.
#    - The arrow and the frame gaps encode the same direction redundantly. That redundancy is
#      deliberate: it prevents ambiguity from noisy backgrounds and makes the fusion human-readable.
# c) Implicit assumptions audit
#    - Original frame detection assumed every border edge was complete; that is false now, so the
#      detector must allow one contiguous 1-2 cell hole on exactly one side.
#    - Original direction inference tried both training orders and picked a heuristic winner; that
#      assumption is removed. The new solver expects the pair direction to be explicitly stated by
#      both the gap side and the arrow.
#    - Original output logic discarded the scene geometry after inference. That is invalid here,
#      because the blank frame is the destination carrier and the output must preserve the full grid.
#    - The new solver assumes the test frame and the blank frame also form an adjacent directed
#      pair on the same axis. The generator enforces that layout.
# d) Placement legality
#    - Frames are placed on a 2x2 structured layout with minimum corridor width, so arrows fit
#      cleanly between paired frames without overlap or boundary issues.
#    - Drawing order is background -> frame squares cleared to 0 -> frame borders -> border gap
#      erased -> interior sprite pasted -> corridor arrows drawn. This keeps each layer separable.
# e) Visual observability
#    - A human sees four obviously broken frames, two arrows, and one empty destination frame.
#      Across input/output examples the only changed area is the blank frame interior, so the
#      semantics "fill the empty directed destination" remains legible.
# f) Visual cleanliness
#    - The marker color is unique and absent from background, border, and sprites.
#    - Gap cells expose the black frame interior rather than noisy background, so the missing side
#      reads as part of the frame object rather than accidental corruption.
# g) Color/change inferability
#    - No extra color mapping is introduced by the fusion. Colors inside sprites still obey the
#      original rule families, while direction is conveyed structurally by gap orientation and arrow
#      placement instead of by hidden numeric color identities.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, draw_line, paste


GAP_SIDES = ("top", "right", "bottom", "left")
OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DIR_TO_GAP = {"right": "right", "left": "left", "down": "bottom", "up": "top"}
GAP_TO_DIR = {"right": "right", "left": "left", "bottom": "down", "top": "up"}


def _edge_gap_info(edge: np.ndarray) -> tuple[int, int] | str | None:
    """Return 'full' for a complete edge, (start,width) for a valid gap, else None."""
    if edge[0] == 0 or edge[-1] == 0:
        return None
    missing = np.where(edge == 0)[0]
    if missing.size == 0:
        return "full"
    if missing.size not in (1, 2):
        return None
    if missing[0] == 0 or missing[-1] == len(edge) - 1:
        return None
    if int(missing[-1] - missing[0] + 1) != int(missing.size):
        return None
    return int(missing[0]), int(missing.size)


def find_complete_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if col == 0:
                    continue
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                cands.append((size, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
    return picked


def find_gap_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, str, int, int]]:
    """
    Detect near-square frames whose border is complete except for one contiguous 1-2 cell gap.

    Returns tuples:
    (size, color, r0, c0, r1, c1, gap_side, gap_start_along_side, gap_width)
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if col == 0:
                    continue
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                edges = {
                    "top": (g[r0, c0 : c1 + 1] == col).astype(int),
                    "right": (g[r0 : r1 + 1, c1] == col).astype(int),
                    "bottom": (g[r1, c0 : c1 + 1] == col).astype(int),
                    "left": (g[r0 : r1 + 1, c0] == col).astype(int),
                }
                gap_info = []
                ok = True
                for side in GAP_SIDES:
                    info = _edge_gap_info(edges[side])
                    if info is None:
                        ok = False
                        break
                    if info != "full":
                        gap_info.append((side, info[0], info[1]))
                if not ok or len(gap_info) != 1:
                    continue
                gap_side, gap_start, gap_width = gap_info[0]
                cands.append((size, col, r0, c0, r1, c1, gap_side, gap_start, gap_width))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1, _, _, _ = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1, _, _, _ in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
        if len(picked) >= 6:
            break
    return picked


def stable_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    st = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            st.add(c)
    return st


def pick_var_color(x: np.ndarray, st: set[int]) -> int | None:
    vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
    if not vals:
        return None
    best = None
    for v in vals:
        cnt = int(np.sum(x == v))
        if best is None or cnt > best[0]:
            best = (cnt, v)
    return best[1]


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def sym_fill_base(arr: np.ndarray) -> np.ndarray | None:
    y = np.zeros_like(arr)
    for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
        zz = t(arr)
        rr, cc = np.where(zz != 0)
        for r, c in zip(rr.tolist(), cc.tolist()):
            val = int(zz[r, c])
            if y[r, c] != 0 and int(y[r, c]) != val:
                return None
            y[r, c] = val
    return y


def infer_template(src: np.ndarray, dst: np.ndarray):
    st = stable_colors(src, dst)
    v = pick_var_color(dst, st) or pick_var_color(src, st)

    if np.array_equal(dst == 0, src != 0):
        nz = np.argwhere(dst != 0)
        if nz.size > 0:
            even_cols = set()
            odd_cols = set()
            for r, c in nz:
                col = int(dst[int(r), int(c)])
                if ((int(r) + int(c)) & 1) == 0:
                    even_cols.add(col)
                else:
                    odd_cols.add(col)
            if len(even_cols) == 1 and len(odd_cols) == 1:
                return ("comp_checker", {"stable": st})

    sym_src = sym_fill_base(src)
    if sym_src is not None:
        if np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
            fill_cols = sorted(set(int(x) for x in dst[sym_src == 0].ravel() if int(x) != 0))
            if len(fill_cols) <= 1:
                fill_col = fill_cols[0] if fill_cols else None
                ok = True
                if fill_col is not None:
                    bad = (sym_src == 0) & (dst != 0) & (dst != fill_col)
                    if np.any(bad):
                        ok = False
                if ok:
                    keep0 = (sym_src == 0) & (dst == 0)
                    return ("sym_fill", {"stable": st, "keep0": keep0})

    if v is not None:
        pred = src.copy()
        m0 = src == 0
        mv = src == v
        pred[m0] = v
        pred[mv] = 0
        for c in st:
            pred[dst == c] = c
        if np.array_equal(pred, dst):
            return ("invert0", {"stable": st})

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            return ("sym", {"idx": idx})

    if v is not None:
        coords = np.argwhere(src == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        pred = src.copy()
        pred[src == v] = 0
        pred[keep] = v
        if np.array_equal(pred, dst):
            return ("endpoints", {"stable": st})

        pred = src.copy()
        h, w = pred.shape
        stable_mask = np.zeros_like(pred, dtype=bool)
        for c in st:
            stable_mask |= pred == c
        for r in range(h):
            nz = np.where(pred[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if pred[r, cc] == 0 and not stable_mask[r, cc]:
                        pred[r, cc] = v
        for c in range(w):
            nz = np.where(pred[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if pred[rr, c] == 0 and not stable_mask[rr, c]:
                        pred[rr, c] = v
        if np.array_equal(pred, dst):
            return ("fill", {"stable": st})

    return None


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind == "sym":
        return symmetries(x)[params["idx"]].copy()

    st = set(params.get("stable", set()))
    v = pick_var_color(x, st)

    if kind == "comp_checker":
        h, w = x.shape
        rr, cc = np.indices((h, w))
        parity = (rr + cc) & 1
        even_mask = (x != 0) & (parity == 0)
        odd_mask = (x != 0) & (parity == 1)
        even_counts = _bincount_colors(x[even_mask])
        odd_counts = _bincount_colors(x[odd_mask])
        if even_counts:
            c_even = max(even_counts.items(), key=lambda t: t[1])[0]
        elif odd_counts:
            c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
        else:
            return x.copy()
        c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
        y = np.where(parity == 0, c_even, c_odd)
        y[x != 0] = 0
        return y.astype(x.dtype)

    if kind == "sym_fill":
        h, w = x.shape
        nz = np.argwhere(x != 0)
        if nz.size == 0:
            return x.copy()
        best = None
        counts = {}
        for r, c in nz:
            col = int(x[int(r), int(c)])
            d = min(int(r), int(c), h - 1 - int(r), w - 1 - int(c))
            counts[col] = counts.get(col, 0) + 1
            if best is None or d > best[0] or (d == best[0] and counts[col] > best[1]):
                best = (d, counts[col], col)
        fill_col = best[2]
        y = sym_fill_base(x)
        if y is None:
            return x.copy()
        keep0 = params.get("keep0", None)
        if keep0 is None or keep0.shape != y.shape:
            keep0 = np.zeros_like(y, dtype=bool)
        y[(y == 0) & ~keep0] = fill_col
        return y

    if v is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        m0 = x == 0
        mv = x == v
        y[m0] = v
        y[mv] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        y = x.copy()
        y[x == v] = 0
        y[keep] = v
        return y

    if kind == "fill":
        y = x.copy()
        h, w = y.shape
        stable_mask = np.zeros_like(y, dtype=bool)
        for c in st:
            stable_mask |= y == c
        for r in range(h):
            nz = np.where(y[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if y[r, cc] == 0 and not stable_mask[r, cc]:
                        y[r, cc] = v
        for c in range(w):
            nz = np.where(y[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if y[rr, c] == 0 and not stable_mask[rr, c]:
                        y[rr, c] = v
        return y

    return x.copy()


def write_into_frame(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    sprite: np.ndarray,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    out = np.array(grid, copy=True)
    out[r0 + 1 : r1, c0 + 1 : c1] = 0
    paste(out, sprite, r0 + 1, c0 + 1)
    return out


def _pair_axis(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> str:
    ra = (box_a[0] + box_a[2]) / 2.0
    ca = (box_a[1] + box_a[3]) / 2.0
    rb = (box_b[0] + box_b[2]) / 2.0
    cb = (box_b[1] + box_b[3]) / 2.0
    return "horizontal" if abs(ca - cb) >= abs(ra - rb) else "vertical"


def _direction_from_component(pts: list[tuple[int, int]], axis: str) -> str | None:
    if not pts:
        return None
    if axis == "horizontal":
        col_counts: dict[int, int] = {}
        for _r, c in pts:
            col_counts[c] = col_counts.get(c, 0) + 1
        cols = sorted(col_counts)
        if len(cols) < 2:
            return None
        if col_counts[cols[-1]] == 1 and col_counts[cols[-2]] >= 2:
            return "right"
        if col_counts[cols[0]] == 1 and col_counts[cols[1]] >= 2:
            return "left"
        return None
    row_counts: dict[int, int] = {}
    for r, _c in pts:
        row_counts[r] = row_counts.get(r, 0) + 1
    rows = sorted(row_counts)
    if len(rows) < 2:
        return None
    if row_counts[rows[-1]] == 1 and row_counts[rows[-2]] >= 2:
        return "down"
    if row_counts[rows[0]] == 1 and row_counts[rows[1]] >= 2:
        return "up"
    return None


def decode_arrow_direction(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int, int, int, str, int, int]],
    pair: tuple[int, int],
) -> str | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    reserved = np.zeros((h, w), dtype=bool)
    for _, _, r0, c0, r1, c1, _, _, _ in frames:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    boxes = {idx: (frames[idx][2], frames[idx][3], frames[idx][4], frames[idx][5]) for idx in pair}
    box_a = boxes[pair[0]]
    box_b = boxes[pair[1]]
    axis = _pair_axis(box_a, box_b)
    candidates = []

    if axis == "horizontal":
        left_idx, right_idx = sorted(pair, key=lambda i: boxes[i][1])
        left_box = boxes[left_idx]
        right_box = boxes[right_idx]
        gap_c0 = left_box[3] + 1
        gap_c1 = right_box[1] - 1
        mid_r = (left_box[0] + left_box[2]) // 2
        row_lo = max(0, mid_r - 1)
        row_hi = min(h - 1, mid_r + 1)
        for color in colors_present(g, exclude={0}):
            comps = components_mask((g == color) & ~reserved, connectivity=4)
            for comp in comps:
                if comp.size == 0:
                    continue
                pts = [(int(r), int(c)) for r, c in comp]
                r0, c0, r1, c1 = bbox_coords(comp)
                if c0 < gap_c0 or c1 > gap_c1:
                    continue
                if r0 < row_lo or r1 > row_hi:
                    continue
                if (r1 - r0 + 1) > 3 or (c1 - c0 + 1) < 2:
                    continue
                candidates.append((len(pts), pts))
    else:
        top_idx, bottom_idx = sorted(pair, key=lambda i: boxes[i][0])
        top_box = boxes[top_idx]
        bottom_box = boxes[bottom_idx]
        gap_r0 = top_box[2] + 1
        gap_r1 = bottom_box[0] - 1
        mid_c = (top_box[1] + top_box[3]) // 2
        col_lo = max(0, mid_c - 1)
        col_hi = min(w - 1, mid_c + 1)
        for color in colors_present(g, exclude={0}):
            comps = components_mask((g == color) & ~reserved, connectivity=4)
            for comp in comps:
                if comp.size == 0:
                    continue
                pts = [(int(r), int(c)) for r, c in comp]
                r0, c0, r1, c1 = bbox_coords(comp)
                if r0 < gap_r0 or r1 > gap_r1:
                    continue
                if c0 < col_lo or c1 > col_hi:
                    continue
                if (c1 - c0 + 1) > 3 or (r1 - r0 + 1) < 2:
                    continue
                candidates.append((len(pts), pts))

    if not candidates:
        return None
    pts = max(candidates, key=lambda t: t[0])[1]
    return _direction_from_component(pts, axis)


def decode_gap_direction(
    frames: list[tuple[int, int, int, int, int, int, str, int, int]],
    pair: tuple[int, int],
) -> str | None:
    a = frames[pair[0]]
    b = frames[pair[1]]
    box_a = (a[2], a[3], a[4], a[5])
    box_b = (b[2], b[3], b[4], b[5])
    axis = _pair_axis(box_a, box_b)
    if a[6] != b[6]:
        return None
    gap_side = a[6]
    if axis == "horizontal" and gap_side in ("left", "right"):
        return GAP_TO_DIR[gap_side]
    if axis == "vertical" and gap_side in ("top", "bottom"):
        return GAP_TO_DIR[gap_side]
    return None


def decode_pair_direction(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int, int, int, str, int, int]],
    pair: tuple[int, int],
) -> tuple[int, int] | None:
    gap_dir = decode_gap_direction(frames, pair)
    arrow_dir = decode_arrow_direction(grid, frames, pair)
    if gap_dir is None or arrow_dir is None or gap_dir != arrow_dir:
        return None

    boxes = {idx: (frames[idx][2], frames[idx][3], frames[idx][4], frames[idx][5]) for idx in pair}
    if gap_dir == "right":
        return tuple(sorted(pair, key=lambda i: boxes[i][1]))  # type: ignore[return-value]
    if gap_dir == "left":
        return tuple(sorted(pair, key=lambda i: boxes[i][1], reverse=True))  # type: ignore[return-value]
    if gap_dir == "down":
        return tuple(sorted(pair, key=lambda i: boxes[i][0]))  # type: ignore[return-value]
    if gap_dir == "up":
        return tuple(sorted(pair, key=lambda i: boxes[i][0], reverse=True))  # type: ignore[return-value]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a directed sprite transformation from broken framed pairs, then fill the empty destination frame.

    Input structure:
    - The board contains four same-size square frames on a noisy 30x30 canvas, and every frame has
      exactly one 1-2 cell border gap on a single side.
    - Two non-blank frames share the same non-zero palette and form the training pair; one other
      non-blank frame is the test sprite, and one frame has an all-zero interior and serves as the
      output container.
    - The training pair and the test/blank pair each have a thin arrow in the corridor between them,
      and the arrow direction matches the shared gap side of that pair.

    Rule:
    - Detect the gapped frames, identify the training pair by matching interior palettes, and decode
      the source->destination order from the combined gap-side cue and corridor arrow.
    - Infer the inner sprite transform demonstrated by that directed training pair using the original
      rule families (fill, endpoints, symmetry, symmetry-fill, invert-zero, checker complement).
    - Apply the same transform to the test sprite and write the result into the blank frame interior,
      returning the full grid.

    Key insight:
    - The directional information is explicit twice over: frame gaps and extra marker pixels both
      point along the transformation flow. Solving requires using those cues to orient the analogy
      and then placing the answer in the empty directed destination frame rather than returning a crop.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_gap_square_frames(g)
    if len(frames) < 4:
        return g.copy()

    inners = []
    for size, col, r0, c0, r1, c1, gap_side, gap_start, gap_width in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inners.append(((r0, c0, r1, c1, gap_side, gap_start, gap_width), inner))

    blank_idx = None
    for i, (_, inner) in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for _, inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        inner = inners[i][1]
        nz = sorted(set(int(x) for x in inner.ravel() if int(x) != 0))
        sigs.append((i, tuple(nz)))

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return g.copy()

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return g.copy()
    test_idx = remaining[0]

    train_dir = decode_pair_direction(g, frames, pair)
    if train_dir is None:
        return g.copy()

    test_blank_dir = decode_pair_direction(g, frames, (test_idx, blank_idx))
    if test_blank_dir is None or test_blank_dir[1] != blank_idx:
        return g.copy()

    blank_box, blank_inner = inners[blank_idx]
    test_inner = inners[test_idx][1]
    if blank_inner.shape != test_inner.shape:
        return g.copy()

    src_idx, dst_idx = train_dir
    src = inners[src_idx][1]
    dst = inners[dst_idx][1]

    tpl = infer_template(src, dst)
    if tpl is None:
        return g.copy()
    kind, params = tpl
    transformed = apply_template(kind, params, test_inner)
    return write_into_frame(g, blank_box[:4], transformed)


# --------------- generator ---------------

# Canvas
H = W = 30

# Palette pools
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (3, 4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

# Frame layout
FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
MIN_AXIS_GAP = 4
GAP_WIDTH_CHOICES = (1, 2)

# Rule set
RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]
SYM_DIRECTIONAL_IDXS = (2, 6)

# Sprite pixel-count bounds
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32

# Rule-specific build bounds
SYMFILL_KEEP0_PROB = 0.4
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
COMP_CHECKER_M_MAX = 10
INVERT0_STABLE_MAX = 3
INVERT0_VAR_MAX = 12
SYM_BUILD_RETRIES = 30
SPARSE_COUNT_MAX = 10

# Outer retry loop
GEN_MAX_ATTEMPTS = 500


def orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return sorted(cells)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def build_fill_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    dst = np.zeros((n, n), dtype=int)
    while True:
        h = rng.randint(2, n)
        w = rng.randint(2, n)
        if FILL_RECT_AREA_MIN <= h * w <= FILL_RECT_AREA_MAX:
            break
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)
    dst[r0 : r0 + h, c0 : c0 + w] = v
    src = np.zeros_like(dst)
    src[r0, c0] = v
    src[r0, c0 + w - 1] = v
    src[r0 + h - 1, c0] = v
    src[r0 + h - 1, c0 + w - 1] = v
    return src, dst


def build_endpoints_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    src = np.zeros((n, n), dtype=int)
    dst = np.zeros_like(src)
    row_pairs = [(0, n - 1)]
    if n >= 5:
        row_pairs.extend([(0, n - 2), (1, n - 1)])
    r1, r2 = rng.choice(row_pairs)
    len1 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    len2 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    c1 = rng.randint(0, n - len1)
    c2 = rng.randint(0, n - len2)
    draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=v)
    draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=v)
    dst[r1, c1] = v
    dst[r1, c1 + len1 - 1] = v
    dst[r2, c2] = v
    dst[r2, c2 + len2 - 1] = v
    return src, dst


def build_comp_checker_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if len(colors) == 1:
        even_color = odd_color = colors[0]
    else:
        even_color, odd_color = colors
    min_src = 4 if n == 6 else 3
    m = rng.randint(min_src, min(COMP_CHECKER_M_MAX, n * n - 3))
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    while True:
        chosen = rng.sample(all_cells, m)
        parities = {(r + c) & 1 for r, c in chosen}
        if len(colors) == 1 or len(parities) == 2:
            break
    rr, cc = np.indices((n, n))
    full = np.where(((rr + cc) & 1) == 0, even_color, odd_color).astype(int)
    hole_mask = np.zeros((n, n), dtype=bool)
    for r, c in chosen:
        hole_mask[r, c] = True
    src = np.where(hole_mask, full, 0)
    dst = np.where(hole_mask, 0, full)
    return src, dst


def build_invert0_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v, s = colors
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    stable_n = rng.randint(1, min(INVERT0_STABLE_MAX, n))
    stable_cells = set(rng.sample(all_cells, stable_n))
    remaining = [cell for cell in all_cells if cell not in stable_cells]
    var_n = rng.randint(2, max(2, min(len(remaining) - 2, INVERT0_VAR_MAX)))
    var_cells = set(rng.sample(remaining, var_n))
    dst = np.zeros((n, n), dtype=int)
    for r, c in stable_cells:
        dst[r, c] = s
    for r, c in var_cells:
        dst[r, c] = v
    src = dst.copy()
    src[dst == 0] = v
    src[dst == v] = 0
    return src, dst


def inverse_symmetry(arr: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0:
        return arr.copy()
    if idx == 1:
        return np.fliplr(arr)
    if idx == 2:
        return np.rot90(arr, -1)
    if idx == 3:
        return np.rot90(np.fliplr(arr), -1)
    if idx == 4:
        return np.rot90(arr, 2)
    if idx == 5:
        return np.flipud(arr)
    if idx == 6:
        return np.rot90(arr, 1)
    if idx == 7:
        return np.rot90(np.fliplr(arr), 1)
    raise ValueError(idx)


def random_sparse_pattern(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    count = rng.randint(3, min(SPARSE_COUNT_MAX, n * n - 1))
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], count)
    used = [colors[0]] * count
    if len(colors) == 2:
        for i in range(count):
            used[i] = colors[rng.randint(0, 1)]
        used[0] = colors[0]
        used[1] = colors[1]
    for (r, c), col in zip(cells, used):
        arr[r, c] = col
    return arr


def build_sym_pair(n: int, colors: list[int], rng: random.Random, idx: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SYM_BUILD_RETRIES):
        dst = random_sparse_pattern(n, colors, rng)
        src = inverse_symmetry(dst, idx)
        if not np.array_equal(src, dst):
            return src, dst
    raise ValueError("failed to build sym pair")


def build_sym_fill_pair(
    n: int,
    colors: list[int],
    rng: random.Random,
    keep0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fill_col, accent_col = colors
    src = np.zeros((n, n), dtype=int)
    orbits = []
    seen = set()
    for r in range(n):
        for c in range(n):
            orb = tuple(orbit_cells(n, r, c))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))
    orbits.sort(
        key=lambda orb: (
            -max(min(r, c, n - 1 - r, n - 1 - c) for r, c in orb),
            -len(orb),
            orb[0],
        )
    )
    fill_orbit = next(orb for orb in orbits if not any(keep0[r, c] for r, c in orb))
    src[fill_orbit[0]] = fill_col
    accent_orbits = [orb for orb in orbits if orb != fill_orbit and not any(keep0[r, c] for r, c in orb)]
    picks = []
    for orb in accent_orbits:
        if len(orb) >= 2:
            picks.append(orb)
        if len(picks) >= 2:
            break
    if len(picks) < 2:
        raise ValueError("not enough accent orbits")
    for orb in picks:
        src[orb[0]] = accent_col
    dst = apply_template("sym_fill", {"keep0": keep0}, src)
    return src, dst


def choose_rule_setup(n: int, rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "invert0":
        color_count = 2
    elif kind == "sym_fill":
        color_count = 2
        keep0 = np.zeros((n, n), dtype=bool)
        if n == 6 or rng.random() < SYMFILL_KEEP0_PROB:
            for r, c in orbit_cells(n, 0, 0):
                keep0[r, c] = True
        params["keep0"] = keep0
    elif kind == "sym":
        color_count = rng.choice([1, 2])
        params["idx"] = rng.choice(SYM_DIRECTIONAL_IDXS)
    elif kind == "comp_checker":
        color_count = rng.choice([1, 2])
    else:
        color_count = 1
    return kind, params, color_count


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, colors, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, colors, rng)
    if kind == "comp_checker":
        return build_comp_checker_pair(n, colors, rng)
    if kind == "invert0":
        return build_invert0_pair(n, colors, rng)
    if kind == "sym":
        return build_sym_pair(n, colors, rng, params["idx"])
    if kind == "sym_fill":
        return build_sym_fill_pair(n, colors, rng, params["keep0"])
    raise ValueError(kind)


def structured_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    top_low, top_high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    while True:
        top = rng.randint(top_low, top_high)
        bottom = rng.randint(bottom_low, bottom_high)
        left = rng.randint(top_low, top_high)
        right = rng.randint(bottom_low, bottom_high)
        if right - (left + s) < MIN_AXIS_GAP:
            continue
        if bottom - (top + s) < MIN_AXIS_GAP:
            continue
        return [(top, left), (top, right), (bottom, left), (bottom, right)]


def render_background(
    frame_boxes: list[tuple[int, int, int, int]],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((H, W))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_cells = [(r, c) for r, c in zip(*np.where(~reserved))]
    lo, hi = BG_PERTURB_FRACTION
    perturb_n = int(round(len(bg_cells) * rng.uniform(lo, hi)))
    noise_choices = list(bg_palette) + [c for c in BG_EXTRA_NOISE if c not in bg_palette]
    for r, c in rng.sample(bg_cells, perturb_n):
        grid[r, c] = rng.choice(noise_choices)
    return grid


def choose_gap_start(s: int, width: int, rng: random.Random) -> int:
    starts = list(range(1, s - width))
    centered = [start for start in starts if start >= 2 and start + width <= s - 2]
    if centered:
        starts = centered
    return rng.choice(starts)


def erase_gap(grid: np.ndarray, r0: int, c0: int, s: int, gap_side: str, gap_start: int, gap_width: int) -> None:
    if gap_side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_width] = 0
    elif gap_side == "bottom":
        grid[r0 + s - 1, c0 + gap_start : c0 + gap_start + gap_width] = 0
    elif gap_side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0] = 0
    elif gap_side == "right":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0 + s - 1] = 0
    else:
        raise ValueError(gap_side)


def draw_gapped_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    s: int,
    frame_color: int,
    sprite: np.ndarray,
    gap_side: str,
    gap_start: int,
    gap_width: int,
) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    erase_gap(grid, r0, c0, s, gap_side, gap_start, gap_width)
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def draw_direction_arrow(
    grid: np.ndarray,
    src_box: tuple[int, int, int, int],
    dst_box: tuple[int, int, int, int],
    marker_color: int,
) -> None:
    sr0, sc0, sr1, sc1 = src_box
    dr0, dc0, dr1, dc1 = dst_box
    if sc1 < dc0:
        mid = (sr0 + sr1) // 2
        shaft_start = sc1 + 1
        shaft_end = dc0 - 2
        draw_line(grid, mid, shaft_start, end_r=mid, end_c=shaft_end, color=marker_color)
        grid[mid - 1, shaft_end] = marker_color
        grid[mid + 1, shaft_end] = marker_color
        grid[mid, dc0 - 1] = marker_color
        return
    if dc1 < sc0:
        mid = (sr0 + sr1) // 2
        shaft_start = dc1 + 2
        shaft_end = sc0 - 1
        draw_line(grid, mid, shaft_start, end_r=mid, end_c=shaft_end, color=marker_color)
        grid[mid - 1, shaft_start] = marker_color
        grid[mid + 1, shaft_start] = marker_color
        grid[mid, dc1 + 1] = marker_color
        return
    if sr1 < dr0:
        mid = (sc0 + sc1) // 2
        shaft_start = sr1 + 1
        shaft_end = dr0 - 2
        draw_line(grid, shaft_start, mid, end_r=shaft_end, end_c=mid, color=marker_color)
        grid[shaft_end, mid - 1] = marker_color
        grid[shaft_end, mid + 1] = marker_color
        grid[dr0 - 1, mid] = marker_color
        return
    mid = (sc0 + sc1) // 2
    shaft_start = dr1 + 2
    shaft_end = sr0 - 1
    draw_line(grid, shaft_start, mid, end_r=shaft_end, end_c=mid, color=marker_color)
    grid[shaft_start, mid - 1] = marker_color
    grid[shaft_start, mid + 1] = marker_color
    grid[dr1 + 1, mid] = marker_color


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    setup_params: dict,
    n: int,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != kind:
        return None
    inferred_kind, inferred_params = inferred

    test_src, _ = build_training_pair(kind, setup_params, n, p2, rng)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_scene_colors(used: set[int], rng: random.Random) -> tuple[int, int, list[int]]:
    free = [c for c in ALL_COLORS if c not in used and c != 0]
    if len(free) < 5:
        raise ValueError("not enough free colors")
    frame_color = rng.choice(free)
    free = [c for c in free if c != frame_color]
    marker_color = rng.choice(free)
    free = [c for c in free if c != marker_color]
    bg_sizes = [p for p in BG_PERIOD_CHOICES if len(free) >= p]
    if not bg_sizes:
        raise ValueError("not enough colors left for background")
    bg_palette = rng.sample(free, rng.choice(bg_sizes))
    return frame_color, marker_color, bg_palette


def _assign_roles(
    axis: str,
    direction: str,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    if axis == "horizontal":
        training_row = rng.choice([0, 1])
        rows = [(0, 1), (2, 3)]
        train_pair = rows[training_row]
        test_pair = rows[1 - training_row]
        if direction == "right":
            src_idx, dst_idx = train_pair
            test_idx, blank_idx = test_pair
        else:
            dst_idx, src_idx = train_pair
            blank_idx, test_idx = test_pair
        return (src_idx, dst_idx), (test_idx, blank_idx), test_idx, blank_idx

    training_col = rng.choice([0, 1])
    cols = [(0, 2), (1, 3)]
    train_pair = cols[training_col]
    test_pair = cols[1 - training_col]
    if direction == "down":
        src_idx, dst_idx = train_pair
        test_idx, blank_idx = test_pair
    else:
        dst_idx, src_idx = train_pair
        blank_idx, test_idx = test_pair
    return (src_idx, dst_idx), (test_idx, blank_idx), test_idx, blank_idx


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    marker_color: int,
    gap_side: str,
    gap_widths: list[int],
    train_dir: tuple[int, int],
    test_blank_dir: tuple[int, int],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for idx, ((r0, c0), sprite) in enumerate(zip(positions, sprites)):
        gap_width = gap_widths[idx]
        gap_start = choose_gap_start(s, gap_width, rng)
        draw_gapped_frame(grid, r0, c0, s, frame_color, sprite, gap_side, gap_start, gap_width)
    draw_direction_arrow(grid, frame_boxes[train_dir[0]], frame_boxes[train_dir[1]], marker_color)
    draw_direction_arrow(grid, frame_boxes[test_blank_dir[0]], frame_boxes[test_blank_dir[1]], marker_color)
    return grid


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_gap_square_frames(grid)
    if len(frames) != 4:
        return False
    if len(find_complete_square_frames(grid)) != 0:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    if expected_output.shape != grid.shape:
        return False

    gap_sides = {gap_side for _, _, _, _, _, _, gap_side, _, _ in frames}
    if len(gap_sides) != 1:
        return False
    widths = [gap_width for _, _, _, _, _, _, _, _, gap_width in frames]
    if any(width not in GAP_WIDTH_CHOICES for width in widths):
        return False

    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1, _, _, _ in frames]
    blank_ids = [i for i, inner in enumerate(inners) if np.all(inner == 0)]
    if len(blank_ids) != 1:
        return False
    blank_idx = blank_ids[0]

    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    sig_to_idxs: dict[tuple[int, ...], list[int]] = {}
    for idx, sig in enumerate(nonblank_sigs):
        if sig:
            sig_to_idxs.setdefault(sig, []).append(idx)
    matching_pairs = [tuple(v) for v in sig_to_idxs.values() if len(v) == 2]
    if len(matching_pairs) != 1:
        return False
    pair = matching_pairs[0]

    train_dir = decode_pair_direction(grid, frames, pair)
    if train_dir is None:
        return False
    test_idx = next(i for i in range(4) if i not in pair and i != blank_idx)
    test_blank_dir = decode_pair_direction(grid, frames, (test_idx, blank_idx))
    if test_blank_dir is None or test_blank_dir[1] != blank_idx:
        return False
    if decode_gap_direction(frames, pair) != decode_gap_direction(frames, (test_idx, blank_idx)):
        return False

    _, _, r0, c0, r1, c1, _, _, _ = frames[blank_idx]
    allowed = np.zeros_like(grid, dtype=bool)
    allowed[r0 + 1 : r1, c0 + 1 : c1] = True
    diff = expected_output != grid
    if not np.any(diff & allowed):
        return False
    if np.any(diff & ~allowed):
        return False
    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))
    built = _build_sprites(kind, setup_params, n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    try:
        frame_color, marker_color, bg_palette = _choose_scene_colors(set(p1) | set(p2), rng)
    except ValueError:
        return None

    positions = structured_positions(s, rng)
    axis = rng.choice(["horizontal", "vertical"])
    direction = rng.choice(["right", "left"]) if axis == "horizontal" else rng.choice(["down", "up"])
    gap_side = DIR_TO_GAP[direction]
    train_dir, test_blank_dir, test_idx, blank_idx = _assign_roles(axis, direction, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_dir[0]] = train_src
    sprites[train_dir[1]] = train_dst
    sprites[test_idx] = test_src
    gap_widths = [rng.choice(GAP_WIDTH_CHOICES) for _ in range(4)]

    grid = _render_scene(
        s,
        positions,
        sprites,
        frame_color,
        marker_color,
        gap_side,
        gap_widths,
        train_dir,
        test_blank_dir,
        bg_palette,
        rng,
    )

    blank_r0, blank_c0 = positions[blank_idx]
    expected_output = write_into_frame(
        grid,
        (blank_r0, blank_c0, blank_r0 + s - 1, blank_c0 + s - 1),
        test_dst,
    )
    if not validate_scene(grid, expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
