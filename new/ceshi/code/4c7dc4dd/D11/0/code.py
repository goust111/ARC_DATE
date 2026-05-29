# Variant of 4c7dc4dd
# Dimension: D11 — 训练对方向性 (match_cond)
# Change: solver 尝试两个方向（A→B 和 B→A），选择得分更高的方向作为变换方向... → 显式方向标记
#
# design_analysis:
#   concrete_behavior:
#     - The scene still contains 4 same-size square frames on a noisy 30x30 canvas.
#     - Two non-blank frames still share the same sprite palette and form the training pair.
#     - The D11 change is a single-color thin arrow placed in the open corridor between those two
#       training frames. The arrow is horizontal or vertical and points from source frame to
#       destination frame.
#     - Example trace:
#       original:
#         - pair frames are found by shared palette
#         - solver tries src=A,dst=B and src=B,dst=A
#         - whichever inferred rule scores higher becomes the chosen direction
#       variant:
#         - pair frames are found by shared palette
#         - solver locates the arrow component in the gap between those two frames
#         - arrow tip decides which frame is destination, so only one direction is inferred
#     - Pixel realization:
#       - horizontal arrow = 1-pixel shaft plus a 3-pixel head inside the frame gap
#       - vertical arrow = same idea rotated
#       - marker color is unique in the whole scene and never used by background/frame/sprites
#   complexity_calibration:
#     - I keep the original 6 inner transformation families unchanged.
#     - I only constrain the training-pair placement so the pair is horizontally or vertically
#       adjacent, which makes the arrow legible and avoids diagonal-direction ambiguity.
#     - For the symmetry rule I only sample 90-degree rotations (idx 2 or 6), because
#       self-inverse symmetries would hide the importance of the new direction cue.
#   implicit_assumption_audit:
#     - find_square_frames(): still valid because the arrow never touches a frame border.
#     - blank-frame detection: unchanged; the arrow lives outside all interiors.
#     - palette matching for the training pair: unchanged; the marker color is disjoint from every
#       interior color so it does not pollute signatures.
#     - original "best over both directions" loop depended on direction being implicit; that
#       assumption is removed. The new code assumes a unique arrow component exists in the pair gap.
#     - generator guarantees that assumption by using one adjacent training pair, one marker color,
#       and enough empty corridor cells for the arrow.
#   placement_legality:
#     - Frames remain non-overlapping and in four quadrants.
#     - A minimum horizontal and vertical gap is enforced before choosing the quadrant anchors.
#     - The arrow is drawn strictly inside the corridor between the two training frames, so it
#       never overlaps frames, sprites, or canvas borders.
#   visual_observability:
#     - A human can spot the source->destination relation directly from the arrow before reasoning
#       about the inner sprite rule.
#     - The marker is always a thin, isolated component with a unique color, so it stays readable
#       against the noisy background.
#   visual_cleanliness:
#     - No region mixes multiple overlays: background first, then frames, then the arrow.
#     - The arrow occupies only the gap between the two training frames.
#   color_inference:
#     - No hidden color ordering is used. The marker color does not encode a mapping; it only
#       separates the directional cue from the sprite colors by uniqueness and position.

import numpy as np
import random

from arc_utils import bbox_coords, colors_present, components_mask, draw_line, paste


def find_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
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


def decode_training_direction(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int, int, int]],
    pair: tuple[int, int],
) -> tuple[int, int] | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    reserved = np.zeros((h, w), dtype=bool)
    for _, _, r0, c0, r1, c1 in frames:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    boxes = {
        idx: (frames[idx][2], frames[idx][3], frames[idx][4], frames[idx][5])
        for idx in pair
    }
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
        if not candidates:
            return None
        pts = max(candidates, key=lambda t: t[0])[1]
        arrow = _direction_from_component(pts, axis)
        if arrow == "right":
            return left_idx, right_idx
        if arrow == "left":
            return right_idx, left_idx
        return None

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
    arrow = _direction_from_component(pts, axis)
    if arrow == "down":
        return top_idx, bottom_idx
    if arrow == "up":
        return bottom_idx, top_idx
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a framed training analogy with an explicit arrow cue, then transform the test sprite.

    Input structure:
    - The board contains four hollow square frames of one size on a noisy background.
    - One frame interior is blank, two non-blank interiors share a palette and form the training pair,
      and the remaining non-blank frame is the test sprite.
    - A thin single-color arrow sits between the two training frames and points from the source frame to
      the destination frame.

    Rule:
    - Match the two training frames by their interior color set, then decode the arrow to decide which
      member is the example input and which member is the example output.
    - Infer the inner sprite transformation in that marked direction using the same rule families as the
      original task (fill, endpoints, symmetry, symmetry-fill, invert-zero, checker complement).
    - Apply the inferred transformation to the test sprite and return the transformed test interior.

    Key insight:
    - The analogy direction is no longer guessed from heuristic scoring; it is stated in the picture by
      the arrow marker, so the solver must parse that cue before learning the sprite rule.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for size, col, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inners.append(((r0, c0, r1, c1), inner))

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
        return inners[blank_idx][1].copy()

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx][1].copy()
    test_idx = remaining[0]

    directed_pair = decode_training_direction(g, frames, pair)
    if directed_pair is None:
        return inners[test_idx][1].copy()

    src_idx, dst_idx = directed_pair
    src = inners[src_idx][1]
    dst = inners[dst_idx][1]
    test_inner = inners[test_idx][1]

    tpl = infer_template(src, dst)
    if tpl is None:
        return test_inner.copy()
    kind, params = tpl
    return apply_template(kind, params, test_inner)


# --------------- generator ---------------

# Canvas
H = W = 30

# Palette pools
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

# Frame layout
FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
MIN_AXIS_GAP = 4
TRAIN_PAIR_CHOICES = ((0, 1), (2, 3), (0, 2), (1, 3))

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
GEN_MAX_ATTEMPTS = 400


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


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
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


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    sig_to_idxs: dict[tuple[int, ...], list[int]] = {}
    for idx, sig in enumerate(tuple(colors_present(inner, exclude={0})) for inner in inners):
        if sig:
            sig_to_idxs.setdefault(sig, []).append(idx)
    pair = None
    for idxs in sig_to_idxs.values():
        if len(idxs) == 2:
            pair = (idxs[0], idxs[1])
            break
    if pair is None or decode_training_direction(grid, frames, pair) is None:
        return False
    return np.array_equal(main(grid), expected_output)


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
    free = [c for c in ALL_COLORS if c not in used]
    if len(free) < 6:
        raise ValueError("not enough free colors for frame/marker/background")
    frame_color = rng.choice(free)
    free = [c for c in free if c != frame_color]
    marker_color = rng.choice(free)
    free = [c for c in free if c != marker_color]
    bg_sizes = [p for p in BG_PERIOD_CHOICES if len(free) >= p]
    if not bg_sizes:
        raise ValueError("not enough colors left for background")
    bg_palette = rng.sample(free, rng.choice(bg_sizes))
    return frame_color, marker_color, bg_palette


def _assign_slots(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int], int]:
    train_pair = rng.choice(TRAIN_PAIR_CHOICES)
    if rng.random() < 0.5:
        directed = (train_pair[0], train_pair[1])
    else:
        directed = (train_pair[1], train_pair[0])
    remaining = [i for i in range(4) if i not in train_pair]
    test_slot = rng.choice(remaining)
    return train_pair, directed, test_slot


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    marker_color: int,
    train_direction: tuple[int, int],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    src_box = frame_boxes[train_direction[0]]
    dst_box = frame_boxes[train_direction[1]]
    draw_direction_arrow(grid, src_box, dst_box, marker_color)
    return grid


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
    _train_pair, train_direction, test_slot = _assign_slots(rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_direction[0]] = train_src
    sprites[train_direction[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, marker_color, train_direction, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
