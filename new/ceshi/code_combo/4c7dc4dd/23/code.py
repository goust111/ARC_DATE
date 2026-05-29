# Combo variant of 4c7dc4dd (index 23)
# combo_spec: D5/4+D11/1+D15/2
# Name: 定向连线并排验证

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    Original task:
#    - Find the two same-palette training frames.
#    - Try both A->B and B->A across a mixed rule library.
#    - Return only the transformed test interior.
#    Combo behavior:
#    - Keep the framed-sprite analogy layout, but the only transform is
#      connector_draw: same-color endpoints get bridged along clear horizontal
#      or vertical spans, while the accent color stays fixed.
#    - Direction is not guessed. The earlier training frame in row-major space
#      is the source; the later one is the destination.
#    - Output is a packed verification board containing two panels:
#      training destination first, then transformed test result, separated by
#      one black row/column. The packing axis mirrors the training-pair axis.
#    Small trace:
#      src row : [v, 0, v, 0, s]
#      old fill: [v, v, v, v, s]
#      new dst : [v, v, v, 0, s]
#      packed  : [train_dst | 0 | test_dst]  or stacked vertically.
#
# b) Complexity calibration
#    - Generator uses exactly two sprite colors per non-blank frame:
#      one active connector color and one stable accent color.
#    - Each sprite includes at least one witness where connector_draw differs
#      from naive span fill, so the D5 change is visible.
#    - Training slots are chosen from same-row or same-column quadrant pairs,
#      giving a repeated, human-visible cue for both direction and output axis.
#
# c) Hidden-assumption audit for main()
#    - Pair matching by shared non-zero palette still works because generator
#      keeps exactly one duplicated palette among non-blank frames.
#    - Sorting the training pair by bbox assumes spatial order is meaningful;
#      generator guarantees that the intended source is earlier in row-major
#      order.
#    - Axis decoding from bbox deltas assumes the pair is clearly horizontal or
#      vertical; generator uses only same-row or same-column quadrant pairs.
#    - pick_connector_color() assumes only one non-stable color can actually
#      grow into bridges; generator validates this structurally on train/test.
#    - Packed output assumes both panels share the same interior size; all four
#      frames still share one sampled side length S.
#
# d) Placement legality
#    - Frames stay inside the 30x30 board, one per quadrant, non-overlapping.
#    - The two training frames are sampled from legal row/column pairs:
#      top/bottom row for horizontal, left/right column for vertical.
#    - Test and blank occupy the remaining two quadrant slots.
#
# e) Visual observability
#    - Humans can first identify the training pair by shared palette, then read
#      source->destination directly from spatial order, then notice the output
#      packs the known destination next to the inferred test result on the same
#      dominant axis.
#    - No invisible ordering by color id or pixel count is used.
#
# f) Visual cleanliness
#    - Connector sprites stay sparse: a small number of endpoint pairs plus one
#      or two accent witnesses. This avoids clutter inside 4x4 to 6x6 interiors.
#    - The packed output uses a one-cell black separator so the two verification
#      panels never merge visually.
#
# g) Change inferability
#    - There is no arbitrary color remapping. The active color extends, the
#      accent color stays put, and the same structural rule transfers to the
#      test palette.
#    - The extra D15 packing does not introduce new color semantics; it only
#      changes how the two already-computable results are arranged.
#
# h) New solver assumptions reviewed against generator
#    - If infer_template() fails, main() falls back to packing the visible train
#      destination with the raw test sprite; generator rejects such scenes.
#    - main() never searches for reverse direction or alternative rules; every
#      generated sample is verified under this stricter directed solver.


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
CONNECTOR_BUILD_RETRIES = 100
GEN_MAX_ATTEMPTS = 300
NEXT_HORIZONTAL_TRAIN_PAIR = True


def find_square_frames(
    grid: np.ndarray,
    limit: int | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
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
        if limit is not None and len(picked) >= limit:
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
        cand = (cnt, v)
        if best is None or cand > best:
            best = cand
    return best[1]


def span_fill_transform(x: np.ndarray, st: set[int], v: int | None) -> np.ndarray:
    y = x.copy()
    if v is None:
        return y
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


def connector_draw_transform(x: np.ndarray, st: set[int], v: int | None) -> np.ndarray:
    y = x.copy()
    if v is None:
        return y
    base = np.asarray(x, dtype=int)
    h, w = base.shape
    for r in range(h):
        cols = np.where(base[r, :] == v)[0].tolist()
        for i in range(0, len(cols) - 1, 2):
            c0 = int(cols[i])
            c1 = int(cols[i + 1])
            segment = base[r, c0 : c1 + 1]
            if np.any((segment != 0) & (segment != v)):
                continue
            for cc in range(c0 + 1, c1):
                if base[r, cc] == 0:
                    y[r, cc] = v
    for c in range(w):
        rows = np.where(base[:, c] == v)[0].tolist()
        for i in range(0, len(rows) - 1, 2):
            r0 = int(rows[i])
            r1 = int(rows[i + 1])
            segment = base[r0 : r1 + 1, c]
            if np.any((segment != 0) & (segment != v)):
                continue
            for rr in range(r0 + 1, r1):
                if base[rr, c] == 0:
                    y[rr, c] = v
    return y


def pick_connector_color(x: np.ndarray, st: set[int]) -> int | None:
    candidates = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
    if not candidates:
        candidates = [int(v) for v in np.unique(x) if int(v) != 0]
    best = None
    for color in candidates:
        pred = connector_draw_transform(x, st, color)
        added = int(np.sum(pred == color) - np.sum(x == color))
        cnt = int(np.sum(x == color))
        cand = (added, cnt, color)
        if best is None or cand > best:
            best = cand
    if best is not None and best[0] > 0:
        return best[2]
    return pick_var_color(x, st)


def infer_template(src: np.ndarray, dst: np.ndarray):
    st = stable_colors(src, dst)
    v = pick_connector_color(src, st)
    pred = connector_draw_transform(src, st, v)
    if np.array_equal(pred, dst):
        return ("connector_draw", {"stable": st})
    return None


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind != "connector_draw":
        return x.copy()
    st = set(params.get("stable", set()))
    v = pick_connector_color(x, st)
    return connector_draw_transform(x, st, v)


def pair_is_horizontal(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> bool:
    ar = (int(a_bbox[0]) + int(a_bbox[2])) / 2.0
    ac = (int(a_bbox[1]) + int(a_bbox[3])) / 2.0
    br = (int(b_bbox[0]) + int(b_bbox[2])) / 2.0
    bc = (int(b_bbox[1]) + int(b_bbox[3])) / 2.0
    return abs(bc - ac) >= abs(br - ar)


def pack_outputs(train_dst: np.ndarray, test_dst: np.ndarray, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.zeros((train_dst.shape[0], 1), dtype=int)
        return np.concatenate([train_dst, sep, test_dst], axis=1)
    sep = np.zeros((1, train_dst.shape[1]), dtype=int)
    return np.concatenate([train_dst, sep, test_dst], axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a directed connector rule from framed sprites, then return a packed verification output.

    Input structure:
    - The board contains four same-size hollow square frames on a noisy background.
    - One frame interior is blank, two non-blank frames share a palette and form the training pair, and the remaining non-blank frame is the test sprite.
    - The training pair is deliberately arranged in the same row or the same column so its spatial order is visible.

    Rule:
    - Match the training pair by shared non-zero color set, then sort those two frames by spatial reading order; the earlier frame is the source and the later frame is the destination.
    - The source-to-destination change is connector_draw: one active color bridges aligned same-color endpoint pairs through empty cells, while accent cells remain unchanged.
    - Apply that directed rule to the test sprite, then output the visible training destination together with the computed test result, packed along the training pair's main axis with a one-cell black separator.

    Key insight:
    - The puzzle is still an analogy, but now two cues must be fused: placement tells you which training sprite is "before" the other, and the final answer is a two-panel check that repeats the known destination next to the inferred test transformation.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g, limit=6)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(((r0, c0, r1, c1), g[r0 + 1 : r1, c0 + 1 : c1]))

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
        counts = [(i, int(np.count_nonzero(inners[i][1]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    pair = tuple(sorted(pair, key=lambda i: inners[i][0]))
    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[pair[1]][1].copy()

    test_idx = remaining[0]
    src_bbox, src_inner = inners[pair[0]]
    dst_bbox, dst_inner = inners[pair[1]]
    test_inner = inners[test_idx][1]

    inferred = infer_template(src_inner, dst_inner)
    if inferred is None:
        test_dst = test_inner.copy()
    else:
        kind, params = inferred
        test_dst = apply_template(kind, params, test_inner)

    horizontal = pair_is_horizontal(src_bbox, dst_bbox)
    return pack_outputs(dst_inner, test_dst, horizontal)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def _line_cells(orient: str, fixed: int, coords: list[int]) -> list[tuple[int, int]]:
    if orient == "h":
        return [(fixed, c) for c in coords]
    return [(r, fixed) for r in coords]


def _counts_ok(var_cells: set[tuple[int, int]]) -> bool:
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in var_cells:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1
        if row_counts[r] > 2 or col_counts[c] > 2:
            return False
    return True


def _clear_candidates(
    n: int,
    occupied: set[tuple[int, int]],
    var_cells: set[tuple[int, int]],
) -> list[tuple[str, int, int, int, list[tuple[int, int]]]]:
    cands = []
    for orient in ("h", "v"):
        for fixed in range(n):
            for a in range(n):
                for b in range(a + 2, n):
                    cells = _line_cells(orient, fixed, [a, b])
                    if any(cell in occupied for cell in cells):
                        continue
                    if not _counts_ok(var_cells | set(cells)):
                        continue
                    cands.append((orient, fixed, a, b, cells))
    return cands


def _tail_witness_candidates(
    n: int,
    occupied: set[tuple[int, int]],
    var_cells: set[tuple[int, int]],
) -> list[tuple[str, int, int, int, int, list[tuple[int, int]]]]:
    cands = []
    for orient in ("h", "v"):
        for fixed in range(n):
            for a in range(n):
                for b in range(a + 2, n):
                    for p in range(n):
                        if p in range(a, b + 1):
                            continue
                        if p < a and a - p < 2:
                            continue
                        if p > b and p - b < 2:
                            continue
                        cells = _line_cells(orient, fixed, [a, b, p])
                        if any(cell in occupied for cell in cells):
                            continue
                        if not _counts_ok(var_cells | set(cells[:2])):
                            continue
                        cands.append((orient, fixed, a, b, p, cells))
    return cands


def _blocked_witness_candidates(
    n: int,
    occupied: set[tuple[int, int]],
    var_cells: set[tuple[int, int]],
) -> list[tuple[str, int, int, int, int, list[tuple[int, int]]]]:
    cands = []
    for orient in ("h", "v"):
        for fixed in range(n):
            for a in range(n):
                for b in range(a + 2, n):
                    for p in range(a + 1, b):
                        cells = _line_cells(orient, fixed, [a, b, p])
                        if any(cell in occupied for cell in cells):
                            continue
                        if not _counts_ok(var_cells | set(cells[:2])):
                            continue
                        cands.append((orient, fixed, a, b, p, cells))
    return cands


def build_connector_draw_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v, s = colors
    for _ in range(CONNECTOR_BUILD_RETRIES):
        src = np.zeros((n, n), dtype=int)
        occupied: set[tuple[int, int]] = set()
        var_cells: set[tuple[int, int]] = set()

        clear = _clear_candidates(n, occupied, var_cells)
        if not clear:
            continue
        _, _, _, _, cells = rng.choice(clear)
        for r, c in cells:
            src[r, c] = v
        occupied |= set(cells)
        var_cells |= set(cells)

        witness_kind = "tail" if (n >= 5 and rng.random() < 0.65) else "blocked"
        if witness_kind == "tail":
            witness = _tail_witness_candidates(n, occupied, var_cells)
            if not witness:
                witness_kind = "blocked"
        if witness_kind == "blocked":
            witness = _blocked_witness_candidates(n, occupied, var_cells)
        if not witness:
            continue

        _, _, _, _, _, cells = rng.choice(witness)
        for r, c in cells[:2]:
            src[r, c] = v
        sr, sc = cells[2]
        src[sr, sc] = s
        occupied |= set(cells)
        var_cells |= set(cells[:2])

        if rng.random() < 0.35:
            extra = _clear_candidates(n, occupied, var_cells)
            if extra:
                _, _, _, _, cells = rng.choice(extra)
                for r, c in cells:
                    src[r, c] = v
                occupied |= set(cells)
                var_cells |= set(cells)

        dst = apply_template("connector_draw", {"stable": {s}}, src)
        fill_dst = span_fill_transform(src, {s}, v)
        added = int(np.sum(dst == v) - np.sum(src == v))
        if added < 2:
            continue
        if np.array_equal(dst, src) or np.array_equal(fill_dst, dst):
            continue
        inferred = infer_template(src, dst)
        if inferred is None or inferred[0] != "connector_draw":
            continue
        return src, dst

    raise ValueError("failed to build connector_draw pair")


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    low, high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    return [
        (rng.randint(low, high), rng.randint(low, high)),
        (rng.randint(low, high), rng.randint(bottom_low, bottom_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(low, high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)),
    ]


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


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _choose_training_slots(
    rng: random.Random,
    horizontal: bool | None = None,
) -> tuple[list[int], bool]:
    if horizontal is None:
        horizontal = rng.random() < 0.5
    if horizontal:
        return list(rng.choice([(0, 1), (2, 3)])), True
    return list(rng.choice([(0, 2), (1, 3)])), False


def _build_sprites(
    n: int,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
):
    train_src, train_dst = build_connector_draw_pair(n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None or inferred[0] != "connector_draw":
        return None
    _, params = inferred

    test_src, _ = build_connector_draw_pair(n, p2, rng)
    test_dst = apply_template("connector_draw", params, test_src)

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

    st_train = stable_colors(train_src, train_dst)
    v_train = pick_connector_color(train_src, st_train)
    st_test = stable_colors(test_src, test_dst)
    v_test = pick_connector_color(test_src, st_test)
    if np.array_equal(span_fill_transform(train_src, st_train, v_train), train_dst):
        return None
    if np.array_equal(span_fill_transform(test_src, st_test, v_test), test_dst):
        return None
    return train_src, train_dst, test_src, test_dst


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


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

    if not np.array_equal(main(grid), expected_output):
        return False
    return True


def _gen_attempt(force_horizontal: bool | None = None) -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    p1 = choose_palette(rng, 2, set())
    p2 = choose_palette(rng, 2, set(p1))
    built = _build_sprites(n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(s, rng)
    train_slots, horizontal = _choose_training_slots(rng, force_horizontal)
    remaining = [i for i in range(4) if i not in train_slots]
    test_slot = rng.choice(remaining)
    blank_slot = remaining[0] if remaining[1] == test_slot else remaining[1]

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src
    sprites[blank_slot] = np.zeros((n, n), dtype=int)

    expected_output = pack_outputs(train_dst, test_dst, horizontal)
    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    global NEXT_HORIZONTAL_TRAIN_PAIR

    preferred_horizontal = NEXT_HORIZONTAL_TRAIN_PAIR
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(force_horizontal=preferred_horizontal)
        if grid is not None:
            NEXT_HORIZONTAL_TRAIN_PAIR = not preferred_horizontal
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
