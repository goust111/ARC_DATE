# Combo variant of 4c7dc4dd (index 3)
# combo_spec: D3/2+D5/4
# Name: 分量内连线绘制

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, paste


# Design analysis for the fused D3/2 + D5/4 variant
# --------------------------------------------------
# a) Concrete behavior and trace
#    The original task treated each framed interior as one sprite and tried a
#    small menu of whole-sprite transforms. This fusion changes two things at
#    once:
#    - the active rule is always connector_draw
#    - the source interior contains multiple disconnected 4-connected components,
#      and the connector rule must be applied inside each component's own bbox
#      rather than across the whole inner square.
#
#    A local component prototype is:
#        [[v, 0, v],
#         [v, 0, s],
#         [v, v, v]]
#    Local connector_draw adds only the top-middle cell:
#        [[v, v, v],
#         [v, 0, s],
#         [v, v, v]]
#    Old whole-canvas behavior would be wrong once two such components are
#    placed with a one-cell moat inside the same 7x7 interior: the global rule
#    would also bridge across that moat because it sees aligned same-color pixels
#    belonging to different components. The fused rule forbids that by cropping
#    each component before drawing.
#
# b) Complexity calibration
#    The inner square is expanded from the source task's common 4-6 range to a
#    fixed 7x7 so two 3x3 connected components can be shown with an explicit
#    separating row/column of zeros. Without that moat, D3/2 would often be
#    visually present but causally irrelevant because global connector_draw would
#    match the component-wise result.
#    Each frame still uses only two sprite colors: one variable connector color
#    and one stable accent color. This keeps the rule legible and avoids palette
#    ambiguity.
#
# c) Hidden-assumption audit for main()
#    - The source task assumed a single working canvas for fill-like rules.
#      Here that assumption is invalid, so the solver must enumerate 4-connected
#      components and crop each to its tight bbox before inference/application.
#    - stable_colors() and pick_var_color() still assume there is exactly one
#      transforming non-stable color. The generator preserves that.
#    - apply_componentwise() assumes the transform never needs to draw outside a
#      component's bbox. The generator uses a component prototype whose bridges
#      stay strictly inside the local crop.
#    - The pair-identification heuristic still relies on the shared non-zero
#      color set of the training pair. The generator keeps that invariant.
#
# d) Placement legality
#    All four outer frames remain non-overlapping and quadrant-separated on the
#    30x30 board. Inside a 7x7 interior, the generator places two 3x3 components
#    either horizontally or vertically with a guaranteed one-cell zero moat, so
#    components never overlap, never merge, and never leave the frame.
#
# e) Visual observability
#    Across examples, a human sees the same color draw short straight bridges
#    inside each disconnected piece, while the blank moat between pieces remains
#    empty. That repeated "bridge inside, not across" behavior is directly
#    visible from the training pair and does not depend on color ids or ordering.
#
# f) Visual cleanliness
#    Each component is only 3x3 and uses a single stable accent cell cluster, so
#    the framed interiors stay sparse even with two components present. The moat
#    between components prevents the scene from reading as one messy object.
#
# g) Change inferability
#    There is no arbitrary color remapping. The same variable color expands into
#    new bridge pixels and the accent color stays fixed. The extra D3 signal is
#    structural: the solver must notice that disconnected pieces repeat the same
#    rule independently.


H = W = 30

FRAME_SIZE = 9
INNER_SIZE = FRAME_SIZE - 2

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 6, 18
GEN_MAX_ATTEMPTS = 200

BASE_COMPONENT = np.array(
    [
        [1, 0, 1],
        [1, 0, 2],
        [1, 1, 1],
    ],
    dtype=int,
)


def find_square_frames(grid: np.ndarray, limit: int | None = None) -> list[tuple[int, int, int, int, int, int]]:
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


def component_bboxes4(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = arr != 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            out.append((min(rows), min(cols), max(rows), max(cols)))
    out.sort()
    return out


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
        return None
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


def infer_local_template(src: np.ndarray, dst: np.ndarray):
    st = stable_colors(src, dst)
    v = pick_var_color(dst, st) or pick_var_color(src, st)
    pred = connector_draw_transform(src, st, v)
    if np.array_equal(pred, dst):
        return ("connector_draw", {"stable": st})
    return None


def apply_local_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind != "connector_draw":
        return x.copy()
    st = set(params.get("stable", set()))
    return connector_draw_transform(x, st, pick_connector_color(x, st))


def _freeze_param_value(v):
    if isinstance(v, np.ndarray):
        return tuple(tuple(int(x) for x in row) for row in v.tolist())
    if isinstance(v, set):
        return tuple(sorted(int(x) for x in v))
    return v


def _param_signature(kind: str, params: dict) -> tuple:
    return (kind, tuple(sorted((k, _freeze_param_value(v)) for k, v in params.items())))


def infer_componentwise(src: np.ndarray, dst: np.ndarray):
    bboxes = component_bboxes4(src)
    if len(bboxes) < 2:
        return None
    covered = np.zeros_like(src, dtype=bool)
    first = None
    first_sig = None
    for r0, c0, r1, c1 in bboxes:
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
        local_src = src[r0 : r1 + 1, c0 : c1 + 1]
        local_dst = dst[r0 : r1 + 1, c0 : c1 + 1]
        tpl = infer_local_template(local_src, local_dst)
        if tpl is None:
            return None
        sig = _param_signature(tpl[0], tpl[1])
        if first is None:
            first = tpl
            first_sig = sig
        elif sig != first_sig:
            return None
    if np.any((dst != 0) & ~covered):
        return None
    return first


def apply_componentwise(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    bboxes = component_bboxes4(x)
    if not bboxes:
        return x.copy()
    y = np.zeros_like(x)
    for r0, c0, r1, c1 in bboxes:
        local = x[r0 : r1 + 1, c0 : c1 + 1]
        transformed = apply_local_template(kind, params, local)
        y[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return y


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a component-local connector rule from a framed training pair, then use it on a test sprite.

    Input structure:
    - The board contains several hollow square frames on a noisy background, and one frame is blank.
    - Two non-blank frames share the same non-zero palette and form the training pair; another non-blank frame is the test source.
    - In this fused variant, each non-blank interior contains multiple disconnected foreground components rather than one monolithic sprite.

    Rule:
    - Find the training pair as usual, then segment the source interior into 4-connected components.
    - For each component, crop to its own tight bbox and infer the same-color connector rule there: aligned pixels of the active color draw the shortest horizontal or vertical bridge through zeros, while stable accent pixels stay fixed.
    - Apply that same local connector rule independently to every disconnected component of the test interior, and return only the transformed inner square.

    Key insight:
    - The important visual cue is not just "same-color dots connect"; it is that they connect only inside each disconnected piece. A solver that treats the whole frame as one canvas will draw illegal bridges across the zero moat between components.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g, limit=6)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(g[r0 + 1 : r1, c0 + 1 : c1])

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        inner = inners[i]
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
        counts = [(i, int(np.count_nonzero(inners[i]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        tpl = infer_componentwise(src, dst)
        if tpl is None:
            continue
        kind, params = tpl
        st = set(params.get("stable", set()))
        v_src = pick_connector_color(src, st)
        v_dst = pick_var_color(dst, st)
        added = 0
        if v_src is not None and v_dst is not None:
            added = int(np.sum(dst == v_dst) - np.sum(src == v_src))
        comp_n = len(component_bboxes4(src))
        score = added * 100 + comp_n * 10 - abs(int(np.count_nonzero(src)) - int(np.count_nonzero(test_inner)))
        if best is None or score > best[0]:
            best = (score, kind, params)

    if best is None:
        return test_inner.copy()

    _, kind, params = best
    return apply_componentwise(kind, params, test_inner)


def colorize_blueprint(bp: np.ndarray, colors: list[int]) -> np.ndarray:
    y = np.zeros_like(bp)
    y[bp == 1] = colors[0]
    y[bp == 2] = colors[1]
    return y


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


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


def whole_connector_output(x: np.ndarray) -> np.ndarray:
    st = set()
    vals = [int(v) for v in np.unique(x) if int(v) != 0]
    if len(vals) >= 2:
        # In generated samples the rarer color is the stable accent.
        counts = [(int(np.sum(x == v)), v) for v in vals]
        counts.sort()
        st = {counts[0][1]}
    return connector_draw_transform(x, st, pick_connector_color(x, st))


def unique_component_blueprints() -> list[np.ndarray]:
    seen = set()
    out = []
    for blueprint in symmetries(BASE_COMPONENT):
        key = tuple(tuple(int(x) for x in row) for row in blueprint.tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(np.asarray(blueprint, dtype=int))
    return out


def build_component_variant(
    rng: random.Random,
    colors: list[int],
    blueprint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if blueprint is None:
        blueprint = rng.choice(unique_component_blueprints())
    src = colorize_blueprint(blueprint, colors)
    dst = apply_local_template("connector_draw", {"stable": {colors[1]}}, src)
    return src, dst


def build_multi_component_pair(colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    blueprints = unique_component_blueprints()
    for _ in range(64):
        chosen = rng.sample(blueprints, 2)
        locals_pair = [build_component_variant(rng, colors, blueprint=bp) for bp in chosen]
        layout = rng.choice(["horizontal", "vertical"])
        src = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
        dst = np.zeros_like(src)
        if layout == "horizontal":
            row0 = rng.choice([0, INNER_SIZE - 3])
            anchors = [(row0, 0), (row0, INNER_SIZE - 3)]
        else:
            col0 = rng.choice([0, INNER_SIZE - 3])
            anchors = [(0, col0), (INNER_SIZE - 3, col0)]
        for (r0, c0), (local_src, local_dst) in zip(anchors, locals_pair):
            paste(src, local_src, r0, c0, transparent=0)
            paste(dst, local_dst, r0, c0, transparent=0)
        if len(component_bboxes4(src)) != 2:
            continue
        if np.array_equal(src, dst):
            continue
        inferred = infer_componentwise(src, dst)
        if inferred is None or inferred[0] != "connector_draw":
            continue
        if np.array_equal(whole_connector_output(src), dst):
            continue
        return src, dst
    raise ValueError("failed to build multi-component connector pair")


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if sizes != {FRAME_SIZE}:
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
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


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


def _build_sprites(p1: list[int], p2: list[int], rng: random.Random):
    train_src, train_dst = build_multi_component_pair(p1, rng)
    inferred = infer_componentwise(train_src, train_dst)
    if inferred is None or inferred[0] != "connector_draw":
        return None

    test_src, _ = build_multi_component_pair(p2, rng)
    test_dst = apply_componentwise(inferred[0], inferred[1], test_src)

    if np.array_equal(test_src, test_dst):
        return None
    if np.array_equal(whole_connector_output(test_src), test_dst):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    if len(component_bboxes4(train_src)) != 2 or len(component_bboxes4(test_src)) != 2:
        return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    p1 = choose_palette(rng, 2, set())
    p2 = choose_palette(rng, 2, set(p1))

    built = _build_sprites(p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(FRAME_SIZE, rng)
    train_slots, test_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((INNER_SIZE, INNER_SIZE), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(FRAME_SIZE, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
