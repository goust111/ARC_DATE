# Combo variant of 4c7dc4dd (index 25)
# combo_spec: D5/0+D7/1+D20/0
# Name: 棋盘背景下的形态学消歧

# Design analysis
# ===============
# a) Concrete behavior and trace
#    The preserved scaffold is still the original ARC analogy scene: a 30x30 canvas,
#    four same-size hollow square frames, one blank interior, one training pair, and
#    one held-out test sprite. The fusion changes two places together:
#    - outside the frames, the background is now a noisy 2-color checkerboard rather
#      than diagonal stripes;
#    - inside the matched training pair, the true rule is morphology, but the example
#      is chosen so a checker-based distractor also fits.
#    The operational ambiguity is intentionally local and visual:
#    - There are two balanced morphology branches:
#      1. erosion: a solid 5-cell straight strip shrinks to its 3-cell interior core;
#      2. dilation: a solid 3-cell straight strip grows into its 4-neighbor checker ribbon.
#    - Each training example is intentionally checker-ambiguous:
#      1. the erosion example also matches "keep the strip's checker core";
#      2. the dilation example also matches "complete the strip's checker ribbon".
#    - Test source: a bent path with the same checker coloring discipline.
#      Morphology still applies, but the checker-strip distractor no longer applies
#      because the sprite is not a single straight strip. So the solver must carry
#      both candidates from training and validate them on the test sprite.
#    Small concrete traces on a 6x6 interior:
#      erosion train src row segment: 0 1 2 1 2 1 0
#      erosion train dst: keep the middle 3 cells
#      dilation train src row segment: 0 0 1 2 1 0 0
#      dilation train dst: thicken it into the local checker ribbon
#      test src: an L-shaped path
#      test dst: the same path after the chosen morphology operator
#    Difference from the original task:
#    - original code chose among six unrelated transforms by first-match heuristics;
#    - this fusion narrows the true transform family to morphology, introduces a
#      checker-shaped distractor because of the new background semantics, and makes
#      test-time validation part of the reasoning chain.
#
# b) Complexity calibration
#    The D5 variant keeps both dilation and erosion active, but each branch is
#    calibrated to the same visual pattern family:
#    - erosion uses a 5->3 straight strip, so the checker-core distractor is visible;
#    - dilation uses a 3-cell straight strip whose thickened output is also readable
#      as a checker-ribbon completion;
#    - in both cases the test sprite is bent, so the straight-strip distractor drops
#      out and exactly one morphology rule remains applicable.
#    Sprites are parity-colored with exactly two non-zero colors. This makes the
#    checker motif explicit without introducing arbitrary color remapping.
#
# c) Implicit assumptions audited from the original main()
#    - Frame finding still assumes the noisy background does not accidentally create
#      extra hollow monochrome squares; the generator keeps quadrant placement and
#      validates that exactly four frames are detected.
#    - Training-pair identification still depends on matching non-zero color sets.
#      That would break if erosion collapsed to one color, so the generator uses a
#      5-cell strip whose 3-cell core still contains both parities/colors.
#    - The original solver assumed the training pair uniquely determined one rule.
#      That is no longer true. `main()` therefore collects every training-compatible
#      template, applies them to the test sprite, and keeps only the candidates that
#      remain applicable there.
#    - The new solver assumes sprite colors follow checker parity consistently. The
#      generator enforces that by recoloring every occupied cell from its local
#      interior parity.
#    - The checker distractor only makes sense on solid straight strips. The test
#      generator intentionally violates that precondition so the ambiguity is resolved
#      by visible geometry, not hidden color ids.
#
# d) Placement legality
#    All four outer frames keep the original non-overlapping quadrant layout. The
#    generator fixes the inner size to 6x6 (outer size 8) so the 5-cell strip and
#    bent test path both fit comfortably with margin. Background is rendered first,
#    then the full frame square is zeroed, then the hollow border is drawn, then the
#    sprite is pasted, so objects neither overlap nor leak outside their windows.
#
# e) Visual observability
#    A human can see the checkerboard story because every sprite uses two colors tied
#    to parity, and the background uses the same alternating motif. The training pair
#    really is ambiguous by inspection: "remove the two end pixels" and "keep the
#    checker strip's interior core" agree on the strip example. The test pair breaks
#    that because the bent path is no longer a single strip while morphology still
#    reads naturally.
#
# f) Visual cleanliness
#    Only one sprite appears inside each frame and each interior is cleared to black
#    before stamping. The checkerboard noise lives outside the frames, so it cues the
#    distractor without obscuring the local sprite geometry.
#
# g) Change inferability and generator guarantees
#    No arbitrary color permutation is hidden in the task. Colors simply encode local
#    checker parity. The generator guarantees:
#    - exactly one blank frame;
#    - exactly one duplicated non-zero color signature pair;
#    - the training pair admits exactly one morphology explanation plus one
#      checker-strip explanation;
#    - the test sprite rejects the checker-strip explanation but accepts the same
#      morphology operator;
#    - `main(generate_input())` returns the morphology output.

import random

import numpy as np

from arc_utils import colors_present, paste


H = W = 30
FRAME_SIZE_CHOICES = (8,)
INNER_SIZE = 6
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERTURB_FRACTION = (0.30, 0.45)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
GEN_MAX_ATTEMPTS = 400
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
RULES = ("erode", "dilate")
_GEN_RULE_INDEX = 0


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
        if len(picked) >= 6:
            break
    return picked


def nonzero_color_signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(colors_present(arr, exclude={0}))


def parity_palette(arr: np.ndarray) -> tuple[int, int] | None:
    even_colors = {int(arr[r, c]) for r, c in np.argwhere(arr != 0) if ((int(r) + int(c)) & 1) == 0}
    odd_colors = {int(arr[r, c]) for r, c in np.argwhere(arr != 0) if ((int(r) + int(c)) & 1) == 1}
    if not even_colors or not odd_colors:
        return None
    if len(even_colors) != 1 or len(odd_colors) != 1:
        return None
    return next(iter(even_colors)), next(iter(odd_colors))


def recolor_mask(mask: np.ndarray, even_color: int, odd_color: int) -> np.ndarray:
    h, w = mask.shape
    rr, cc = np.indices((h, w))
    out = np.zeros((h, w), dtype=int)
    out[mask & (((rr + cc) & 1) == 0)] = even_color
    out[mask & (((rr + cc) & 1) == 1)] = odd_color
    return out


def erode4_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    remove = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        deg = 0
        if rr > 0 and bool(mask[rr - 1, cc]):
            deg += 1
        if rr + 1 < h and bool(mask[rr + 1, cc]):
            deg += 1
        if cc > 0 and bool(mask[rr, cc - 1]):
            deg += 1
        if cc + 1 < w and bool(mask[rr, cc + 1]):
            deg += 1
        if deg == 1:
            remove.append((rr, cc))
    for rr, cc in remove:
        out[rr, cc] = False
    return out


def dilate4_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        if rr > 0:
            out[rr - 1, cc] = True
        if rr + 1 < h:
            out[rr + 1, cc] = True
        if cc > 0:
            out[rr, cc - 1] = True
        if cc + 1 < w:
            out[rr, cc + 1] = True
    return out


def component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = set()
    count = 0
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        count += 1
        stack = [cell]
        seen.add(cell)
        while stack:
            rr, cc = stack.pop()
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return count


def straight_segment_info(mask: np.ndarray) -> tuple[str, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        seg = sorted(int(c) for _, c in coords)
        if seg == list(range(seg[0], seg[-1] + 1)):
            return ("h", row, seg[0], seg[-1])
    if len(cols) == 1:
        col = cols[0]
        seg = sorted(int(r) for r, _ in coords)
        if seg == list(range(seg[0], seg[-1] + 1)):
            return ("v", col, seg[0], seg[-1])
    return None


def checker_strip_core(arr: np.ndarray) -> np.ndarray | None:
    pal = parity_palette(arr)
    if pal is None:
        return None
    info = straight_segment_info(arr != 0)
    if info is None:
        return None
    orientation, fixed, lo, hi = info
    if hi - lo + 1 < 5:
        return None
    core_mask = np.zeros_like(arr, dtype=bool)
    if orientation == "h":
        core_mask[fixed, lo + 1 : hi] = True
    else:
        core_mask[lo + 1 : hi, fixed] = True
    if int(np.count_nonzero(core_mask)) < 3:
        return None
    return recolor_mask(core_mask, pal[0], pal[1])


def checker_strip_ribbon(arr: np.ndarray) -> np.ndarray | None:
    pal = parity_palette(arr)
    if pal is None:
        return None
    info = straight_segment_info(arr != 0)
    if info is None:
        return None
    ribbon_mask = dilate4_mask(arr != 0)
    if np.array_equal(ribbon_mask, arr != 0):
        return None
    return recolor_mask(ribbon_mask, pal[0], pal[1])


def infer_templates(src: np.ndarray, dst: np.ndarray) -> list[tuple[str, dict]]:
    if nonzero_color_signature(src) != nonzero_color_signature(dst):
        return []
    pal = parity_palette(src)
    if pal is None or parity_palette(dst) != pal:
        return []
    out: list[tuple[str, dict]] = []

    eroded = recolor_mask(erode4_mask(src != 0), pal[0], pal[1])
    if not np.array_equal(eroded, src) and np.array_equal(eroded, dst):
        out.append(("erode", {"palette": pal}))

    dilated = recolor_mask(dilate4_mask(src != 0), pal[0], pal[1])
    if not np.array_equal(dilated, src) and np.array_equal(dilated, dst):
        out.append(("dilate", {"palette": pal}))

    checker = checker_strip_core(src)
    if checker is not None and np.array_equal(checker, dst):
        out.append(("checker_core", {"palette": pal}))

    ribbon = checker_strip_ribbon(src)
    if ribbon is not None and np.array_equal(ribbon, dst):
        out.append(("checker_ribbon", {"palette": pal}))

    return out


def apply_template(kind: str, arr: np.ndarray) -> np.ndarray | None:
    pal = parity_palette(arr)
    if pal is None:
        return None
    mask = arr != 0
    if kind == "erode":
        return recolor_mask(erode4_mask(mask), pal[0], pal[1])
    if kind == "dilate":
        return recolor_mask(dilate4_mask(mask), pal[0], pal[1])
    if kind == "checker_core":
        return checker_strip_core(arr)
    if kind == "checker_ribbon":
        return checker_strip_ribbon(arr)
    return None


def solve_triplet_disambiguated(src_a: np.ndarray, src_b: np.ndarray, test_inner: np.ndarray) -> tuple[np.ndarray, dict]:
    candidates = []
    for direction, (src, dst) in enumerate(((src_a, src_b), (src_b, src_a))):
        for kind, params in infer_templates(src, dst):
            test_out = apply_template(kind, test_inner)
            candidates.append(
                {
                    "direction": direction,
                    "kind": kind,
                    "train_src": src,
                    "train_dst": dst,
                    "test_out": test_out,
                    "applicable": test_out is not None and not np.array_equal(test_out, test_inner),
                }
            )

    applicable = [cand for cand in candidates if cand["applicable"]]
    if not applicable:
        return test_inner.copy(), {"candidate_count": len(candidates), "applicable_count": 0}

    groups: list[dict] = []
    for cand in applicable:
        matched = None
        for group in groups:
            if np.array_equal(group["output"], cand["test_out"]):
                matched = group
                break
        if matched is None:
            groups.append({"output": cand["test_out"], "members": [cand]})
        else:
            matched["members"].append(cand)

    if len(groups) == 1:
        best = groups[0]["members"][0]
    else:
        best = None
        for group in groups:
            morphs = [cand for cand in group["members"] if cand["kind"] in {"erode", "dilate"}]
            choice = morphs[0] if morphs else group["members"][0]
            if best is None:
                best = choice
                continue
            current_is_morph = choice["kind"] in {"erode", "dilate"}
            best_is_morph = best["kind"] in {"erode", "dilate"}
            if current_is_morph and not best_is_morph:
                best = choice

    metrics = {
        "candidate_count": len(candidates),
        "applicable_count": len(applicable),
        "kinds": [cand["kind"] for cand in candidates],
        "applicable_kinds": [cand["kind"] for cand in applicable],
        "winner": best["kind"],
        "unique_output_count": len(groups),
    }
    return best["test_out"].copy(), metrics


def extract_triplet(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    frames = find_square_frames(grid)
    if len(frames) < 3:
        return None
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_idx = next((i for i, inner in enumerate(inners) if inner.size > 0 and np.all(inner == 0)), None)
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))
    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, nonzero_color_signature(inners[i])) for i in nonblank]
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return None
    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return None
    return inners[pair[0]], inners[pair[1]], inners[remaining[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover an ambiguous framed checker-sprite rule, then resolve it on the held-out test sprite.

    Input structure:
    - The grid contains four non-overlapping hollow square frames on a noisy checkerboard background.
    - One frame interior is blank, two interiors form the training pair, and the remaining non-blank interior is the test sprite.
    - Every non-zero sprite cell is colored by local checker parity, so the training pair shares one two-color signature while the test sprite uses a different two-color signature.

    Rule:
    - Match the training pair by identical non-zero color sets.
    - Enumerate every training-compatible transform: the true four-neighbor morphology rule and the checker-strip distractor that happens to fit the same straight-strip example.
    - Apply those candidates to the test sprite and keep the one that still makes sense there; return that transformed interior.

    Key insight:
    - The training strip is intentionally over-explained: it supports one morphology reading and one checker-strip reading, so the real decision only becomes visible on the bent test sprite.
    """
    g = np.asarray(grid, dtype=int)
    triplet = extract_triplet(g)
    if triplet is None:
        return g.copy()
    src_a, src_b, test_inner = triplet
    out, _ = solve_triplet_disambiguated(src_a, src_b, test_inner)
    return out


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def paint_checker_cells(shape: tuple[int, int], cells: list[tuple[int, int]], even_color: int, odd_color: int) -> np.ndarray:
    arr = np.zeros(shape, dtype=int)
    for r, c in cells:
        arr[r, c] = even_color if ((r + c) & 1) == 0 else odd_color
    return arr


def rotate_cells(cells: list[tuple[int, int]], quarter_turns: int, n: int) -> list[tuple[int, int]]:
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(quarter_turns % 4):
        out = [(c, n - 1 - r) for r, c in out]
    return out


def shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def place_cells_within(cells: list[tuple[int, int]], n: int, rng: random.Random) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    dr = rng.randint(-min_r, n - 1 - max_r)
    dc = rng.randint(-min_c, n - 1 - max_c)
    return shift_cells(cells, dr, dc)


def build_train_pair(kind: str, n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if kind == "erode":
        base = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    else:
        base = [(2, 1), (2, 2), (2, 3)]
    rot = rng.choice([0, 1])
    cells = rotate_cells(base, rot, n)
    cells = place_cells_within(cells, n, rng)
    src = paint_checker_cells((n, n), cells, colors[0], colors[1])
    if kind == "erode":
        dst = recolor_mask(erode4_mask(src != 0), colors[0], colors[1])
    else:
        dst = recolor_mask(dilate4_mask(src != 0), colors[0], colors[1])
    return src, dst


def build_test_pair(kind: str, n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if kind == "erode":
        base = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
    else:
        base = [(1, 1), (1, 2), (2, 2)]
    rot = rng.randint(0, 3)
    cells = rotate_cells(base, rot, 5)
    cells = place_cells_within(cells, n, rng)
    src = paint_checker_cells((n, n), cells, colors[0], colors[1])
    if kind == "erode":
        dst = recolor_mask(erode4_mask(src != 0), colors[0], colors[1])
    else:
        dst = recolor_mask(dilate4_mask(src != 0), colors[0], colors[1])
    return src, dst


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


def render_background(frame_boxes: list[tuple[int, int, int, int]], bg_colors: list[int], rng: random.Random) -> np.ndarray:
    rr, cc = np.indices((H, W))
    grid = np.where(((rr + cc) & 1) == 0, bg_colors[0], bg_colors[1]).astype(int)
    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    cells = [(int(r), int(c)) for r, c in zip(*np.where(~reserved))]
    lo, hi = BG_PERTURB_FRACTION
    perturb_n = int(round(len(cells) * rng.uniform(lo, hi)))
    noise_choices = list(bg_colors) + [0]
    for r, c in rng.sample(cells, perturb_n):
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


def variant_metrics_from_triplet(train_src: np.ndarray, train_dst: np.ndarray, test_src: np.ndarray) -> dict:
    train_templates = infer_templates(train_src, train_dst)
    train_kinds = [kind for kind, _ in train_templates]
    morph_kind = "erode" if "erode" in train_kinds else "dilate" if "dilate" in train_kinds else None
    checker_kind = "checker_core" if "checker_core" in train_kinds else "checker_ribbon" if "checker_ribbon" in train_kinds else None
    checker_test = apply_template(checker_kind, test_src) if checker_kind is not None else None
    morph_test = apply_template(morph_kind, test_src) if morph_kind is not None else None
    solved, solve_metrics = solve_triplet_disambiguated(train_src, train_dst, test_src)
    return {
        "train_kinds": train_kinds,
        "morph_kind": morph_kind,
        "checker_kind": checker_kind,
        "train_ambiguous": (
            morph_kind is not None
            and checker_kind is not None
            and set(train_kinds) == {morph_kind, checker_kind}
        ),
        "checker_applicable_on_test": checker_test is not None and not np.array_equal(checker_test, test_src),
        "morph_applicable_on_test": morph_test is not None and not np.array_equal(morph_test, test_src),
        "resolved_to_morph": morph_kind is not None and solve_metrics.get("winner") == morph_kind,
        "solver_output_changes_test": not np.array_equal(solved, test_src),
        "solver_unique_output": solve_metrics.get("unique_output_count") == 1,
    }


def variant_metrics(grid: np.ndarray) -> dict:
    triplet = extract_triplet(np.asarray(grid, dtype=int))
    if triplet is None:
        return {"train_ambiguous": False}
    return variant_metrics_from_triplet(*triplet)


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if sizes != {FRAME_SIZE_CHOICES[0]}:
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [nonzero_color_signature(inner) for inner in inners if np.any(inner != 0)]
    dup_count = sum(
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    triplet = extract_triplet(grid)
    if triplet is None:
        return False
    train_src, train_dst, test_src = triplet
    metrics = variant_metrics_from_triplet(train_src, train_dst, test_src)
    return (
        np.array_equal(main(grid), expected_output)
        and metrics["train_ambiguous"]
        and not metrics["checker_applicable_on_test"]
        and metrics["morph_applicable_on_test"]
        and metrics["resolved_to_morph"]
        and metrics["solver_output_changes_test"]
        and metrics["solver_unique_output"]
    )


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_colors = rng.sample([c for c in ALL_COLORS if c not in used and c != frame_color], 2)
    return frame_color, bg_colors


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
    bg_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_colors, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def next_rule_kind() -> str:
    global _GEN_RULE_INDEX
    kind = RULES[_GEN_RULE_INDEX % len(RULES)]
    _GEN_RULE_INDEX += 1
    return kind


def _gen_attempt(kind: str) -> np.ndarray | None:
    rng = random.Random()
    s = FRAME_SIZE_CHOICES[0]
    n = INNER_SIZE

    train_colors = choose_palette(rng, 2, set())
    test_colors = choose_palette(rng, 2, set(train_colors))
    train_src, train_dst = build_train_pair(kind, n, train_colors, rng)
    test_src, test_dst = build_test_pair(kind, n, test_colors, rng)

    if not all(_valid_nz(arr) for arr in (train_src, train_dst, test_src, test_dst)):
        return None
    if nonzero_color_signature(train_src) != nonzero_color_signature(train_dst):
        return None
    if nonzero_color_signature(test_src) == nonzero_color_signature(train_src):
        return None

    metrics = variant_metrics_from_triplet(train_src, train_dst, test_src)
    if not (
        metrics["train_ambiguous"]
        and not metrics["checker_applicable_on_test"]
        and metrics["morph_applicable_on_test"]
        and metrics["resolved_to_morph"]
        and metrics["solver_unique_output"]
    ):
        return None

    frame_color, bg_colors = _choose_frame_and_bg(set(train_colors) | set(test_colors), rng)
    positions = quadrant_positions(s, rng)
    train_slots, test_slot = _assign_slots(positions, rng)
    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, bg_colors, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    kind = next_rule_kind()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(kind)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
