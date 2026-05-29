# Combo variant of 4c7dc4dd (index 15)
# combo_spec: D5/2+D11/0+D13/0
# Name: 三色定向边界提取

from collections import deque
import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, draw_line, paste


# Design analysis
# design_analysis:
#   a) concretization_and_behavior_trace:
#     - Scene layout stays close to the original task: a 30x30 noisy canvas, four same-size hollow square
#       frames, one blank interior, one training pair, and one test sprite.
#     - The fused rule is no longer "guess one of several transforms". It is always:
#       1) use the arrow between the two training frames to read source -> destination,
#       2) identify the three sprite roles by count,
#       3) keep the smallest-count stable color unchanged,
#       4) extract the 4-connected outer boundary of the two variable colors together,
#          preserving their original colors on surviving boundary cells.
#     - Concrete trace on a small 6x6 inner sprite:
#       original task:
#         - find the matching palette pair,
#         - try both directions,
#         - infer one rule from {fill/endpoints/sym/...}.
#       fused task:
#         - find the matching 3-color pair,
#         - read the arrow to fix direction,
#         - the large thick color loses interior pixels,
#         - the thin secondary color usually survives because every one of its pixels is already boundary,
#         - the tiny stable color remains unchanged and is excluded from the boundary computation.
#     - Difference from plain D5 boundary:
#       - boundary is not computed on a single-color sprite;
#       - the output retains two unchanged role colors (stable + thin secondary),
#         while only the thick primary region visibly shrinks.
#   b) complexity_calibration:
#     - To avoid ambiguity, every non-blank sprite uses exactly 3 non-zero colors with strictly ordered
#       counts: stable < secondary < primary.
#     - The primary color is generated as a thick, hole-free blob so boundary extraction is visible.
#     - The secondary color is generated as a 1-cell-thick line, so it acts as a visible second variable
#       role without introducing another large area that could obscure the boundary effect.
#     - The arrow only appears between horizontally or vertically adjacent training frames, which keeps the
#       directional cue legible.
#   c) implicit_assumptions_audit:
#     - Original code assumed the transform family had to be inferred by trying both directions; this is
#       invalid once direction is explicitly marked, so the solver now decodes the arrow first.
#     - Original code also assumed at most one moving color. That breaks in the 3-color setting, so the
#       solver re-identifies roles from color counts instead of using a single "variable color".
#     - A new assumption is introduced: counts of the three non-zero colors are distinct. The generator
#       enforces this on every non-blank sprite.
#     - Another new assumption is that the stable role is the smallest-count color. The generator fixes
#       stable count at 2 and keeps the secondary line longer than that.
#   d) placement_legality:
#     - Frames are rendered in four non-overlapping quadrant anchors with enough horizontal/vertical gap
#       for a thin arrow to fit between any allowed training pair.
#     - Sprite pixels live strictly inside frame interiors. The arrow is drawn only in the corridor
#       between the training frames and never overlaps a frame border.
#   e) visual_observability:
#     - A human can see the arrow before reasoning about the transform.
#     - The thick primary region visibly hollows into an outline, which is the main D5 signal.
#     - The secondary line survives unchanged, so the example clearly shows that not every non-stable
#       color shrinks the same way.
#     - The stable color is always the tiniest color count, giving a repeated cross-example cue.
#   f) visual_cleanliness:
#     - Stable cells are small markers near the primary blob but not overlapping it.
#     - Secondary cells are a thin disjoint line, so the three roles remain separable even in 5x5/6x6
#       interiors.
#     - Background uses only a 2-color stripe because the fused palette budget is tight; strong noise keeps
#       it from looking like a meaningful object.
#   g) change_inferability:
#     - No color-id mapping is hidden in the task. Colors only encode roles by count and geometry.
#     - The transform is structure-based: thick areas shrink to their exterior contour, thin areas remain,
#       and the smallest stable accent is copied through unchanged.
#   h) quantitative_core_behavior:
#     - Source primary count is forced to exceed secondary count by several cells.
#     - Boundary extraction removes at least 2 primary cells.
#     - Secondary and stable masks are preserved exactly from source to destination.
#   i) main_assumptions_vs_generator_guarantees:
#     - `extract_roles_from_sprite()` assumes exactly 3 non-zero colors with distinct counts:
#       guaranteed by the builders.
#     - `decode_training_direction()` assumes one isolated arrow component sits in the gap between the
#       paired training frames: guaranteed by structured placement and unique marker color.
#     - `infer_template()` assumes the smallest color is stable and that applying the fused boundary rule
#       to the training source yields the training destination: guaranteed by output-first generation and
#       final validation through `main(grid) == expected_output`.


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (2,)
BG_PERTURB_FRACTION = (0.40, 0.60)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
MIN_AXIS_GAP = 4
TRAIN_PAIR_CHOICES = ((0, 1), (2, 3), (0, 2), (1, 3))

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 8, 32
STABLE_COUNT = 2
SECONDARY_LEN_CHOICES = (3, 4)
PRIMARY_SRC_MIN = 8
PRIMARY_REMOVED_MIN = 2
PRIMARY_BUILD_RETRIES = 120
SECOND_RECT_PROB = 0.8
GEN_MAX_ATTEMPTS = 500


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


def nonzero_counts(arr: np.ndarray) -> dict[int, int]:
    return {int(c): int(np.sum(arr == c)) for c in np.unique(arr) if int(c) != 0}


def extract_roles_from_sprite(arr: np.ndarray) -> tuple[int, int, int] | None:
    counts = nonzero_counts(arr)
    if len(counts) != 3:
        return None
    items = sorted((cnt, color) for color, cnt in counts.items())
    if len({cnt for cnt, _ in items}) != 3:
        return None
    stable = items[0][1]
    secondary = items[1][1]
    primary = items[2][1]
    return stable, primary, secondary


def exterior_zero_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if not mask[r, c] and not ext[r, c]:
            ext[r, c] = True
            q.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def has_hole(mask: np.ndarray) -> bool:
    ext = exterior_zero_mask(mask)
    return bool(np.any((~mask) & ~ext))


def boundary_keep_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    ext = exterior_zero_mask(mask)
    keep = np.zeros((h, w), dtype=bool)
    rr, cc = np.where(mask)
    for r, c in zip(rr.tolist(), cc.tolist()):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or ext[nr, nc]:
                keep[r, c] = True
                break
    return keep


def boundary_transform(arr: np.ndarray, stable_color: int | None) -> np.ndarray:
    stable_mask = np.zeros_like(arr, dtype=bool)
    if stable_color is not None:
        stable_mask = arr == stable_color
    active = (arr != 0) & ~stable_mask
    keep = boundary_keep_mask(active)
    out = np.zeros_like(arr)
    if stable_color is not None:
        out[stable_mask] = stable_color
    out[keep] = arr[keep]
    return out


def primary_only_boundary(arr: np.ndarray, stable_color: int, primary_color: int) -> np.ndarray:
    out = np.zeros_like(arr)
    out[arr == stable_color] = stable_color
    keep = boundary_keep_mask(arr == primary_color)
    out[keep] = primary_color
    return out


def infer_template(src: np.ndarray, dst: np.ndarray) -> dict | None:
    roles = extract_roles_from_sprite(src)
    if roles is None:
        return None
    stable, primary, secondary = roles
    if tuple(colors_present(src, exclude={0})) != tuple(colors_present(dst, exclude={0})):
        return None
    if not np.array_equal(src == stable, dst == stable):
        return None
    if not np.array_equal(src == secondary, dst == secondary):
        return None
    if np.array_equal(src == primary, dst == primary):
        return None
    pred = boundary_transform(src, stable)
    if not np.array_equal(pred, dst):
        return None
    if int(np.sum(src == primary)) - int(np.sum(dst == primary)) < PRIMARY_REMOVED_MIN:
        return None
    return {"mode": "three_color_boundary"}


def apply_template(_params: dict, arr: np.ndarray) -> np.ndarray:
    roles = extract_roles_from_sprite(arr)
    if roles is None:
        return arr.copy()
    stable, _primary, _secondary = roles
    return boundary_transform(arr, stable)


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
    Solve a framed sprite analogy where a directed arrow and three color roles specify a contour rule.

    Input structure:
    - The canvas contains four same-size hollow square frames on a noisy background.
    - One frame interior is blank, two non-blank interiors share the same 3-color palette and form the
      training pair, and the remaining non-blank interior is the test sprite.
    - A thin single-color arrow lies between the two training frames and points from the example source
      to the example destination.

    Rule:
    - Match the training pair by identical non-zero palettes, then decode the arrow to determine the
      source -> destination direction.
    - In each 3-color sprite, treat the smallest-count color as stable, exclude it from boundary
      computation, and keep it unchanged; the other two colors form the active foreground whose 4-connected
      outer boundary is preserved with original colors.
    - Apply that same three-role boundary rule to the test sprite and return only the transformed interior.

    Key insight:
    - The arrow removes direction ambiguity, while the color counts reveal the stable role.
    - The large thick color shrinks to its exposed contour, the thin secondary color survives because it
      is already boundary-like, and the tiny stable accent is copied through unchanged.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
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
    sigs = [(i, tuple(colors_present(inners[i], exclude={0}))) for i in nonblank]

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return inners[blank_idx].copy()

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    directed_pair = decode_training_direction(g, frames, pair)
    if directed_pair is None:
        return inners[test_idx].copy()

    src = inners[directed_pair[0]]
    dst = inners[directed_pair[1]]
    test_inner = inners[test_idx]

    params = infer_template(src, dst)
    if params is None:
        return test_inner.copy()
    return apply_template(params, test_inner)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def random_primary_mask(n: int, rng: random.Random) -> np.ndarray:
    for _ in range(PRIMARY_BUILD_RETRIES):
        mask = np.zeros((n, n), dtype=bool)

        h1 = rng.randint(3, n)
        w1 = rng.randint(3, n)
        r1 = rng.randint(0, n - h1)
        c1 = rng.randint(0, n - w1)
        mask[r1 : r1 + h1, c1 : c1 + w1] = True

        if rng.random() < SECOND_RECT_PROB:
            for _ in range(40):
                h2 = rng.randint(2, n)
                w2 = rng.randint(2, n)
                r2 = rng.randint(0, n - h2)
                c2 = rng.randint(0, n - w2)
                rect2 = np.zeros((n, n), dtype=bool)
                rect2[r2 : r2 + h2, c2 : c2 + w2] = True
                if np.any(mask & rect2) and np.any(rect2 & ~mask):
                    mask |= rect2
                    break

        src_n = int(np.count_nonzero(mask))
        if src_n < PRIMARY_SRC_MIN or src_n > SPRITE_NZ_MAX - STABLE_COUNT - min(SECONDARY_LEN_CHOICES):
            continue
        if has_hole(mask):
            continue
        keep = boundary_keep_mask(mask)
        dst_n = int(np.count_nonzero(keep))
        if src_n - dst_n < PRIMARY_REMOVED_MIN:
            continue
        if dst_n <= max(SECONDARY_LEN_CHOICES):
            continue
        return mask
    raise ValueError("failed to build primary mask")


def _line_mask(n: int, orientation: str, start_r: int, start_c: int, length: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if orientation == "horizontal":
        mask[start_r, start_c : start_c + length] = True
    else:
        mask[start_r : start_r + length, start_c] = True
    return mask


def build_secondary_mask(n: int, primary_mask: np.ndarray, rng: random.Random) -> np.ndarray | None:
    candidates = []
    for length in SECONDARY_LEN_CHOICES:
        for orientation in ("horizontal", "vertical"):
            max_r = n if orientation == "horizontal" else n - length + 1
            max_c = n - length + 1 if orientation == "horizontal" else n
            for r0 in range(max_r):
                for c0 in range(max_c):
                    mask = _line_mask(n, orientation, r0, c0, length)
                    if np.any(mask & primary_mask):
                        continue
                    touching = False
                    rr, cc = np.where(mask)
                    for r, c in zip(rr.tolist(), cc.tolist()):
                        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if 0 <= nr < n and 0 <= nc < n and primary_mask[nr, nc]:
                                touching = True
                                break
                        if touching:
                            break
                    if touching:
                        continue
                    candidates.append(mask)
    if not candidates:
        return None
    return rng.choice(candidates)


def build_stable_mask(primary_mask: np.ndarray, secondary_mask: np.ndarray, rng: random.Random) -> np.ndarray | None:
    n = primary_mask.shape[0]
    blocked = primary_mask | secondary_mask
    candidates = []
    rr, cc = np.where(primary_mask)
    for r, c in zip(rr.tolist(), cc.tolist()):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < n and 0 <= nc < n and not blocked[nr, nc]:
                candidates.append((nr, nc))
    candidates = sorted(set(candidates))
    if len(candidates) < STABLE_COUNT:
        return None
    rng.shuffle(candidates)
    pair = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2:
                pair = [a, b]
                break
        if pair is not None:
            break
    if pair is None:
        pair = candidates[:STABLE_COUNT]
    mask = np.zeros_like(primary_mask, dtype=bool)
    for r, c in pair:
        mask[r, c] = True
    return mask


def build_three_color_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    stable_color, primary_color, secondary_color = colors
    for _ in range(PRIMARY_BUILD_RETRIES):
        primary_mask = random_primary_mask(n, rng)
        secondary_mask = build_secondary_mask(n, primary_mask, rng)
        if secondary_mask is None:
            continue
        stable_mask = build_stable_mask(primary_mask, secondary_mask, rng)
        if stable_mask is None:
            continue

        src = np.zeros((n, n), dtype=int)
        src[primary_mask] = primary_color
        src[secondary_mask] = secondary_color
        src[stable_mask] = stable_color
        dst = boundary_transform(src, stable_color)

        roles = extract_roles_from_sprite(src)
        if roles is None:
            continue
        stable, primary, secondary = roles
        if stable != stable_color or primary != primary_color or secondary != secondary_color:
            continue
        if tuple(colors_present(dst, exclude={0})) != tuple(sorted(colors)):
            continue
        if not np.array_equal(src == stable_color, dst == stable_color):
            continue
        if not np.array_equal(src == secondary_color, dst == secondary_color):
            continue
        if np.array_equal(src == primary_color, dst == primary_color):
            continue
        if int(np.sum(src == primary_color)) - int(np.sum(dst == primary_color)) < PRIMARY_REMOVED_MIN:
            continue
        if int(np.sum(dst == primary_color)) <= int(np.sum(src == secondary_color)):
            continue

        all_nonzero = boundary_transform(src, None)
        if np.array_equal(all_nonzero, dst):
            continue
        primary_only = primary_only_boundary(src, stable_color, primary_color)
        if np.array_equal(primary_only, dst):
            continue
        return src, dst
    raise ValueError("failed to build fused three-color pair")


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
    nonblank = [inner for inner in inners if not np.all(inner == 0)]
    if any(len(colors_present(inner, exclude={0})) != 3 for inner in nonblank):
        return False
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in nonblank]
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


def _valid_nonzero(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(n: int, p1: list[int], p2: list[int], rng: random.Random):
    train_src, train_dst = build_three_color_pair(n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    if inferred is None:
        return None

    test_src, _ = build_three_color_pair(n, p2, rng)
    test_dst = apply_template(inferred, test_src)

    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nonzero(arr):
            return None
        if len(colors_present(arr, exclude={0})) != 3:
            return None
        if extract_roles_from_sprite(arr) is None:
            return None
    if tuple(colors_present(train_src, exclude={0})) != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) != tuple(colors_present(test_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == tuple(colors_present(train_src, exclude={0})):
        return None
    if np.array_equal(test_src, test_dst):
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_scene_colors(used: set[int], rng: random.Random) -> tuple[int, int, list[int]]:
    free = [c for c in ALL_COLORS if c not in used]
    nonzero_free = [c for c in free if c != 0]
    if len(nonzero_free) < 2:
        raise ValueError("not enough nonzero free colors for frame and marker")
    frame_color = rng.choice(nonzero_free)
    free = [c for c in free if c != frame_color]
    nonzero_free = [c for c in free if c != 0]
    marker_color = rng.choice(nonzero_free)
    free = [c for c in free if c != marker_color]
    bg_sizes = [p for p in BG_PERIOD_CHOICES if len(free) >= p]
    if not bg_sizes:
        raise ValueError("not enough colors left for background")
    bg_palette = rng.sample(free, rng.choice(bg_sizes))
    return frame_color, marker_color, bg_palette


def _assign_slots(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int], int]:
    train_pair = rng.choice(TRAIN_PAIR_CHOICES)
    directed = train_pair if rng.random() < 0.5 else (train_pair[1], train_pair[0])
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
    draw_direction_arrow(grid, frame_boxes[train_direction[0]], frame_boxes[train_direction[1]], marker_color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    p1 = choose_palette(rng, 3, set())
    p2 = choose_palette(rng, 3, set(p1))

    built = _build_sprites(n, p1, p2, rng)
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
