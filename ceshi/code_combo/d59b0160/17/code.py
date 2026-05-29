# Combo variant of d59b0160 (index 17)
# combo_spec: D2/3+D17/1
# Name: 匹配分量镜像翻转+仅删除最大匹配分量

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - The scene keeps the original structure: a 16x16 grid with bg=7, a 4x4 top-left legend framed by color 3,
#      and several separated rectangle-shaped 4-connected components elsewhere.
#    - The legend interior still lists exactly three target colors.
#    - Original task: every component containing all three target colors is deleted.
#    - D2/3 alone would mirror every matching component horizontally in its own bbox.
#    - D17/1 alone would delete only the unique largest matching component and keep the smaller matches unchanged.
#    - Combo behavior:
#      1. collect all matching components,
#      2. compare their areas,
#      3. delete only the unique largest match,
#      4. horizontally mirror every other matching component in place,
#      5. leave non-matching components unchanged.
#    - Example: if matching component areas are 24, 15, 10, then area-24 disappears while area-15 and area-10 stay
#      but their internal left/right color layouts are reversed. In the source task all three would vanish; in D17/1
#      only area-24 would vanish; in this combo the smaller matches survive but visibly change.
#
# b) Complexity calibration
#    - The generator always creates 2 or 3 matching components so the area competition is real.
#    - Matching areas are forced to be distinct, giving one unambiguous largest deletion target.
#    - Every kept matching component has width >= 3 and is forced to differ from its horizontal mirror, so the flip
#      is visible rather than collapsing into "looks unchanged".
#    - The number of non-matching components stays fixed at 3, preserving repeated evidence for the legend filter
#      without overcrowding the 16x16 board.
#
# c) Implicit assumption audit
#    - `_bg_boundary` still assumes the outer frame is mostly background; neither deletion nor in-place reflection
#      changes the frame, so this remains valid.
#    - `_find_legend` still assumes the same top-left legend format; the generator preserves it exactly.
#    - `components_mask(g != bg, connectivity=4)` still works because every non-legend object is one disjoint
#      rectangle of non-bg cells.
#    - The old baseline could act on matches immediately. The combo cannot: it must first collect all matches, because
#      the action depends on relative area among the matching set.
#    - `_mirror_component_in_bbox` assumes the component's bbox contains only that component. The generator enforces
#      full-rectangle components with one-cell padding, so no foreign pixels can occupy the bbox.
#
# d) Placement legality
#    - Each patch is placed with `random_free_location(..., padding=1, padding_connectivity=8)`, preventing overlap,
#      edge overflow, and accidental component merging.
#    - Reflection never expands the bbox, so any legal input placement stays legal in the output.
#    - Area bounds for matching and non-matching components are kept moderate so that 5-6 objects fit reliably.
#
# e) Visual observability
#    - Humans can infer the competition by seeing multiple objects that all contain the legend palette while only the
#      largest one disappears.
#    - They can infer the second rule because the surviving matching objects do not merely stay: their left/right
#      internal arrangements reverse.
#    - The ranking uses visible geometry only (component area), not hidden properties such as color ids or creation
#      order.
#
# f) Visual cleanliness
#    - Objects remain separated rectangular blocks on a uniform background, with one compact legend in the corner.
#    - Only one transform type is used for survivors: horizontal mirror in place. No mixed rotations or translations
#      are introduced.
#
# g) Change inferability
#    - No new color mapping is introduced. Colors keep their original role as membership evidence for the legend set.
#    - The combo is deducible from a few examples because both sub-rules repeat: "largest matching one disappears" and
#      "smaller matching ones mirror".
#
# New solver assumptions and generator guarantees
#    - main assumes there are at least two matching components if the combo behavior is to be demonstrated; the
#      generator guarantees that.
#    - main resolves the largest match by area using component size. The generator guarantees a unique largest match.
#    - main mirrors kept matches from the original input grid rather than from partially edited output; the generator
#      guarantees those mirrors are visibly different from the input arrangement.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend color set is the set of colors in the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    assert False, "no legend border found in grid"


def _component_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    rs = pts[:, 0]
    cs = pts[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _mirror_component_in_bbox(out: np.ndarray, g: np.ndarray, pts: np.ndarray, bg: int) -> None:
    r0, c0, r1, c1 = _component_bbox(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in pts:
        patch[int(r) - r0, int(c) - c0] = int(g[int(r), int(c)])
    paste(out, np.fliplr(patch), r0, c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete only the largest matching component, and horizontally mirror the other matching ones.

    Input structure:
    - The outer frame reveals the background color.
    - A small square legend sits in the top-left corner, with one shared border color along its bottom edge and right edge.
    - Outside the legend, the non-background cells form separated 4-connected multicolor components on the background.

    Rule:
    - Read the target color set from the legend interior.
    - Find every non-background 4-connected component whose color set contains all target colors.
    - Among those matches, erase only the largest component by painting it with the background.
    - For every other matching component, reflect its content left-to-right inside its own bounding box; leave non-matching components unchanged.

    Key insight: The legend still decides which components qualify, but matching is no longer the whole story; the largest qualifying object disappears, while the smaller qualifying ones survive with a visible horizontal mirror transformation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    matches: list[np.ndarray] = []
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[int(r), int(c)]) for r, c in pts}
        if legend_colors.issubset(cols):
            matches.append(pts)

    if not matches:
        return out

    largest_idx = max(range(len(matches)), key=lambda i: int(matches[i].shape[0]))
    for i, pts in enumerate(matches):
        if i == largest_idx:
            for r, c in pts:
                out[int(r), int(c)] = bg
        else:
            _mirror_component_in_bbox(out, g, pts, bg)
    return out


# --------------- generator ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of colored pixels among the base-0 rectangle) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Variant-specific component mix --
NONMATCH_COUNT = 3
MATCH_COUNTS = [2, 3]
MATCH_WEIGHTS = [4, 1]
MIN_DELETE_AREA = 18
MAX_DELETE_AREA = 36
MIN_KEEP_MATCH_AREA = 8
MAX_NONMATCH_AREA = 24
MIN_MATCH_AREA_GAP = 4

# -- Variant observability controls --
PATCH_RETRIES = 80
MIN_FLIP_DIFF = 2

# -- Generation retry budget --
MAX_RETRIES = 700


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box (border color 3) at top-left and place target colors."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _sample_dims(
    *,
    min_area: int = 1,
    max_area: int | None = None,
    min_w: int = 2,
) -> tuple[int, int]:
    """Sample component dimensions from the original shape families, with optional area and width bounds."""
    for _ in range(200):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        area = h * w
        if w < min_w:
            continue
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        return h, w

    candidates: list[tuple[int, int]] = []
    for h in range(3, GRID_H + 1):
        for w in range(min_w, GRID_W + 1):
            area = h * w
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            candidates.append((h, w))
    if not candidates:
        raise ValueError("no component dimensions satisfy the requested bounds")
    return random.choice(candidates)


def _sample_matching_dims(match_count: int) -> list[tuple[int, int]]:
    """Return matching dimensions with one unique largest area and flippable smaller matches."""
    for _ in range(200):
        delete_h, delete_w = _sample_dims(min_area=MIN_DELETE_AREA, max_area=MAX_DELETE_AREA)
        delete_area = delete_h * delete_w
        max_other_area = delete_area - MIN_MATCH_AREA_GAP
        if max_other_area < MIN_KEEP_MATCH_AREA:
            continue

        dims = [(delete_h, delete_w)]
        used_areas = {delete_area}
        ok = True
        for _ in range(match_count - 1):
            for _ in range(200):
                h, w = _sample_dims(
                    min_area=MIN_KEEP_MATCH_AREA,
                    max_area=max_other_area,
                    min_w=3,
                )
                area = h * w
                if area in used_areas:
                    continue
                dims.append((h, w))
                used_areas.add(area)
                break
            else:
                ok = False
                break
        if ok:
            return dims
    raise RuntimeError("failed to sample matching dimensions with a unique largest area")


def _random_patch(h: int, w: int, colors: list[int], n_colored: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    fill = [random.choice(colors) for _ in range(n_colored)]
    random.shuffle(fill)
    rs, cs = zip(*coords)
    patch[rs, cs] = fill
    return patch


def _flip_diff(patch: np.ndarray) -> int:
    return int(np.count_nonzero(patch != patch[:, ::-1]))


def _component_patch(h: int, w: int, targets: list[int], kind: str) -> np.ndarray | None:
    """Build an h x w rectangle with base color 0 and sprinkled colored pixels."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    non_targets = [v for v in PALETTE if v not in targets]

    for _ in range(PATCH_RETRIES):
        if kind in {"delete_match", "keep_match"}:
            lo_here = max(lo, NUM_TARGETS)
            n_colored = random.randint(lo_here, hi)
            coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
            colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
            random.shuffle(colors)

            patch = np.zeros((h, w), dtype=int)
            rs, cs = zip(*coords)
            patch[rs, cs] = colors
            if kind == "keep_match" and _flip_diff(patch) < MIN_FLIP_DIFF:
                continue
            return patch

        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        n_colored = random.randint(lo, hi)
        patch = _random_patch(h, w, allowed, n_colored)
        cols = {int(v) for v in np.unique(patch)}
        if set(targets).issubset(cols):
            continue
        return patch

    return None


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output follows the combo rule."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    match_count = random.choices(MATCH_COUNTS, weights=MATCH_WEIGHTS)[0]
    matching_dims = _sample_matching_dims(match_count)

    specs: list[dict[str, object]] = []
    delete_h, delete_w = matching_dims[0]
    specs.append({"h": delete_h, "w": delete_w, "kind": "delete_match"})
    for h, w in matching_dims[1:]:
        specs.append({"h": h, "w": w, "kind": "keep_match"})
    for _ in range(NONMATCH_COUNT):
        h, w = _sample_dims(min_area=MIN_KEEP_MATCH_AREA, max_area=MAX_NONMATCH_AREA)
        specs.append({"h": h, "w": w, "kind": "nonmatch"})

    random.shuffle(specs)

    changed_flipped = 0
    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        kind = str(spec["kind"])
        patch = _component_patch(h, w, targets, kind)
        if patch is None:
            return None
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        if kind == "delete_match":
            continue
        if kind == "keep_match":
            mirrored = patch[:, ::-1]
            paste(out, mirrored, r, c)
            if not np.array_equal(patch, mirrored):
                changed_flipped += 1
            continue
        paste(out, patch, r, c)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    bg = _bg_boundary(inp)
    legend_colors = _find_legend(inp, bg)[2]
    match_infos: list[tuple[int, int]] = []
    for pts in components_mask(inp != bg, connectivity=4):
        cols = {int(inp[int(r), int(c)]) for r, c in pts}
        if legend_colors.issubset(cols):
            match_infos.append((int(pts.shape[0]), _flip_diff(inp[
                _component_bbox(pts)[0]:_component_bbox(pts)[2] + 1,
                _component_bbox(pts)[1]:_component_bbox(pts)[3] + 1,
            ])))

    if len(match_infos) < 2:
        return None
    areas = sorted(area for area, _diff in match_infos)
    if areas[-1] == areas[-2]:
        return None
    if changed_flipped == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
