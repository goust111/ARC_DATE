# Combo variant of 9385bd28 (index 20)
# combo_spec: D4/3+D10/2+D17/2
# Name: 隐式颜色映射+面积编码优先级+无操作干扰

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# concretization:
#   - The explicit bottom-left legend is removed entirely. The input now consists only
#     of 4-5 colored marker objects on a plain background. Each color appears in exactly
#     one object, usually an L-pair and occasionally a 2x2 block.
#   - The key visible quantity is each object's bounding-box area. Example trace with
#     four objects of areas 35, 24, 12, 8 and colors [A, B, C, D]:
#       source task: a legend would directly say A->x, B->y, C->z, D->w and give order.
#       combo task: no legend exists. Instead, the area ranking itself creates the
#       hidden lookup chain. A (largest) fills with B, B fills with C, and C/D are
#       no-ops. The same ranking also gives processing order: A first, then B.
#   - This makes the three requested changes interact instead of living separately:
#       1. implicit mapping: fill color comes from another marker color, not a legend;
#       2. area-coded priority: larger bbox means earlier execution;
#       3. no-op interference: the 2-3 smallest objects only contribute colors to the
#          chain and never perform a fill of their own.
# complexity calibration:
#   - The generator uses 4-5 objects and fixes the active count at exactly two. That
#     guarantees 2-3 genuine no-op distractors in every sample without flooding the grid.
#   - Bounding-box areas are forced to be distinct, so "largest / second / third" is
#     visible and unambiguous. This avoids tie-based hidden logic.
#   - A 2x2 block may still appear, but only as a small tail object; that keeps the
#     original shape vocabulary while preventing the block from dominating the area order.
# implicit-assumption review of main():
#   - The original `_find_legend(...)` assumption is gone; the solver must recover all
#     actionable structure from the colored objects themselves.
#   - The old loop `for legend entry in bottom->top order` depended on externally given
#     order and fill colors. In the combo variant, both are inferred from the same sorted
#     object list, so generator and solver must agree on unique bbox areas.
#   - The old `if len(marker_pts) < 2: continue` skip path was for empty legend rows.
#     Here there are no empty rows; instead the tail objects are explicit no-ops because
#     they rank below the two active objects.
#   - Future-marker protection is still required. Earlier large fills can cross cells
#     inside smaller bboxes, so later marker pixels stay protected exactly as before.
# placement legality:
#   - Objects are placed with non-overlapping marker pixels, within bounds, and only use
#     one color per object. The generator also rejects any scene whose bbox areas repeat.
#   - Because the solver infers the chain from the global area ranking, generation must
#     guarantee at least three objects, and in practice always emits 4-5.
# visual observability:
#   - A human can compare input/output pairs and see that only the two largest marker
#     objects create new filled diamonds, while the smaller objects stay unchanged.
#   - The new fill colors are not arbitrary: each active region is recolored using a
#     color already present on the next smaller object, so the mapping is visible in-grid.
# cleanliness / inferability:
#   - No ordering step depends on color ids, black=0, scan order of disconnected pieces,
#     or any other non-visual attribute. Size is the only ranking key that matters.
#   - Limiting transformation to the top two ranks keeps the scene legible: the no-op
#     objects act as distractors without creating a tangled stack of overlapping fills.


ACTIVE_OBJECTS = 2


def _l_anchor_present(
    pts: set[tuple[int, int]],
    r: int,
    c: int,
    arm1: tuple[int, int],
    arm2: tuple[int, int],
) -> bool:
    """
    Marker corners are L-shapes:
      TL: (r,c), (r,c+1), (r+1,c)
      TR: (r,c), (r,c-1), (r+1,c)
      BL: (r,c), (r,c+1), (r-1,c)
      BR: (r,c), (r,c-1), (r-1,c)
    """
    dr1, dc1 = arm1
    dr2, dc2 = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """Mask of shortest-path cells between corners a and b inside one bbox."""
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _object_record(color: int, pts: set[tuple[int, int]]) -> dict:
    """Extract bbox/type metadata for one marker color."""
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    bh = rmax - rmin + 1
    bw = cmax - cmin + 1
    bbox_area = bh * bw

    is_block = len(pts) == 4 and bh == 2 and bw == 2 and all(
        (r, c) in pts for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)
    )

    return {
        "color": int(color),
        "pts": pts,
        "rmin": int(rmin),
        "rmax": int(rmax),
        "cmin": int(cmin),
        "cmax": int(cmax),
        "bbox_area": int(bbox_area),
        "is_block": bool(is_block),
    }


def _scene_objects(g: np.ndarray) -> tuple[int, list[dict]]:
    """Return background color and marker objects sorted by descending bbox area."""
    bg = get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    objects: list[dict] = []
    for color in colors:
        pts = set(map(tuple, np.argwhere(g == color)))
        assert pts, "every non-background color must define an object"
        rec = _object_record(color, pts)
        if not rec["is_block"]:
            assert len(pts) >= 6, "non-block objects must be L-pairs"
            assert rec["rmin"] != rec["rmax"] and rec["cmin"] != rec["cmax"], "L-pair bbox must be 2D"
        objects.append(rec)

    assert len(objects) >= 3, "scene must contain at least three colored objects"
    objects.sort(key=lambda obj: (-obj["bbox_area"], obj["rmin"], obj["cmin"], obj["color"]))
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the largest marker objects using colors borrowed from the next-smaller ones.

    Input structure:
    - There is no explicit legend anywhere in the grid.
    - Instead, the non-background colors themselves form 4-5 marker objects, each appearing once as either
      an opposite-corner L-pair or a solid 2x2 block.
    - The objects have distinct bounding-box areas; the smallest 2-3 are present only as distractors.

    Rule:
    - Rank all colored objects by bounding-box area, largest first.
    - Process only the two largest objects. The largest fills with the second-largest object's color, and the
      second-largest fills with the third-largest object's color. All remaining smaller objects do nothing.
    - Geometry is solved exactly as in the base task: a 2x2 block is recolored directly, while an L-pair
      reveals a bounding box whose opposite diagonal defines the diamond to fill.
    - While processing, preserve marker pixels of the current object and of every smaller-ranked object.

    Key insight:
    - The bbox-area ranking replaces the legend entirely: the same visible size order simultaneously tells you
      which objects act, in what order they act, and which later color each active object should borrow.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _bg, objects = _scene_objects(g)
    out = g.copy()

    active_count = min(ACTIVE_OBJECTS, len(objects) - 1)
    color_masks = {obj["color"]: (g == obj["color"]) for obj in objects}

    for i in range(active_count):
        obj = objects[i]
        fill_color = objects[i + 1]["color"]

        protect = np.zeros_like(g, dtype=bool)
        for future in objects[i + 1 :]:
            protect |= color_masks[future["color"]]

        if obj["is_block"]:
            block = np.zeros_like(g, dtype=bool)
            for r, c in obj["pts"]:
                block[r, c] = True
            out[block & (~protect)] = int(fill_color)
            continue

        protect |= color_masks[obj["color"]]

        rmin, rmax = obj["rmin"], obj["rmax"]
        cmin, cmax = obj["cmin"], obj["cmax"]
        tl = (rmin, cmin)
        tr = (rmin, cmax)
        bl = (rmax, cmin)
        br = (rmax, cmax)
        pts = obj["pts"]

        tl_anchor = _l_anchor_present(pts, tl[0], tl[1], (0, 1), (1, 0))
        tr_anchor = _l_anchor_present(pts, tr[0], tr[1], (0, -1), (1, 0))
        bl_anchor = _l_anchor_present(pts, bl[0], bl[1], (0, 1), (-1, 0))
        br_anchor = _l_anchor_present(pts, br[0], br[1], (0, -1), (-1, 0))

        score_tlbr = int(tl_anchor) + int(br_anchor)
        score_trbl = int(tr_anchor) + int(bl_anchor)
        assert score_tlbr != 0 or score_trbl != 0, "at least one diagonal must have an L-anchor"
        assert score_tlbr != score_trbl, "diagonal scores must not tie"

        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill_color)

    return out


# --------------- generator ---------------

# ── CFG: canvas & palette ──────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)
OBJECT_COUNT_RANGE = (4, 5)

# ── CFG: active/no-op split ────────────────────────────────────────
NOOP_COUNT_RANGE = (2, 3)  # enforced by choosing 4-5 objects while keeping only top two active

# ── CFG: object shapes ─────────────────────────────────────────────
BLOCK_PROB = 0.35          # optional 2x2 block among the smallest objects only
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.5

# ── CFG: placement & retries ──────────────────────────────────────
SIZE_RETRIES = 80
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
GEN_RETRIES = 400

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the six cells for an L-pair at opposite bbox corners."""
    if diag == "TLBR":
        anchors = [("TL", rmin, cmin), ("BR", rmax, cmax)]
    else:
        anchors = [("TR", rmin, cmax), ("BL", rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        for dr, dc in _L_OFFSETS[name]:
            cells.append((r + dr, c + dc))
    return cells


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    """True if (h, w) avoids an ambiguous L-anchor tie."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _candidate_l_sizes(H: int, W: int) -> list[tuple[int, int]]:
    """All legal L-pair bbox sizes that fit in the canvas."""
    sizes: list[tuple[int, int]] = []
    for bh in range(2, min(BBOX_H_MAX, H) + 1):
        for bw in range(2, min(BBOX_W_MAX, W) + 1):
            if _l_pair_valid_bbox(bh, bw):
                sizes.append((bh, bw))
    return sizes


def _pick_sizes(H: int, W: int, n_objects: int, include_block: bool) -> list[tuple[str, int, int]]:
    """
    Pick object shapes with strictly decreasing bbox areas.

    Returns entries as (kind, bh, bw) already sorted largest -> smallest.
    """
    l_sizes = _candidate_l_sizes(H, W)
    if not l_sizes:
        return []

    big_pool = [sz for sz in l_sizes if 18 <= sz[0] * sz[1] <= 45]
    mid_pool = [sz for sz in l_sizes if 12 <= sz[0] * sz[1] <= 30]
    small_pool = [sz for sz in l_sizes if 8 <= sz[0] * sz[1] <= 18]
    if not big_pool or not mid_pool or len(small_pool) < 2:
        return []

    for _ in range(SIZE_RETRIES):
        chosen: list[tuple[str, int, int]] = []
        used_areas: set[int] = set()

        bh, bw = random.choice(big_pool)
        chosen.append(("L", bh, bw))
        used_areas.add(bh * bw)

        second = [sz for sz in mid_pool if sz[0] * sz[1] not in used_areas and sz[0] * sz[1] < chosen[-1][1] * chosen[-1][2]]
        if not second:
            continue
        bh, bw = random.choice(second)
        chosen.append(("L", bh, bw))
        used_areas.add(bh * bw)

        tail_needed = n_objects - 2
        tail: list[tuple[str, int, int]] = []
        if include_block:
            tail.append(("block", 2, 2))
            used_areas.add(4)
            tail_needed -= 1

        prev_area = chosen[-1][1] * chosen[-1][2]
        ok = True
        for _tail_idx in range(tail_needed):
            pool = [
                sz for sz in small_pool
                if sz[0] * sz[1] not in used_areas and sz[0] * sz[1] < prev_area
            ]
            if not pool:
                ok = False
                break
            bh, bw = random.choice(pool)
            tail.append(("L", bh, bw))
            used_areas.add(bh * bw)
            prev_area = bh * bw
        if not ok:
            continue

        chosen.extend(tail)
        chosen.sort(key=lambda item: -(item[1] * item[2]))
        areas = [bh * bw for _, bh, bw in chosen]
        if len(areas) == n_objects and len(set(areas)) == n_objects:
            return chosen

    return []


def _bbox_area(kind: str, bh: int, bw: int) -> int:
    return bh * bw


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True when two inclusive rectangles overlap."""
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    bh: int,
    bw: int,
    retries: int,
    can_place,
    do_place,
) -> bool:
    """Try to place one L-pair with fixed bbox size."""
    for _ in range(retries):
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        bbox = (rmin, rmin + bh - 1, cmin, cmin + bw - 1)
        if can_place(cells, bbox):
            do_place(cells, color, bbox)
            return True
    return False


def _try_place_block(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    """Try to place a 2x2 block."""
    for _ in range(retries):
        r0 = random.randint(0, H - 2)
        c0 = random.randint(0, W - 2)
        cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
        bbox = (r0, r0 + 1, c0, c0 + 1)
        if can_place(cells, bbox):
            do_place(cells, color, bbox)
            return True
    return False


def _scene_signature(g: np.ndarray) -> dict:
    """Small quantitative summary used for generator validation and external checks."""
    _bg, objects = _scene_objects(g)
    areas = [obj["bbox_area"] for obj in objects]
    changed_fill_colors = [objects[i + 1]["color"] for i in range(min(ACTIVE_OBJECTS, len(objects) - 1))]
    bboxes = [(obj["rmin"], obj["rmax"], obj["cmin"], obj["cmax"]) for obj in objects]
    disjoint_bboxes = True
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _rectangles_overlap(bboxes[i], bboxes[j]):
                disjoint_bboxes = False
                break
        if not disjoint_bboxes:
            break
    return {
        "object_count": len(objects),
        "areas": areas,
        "active_count": min(ACTIVE_OBJECTS, len(objects) - 1),
        "noop_count": len(objects) - min(ACTIVE_OBJECTS, len(objects) - 1),
        "changed_fill_colors": changed_fill_colors,
        "disjoint_bboxes": disjoint_bboxes,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    include_block = n_objects == 5 and random.random() < BLOCK_PROB

    shape_plan = _pick_sizes(H, W, n_objects, include_block)
    if not shape_plan:
        return None

    colors = random.sample([c for c in range(10) if c != bg], n_objects)
    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    placed_bboxes: list[tuple[int, int, int, int]] = []

    def can_place(cells, bbox):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c]:
                return False
        for other_bbox in placed_bboxes:
            if _rectangles_overlap(bbox, other_bbox):
                return False
        return True

    def do_place(cells, color, bbox):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        placed_bboxes.append(bbox)

    for color, (kind, bh, bw) in zip(colors, shape_plan):
        if kind == "block":
            if not _try_place_block(H, W, color, BLOCK_RETRIES, can_place, do_place):
                return None
        else:
            if not _try_place_l_pair(H, W, color, bh, bw, LPAIR_RETRIES, can_place, do_place):
                return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    sig = _scene_signature(grid)
    if sig["object_count"] != n_objects:
        return None
    if len(set(sig["areas"])) != n_objects:
        return None
    if sig["active_count"] != ACTIVE_OBJECTS:
        return None
    if sig["noop_count"] not in NOOP_COUNT_RANGE:
        return None
    if not sig["disjoint_bboxes"]:
        return None

    # Fusion check: only the two largest objects should cause any change, and each uses
    # the next smaller object's marker color rather than an external legend color.
    changed_colors = set(int(c) for c in np.unique(out[grid != out]))
    expected_colors = set(sig["changed_fill_colors"])
    if not expected_colors.issubset(changed_colors):
        return None
    objects = _scene_objects(grid)[1]
    active_boxes = [
        (obj["rmin"], obj["rmax"], obj["cmin"], obj["cmax"])
        for obj in objects[:ACTIVE_OBJECTS]
    ]
    for r, c in np.argwhere(grid != out):
        if not any(r0 <= r <= r1 and c0 <= c <= c1 for r0, r1, c0, c1 in active_boxes):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
