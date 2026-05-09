# Combo variant of 800d221b (index 33)
# combo_spec: D6/2+D15/0+D16/2
# Name: 方向匹配+角落hub+对边种子
#
# Design analysis:
# - Concrete trace:
#   In the base task, a non-hub P component is recolored from the seed region it physically touches most.
#   In this combo, the 3x3 hub is anchored in a corner, so only two outward directions are available. A
#   concrete top-left example has one seed on the right edge and one seed on the bottom edge. The right-going
#   branch component can stop in empty background without touching any seed at all, yet it still recolors to
#   the majority color of the right-edge seed because direction, not contact, is the binding signal. The same
#   holds for the downward branch versus the bottom-edge seed. The output therefore changes even when the old
#   contact-based rule has zero evidence.
# - Complexity calibration:
#   The fused variant keeps only two seed reference regions and two branch directions. That is the smallest
#   stable configuration where the direction-to-seed mapping is visible, repeated, and not cluttered. The
#   generator varies the corner among all four orientations and varies how many hub-adjacent pixels each branch
#   family uses, so the center-pixel vote remains meaningful without introducing extra object types.
# - Hidden-assumption audit against the original solver:
#   * "Every recolored P component touches a seed" is deliberately false now. The new solver must classify
#     hub-connected P components by which face of the corner hub they leave from, not by seed adjacency.
#   * "The largest solid P square can be searched anywhere" is tightened: because the combo requires a corner
#     hub, the solver looks for the unique solid 3x3 P square anchored at a grid corner.
#   * "All non-hub P components should be recolored" is also false now: detached seed frames stay P because
#     only hub-connected branches participate in the directional recolor.
#   * "Seed placement can be inferred from arbitrary edge contact" is specialized: each seed must touch exactly
#     one of the two far edges associated with the chosen corner orientation. The generator enforces that.
# - Placement legality:
#   Base generation is done in a canonical top-left layout, then mirrored to the other three corners. The two
#   seed rectangles occupy the two far edges of that corner, their framed zones are separated, and the two
#   branch fans stay in the opposite quadrant so they never overlap seeds, go out of bounds, or touch each
#   other orthogonally.
# - Visual observability:
#   A human sees a corner hub, two detached edge seeds, and two groups of spokes pointing toward those edge
#   seeds. Because the branches do not contact the seeds, the recolor cannot be explained by local adjacency;
#   the only stable cue is the initial direction from the hub.
# - Visual cleanliness:
#   The picture still uses only four colors and only the original object families: hub, thin P branches, and
#   framed seed rectangles. The new information is carried by corner placement and empty space, not by extra
#   markers layered on top of the grid.
# - Inferability / anti-degeneration metric:
#   Accepted samples are constructed so every hub-connected branch component has zero physical seed contact.
#   A counterfactual contact-based solver would therefore leave those components unchanged, while the fused
#   solver recolors them by direction. This makes the D6/2 change active on every generated sample.
# - New solver assumptions guaranteed by the generator:
#   there is exactly one corner hub; exactly two detached seed interiors exist on the two far edges of that
#   corner; each hub-connected P component touches exactly one exposed hub face; and the hub-center vote has a
#   unique winner because the two direction families use different counts of hub-adjacent pixels.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT, ROOT / "solvers"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.insert(0, extra_s)

from arc_utils import get_bg, label_mask


HUB_SIZE = 3

GRID_SIZE_MIN = 15
GRID_SIZE_MAX = 24

SEED_INTERIOR_MIN = 3
SEED_INTERIOR_MAX = 5
MINORITY_FRAC_LO = 0.12
MINORITY_FRAC_HI = 0.38

GEN_OUTER_RETRIES = 200

_CORNERS = ("tl", "tr", "bl", "br")
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _corner_anchor(corner: str, H: int, W: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, W - HUB_SIZE
    if corner == "bl":
        return H - HUB_SIZE, 0
    return H - HUB_SIZE, W - HUB_SIZE


def _corner_exposed_faces(corner: str) -> tuple[str, str]:
    if corner == "tl":
        return "right", "bottom"
    if corner == "tr":
        return "left", "bottom"
    if corner == "bl":
        return "right", "top"
    return "left", "top"


def _transform_from_base(g: np.ndarray, corner: str) -> np.ndarray:
    out = np.array(g, copy=True)
    if corner in ("tr", "br"):
        out = np.fliplr(out)
    if corner in ("bl", "br"):
        out = np.flipud(out)
    return out


def _seed_majorities(g: np.ndarray, bg: int, P: int) -> dict[str, int]:
    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    majors: dict[str, int] = {}
    H, W = g.shape
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        major = int(Counter(vals).most_common(1)[0][0])
        rows = pts[:, 0]
        cols = pts[:, 1]
        side_hits = []
        if int(rows.min()) == 0:
            side_hits.append("top")
        if int(rows.max()) == H - 1:
            side_hits.append("bottom")
        if int(cols.min()) == 0:
            side_hits.append("left")
        if int(cols.max()) == W - 1:
            side_hits.append("right")
        assert len(side_hits) == 1, "each seed must touch exactly one grid edge"
        majors[side_hits[0]] = int(major)
    assert len(majors) == 2, "combo expects exactly two seed regions"
    return majors


def _find_path_color_and_hub(g: np.ndarray, bg: int) -> tuple[int, str, np.ndarray]:
    H, W = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    candidates: list[tuple[int, str]] = []
    for col in colors:
        mask = g == int(col)
        for corner in _CORNERS:
            r0, c0 = _corner_anchor(corner, H, W)
            block = mask[r0 : r0 + HUB_SIZE, c0 : c0 + HUB_SIZE]
            if block.shape == (HUB_SIZE, HUB_SIZE) and bool(np.all(block)):
                candidates.append((int(col), corner))
    assert len(candidates) == 1, "expected unique corner hub color"
    P, corner = candidates[0]
    r0, c0 = _corner_anchor(corner, H, W)
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r0 + HUB_SIZE, c0 : c0 + HUB_SIZE] = True
    return int(P), corner, hub


def _component_face(
    pts: np.ndarray,
    hub: np.ndarray,
    corner: str,
    H: int,
    W: int,
) -> str | None:
    faces = Counter()
    allowed = set(_corner_exposed_faces(corner))
    for r, c in pts:
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS4:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if not bool(hub[rr, cc]):
                continue
            if dr == 1:
                faces["top"] += 1
            elif dr == -1:
                faces["bottom"] += 1
            elif dc == 1:
                faces["left"] += 1
            else:
                faces["right"] += 1
    if not faces:
        return None
    live = {side: count for side, count in faces.items() if side in allowed}
    assert len(live) == 1, "hub-connected component must leave from exactly one exposed face"
    return next(iter(live))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a corner-anchored branch fan by directional seed matching.

    Input structure:
    - bg is the most common color.
    - Path color P forms a solid 3x3 hub in one grid corner plus detached thin P branches leaving only from
      the two exposed faces of that corner.
    - Exactly two seed regions appear as framed rectangles on the two far edges associated with that corner;
      their mixed interiors determine the two available recolor targets.

    Rule:
    - Find P by locating the unique 3x3 solid corner hub and infer which corner it occupies.
    - Compute the majority color of each seed interior and index those majorities by edge side.
    - Recolor only the P components that are connected to the hub: a component inherits the majority color of
      the seed on the edge matching the component's initial hub face, even if that component never touches a
      seed physically.
    - Recolor the hub center pixel to the color with the most hub-adjacent recolored branch pixels.

    Key insight:
    Local adjacency is a decoy in this combo. The decisive cue is the branch's first outward direction from the
    corner hub, which selects the seed on the corresponding far edge.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    P, corner, hub = _find_path_color_and_hub(g, bg)
    seed_major = _seed_majorities(g, bg, P)
    exposed = set(_corner_exposed_faces(corner))
    assert exposed == set(seed_major), "seeds must lie on the two far edges for the hub corner"

    out = np.array(g, copy=True)
    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    hub_touch_colors = Counter()
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        face = _component_face(pts, hub, corner, H, W)
        if face is None:
            continue
        new_col = int(seed_major[face])
        for r, c in pts:
            out[int(r), int(c)] = int(new_col)
        hub_touch_colors[int(new_col)] += sum(
            1
            for r, c in pts
            for dr, dc in _DIRS4
            if 0 <= int(r + dr) < H
            and 0 <= int(c + dc) < W
            and bool(hub[int(r + dr), int(c + dc)])
        )

    assert hub_touch_colors, "hub must have at least one outgoing branch component"
    cr = int(np.argwhere(hub)[:, 0].min() + HUB_SIZE // 2)
    cc = int(np.argwhere(hub)[:, 1].min() + HUB_SIZE // 2)
    out[cr, cc] = int(hub_touch_colors.most_common(1)[0][0])
    return out


def _fill_seed_interior(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    majority: int,
    minority: int,
) -> None:
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    n = len(cells)
    min_minor = max(1, int(round(n * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(round(n * MINORITY_FRAC_HI)))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n - n_minor:
        n_minor = max(1, (n - 1) // 2)
    random.shuffle(cells)
    for r, c in cells[:n_minor]:
        g[r, c] = int(minority)
    for r, c in cells[n_minor:]:
        g[r, c] = int(majority)


def _draw_frame(
    g: np.ndarray,
    P: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    edge: str,
) -> None:
    if edge != "top" and r0 > 0:
        g[r0 - 1, c0 : c1 + 1] = int(P)
    if edge != "bottom" and r1 + 1 < g.shape[0]:
        g[r1 + 1, c0 : c1 + 1] = int(P)
    if edge != "left" and c0 > 0:
        g[r0 : r1 + 1, c0 - 1] = int(P)
    if edge != "right" and c1 + 1 < g.shape[1]:
        g[r0 : r1 + 1, c1 + 1] = int(P)
    if edge not in ("top", "left") and r0 > 0 and c0 > 0:
        g[r0 - 1, c0 - 1] = int(P)
    if edge not in ("top", "right") and r0 > 0 and c1 + 1 < g.shape[1]:
        g[r0 - 1, c1 + 1] = int(P)
    if edge not in ("bottom", "left") and r1 + 1 < g.shape[0] and c0 > 0:
        g[r1 + 1, c0 - 1] = int(P)
    if edge not in ("bottom", "right") and r1 + 1 < g.shape[0] and c1 + 1 < g.shape[1]:
        g[r1 + 1, c1 + 1] = int(P)


def _add_right_fan(g: np.ndarray, P: int, max_c: int, touch_count: int) -> None:
    rows = list(range(touch_count))
    cur_end = max_c
    for r in rows:
        cur_end = max(HUB_SIZE + 1, cur_end - random.randint(0, 2))
        g[r, HUB_SIZE : cur_end + 1] = int(P)


def _add_down_fan(g: np.ndarray, P: int, max_r: int, touch_count: int) -> None:
    cols = list(range(touch_count))
    cur_end = max_r
    for c in cols:
        cur_end = max(HUB_SIZE + 1, cur_end - random.randint(0, 2))
        g[HUB_SIZE : cur_end + 1, c] = int(P)


def _generate_base_top_left() -> np.ndarray | None:
    H = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    W = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)
    g = np.full((H, W), bg, dtype=int)

    g[:HUB_SIZE, :HUB_SIZE] = int(P)

    ih_r = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw_r = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    ih_b = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw_b = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)

    right_ir0_hi = max(4, H // 2 - ih_r)
    bottom_ic0_hi = max(4, W // 2 - iw_b)
    if right_ir0_hi < 4 or bottom_ic0_hi < 4:
        return None

    right_ir0 = random.randint(4, right_ir0_hi)
    right_ir1 = right_ir0 + ih_r - 1
    right_ic1 = W - 1
    right_ic0 = W - iw_r

    bottom_ic0 = random.randint(4, bottom_ic0_hi)
    bottom_ic1 = bottom_ic0 + iw_b - 1
    bottom_ir1 = H - 1
    bottom_ir0 = H - ih_b

    _fill_seed_interior(g, right_ir0, right_ic0, right_ir1, right_ic1, A, B)
    _fill_seed_interior(g, bottom_ir0, bottom_ic0, bottom_ir1, bottom_ic1, B, A)
    _draw_frame(g, P, right_ir0, right_ic0, right_ir1, right_ic1, "right")
    _draw_frame(g, P, bottom_ir0, bottom_ic0, bottom_ir1, bottom_ic1, "bottom")

    if not (right_ir0 > HUB_SIZE + 1 and bottom_ic0 > HUB_SIZE + 1):
        return None

    touch_pair = random.choice(((2, 3), (3, 2)))
    right_touch, down_touch = touch_pair

    right_max_c = min(right_ic0 - 3, max(HUB_SIZE + 2, W // 2 + random.randint(-1, 1)))
    down_max_r = min(bottom_ir0 - 3, max(HUB_SIZE + 2, H // 2 + random.randint(-1, 1)))
    if right_max_c <= HUB_SIZE or down_max_r <= HUB_SIZE:
        return None

    _add_right_fan(g, P, right_max_c, right_touch)
    _add_down_fan(g, P, down_max_r, down_touch)

    if np.count_nonzero(g == bg) <= (H * W) // 2:
        return None
    return g


def _seed_contact_free_metric(inp: np.ndarray) -> tuple[int, int]:
    bg = int(get_bg(inp))
    P, _corner, hub = _find_path_color_and_hub(inp, bg)
    seed_mask = (inp != int(bg)) & (inp != int(P))
    branch_mask = (inp == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    seed_touch_free = 0
    total_hub_components = 0
    H, W = inp.shape
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches_hub = any(
            0 <= int(r + dr) < H
            and 0 <= int(c + dc) < W
            and bool(hub[int(r + dr), int(c + dc)])
            for r, c in pts
            for dr, dc in _DIRS4
        )
        if not touches_hub:
            continue
        total_hub_components += 1
        touches_seed = any(
            0 <= int(r + dr) < H
            and 0 <= int(c + dc) < W
            and bool(seed_mask[int(r + dr), int(c + dc)])
            for r, c in pts
            for dr, dc in _DIRS4
        )
        if not touches_seed:
            seed_touch_free += 1
    return seed_touch_free, total_hub_components


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        base = _generate_base_top_left()
        if base is None:
            continue
        corner = random.choice(_CORNERS)
        inp = _transform_from_base(base, corner).astype(int)
        try:
            main(inp)
            free_count, total_count = _seed_contact_free_metric(inp)
            if total_count != 2 or free_count != total_count:
                continue
        except Exception:
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")
