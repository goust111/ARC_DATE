# Combo variant of dbff022c (index 13)
# combo_spec: D8/0+D11/1
# Name: 连通分量内外判定 + 精灵嵌套

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Keep the edge palette mechanic: a 2xN or Nx2 legend on the frame maps shell-border
#   colors to fill colors.
# - Replace the original "independent hollow sprites" layout with a mandatory nested
#   stack: one large outer shell contains one broken middle shell, and that middle shell
#   contains one tiny inner shell.
# - The middle shell has a deliberate gap, so its cavity merges into the large shell's
#   cavity. This makes connected-component reasoning visible in every sample.
# - Trace on a sample:
#   1. A bbox-style or adjacency-only intuition might still treat the space inside the
#      middle shell as "its hole" because the middle shell visually surrounds that area.
#   2. The combo solver instead checks edge reachability. Because the middle shell has a
#      gap, that space can escape into the outer cavity, so it is not the middle shell's
#      interior.
#   3. The merged region is owned by the outer shell, while the tiny inner hole is still
#      owned by the tiny shell because that one remains closed.
#
# b) Complexity calibration
# - The generator uses one clean three-level nested structure instead of many unrelated
#   sprites. This keeps the topology readable while still forcing multi-layer reasoning.
# - The main topological twist is repeated rather than multiplied: every sample uses one
#   broken middle shell plus one closed tiny shell. That is enough to expose the D8 change
#   without exploding the rule space.
# - Palette size stays small (3-4 entries), so the visible mapping remains inferable from
#   examples. One extra unused mapping may appear, but never many.
#
# c) Implicit-assumption audit of the original main()
# - The original `_fill_enclosed_bg_regions` assumes every non-boundary bg component can be
#   assigned by dominant adjacent color alone. The nested tail breaks that assumption.
# - The original logic also assumes a hole belongs to the shell that merely touches it most.
#   In the combo task, ownership depends on which shell disconnects the region from the grid
#   edge when treated as the wall.
# - The original in-place update pattern is risky once shells are nested, because a later
#   decision must depend on the original topology, not on earlier fills. The new solver reads
#   from an immutable snapshot and writes to a separate output grid.
# - Palette detection itself does not rely on the old scattered layout, so `_best_palette`
#   and `_palette_mapping` can be preserved.
#
# d) Placement legality
# - The palette stays on the left or right edge and gets a 1-cell moat, so it is visually
#   separate and cannot be mistaken for part of a shell.
# - The outer shell is always at least one cell away from the global frame, guaranteeing that
#   its interior is not automatically external.
# - The middle shell fits strictly inside the outer shell with a bg moat on all sides.
# - The tiny inner shell fits strictly inside the middle shell's main cavity.
# - In the open-middle mode, the gap is placed so the middle cavity merges with the outer
#   annulus but the tiny shell remains isolated inside that merged region.
#
# e) Visual observability
# - Every sample contains at least one bg region that touches multiple mapped shell colors.
#   This makes the "pick the true enclosing shell" signal visible in the I/O pairs.
# - The broken-middle mode is visible by eye: one nested cavity stops behaving like its own
#   interior once the gap appears, even though it still sits inside the large shell.
# - The tail/bar is not decorative noise; it creates repeated evidence that adjacency count
#   alone is insufficient.
#
# f) Visual cleanliness
# - The play area only contains four semantic objects: the palette, the outer shell, the
#   middle shell with one tail/bar, and the tiny inner shell.
# - All shells are simple ring-like shapes. No extra markers, text, or scattered distractors
#   are added, so the nesting is easy to parse.
#
# g) Color-change inferability
# - Every fill color is justified by the visible edge legend; no numeric ordering or hidden
#   color preference is used.
# - Only three shell colors are active in the nested structure, which keeps the examples
#   readable even when three different fill colors appear in the output.
# - The broken-middle case remains inferable because no new color rule is introduced: the same
#   palette mapping applies, but the owning shell changes with topology.
#
# 9) New solver assumptions to check against the generator
# - `_select_region_border` assumes each relevant shell is one connected non-bg component.
#   The generator builds each shell as exactly one connected component.
# - The solver assumes the true owner of a region is adjacent to that region. The generator's
#   nested rings guarantee this for every fillable interior.
# - The solver prefers enclosing adjacent components with the smallest bbox area if more than
#   one candidate blocks the edge. The generator's three-level nesting makes the immediate
#   parent shell the smallest enclosing adjacent shell.
# - The solver assumes the palette is isolated from shells by at least one cell so palette
#   pixels do not become candidate owners. The generator enforces that moat.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[0, :]
        inner = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(outer_colors | inner_colors) < 3 or not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(outer_colors | inner_colors), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[h - 1, :]
        inner = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(outer_colors | inner_colors) < 3 or not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(outer_colors | inner_colors), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, 0]
        inner = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(outer_colors | inner_colors) < 3 or not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(outer_colors | inner_colors), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, w - 1]
        inner = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(outer_colors | inner_colors) < 3 or not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(outer_colors | inner_colors), -(r1 - r0), nonbg), "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Returns (mapping border->fill, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b != bg:
            mapping[b] = f

    return mapping, pal_mask


def _sprite_components(g: np.ndarray, bg: int, pal_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Connected non-bg components outside the palette, with per-cell component ids."""
    h, w = g.shape
    comp_ids = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    next_id = 0

    for color in sorted(int(x) for x in np.unique(g)):
        if color == bg:
            continue
        mask = (g == color) & (~pal_mask)
        if not mask.any():
            continue
        for pts in components_mask(mask, connectivity=4):
            pts = np.asarray(pts, dtype=int)
            rr = pts[:, 0]
            cc = pts[:, 1]
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[rr, cc] = True
            r0 = int(rr.min())
            c0 = int(cc.min())
            r1 = int(rr.max())
            c1 = int(cc.max())
            comps.append(
                {
                    "id": next_id,
                    "color": color,
                    "mask": comp_mask,
                    "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
            comp_ids[rr, cc] = next_id
            next_id += 1

    return comps, comp_ids


def _can_reach_boundary_without_crossing(start: tuple[int, int], blocked: np.ndarray) -> bool:
    """
    Return True if `start` can reach any grid-edge cell while treating `blocked`
    as the only obstacle. This is the component-based inside/outside test.
    """
    h, w = blocked.shape
    sr, sc = start
    if blocked[sr, sc]:
        return False

    seen = np.zeros((h, w), dtype=bool)
    q = deque([(int(sr), int(sc))])
    seen[sr, sc] = True

    while q:
        r, c = q.popleft()
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            return True
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return False


def _reachable_from_boundary(walkable: np.ndarray) -> np.ndarray:
    """Return the walkable cells that can be reached from the grid boundary."""
    h, w = walkable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for c in range(w):
        if walkable[0, c] and not seen[0, c]:
            seen[0, c] = True
            q.append((0, c))
        if walkable[h - 1, c] and not seen[h - 1, c]:
            seen[h - 1, c] = True
            q.append((h - 1, c))
    for r in range(1, h - 1):
        if walkable[r, 0] and not seen[r, 0]:
            seen[r, 0] = True
            q.append((r, 0))
        if walkable[r, w - 1] and not seen[r, w - 1]:
            seen[r, w - 1] = True
            q.append((r, w - 1))

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and walkable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _mapped_adjacent_counts(
    pts: np.ndarray,
    base: np.ndarray,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
    comp_ids: np.ndarray,
    comps: list[dict],
) -> tuple[dict[int, int], set[int]]:
    """Count mapped adjacent shell colors and collect adjacent component ids."""
    counts: dict[int, int] = {}
    candidate_ids: set[int] = set()
    h, w = base.shape

    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w:
                if pal_mask[rr, cc]:
                    continue
                cid = int(comp_ids[rr, cc])
                if cid < 0:
                    continue
                candidate_ids.add(cid)
                color = int(comps[cid]["color"])
                if color in mapping:
                    counts[color] = counts.get(color, 0) + 1
    return counts, candidate_ids


def _dominant_mapped_adjacent_color(counts: dict[int, int]) -> tuple[int | None, bool]:
    """Return (dominant_color, is_unique_max)."""
    if not counts:
        return None, False
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) == 1:
        return int(items[0][0]), True
    return int(items[0][0]), items[0][1] > items[1][1]


def _select_region_border(
    pts: np.ndarray,
    base: np.ndarray,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
    comps: list[dict],
    comp_ids: np.ndarray,
) -> int | None:
    """
    Choose the border color that owns this bg region.

    A mapped shell owns the region if, when that shell alone is treated as a wall,
    the region cannot reach the grid edge. This handles nested shells and shells
    with gaps. If no enclosing mapped shell is found, fall back to dominant mapped
    adjacency.
    """
    counts, candidate_ids = _mapped_adjacent_counts(pts, base, mapping, pal_mask, comp_ids, comps)
    if not counts:
        return None

    start = tuple(map(int, pts[0]))
    enclosing: list[tuple[int, int, int]] = []
    for cid in candidate_ids:
        color = int(comps[cid]["color"])
        if color not in mapping:
            continue
        if not _can_reach_boundary_without_crossing(start, comps[cid]["mask"]):
            enclosing.append((int(comps[cid]["bbox_area"]), -counts.get(color, 0), color))

    if enclosing:
        enclosing.sort()
        return int(enclosing[0][2])

    dominant, _ = _dominant_mapped_adjacent_color(counts)
    return dominant


def _owned_bg_regions(base: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> list[dict]:
    """List interior bg regions with their owning shell color (or None)."""
    comps, comp_ids = _sprite_components(base, bg, pal_mask)
    walkable = (base == bg) | pal_mask
    exterior = _reachable_from_boundary(walkable)
    bg_mask = (base == bg) & (~exterior) & (~pal_mask)
    regions: list[dict] = []

    for pts in components_mask(bg_mask, connectivity=4):
        pts_arr = np.asarray(pts, dtype=int)
        owner = _select_region_border(pts_arr, base, mapping, pal_mask, comps, comp_ids)
        regions.append({"pts": pts_arr, "owner": owner, "counts": _mapped_adjacent_counts(pts_arr, base, mapping, pal_mask, comp_ids, comps)[0]})

    return regions


def _differential_region_count(base: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> int:
    """
    Count interior bg regions where the combo owner differs from the old dominant-adjacency
    heuristic. The generator requires at least one such region to avoid degeneration.
    """
    total = 0
    for region in _owned_bg_regions(base, bg, mapping, pal_mask):
        owner = region["owner"]
        dominant, unique = _dominant_mapped_adjacent_color(region["counts"])
        if owner is not None and dominant is not None and unique and dominant != owner and len(region["counts"]) >= 2:
            total += 1
    return total


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """Fill all interior bg regions according to their owning enclosing shell."""
    base = out.copy()
    for region in _owned_bg_regions(base, bg, mapping, pal_mask):
        owner = region["owner"]
        if owner is None:
            continue
        pts = region["pts"]
        out[pts[:, 0], pts[:, 1]] = int(mapping[owner])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill interiors of nested hollow shells using an edge legend, with topology-aware ownership.

    Input structure:
    - The background dominates the frame, and a 2×N or N×2 palette sits on one grid edge.
    - The main object area contains a large hollow shell, a smaller broken shell nested inside
      it, and a tiny closed shell nested inside that smaller shell.

    Rule:
    - Read the edge palette as a border-color to fill-color mapping.
    - Use connected-component reasoning to separate exterior from interior: bg connected to the
      grid edge is exterior, while bg cut off from the edge is a candidate interior region.
    - For each candidate interior region, choose the adjacent shell that truly disconnects that
      region from the grid edge, then fill the region with that shell's mapped color.

    Key insight:
    - In nested layouts, a bg region can touch several shell colors, so the most frequent
      neighboring color is not necessarily the owner. The correct owner is the shell whose wall
      makes that region topologically internal relative to the whole grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

_GRID_H, _GRID_W = 13, 15
_OUTER_H, _OUTER_W = 11, 11
_PALETTE_N_MIN, _PALETTE_N_MAX = 3, 4
_P_MIDDLE_OPEN = 1.0
_MAX_RETRIES = 300


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the given edge side."""
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, side: str, pal: np.ndarray, rng: random.Random) -> np.ndarray:
    """Place the palette and return its mask."""
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    else:
        r0, c0 = rng.randint(0, h - ph), w - pw

    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    pal_mask = np.zeros((h, w), dtype=bool)
    pal_mask[r0 : r0 + ph, c0 : c0 + pw] = True
    return pal_mask


def _choose_colors(
    rng: random.Random, n: int
) -> tuple[list[int], list[int], dict[int, int], int, int, int]:
    """Pick palette keys/fills and assign three distinct mapped shell colors."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:n]
    fill_values = all_colors[n : 2 * n]
    mapping = dict(zip(border_keys, fill_values))
    shell_colors = rng.sample(border_keys, 3)
    outer_color, middle_color, inner_color = map(int, shell_colors)
    return border_keys, fill_values, mapping, outer_color, middle_color, inner_color


def _paint_ring(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    """Draw a 1-cell-thick rectangular ring."""
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    """Paint a boolean mask at (r0, c0)."""
    sub = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    sub[mask] = color


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """Flood-fill the True-valued component containing `seed`."""
    sr, sc = seed
    if not mask[sr, sc]:
        raise ValueError("_component_from_seed: seed must lie inside the target mask")
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    q = deque([(int(sr), int(sc))])
    out[sr, sc] = True

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                q.append((rr, cc))
    return out


def _make_middle_component(open_gap: bool) -> np.ndarray:
    """
    Build the nested middle shell.

    The 7x7 shell is deliberately broken at the top center so its cavity merges with the
    outer cavity. The `open_gap` flag is kept for clarity even though the generator uses
    the broken mode for every accepted sample.
    """
    mask = np.zeros((7, 7), dtype=bool)
    mask[0, :] = True
    mask[6, :] = True
    mask[:, 0] = True
    mask[:, 6] = True
    if open_gap:
        mask[0, 3] = False
    return mask


def _inner_ring_mask() -> np.ndarray:
    """Small closed shell with a 1-cell hole."""
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    return mask


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    side = rng.choice(["left", "right"])
    n = rng.randint(_PALETTE_N_MIN, _PALETTE_N_MAX)
    middle_open = rng.random() < _P_MIDDLE_OPEN

    border_keys, fill_values, mapping, outer_color, middle_color, inner_color = _choose_colors(rng, n)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((_GRID_H, _GRID_W), bg, dtype=int)
    pal_mask = _place_palette(grid, side, pal, rng)

    outer_r0 = 1
    outer_c0 = 3 if side == "left" else 1
    _paint_ring(grid, outer_r0, outer_c0, _OUTER_H, _OUTER_W, outer_color)

    middle_mask = _make_middle_component(middle_open)
    middle_r0 = outer_r0 + 2
    middle_c0 = outer_c0 + 2
    _paint_mask(grid, middle_r0, middle_c0, middle_mask, middle_color)

    inner_mask = _inner_ring_mask()
    inner_r0 = middle_r0 + 2
    inner_c0 = middle_c0 + 2
    _paint_mask(grid, inner_r0, inner_c0, inner_mask, inner_color)

    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    bg_mask = grid == bg
    outer_seed = (outer_r0 + 1, outer_c0 + 1)
    inner_seed = (inner_r0 + 1, inner_c0 + 1)
    outer_region = _component_from_seed(bg_mask, outer_seed)
    inner_region = _component_from_seed(bg_mask, inner_seed)

    output = grid.copy()
    output[outer_region] = int(mapping[outer_color])
    if not middle_open:
        middle_seed = (middle_r0 + 1, middle_c0 + 1)
        middle_region = _component_from_seed(bg_mask, middle_seed)
        output[middle_region] = int(mapping[middle_color])
    output[inner_region] = int(mapping[inner_color])

    if _differential_region_count(grid, bg, mapping, pal_mask) < 1:
        return None

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
