# Variant of e3721c99
# Dimension: D11 — 多步处理逻辑 (multi_step)
# Change: 单步处理——识别图例、计算洞数、直接着色，无迭代或链式操作... → 链式传播（已着色精灵影响相邻未着色精灵）

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the D11 variant
# a) Concrete behavior and trace:
#    - Keep the original legend panel: each colored legend shape still defines hole_count -> color.
#    - Grey sprites in the workspace are still separate 4-connected components, but some are arranged so that
#      two different components touch only diagonally. That creates a visible "chain" without merging them into
#      one component.
#    - Example: suppose the legend says 2 holes -> red. Three grey components A,B,C are arranged as a diagonal
#      chain. A has 2 holes, B and C have non-legend hole counts. Original task: A becomes red, B/C are erased.
#      Variant: round 1 colors A from the legend; round 2 B inherits red because it diagonally touches A;
#      round 3 C inherits red because it diagonally touches the now-colored B.
# b) Complexity calibration:
#    - Propagation uses only one visible adjacency mode: 8-neighbor contact between different 4-components,
#      which in practice means diagonal touching. This is strong enough to be observable but still easy to infer.
#    - Follower sprites intentionally use hole counts outside the legend set, so their recoloring cannot be
#      explained by the original single-step rule.
#    - The generator guarantees multiple propagation edges per sample so the new logic is repeated, not anecdotal.
# c) Hidden assumptions in the original main():
#    - Original loop assumed every grey component can be solved independently from its own hole count.
#      That is no longer safe because a non-legend component may still survive via propagation.
#    - Original "else -> bg" assignment was harmless only because colors never depended on other sprites.
#      In the variant, erasing too early would destroy components that should be recolored in later rounds.
#    - Therefore the new solver first identifies seed components, then runs iterative propagation, and only
#      after propagation stabilizes does it erase still-uncolored components.
# d) Placement legality:
#    - Legend/divider placement remains unchanged.
#    - Each propagation cluster is first assembled off-grid as a chain of grey components with diagonal contact
#      and no orthogonal contact; then the whole cluster is placed into the workspace with at least one-cell
#      clearance from other clusters and from the blocked legend/divider region.
# e) Visual observability:
#    - The new rule is visible because some recolored sprites have the "wrong" hole count for every legend entry
#      yet still turn into the same color as a touching neighbor.
#    - Diagonal chains appear in the input and become monochrome regions in the output, making the multi-step
#      nature inferable from 2-3 examples.
# f) Visual cleanliness:
#    - Different clusters never 8-touch each other, so the only touching relations the solver sees are the
#      intended within-cluster ones.
#    - We avoid branching and ambiguous cross-color contacts by building simple chains.
# g) Color-change inferability:
#    - Colors still come from the legend only; propagation does not introduce new color mappings.
#    - Follower components inherit a nearby colored component's color, so the color change has a clear spatial cue.
# h) Quantitative anti-degeneration target:
#    - Every generated grid contains at least two follower components that do not match any legend hole count
#      but are recolored via propagation.
# i) New main() assumptions checked against the generator:
#    - Assumes at least one seed component exists: generator places at least one direct legend match per legend.
#    - Assumes propagation ambiguity is absent or rare: generator isolates clusters and gives each follower chain
#      a unique already-colored neighbor frontier.
#    - Assumes adjacency is based on separate components, not merged blobs: generator uses diagonal-only contact.


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding box border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg,grey) color.
    Map hole_count -> legend_color.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == c
        hc = _hole_count(sub)
        assert hc not in mapping, f"legend hole count {hc} must map to exactly one color"
        mapping[hc] = c
    return mapping


def _component_infos(g: np.ndarray, grey: int) -> list[dict]:
    """Collect grey components with their coords, coord-set, and hole count."""
    infos: list[dict] = []
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        coords = []
        coord_set = set()
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            sub[rr - r0, cc - c0] = True
            coords.append((rr, cc))
            coord_set.add((rr, cc))
        infos.append({
            "pts": coords,
            "set": coord_set,
            "hole_count": _hole_count(sub),
        })
    return infos


def _touches8(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    """Whether two separate 4-components are spatially adjacent by 8-neighborhood."""
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in b:
                    return True
    return False


def _adjacency_graph(infos: list[dict]) -> list[set[int]]:
    """Build an undirected graph over grey components using 8-neighbor adjacency."""
    n = len(infos)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _touches8(infos[i]["set"], infos[j]["set"]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by legend matches, then propagate those colors through touching chains.

    Input structure:
    - bg is the most common color; grey (5) sprites occupy the workspace while the colored legend panel stays near
      an edge or corner behind a divider.
    - Some grey sprites directly match a legend hole count; other grey sprites may have non-legend hole counts but
      sit diagonally adjacent to those matched sprites, forming chains of separate components.

    Rule:
    - Read the legend exactly as in the original task: each colored legend shape maps its hole count to a color.
    - First recolor every grey component whose own hole count appears in the legend.
    - Then repeatedly recolor any still-grey component that touches already recolored components by 8-neighborhood;
      if all recolored neighbors agree on one color, the untouched component inherits that color. Components that
      never receive a propagated color are erased to background.

    Key insight: Hole counts only create the initial seeds; the full output emerges by iterative spatial propagation
    across diagonal chains of otherwise separate grey components.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    infos = _component_infos(g, grey)
    adj = _adjacency_graph(infos)

    colors: list[int | None] = []
    for info in infos:
        hc = int(info["hole_count"])
        colors.append(int(legend[hc]) if hc in legend else None)

    while True:
        snapshot = list(colors)
        updates: list[tuple[int, int]] = []
        for i, current in enumerate(snapshot):
            if current is not None:
                continue
            neighbor_colors = {snapshot[j] for j in adj[i] if snapshot[j] is not None}
            if len(neighbor_colors) == 1:
                updates.append((i, int(next(iter(neighbor_colors)))))
        if not updates:
            break
        for i, color in updates:
            colors[i] = int(color)

    out = g.copy()
    for info, color in zip(infos, colors):
        newc = bg if color is None else int(color)
        for r, c in info["pts"]:
            out[r, c] = newc

    return out


# --------------- generator (holes-first organic blob + propagation chains) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
SPRITE_RETRIES = 30
SPRITE_AREA_0 = (5, 25)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 12
SPRITE_MAX_W = 18
SPRITE_MAX_ASPECT = 2.5

# -- propagation cluster generation --
MIN_PROPAGATING_CLUSTERS = 2
MAX_PROPAGATING_CLUSTERS = 3
MAX_CHAIN_LEN = 4
MAX_ISOLATED_DISTRACTORS = 3
CLUSTER_CANVAS = 80
CLUSTER_RETRIES = 50

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)
_DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _legend_span(hole_count: int) -> int:
    """Width (or height) of a single legend patch along the arrangement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan (horizontal) or Spanx3 (vertical) legend patch."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if hole_count:
            patch[1, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if hole_count:
            patch[1::2, 1] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 2-4 legends with distinct hole counts that fit within max_span."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row horizontal divider with legend panel above or below."""
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - 4)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-column vertical divider with legend panel left or right."""
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - 4)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend placement bounds for an L-corner."""
    n = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place an L-shaped divider with legend panel in one corner."""
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, arrangement, total
    )

    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Randomly choose and apply a divider layout."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    """Sample a random sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Find well-separated seed positions for holes inside the canvas."""
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
    positions = [
        (r, c)
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
        if margin <= r < csz - margin and margin <= c < csz - margin
    ]
    random.shuffle(positions)
    seeds = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(
    seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int
) -> set[tuple[int, int]]:
    """Grow irregular blob regions around each hole seed."""
    margin = HOLE_SEED_MARGIN
    max_hole = (
        max(2, area // (hole_count + 2))
        if hole_count <= 2
        else max(1, area // (hole_count * 2 + 1))
    )
    forbidden = set()
    for sr, sc in seeds:
        forbidden.add((sr, sc))
        blob = {(sr, sc)}
        target = random.randint(1, max(1, max_hole))
        while len(blob) < target:
            cands = []
            for r, c in blob:
                for nr, nc in neighbors4(r, c):
                    if not (margin <= nr < csz - margin and margin <= nc < csz - margin):
                        continue
                    if (nr, nc) in forbidden:
                        continue
                    if any(
                        (nnr, nnc) in forbidden and (nnr, nnc) not in blob
                        for nnr, nnc in neighbors4(nr, nc)
                    ):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            p = random.choice(cands)
            forbidden.add(p)
            blob.add(p)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    """Find the nearest non-forbidden cell to the canvas center."""
    if (center, center) not in forbidden:
        return center, center
    for d in range(1, csz):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = center + dr, center + dc
                if 0 <= nr < csz and 0 <= nc < csz and (nr, nc) not in forbidden:
                    return nr, nc
    return center, center


def _grow_mask(
    csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]
) -> np.ndarray:
    """Grow a random connected blob of target area, avoiding forbidden cells."""
    mask = np.zeros((csz, csz), dtype=bool)
    sr, sc = start
    mask[sr, sc] = True
    frontier = list(neighbors4(sr, sc))
    filled = 1
    while filled < area and frontier:
        idx = random.randrange(len(frontier))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        r, c = frontier.pop()
        if not (0 <= r < csz and 0 <= c < csz) or mask[r, c] or (r, c) in forbidden:
            continue
        mask[r, c] = True
        filled += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc]:
                frontier.append((nr, nc))
    return mask


def _seal_and_connect(mask: np.ndarray, forbidden: set[tuple[int, int]]) -> np.ndarray:
    """Seal gaps around forbidden cells and keep only the largest connected component."""
    csz = mask.shape[0]
    for _ in range(SEAL_ITERS):
        sealed = False
        for r, c in sorted(forbidden):
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc] and (nr, nc) not in forbidden:
                    mask[nr, nc] = True
                    sealed = True
        if not sealed:
            break
    if np.any(mask):
        labeled, n_comp = label_mask(mask)
        if n_comp > 1:
            best = max(range(1, n_comp + 1), key=lambda lbl: int(np.sum(labeled == lbl)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop mask to tight bbox (including forbidden cells), validate shape and holes."""
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    for r, c in forbidden:
        rmin, cmin = min(rmin, r), min(cmin, c)
        rmax, cmax = max(rmax, r), max(cmax, c)
    h, w = rmax - rmin + 1, cmax - cmin + 1
    if h > SPRITE_MAX_H or w > SPRITE_MAX_W or max(h, w) > SPRITE_MAX_ASPECT * min(h, w):
        return None
    cropped = mask[rmin : rmax + 1, cmin : cmax + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    """Generate a random amorphous grey sprite with the specified number of holes."""
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2

        forbidden = set()
        if hole_count > 0:
            seeds = _place_hole_seeds(hole_count, center, csz, area)
            if seeds is None:
                continue
            forbidden = _grow_hole_blobs(seeds, hole_count, area, csz)

        start = _find_start(center, csz, forbidden)
        mask = _grow_mask(csz, area, start, forbidden)
        mask = _seal_and_connect(mask, forbidden)

        result = _crop_and_validate(mask, forbidden, hole_count)
        if result is not None:
            return result
    return None


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _sprite_coords(sprite: np.ndarray) -> list[tuple[int, int]]:
    """Coordinates of grey cells inside a sprite patch."""
    return [tuple(map(int, pt)) for pt in np.argwhere(sprite == GREY)]


def _component_patch_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    """Convert absolute coords into a cropped GREY/BG patch."""
    pts = np.array(sorted(coords), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = GREY
    return patch


def _try_attach_sprite(
    parent_coords: set[tuple[int, int]],
    occupied_coords: set[tuple[int, int]],
    sprite: np.ndarray,
) -> set[tuple[int, int]] | None:
    """Attach a sprite to the parent by diagonal contact only, without touching others."""
    sprite_pts = _sprite_coords(sprite)
    offsets = set()
    for pr, pc in parent_coords:
        for sr, sc in sprite_pts:
            for dr, dc in _DIAGS:
                offsets.add((pr + dr - sr, pc + dc - sc))
    offsets = list(offsets)
    random.shuffle(offsets)

    other_coords = occupied_coords - parent_coords
    for off_r, off_c in offsets:
        placed = {(off_r + sr, off_c + sc) for sr, sc in sprite_pts}
        if any(r < 0 or c < 0 or r >= CLUSTER_CANVAS or c >= CLUSTER_CANVAS for r, c in placed):
            continue
        if placed & occupied_coords:
            continue
        invalid = False
        touches_parent = False
        for r, c in placed:
            for rr, cc in neighbors4(r, c):
                if (rr, cc) in occupied_coords:
                    invalid = True
                    break
            if invalid:
                break
            for dr, dc in _DIAGS:
                nb = (r + dr, c + dc)
                if nb in parent_coords:
                    touches_parent = True
                elif nb in other_coords:
                    invalid = True
                    break
            if invalid:
                break
        if not invalid and touches_parent:
            return placed
    return None


def _build_cluster_patch(sprites: list[np.ndarray]) -> np.ndarray | None:
    """Build a diagonal-touch chain patch from a list of sprites."""
    if not sprites:
        return None
    canvas = np.zeros((CLUSTER_CANVAS, CLUSTER_CANVAS), dtype=bool)
    placed_components: list[set[tuple[int, int]]] = []

    first = sprites[0]
    h, w = first.shape
    r0 = CLUSTER_CANVAS // 2 - h // 2
    c0 = CLUSTER_CANVAS // 2 - w // 2
    first_coords = {(r0 + r, c0 + c) for r, c in _sprite_coords(first)}
    if any(r < 0 or c < 0 or r >= CLUSTER_CANVAS or c >= CLUSTER_CANVAS for r, c in first_coords):
        return None
    for r, c in first_coords:
        canvas[r, c] = True
    placed_components.append(first_coords)
    occupied = set(first_coords)

    for sprite in sprites[1:]:
        new_coords = _try_attach_sprite(placed_components[-1], occupied, sprite)
        if new_coords is None:
            return None
        for r, c in new_coords:
            canvas[r, c] = True
        placed_components.append(new_coords)
        occupied |= new_coords

    patch = _component_patch_from_coords(occupied)
    assert len(components_mask(patch == GREY, connectivity=4)) == len(sprites)
    return patch


def _random_patch_location(
    static_blocked: np.ndarray, occupied: np.ndarray, patch: np.ndarray
) -> tuple[int, int] | None:
    """Find a random placement for a whole patch, isolating different patches by one cell."""
    h, w = patch.shape
    patch_mask = patch == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & patch_mask):
            continue
        return int(r), int(c)
    return None


def _plan_items(legends: list[tuple[int, int]]) -> list[dict]:
    """Plan cluster/singleton items while guaranteeing visible propagation."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    legend_holes = [h for h, _ in legends]
    nonlegend_holes = [h for h in range(MAX_HOLE_COUNT) if h not in set(legend_holes)]

    extra_budget = total - len(legends)
    prop_count = random.randint(
        MIN_PROPAGATING_CLUSTERS,
        min(MAX_PROPAGATING_CLUSTERS, len(legends), extra_budget),
    )
    prop_indices = set(random.sample(range(len(legends)), prop_count))

    items = []
    used = 0
    for i, (seed_hole, color) in enumerate(legends):
        followers = []
        if i in prop_indices:
            followers.append(random.choice(nonlegend_holes))
        items.append({"seed_hole": seed_hole, "color": color, "followers": followers})
        used += 1 + len(followers)

    remaining = total - used
    if remaining > 0:
        n_isolated = random.randint(0, min(MAX_ISOLATED_DISTRACTORS, remaining))
    else:
        n_isolated = 0

    extra_followers = remaining - n_isolated
    if extra_followers > 0:
        heavy_candidates = [idx for idx in range(len(items)) if len(items[idx]["followers"]) > 0]
        if heavy_candidates and all(len(items[idx]["followers"]) < 2 for idx in heavy_candidates):
            idx = random.choice(heavy_candidates)
            items[idx]["followers"].append(random.choice(nonlegend_holes))
            extra_followers -= 1
        for _ in range(extra_followers):
            candidates = [idx for idx in range(len(items)) if len(items[idx]["followers"]) < (MAX_CHAIN_LEN - 1)]
            idx = random.choice(candidates)
            items[idx]["followers"].append(random.choice(nonlegend_holes))

    for _ in range(n_isolated):
        items.append({"seed_hole": random.choice(nonlegend_holes), "color": None, "followers": []})

    random.shuffle(items)
    return items


def _make_item_patch(spec: dict) -> np.ndarray | None:
    """Generate one patch for either a singleton or a propagation chain."""
    for _ in range(CLUSTER_RETRIES):
        sprites = []
        seed = _make_sprite(int(spec["seed_hole"]))
        if seed is None:
            continue
        sprites.append(seed)
        ok = True
        for hole_count in spec["followers"]:
            spr = _make_sprite(int(hole_count))
            if spr is None:
                ok = False
                break
            sprites.append(spr)
        if not ok:
            continue
        if len(sprites) == 1:
            return sprites[0]
        patch = _build_cluster_patch(sprites)
        if patch is not None:
            return patch
    return None


def _place_items(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place propagation clusters, singleton matches, and isolated distractors."""
    specs = _plan_items(legends)
    built = []
    for spec in specs:
        patch = _make_item_patch(spec)
        if patch is None:
            return False
        built.append((int(np.sum(patch == GREY)), patch, spec["color"]))
    built.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, patch, color in built:
        loc = _random_patch_location(blocked, occupied, patch)
        if loc is None:
            return False
        r, c = loc
        paste(grid, patch, r, c, transparent=BG)
        paste(occupied, patch, r, c, transparent=BG)
        if color is not None:
            out_patch = np.where(patch == GREY, int(color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _count_propagated_followers(grid: np.ndarray) -> int:
    """Number of non-legend-hole components that survive only because of propagation."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _legend_mapping(g, bg, GREY)
    out = main(g)
    count = 0
    for info in _component_infos(g, GREY):
        hc = int(info["hole_count"])
        sample_r, sample_c = info["pts"][0]
        out_color = int(out[sample_r, sample_c])
        if hc not in legend and out_color != bg:
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place clusters, validate via oracle and anti-degeneration checks."""
    n = GRID_SIZE
    grid = np.zeros((n, n), dtype=int)
    expected = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_items(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if _count_propagated_followers(grid) < 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 D11 input")
