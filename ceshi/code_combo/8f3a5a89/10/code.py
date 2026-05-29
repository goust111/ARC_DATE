# Combo variant of 8f3a5a89 (index 10)
# combo_spec: D15/0+D19/1
# Name: 形状标记+标记形状编码边界厚度

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the combo variant
# a) Concreteization and behavior trace:
#    The original task had one magenta pixel at the bottom-left corner and always painted a
#    1-cell orange inner border. This combo replaces that singleton with a tiny legal corner
#    sprite and simultaneously turns its area into a number. Example trace:
#    - Old input: one magenta pixel at (H-1, 0). Solver floods from that cell and paints only
#      the first reachable boundary layer.
#    - New input: a 4-cell 2x2 magenta block or a 4-cell tall L touching the same corner.
#      Solver must first extract the whole corner component, verify it is one of the legal
#      corner templates, count its 4 pixels, then flood from all 4 cells and paint the first
#      4 geodesic layers of the reachable inner boundary.
#    - Difference: "which cells are the start?" is now answered by local shape recognition,
#      and "how thick is the orange border?" is answered by the same shape's pixel count.
# b) Complexity calibration:
#    Thickness is limited to 2/3/4, so examples stay inferable from 2-4 pairs. The legal
#    marker family is also deliberately tiny: compact corner sprites with footprints no larger
#    than 3x2. That keeps the new cue visually obvious without turning the task into template
#    memorization over many unrelated symbols.
# c) Implicit assumption audit of the original solver:
#    - Original corner detection assumed count(marker_color) == 1. That fails here; the new
#      solver must read the full same-color component anchored at bottom-left.
#    - Original flood helper accepted one start cell. The combo needs multi-source flood from
#      every marker pixel.
#    - Original orange border thickness was hard-coded to one layer. The combo must compute the
#      original seed boundary first, then expand inward by the decoded thickness.
#    - Original generator only reserved one marker cell. The combo must reserve the entire sprite.
#    - Original fill-connectivity checks become unreliable because a multi-cell corner sprite can
#      split an edge-hugging orange band into multiple components. The rule never required one
#      orange component, so the combo generator must not enforce that artifact.
# d) Placement legality:
#    Marker cells are generated first from legal in-bounds templates anchored at bottom-left.
#    Structural walls are forbidden from overlapping any marker cell. Visible decorations are
#    restricted to reachable background cells outside the orange band so they cannot hide the
#    encoded thickness; hidden decorations stay off the border so they are not misread as walls.
# e) Visual observability:
#    Humans can see the new rule in two ways at once: the corner magenta sprite is visibly not a
#    singleton, and its pixel count matches the thickness of the orange band. Generator filters
#    out cases where the deepest requested layer is absent or where no untouched interior remains,
#    so the thickness does not collapse back to a thin border or to a fully orange region.
# f) Visual cleanliness:
#    The sprite stays compact and corner-anchored, walls remain blue, fill remains orange, and
#    reachable decorations are kept away from the orange band. This prevents the corner cue,
#    structural boundary, and thickness signal from visually overlapping in a confusing way.
# g) Change inferability:
#    No new colors are introduced, so the only new abstraction is geometric: recognize the legal
#    magenta corner sprite, count its cells, and use that count as the border thickness. Because
#    some different legal shapes share the same area, the intended invariant is the pixel count,
#    not a one-to-one mapping from specific silhouette to specific behavior.
# 8) Quantitative anti-degeneracy target:
#    After implementation, the intended check is that generated samples with marker sizes 2/3/4
#    produce strictly more orange cells than a fixed 1-layer baseline on the same input, showing
#    that the numeric control variable is active rather than silently reducing to the original task.
# 9) New main() assumptions and generator guarantees:
#    - main() assumes the bottom-left cell belongs to the marker component.
#      generate_input() always anchors the marker there.
#    - main() assumes that bottom-left marker component matches one of the legal local templates.
#      generate_input() only samples from that template set.
#    - main() assumes marker color is not reused elsewhere.
#      generate_input() uses the marker color only for that corner sprite.
#    - main() assumes there is enough reachable depth to realize the decoded thickness.
#      generate_input() rejects layouts lacking the deepest painted layer or leftover interior.


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

MARKER_THICKNESS_CHOICES = (2, 3, 4)
REACH_RATIO_RANGE = (0.42, 0.87)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 600

MARKER_SHAPES_BY_THICKNESS = {
    2: (
        frozenset({(0, 0), (0, 1)}),                  # horizontal domino
        frozenset({(0, 0), (1, 0)}),                  # vertical domino
    ),
    3: (
        frozenset({(0, 0), (0, 1), (1, 0)}),          # elbow
        frozenset({(0, 0), (1, 0), (1, 1)}),          # raised elbow
    ),
    4: (
        frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),  # 2x2 block
        frozenset({(0, 0), (0, 1), (1, 0), (2, 0)}),  # tall L
    ),
}
LEGAL_MARKER_TEMPLATES = {shape for shapes in MARKER_SHAPES_BY_THICKNESS.values() for shape in shapes}


def _flood(
    starts: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...],
    passable,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    q = deque()
    vis: set[tuple[int, int]] = set()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and passable(r, c) and (r, c) not in vis:
            vis.add((r, c))
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _same_color_component(g: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = g.shape
    sr, sc = start
    col = int(g[sr, sc])
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == col:
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _marker_local_shape(cells: set[tuple[int, int]], anchor: tuple[int, int]) -> frozenset[tuple[int, int]]:
    ar, ac = anchor
    return frozenset((ar - r, c - ac) for r, c in cells)


def _extract_marker(g: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], int, int]:
    h, _w = g.shape
    anchor = (h - 1, 0)
    marker_col = int(g[anchor])
    assert marker_col != bg, "bottom-left marker sprite missing"
    marker_cells = _same_color_component(g, anchor)
    local_shape = _marker_local_shape(marker_cells, anchor)
    assert local_shape in LEGAL_MARKER_TEMPLATES, "bottom-left corner component is not a legal marker sprite"
    thickness = len(marker_cells)
    assert thickness in MARKER_THICKNESS_CHOICES, "marker sprite encodes unsupported thickness"
    return marker_cells, marker_col, thickness


def _distance_from_seed(candidate_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    h, w = candidate_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in np.argwhere(candidate_mask & seed_mask):
        rr, cc = int(r), int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and candidate_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _compute_seed_mask(
    out: np.ndarray,
    reach: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = out.shape
    candidate = np.zeros((h, w), dtype=bool)
    seed = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        candidate[r, c] = True
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seed[r, c] = True
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            seed[r, c] = True
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            seed[r, c] = True
    return candidate, seed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a shape-decoded inner border around the region reachable from a corner marker sprite.

    Input structure:
    - The dominant background is azure, blue components act as wall-like barriers, and magenta is
      reserved for one compact corner sprite anchored at the bottom-left.
    - That magenta start marker is a small legal polyomino rather than a singleton pixel.
    - The sprite's area is meaningful: its number of pixels encodes the target orange border thickness.

    Rule:
    - Detect the full bottom-left marker component, verify it is one of the legal corner templates,
      and use all of its cells as the flood-fill source.
    - Identify which blue components are truly structural by checking border contact and adjacency to
      the first reachable side; erase blue decorations hidden behind those structural walls.
    - Build the original 1-layer reachable inner boundary, then extend it inward by as many 4-neighbor
      layers as the marker sprite's pixel count specifies.

    Key insight:
    - The same corner shape provides both pieces of information the solver needs: geometry tells you
      which cells are the starting region, and cell count tells you how thick the orange boundary must be.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    marker_cells, marker_col, thickness = _extract_marker(g, bg)
    marker_list = sorted(marker_cells)

    counts = Counter(int(v) for v in g.ravel())
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        marker_list,
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    reach = _flood(
        marker_list,
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    candidate, seed = _compute_seed_mask(out, reach, walls, bg)
    dist = _distance_from_seed(candidate, seed)
    fill_mask = candidate & (dist >= 0) & (dist < thickness)
    out[fill_mask] = FILL
    return out


def _shape_vertical_line(h, w, rng):
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h, w, rng):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h, w, rng):
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(2, max(2, w // RECT_MAX_DIM_DIV))
    anchor = rng.choice(["top", "right", "tr", "top_mid", "right_mid", "bottom_right"])
    if anchor == "top":
        r0, c0 = 0, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "right":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), w - rw
    elif anchor == "tr":
        r0, c0 = 0, w - rw
    elif anchor == "top_mid":
        r0, c0 = 0, rng.randint(2, max(2, w - rw - 2))
    elif anchor == "right_mid":
        r0, c0 = rng.randint(2, max(2, h - rh - 2)), w - rw
    else:
        r0, c0 = h - rh, w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]


def _shape_left_diamond(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 3:
        return None
    size = rng.randint(3, max_size)
    r_center = rng.randint(size, h - size - 1)
    top = r_center - (size - 1)
    cells = []
    for i in range(size):
        cells.extend((top + i, c) for c in range(i + 1))
    for i in range(size - 1):
        cells.extend((top + size + i, c) for c in range(size - 1 - i))
    return cells


def _shape_top_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // 2, w // 2)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    r_start = rng.randint(0, max(0, h - size - 1))
    cells = []
    for i in range(size):
        length = i + 1
        cells.extend((r_start + i, c) for c in range(w - length, w))
    return cells


SHAPE_FACTORIES = [
    _shape_vertical_line,
    _shape_horizontal_line,
    _shape_rectangle,
    _shape_left_diamond,
    _shape_top_triangle,
    _shape_right_triangle,
]


def _flood_mask(starts: list[tuple[int, int]], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    q = deque()
    for r, c in starts:
        if blocked[r, c] or reach[r, c]:
            continue
        reach[r, c] = True
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _touches_border(mask: np.ndarray) -> bool:
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _choose_marker_shape(rng) -> tuple[int, frozenset[tuple[int, int]]]:
    thickness = rng.choice(MARKER_THICKNESS_CHOICES)
    return thickness, rng.choice(MARKER_SHAPES_BY_THICKNESS[thickness])


def _marker_cells_from_shape(h: int, local_shape: frozenset[tuple[int, int]]) -> list[tuple[int, int]]:
    ar, ac = h - 1, 0
    return [(ar - dr, ac + dc) for dr, dc in sorted(local_shape)]


def _place_structural_walls(h: int, w: int, forbidden: np.ndarray, rng) -> np.ndarray | None:
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if any(forbidden[r, c] or structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    return structural if placed > 0 else None


def _validate_structural(structural: np.ndarray, reach: np.ndarray, h: int, w: int) -> bool:
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _compute_fill_mask(
    structural: np.ndarray,
    reach: np.ndarray,
    marker_mask: np.ndarray,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_mask] = MARKER

    walls = {(int(r), int(c)) for r, c in np.argwhere(structural)}
    reach_set = {(int(r), int(c)) for r, c in np.argwhere(reach)}
    candidate, seed = _compute_seed_mask(out, reach_set, walls, BG)
    dist = _distance_from_seed(candidate, seed)
    fill = candidate & (dist >= 0) & (dist < thickness)
    return seed, fill, dist


def _paint_output(
    structural: np.ndarray,
    reach: np.ndarray,
    marker_mask: np.ndarray,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_mask] = MARKER
    seed, fill, dist = _compute_fill_mask(structural, reach, marker_mask, thickness)
    out[fill] = FILL
    return out, fill, seed, dist


def _place_decor(allowed_mask: np.ndarray, count: int, rng) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    thickness, marker_shape = _choose_marker_shape(rng)
    marker_cells = _marker_cells_from_shape(h, marker_shape)
    marker_mask = np.zeros((h, w), dtype=bool)
    for r, c in marker_cells:
        marker_mask[r, c] = True

    structural = _place_structural_walls(h, w, marker_mask, rng)
    if structural is None or (structural & marker_mask).any():
        return None

    reach = _flood_mask(marker_cells, structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    out, fill_mask, _seed_mask, dist = _paint_output(structural, reach, marker_mask, thickness)
    if not fill_mask.any():
        return None
    if not np.any(dist == thickness - 1):
        return None
    if dist.max() < thickness:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & ~(structural | marker_mask | fill_mask)
    hidden_allowed = (~reach) & (~structural) & (~marker_mask)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    if not np.array_equal(main(inp), out_final):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant")
