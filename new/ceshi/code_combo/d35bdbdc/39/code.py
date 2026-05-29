# Combo variant of d35bdbdc (index 39)
# combo_spec: D6/0+D8/0+D9/0
# Name: 保留未连接精灵+删除路径+紧凑裁剪

import random
from collections import deque
import itertools
from pathlib import Path
import sys

import numpy as np


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "solvers" / "arc_utils.py").exists():
            return parent
    raise RuntimeError("could not locate project root containing solvers/arc_utils.py")


ROOT = _project_root()
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, neighbors4, neighbors8, paste


# Design analysis for the combo variant
# - a) Concrete behavior + trace:
#   The input still has one grey one-stroke trail and 4-5 bordered sprites. Only the
#   two sprites touched by the trail endpoints are destinations, and each destination
#   still uses its original center color as a lookup key into another sprite's border
#   color. The fusion changes the post-processing in three places at once:
#   original task = transfer on the 2 destinations -> delete every other sprite ->
#   keep the grey trail -> return full 10x10 canvas.
#   combo task = transfer on the 2 destinations -> keep every non-destination sprite
#   unchanged -> erase the grey trail to background -> crop to the smallest rectangle
#   containing all remaining non-background pixels. Example trace: if a destination
#   centered blue points to a green-bordered source whose center is yellow, only that
#   destination center becomes yellow; all bystanders stay visible; the grey path then
#   disappears; the output is the tight crop of the sprite-only arrangement.
# - b) Complexity calibration:
#   Keeping all sprites makes the color-transfer rule harder to notice unless the path
#   deletion and crop are both visible. The generator therefore keeps 4 sprites in
#   every example, so there are always exactly two preserved bystanders and enough
#   repeated evidence that "only endpoint-connected sprites
#   change". To keep the cropped all-sprites layout physically feasible inside a 10x10
#   grid, the generator fixes the sprite shape to the sparser plus/cross form; border
#   colors remain unique, preserving one-to-one source lookup and preventing ambiguous
#   many-to-many color matches.
# - c) Hidden assumptions reviewed:
#   The original solver assumed non-destination material must disappear, so both the
#   sprite-deletion loop and the cleanup flood-fill are invalid here. D8 also breaks
#   the assumption that the grey path is part of the final support set, so cropping
#   must happen after the path is erased. The destination-selection logic still relies
#   on the input trail being present, and the border->source lookup still relies on
#   reading sprite metadata from the unmodified input before any center updates.
# - d) Placement legality:
#   The generator uses interior sprite layouts whose visible cells stay away from the
#   outer border, ensuring a real crop is possible. Destinations occupy opposite
#   corners of that compact layout so their valid ports point inward, while the grey
#   trail is generated inside an inner frame and rejected if it creates any valid grey
#   port for a non-destination sprite. This keeps objects disjoint, on-grid, and
#   semantically legal for the solver.
# - e) Visual observability:
#   A human can infer the fused rule from a few examples because three visible facts
#   repeat together: preserved bystanders remain in the output, no grey cells survive,
#   and the output bounds hug the sprite arrangement instead of the full 10x10 input.
#   Nothing depends on sorting by numeric color ids; the only structural cue is endpoint
#   connectivity along the visible trail in the input.
# - f) Visual cleanliness:
#   The final output contains only bordered sprites with no path overlay. The generator
#   rejects layouts where bystanders accidentally gain grey ports, so there is no
#   misleading half-connected clutter. Compact interior layouts also avoid giant empty
#   areas after cropping.
# - g) Color-rule inferability:
#   No new color mechanics are introduced beyond the original border-to-center lookup.
#   Because unchanged bystanders remain visible, the solver can compare changed and
#   unchanged sprites in the same scene and infer that connectivity selects which
#   centers update, while border colors explain what new center each destination gets.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the 2 endpoints of the grey one-stroke trail under 8-connectivity."""
    h, w = g.shape
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        border = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == border:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    n = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == n:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
    return path[0], path[-1]


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color == bg or border_color == grey or center_color == grey:
            continue

        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return endpoint cells that satisfy the 2-step port rule for this sprite."""
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(g != bg)
    assert coords.size > 0, "expected some retained non-background cells"
    r0, c0, r1, c1 = bbox_coords(coords)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer endpoint-selected center colors, erase the temporary trail, and crop to the sprite-only region.

    Input structure:
    - The input contains background, a grey (5) one-stroke trail, and several two-color bordered sprites.
    - Each sprite has one repeated border color and a single center pixel of another color.
    - Only the two sprites touched by the trail endpoints through valid 2-step ports are destinations; every
      other sprite is still present in the input but is not trail-connected.

    Rule:
    - Use the trail endpoints to identify the two destination sprites.
    - For each destination, treat its original center color as a lookup key for the unique source sprite whose
      border color matches that key, and copy that source center color into the destination center.
    - Keep every non-destination sprite unchanged, remove the entire grey trail to background, then return the
      tight crop of all remaining non-background pixels.

    Key insight:
    - The grey path is only a temporary selector: it tells you which sprites are allowed to change, but it does
      not survive in the output. After the transfer, the final picture is just the compact sprite arrangement,
      where connected sprites may have new centers and unconnected sprites stay exactly as they were.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[te1]) <= len(opts[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()
    out[out == grey] = bg

    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return _tight_crop_non_bg(out, bg)


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SOURCE_COUNTS = [1, 2]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 16
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
PATH_MARGIN = 1
GEN_RETRIES = 500


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus sprite centered at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap and 4-touch against non-background, non-grey cells."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _in_path_frame(r: int, c: int) -> bool:
    return PATH_MARGIN <= r < GRID_H - PATH_MARGIN and PATH_MARGIN <= c < GRID_W - PATH_MARGIN


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check if adding (r,c) would make any 8-neighborhood too dense."""
    nbs = [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (r + dr, c + dc) in occupied
    ]
    if len(nbs) > MAX_THICKNESS:
        return True
    for nr, nc in nbs:
        deg = sum(
            1
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (nr + dr, nc + dc) in occupied
        )
        if deg + 1 > MAX_THICKNESS:
            return True
    return False


def _gen_winding_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int
) -> list[tuple[int, int]] | None:
    """Generate an 8-connected grey trail inside the inner frame."""
    if not (_in_path_frame(*start) and _in_path_frame(*end)):
        return None

    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int, bool]) -> int:
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

    for _ in range(target_len + 40):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if not _in_path_frame(nr, nc):
                    continue
                if (nr, nc) in visited or grid[nr, nc] != BG or _too_thick(nr, nc, visited):
                    continue
                moves.append((nr, nc, dr != 0 and dc != 0))

        if not moves:
            return None

        dist = abs(cr - end[0]) + abs(cc - end[1])
        budget = target_len - len(path)
        diag = [move for move in moves if move[2]]
        away = [move for move in moves if score(move) > dist]

        if budget > dist + 4:
            if away and random.random() < PATH_WANDER_AWAY_P:
                pool = away
            elif diag and random.random() < PATH_WANDER_DIAG_P:
                pool = diag
            else:
                pool = moves
            chosen = random.choice(pool)
        elif budget > dist + 1:
            pool = diag if diag and random.random() < PATH_WANDER_DIAG_P else moves
            if random.random() < PATH_MODERATE_RANDOM_P:
                chosen = random.choice(pool)
            else:
                chosen = min(pool, key=score)
        else:
            best = sorted(moves, key=score)
            top = [move for move in best if score(move) <= score(best[0]) + 1]
            dtop = [move for move in top if move[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if not _in_path_frame(*end) or grid[end] != BG or _too_thick(end[0], end[1], visited):
            return None
        path.append(end)

    return path


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return len(visited) == len(cell_set)


def _port_options(grid: np.ndarray, cr: int, cc: int, border_color: int) -> list[tuple[int, int]]:
    """Return all valid empty port cells for a sprite center/border pair."""
    options: list[tuple[int, int]] = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if not _in_path_frame(er, ec):
            continue
        if grid[ir, ic] == border_color and grid[er, ec] == BG:
            options.append((er, ec))
    return options


def _grey_ports_for_sprite(
    center_pos: tuple[int, int], border_color: int, g: np.ndarray
) -> list[tuple[int, int]]:
    """Return all grey cells that satisfy the 2-step port rule for this sprite."""
    cr, cc = center_pos
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(g[ir, ic]) == border_color and int(g[er, ec]) == GREY:
            ports.append((er, ec))
    return ports


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    """Assign roles (dest/source/distractor) and pick center colors for all sprites."""
    remaining = list(range(2, num_sprites))
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        dest_center_colors = [border_colors[source_indices[0]]] * 2
    else:
        dest_center_colors = [border_colors[si] for si in source_indices]

    source_center_colors = []
    for j, si in enumerate(source_indices):
        forbidden = {BG, GREY, border_colors[si]}
        if num_sources == 1:
            forbidden.update([border_colors[0], border_colors[1]])
        else:
            forbidden.add(border_colors[j])
        candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        source_center_colors.append(random.choice(candidates))

    distractor_center_colors = [
        random.choice([c for c in range(1, 10) if c not in {GREY, border_colors[di]}])
        for di in distractor_indices
    ]

    output_dest_centers = (source_center_colors * 2)[:2] if num_sources == 1 else source_center_colors
    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
    }


def _dest_position_pairs() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Opposite-corner destination pairs inside the crop-safe inner area."""
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dr in [0, 1]:
        for dc in [0, 1]:
            tl = (1 + dr, 1 + dc)
            tr = (1 + dr, 5 + dc)
            bl = (5 + dr, 1 + dc)
            br = (5 + dr, 5 + dc)
            pairs.append((tl, br))
            pairs.append((tr, bl))
    return pairs


def _layout_candidates() -> list[list[tuple[int, int]]]:
    """Return compact destination pairs inside the crop-safe inner area."""
    layouts: list[list[tuple[int, int]]] = []
    for dr in [0, 1]:
        for dc in [0, 1]:
            tl = (1 + dr, 1 + dc)
            tr = (1 + dr, 5 + dc)
            bl = (5 + dr, 1 + dc)
            br = (5 + dr, 5 + dc)
            layouts.append([tl, br])
            layouts.append([tr, bl])
    return layouts


def _sprite_positions_inner() -> list[tuple[int, int]]:
    """All fully interior 3x3 placements, leaving a 1-cell border for the crop to remove."""
    return [
        (r0, c0)
        for r0 in range(1, GRID_H - 3)
        for c0 in range(1, GRID_W - 3)
    ]


def _tight_bbox_inside(full_output: np.ndarray) -> bool:
    coords = np.argwhere(full_output != BG)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0) > 0 and int(c0) > 0 and int(r1) < GRID_H - 1 and int(c1) < GRID_W - 1


def _endpoint_clean(path_cells: list[tuple[int, int]], endpoint: tuple[int, int]) -> bool:
    path_set = set(path_cells)
    return sum((nr, nc) in path_set for nr, nc in neighbors8(*endpoint)) == 1


def _compatible_slot_assignment(
    shape_type: str,
    border_colors: list[int],
    bystander_indices: list[int],
    dest_grid: np.ndarray,
    path_cells: list[tuple[int, int]],
) -> list[tuple[int, tuple[int, int]]] | None:
    """Find bystander placements that avoid path overlap and grey-port confusion."""
    base = dest_grid.copy()
    for r, c in path_cells:
        base[r, c] = GREY

    candidates = _sprite_positions_inner()
    random.shuffle(candidates)
    for order in itertools.permutations(bystander_indices):
        test_grid = base.copy()
        assignment: list[tuple[int, tuple[int, int]]] = []
        ok = True
        for sprite_idx in order:
            patch = _make_sprite_patch(shape_type, border_colors[sprite_idx], border_colors[sprite_idx])
            placed = False
            for r0, c0 in candidates:
                if _overlaps(test_grid, patch, r0, c0):
                    continue
                _place_sprite(test_grid, patch, r0, c0)
                if _grey_ports_for_sprite((r0 + 1, c0 + 1), border_colors[sprite_idx], test_grid):
                    _erase_sprite(test_grid, patch, r0, c0)
                    continue
                assignment.append((sprite_idx, (r0, c0)))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return assignment
    return None


def _gen_attempt() -> np.ndarray | None:
    shape_type = "plus"
    num_sprites = 4
    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    dest_positions = random.choice(_layout_candidates())
    others = (
        [(int(si), int(colors["source_center_colors"][j])) for j, si in enumerate(colors["source_indices"])]
        + [(int(di), int(colors["distractor_center_colors"][j])) for j, di in enumerate(colors["distractor_indices"])]
    )
    bystander_indices = [sprite_idx for sprite_idx, _ in others]
    dest_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    sprite_centers: list[tuple[int, int] | None] = [None] * num_sprites
    for i, (r0, c0) in enumerate(dest_positions):
        patch = _make_sprite_patch(shape_type, border_colors[i], int(colors["output_dest_centers"][i]))
        if _overlaps(dest_grid, patch, r0, c0):
            return None
        _place_sprite(dest_grid, patch, r0, c0)
        sprite_centers[i] = (r0 + 1, c0 + 1)

    assert sprite_centers[0] is not None and sprite_centers[1] is not None
    port_opts0 = _port_options(dest_grid, sprite_centers[0][0], sprite_centers[0][1], border_colors[0])
    port_opts1 = _port_options(dest_grid, sprite_centers[1][0], sprite_centers[1][1], border_colors[1])
    if not port_opts0 or not port_opts1:
        return None

    path_cells = None
    chosen_ports = None
    slot_assignment = None
    target_lengths = list(range(PATH_LEN_MIN, PATH_LEN_MAX + 1))
    random.shuffle(target_lengths)
    for target_len in target_lengths:
        min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)
        port_pairs = [
            (abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]), p0, p1)
            for p0 in port_opts0
            for p1 in port_opts1
            if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) >= min_port_dist
        ]
        random.shuffle(port_pairs)
        port_pairs.sort(key=lambda item: item[0], reverse=True)
        for _, p0, p1 in port_pairs:
            trial = _gen_winding_path(dest_grid, p0, p1, target_len)
            if trial is None or not (PATH_LEN_MIN <= len(trial) <= PATH_LEN_MAX):
                continue
            if not _is_8_connected(trial):
                continue
            if not _endpoint_clean(trial, p0) or not _endpoint_clean(trial, p1):
                continue
            assign = _compatible_slot_assignment(shape_type, border_colors, bystander_indices, dest_grid, trial)
            if assign is None:
                continue
            path_cells = trial
            chosen_ports = (p0, p1)
            slot_assignment = assign
            break
        if path_cells is not None:
            break
    if path_cells is None or chosen_ports is None or slot_assignment is None:
        return None

    full_output = dest_grid.copy()
    input_grid = dest_grid.copy()
    for i in range(2):
        cr, cc = sprite_centers[i]
        assert cr is not None and cc is not None
        input_grid[cr, cc] = int(colors["dest_center_colors"][i])

    for r, c in path_cells:
        if input_grid[r, c] != BG:
            return None
        input_grid[r, c] = GREY

    center_by_index = {sprite_idx: center_color for sprite_idx, center_color in others}
    for sprite_idx, (r0, c0) in slot_assignment:
        center_color = center_by_index[sprite_idx]
        patch = _make_sprite_patch(shape_type, border_colors[sprite_idx], center_color)
        if _overlaps(full_output, patch, r0, c0) or _overlaps(input_grid, patch, r0, c0):
            return None
        _place_sprite(full_output, patch, r0, c0)
        _place_sprite(input_grid, patch, r0, c0)
        center = (r0 + 1, c0 + 1)
        if _grey_ports_for_sprite(center, border_colors[sprite_idx], input_grid):
            return None
        sprite_centers[sprite_idx] = center

    if not _tight_bbox_inside(full_output):
        return None

    for i in range(num_sprites):
        assert sprite_centers[i] is not None
        ports = _grey_ports_for_sprite(sprite_centers[i], border_colors[i], input_grid)
        if i == 0 and chosen_ports[0] not in ports:
            return None
        if i == 1 and chosen_ports[1] not in ports:
            return None
        if i >= 2 and ports:
            return None

    bg = _bg_boundary(input_grid)
    sprites = _detect_bordered_sprites(input_grid, bg, GREY)
    if len(sprites) != num_sprites:
        return None
    if set(_find_trail_endpoints(input_grid, GREY, sprites)) != set(chosen_ports):
        return None

    expected_output = _tight_crop_non_bg(full_output, BG)
    try:
        out = main(input_grid)
    except Exception:
        return None
    if np.array_equal(out, expected_output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
