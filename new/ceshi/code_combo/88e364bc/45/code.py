# Combo variant of 88e364bc (index 45)
# combo_spec: D5/0+D10/1
# Name: 2x2色块+按颜色分组堆叠

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis for combo D5/0 + D10/1
# spec_variant:
#   core_change:
#     from:
#       - every mover is a yellow single cell
#       - all movers block each other uniformly
#     to:
#       - every mover is a solid 2x2 sprite, and movers use two visible colors
#       - movement is resolved on 1D slide tracks of 2x2 anchors
#       - within one track, same-color sprites stack together; different colors can pass through
#         and therefore become color-grouped near the destination wall
#   concrete_visual_form:
#     - legends and wall-linked directions remain exactly the control mechanism
#     - enclosed regions now contain chunky 2x2 green/yellow blocks instead of lone yellow pixels
#     - at least one region per generated grid shows one mixed-color track with an input order like
#       A, B, A along the slide direction; the output becomes A, A, B on that same track
#     - the walls still define the available interior, so the new idea is visible without adding any
#       extra annotation layer
#   behavior_trace:
#     original:
#       - read every yellow cell independently
#       - move it one cell at a time until a wall or any earlier yellow stops it
#       - equal treatment of all movers means spatial order is preserved
#     combo:
#       - first recover each 2x2 same-color component as one rigid sprite with a top-left anchor
#       - build that region's valid anchor tracks for the legend direction
#       - on each track, read the mover colors from the wall side outward
#       - preserve only the first-appearance order of colors on that track, count how many sprites
#         each color has, then repack the far-side physical 2x2 slots by color groups
#     small_example:
#       - direction = right, one track has input anchors with near-to-far colors [green, yellow, green]
#       - color-blind rigid packing would keep [green, yellow, green] against the wall
#       - the fused rule instead keeps the wall-side color order [green, yellow] but merges equal colors,
#         so the output becomes [green, green, yellow]
#       - because sprites are 2x2, the packed slots are spaced by full non-overlapping block positions,
#         not by single pixels
#   complexity_calibration:
#     - only two mover colors are used, so the grouping rule is strong and repeated rather than noisy
#     - each active region uses one clearly readable mixed-color track with three 2x2 sprites
#     - walls, legends, and region irregularity stay from the base task, preserving the original
#       "read legend color, then slide inside that enclosure" reasoning scaffold
#     - the track-local grouping rule is enough to create visible color sorting without introducing
#       hidden numeric color-order conventions
#   hidden_assumptions_audited:
#     - old `movers = all yellow cells` is invalid: one object now spans four cells and has a color
#     - old passable logic must treat both mover colors as interior cells
#     - old per-step collision on single cells is invalid because 2x2 anchors overlap after a 1-cell shift
#     - "different colors pass through" cannot be implemented as literal independent occupancy, because
#       that would let two colors demand the same final cells; the solver therefore works on explicit
#       non-overlapping track slots and groups those slots by color
#     - extraction now assumes each mover component is a filled 2x2 square of one color; the generator
#       guarantees this by painting only separated slot positions
#     - diagonal motion still needs anti-corner-cut checks for the whole 2x2 anchor, not just one cell
#   placement_legality:
#     - generator only accepts regions whose anchor geometry contains at least one track with >= 5 valid
#       non-overlapping 2x2 slots, enough for a 3-block grouped output and a reverse-slid input
#     - input sprites are placed only on those legal non-overlapping slots, so no two 2x2 blocks overlap
#     - track colors are chosen as [A, B, A] from wall side outward, guaranteeing visible regrouping
#       into [A, A, B] after solving
#   observability:
#     - before solving, humans can already see 2x2 colored blocks rather than single yellow dots
#     - in the output, repeated colors become contiguous piles on a track; that repeated A-B-A -> A-A-B
#       motif is strong enough to infer "same colors gather together"
#     - every generated sample contains at least one track where the fused solver differs from a
#       color-blind rigid-block baseline, so the combo does not collapse back to ordinary stacking
#   cleanliness:
#     - mover colors are reserved away from wall colors, so walls/legends never masquerade as movers
#     - active tracks are few and region count is capped, keeping the grid readable
#     - no hidden color-id ordering is used; only visible track order from the destination side matters
#   new_solver_assumptions_and_generator_support:
#     - every mover component is a full 2x2 square in one of the reserved mover colors
#     - each active region contains at least one long enough slot track aligned with its legend direction
#     - mixed-color input order on that track is not already grouped, so regrouping is exercised
#     - the wall-side first color on a mixed track is preserved in input construction, matching the
#       solver's "first appearance order defines group order" rule


GRID_H, GRID_W = 20, 20
SPRITE_SIZE = 2
MOVER_COLORS = (3, 4)

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [5, 6, 7, 8, 9]
CARDINAL_PROB = 0.8
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 3
REGION_SIZE_BOUNDS = {
    1: (34, 90),
    2: (30, 64),
    3: (28, 44),
}
BLOB_GROW_RETRIES = 40
MIN_FEATURE_SLOTS = 5
MAX_INPUT_SPRITES_PER_REGION = 3

GEN_RETRIES = 450


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        if len(others) < 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _block_cells(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = anchor
    return [(r + dr, c + dc) for dr in range(SPRITE_SIZE) for dc in range(SPRITE_SIZE)]


def _block_cell_set(anchor: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_block_cells(anchor))


def _anchor_fits(region_set: set[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    return all(cell in region_set for cell in _block_cells(anchor))


def _anchor_cells(region_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    anchors = set()
    for r, c in region_set:
        anchor = (int(r), int(c))
        if _anchor_fits(region_set, anchor):
            anchors.add(anchor)
    return anchors


def _block_step_valid(region_set: set[tuple[int, int]], anchor: tuple[int, int], dr: int, dc: int) -> bool:
    if not _anchor_fits(region_set, anchor):
        return False
    r, c = anchor
    nxt = (r + dr, c + dc)
    if not _anchor_fits(region_set, nxt):
        return False
    if dr != 0 and dc != 0:
        if not _anchor_fits(region_set, (r + dr, c)):
            return False
        if not _anchor_fits(region_set, (r, c + dc)):
            return False
    return True


def _extract_sprite_anchors(
    g: np.ndarray, pts: np.ndarray
) -> tuple[list[tuple[tuple[int, int], int]], list[tuple[int, int]]]:
    anchors_with_color: list[tuple[tuple[int, int], int]] = []
    mover_cells: list[tuple[int, int]] = []
    pts_list = [(int(r), int(c)) for r, c in pts]
    for mover in MOVER_COLORS:
        cells = {(r, c) for r, c in pts_list if int(g[r, c]) == mover}
        remaining = set(cells)
        while remaining:
            start = next(iter(remaining))
            stack = [start]
            comp = set()
            while stack:
                cur = stack.pop()
                if cur in comp or cur not in remaining:
                    continue
                comp.add(cur)
                remaining.remove(cur)
                for nxt in neighbors4(*cur):
                    if nxt in remaining:
                        stack.append(nxt)
            if len(comp) != 4:
                raise ValueError("mover component is not a 2x2 sprite")
            coords = np.array(list(comp), dtype=int)
            r0, c0, r1, c1 = bbox_coords(coords)
            if (r1 - r0 + 1, c1 - c0 + 1) != (SPRITE_SIZE, SPRITE_SIZE):
                raise ValueError("mover component bbox is not 2x2")
            expected = _block_cell_set((r0, c0))
            if comp != expected:
                raise ValueError("mover component is not a filled 2x2 block")
            anchors_with_color.append(((r0, c0), mover))
            mover_cells.extend(sorted(comp))
    return anchors_with_color, mover_cells


def _region_border_color(g: np.ndarray, pts: np.ndarray, pset: set[tuple[int, int]], bg: int) -> int | None:
    h, w = g.shape
    border_counts: dict[int, int] = {}
    for r, c in pts:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in pset:
                continue
            v = int(g[nr, nc])
            if v == bg or v in MOVER_COLORS:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None
    return max(border_counts, key=border_counts.get)


def _anchor_tracks(region_set: set[tuple[int, int]], dr: int, dc: int) -> list[list[tuple[int, int]]]:
    anchors = _anchor_cells(region_set)
    if not anchors:
        return []

    def has_prev(anchor: tuple[int, int]) -> bool:
        prev = (anchor[0] - dr, anchor[1] - dc)
        return _block_step_valid(region_set, prev, dr, dc)

    starts = sorted(
        [anchor for anchor in anchors if not has_prev(anchor)],
        key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]),
    )
    tracks: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in starts:
        if start in seen:
            continue
        track = [start]
        seen.add(start)
        cur = start
        while _block_step_valid(region_set, cur, dr, dc):
            cur = (cur[0] + dr, cur[1] + dc)
            if cur in seen:
                break
            track.append(cur)
            seen.add(cur)
        tracks.append(track)

    for anchor in sorted(anchors):
        if anchor in seen:
            continue
        cur = anchor
        while has_prev(cur):
            cur = (cur[0] - dr, cur[1] - dc)
        track = [cur]
        seen.add(cur)
        while _block_step_valid(region_set, cur, dr, dc):
            cur = (cur[0] + dr, cur[1] + dc)
            if cur in seen:
                break
            track.append(cur)
            seen.add(cur)
        tracks.append(track)
    return tracks


def _track_slots_near(track: list[tuple[int, int]]) -> list[tuple[int, int]]:
    slots: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for anchor in reversed(track):
        cells = _block_cell_set(anchor)
        if cells & occupied:
            continue
        slots.append(anchor)
        occupied.update(cells)
    return slots


def _slide_grouped_blocks(
    region_set: set[tuple[int, int]],
    anchors_with_color: list[tuple[tuple[int, int], int]],
    dr: int,
    dc: int,
) -> list[tuple[tuple[int, int], int]]:
    anchor_to_color = {anchor: color for anchor, color in anchors_with_color}
    out: list[tuple[tuple[int, int], int]] = []
    for track in _anchor_tracks(region_set, dr, dc):
        present_near = [(anchor, anchor_to_color[anchor]) for anchor in reversed(track) if anchor in anchor_to_color]
        if not present_near:
            continue
        color_order: list[int] = []
        counts: Counter[int] = Counter()
        for _, color in present_near:
            counts[color] += 1
            if color not in color_order:
                color_order.append(color)
        slots_near = _track_slots_near(track)
        if len(slots_near) < len(present_near):
            raise ValueError("track does not have enough physical slots for grouped packing")
        idx = 0
        for color in color_order:
            for _ in range(counts[color]):
                out.append((slots_near[idx], color))
                idx += 1
    return out


def _slide_colorblind_blocks(
    region_set: set[tuple[int, int]],
    anchors_with_color: list[tuple[tuple[int, int], int]],
    dr: int,
    dc: int,
) -> list[tuple[tuple[int, int], int]]:
    anchor_to_color = {anchor: color for anchor, color in anchors_with_color}
    out: list[tuple[tuple[int, int], int]] = []
    for track in _anchor_tracks(region_set, dr, dc):
        present_near = [(anchor, anchor_to_color[anchor]) for anchor in reversed(track) if anchor in anchor_to_color]
        if not present_near:
            continue
        slots_near = _track_slots_near(track)
        if len(slots_near) < len(present_near):
            raise ValueError("track does not have enough physical slots for baseline packing")
        for idx, (_, color) in enumerate(present_near):
            out.append((slots_near[idx], color))
    return out


def _apply_solver(grid: np.ndarray, slide_fn) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | np.isin(g, MOVER_COLORS)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        anchors_with_color, mover_cells = _extract_sprite_anchors(g, pts)
        if not anchors_with_color:
            continue
        pset = {(int(r), int(c)) for r, c in pts}
        border_col = _region_border_color(g, pts, pset, bg)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        for r, c in mover_cells:
            out[r, c] = bg
        for anchor, color in slide_fn(pset, anchors_with_color, dr, dc):
            for r, c in _block_cells(anchor):
                out[r, c] = color
    return out


def _colorblind_baseline(grid: np.ndarray) -> np.ndarray:
    return _apply_solver(grid, _slide_colorblind_blocks)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored 2x2 blocks inside walled regions using border legends, then regroup them by color on each track.

    Input structure:
    - Background is the most common color. Border-touching legend rectangles still encode directions, and each legend's fill color matches the wall color of one or more enclosed regions.
    - Inside each enclosed region, movers are no longer single yellow cells: they are separated solid 2x2 sprites in two reserved colors.
    - A region's interior defines the legal 2x2 anchor tracks for its legend direction; movers on the same track are the only ones that can interact.

    Rule:
    - For each enclosed region, read its border color, fetch the matching legend direction, and recover every 2x2 mover sprite with its color.
    - On each slide track, scan movers from the destination wall outward. The first appearance order of colors on that track determines the wall-side order of color groups.
    - Repack the farthest non-overlapping 2x2 slots on that track by color groups: same-color sprites stack contiguously, while differently colored sprites may pass through one another to join their group.

    Key insight:
    - The legend-to-wall-color link from the base task still chooses the direction, but motion is now solved on rigid 2x2 anchor tracks rather than on single pixels.
    - The crucial new cue is not numeric color id; it is the visible color sequence along a track from the destination side. That sequence's first-occurrence order stays fixed while repeated colors collapse into contiguous groups.
    """
    return _apply_solver(grid, _slide_grouped_blocks)


def _grow_blob(
    canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _track_infos(region_set: set[tuple[int, int]], dr: int, dc: int) -> list[dict]:
    infos = []
    for track in _anchor_tracks(region_set, dr, dc):
        slots_near = _track_slots_near(track)
        if len(slots_near) >= 2:
            infos.append({"anchors_far": track, "slots_near": slots_near})
    infos.sort(key=lambda info: (-len(info["slots_near"]), info["slots_near"][0]))
    return infos


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
    dr: int,
    dc: int,
) -> tuple[set[tuple[int, int]], list[dict]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        tracks = _track_infos(cand, dr, dc)
        if not any(len(track["slots_near"]) >= MIN_FEATURE_SLOTS for track in tracks):
            continue
        return cand, tracks
    return None


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wc in zip(corners, wall_colors):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wc, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))
    return legend_occupied, legends


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    region_colors = []
    for wall_color in all_colors:
        region_colors.extend([wall_color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
        dr, dc = color_dir[rc]
        target_size = rng.randint(size_min, size_max)

        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        grown = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w, dr, dc)
        if grown is None:
            continue
        blob, tracks = grown
        walls = _blob_wall_cells(blob, h, w)
        feature_tracks = [track for track in tracks if len(track["slots_near"]) >= MIN_FEATURE_SLOTS]
        if not feature_tracks:
            continue
        regions.append(
            {
                "color": rc,
                "dir": (dr, dc),
                "interior": blob,
                "walls": walls,
                "tracks": tracks,
                "feature_tracks": feature_tracks,
            }
        )

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(all_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _paint_block(grid: np.ndarray, anchor: tuple[int, int], color: int) -> None:
    for r, c in _block_cells(anchor):
        grid[r, c] = color


def _place_region_sprites(rng: random.Random, reg: dict, grid: np.ndarray) -> bool:
    track = rng.choice(reg["feature_tracks"])
    slots_near = track["slots_near"]
    if len(slots_near) < MIN_FEATURE_SLOTS:
        return False

    group0, group1 = rng.sample(list(MOVER_COLORS), 2)
    output_assignments = [
        (slots_near[0], group0),
        (slots_near[1], group0),
        (slots_near[2], group1),
    ]

    candidate_indices = list(range(1, len(slots_near)))
    if len(candidate_indices) < MAX_INPUT_SPRITES_PER_REGION:
        return False
    input_indices = sorted(rng.sample(candidate_indices, MAX_INPUT_SPRITES_PER_REGION))
    input_assignments = [
        (slots_near[input_indices[0]], group0),
        (slots_near[input_indices[1]], group1),
        (slots_near[input_indices[2]], group0),
    ]

    for anchor, color in input_assignments:
        _paint_block(grid, anchor, color)

    reg["output_assignments"] = output_assignments
    reg["input_assignments"] = input_assignments
    return True


def _place_sprites(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        if not _place_region_sprites(rng, reg, grid):
            return False
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (~np.isin(input_grid, MOVER_COLORS)) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int(np.isin(solver_out, MOVER_COLORS).sum()) != int(np.isin(input_grid, MOVER_COLORS).sum()):
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[np.isin(solver_out, MOVER_COLORS)])):
        return False
    return True


def _behavior_signature(grid: np.ndarray) -> dict[str, int]:
    grouped = main(grid)
    baseline = _colorblind_baseline(grid)
    return {
        "baseline_diff_cells": int(np.sum(grouped != baseline)),
        "mover_cells": int(np.isin(grid, MOVER_COLORS).sum()),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_sprites(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    sig = _behavior_signature(input_grid)
    if sig["baseline_diff_cells"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
