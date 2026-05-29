# Combo variant of d35bdbdc (index 21)
# combo_spec: D1/4+D10/1
# Name: 尺寸排序匹配+多源对一目标

"""
Design analysis:
  fusion_goal:
    keep:
      - 10x10 canvas, one thin grey trail, and the original family of clipped 3x3 bordered source sprites.
      - The grey trail still selects the kept object, and every detached colored object is deleted in the output.
    change:
      - Replace the two independent small destinations with one larger rectangular aggregation target.
      - Replace border-color lookup with visible-area ordering: detached sources are sorted by clipped area,
        then their center colors are written into the target slots in that size order.

  a_specificization_and_trace:
    concrete_scene:
      - One full rectangular target of size 3x4/3x5 or 4x3/5x3 has a uniform border and a central strip of 2-3 slots.
      - A grey trail connects the two external target ports, so both trail endpoints point to that same target.
      - Elsewhere there are 2-3 small bordered sprites of one shared shape type (all box or all plus).
      - Those small sprites are clipped to distinct visible areas:
        box areas in {4, 6, 9} or plus areas in {3, 4, 5}.
    traced_step:
      original_logic:
        - Endpoint A selected one small destination, endpoint B selected another.
        - Each destination center color was a lookup key into source border colors.
      fused_logic:
        - The two endpoints now jointly identify one large target strip.
        - Ignore the target slot colors as keys; they are just placeholders.
        - Sort detached sources by visible area from largest to smallest.
        - Write the sorted source center colors into the target slots in spatial order
          (left-to-right for horizontal targets, top-to-bottom for vertical targets).
      difference:
        - Matching is no longer color-driven.
        - Multiple sources now aggregate into one kept target instead of recoloring two separate destinations.

  b_complexity_calibration:
    controls:
      - Source count is restricted to 2 or 3, matching the 2 or 3 target slots.
      - Source areas are always distinct, so the size order is decisive and not hidden behind tie-breaking.
      - No extra distractor sprites are added, because once the rule depends on the detached source set,
        unlabeled extras would create ambiguity without adding useful visual evidence.
    visibility:
      - The size signal is coarse and human-countable because it comes from clipped area classes only.
      - The target input slots all share one placeholder color that is not any source border color,
        so the old D10-style color-lookup explanation is visibly disabled.

  c_implicit_assumption_audit:
    solver_changes:
      - The original small-sprite detector assumed every important object had exactly one center pixel.
        That is still valid for sources, but the kept target violates it, so target detection is separate.
      - The original two-endpoint bipartite destination match assumed there were two kept sprites.
        That assumption is removed; both endpoints must belong to one target's two ports.
      - The original `want_border = dest.center` lookup is invalid after D1/4 and is fully removed.
      - Recoloring now depends on `area`, so source detection must record visible component size.
    new_assumptions:
      - Exactly one target component exists and it has 2 or 3 ordered slots.
      - The number of detached small sprites equals the number of target slots.
      - Detached source areas are pairwise distinct.
      - Source placement must avoid visually claiming the target endpoints as alternate ports.

  d_placement_legality:
    guarantees:
      - The large target is always fully in-bounds and leaves both external ports on-grid.
      - Sources are placed only at top-left positions that realize the requested clipped area.
      - `_overlaps` blocks overlap and 4-neighbor touching among all colored components.
      - The grey trail is generated after placing the target, so it can reserve the exact two ports.
      - Sources are added after the trail and rejected if they would also appear connected to a trail endpoint.

  e_visual_observability:
    human_view:
      - The kept object is obvious: the only large rectangle touched by the grey path at both ends.
      - The detached small sprites visibly come in different clipped sizes, and the output preserves only
        the large target whose slots are filled with the detached sprites' center colors in size order.
      - Because all input slots start as the same placeholder color, the solver cannot be explained by
        slot-color lookup; the size-ranked source list is the only stable cue.

  f_cleanliness:
    layout:
      - The scene stays sparse: one target, one trail, and 2-3 detached sources.
      - The target is a clean full rectangle, while sources remain small clipped motifs, so the two object
        roles are easy to distinguish without overlapping visual encodings.

  g_color_change_inferability:
    palette_use:
      - Target border, source borders, source centers, and the shared placeholder color occupy a small palette.
      - Source center colors are distinct and become the ordered slot outputs, so the transformation is
        legible as an ordered transfer rather than a noisy recoloring.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Small bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
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
        if border_color in (bg, grey) or center_color == grey:
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
                "area": int(len(pts)),
            }
        )
    return sprites


def _try_detect_target(g: np.ndarray, pts: np.ndarray, bg: int, grey: int) -> dict | None:
    """
    Aggregation target = a full 3x4/3x5 or 4x3/5x3 rectangle:
      - outer ring is one border color
      - central row/column holds 2-3 non-border slot cells
      - the two ports are immediately outside the short-side midpoints
    """
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if (h, w) not in {(3, 4), (3, 5), (4, 3), (5, 3)}:
        return None
    if len(pts) != h * w:
        return None

    bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    comp_cells = {(int(r), int(c)) for r, c in pts}
    if bbox_cells != comp_cells:
        return None

    if h == 3:
        orientation = "horizontal"
        slot_positions = [(r0 + 1, c) for c in range(c0 + 1, c1)]
        ports = [(r0 + 1, c0 - 1), (r0 + 1, c1 + 1)]
    else:
        orientation = "vertical"
        slot_positions = [(r, c0 + 1) for r in range(r0 + 1, r1)]
        ports = [(r0 - 1, c0 + 1), (r1 + 1, c0 + 1)]
    if any(not (0 <= pr < g.shape[0] and 0 <= pc < g.shape[1]) for pr, pc in ports):
        return None

    slot_set = set(slot_positions)
    border_positions = sorted(bbox_cells - slot_set)
    border_color = int(g[border_positions[0]])
    if border_color in (bg, grey):
        return None
    if any(int(g[r, c]) != border_color for r, c in border_positions):
        return None

    slot_colors = [int(g[r, c]) for r, c in slot_positions]
    if len(slot_colors) not in (2, 3):
        return None
    if any(col in (bg, grey, border_color) for col in slot_colors):
        return None

    return {
        "pts": pts,
        "border": border_color,
        "slot_positions": slot_positions,
        "slot_colors": slot_colors,
        "ports": ports,
        "orientation": orientation,
    }


def _detect_targets(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    targets: list[dict] = []
    for pts in comps:
        detected = _try_detect_target(g, pts, bg, grey)
        if detected is not None:
            targets.append(detected)
    return targets


def _find_trail_endpoints(
    g: np.ndarray, grey: int, targets: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two endpoint cells of the grey one-stroke trail."""
    cells = {(int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False)}
    assert cells, "expected grey trail cells"

    target_ports = {
        port
        for target in targets
        for port in target["ports"]
        if port in cells
    }
    if len(target_ports) == 2:
        a, b = sorted(target_ports)
        return a, b

    adj = {cell: [nb for nb in neighbors8(*cell) if nb in cells] for cell in cells}
    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]
    raise AssertionError("could not identify grey trail endpoints")


def _rank_sources_by_area_desc(sprites: list[dict]) -> list[dict]:
    """Largest visible sprite first; deterministic tie-breaks keep behavior stable."""
    return sorted(
        sprites,
        key=lambda sp: (-int(sp["area"]), int(sp["center_pos"][0]), int(sp["center_pos"][1]), int(sp["border"])),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aggregate detached source colors into one trail-selected target strip by visible-size order, then remove everything else.

    Input structure:
    - The grid contains a grey (5) one-stroke trail, exactly one larger rectangular target, and 2-3 detached small bordered sprites.
    - The large target is a full 3x4/3x5 or 4x3/5x3 rectangle with one border color and a central row or column of 2-3 slots.
    - The grey trail touches that same target at both external ports, while the detached small sprites may be clipped by the grid edge.

    Rule:
    - Use the two grey trail endpoints to identify the single large target they jointly attach to.
    - Delete every colored component except that target.
    - Rank the detached small sprites by visible area from largest to smallest.
    - Fill the target slots in their spatial order using the ranked source center colors.

    Key insight:
    - The grey path still selects what survives, but the slot values are not lookup keys anymore.
      The essential cue is that the detached sources have distinct clipped sizes, and those sizes determine the order in which their center colors are merged into the one kept target.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    targets = _detect_targets(g, bg, grey)
    assert len(targets) == 1, "expected exactly one aggregation target"
    sources = _detect_bordered_sprites(g, bg, grey)
    assert sources, "expected detached bordered source sprites"

    te1, te2 = _find_trail_endpoints(g, grey, targets)
    trail_eps = {te1, te2}
    dests = [target for target in targets if set(target["ports"]) == trail_eps]
    assert len(dests) == 1, "expected exactly one target touched by both trail endpoints"
    dest = dests[0]

    ranked_sources = _rank_sources_by_area_desc(sources)
    assert len(ranked_sources) == len(dest["slot_positions"]), "source count must match target slot count"

    out = g.copy()
    keep = {(int(r), int(c)) for r, c in dest["pts"]}
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if out[r, c] in (bg, grey):
                continue
            if (r, c) not in keep:
                out[r, c] = bg

    for (r, c), src in zip(dest["slot_positions"], ranked_sources, strict=True):
        out[r, c] = int(src["center"])

    return out


# --------------- generator (combo) ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SOURCE_COUNTS = [2, 3]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 600


def _make_small_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus sprite with a single center cell."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _make_target_patch(orientation: str, border_color: int, slot_colors: list[int]) -> np.ndarray:
    """Build the large target whose center strip has 2-3 slots."""
    n = len(slot_colors)
    if orientation == "horizontal":
        patch = np.full((3, n + 2), border_color, dtype=int)
        for i, color in enumerate(slot_colors, start=1):
            patch[1, i] = color
    else:
        patch = np.full((n + 2, 3), border_color, dtype=int)
        for i, color in enumerate(slot_colors, start=1):
            patch[i, 1] = color
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    for dr in range(h):
        for dc in range(w):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap or 4-touch with existing non-bg, non-grey objects."""
    h, w = patch.shape
    patch_cells = set()
    for dr in range(h):
        for dc in range(w):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < GRID_H and 0 <= cc < GRID_W):
                continue
            if grid[rr, cc] != BG:
                return True
            patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(grid[nr, nc]) not in (BG, GREY):
                return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid top-lefts for a small 3x3 sprite whose center stays on-grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _visible_area(patch: np.ndarray, r0: int, c0: int) -> int:
    area = 0
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                area += 1
    return int(area)


def _positions_by_area(shape_type: str) -> dict[int, list[tuple[int, int]]]:
    """Group candidate top-left positions by realized visible clipped area."""
    probe = _make_small_sprite_patch(shape_type, 1, 2)
    out: dict[int, list[tuple[int, int]]] = {}
    for r0, c0 in _sprite_positions():
        area = _visible_area(probe, r0, c0)
        out.setdefault(area, []).append((r0, c0))
    return out


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check if adding (r,c) would push any cell's 8-degree above MAX_THICKNESS."""
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
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int,
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path with no overlap into occupied cells."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(m: tuple[int, int, bool]) -> int:
        return abs(m[0] - end[0]) + abs(m[1] - end[1])

    for _ in range(target_len + 20):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if (nr, nc) in visited or grid[nr, nc] != BG:
                    continue
                if _too_thick(nr, nc, visited):
                    continue
                moves.append((nr, nc, dr != 0 and dc != 0))

        if not moves:
            return None

        dist = abs(cr - end[0]) + abs(cc - end[1])
        budget = target_len - len(path)
        diag = [m for m in moves if m[2]]
        away = [m for m in moves if score(m) > dist]

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
            top = [m for m in best if score(m) <= score(best[0]) + 1]
            dtop = [m for m in top if m[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if _too_thick(end[0], end[1], visited):
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
        for nb in neighbors8(r, c):
            if nb in cell_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(cell_set)


def _target_layout(
    orientation: str, num_slots: int, r0: int, c0: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if orientation == "horizontal":
        width = num_slots + 2
        slots = [(r0 + 1, c0 + i) for i in range(1, width - 1)]
        ports = [(r0 + 1, c0 - 1), (r0 + 1, c0 + width)]
    else:
        height = num_slots + 2
        slots = [(r0 + i, c0 + 1) for i in range(1, height - 1)]
        ports = [(r0 - 1, c0 + 1), (r0 + height, c0 + 1)]
    return slots, ports


def _target_positions(orientation: str, num_slots: int) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        h, w = 3, num_slots + 2
        return [(r0, c0) for r0 in range(0, GRID_H - h + 1) for c0 in range(1, GRID_W - w)]
    h, w = num_slots + 2, 3
    return [(r0, c0) for r0 in range(1, GRID_H - h) for c0 in range(0, GRID_W - w + 1)]


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, bc: int) -> tuple[int, int] | None:
    """Find a valid outward port cell for a small source sprite."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == BG:
            return (er, ec)
    return None


def _sprite_claims_endpoint(
    cr: int, cc: int, bc: int, endpoints: set[tuple[int, int]], grid: np.ndarray,
) -> bool:
    """Reject source placements that would also look attached to a target endpoint."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) in endpoints and 0 <= ir < GRID_H and 0 <= ic < GRID_W and grid[ir, ic] == bc:
            return True
    return False


def _scan_order_indices(items: list[dict]) -> list[int]:
    return sorted(
        range(len(items)),
        key=lambda i: (int(items[i]["center_pos"][0]), int(items[i]["center_pos"][1]), int(items[i]["border"])),
    )


def _assign_colors(num_sources: int) -> dict | None:
    palette = [c for c in range(1, 10) if c != GREY]
    needed = 1 + num_sources + num_sources + 1
    if needed > len(palette):
        return None

    shuffled = random.sample(palette, len(palette))
    target_border = shuffled[0]
    source_borders = shuffled[1 : 1 + num_sources]
    source_centers = shuffled[1 + num_sources : 1 + 2 * num_sources]
    filler_color = shuffled[1 + 2 * num_sources]
    return {
        "target_border": target_border,
        "source_borders": source_borders,
        "source_centers": source_centers,
        "filler_color": filler_color,
    }


def _choose_source_areas(shape_type: str, num_sources: int) -> list[int]:
    area_levels = sorted(_positions_by_area(shape_type))
    return random.sample(area_levels, num_sources)


def _place_target(output_grid: np.ndarray, orientation: str, colors: dict, num_slots: int) -> dict | None:
    slot_colors = colors["output_slots"]
    patch = _make_target_patch(orientation, int(colors["target_border"]), slot_colors)
    positions = _target_positions(orientation, num_slots)
    random.shuffle(positions)

    for r0, c0 in positions:
        if _overlaps(output_grid, patch, r0, c0):
            continue
        _place_patch(output_grid, patch, r0, c0)
        slots, ports = _target_layout(orientation, num_slots, r0, c0)
        if all(0 <= pr < GRID_H and 0 <= pc < GRID_W and output_grid[pr, pc] == BG for pr, pc in ports):
            return {
                "r0": r0,
                "c0": c0,
                "orientation": orientation,
                "slots": slots,
                "ports": ports,
                "patch": patch,
            }
        _erase_patch(output_grid, patch, r0, c0)
    return None


def _place_sources(
    input_grid: np.ndarray,
    shape_type: str,
    source_specs: list[dict],
    endpoint_ports: set[tuple[int, int]],
) -> list[dict] | None:
    """Place all detached sources with the requested visible areas."""
    positions_by_area = _positions_by_area(shape_type)
    placement_order = sorted(range(len(source_specs)), key=lambda i: len(positions_by_area[int(source_specs[i]["area"])]))
    placed: list[dict | None] = [None] * len(source_specs)

    for idx in placement_order:
        spec = source_specs[idx]
        patch = _make_small_sprite_patch(shape_type, int(spec["border"]), int(spec["center"]))
        pool = list(positions_by_area[int(spec["area"])])
        random.shuffle(pool)

        success = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_patch(input_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            if _sprite_claims_endpoint(cr, cc, int(spec["border"]), endpoint_ports, input_grid):
                _erase_patch(input_grid, patch, r0, c0)
                continue
            placed[idx] = {
                **spec,
                "center_pos": (cr, cc),
            }
            success = True
            break

        if not success:
            return None

    result = [p for p in placed if p is not None]
    if len(result) != len(source_specs):
        return None
    area_rank = sorted(range(len(result)), key=lambda i: (-int(result[i]["area"]), int(result[i]["border"])))
    if area_rank == _scan_order_indices(result):
        return None
    return result


def _gen_attempt() -> np.ndarray | None:
    num_sources = random.choice(SOURCE_COUNTS)
    colors = _assign_colors(num_sources)
    if colors is None:
        return None

    orientation = random.choice(["horizontal", "vertical"])
    shape_type = random.choice(["box", "plus"])
    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    source_areas = _choose_source_areas(shape_type, num_sources)

    source_specs = [
        {
            "area": int(source_areas[i]),
            "border": int(colors["source_borders"][i]),
            "center": int(colors["source_centers"][i]),
        }
        for i in range(num_sources)
    ]
    ranked_sources = sorted(source_specs, key=lambda sp: (-int(sp["area"]), int(sp["border"])))
    colors["output_slots"] = [int(src["center"]) for src in ranked_sources]

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    target = _place_target(output_grid, orientation, colors, num_sources)
    if target is None:
        return None

    path_cells = _gen_winding_path(output_grid, target["ports"][0], target["ports"][1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    for pr, pc in target["ports"]:
        deg4 = sum(
            1
            for nr, nc in neighbors4(pr, pc)
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg4 > 1:
            return None

    input_grid = output_grid.copy()
    for r, c in target["slots"]:
        input_grid[r, c] = int(colors["filler_color"])

    placed_sources = _place_sources(input_grid, shape_type, source_specs, set(target["ports"]))
    if placed_sources is None:
        return None

    if len(_detect_targets(input_grid, BG, GREY)) != 1:
        return None
    if len(_detect_bordered_sprites(input_grid, BG, GREY)) != num_sources:
        return None
    if len({int(sp["area"]) for sp in placed_sources}) != num_sources:
        return None

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            return input_grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
