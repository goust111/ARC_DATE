# Combo variant of d35bdbdc (index 31)
# combo_spec: D9/2+D12/0
# Name: 等间距重排+精灵沿边缘排列

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paste


# Design analysis for the fused variant:
# - Concrete behavior trace:
#   Originally the two endpoint-linked sprites stayed in place and the grey trail stayed visible.
#   Here every sprite is edge-anchored in the input, so the trail must be traced from the border
#   inward. After the same color-lookup transfer, the trail is discarded and the two kept sprites
#   are repacked onto a wider clean strip with a fixed large gap. Example: if the connected sprites
#   sit at input centers (1,8) and (11,4), the upper one becomes the left packed sprite because the
#   output order uses visible reading order of the input centers rather than any hidden color value.
# - Complexity calibration:
#   The palette logic stays small: 4-5 total sprites, exactly 2 destinations, and only 1-2 sources.
#   Border colors remain unique, so every destination center has at most one source match. Keeping
#   just two packed outputs makes the spatial rewrite obvious without adding extra ambiguous objects.
# - Hidden assumptions reviewed:
#   The original `_bg_boundary` trick is unsafe once sprites occupy the outer frame, so `main()`
#   now uses global background frequency. Endpoint matching still assumes each grey endpoint is a
#   valid port for exactly one destination; the generator enforces unique inward ports and blocks
#   the path from touching any non-destination port. Output ordering is by geometry, not by color.
# - Placement legality:
#   Input sprites use a pre-validated ring of edge slots, which guarantees no overlap or 4-touching.
#   The path is generated only through free interior cells. Output packing uses equal-size slots with
#   a fixed gap, so destination patches never collide even if a future sample used clipped variants.
# - Visual observability and cleanliness:
#   The input has a clear edge-versus-interior separation: colored sprites on the border, grey trail
#   inside. The output has only two transformed sprites on an empty wider canvas. Because all source
#   clues remain ordinary bordered sprites, the color-transfer rule is still supported by visible
#   repeated structure instead of arbitrary recoloring.


GREY = 5
BG = 0

PACK_PAD = 1
PACK_GAP = 7

INPUT_H = 13
INPUT_W = 13
EDGE_SLOTS = [
    (1, 4), (1, 8),
    (4, 11), (8, 11),
    (11, 8), (11, 4),
    (8, 1), (4, 1),
]

SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
PATH_LEN_MIN = 8
PATH_LEN_MAX = 14
GEN_RETRIES = 400


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the 2 endpoints of the grey trail under 8-connectivity."""
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    assert cells, "expected a grey trail"
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) == 2:
        return deg1[0], deg1[1]

    h, w = g.shape
    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == b:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    n = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == n:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                got = dfs(nb, visited | {nb})
                if got is not None:
                    return [cur] + got
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace the grey trail"
    return path[0], path[-1]


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Find 4-connected two-color components whose rarer color occurs exactly once.

    The repeated color is treated as the border and the singleton as the center.
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
        if border_color in (bg, grey) or center_color in (bg, grey):
            continue

        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue

        rs = [int(r) for r, _c in pts]
        cs = [int(c) for _r, c in pts]
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """List endpoint cells that can serve as this sprite's inward port."""
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def _sprite_patch(g: np.ndarray, sprite: dict, bg: int) -> np.ndarray:
    """Crop the sprite to its bbox, preserving its exact visible shape."""
    r0, c0, r1, c1 = sprite["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in sprite["pts"]:
        patch[int(r) - r0, int(c) - c0] = int(g[r, c])
    return patch


def _pack_patches(patches: list[np.ndarray], bg: int) -> np.ndarray:
    """Arrange patches left-to-right using equal slot sizes and a wide fixed gap."""
    assert patches, "expected at least one patch to pack"
    slot_h = max(int(p.shape[0]) for p in patches)
    slot_w = max(int(p.shape[1]) for p in patches)
    out_h = 2 * PACK_PAD + slot_h
    out_w = 2 * PACK_PAD + len(patches) * slot_w + (len(patches) - 1) * PACK_GAP
    out = np.full((out_h, out_w), int(bg), dtype=int)
    step = slot_w + PACK_GAP
    for i, patch in enumerate(patches):
        rr = PACK_PAD + (slot_h - int(patch.shape[0])) // 2
        cc = PACK_PAD + i * step + (slot_w - int(patch.shape[1])) // 2
        paste(out, patch, rr, cc, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the grey trail to identify two border-framed sprites, transfer matched center colors,
    then discard the trail and repack the kept sprites into an evenly spaced strip.

    Input structure:
    - Background plus a single grey one-stroke trail running through the interior.
    - Four or five small bordered sprites sit along the outer edge of the grid.
    - Exactly two sprites touch the trail endpoints through valid inward ports; the others are
      edge distractors or sources for the color lookup.

    Rule:
    - Use the two grey endpoints to select the destination sprites.
    - For each destination, look for the unique source sprite whose border color equals that
      destination's original center color, and copy the source center color into the destination.
    - Remove every non-destination sprite and the entire grey trail, then place the two transformed
      destinations onto a new clean canvas from left to right in reading order, with fixed spacing.

    Key insight:
    - The grey path is only a selector. The real content is the indirect border->center color match,
      and after that lookup the edge layout is normalized away into a canonical packed arrangement.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    grey = GREY

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) >= 3, "expected multiple bordered sprites"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    opts = {
        te1: [sp for sp in sprites if te1 in _ports_for_sprite(sp, g, trail_eps)],
        te2: [sp for sp in sprites if te2 in _ports_for_sprite(sp, g, trail_eps)],
    }

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    eps_pair = [te1, te2]
    order = [0, 1] if len(opts[te1]) <= len(opts[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        assert cands, "each endpoint should identify a destination sprite"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    ordered_dest = sorted(dest, key=lambda sp: sp["center_pos"])
    packed_patches: list[np.ndarray] = []
    for sp in ordered_dest:
        want_border = int(sp["center"])
        srcs = by_border.get(want_border, [])
        new_center = int(sp["center"])
        if srcs:
            assert len(srcs) == 1, "each destination lookup should have at most one source"
            new_center = int(srcs[0]["center"])

        patch = _sprite_patch(g, sp, bg)
        r0, c0, _r1, _c1 = sp["bbox"]
        cr, cc = sp["center_pos"]
        patch[cr - r0, cc - c0] = new_center
        packed_patches.append(patch)

    return _pack_patches(packed_patches, bg)


# --------------- generator ---------------


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus sprite patch."""
    if shape_type == "box":
        patch = np.full((3, 3), int(border_color), dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = int(border_color)
    patch[1, 1] = int(center_color)
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    """Place a 3x3 patch so that its center lands at `center`."""
    cr, cc = center
    paste(grid, patch, cr - 1, cc - 1, transparent=BG)


def _inward_port(center: tuple[int, int]) -> tuple[int, int]:
    """Unique interior port cell for an edge-anchored sprite center."""
    r, c = center
    if r == 1:
        return 3, c
    if r == INPUT_H - 2:
        return INPUT_H - 4, c
    if c == 1:
        return r, 3
    if c == INPUT_W - 2:
        return r, INPUT_W - 4
    raise ValueError(f"center {center} is not on the edge ring")


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    """Assign roles and center colors while preserving unique border lookups."""
    remaining = list(range(2, num_sprites))
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None

    random.shuffle(remaining)
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        dest_center_colors = [border_colors[source_indices[0]]] * 2
    else:
        chosen = source_indices[:2]
        dest_center_colors = [border_colors[chosen[0]], border_colors[chosen[1]]]

    source_center_colors: list[int] = []
    for j, si in enumerate(source_indices):
        forbidden = {BG, GREY, border_colors[si]}
        if num_sources == 1:
            forbidden.update([border_colors[0], border_colors[1]])
        elif j < 2:
            forbidden.add(border_colors[j])
        candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        source_center_colors.append(random.choice(candidates))

    distractor_center_colors = []
    for di in distractor_indices:
        candidates = [c for c in range(1, 10) if c not in {GREY, border_colors[di]}]
        distractor_center_colors.append(random.choice(candidates))

    output_dest_centers = (source_center_colors * 2)[:2] if num_sources == 1 else source_center_colors[:2]

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
    }


def _king_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _strict_candidate(
    cell: tuple[int, int],
    prev: tuple[int, int],
    visited: set[tuple[int, int]],
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    end: tuple[int, int],
) -> bool:
    r, c = cell
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    if grid[r, c] != BG or cell in visited:
        return False
    if cell in blocked and cell != end:
        return False
    touched = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in visited]
    return all(nb == prev for nb in touched)


def _find_path_exact(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    target_len: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Backtracking search for a simple 8-connected path with a chosen length."""
    path = [start]
    visited = {start}

    def dfs(cur: tuple[int, int]) -> bool:
        steps_left = target_len - len(path)
        if _king_dist(cur, end) > steps_left:
            return False
        if len(path) == target_len:
            return cur == end
        if cur == end:
            return False

        moves: list[tuple[tuple[int, int], int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (cur[0] + dr, cur[1] + dc)
                if not _strict_candidate(nxt, cur, visited, grid, blocked, end):
                    continue
                moves.append((nxt, _king_dist(nxt, end)))

        random.shuffle(moves)
        if steps_left > _king_dist(cur, end) + 2:
            moves.sort(key=lambda item: -item[1])
        else:
            moves.sort(key=lambda item: item[1])

        for nxt, _dist in moves:
            path.append(nxt)
            visited.add(nxt)
            if dfs(nxt):
                return True
            visited.remove(nxt)
            path.pop()
        return False

    return path.copy() if dfs(start) else None


def _endpoint_degree_ok(path: list[tuple[int, int]]) -> bool:
    cells = set(path)
    deg1 = 0
    for r, c in cells:
        deg = sum(1 for nr, nc in neighbors8(r, c) if (nr, nc) in cells)
        if deg == 1:
            deg1 += 1
        elif deg > 2:
            return False
    return deg1 == 2


def _build_expected_output(
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    dest_positions: list[tuple[int, int]],
) -> np.ndarray:
    """Construct the packed output directly from logical destination specs."""
    ordered_idx = sorted(range(2), key=lambda i: dest_positions[i])
    patches = [
        _make_sprite_patch(shape_type, border_colors[i], output_dest_centers[i])
        for i in ordered_idx
    ]
    return _pack_patches(patches, BG)


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    shape_type = random.choice(["box", "plus"])
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample([c for c in range(1, 10) if c != GREY], num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    chosen_slots = random.sample(EDGE_SLOTS, num_sprites)
    dest_positions = chosen_slots[:2]
    other_positions = chosen_slots[2:]

    input_grid = np.full((INPUT_H, INPUT_W), BG, dtype=int)

    for i in range(2):
        patch = _make_sprite_patch(shape_type, border_colors[i], colors["dest_center_colors"][i])
        _place_sprite(input_grid, patch, dest_positions[i])

    source_pairs = list(zip(colors["source_indices"], colors["source_center_colors"], strict=False))
    distractor_pairs = list(zip(colors["distractor_indices"], colors["distractor_center_colors"], strict=False))
    for (idx, center_color), pos in zip(source_pairs + distractor_pairs, other_positions, strict=False):
        patch = _make_sprite_patch(shape_type, border_colors[idx], center_color)
        _place_sprite(input_grid, patch, pos)

    port_cells = [_inward_port(pos) for pos in dest_positions]
    blocked_ports = {_inward_port(pos) for pos in other_positions}

    lengths = list(range(PATH_LEN_MIN, PATH_LEN_MAX + 1))
    random.shuffle(lengths)
    path_cells = None
    for target_len in lengths:
        path_cells = _find_path_exact(input_grid, port_cells[0], port_cells[1], target_len, blocked_ports)
        if path_cells is not None and _endpoint_degree_ok(path_cells):
            break
        path_cells = None
    if path_cells is None:
        return None

    for r, c in path_cells:
        if input_grid[r, c] != BG:
            return None
        input_grid[r, c] = GREY

    expected = _build_expected_output(
        shape_type,
        border_colors,
        colors["output_dest_centers"],
        dest_positions,
    )

    try:
        produced = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    stats = {
        "num_sprites": num_sprites,
        "dest_positions": dest_positions,
        "path_len": len(path_cells),
        "input_shape": tuple(input_grid.shape),
        "output_shape": tuple(expected.shape),
        "blocked_ports_used": sum(1 for cell in path_cells if cell in blocked_ports),
    }
    return input_grid, stats


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        got = _gen_attempt()
        if got is not None:
            return got[0]
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
