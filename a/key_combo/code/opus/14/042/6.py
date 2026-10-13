import random
from collections import Counter, defaultdict, deque

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
PALETTE = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    allowed_mask: np.ndarray | None = None,
) -> list[dict]:
    h, w = grid.shape
    if allowed_mask is None:
        allowed_mask = grid != bg
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not allowed_mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and allowed_mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            box = _bbox(cells)
            out.append(
                {
                    "cells": cells,
                    "bbox": box,
                    "area": len(cells),
                    "colors": Counter(int(grid[rr, cc]) for rr, cc in cells),
                }
            )
    return out


def _paste(dst: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = sprite != bg
    target = dst[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    assert target.shape == sprite.shape
    assert not np.any(mask & (target != bg))
    target[mask] = sprite[mask]


def _apply_chain(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    for src, dst in pairs:
        if current == src:
            current = dst
    return current


def _chain_steps(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    steps = 0
    for src, dst in pairs:
        if current == src:
            current = dst
            steps += 1
    return steps


def _legend_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    candidates = []
    for comp in _find_components(grid, bg=bg, connectivity=8):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if w != 2 or h not in (3, 4):
            continue
        if comp["area"] != 2 * h:
            continue
        sub = _crop(grid, comp["bbox"])
        if np.any(sub == bg):
            continue
        if len(set(map(int, sub.ravel()))) < 3:
            continue
        candidates.append(comp["bbox"])
    return sorted(candidates)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates = _legend_candidates(grid, bg)
    assert candidates
    best = candidates[0]
    assert all((r0, c0) != (best[0], best[1]) for r0, c0, _r1, _c1 in candidates[1:])
    return best


def _connector_dir(sprite: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int]:
    h, w = sprite.shape
    dirs = []
    if r == 0:
        dirs.append((-1, 0))
    if r == h - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == w - 1:
        dirs.append((0, 1))
    assert len(dirs) == 1
    dr, dc = dirs[0]
    nr, nc = r + dr, c + dc
    if 0 <= nr < h and 0 <= nc < w:
        assert sprite[nr, nc] == bg
    return dr, dc


def _parse_objects(grid: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int], pairs: list[tuple[int, int]]) -> list[dict]:
    allowed = (grid != bg).copy()
    r0, c0, r1, c1 = legend_bbox
    allowed[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _find_components(grid, bg=bg, connectivity=4, allowed_mask=allowed)
    objects = []
    for comp in comps:
        sprite = _crop(grid, comp["bbox"])
        vals = [int(v) for v in sprite.ravel() if int(v) != bg]
        counts = Counter(vals)
        body, body_count = max(counts.items(), key=lambda item: item[1])
        assert body_count > sum(n for color, n in counts.items() if color != body)
        markers = []
        for rr, cc in np.argwhere((sprite != bg) & (sprite != body)):
            rr = int(rr)
            cc = int(cc)
            surface = int(sprite[rr, cc])
            direction = _connector_dir(sprite, rr, cc, bg)
            markers.append(
                {
                    "surface": surface,
                    "true": _apply_chain(surface, pairs),
                    "pos": (rr, cc),
                    "dir": direction,
                }
            )
        assert 1 <= len(markers) <= 2
        objects.append({"sprite": sprite, "markers": markers})
    assert 3 <= len(objects) <= 4
    return objects


def _assemble(objects: list[dict], bg: int) -> np.ndarray:
    marker_occurrences: dict[int, list[tuple[int, tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    for idx, obj in enumerate(objects):
        for marker in obj["markers"]:
            marker_occurrences[int(marker["true"])].append((idx, marker["pos"], marker["dir"]))
    adjacency = [[] for _ in objects]
    for true_color, occs in marker_occurrences.items():
        assert len(occs) == 2, true_color
        (a_idx, a_pos, a_dir), (b_idx, b_pos, b_dir) = occs
        assert a_idx != b_idx
        assert b_dir == (-a_dir[0], -a_dir[1])
        delta = (a_pos[0] + a_dir[0] - b_pos[0], a_pos[1] + a_dir[1] - b_pos[1])
        adjacency[a_idx].append((b_idx, delta))
        adjacency[b_idx].append((a_idx, (-delta[0], -delta[1])))
    placements = {0: (0, 0)}
    queue = deque([0])
    while queue:
        idx = queue.popleft()
        base_r, base_c = placements[idx]
        for nbr, (dr, dc) in adjacency[idx]:
            pos = (base_r + dr, base_c + dc)
            if nbr in placements:
                assert placements[nbr] == pos
            else:
                placements[nbr] = pos
                queue.append(nbr)
    assert len(placements) == len(objects)

    fg_cells = []
    for idx, obj in enumerate(objects):
        br, bc = placements[idx]
        for rr, cc in np.argwhere(obj["sprite"] != bg):
            fg_cells.append((br + int(rr), bc + int(cc)))
    r0, c0, r1, c1 = _bbox(fg_cells)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for idx, obj in enumerate(objects):
        br, bc = placements[idx]
        _paste(out, obj["sprite"], br - r0, bc - c0, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the legend-driven connector colors and stitch the objects by their hidden docking colors.

    Input structure:
    - The grid has a black background with one isolated two-column legend near the upper-left; each legend row is a source/target color pair.
    - Elsewhere there are 3 or 4 separated 4-connected foreground objects with one dominant body color each.
    - Every object has 1 or 2 single-pixel connector markers on non-corner bbox edges, and marker surface colors do not directly reveal their partners.

    Rule:
    - Read the legend from top to bottom as an ordered replacement chain from left color to right color.
    - For every connector marker, apply the full chain to its surface color; the final color is the marker's hidden docking color.
    - Pair the two markers with the same hidden docking color, use each marker's exposed edge direction to translate the objects until the paired markers become 4-neighbors, then crop away the background and omit the legend.

    Key insight:
    - The visible marker colors are decoys: only the legend's ordered chain reveals which ports truly match, and the port side determines the exact stitching translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    legend_bbox = _find_legend(g, bg)
    r0, c0, r1, c1 = legend_bbox
    pairs = [(int(g[r, c0]), int(g[r, c1])) for r in range(r0, r1 + 1)]
    objects = _parse_objects(g, bg, legend_bbox, pairs)
    return _assemble(objects, bg)


def _random_side_walk(length: int) -> list[str] | None:
    dirs = list(DIRS)
    for _ in range(200):
        used = {(0, 0)}
        pos = (0, 0)
        walk = []
        prev = None
        ok = True
        for _step in range(length):
            choices = []
            for side in dirs:
                if prev is not None and side == OPPOSITE[prev]:
                    continue
                dr, dc = DIRS[side]
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt not in used:
                    choices.append(side)
            if not choices:
                ok = False
                break
            side = random.choice(choices)
            walk.append(side)
            dr, dc = DIRS[side]
            pos = (pos[0] + dr, pos[1] + dc)
            used.add(pos)
            prev = side
        if ok:
            return walk
    return None


def _build_sprite(body: int, ports: list[tuple[int, str]]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    sprite = np.full((h, w), body, dtype=int)
    port_pos = {}
    for color, side in ports:
        if side in ("U", "D"):
            c = random.randint(1, w - 2)
            r = 0 if side == "U" else h - 1
        else:
            r = random.randint(1, h - 2)
            c = 0 if side == "L" else w - 1
        if side in port_pos:
            return None, None
        if (r, c) in port_pos.values():
            return None, None
        port_pos[side] = (r, c)
        sprite[r, c] = color

    if len(ports) == 2 and random.random() < 0.6:
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] == body
            and 0 < r < h - 1
            and 0 < c < w - 1
            and (r, c) not in port_pos.values()
        ]
        if candidates:
            rr, cc = random.choice(candidates)
            sprite[rr, cc] = 0
            mask = sprite != 0
            seen = set()
            q = deque([next(iter(zip(*np.where(mask))))])
            while q:
                cell = q.popleft()
                if cell in seen:
                    continue
                seen.add(cell)
                for nr, nc in _neighbors(cell[0], cell[1], 4):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                        q.append((nr, nc))
            if len(seen) != int(mask.sum()):
                sprite[rr, cc] = body
    return sprite, port_pos


def _legend_plan(n_connections: int) -> tuple[list[tuple[int, int]], list[int]]:
    if n_connections == 3:
        color_count = 7
    else:
        color_count = 5
    colors = list(random.sample(PALETTE, color_count))
    if n_connections == 3:
        a, b, c, d, e, f, g = colors[:7]
        rows = [(a, b), (b, c), (d, e), (f, g)]
        chain_rows = rows[:2]
        others = rows[2:]
        random.shuffle(others)
        pairs = [chain_rows[0]]
        if random.random() < 0.5:
            pairs.extend(others)
            pairs.append(chain_rows[1])
        else:
            pairs.append(chain_rows[1])
            pairs.extend(others)
        return pairs, [c, e, g]

    if len(colors) == 5:
        a, b, c, d, e = colors
        rows = [(a, b), (b, c), (d, e)]
        insert = random.randint(0, 1)
        pairs = rows[:1] + [rows[2]] + rows[1:] if insert else rows
        return pairs, [c, e]

    a, b, c, d, e, f = colors[:6]
    first = [(a, b), (b, c)]
    second = [(d, e), (e, f)]
    merged = []
    i = j = 0
    while i < 2 or j < 2:
        take_first = j == 2 or (i < 2 and random.random() < 0.5)
        if take_first:
            merged.append(first[i])
            i += 1
        else:
            merged.append(second[j])
            j += 1
    return merged, [c, f]


def _surface_choices(pairs: list[tuple[int, int]], finals: list[int]) -> list[tuple[int, int]]:
    choices = []
    used = set()
    for index, final in enumerate(finals):
        candidates = [c for c in PALETTE if _apply_chain(c, pairs) == final]
        candidates.sort(key=lambda color: (_chain_steps(color, pairs), color), reverse=True)
        valid = [(a, b) for a in candidates for b in candidates if a < b]
        valid = [pair for pair in valid if pair[0] not in used and pair[1] not in used]
        assert valid
        if index == 0:
            changing = [pair for pair in valid if _chain_steps(pair[0], pairs) > 0 and _chain_steps(pair[1], pairs) > 0]
            if changing:
                pair = random.choice(changing)
            else:
                pair = valid[0]
        else:
            pair = random.choice(valid)
        used.update(pair)
        choices.append(pair)
    return choices


def _render_output(sprites: list[np.ndarray], bg: int, walk: list[str], edge_marker_info: list[tuple[int, int, tuple[int, int], tuple[int, int], int]]) -> np.ndarray:
    objects = []
    for sprite in sprites:
        markers = []
        for rr, cc in np.argwhere(sprite != bg):
            color = int(sprite[rr, cc])
            body = Counter(int(v) for v in sprite.ravel() if int(v) != bg).most_common(1)[0][0]
            if color == body:
                continue
            markers.append({"surface": color, "true": None, "pos": (int(rr), int(cc)), "dir": _connector_dir(sprite, int(rr), int(cc), bg)})
        objects.append({"sprite": sprite, "markers": markers})
    for idx_a, idx_b, pos_a, pos_b, true_color in edge_marker_info:
        for marker in objects[idx_a]["markers"]:
            if marker["pos"] == pos_a:
                marker["true"] = true_color
        for marker in objects[idx_b]["markers"]:
            if marker["pos"] == pos_b:
                marker["true"] = true_color
    return _assemble(objects, bg)


def _make_instance() -> tuple[np.ndarray, np.ndarray]:
    bg = 0
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    n_objects = random.choice([3, 4])
    n_connections = n_objects - 1
    walk = _random_side_walk(n_connections)
    if walk is None:
        raise ValueError("walk generation failed")

    legend_pairs, true_colors = _legend_plan(n_connections)
    marker_surface_pairs = _surface_choices(legend_pairs, true_colors)

    port_specs = [[] for _ in range(n_objects)]
    for edge_index, side in enumerate(walk):
        left_color, right_color = marker_surface_pairs[edge_index]
        port_specs[edge_index].append((left_color, side))
        port_specs[edge_index + 1].append((right_color, OPPOSITE[side]))

    body_colors = []
    for spec in port_specs:
        blocked = {color for color, _side in spec}
        choices = [color for color in PALETTE if color not in blocked]
        body_colors.append(random.choice(choices))

    sprites = []
    port_positions = []
    for obj_index in range(n_objects):
        sprite, side_to_pos = _build_sprite(body_colors[obj_index], port_specs[obj_index])
        if sprite is None:
            raise ValueError("sprite generation failed")
        sprites.append(sprite)
        port_positions.append(side_to_pos)

    edge_marker_info = []
    for edge_index, side in enumerate(walk):
        true_color = true_colors[edge_index]
        pos_a = port_positions[edge_index][side]
        pos_b = port_positions[edge_index + 1][OPPOSITE[side]]
        edge_marker_info.append((edge_index, edge_index + 1, pos_a, pos_b, true_color))

    assembled_objects = []
    for obj_index, sprite in enumerate(sprites):
        markers = []
        for surface, side in port_specs[obj_index]:
            pos = port_positions[obj_index][side]
            markers.append(
                {
                    "surface": surface,
                    "true": next(
                        true
                        for a, b, pa, pb, true in edge_marker_info
                        if (a == obj_index and pa == pos) or (b == obj_index and pb == pos)
                    ),
                    "pos": pos,
                    "dir": DIRS[side],
                }
            )
        assembled_objects.append({"sprite": sprite, "markers": markers})

    output = _assemble(assembled_objects, bg)
    if max(output.shape) > 30:
        raise ValueError("output too large")

    legend_h = len(legend_pairs)
    grid = np.full((h, w), bg, dtype=int)
    legend_top = random.randint(1, 3)
    legend_left = random.randint(1, 3)
    for r, (src, dst) in enumerate(legend_pairs):
        grid[legend_top + r, legend_left] = src
        grid[legend_top + r, legend_left + 1] = dst

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[legend_top - 1 : legend_top + legend_h + 1, legend_left - 1 : legend_left + 3] = True

    order = list(range(n_objects))
    random.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        sh, sw = sprite.shape
        spots = []
        for top in range(1, h - sh):
            for left in range(1, w - sw):
                if occupied[top - 1 : top + sh + 1, left - 1 : left + sw + 1].any():
                    continue
                spots.append((top, left))
        if not spots:
            raise ValueError("input placement failed")
        top, left = random.choice(spots)
        block = grid[top : top + sh, left : left + sw]
        mask = sprite != bg
        assert not np.any(block[mask] != bg)
        block[mask] = sprite[mask]
        occupied[top - 1 : top + sh + 1, left - 1 : left + sw + 1] = True

    return grid, output


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> None:
    bg = _get_bg(inp)
    legend_bbox = _find_legend(inp, bg)
    r0, c0, r1, c1 = legend_bbox
    pairs = [(int(inp[r, c0]), int(inp[r, c1])) for r in range(r0, r1 + 1)]
    objects = _parse_objects(inp, bg, legend_bbox, pairs)
    marker_surfaces = [marker["surface"] for obj in objects for marker in obj["markers"]]
    assert len(marker_surfaces) == len(set(marker_surfaces))
    decoded = defaultdict(list)
    for obj in objects:
        for marker in obj["markers"]:
            decoded[marker["true"]].append(marker)
    assert len(decoded) == len(objects) - 1
    assert any(_chain_steps(marker["surface"], pairs) >= 2 for obj in objects for marker in obj["markers"])
    for occs in decoded.values():
        assert len(occs) == 2
        assert occs[0]["dir"] == (-occs[1]["dir"][0], -occs[1]["dir"][1])
        assert occs[0]["surface"] != occs[1]["surface"]
    out = main(inp)
    assert np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, expected = _make_instance()
            _validate_instance(inp, expected)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid instance")
