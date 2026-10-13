import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERP_DIRS = {
    "up": [(0, -1), (0, 1)],
    "down": [(0, -1), (0, 1)],
    "left": [(-1, 0), (1, 0)],
    "right": [(-1, 0), (1, 0)],
}

BG = 0
BODY_COLORS = (1, 3)
CONNECTOR_COLORS = (2, 4)
GEN_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_of(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _connector_side(connector, bbox):
    r0, c0, r1, c1 = bbox
    r, c = connector
    sides = []
    if r == r0:
        sides.append("up")
    if r == r1:
        sides.append("down")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    assert len(sides) == 1, "connector must lie on exactly one bbox side"
    return sides[0]


def _parse_sprite(grid: np.ndarray, coords):
    g = np.asarray(grid, dtype=int)
    colors = Counter(int(g[r, c]) for r, c in coords)
    assert len(colors) == 2, "each sprite must have one body color and one connector color"
    body_color, body_count = colors.most_common(1)[0]
    connector_colors = [color for color in colors if color != body_color]
    assert len(connector_colors) == 1
    connector_color = int(connector_colors[0])
    assert int(colors[connector_color]) == 1, "connector must be a single pixel"
    assert int(body_count) >= 3, "body must contain at least three pixels"

    connector = next((r, c) for r, c in coords if int(g[r, c]) == connector_color)
    bbox = _bbox_of(coords)
    r0, c0, r1, c1 = bbox
    side = _connector_side(connector, bbox)

    local_cells = []
    for r, c in coords:
        local_cells.append((r - r0, c - c0, int(g[r, c])))

    return {
        "bbox": bbox,
        "top_left": (r0, c0),
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "body_color": int(body_color),
        "connector_color": connector_color,
        "connector_global": connector,
        "connector_local": (connector[0] - r0, connector[1] - c0),
        "side": side,
        "cells_local": local_cells,
    }


def _place_sprite(canvas: np.ndarray, sprite, top_left):
    h, w = canvas.shape
    tr, tc = map(int, top_left)
    for lr, lc, color in sprite["cells_local"]:
        rr = tr + int(lr)
        cc = tc + int(lc)
        assert 0 <= rr < h and 0 <= cc < w, "translated sprite must stay in bounds"
        assert int(canvas[rr, cc]) == BG, "translated sprites may not overlap"
        canvas[rr, cc] = int(color)


def _final_sprite(sprite, top_left):
    tr, tc = map(int, top_left)
    connector_local = sprite["connector_local"]
    connector_global = (tr + connector_local[0], tc + connector_local[1])
    body_cells = set()
    all_cells = set()
    for lr, lc, color in sprite["cells_local"]:
        rr = tr + int(lr)
        cc = tc + int(lc)
        all_cells.add((rr, cc))
        if int(color) == int(sprite["body_color"]):
            body_cells.add((rr, cc))
    return {
        "body_color": sprite["body_color"],
        "connector_color": sprite["connector_color"],
        "connector": connector_global,
        "body_cells": body_cells,
        "all_cells": all_cells,
        "top_left": (tr, tc),
    }


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG, "generated task uses black background"

    components = _find_components(g, bg)
    assert len(components) == 4, "expected exactly four sprites"
    sprites = [_parse_sprite(g, coords) for coords in components]

    by_connector = defaultdict(list)
    for sprite in sprites:
        by_connector[sprite["connector_color"]].append(sprite)
    assert len(by_connector) == 2, "expected exactly two connector colors"

    final_sprites = []
    pair_meta = []
    for connector_color, pair in by_connector.items():
        assert len(pair) == 2, "each connector color must appear on exactly two sprites"
        side_set = {sprite["side"] for sprite in pair}
        if side_set == {"right", "left"}:
            axis = "horizontal"
            anchor = next(sprite for sprite in pair if sprite["side"] == "right")
            partner = next(sprite for sprite in pair if sprite["side"] == "left")
            dock_dr, dock_dc = 0, 1
        elif side_set == {"down", "up"}:
            axis = "vertical"
            anchor = next(sprite for sprite in pair if sprite["side"] == "down")
            partner = next(sprite for sprite in pair if sprite["side"] == "up")
            dock_dr, dock_dc = 1, 0
        else:
            raise AssertionError("paired connectors must face each other on opposite sides")

        ar, ac = anchor["connector_global"]
        pr, pc = partner["connector_global"]
        partner_shift = (ar + dock_dr - pr, ac + dock_dc - pc)

        anchor_final = _final_sprite(anchor, anchor["top_left"])
        partner_final = _final_sprite(
            partner,
            (
                partner["top_left"][0] + partner_shift[0],
                partner["top_left"][1] + partner_shift[1],
            ),
        )
        final_sprites.extend([anchor_final, partner_final])
        pair_meta.append(
            {
                "axis": axis,
                "connector_color": int(connector_color),
                "perp_dirs": PERP_DIRS[anchor["side"]],
                "connector_cells": [
                    (anchor_final["connector"], anchor_final["body_color"]),
                    (partner_final["connector"], partner_final["body_color"]),
                ],
                "body_cells": anchor_final["all_cells"] | partner_final["all_cells"],
            }
        )

    body_grid = np.full_like(g, bg)
    for sprite in final_sprites:
        for (r, c) in sprite["all_cells"]:
            color = sprite["connector_color"] if (r, c) == sprite["connector"] else sprite["body_color"]
            assert int(body_grid[r, c]) == bg, "final docked sprites may not overlap"
            body_grid[r, c] = int(color)

    body_mask = body_grid != bg
    out = body_grid.copy()
    h, w = out.shape

    for pair in sorted(pair_meta, key=lambda item: item["connector_color"]):
        ray_cells = set()
        for (start_r, start_c), _body_color in pair["connector_cells"]:
            for dr, dc in pair["perp_dirs"]:
                rr = start_r + dr
                cc = start_c + dc
                while 0 <= rr < h and 0 <= cc < w and not bool(body_mask[rr, cc]):
                    out[rr, cc] = int(pair["connector_color"])
                    ray_cells.add((rr, cc))
                    rr += dr
                    cc += dc
        pair["ray_cells"] = ray_cells

    for pair in pair_meta:
        for (r, c), body_color in pair["connector_cells"]:
            out[r, c] = int(body_color)

    if return_meta:
        return out, pair_meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Dock each matching connector pair, then project connector-colored rays perpendicular to the docking seam.

    Input structure:
    - The background is black.
    - There are four small 4-connected multicolor sprites, grouped into two pairs by
      a shared one-pixel connector color.
    - Inside each sprite, the body is blue or green and the unique connector sits on
      exactly one side of the sprite bbox.

    Rule:
    - Pair sprites by connector color.
    - For each pair, use the connector side to identify the docking axis: right faces
      left, and down faces up.
    - Keep the right-facing or down-facing sprite fixed, translate its partner until
      the two connector pixels become adjacent, then recolor those connector pixels to
      the pair's body color.
    - From both docking-seam connector locations, emit rays in the two perpendicular
      directions until the grid edge or another docked sprite body is reached.
    - If rays from the two pairs cross, the pair with the larger connector color is
      drawn last and overwrites the earlier ray color at those crossing cells.

    Key insight:
    - The same connector pixel simultaneously determines the pairing, the docking face,
      and the perpendicular ray direction after docking.
    """
    return _solve(np.asarray(grid, dtype=int))


def _normalize_component(body_cells, connector, body_color, connector_color):
    all_cells = list(body_cells) + [connector]
    r0, c0, r1, c1 = _bbox_of(all_cells)
    local_body = {(r - r0, c - c0) for r, c in body_cells}
    local_connector = (connector[0] - r0, connector[1] - c0)
    side = _connector_side(local_connector, (0, 0, r1 - r0, c1 - c0))

    local_cells = [(r, c, int(body_color)) for r, c in sorted(local_body)]
    local_cells.append((local_connector[0], local_connector[1], int(connector_color)))
    return {
        "body_color": int(body_color),
        "connector_color": int(connector_color),
        "connector_local": local_connector,
        "cells_local": local_cells,
        "side": side,
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
    }


def _sample_component(side, body_color, connector_color):
    for _ in range(200):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        max_area = min(8, box_h * box_w)
        area = random.randint(3, max_area)

        if side == "right":
            attach = (random.randint(0 if box_h < 3 else 1, box_h - 1 if box_h < 3 else box_h - 2), box_w - 1)
            connector = (attach[0], attach[1] + 1)
            preferred = [(attach[0] - 1, attach[1]), (attach[0] + 1, attach[1])]
        elif side == "left":
            attach = (random.randint(0 if box_h < 3 else 1, box_h - 1 if box_h < 3 else box_h - 2), 0)
            connector = (attach[0], attach[1] - 1)
            preferred = [(attach[0] - 1, attach[1]), (attach[0] + 1, attach[1])]
        elif side == "down":
            attach = (box_h - 1, random.randint(0 if box_w < 3 else 1, box_w - 1 if box_w < 3 else box_w - 2))
            connector = (attach[0] + 1, attach[1])
            preferred = [(attach[0], attach[1] - 1), (attach[0], attach[1] + 1)]
        else:
            attach = (0, random.randint(0 if box_w < 3 else 1, box_w - 1 if box_w < 3 else box_w - 2))
            connector = (attach[0] - 1, attach[1])
            preferred = [(attach[0], attach[1] - 1), (attach[0], attach[1] + 1)]

        body = {attach}
        pref = preferred[:]
        random.shuffle(pref)
        for cell in pref:
            if len(body) >= area:
                break
            r, c = cell
            if 0 <= r < box_h and 0 <= c < box_w and random.random() < 0.8:
                body.add(cell)

        while len(body) < area:
            frontier = set()
            for r, c in body:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in body:
                        frontier.add((nr, nc))
            if not frontier:
                break
            body.add(random.choice(tuple(frontier)))
        if len(body) != area:
            continue

        try:
            component = _normalize_component(body, connector, body_color, connector_color)
        except AssertionError:
            continue
        if component["side"] != side:
            continue
        conn_r, conn_c = component["connector_local"]
        if side in {"right", "left"} and component["height"] >= 3 and conn_r in {0, component["height"] - 1}:
            continue
        if side in {"up", "down"} and component["width"] >= 3 and conn_c in {0, component["width"] - 1}:
            continue
        return component
    raise RuntimeError("failed to sample a valid component")


def _cells_at(component, top_left):
    tr, tc = map(int, top_left)
    cells = {}
    for r, c, color in component["cells_local"]:
        cells[(tr + int(r), tc + int(c))] = int(color)
    return cells


def _sprite_coords(component, top_left):
    return set(_cells_at(component, top_left))


def _min_separation(coords_a, coords_b):
    best_l1 = 10**9
    best_cheb = 10**9
    for ra, ca in coords_a:
        for rb, cb in coords_b:
            best_l1 = min(best_l1, abs(ra - rb) + abs(ca - cb))
            best_cheb = min(best_cheb, max(abs(ra - rb), abs(ca - cb)))
    return best_l1, best_cheb


def _build_pair_spec(axis, connector_color):
    body_color = random.choice(BODY_COLORS)
    if axis == "horizontal":
        anchor = _sample_component("right", body_color, connector_color)
        partner = _sample_component("left", body_color, connector_color)
        step = (0, 1)
    else:
        anchor = _sample_component("down", body_color, connector_color)
        partner = _sample_component("up", body_color, connector_color)
        step = (1, 0)

    gap = random.randint(3, 7)
    delta_final = (
        anchor["connector_local"][0] + step[0] - partner["connector_local"][0],
        anchor["connector_local"][1] + step[1] - partner["connector_local"][1],
    )
    delta_input = (delta_final[0] + gap * step[0], delta_final[1] + gap * step[1])
    return {
        "axis": axis,
        "connector_color": int(connector_color),
        "body_color": int(body_color),
        "anchor": anchor,
        "partner": partner,
        "delta_final": delta_final,
        "delta_input": delta_input,
    }


def _anchor_ranges(pair, h, w):
    anchor = pair["anchor"]
    partner = pair["partner"]
    rf, cf = pair["delta_final"]
    ri, ci = pair["delta_input"]

    r_lo = 0
    r_hi = h - anchor["height"]
    r_lo = max(r_lo, -rf, -ri)
    r_hi = min(r_hi, h - partner["height"] - rf, h - partner["height"] - ri)

    c_lo = 0
    c_hi = w - anchor["width"]
    c_lo = max(c_lo, -cf, -ci)
    c_hi = min(c_hi, w - partner["width"] - cf, w - partner["width"] - ci)
    return r_lo, r_hi, c_lo, c_hi


def _compose_input(h, w, pair_specs, anchor_tops):
    grid = np.full((h, w), BG, dtype=int)
    sprites = []
    for pair, anchor_top in zip(pair_specs, anchor_tops):
        ar, ac = anchor_top
        dr, dc = pair["delta_input"]
        partner_top = (ar + dr, ac + dc)
        for component, top in ((pair["anchor"], anchor_top), (pair["partner"], partner_top)):
            cells = _cells_at(component, top)
            for (r, c), color in cells.items():
                if not (0 <= r < h and 0 <= c < w):
                    return None, None
                if int(grid[r, c]) != BG:
                    return None, None
                grid[r, c] = int(color)
            sprites.append(set(cells))
    return grid, sprites


def _valid_spacing(sprite_sets):
    for i in range(len(sprite_sets)):
        for j in range(i + 1, len(sprite_sets)):
            l1, cheb = _min_separation(sprite_sets[i], sprite_sets[j])
            if l1 < 3 or cheb < 2:
                return False
    return True


def _pair_color_by_axis(pair_meta, axis):
    for pair in pair_meta:
        if pair["axis"] == axis:
            return pair["connector_color"]
    raise AssertionError("missing axis")


def _candidate_ok(grid):
    try:
        out, pair_meta = _solve(grid, return_meta=True)
    except AssertionError:
        return False
    axes = {pair["axis"] for pair in pair_meta}
    if axes != {"horizontal", "vertical"}:
        return False

    h_color = _pair_color_by_axis(pair_meta, "horizontal")
    v_color = _pair_color_by_axis(pair_meta, "vertical")
    high_color = max(h_color, v_color)

    horizontal = next(pair for pair in pair_meta if pair["axis"] == "horizontal")
    vertical = next(pair for pair in pair_meta if pair["axis"] == "vertical")
    crossings = horizontal["ray_cells"] & vertical["ray_cells"]
    if len(crossings) < 4:
        return False
    if any(int(out[r, c]) != high_color for r, c in crossings):
        return False
    if any(len(pair["ray_cells"]) < 12 for pair in pair_meta):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(18, 26)
        w = random.randint(18, 26)

        color_order = list(CONNECTOR_COLORS)
        random.shuffle(color_order)
        pair_specs = [
            _build_pair_spec("horizontal", color_order[0]),
            _build_pair_spec("vertical", color_order[1]),
        ]
        random.shuffle(pair_specs)

        anchor_tops = []
        ok = True
        for pair in pair_specs:
            r_lo, r_hi, c_lo, c_hi = _anchor_ranges(pair, h, w)
            if r_lo > r_hi or c_lo > c_hi:
                ok = False
                break
            anchor_tops.append(
                (
                    random.randint(r_lo, r_hi),
                    random.randint(c_lo, c_hi),
                )
            )
        if not ok:
            continue

        grid, sprite_sets = _compose_input(h, w, pair_specs, anchor_tops)
        if grid is None or not _valid_spacing(sprite_sets):
            continue
        if not _candidate_ok(grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 2")
