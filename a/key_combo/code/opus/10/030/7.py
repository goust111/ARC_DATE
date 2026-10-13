from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
WALL = 5
MARKER = 6
CATALOG_BG = 9

PALETTE = (1, 2, 3, 4, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = DIR4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
RETRY_LIMIT = 400


def neighbors(r: int, c: int, connectivity: int = 4):
    dirs = DIR4 if connectivity == 4 else DIR8
    for dr, dc in dirs:
        yield r + dr, c + dc


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _catalog_component(grid: np.ndarray) -> np.ndarray:
    comps = components_mask(grid != WALL, connectivity=4)
    scored = []
    for comp in comps:
        maroon = int(np.sum(grid[comp[:, 0], comp[:, 1]] == CATALOG_BG))
        if maroon:
            scored.append((maroon, -int(comp[:, 0].min()), comp))
    assert scored, "expected a catalog region containing color 9"
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    mask = np.zeros_like(grid, dtype=bool)
    mask[scored[0][2][:, 0], scored[0][2][:, 1]] = True
    return mask


def _template_border_color(sprite: np.ndarray) -> int:
    visible = sprite[(sprite != BLACK) & (sprite != CATALOG_BG)]
    assert visible.size > 0, "template must contain at least one non-black visible color"
    counts = Counter(map(int, visible.flat))
    border_color, _count = counts.most_common(1)[0]
    return int(border_color)


def _extract_templates(grid: np.ndarray) -> dict[tuple[int, int, int], dict]:
    catalog = _catalog_component(grid)
    visible = catalog & (grid != CATALOG_BG)
    templates: dict[tuple[int, int, int], dict] = {}
    for comp in components_mask(visible, connectivity=4):
        box = bbox_coords(comp)
        sprite = crop_bbox(grid, box)
        h, w = sprite.shape
        key = (_template_border_color(sprite), int(h), int(w))
        assert key not in templates, "template keys must be unique"
        templates[key] = {"sprite": sprite, "box": box}
    assert len(templates) >= 2, "expected 2-3 catalog templates"
    return templates


def _chambers(grid: np.ndarray) -> list[np.ndarray]:
    catalog = _catalog_component(grid)
    comps = []
    for comp in components_mask(grid != WALL, connectivity=4):
        mask = np.zeros_like(grid, dtype=bool)
        mask[comp[:, 0], comp[:, 1]] = True
        if np.array_equal(mask, catalog):
            continue
        if np.any(grid[mask] == MARKER):
            comps.append(mask)
    assert 2 <= len(comps) <= 3, "expected 2-3 marked lower chambers"
    return comps


def _find_rectangle(grid: np.ndarray, chamber: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    rect_mask = chamber & (~np.isin(grid, (BLACK, WALL, MARKER, CATALOG_BG)))
    comps = components_mask(rect_mask, connectivity=4)
    assert len(comps) == 1, "each chamber must contain one colored rectangle"
    coords = comps[0]
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    color = int(grid[coords[0, 0], coords[0, 1]])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == color), "the chamber object must be a solid rectangle"
    return box, color


def _stamp_templates(grid: np.ndarray, templates: dict[tuple[int, int, int], dict]) -> tuple[np.ndarray, list[dict]]:
    out = grid.copy()
    chamber_info: list[dict] = []
    for chamber in _chambers(grid):
        marker_cells = np.argwhere(chamber & (grid == MARKER))
        assert len(marker_cells) == 1, "each chamber must contain exactly one marker"
        rect_box, rect_color = _find_rectangle(grid, chamber)
        r0, c0, r1, c1 = rect_box
        key = (rect_color, int(r1 - r0 + 1), int(c1 - c0 + 1))
        assert key in templates, "rectangle must uniquely match a catalog template"
        sprite = templates[key]["sprite"]
        for rr in range(sprite.shape[0]):
            for cc in range(sprite.shape[1]):
                if int(sprite[rr, cc]) != CATALOG_BG:
                    out[r0 + rr, c0 + cc] = int(sprite[rr, cc])
        chamber_info.append(
            {
                "mask": chamber,
                "marker": tuple(map(int, marker_cells[0])),
                "rect_box": rect_box,
                "key": key,
            }
        )
    return out, chamber_info


def _flood_markers(grid: np.ndarray, chamber_info: list[dict]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for item in chamber_info:
        sr, sc = item["marker"]
        seen = {(sr, sc)}
        queue = deque([(sr, sc)])
        while queue:
            r, c = queue.popleft()
            for nr, nc in neighbors(r, c, 4):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in seen or int(out[nr, nc]) != BLACK:
                    continue
                seen.add((nr, nc))
                out[nr, nc] = MARKER
                queue.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each chamber's solid rectangle to a catalog template, stamp it, then flood-fill through the new template-created topology.

    Input structure
    - Grey wall pixels split the grid into one upper catalog region and 2-3 lower chambers.
    - The upper catalog sits on maroon background and contains 2-3 multi-color templates whose bbox interiors may also include black holes and maroon transparent cells.
    - Each lower chamber contains exactly one solid colored rectangle plus one magenta marker on surrounding black floor.

    Rule
    - Extract each catalog template by its tight bbox and key it by its dominant visible border color together with bbox height and width.
    - In every lower chamber, find the unique solid rectangle, match it to the template with the same color and bbox size, and stamp the full template over that rectangle with maroon cells treated as transparent.
    - After all stamps are placed, flood-fill 4-connected reachable black cells from each magenta marker without crossing grey walls or any non-black stamped template cell.

    Key insight
    - The catalog rectangle is only a lookup key: the real effect comes from stamping the template's internal black tunnels and pockets, because those newly carved passages decide which chamber cells the later flood-fill can reach.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    templates = _extract_templates(g)
    stamped, chamber_info = _stamp_templates(g, templates)
    return _flood_markers(stamped, chamber_info)


def _boundary_openings(sprite: np.ndarray) -> list[tuple[int, int, str]]:
    h, w = sprite.shape
    openings: list[tuple[int, int, str]] = []
    for c in range(w):
        if int(sprite[0, c]) == BLACK:
            openings.append((0, c, "top"))
        if int(sprite[h - 1, c]) == BLACK:
            openings.append((h - 1, c, "bottom"))
    for r in range(h):
        if int(sprite[r, 0]) == BLACK:
            openings.append((r, 0, "left"))
        if int(sprite[r, w - 1]) == BLACK:
            openings.append((r, w - 1, "right"))
    dedup = []
    seen = set()
    for item in openings:
        if item[:2] not in seen:
            seen.add(item[:2])
            dedup.append(item)
    assert dedup, "template must expose at least one black boundary opening"
    return dedup


def _make_template(
    height: int,
    width: int,
    border_color: int,
    accent_color: int,
    pattern: str,
    rng: random.Random,
) -> dict:
    sprite = np.full((height, width), int(border_color), dtype=int)
    sprite[1:-1, 1:-1] = int(accent_color)

    if pattern == "h_tunnel":
        row = height // 2
        sprite[row, :] = BLACK
        sprite[1, 1] = BLACK
        sprite[1, width - 2] = CATALOG_BG
    elif pattern == "v_tunnel":
        col = width // 2
        sprite[:, col] = BLACK
        sprite[1, 1] = BLACK
        sprite[height - 2, 1] = CATALOG_BG
    elif pattern == "elbow":
        row = height // 2
        col = width // 2
        sprite[row, : col + 1] = BLACK
        sprite[row:, col] = BLACK
        sprite[1, 1] = BLACK
        sprite[1, width - 2] = CATALOG_BG
    else:
        raise ValueError(f"unknown pattern {pattern}")

    # Small random accent tweak for variety without changing the connectivity role.
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if int(sprite[r, c]) == border_color and rng.random() < 0.15
    ]
    for r, c in candidates[:1]:
        sprite[r, c] = int(accent_color)

    visible = sprite[(sprite != BLACK) & (sprite != CATALOG_BG)]
    counts = Counter(map(int, visible.flat))
    assert counts and counts.most_common(1)[0][0] == border_color
    assert np.any(sprite == CATALOG_BG), "need transparent maroon cells to exercise stamping"
    return {
        "sprite": sprite,
        "border": int(border_color),
        "size": (int(height), int(width)),
        "pattern": pattern,
        "openings": _boundary_openings(sprite),
    }


def _build_template_set(rng: random.Random) -> tuple[list[dict], dict]:
    colors = list(PALETTE)
    rng.shuffle(colors)
    border_a, border_b = colors[:2]
    accent_a, accent_b, accent_c = colors[2:5]

    base_size = rng.choice([(4, 4), (4, 5), (5, 4), (5, 5)])
    alt_sizes = [size for size in ((4, 4), (4, 5), (5, 4), (5, 5)) if size != base_size]
    alt_size = rng.choice(alt_sizes)

    t0 = _make_template(base_size[0], base_size[1], border_a, accent_a, "h_tunnel", rng)
    t1 = _make_template(base_size[0], base_size[1], border_b, accent_b, "v_tunnel", rng)
    t2 = _make_template(alt_size[0], alt_size[1], border_a, accent_c, "elbow", rng)
    trio = [t0, t1, t2]

    template_count = rng.choice([2, 3])
    if template_count == 3:
        chosen = trio
    else:
        chosen = [t0, t1] if rng.random() < 0.5 else [t0, t2]

    meta = {
        "template_count": template_count,
        "has_size_ambiguity": any(
            a["size"] == b["size"] and a["border"] != b["border"]
            for i, a in enumerate(chosen)
            for b in chosen[i + 1 :]
        ),
        "has_color_ambiguity": any(
            a["border"] == b["border"] and a["size"] != b["size"]
            for i, a in enumerate(chosen)
            for b in chosen[i + 1 :]
        ),
    }
    return chosen, meta


def _random_slots(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted([0, total] + [rng.randint(0, total) for _ in range(parts - 1)])
    return [cuts[i + 1] - cuts[i] for i in range(parts)]


def _catalog_positions(inner_width: int, cat_h: int, templates: list[dict], rng: random.Random) -> list[tuple[int, int]]:
    order = templates[:]
    rng.shuffle(order)
    widths = [tpl["size"][1] for tpl in order]
    total_w = sum(widths)
    min_gap = len(order) + 1
    slack = inner_width - total_w - min_gap
    assert slack >= 0
    extras = _random_slots(slack, len(order) + 1, rng)
    gaps = [1 + extra for extra in extras]
    cols = []
    x = gaps[0]
    for tpl, gap in zip(order, gaps[1:]):
        rowslack = cat_h - tpl["size"][0]
        y = rng.randint(0, rowslack)
        cols.append((y, x))
        x += tpl["size"][1] + gap

    mapping = {}
    for tpl, pos in zip(order, cols):
        mapping[id(tpl)] = pos
    return [mapping[id(tpl)] for tpl in templates]


def _choose_assignments(templates: list[dict], chamber_count: int, rng: random.Random) -> list[int]:
    idxs = list(range(len(templates)))
    if chamber_count <= len(templates):
        rng.shuffle(idxs)
        return idxs[:chamber_count]
    pick = idxs[:]
    pick.append(rng.choice(idxs))
    rng.shuffle(pick)
    return pick


def _marker_position(
    chamber_h: int,
    chamber_w: int,
    rect_r: int,
    rect_c: int,
    template: dict,
    rng: random.Random,
) -> tuple[int, int] | None:
    th, tw = template["size"]
    candidates = []
    for orow, ocol, side in template["openings"]:
        gr = rect_r + orow
        gc = rect_c + ocol
        if side == "left":
            mr, mc = gr, gc - 2
        elif side == "right":
            mr, mc = gr, gc + 2
        elif side == "top":
            mr, mc = gr - 2, gc
        else:
            mr, mc = gr + 2, gc
        if not (0 <= mr < chamber_h and 0 <= mc < chamber_w):
            continue
        if rect_r - 1 <= mr <= rect_r + th and rect_c - 1 <= mc <= rect_c + tw:
            continue
        candidates.append((mr, mc))
    if not candidates:
        return None
    return rng.choice(candidates)


def _place_chamber_object(chamber_h: int, chamber_w: int, template: dict, rng: random.Random) -> tuple[int, int, int, int] | None:
    th, tw = template["size"]
    placements = []
    for rect_r in range(1, chamber_h - th):
        for rect_c in range(1, chamber_w - tw):
            marker = _marker_position(chamber_h, chamber_w, rect_r, rect_c, template, rng)
            if marker is not None:
                placements.append((rect_r, rect_c, marker[0], marker[1]))
    if not placements:
        return None
    return rng.choice(placements)


def _flood_only_input(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for sr, sc in np.argwhere(out == MARKER):
        sr = int(sr)
        sc = int(sc)
        queue = deque([(sr, sc)])
        seen = {(sr, sc)}
        while queue:
            r, c = queue.popleft()
            for nr, nc in neighbors(r, c, 4):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in seen or int(out[nr, nc]) != BLACK:
                    continue
                seen.add((nr, nc))
                out[nr, nc] = MARKER
                queue.append((nr, nc))
    return out


def _render_wrong_output(
    grid: np.ndarray,
    chamber_rects: list[tuple[tuple[int, int, int, int], dict]],
    wrong_index: int,
    wrong_template: dict,
) -> np.ndarray:
    stamped = grid.copy()
    chamber_info = []
    for idx, (rect_box, template) in enumerate(chamber_rects):
        r0, c0, _r1, _c1 = rect_box
        sprite = wrong_template["sprite"] if idx == wrong_index else template["sprite"]
        for rr in range(sprite.shape[0]):
            for cc in range(sprite.shape[1]):
                if int(sprite[rr, cc]) != CATALOG_BG:
                    stamped[r0 + rr, c0 + cc] = int(sprite[rr, cc])
        marker = np.argwhere((grid == MARKER) & _single_chamber_mask(grid, rect_box))
        assert len(marker) == 1
        chamber_info.append({"marker": tuple(map(int, marker[0]))})
    return _flood_markers(stamped, chamber_info)


def _single_chamber_mask(grid: np.ndarray, rect_box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect_box
    mask = np.zeros_like(grid, dtype=bool)
    for comp in _chambers(grid):
        if comp[r0, c0] or comp[r1, c1]:
            return comp
    raise AssertionError("rectangle bbox not found in any chamber")


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    templates: list[dict],
    chamber_rects: list[tuple[tuple[int, int, int, int], dict]],
    meta: dict,
) -> bool:
    baseline = _flood_only_input(inp)
    if np.array_equal(baseline, out):
        return False

    for rect_box, _tpl in chamber_rects:
        r0, c0, r1, c1 = rect_box
        if np.array_equal(baseline[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
            return False

    if meta["has_size_ambiguity"]:
        found_difference = False
        for idx, (rect_box, used_tpl) in enumerate(chamber_rects):
            same_size = [
                tpl for tpl in templates if tpl["size"] == used_tpl["size"] and tpl["border"] != used_tpl["border"]
            ]
            for wrong_tpl in same_size:
                wrong_out = _render_wrong_output(inp, chamber_rects, idx, wrong_tpl)
                if not np.array_equal(wrong_out, out):
                    found_difference = True
                    break
            if found_difference:
                break
        if not found_difference:
            return False

    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    templates, meta = _build_template_set(rng)
    chamber_count = rng.choice([2, 3])
    assignment = _choose_assignments(templates, chamber_count, rng)
    assigned = [templates[idx] for idx in assignment]

    chamber_widths = [tpl["size"][1] + 3 for tpl in assigned]
    chamber_h = max(tpl["size"][0] for tpl in assigned) + 4

    bottom_inner_w = sum(chamber_widths) + (chamber_count - 1)
    cat_h = max(6, min(8, max(tpl["size"][0] for tpl in templates) + 2))
    catalog_need = sum(tpl["size"][1] for tpl in templates) + (len(templates) + 1)
    inner_w = max(bottom_inner_w, catalog_need)
    if inner_w > 28:
        return None
    grid_h = cat_h + chamber_h + 3
    grid_w = inner_w + 2
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    sep_r = cat_h + 1
    grid[sep_r, :] = WALL
    grid[1 : sep_r, 1:-1] = CATALOG_BG

    template_positions = _catalog_positions(inner_w, cat_h, templates, rng)
    for tpl, (tr, tc) in zip(templates, template_positions):
        th, tw = tpl["size"]
        grid[1 + tr : 1 + tr + th, 1 + tc : 1 + tc + tw] = tpl["sprite"]

    chamber_rects: list[tuple[tuple[int, int, int, int], dict]] = []
    x = 1
    for idx, tpl in enumerate(assigned):
        cw = chamber_widths[idx]
        chamber_top = sep_r + 1
        chamber_left = x
        chamber_right = x + cw - 1
        if idx > 0:
            grid[chamber_top:, chamber_left - 1] = WALL

        placed = _place_chamber_object(chamber_h, cw, tpl, rng)
        if placed is None:
            return None
        rect_r, rect_c, marker_r, marker_c = placed
        th, tw = tpl["size"]

        abs_r = chamber_top + rect_r
        abs_c = chamber_left + rect_c
        grid[abs_r : abs_r + th, abs_c : abs_c + tw] = int(tpl["border"])
        grid[chamber_top + marker_r, chamber_left + marker_c] = MARKER

        chamber_rects.append(((abs_r, abs_c, abs_r + th - 1, abs_c + tw - 1), tpl))
        x += cw + 1

    out = main(grid)
    if not _validate_semantics(grid, out, templates, chamber_rects, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(RETRY_LIMIT):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
