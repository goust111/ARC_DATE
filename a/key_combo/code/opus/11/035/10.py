import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREY = 5
MAGENTA = 6
PATTERN_COLORS = (BLUE, GREY)
RECT_W = 3
GEN_ATTEMPTS = 400


def _bbox(coords):
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected(coords):
    pts = {tuple(map(int, p)) for p in coords}
    if not pts:
        return False
    todo = deque([next(iter(pts))])
    seen = {todo[0]}
    while todo:
        r, c = todo.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in pts and nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return len(seen) == len(pts)


def _components_of_color(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            todo = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while todo:
                rr, cc = todo.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        todo.append((nr, nc))
            comps.append(coords)
    return comps


def _tile_color(tile, r, c):
    return int(tile[int(r) % tile.shape[0], int(c) % tile.shape[1]])


def _min_period_area(tile):
    h, w = tile.shape
    best = h * w
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best = min(best, ph * pw)
    return best


def _infer_min_tile(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            tile = np.full((ph, pw), -1, dtype=int)
            ok = True
            for r in range(h):
                for c in range(w):
                    v = int(g[r, c])
                    if v not in PATTERN_COLORS:
                        continue
                    rr, cc = r % ph, c % pw
                    if tile[rr, cc] == -1:
                        tile[rr, cc] = v
                    elif int(tile[rr, cc]) != v:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or np.any(tile == -1):
                continue
            if set(map(int, tile.ravel())) != set(PATTERN_COLORS):
                continue
            candidates.append((ph * pw, ph, pw, tile))
    assert candidates, "no consistent periodic tile"
    min_area = min(area for area, _ph, _pw, _tile in candidates)
    minimal = [(ph, pw, tile) for area, ph, pw, tile in candidates if area == min_area]
    assert len(minimal) == 1, "minimum-area tile must be unique"
    return minimal[0]


def _make_output_background(h, w, tile):
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = _tile_color(tile, r, c)
    return out


def _catalog_tiles():
    out = {(2, 2): [], (3, 2): []}
    for shape in ((2, 2), (3, 2)):
        h, w = shape
        for bits in range(1 << (h * w)):
            tile = np.full((h, w), GREY, dtype=int)
            for idx in range(h * w):
                if (bits >> idx) & 1:
                    tile[idx // w, idx % w] = BLUE
            if set(map(int, tile.ravel())) != set(PATTERN_COLORS):
                continue
            if _min_period_area(tile) != h * w:
                continue
            out[shape].append(tile)
    return out


def _catalog_fragments():
    out = {3: [], 4: []}
    for rect_h in (3, 4):
        full = {(r, c) for r in range(rect_h) for c in range(RECT_W)}
        for bits in range(1, (1 << (rect_h * RECT_W)) - 1):
            frag = {(idx // RECT_W, idx % RECT_W) for idx in range(rect_h * RECT_W) if (bits >> idx) & 1}
            anchor = full - frag
            if not _connected(frag) or not _connected(anchor):
                continue
            fr0, fc0, fr1, fc1 = _bbox(frag)
            ar0, ac0, _ar1, ac1 = _bbox(anchor)
            if fr0 == 0 or fr1 != rect_h - 1:
                continue
            if fc0 != 0 or fc1 != RECT_W - 1:
                continue
            if ac0 != 0 or ac1 != RECT_W - 1:
                continue
            top_cols = [c for r, c in frag if r == fr0]
            out[rect_h].append(
                {
                    "rect_h": rect_h,
                    "frag_cells": sorted(frag),
                    "anchor_cells": sorted(anchor),
                    "frag_top_offset": fr0,
                    "frag_h": fr1 - fr0 + 1,
                    "top_cols": sorted(top_cols),
                    "frag_area": len(frag),
                }
            )
    return out


TILE_CATALOG = _catalog_tiles()
FRAGMENT_CATALOG = _catalog_fragments()


def _candidate_target_rows(anchor_coords, fragment_coords, grid_shape):
    h, _w = grid_shape
    frag_r0, frag_c0, frag_r1, _frag_c1 = _bbox(fragment_coords)
    norm = [(r - frag_r0, c) for r, c in fragment_coords]
    anchor_set = set(anchor_coords)
    cand = []
    for top in range(0, frag_r0 + 1):
        moved = {(top + rr, c) for rr, c in norm}
        if any(r < 0 or r >= h for r, _c in moved):
            continue
        if moved & anchor_set:
            continue
        union = anchor_set | moved
        r0, c0, r1, c1 = _bbox(union)
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        if rect_w != RECT_W or rect_h not in (3, 4):
            continue
        if len(union) != rect_h * rect_w:
            continue
        if union != {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}:
            continue
        cand.append(top)
    return cand


def _pair_red_objects(grid):
    comps = _components_of_color(grid, RED)
    assert len(comps) == 4, "expected exactly four red components"
    h, _w = grid.shape
    fragments = []
    anchors = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        item = {"coords": comp, "bbox": (r0, c0, r1, c1)}
        if r1 == h - 1:
            fragments.append(item)
        else:
            anchors.append(item)
    assert len(fragments) == 2 and len(anchors) == 2, "expected two bottom fragments and two anchors"

    pairs = []
    used = set()
    for frag in sorted(fragments, key=lambda x: x["bbox"][1]):
        fr0, fc0, fr1, fc1 = frag["bbox"]
        candidates = []
        for idx, anchor in enumerate(anchors):
            if idx in used:
                continue
            ar0, ac0, ar1, ac1 = anchor["bbox"]
            overlap = min(fc1, ac1) - max(fc0, ac0) + 1
            if overlap <= 0 or ar1 >= fr0:
                continue
            candidates.append((overlap, -abs((fc0 + fc1) - (ac0 + ac1)), idx, anchor))
        assert len(candidates) == 1, "fragment-anchor pairing must be unique"
        _overlap, _center_score, idx, anchor = candidates[0]
        used.add(idx)
        pairs.append((frag, anchor))
    return pairs


def _solve_with_details(grid):
    g = np.asarray(grid, dtype=int)
    ph, pw, tile = _infer_min_tile(g)
    pairs = _pair_red_objects(g)
    magenta_positions = {(int(r), int(c)) for r, c in np.argwhere(g == MAGENTA)}

    moves = []
    for frag, anchor in pairs:
        fr0, fc0, _fr1, fc1 = frag["bbox"]
        ar0, _ac0, ar1, _ac1 = anchor["bbox"]
        candidate_rows = _candidate_target_rows(anchor["coords"], frag["coords"], g.shape)
        magenta_rows = {
            r
            for r, c in magenta_positions
            if ar0 <= r < fr0 and fc0 <= c <= fc1
        }
        chosen = sorted(set(candidate_rows) & magenta_rows)
        assert len(chosen) == 1, "anchor marker row must uniquely pick one vertical slide"
        target_top = chosen[0]
        moves.append((frag, anchor, target_top))

    out = g.copy()
    for r, c in np.argwhere(out == MAGENTA):
        out[int(r), int(c)] = _tile_color(tile, int(r), int(c))

    for frag, _anchor, target_top in moves:
        frag_r0, _frag_c0, _frag_r1, _frag_c1 = frag["bbox"]
        norm = [(r - frag_r0, c) for r, c in frag["coords"]]
        for r, c in frag["coords"]:
            out[int(r), int(c)] = _tile_color(tile, int(r), int(c))
        for rr, c in norm:
            out[int(target_top + rr), int(c)] = RED

    return out, {"period": (ph, pw), "tile": tile, "moves": moves}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the periodic background, use the surviving magenta corridor markers to locate the red fragment anchors, and slide each bottom fragment upward to complete a solid rectangle.

    Input structure:
    - The grid is mostly a blue/grey two-color periodic field, partially occluded by magenta pixels.
    - There are exactly four red 4-connected objects: two bottom-touching fragments and two non-bottom anchors.
    - Each fragment shares a vertical column band with exactly one anchor, and magenta pixels appear in the gap between them.

    Rule:
    - Ignore red and magenta cells and infer the unique minimum-area repeating blue/grey tile.
    - Restore every periodic magenta corruption cell to the tile color at that coordinate.
    - For each red anchor/fragment pair, use the remaining magenta row in their shared vertical gap as the target top row for the fragment.
    - Erase the fragment from the bottom edge back to the periodic tile and redraw it at the marked row so that it complements the anchor into a solid 3x3 or 4x3 red rectangle.

    Key insight:
    - Magenta serves two roles at once: most magenta cells are ordinary periodic occlusions, but the magenta row that survives inside each red corridor is the slide anchor that identifies where the fragment must land.
    """
    out, _details = _solve_with_details(grid)
    return out


def _pick_non_overlapping_starts(width):
    starts = list(range(1, width - RECT_W))
    random.shuffle(starts)
    chosen = []
    for start in starts:
        if all(abs(start - prev) >= RECT_W + 2 for prev in chosen):
            chosen.append(start)
        if len(chosen) == 2:
            break
    return sorted(chosen) if len(chosen) == 2 else None


def _place_pair(base, start_col, spec):
    h, _w = base.shape
    rect_h = spec["rect_h"]
    rect_top = spec["rect_top"]
    frag_cells = spec["mask"]["frag_cells"]
    anchor_cells = spec["mask"]["anchor_cells"]
    frag_top_offset = spec["mask"]["frag_top_offset"]
    frag_h = spec["mask"]["frag_h"]
    current_top = h - frag_h
    target_top = rect_top + frag_top_offset

    anchor = [(rect_top + r, start_col + c) for r, c in anchor_cells]
    frag_input = [(current_top + (r - frag_top_offset), start_col + c) for r, c in frag_cells]
    rect = [(rect_top + r, start_col + c) for r in range(rect_h) for c in range(RECT_W)]

    marker_cols = spec["marker_cols"]
    markers = [(target_top, start_col + c) for c in marker_cols]
    distractors = []
    band_rows = [r for r in range(rect_top, current_top) if r != target_top]
    random.shuffle(band_rows)
    for row in band_rows:
        cells = [(row, c) for c in range(start_col, start_col + RECT_W)]
        cells = [p for p in cells if p not in anchor and p not in frag_input and p not in markers]
        if cells:
            distractors.append(random.choice(cells))
        if len(distractors) >= spec["n_distractors"]:
            break
    if len(distractors) < spec["n_distractors"]:
        return None

    info = {
        "anchor": anchor,
        "fragment_input": frag_input,
        "rect": rect,
        "markers": markers,
        "distractors": distractors,
        "target_top": target_top,
        "start_col": start_col,
    }
    return info


def _reserve_visible_cells(grid, period):
    ph, pw = period
    keep = set()
    for rr in range(ph):
        for cc in range(pw):
            options = [(r, c) for r in range(rr, grid.shape[0], ph) for c in range(cc, grid.shape[1], pw) if int(grid[r, c]) in PATTERN_COLORS]
            if not options:
                return None
            keep.add(random.choice(options))
    return keep


def _gen_attempt():
    tile_shape = random.choice([(2, 2), (3, 2)])
    tile = np.array(random.choice(TILE_CATALOG[tile_shape]), dtype=int)
    h = random.randint(17, 22)
    w = random.randint(16, 22)
    starts = _pick_non_overlapping_starts(w)
    if starts is None:
        return None

    base = _make_output_background(h, w, tile)
    output = base.copy()
    input_grid = base.copy()

    pair_infos = []
    for idx, start in enumerate(starts):
        rect_h = random.choice([3, 4])
        mask = random.choice(FRAGMENT_CATALOG[rect_h])
        max_rect_top = h - mask["frag_h"] - 6
        if max_rect_top < 2:
            return None
        rect_top = random.randint(2, max_rect_top)
        marker_pool = mask["top_cols"]
        if not marker_pool:
            return None
        marker_count = min(random.choice([1, 2]), len(marker_pool))
        spec = {
            "rect_h": rect_h,
            "rect_top": rect_top,
            "mask": mask,
            "marker_cols": random.sample(marker_pool, marker_count),
            "n_distractors": random.choice([1, 2]),
        }
        pair = _place_pair(base, start, spec)
        if pair is None:
            return None
        pair_infos.append(pair)
        for r, c in pair["rect"]:
            output[r, c] = RED
        for r, c in pair["anchor"]:
            input_grid[r, c] = RED
        for r, c in pair["fragment_input"]:
            input_grid[r, c] = RED
        for r, c in pair["markers"] + pair["distractors"]:
            input_grid[r, c] = MAGENTA

    protected = _reserve_visible_cells(input_grid, tile.shape)
    if protected is None:
        return None

    eligible = [(r, c) for r in range(h) for c in range(w) if int(input_grid[r, c]) in PATTERN_COLORS and (r, c) not in protected]
    random.shuffle(eligible)
    target_missing = random.randint((h * w) // 9, (h * w) // 6)
    missing = []
    color_hits = {BLUE: 0, GREY: 0}
    for r, c in eligible:
        if len(missing) >= target_missing:
            break
        missing.append((r, c))
        color_hits[int(input_grid[r, c])] += 1
    if not all(color_hits.values()):
        return None
    for r, c in missing:
        input_grid[r, c] = MAGENTA

    solved, details = _solve_with_details(input_grid)
    if not np.array_equal(solved, output):
        return None
    if tuple(details["period"]) != tile.shape:
        return None

    for pair in pair_infos:
        mag_rows = {
            r
            for r, c in np.argwhere(input_grid == MAGENTA)
            if pair["start_col"] <= c < pair["start_col"] + RECT_W and pair["rect"][0][0] <= r < _bbox(pair["fragment_input"])[0]
        }
        if pair["target_top"] not in mag_rows or len(mag_rows) < 2:
            return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
