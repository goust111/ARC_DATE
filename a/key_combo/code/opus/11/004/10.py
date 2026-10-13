import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PAYLOAD = [3, 4, 5, 6, 7, 8, 9]
MASK_SIZE = 4
MAX_ATTEMPTS = 500


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return sprite.copy()
    return np.repeat(np.repeat(sprite, factor, axis=0), factor, axis=1)


def rot180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2)


def component_coords(mask: np.ndarray, connectivity: int = 8):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def is_connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    count = int(mask.sum())
    if count == 0:
        return False
    return len(component_coords(mask, connectivity=connectivity)) == 1


def paste_nonblack(dst: np.ndarray, src: np.ndarray, top: int, left: int):
    h, w = src.shape
    view = dst[top : top + h, left : left + w]
    mask = src != BLACK
    view[mask] = src[mask]


def _find_pattern_component(grid: np.ndarray):
    mask = (grid != BLACK) & (grid != RED)
    blue_cells = np.argwhere(grid == BLUE)
    assert blue_cells.size > 0, "expected one blue mask"
    br, bc = map(int, blue_cells[0])
    for comp in component_coords(mask, connectivity=4):
        comp_set = set(comp)
        if (br, bc) in comp_set:
            return comp
    raise AssertionError("failed to find pattern component")


def _recover_template(grid: np.ndarray):
    blue_box = bbox(grid == BLUE)
    assert blue_box is not None
    r0, c0, r1, c1 = blue_box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (MASK_SIZE, MASK_SIZE)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE)

    comp = _find_pattern_component(grid)
    comp_mask = np.zeros_like(grid, dtype=bool)
    for rr, cc in comp:
        comp_mask[rr, cc] = True
    pattern_box = bbox(comp_mask)
    assert pattern_box is not None
    pr0, pc0, pr1, pc1 = pattern_box
    pattern = crop_bbox(grid, pattern_box)
    local_box = (r0 - pr0, c0 - pc0, r1 - pr0, c1 - pc0)
    template = crop_bbox(rot180(pattern), local_box)
    return pattern_box, blue_box, template


def _match_positions(mask: np.ndarray):
    return tuple(map(tuple, np.argwhere(mask)))


def _enumerate_matches(grid: np.ndarray, object_coords, template: np.ndarray):
    h, w = grid.shape
    obj_mask = np.zeros_like(grid, dtype=bool)
    for r, c in object_coords:
        obj_mask[r, c] = True
    red_mask = obj_mask & (grid == RED)
    red_box = bbox(red_mask)
    assert red_box is not None, "every incomplete object needs red indicators"
    rr0, rc0, rr1, rc1 = red_box
    red_shape = (rr1 - rr0 + 1, rc1 - rc0 + 1)
    red_rel = _match_positions(crop_bbox(red_mask, red_box))

    fixed = {}
    for r, c in object_coords:
        color = int(grid[r, c])
        if color not in (BLACK, RED):
            fixed[(r, c)] = color

    matches = []
    for scale in (1, 2):
        scaled = scale_sprite(template, scale)
        support = scaled != BLACK
        sh, sw = scaled.shape
        rh, rw = red_shape
        if rh > sh or rw > sw:
            continue
        for sr in range(sh - rh + 1):
            for sc in range(sw - rw + 1):
                sub = support[sr : sr + rh, sc : sc + rw]
                if not sub.any():
                    continue
                if _match_positions(sub) != red_rel:
                    continue
                top = rr0 - sr
                left = rc0 - sc
                if top < 0 or left < 0 or top + sh > h or left + sw > w:
                    continue
                ok = True
                for (r, c), color in fixed.items():
                    lr, lc = r - top, c - left
                    if not (0 <= lr < sh and 0 <= lc < sw and int(scaled[lr, lc]) == color):
                        ok = False
                        break
                if ok:
                    matches.append(
                        {
                            "scale": scale,
                            "top": top,
                            "left": left,
                            "sub_top": sr,
                            "sub_left": sc,
                            "scaled": scaled,
                        }
                    )
    return matches


def _fill_object(output: np.ndarray, scaled: np.ndarray, top: int, left: int):
    view = output[top : top + scaled.shape[0], left : left + scaled.shape[1]]
    overwrite = (scaled != BLACK) & ((view == BLACK) | (view == RED))
    view[overwrite] = scaled[overwrite]


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden 4x4 template, then use red anchor subpatterns to place scaled copies that complete the partial objects.

    Input structure:
    - A black background contains one large multicolor region with a solid blue 4x4 occluder.
    - The large region is 180-degree rotationally symmetric once the hidden patch is restored.
    - Outside that region are 2-3 partial objects made of red indicator pixels plus a few correctly colored template pixels.

    Rule:
    - Find the blue 4x4 block inside the large region and rotate that whole region by 180 degrees.
    - Crop the rotated region at the blue block's local coordinates to recover the hidden 4x4 template, then replace the blue block with it.
    - For every outside partial object, match the red pixel topology against a subregion of the template scaled by 1 or 2, use the unique matching offset to place the full scaled template, and overwrite only black/red cells.

    Key insight:
    - The occluded patch is both the answer to the symmetry subproblem and the template needed for the separate completion subproblem; the red pixels encode alignment and scale, not final color.
    """
    g = np.asarray(grid, dtype=int)
    pattern_box, blue_box, template = _recover_template(g)

    out = g.copy()
    r0, c0, r1, c1 = blue_box
    out[r0 : r1 + 1, c0 : c1 + 1] = template

    pr0, pc0, pr1, pc1 = pattern_box
    work = out.copy()
    work[pr0 : pr1 + 1, pc0 : pc1 + 1] = BLACK
    objects = component_coords(work != BLACK, connectivity=8)

    for obj in objects:
        matches = _enumerate_matches(g, obj, template)
        assert len(matches) == 1, "indicator pattern must identify a unique scale and placement"
        match = matches[0]
        _fill_object(out, match["scaled"], match["top"], match["left"])

    return out


def _make_template():
    for _ in range(400):
        mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
        count = random.randint(9, 13)
        chosen = random.sample(range(MASK_SIZE * MASK_SIZE), count)
        for idx in chosen:
            mask[idx // MASK_SIZE, idx % MASK_SIZE] = True
        if mask.all() or not is_connected(mask, connectivity=4):
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        colors = random.sample(PAYLOAD, random.randint(3, 4))
        tpl = np.zeros((MASK_SIZE, MASK_SIZE), dtype=int)
        cells = list(zip(*np.nonzero(mask)))
        assignments = [colors[i % len(colors)] for i in range(len(cells))]
        random.shuffle(assignments)
        for (r, c), color in zip(cells, assignments):
            tpl[r, c] = color
        if len({int(v) for v in tpl.flat if v != BLACK}) < 3:
            continue
        return tpl
    return None


def _non_overlapping_mask_positions(size: int):
    positions = []
    for r in range(size - MASK_SIZE + 1):
        for c in range(size - MASK_SIZE + 1):
            rr = size - MASK_SIZE - r
            cc = size - MASK_SIZE - c
            row_overlap = max(r, rr) <= min(r + MASK_SIZE - 1, rr + MASK_SIZE - 1)
            col_overlap = max(c, cc) <= min(c + MASK_SIZE - 1, cc + MASK_SIZE - 1)
            if not row_overlap or not col_overlap:
                positions.append((r, c))
    return positions


def _make_pattern_region(template: np.ndarray):
    for _ in range(200):
        h = random.randint(10, 12)
        w = random.randint(10, 12)
        candidates = []
        for r in range(h - MASK_SIZE + 1):
            for c in range(w - MASK_SIZE + 1):
                rr = h - MASK_SIZE - r
                cc = w - MASK_SIZE - c
                row_overlap = max(r, rr) <= min(r + MASK_SIZE - 1, rr + MASK_SIZE - 1)
                col_overlap = max(c, cc) <= min(c + MASK_SIZE - 1, cc + MASK_SIZE - 1)
                if not row_overlap or not col_overlap:
                    candidates.append((r, c))
        if not candidates:
            continue
        br, bc = random.choice(candidates)
        orow, ocol = h - MASK_SIZE - br, w - MASK_SIZE - bc
        palette = random.sample(PAYLOAD, random.randint(3, 5))
        region = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                rr, cc = h - 1 - r, w - 1 - c
                if (r > rr) or (r == rr and c > cc):
                    continue
                color = random.choice(palette)
                region[r, c] = color
                region[rr, cc] = color
        region[br : br + MASK_SIZE, bc : bc + MASK_SIZE] = template
        region[orow : orow + MASK_SIZE, ocol : ocol + MASK_SIZE] = rot180(template)
        return region, (br, bc)
    return None, None


def _candidate_keys(template: np.ndarray):
    by_key = {}
    for scale in (1, 2):
        scaled = scale_sprite(template, scale)
        support = scaled != BLACK
        sh, sw = scaled.shape
        total = int(support.sum())
        for r0 in range(sh):
            for c0 in range(sw):
                for r1 in range(r0, sh):
                    for c1 in range(c0, sw):
                        sub = support[r0 : r1 + 1, c0 : c1 + 1]
                        count = int(sub.sum())
                        if count < 3 or count >= total:
                            continue
                        if not sub[0].any() or not sub[-1].any() or not sub[:, 0].any() or not sub[:, -1].any():
                            continue
                        if not is_connected(sub, connectivity=8):
                            continue
                        key = (sub.shape, _match_positions(sub))
                        by_key.setdefault(key, []).append((scale, r0, c0, r1, c1))

    unique = []
    for key, metas in by_key.items():
        if len(metas) != 1:
            continue
        scale, r0, c0, r1, c1 = metas[0]
        unique.append(
            {
                "scale": scale,
                "sub_top": r0,
                "sub_left": c0,
                "sub_bottom": r1,
                "sub_right": c1,
                "shape": key[0],
                "coords": key[1],
            }
        )
    return unique


def _support_graph_positions(support: np.ndarray):
    return {(int(r), int(c)) for r, c in np.argwhere(support)}


def _shortest_path(support: np.ndarray, starts, goal_test):
    q = deque()
    parent = {}
    for s in starts:
        q.append(s)
        parent[s] = None
    while q:
        cur = q.popleft()
        if goal_test(cur):
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        r, c = cur
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if (
                0 <= nr < support.shape[0]
                and 0 <= nc < support.shape[1]
                and support[nr, nc]
                and nxt not in parent
            ):
                parent[nxt] = cur
                q.append(nxt)
    return None


def _build_partial_object(template: np.ndarray, candidate):
    scale = candidate["scale"]
    scaled = scale_sprite(template, scale)
    support = scaled != BLACK
    sr0, sc0, sr1, sc1 = (
        candidate["sub_top"],
        candidate["sub_left"],
        candidate["sub_bottom"],
        candidate["sub_right"],
    )
    red_positions = {(sr0 + r, sc0 + c) for r, c in candidate["coords"]}
    outside_rect = {
        (r, c)
        for r, c in _support_graph_positions(support)
        if not (sr0 <= r <= sr1 and sc0 <= c <= sc1)
    }
    if not outside_rect:
        return None
    path = _shortest_path(support, red_positions, lambda pos: pos in outside_rect)
    if path is None:
        return None
    revealed = set(path) - red_positions
    if not revealed:
        return None

    extra_candidates = list(_support_graph_positions(support) - red_positions - revealed)
    random.shuffle(extra_candidates)
    for cell in extra_candidates[: random.randint(0, min(3, len(extra_candidates)))]:
        r, c = cell
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in red_positions | revealed):
            revealed.add(cell)

    if len(red_positions | revealed) >= int(support.sum()):
        return None

    obj = np.zeros_like(scaled)
    for r, c in red_positions:
        obj[r, c] = RED
    for r, c in revealed:
        obj[r, c] = scaled[r, c]
    if not is_connected(obj != BLACK, connectivity=8):
        return None
    return {
        "scale": scale,
        "sub_top": sr0,
        "sub_left": sc0,
        "scaled": scaled,
        "partial": obj,
    }


def _free_positions(occupied: np.ndarray, shape, padding: int = 1):
    h, w = occupied.shape
    sh, sw = shape
    spots = []
    for r in range(h - sh + 1):
        for c in range(w - sw + 1):
            r0 = max(0, r - padding)
            c0 = max(0, c - padding)
            r1 = min(h, r + sh + padding)
            c1 = min(w, c + sw + padding)
            if not occupied[r0:r1, c0:c1].any():
                spots.append((r, c))
    return spots


def _place_box(occupied: np.ndarray, top: int, left: int, shape):
    h, w = shape
    occupied[top : top + h, left : left + w] = True


def _expected_output(inp: np.ndarray, region_top_left, clean_region, objects):
    out = inp.copy()
    tr, tc = region_top_left
    rh, rw = clean_region.shape
    out[tr : tr + rh, tc : tc + rw] = clean_region
    for obj in objects:
        _fill_object(out, obj["scaled"], obj["top"], obj["left"])
    return out


def _validate_attempt(inp: np.ndarray, expected: np.ndarray, objects):
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    template_box = bbox(inp == BLUE)
    if template_box is None:
        return False
    for obj in objects:
        coords = [(obj["top"] + r, obj["left"] + c) for r, c in zip(*np.nonzero(obj["partial"] != BLACK))]
        matches = _enumerate_matches(inp, coords, obj["template"])
        if len(matches) != 1:
            return False
        match = matches[0]
        if (
            match["scale"] != obj["scale"]
            or match["top"] != obj["top"]
            or match["left"] != obj["left"]
            or match["sub_top"] != obj["sub_top"]
            or match["sub_left"] != obj["sub_left"]
        ):
            return False
    return True


def _gen_attempt():
    template = _make_template()
    if template is None:
        return None
    unique = _candidate_keys(template)
    by_scale = {1: [], 2: []}
    for cand in unique:
        built = _build_partial_object(template, cand)
        if built is not None:
            by_scale[cand["scale"]].append(built)
    if not by_scale[1] or not by_scale[2]:
        return None

    clean_region, local_mask = _make_pattern_region(template)
    if clean_region is None:
        return None
    region_h, region_w = clean_region.shape

    grid_h = random.randint(22, 28)
    grid_w = random.randint(22, 28)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    region_top = random.randint(1, 3)
    region_left = random.randint(1, 3)
    if region_top + region_h > grid_h or region_left + region_w > grid_w:
        return None
    inp_region = clean_region.copy()
    br, bc = local_mask
    inp_region[br : br + MASK_SIZE, bc : bc + MASK_SIZE] = BLUE
    paste_nonblack(grid, inp_region, region_top, region_left)
    _place_box(occupied, region_top, region_left, clean_region.shape)

    n_objects = random.choice([2, 3])
    chosen = [random.choice(by_scale[1]), random.choice(by_scale[2])]
    if n_objects == 3:
        chosen.append(random.choice(by_scale[random.choice([1, 2])]))

    placed_objects = []
    for built in chosen:
        partial = built["partial"]
        scaled = built["scaled"]
        spots = _free_positions(occupied, scaled.shape, padding=1)
        spots = [
            (r, c)
            for r, c in spots
            if r >= region_top + region_h - 1 or c >= region_left + region_w - 1
        ]
        if not spots:
            return None
        top, left = random.choice(spots)
        paste_nonblack(grid, partial, top, left)
        _place_box(occupied, top, left, scaled.shape)
        placed_objects.append(
            {
                "template": template,
                "scale": built["scale"],
                "sub_top": built["sub_top"],
                "sub_left": built["sub_left"],
                "scaled": scaled,
                "partial": partial,
                "top": top,
                "left": left,
            }
        )

    expected = _expected_output(grid, (region_top, region_left), clean_region, placed_objects)

    if not _validate_attempt(grid, expected, placed_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
