import itertools
import random
from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PATH_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_SLOT = 9
MAX_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    steps = DIR4 if int(connectivity) == 4 else [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if not in_bounds(mask.shape, nr, nc):
                        continue
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _is_path_color(v: int) -> bool:
    return int(v) not in (BG, YELLOW, GREY)


def _compress_runs(values: list[int]) -> list[int]:
    palette = []
    for v in values:
        iv = int(v)
        if not palette or palette[-1] != iv:
            palette.append(iv)
    return palette


def _boundary_mask(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    out = np.zeros_like(region, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(region[r, c]):
                continue
            for nr, nc in neighbors4(r, c):
                if not in_bounds(region.shape, nr, nc) or not bool(region[nr, nc]):
                    out[r, c] = True
                    break
    return out


def _cheb_dist(point: tuple[int, int], boundary: np.ndarray) -> int:
    rr = np.abs(boundary[:, 0] - int(point[0]))
    cc = np.abs(boundary[:, 1] - int(point[1]))
    return int(np.maximum(rr, cc).min())


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _trace_path(grid: np.ndarray, marker: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    mr, mc = marker
    starts = []
    for nr, nc in neighbors4(mr, mc):
        if in_bounds((h, w), nr, nc) and _is_path_color(int(grid[nr, nc])):
            starts.append((int(nr), int(nc)))
    assert len(starts) == 1, "marker must touch exactly one path endpoint"

    prev = marker
    curr = starts[0]
    path = []
    seen = set()
    while True:
        assert curr not in seen, "path must be simple"
        seen.add(curr)
        path.append(curr)
        nxt = []
        for nr, nc in neighbors4(*curr):
            if not in_bounds((h, w), nr, nc) or (nr, nc) == prev:
                continue
            if _is_path_color(int(grid[nr, nc])):
                nxt.append((int(nr), int(nc)))
        if not nxt:
            break
        assert len(nxt) == 1, "path must be unbranched"
        prev, curr = curr, nxt[0]
    return path


def _outline_cavity(grid: np.ndarray, outline_coords: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox_coords(outline_coords)
    box = (
        max(0, r0 - 1),
        max(0, c0 - 1),
        min(h - 1, r1 + 1),
        min(w - 1, c1 + 1),
    )
    sub = crop_bbox(grid, box)
    outline_set = {tuple(map(int, xy)) for xy in outline_coords}
    cavities = []
    for comp in components_mask(sub == BG, connectivity=4):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == sub.shape[1] - 1):
            continue
        global_coords = comp + np.array([box[0], box[1]], dtype=int)
        comp_set = {tuple(map(int, xy)) for xy in global_coords}
        boundary = set()
        for r, c in comp_set:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp_set:
                    continue
                if not in_bounds((h, w), nr, nc):
                    boundary.add((nr, nc))
                    continue
                boundary.add((nr, nc))
        if not boundary:
            continue
        if not all(in_bounds((h, w), r, c) and int(grid[r, c]) == YELLOW for r, c in boundary):
            continue
        if not boundary.issubset(outline_set):
            continue
        cavities.append(global_coords)
    assert len(cavities) == 1, "each yellow outline must enclose exactly one black cavity"
    return cavities[0]


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2
    assert max(h, w) <= 30
    assert int(get_bg(g)) == BG

    outline_comps = components_mask(g == YELLOW, connectivity=4)
    assert 2 <= len(outline_comps) <= 3
    outlines = []
    outline_lookup = {}
    for idx, coords in enumerate(outline_comps):
        cavity = _outline_cavity(g, coords)
        boundary = np.array(coords, dtype=int)
        outlines.append(
            {
                "id": idx,
                "coords": boundary,
                "coord_set": {tuple(map(int, xy)) for xy in boundary},
                "cavity": np.array(cavity, dtype=int),
            }
        )
        for r, c in coords:
            outline_lookup[(int(r), int(c))] = idx

    markers = [tuple(map(int, xy)) for xy in np.argwhere(g == GREY)]
    assert 2 <= len(markers) <= 3

    path_cells = {tuple(map(int, xy)) for xy in np.argwhere((g != BG) & (g != YELLOW) & (g != GREY))}
    used_path = set()
    objects = []
    seen_outlines = set()
    for marker in sorted(markers):
        path = _trace_path(g, marker)
        assert path, "each marker must reach a path"
        for cell in path:
            assert cell not in used_path, "paths must be disjoint"
            used_path.add(cell)
        palette = _compress_runs([int(g[r, c]) for r, c in path])
        assert 2 <= len(palette) <= 3, "each path must have 2-3 color segments"
        assert len(set(palette)) == len(palette), "path segment colors must be distinct"

        end = path[-1]
        contacts = {outline_lookup[(nr, nc)] for nr, nc in neighbors4(*end) if (nr, nc) in outline_lookup}
        assert len(contacts) == 1, "path endpoint must touch exactly one yellow outline"
        outline_id = next(iter(contacts))
        assert outline_id not in seen_outlines, "each outline should be paired with one path"
        seen_outlines.add(outline_id)

        objects.append({"marker": marker, "path": path, "palette": palette, "outline_id": outline_id})

    assert used_path == path_cells, "every non-yellow foreground pixel must belong to a traced path"
    assert seen_outlines == {o["id"] for o in outlines}

    return {"outlines": outlines, "objects": objects}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each grey-marked multicolor path, erase it, then fill the contacted yellow cavity with concentric cyclic bands derived from the traversed color order.

    Input structure:
    - The grid has a black background with 2-3 independent yellow closed outlines, each enclosing one black cavity.
    - Each outline is paired with one unbranched 4-connected path whose non-yellow cells are split into 2-3 color runs.
    - A grey marker touches one path endpoint, and the opposite path endpoint touches the paired yellow outline by 4-neighbor contact.

    Rule:
    - For each grey marker, follow its unique 4-connected path cell by cell until the endpoint that touches a yellow outline.
    - Record the path colors in traversal order, compressing each maximal same-color run to one palette entry.
    - Erase the grey marker and every non-yellow path pixel, keep the yellow outline unchanged, then fill the outline's enclosed black cavity.
    - For every cavity cell, compute its Chebyshev distance d to the yellow outline and paint it with palette[d mod len(palette)].

    Key insight:
    - The path is both motion and data: its endpoint selects which cavity to fill, while the ordered color-run sequence along that path becomes the cyclic band palette.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_instance(g)
    out = g.copy()

    for obj in parsed["objects"]:
        marker = obj["marker"]
        path = obj["path"]
        palette = obj["palette"]
        outline = parsed["outlines"][obj["outline_id"]]
        boundary = outline["coords"]
        cavity = outline["cavity"]

        out[marker] = BG
        for r, c in path:
            out[r, c] = BG

        m = len(palette)
        for r, c in cavity:
            d = _cheb_dist((int(r), int(c)), boundary)
            out[int(r), int(c)] = int(palette[d % m])

    return out


def _random_partition(total: int, n: int, minimum: int, rng: random.Random) -> list[int] | None:
    if total < n * minimum:
        return None
    cuts = sorted(rng.sample(range(total - n * minimum + n - 1), n - 1)) if n > 1 else []
    parts = []
    prev = -1
    for cut in cuts + [total - n * minimum + n - 1]:
        parts.append(cut - prev - 1 + minimum)
        prev = cut
    return parts


def _make_slots(rng: random.Random) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    n = rng.choice([2, 2, 3])
    layout = rng.choice(["rows", "cols"])
    gap = 1
    if layout == "rows":
        h_min = n * MIN_SLOT + (n - 1) * gap
        h = rng.randint(h_min, 30)
        w = rng.randint(20, 30)
        heights = _random_partition(h - (n - 1) * gap, n, MIN_SLOT, rng)
        assert heights is not None
        slots = []
        r = 0
        for hh in heights:
            slots.append((r, 0, r + hh - 1, w - 1))
            r += hh + gap
        return (h, w), slots

    w_min = n * MIN_SLOT + (n - 1) * gap
    h = rng.randint(20, 30)
    w = rng.randint(w_min, 30)
    widths = _random_partition(w - (n - 1) * gap, n, MIN_SLOT, rng)
    assert widths is not None
    slots = []
    c = 0
    for ww in widths:
        slots.append((0, c, h - 1, c + ww - 1))
        c += ww + gap
    return (h, w), slots


def _shape_depth(outline: np.ndarray, interior: np.ndarray) -> int:
    if interior.size == 0:
        return 0
    boundary = np.argwhere(outline)
    return max(_cheb_dist((int(r), int(c)), boundary) for r, c in np.argwhere(interior))


def _make_region_shape(max_h: int, max_w: int, rng: random.Random) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None
    for _ in range(200):
        canvas_h = min(12, max_h + 3)
        canvas_w = min(12, max_w + 3)
        region = np.zeros((canvas_h, canvas_w), dtype=bool)

        main_h = rng.randint(5, min(7, max_h))
        main_w = rng.randint(5, min(7, max_w))
        r0 = rng.randint(1, max(1, canvas_h - main_h - 1))
        c0 = rng.randint(1, max(1, canvas_w - main_w - 1))
        region[r0 : r0 + main_h, c0 : c0 + main_w] = True

        bump_attempts = 1 + int(rng.random() < 0.55)
        for _b in range(bump_attempts):
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                if main_w < 4:
                    continue
                bump_w = rng.randint(2, min(4, main_w - 1))
                start_c_low = c0 + 1
                start_c_high = c0 + main_w - bump_w - 1
                if start_c_low > start_c_high:
                    continue
                start_c = rng.randint(start_c_low, start_c_high)
                if side == "top":
                    room = r0
                    if room < 2:
                        continue
                    bump_h = rng.randint(2, min(3, room))
                    region[r0 - bump_h : r0, start_c : start_c + bump_w] = True
                else:
                    room = canvas_h - (r0 + main_h)
                    if room < 2:
                        continue
                    bump_h = rng.randint(2, min(3, room))
                    region[r0 + main_h : r0 + main_h + bump_h, start_c : start_c + bump_w] = True
            else:
                if main_h < 4:
                    continue
                bump_h = rng.randint(2, min(4, main_h - 1))
                start_r_low = r0 + 1
                start_r_high = r0 + main_h - bump_h - 1
                if start_r_low > start_r_high:
                    continue
                start_r = rng.randint(start_r_low, start_r_high)
                if side == "left":
                    room = c0
                    if room < 2:
                        continue
                    bump_w = rng.randint(2, min(3, room))
                    region[start_r : start_r + bump_h, c0 - bump_w : c0] = True
                else:
                    room = canvas_w - (c0 + main_w)
                    if room < 2:
                        continue
                    bump_w = rng.randint(2, min(3, room))
                    region[start_r : start_r + bump_h, c0 + main_w : c0 + main_w + bump_w] = True

        coords = np.argwhere(region)
        r1, c1, r2, c2 = bbox_coords(coords)
        region = region[r1 : r2 + 1, c1 : c2 + 1]
        if region.shape[0] > max_h or region.shape[1] > max_w:
            continue

        outline = _boundary_mask(region)
        interior = region & ~outline
        if int(interior.sum()) < 6:
            continue
        if _shape_depth(outline, interior) < 2 and rng.random() < 0.75:
            continue
        return region
    return None


def _contact_candidates(region: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    outline = _boundary_mask(region)
    h, w = region.shape
    out = []
    for r, c in np.argwhere(outline):
        for dr, dc in DIR4:
            ep_r, ep_c = int(r + dr), int(c + dc)
            if in_bounds((h, w), ep_r, ep_c) and bool(region[ep_r, ep_c]):
                continue
            touches = 0
            for nr, nc in neighbors4(ep_r, ep_c):
                if in_bounds((h, w), nr, nc) and bool(region[nr, nc]):
                    touches += 1
            if touches == 1:
                out.append((int(r), int(c), int(dr), int(dc), int(ep_r), int(ep_c)))
    return out


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    r, c = start
    dr, dc = direction
    cells = []
    while in_bounds((h, w), r, c) and not bool(blocked[r, c]):
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _path_touches_outline_once(path: list[tuple[int, int]], outline_set: set[tuple[int, int]]) -> bool:
    total = 0
    endpoint = path[0]
    for idx, (r, c) in enumerate(path):
        touches = 0
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in outline_set:
                touches += 1
        total += touches
        if idx == 0 and touches != 1:
            return False
        if idx > 0 and touches != 0:
            return False
    return total == 1


def _build_path(
    blocked: np.ndarray,
    endpoint: tuple[int, int],
    outward: tuple[int, int],
    palette_len: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for dir2 in rng.sample(_perpendiculars(outward), k=2):
        avail1 = _ray_cells(endpoint, outward, blocked)
        len1_opts = [k for k in range(2, min(5, len(avail1)) + 1)]
        rng.shuffle(len1_opts)
        for len1 in len1_opts:
            seg1 = avail1[:len1]
            path = list(seg1)
            corner1 = seg1[-1]

            blocked2 = blocked.copy()
            for r, c in path:
                blocked2[r, c] = True

            start2 = (corner1[0] + dir2[0], corner1[1] + dir2[1])
            avail2 = _ray_cells(start2, dir2, blocked2)
            len2_opts = [k for k in range(2, min(4, len(avail2)) + 1)]
            rng.shuffle(len2_opts)
            for len2 in len2_opts:
                seg2 = avail2[:len2]
                path2 = path + seg2
                if palette_len == 2:
                    marker = (path2[-1][0] + dir2[0], path2[-1][1] + dir2[1])
                    if not in_bounds(blocked.shape, *marker) or bool(blocked2[marker]):
                        continue
                    if marker in path2:
                        continue
                    return path2, marker

                blocked3 = blocked.copy()
                for r, c in path2:
                    blocked3[r, c] = True
                start3 = (path2[-1][0] + outward[0], path2[-1][1] + outward[1])
                avail3 = _ray_cells(start3, outward, blocked3)
                len3_opts = [k for k in range(2, min(4, len(avail3)) + 1)]
                rng.shuffle(len3_opts)
                for len3 in len3_opts:
                    seg3 = avail3[:len3]
                    path3 = path2 + seg3
                    marker = (path3[-1][0] + outward[0], path3[-1][1] + outward[1])
                    if not in_bounds(blocked.shape, *marker) or bool(blocked3[marker]):
                        continue
                    if marker in path3:
                        continue
                    return path3, marker
    return None


def _place_object(
    slot_h: int,
    slot_w: int,
    palette: tuple[int, ...],
    rng: random.Random,
    shared_region: np.ndarray | None = None,
) -> tuple[np.ndarray, dict] | None:
    for _ in range(250):
        region = None
        if shared_region is not None and shared_region.shape[0] <= slot_h - 1 and shared_region.shape[1] <= slot_w - 1:
            region = shared_region.copy()
        if region is None:
            region = _make_region_shape(min(8, slot_h - 1), min(8, slot_w - 1), rng)
        if region is None:
            continue

        rh, rw = region.shape
        contacts = _contact_candidates(region)
        if not contacts:
            continue

        if slot_w >= slot_h:
            pref = [c for c in contacts if c[3] != 0]
            contacts = pref or contacts
        else:
            pref = [c for c in contacts if c[2] != 0]
            contacts = pref or contacts
        rng.shuffle(contacts)

        for br, bc, dr, dc, ep_r, ep_c in contacts:
            row_low = max(0, -ep_r)
            row_high = min(slot_h - rh, slot_h - 1 - ep_r)
            col_low = max(0, -ep_c)
            col_high = min(slot_w - rw, slot_w - 1 - ep_c)
            if row_low > row_high or col_low > col_high:
                continue

            row_choices = list(range(row_low, row_high + 1))
            col_choices = list(range(col_low, col_high + 1))
            if dr == -1:
                row_choices = row_choices[len(row_choices) // 2 :] or row_choices
            elif dr == 1:
                row_choices = row_choices[: (len(row_choices) + 1) // 2] or row_choices
            if dc == -1:
                col_choices = col_choices[len(col_choices) // 2 :] or col_choices
            elif dc == 1:
                col_choices = col_choices[: (len(col_choices) + 1) // 2] or col_choices

            for _placement in range(12):
                ro = rng.choice(row_choices)
                co = rng.choice(col_choices)
                outline_local = _boundary_mask(region)
                interior_local = region & ~outline_local
                blocked = np.zeros((slot_h, slot_w), dtype=bool)
                slot_grid = np.zeros((slot_h, slot_w), dtype=int)

                outline_global = []
                outline_set = set()
                for r, c in np.argwhere(outline_local):
                    rr, cc = int(ro + r), int(co + c)
                    blocked[rr, cc] = True
                    slot_grid[rr, cc] = YELLOW
                    outline_global.append((rr, cc))
                    outline_set.add((rr, cc))
                for r, c in np.argwhere(interior_local):
                    blocked[int(ro + r), int(co + c)] = False

                endpoint = (ro + ep_r, co + ep_c)
                built = _build_path(blocked, endpoint, (dr, dc), len(palette), rng)
                if built is None:
                    continue
                cavity_to_marker, marker = built
                marker_to_cavity = list(reversed(cavity_to_marker))

                if not _path_touches_outline_once(cavity_to_marker, outline_set):
                    continue
                if any((nr, nc) in outline_set for nr, nc in neighbors4(*marker)):
                    continue

                ok = True
                for idx, (r, c) in enumerate(marker_to_cavity):
                    if slot_grid[r, c] != BG:
                        ok = False
                        break
                    neigh_path = 0
                    for nr, nc in neighbors4(r, c):
                        if (nr, nc) in marker_to_cavity:
                            neigh_path += 1
                    if idx in (0, len(marker_to_cavity) - 1):
                        if neigh_path != 1:
                            ok = False
                            break
                    elif neigh_path != 2:
                        ok = False
                        break
                if not ok:
                    continue
                if slot_grid[marker] != BG:
                    continue
                if sum((nr, nc) in marker_to_cavity for nr, nc in neighbors4(*marker)) != 1:
                    continue

                cursor = 0
                for color_idx, color in enumerate(palette):
                    remaining = len(marker_to_cavity) - cursor
                    need = len(palette) - color_idx
                    seg_len_min = 2 if remaining >= 2 * need else 1
                    seg_len_max = remaining - (need - 1)
                    seg_len = rng.randint(seg_len_min, seg_len_max)
                    if color_idx == len(palette) - 1:
                        seg_len = remaining
                    for r, c in marker_to_cavity[cursor : cursor + seg_len]:
                        slot_grid[r, c] = int(color)
                    cursor += seg_len

                slot_grid[marker] = GREY
                return slot_grid, {"palette": list(palette), "outline": outline_set}
    return None


def _palette_family(n: int, rng: random.Random) -> tuple[list[tuple[int, ...]], dict]:
    pair_mode = n >= 2 and rng.random() < 0.7
    meta = {"pair_mode": pair_mode}
    palettes = []
    if pair_mode:
        k = 3 if rng.random() < 0.65 else 2
        base = tuple(rng.sample(PATH_COLORS, k))
        perms = [p for p in set(itertools.permutations(base)) if p != base]
        other = rng.choice(perms)
        palettes.extend([base, other])
    while len(palettes) < n:
        k = 3 if rng.random() < 0.55 else 2
        pal = tuple(rng.sample(PATH_COLORS, k))
        palettes.append(pal)
    rng.shuffle(palettes)
    return palettes, meta


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    (h, w), slots = _make_slots(rng)
    n = len(slots)
    palettes, meta = _palette_family(n, rng)
    grid = np.zeros((h, w), dtype=int)

    shared_region = None
    if meta["pair_mode"] and rng.random() < 0.65:
        pair_h = min(slots[0][2] - slots[0][0] + 1, slots[1][2] - slots[1][0] + 1)
        pair_w = min(slots[0][3] - slots[0][1] + 1, slots[1][3] - slots[1][1] + 1)
        shared_region = _make_region_shape(min(8, pair_h - 1), min(8, pair_w - 1), rng)

    object_meta = []
    for idx, slot in enumerate(slots):
        r0, c0, r1, c1 = slot
        slot_h = r1 - r0 + 1
        slot_w = c1 - c0 + 1
        use_shared = shared_region if idx < 2 and shared_region is not None else None
        placed = _place_object(slot_h, slot_w, palettes[idx], rng, use_shared)
        if placed is None:
            return None
        patch, patch_meta = placed
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        object_meta.append(patch_meta)
    meta["objects"] = object_meta
    return grid, meta


def _validate_generated(grid: np.ndarray, meta: dict) -> None:
    parsed = _parse_instance(grid)
    out = main(grid)

    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert len(parsed["objects"]) == len(parsed["outlines"])
    assert 2 <= len(parsed["objects"]) <= 3

    depths = []
    fill_signatures = []
    palettes = []
    for obj in parsed["objects"]:
        outline = parsed["outlines"][obj["outline_id"]]
        boundary = outline["coords"]
        cavity = outline["cavity"]
        depths.append(max(_cheb_dist((int(r), int(c)), boundary) for r, c in cavity))
        fill_signatures.append(tuple(int(out[r, c]) for r, c in cavity))
        palettes.append(tuple(obj["palette"]))

        assert all(int(grid[r, c]) == GREY for r, c in [obj["marker"]])
        assert all(_is_path_color(int(grid[r, c])) for r, c in obj["path"])
        assert all(int(out[r, c]) == BG for r, c in obj["path"])
        assert int(out[obj["marker"]]) == BG
        assert 2 <= len(obj["palette"]) <= 3

    assert any(depth >= 2 for depth in depths), "at least one cavity should show multiple bands"
    assert len(set(fill_signatures)) == len(fill_signatures), "paired objects should not collapse to the same output fill"

    if meta.get("pair_mode") and len(palettes) >= 2:
        matched = False
        for i in range(len(palettes)):
            for j in range(i + 1, len(palettes)):
                if set(palettes[i]) == set(palettes[j]) and palettes[i] != palettes[j]:
                    matched = True
        assert matched, "pair mode must include same-color-set but different-order palettes"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        built = _gen_attempt(rng)
        if built is None:
            continue
        grid, meta = built
        try:
            _validate_generated(grid, meta)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid path-to-cavity band-fill instance")
