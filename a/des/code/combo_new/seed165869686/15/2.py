from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
SKELETON = 5
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _endpoints(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    h, w = mask.shape
    pts: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            deg = 0
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    deg += 1
            if deg <= 1:
                pts.append((int(r), int(c)))
    return tuple(sorted(pts))


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _canonical_rot_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.append(_shape_key(rot))
    return min(keys)


def _matches_with_rotation(fragment_mask: np.ndarray, arc_mask: np.ndarray) -> bool:
    arc_sig = (_shape_key(arc_mask), _endpoints(arc_mask))
    for k in range(4):
        rot = np.rot90(fragment_mask, k)
        sig = (_shape_key(rot), _endpoints(rot))
        if sig == arc_sig and np.array_equal(rot, arc_mask):
            return True
    return False


def _solve_assignment(candidates: dict[int, list[int]]) -> dict[int, int] | None:
    order = sorted(candidates, key=lambda idx: (len(candidates[idx]), idx))
    used: set[int] = set()
    out: dict[int, int] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        frag_idx = int(order[pos])
        for arc_idx in candidates[frag_idx]:
            if arc_idx in used:
                continue
            used.add(int(arc_idx))
            out[frag_idx] = int(arc_idx)
            if dfs(pos + 1):
                return True
            used.remove(int(arc_idx))
            del out[frag_idx]
        return False

    return out if dfs(0) else None


def _component_record(coords: list[tuple[int, int]]) -> dict[str, object]:
    mask, bbox = _mask_from_coords(coords)
    return {
        "coords": tuple(sorted((int(r), int(c)) for r, c in coords)),
        "bbox": bbox,
        "mask": mask,
        "area": int(mask.sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored arc fragments onto the broken gray ring skeleton.

    Input structure:
    - The background is black.
    - Near the center sits a 1-pixel-wide gray ring outline with several single-cell gaps.
    - Away from the ring are several small monochrome fragments, each a 4-connected open arc copied from one gray ring segment.

    Rule:
    - Split the gray outline into its gap-separated 4-connected arc segments.
    - Split the colored cells into monochrome fragment components.
    - For each fragment, restrict candidates to gray arc segments with the same area, then choose the segment whose shape matches under a 0/90/180/270 degree rotation; the fragment's open ends must line up with the arc segment's open ends.
    - Clear the original fragment locations and repaint the matching gray arc cells with the fragment's color, leaving unmatched gray cells unchanged.

    Key insight:
    - The gaps turn the ring into several open paths, so each fragment is identified by an exact rotated path-shape match rather than by color or by absolute position.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)

    arc_components = [_component_record(comp) for comp in _components(g == SKELETON)]
    fragment_components = []
    for comp in _components((g != BG) & (g != SKELETON)):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        rec = _component_record(comp)
        rec["color"] = int(next(iter(colors)))
        fragment_components.append(rec)

    out[g == SKELETON] = SKELETON
    if not arc_components or not fragment_components:
        return out

    arcs_by_area: dict[int, list[int]] = defaultdict(list)
    for arc_idx, arc in enumerate(arc_components):
        arcs_by_area[int(arc["area"])].append(int(arc_idx))

    candidates: dict[int, list[int]] = {}
    for frag_idx, frag in enumerate(fragment_components):
        frag_mask = np.asarray(frag["mask"], dtype=bool)
        options: list[int] = []
        for arc_idx in arcs_by_area.get(int(frag["area"]), []):
            arc_mask = np.asarray(arc_components[arc_idx]["mask"], dtype=bool)
            if _matches_with_rotation(frag_mask, arc_mask):
                options.append(int(arc_idx))
        candidates[int(frag_idx)] = sorted(set(options))

    assignment = _solve_assignment(candidates)
    if assignment is None:
        assignment = {}
        used: set[int] = set()
        for frag_idx in sorted(candidates, key=lambda idx: (len(candidates[idx]), idx)):
            for arc_idx in candidates[frag_idx]:
                if arc_idx not in used:
                    assignment[int(frag_idx)] = int(arc_idx)
                    used.add(int(arc_idx))
                    break

    for frag_idx, arc_idx in assignment.items():
        color = int(fragment_components[frag_idx]["color"])
        for r, c in arc_components[arc_idx]["coords"]:
            out[r, c] = color

    return out


def _build_chain(rows: list[int]) -> list[tuple[int, int]]:
    path = [(int(rows[0]), 0)]
    for x in range(len(rows) - 1):
        cur = int(rows[x])
        nxt = int(rows[x + 1])
        path.append((int(cur), int(x + 1)))
        if nxt > cur:
            for rr in range(cur + 1, nxt + 1):
                path.append((int(rr), int(x + 1)))
        elif nxt < cur:
            for rr in range(cur - 1, nxt - 1, -1):
                path.append((int(rr), int(x + 1)))
    return path


def _vertical_between(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    if r0 == r1:
        return []
    step = 1 if r1 > r0 else -1
    return [(int(r), int(c)) for r in range(r0 + step, r1 + step, step)]


def _cycle_from_rows(top_rows: list[int], bottom_rows: list[int]) -> list[tuple[int, int]]:
    top = _build_chain(top_rows)
    bottom_lr = _build_chain(bottom_rows)
    bottom = list(reversed(bottom_lr))
    right = _vertical_between(int(top_rows[-1]), int(bottom_rows[-1]), len(top_rows) - 1)
    left = _vertical_between(int(bottom_rows[0]), int(top_rows[0]), 0)
    cycle = top + right + bottom[1:] + left[:-1]
    return [(int(r), int(c)) for r, c in cycle]


def _is_simple_cycle(cycle: list[tuple[int, int]]) -> bool:
    pts = set(cycle)
    if len(pts) != len(cycle):
        return False
    for r, c in cycle:
        deg = 0
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in pts:
                deg += 1
        if deg != 2:
            return False
    return True


def _generate_ring_cycle() -> list[tuple[int, int]] | None:
    for _ in range(120):
        width = random.randint(8, 11)
        top = [random.randint(0, 2)]
        for _x in range(1, width):
            prev = int(top[-1])
            options = [v for v in (prev - 1, prev, prev + 1) if 0 <= v <= 2]
            top.append(int(random.choice(options)))

        low_gap = 4
        bottom = [random.randint(top[0] + low_gap, min(9, top[0] + low_gap + 2))]
        ok = True
        for x in range(1, width):
            prev = int(bottom[-1])
            opts = [
                v
                for v in (prev - 1, prev, prev + 1)
                if top[x] + low_gap <= v <= 9
            ]
            if not opts:
                ok = False
                break
            bottom.append(int(random.choice(opts)))
        if not ok:
            continue
        if len(set(top)) == 1 and len(set(bottom)) == 1:
            continue
        cycle = _cycle_from_rows(top, bottom)
        if _is_simple_cycle(cycle):
            return cycle
    return None


def _sample_gap_indices(length: int, gap_count: int, min_arc_len: int) -> list[int] | None:
    choices = list(range(length))
    for _ in range(300):
        gaps = sorted(random.sample(choices, gap_count))
        ok = True
        for i in range(gap_count):
            a = int(gaps[i])
            b = int(gaps[(i + 1) % gap_count])
            span = (b - a - 1) % length
            if span < min_arc_len:
                ok = False
                break
        if ok:
            return gaps
    return None


def _arcs_from_cycle(cycle: list[tuple[int, int]], gaps: list[int]) -> list[list[tuple[int, int]]]:
    length = len(cycle)
    arcs: list[list[tuple[int, int]]] = []
    for i in range(len(gaps)):
        start = int(gaps[i])
        end = int(gaps[(i + 1) % len(gaps)])
        arc: list[tuple[int, int]] = []
        idx = (start + 1) % length
        while idx != end:
            arc.append(cycle[idx])
            idx = (idx + 1) % length
        arcs.append(arc)
    return arcs


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], value: int, offset: tuple[int, int] = (0, 0)) -> None:
    dr, dc = offset
    for r, c in coords:
        grid[int(r + dr), int(c + dc)] = int(value)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _can_place_patch(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = mask.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    pts = np.argwhere(mask)
    for rr, cc in pts:
        gr = int(top + rr)
        gc = int(left + cc)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = int(gr + dr)
                nc = int(gc + dc)
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    pts = np.argwhere(mask)
    for rr, cc in pts:
        gr = int(top + rr)
        gc = int(left + cc)
        grid[gr, gc] = int(color)
        occupied[gr, gc] = True


def _zone_candidates(
    shape: tuple[int, int],
    grid_shape: tuple[int, int],
    skeleton_bbox: tuple[int, int, int, int],
    zone: str,
) -> list[tuple[int, int]]:
    ph, pw = shape
    h, w = grid_shape
    r0, c0, r1, c1 = skeleton_bbox
    rows: range
    cols: range
    if zone == "top":
        rows = range(1, max(1, r0 - ph - 1))
        cols = range(1, w - pw)
    elif zone == "bottom":
        rows = range(r1 + 3, h - ph)
        cols = range(1, w - pw)
    elif zone == "left":
        rows = range(1, h - ph)
        cols = range(1, max(1, c0 - pw - 1))
    else:
        rows = range(1, h - ph)
        cols = range(c1 + 3, w - pw)
    return [(int(r), int(c)) for r in rows for c in cols]


def _build_expected(
    shape: tuple[int, int],
    skeleton_offset: tuple[int, int],
    arcs: list[list[tuple[int, int]]],
    arc_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for arc, color in zip(arcs, arc_colors, strict=True):
        _paint_cells(out, arc, int(color), offset=skeleton_offset)
    return out


def _gen_attempt() -> np.ndarray | None:
    cycle = _generate_ring_cycle()
    if cycle is None:
        return None

    gap_count = random.randint(3, 5)
    gaps = _sample_gap_indices(len(cycle), gap_count, min_arc_len=4)
    if gaps is None:
        return None

    arcs = _arcs_from_cycle(cycle, gaps)
    if any(len(arc) < 4 for arc in arcs):
        return None

    arc_masks = []
    canonical_keys = []
    for arc in arcs:
        mask, _bbox_arc = _mask_from_coords(arc)
        arc_masks.append(mask)
        canonical_keys.append(_canonical_rot_key(mask))
    if len(set(canonical_keys)) != len(canonical_keys):
        return None

    local_h = max(r for r, _ in cycle) + 1
    local_w = max(c for _, c in cycle) + 1
    max_ph = max(mask.shape[0] for mask in arc_masks)
    max_pw = max(mask.shape[1] for mask in arc_masks)
    grid_h = local_h + max_ph + random.randint(6, 8)
    grid_w = local_w + max_pw + random.randint(6, 8)
    if grid_h > 30 or grid_w > 30:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    skel_r0 = max(2, (grid_h - local_h) // 2 + random.randint(-1, 1))
    skel_c0 = max(2, (grid_w - local_w) // 2 + random.randint(-1, 1))
    if skel_r0 + local_h > grid_h - 2 or skel_c0 + local_w > grid_w - 2:
        return None

    for idx, cell in enumerate(cycle):
        if idx in gaps:
            continue
        _paint_cells(inp, [cell], SKELETON, offset=(skel_r0, skel_c0))

    occupied = inp != BG
    skeleton_bbox = (skel_r0, skel_c0, skel_r0 + local_h - 1, skel_c0 + local_w - 1)
    colors = random.sample(list(FRAGMENT_COLORS), len(arcs))
    expected = _build_expected(inp.shape, (skel_r0, skel_c0), arcs, colors)

    zone_pool = ["top", "bottom", "left", "right"]
    zone_assign = random.sample(zone_pool, k=min(len(arcs), 4))
    while len(zone_assign) < len(arcs):
        zone_assign.append(random.choice(zone_pool))
    random.shuffle(zone_assign)

    for arc_idx, arc in enumerate(arcs):
        mask = arc_masks[arc_idx]
        rot = random.randint(0, 3)
        placed_mask = np.rot90(mask, rot)
        preferred = zone_assign[arc_idx]
        zones = [preferred] + [z for z in zone_pool if z != preferred]
        placed = False
        for zone in zones:
            candidates = _zone_candidates(placed_mask.shape, inp.shape, skeleton_bbox, zone)
            random.shuffle(candidates)
            for top, left in candidates:
                if _can_place_patch(occupied, placed_mask, int(top), int(left), pad=1):
                    _place_patch(inp, occupied, placed_mask, int(colors[arc_idx]), int(top), int(left))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
