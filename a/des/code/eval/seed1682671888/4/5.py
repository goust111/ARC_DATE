from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
WINGS = {
    0: [(1, -1), (1, 1)],
    1: [(-1, -1), (1, -1)],
    2: [(-1, -1), (-1, 1)],
    3: [(-1, 1), (1, 1)],
}
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEI8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
PALETTE = list(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = NEI4 if connectivity == 4 else NEI8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], int] | None:
    if len(coords) != 3:
        return None
    pts = set(coords)
    for center in coords:
        cr, cc = center
        for direction in range(4):
            need = {(cr, cc)}
            for dr, dc in WINGS[direction]:
                need.add((cr + dr, cc + dc))
            if need == pts:
                return center, direction
    return None


def _arrow_cells(center: tuple[int, int], direction: int) -> set[tuple[int, int]]:
    cr, cc = center
    cells = {(cr, cc)}
    for dr, dc in WINGS[direction]:
        cells.add((cr + dr, cc + dc))
    return cells


def _rotate_cw(offset: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = offset
    for _ in range(turns % 4):
        dr, dc = dc, -dr
    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored 3-pixel arrows as portals.

    One arrow in each pair points to an object and sends it. The other points to
    empty space and receives it. The pointed object pixel two cells in front of
    the sending arrow is the alignment reference; after rotating by the arrows'
    orientation difference, that reference pixel lands two cells in front of the
    receiving arrow. All arrows are erased.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    arrow_infos: list[dict] = []
    arrow_mask = np.zeros_like(g, dtype=bool)
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        comps = _components(g == color, connectivity=8)
        if len(comps) != 2:
            continue
        infos = []
        ok = True
        for comp in comps:
            hit = _detect_arrow(comp)
            if hit is None:
                ok = False
                break
            center, direction = hit
            infos.append(
                {
                    "color": color,
                    "coords": comp,
                    "center": center,
                    "dir": direction,
                }
            )
        if ok:
            arrow_infos.extend(infos)
            for info in infos:
                for r, c in info["coords"]:
                    arrow_mask[r, c] = True

    by_color: dict[int, list[dict]] = {}
    for info in arrow_infos:
        by_color.setdefault(int(info["color"]), []).append(info)
    assert by_color, "expected at least one arrow pair"
    assert all(len(v) == 2 for v in by_color.values()), "each arrow color must appear exactly twice"

    object_mask = (g != bg) & (~arrow_mask)
    comps = _components(object_mask, connectivity=4)
    coord_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in comp:
            coord_to_comp[cell] = idx

    moves: list[dict] = []
    used_components: set[int] = set()
    for color, pair in by_color.items():
        roles = []
        for info in pair:
            cr, cc = info["center"]
            dr, dc = DIRS[int(info["dir"])]
            gap = (cr + dr, cc + dc)
            ref = (cr + 2 * dr, cc + 2 * dc)
            assert _in_bounds(g.shape, gap[0], gap[1])
            assert _in_bounds(g.shape, ref[0], ref[1])
            hit_comp = coord_to_comp.get(ref)
            roles.append(
                {
                    "info": info,
                    "gap": gap,
                    "ref": ref,
                    "comp": hit_comp,
                }
            )
        senders = [r for r in roles if r["comp"] is not None]
        receivers = [r for r in roles if r["comp"] is None]
        assert len(senders) == 1 and len(receivers) == 1, f"arrow pair color {color} must have one sender and one receiver"

        sender = senders[0]
        receiver = receivers[0]
        comp_idx = int(sender["comp"])
        assert comp_idx not in used_components, "one object cannot be sent twice"
        used_components.add(comp_idx)
        turns = (int(receiver["info"]["dir"]) - int(sender["info"]["dir"])) % 4
        moves.append(
            {
                "comp_idx": comp_idx,
                "src_ref": sender["ref"],
                "dst_ref": receiver["ref"],
                "turns": turns,
            }
        )

    out = g.copy()
    out[arrow_mask] = bg
    for move in moves:
        for r, c in comps[move["comp_idx"]]:
            out[r, c] = bg

    for move in moves:
        sr, sc = move["src_ref"]
        tr, tc = move["dst_ref"]
        turns = int(move["turns"])
        for r, c in comps[move["comp_idx"]]:
            nr_off, nc_off = _rotate_cw((r - sr, c - sc), turns)
            nr, nc = tr + nr_off, tc + nc_off
            assert _in_bounds(g.shape, nr, nc), "moved object must stay inside the grid"
            assert out[nr, nc] == bg, "moved object cannot overlap another remaining object"
            out[nr, nc] = g[r, c]

    return out


GEN_RETRIES = 400
GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
PAIR_COUNT_CHOICES = (1, 2, 3)
DISTRACTOR_COUNT_CHOICES = (0, 1, 2)
MAX_PAIR_PLACEMENT_TRIES = 300
MAX_DISTRACTOR_TRIES = 200


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _random_canonical_shape(rng: random.Random) -> list[tuple[int, int]]:
    while True:
        width = rng.randint(2, 4)
        up = rng.randint(0, 2)
        down = rng.randint(0, 2)
        rows = list(range(-up, down + 1))
        candidates = {(r, c) for r in rows for c in range(width)}
        if len(candidates) >= 3:
            break
    target = rng.randint(3, min(7, len(candidates)))

    cells = {(0, 0)}
    frontier = {(0, 0)}
    while len(cells) < target:
        base = rng.choice(tuple(frontier))
        nbrs = []
        for dr, dc in NEI4:
            nxt = (base[0] + dr, base[1] + dc)
            if nxt in candidates and nxt not in cells:
                nbrs.append(nxt)
        if not nbrs:
            frontier.remove(base)
            if not frontier:
                frontier = set(cells)
            continue
        nxt = rng.choice(nbrs)
        cells.add(nxt)
        frontier.add(nxt)
    return sorted(cells)


def _shape_with_colors(
    rng: random.Random, palette: list[int]
) -> list[tuple[int, int, int]]:
    offsets = _random_canonical_shape(rng)
    color_count = min(len(palette), rng.randint(1, 3))
    colors = rng.sample(palette, color_count)
    return [(r, c, int(rng.choice(colors))) for r, c in offsets]


def _rotate_shape(
    cells: list[tuple[int, int, int]], turns: int
) -> list[tuple[int, int, int]]:
    out = []
    for r, c, color in cells:
        nr, nc = _rotate_cw((r, c), turns)
        out.append((nr, nc, color))
    return out


def _place_shape(
    anchor: tuple[int, int], cells: list[tuple[int, int, int]]
) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c, _ in cells}


def _paint_shape(
    grid: np.ndarray, anchor: tuple[int, int], cells: list[tuple[int, int, int]]
) -> None:
    ar, ac = anchor
    for r, c, color in cells:
        grid[ar + r, ac + c] = int(color)


def _shape_fits(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in cells)


def _random_center(rng: random.Random, shape: tuple[int, int]) -> tuple[int, int]:
    return rng.randrange(1, shape[0] - 1), rng.randrange(1, shape[1] - 1)


def _random_distractor(
    rng: random.Random, palette: list[int]
) -> list[tuple[int, int, int]]:
    while True:
        height = rng.randint(1, 4)
        width = rng.randint(1, 4)
        candidates = {(r, c) for r in range(height) for c in range(width)}
        if len(candidates) >= 2:
            break
    target = rng.randint(2, min(6, len(candidates)))
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier = {start}
    while len(cells) < target:
        base = rng.choice(tuple(frontier))
        nbrs = []
        for dr, dc in NEI4:
            nxt = (base[0] + dr, base[1] + dc)
            if nxt in candidates and nxt not in cells:
                nbrs.append(nxt)
        if not nbrs:
            frontier.remove(base)
            if not frontier:
                frontier = set(cells)
            continue
        nxt = rng.choice(nbrs)
        cells.add(nxt)
        frontier.add(nxt)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    color_count = min(len(palette), rng.randint(1, 2))
    colors = rng.sample(palette, color_count)
    return [(r - min_r, c - min_c, int(rng.choice(colors))) for r, c in sorted(cells)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(PALETTE)
    non_bg = [c for c in PALETTE if c != bg]
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    distractor_count = rng.choice(DISTRACTOR_COUNT_CHOICES)
    if len(non_bg) < pair_count:
        return None

    arrow_colors = rng.sample(non_bg, pair_count)
    object_palette = [c for c in non_bg if c not in arrow_colors]
    if not object_palette:
        return None

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    reserved: set[tuple[int, int]] = set()

    for arrow_color in arrow_colors:
        placed = False
        for _ in range(MAX_PAIR_PLACEMENT_TRIES):
            source_dir = rng.randrange(4)
            turn_diff = rng.randrange(4)
            target_dir = (source_dir + turn_diff) % 4

            canonical = _shape_with_colors(rng, object_palette)
            src_shape = _rotate_shape(canonical, (source_dir - 1) % 4)
            dst_shape = _rotate_shape(canonical, (target_dir - 1) % 4)

            src_center = _random_center(rng, (h, w))
            dst_center = _random_center(rng, (h, w))
            src_dr, src_dc = DIRS[source_dir]
            dst_dr, dst_dc = DIRS[target_dir]
            src_anchor = (src_center[0] + 2 * src_dr, src_center[1] + 2 * src_dc)
            dst_anchor = (dst_center[0] + 2 * dst_dr, dst_center[1] + 2 * dst_dc)

            src_arrow = _arrow_cells(src_center, source_dir)
            dst_arrow = _arrow_cells(dst_center, target_dir)
            src_cells = _place_shape(src_anchor, src_shape)
            dst_cells = _place_shape(dst_anchor, dst_shape)
            gaps = {
                (src_center[0] + src_dr, src_center[1] + src_dc),
                (dst_center[0] + dst_dr, dst_center[1] + dst_dc),
            }
            footprint = src_arrow | dst_arrow | src_cells | dst_cells | gaps
            if len(src_arrow) != 3 or len(dst_arrow) != 3:
                continue
            if len(src_cells) != len(src_shape) or len(dst_cells) != len(dst_shape):
                continue
            if not _shape_fits((h, w), footprint):
                continue
            if _halo(footprint, (h, w)) & reserved:
                continue

            _paint_shape(inp, src_anchor, src_shape)
            _paint_shape(out, dst_anchor, dst_shape)
            for r, c in src_arrow | dst_arrow:
                inp[r, c] = int(arrow_color)
            reserved |= _halo(footprint, (h, w))
            placed = True
            break
        if not placed:
            return None

    for _ in range(distractor_count):
        placed = False
        for _ in range(MAX_DISTRACTOR_TRIES):
            shape = _random_distractor(rng, object_palette)
            base_r = rng.randrange(1, h - 1)
            base_c = rng.randrange(1, w - 1)
            cells = {(base_r + r, base_c + c) for r, c, _ in shape}
            if len(cells) != len(shape):
                continue
            if not _shape_fits((h, w), cells):
                continue
            if _halo(cells, (h, w)) & reserved:
                continue
            _paint_shape(inp, (base_r, base_c), shape)
            _paint_shape(out, (base_r, base_c), shape)
            reserved |= _halo(cells, (h, w))
            placed = True
            break
        if not placed:
            return None

    if Counter(map(int, inp.flatten())).most_common(1)[0][0] != bg:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance")
