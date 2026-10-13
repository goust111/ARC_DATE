from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


GRID_H_RANGE = (18, 25)
GRID_W_MAX = 28
GRID_W_EXTRA = (1, 7)
HEIGHT_CHOICES = (2, 3, 4, 5, 6)
LIFT_MAX = 6
GEN_ATTEMPTS = 600

HOLE_PATTERNS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
)


def _all_colors(grid: np.ndarray) -> list[int]:
    return sorted({int(v) for v in np.asarray(grid).ravel()})


def _infer_roles(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    colors = _all_colors(g)
    assert len(colors) == 3, "expected exactly three colors"
    bg = int(get_bg(g))
    others = [c for c in colors if c != bg]
    assert len(others) == 2, "background must be uniquely identifiable"
    bottom_touch = [c for c in others if np.any(g[-1, :] == c)]
    assert len(bottom_touch) == 1, "exactly one non-background color must touch the bottom edge"
    obstacle = int(bottom_touch[0])
    moving = int(others[0] if others[1] == obstacle else others[1])
    assert not np.any(g[-1, :] == moving), "moving color must stay off the bottom row"
    return bg, obstacle, moving


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_edge = True
                    q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _count_component_holes(coords: np.ndarray) -> int:
    return _hole_count(_component_mask(coords))


def _top_obstacle_rows(grid: np.ndarray, obstacle: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rows = np.full(w, h, dtype=int)
    for c in range(w):
        hits = np.flatnonzero(g[:, c] == int(obstacle))
        if hits.size:
            rows[c] = int(hits[0])
    return rows


def _column_bottoms(coords: np.ndarray) -> dict[int, int]:
    bottoms: dict[int, int] = {}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        bottoms[cc] = rr if cc not in bottoms else max(bottoms[cc], rr)
    return bottoms


def _contact_drop(coords: np.ndarray, top_rows: np.ndarray, height: int) -> int:
    bottoms = _column_bottoms(coords)
    d = height
    for c, bottom_r in bottoms.items():
        blocked_row = int(top_rows[c]) if int(top_rows[c]) < height else height
        d = min(d, blocked_row - bottom_r - 1)
    return int(d)


def _projection_signature(coords: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted({int(c) for c in coords[:, 1]}))


def _shape_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords + np.array([int(r0), int(c0)], dtype=int)
    return coords.astype(int)


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _external_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ext = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def _make_object_mask(holes: int) -> np.ndarray:
    chamber_choices = {
        0: [(1, 2), (2, 2), (1, 3)],
        1: [(1, 1), (1, 2), (2, 1), (2, 2)],
        2: [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)],
        3: [(1, 3), (3, 1), (2, 2)],
    }
    for _ in range(200):
        hr, hc = random.choice(chamber_choices[holes])
        mask = np.ones((2 * hr + 1, 2 * hc + 1), dtype=bool)
        if holes > 0:
            cells = [(r, c) for r in range(hr) for c in range(hc)]
            for r, c in random.sample(cells, holes):
                mask[2 * r + 1, 2 * c + 1] = False
        mask = np.pad(mask, 1, constant_values=False)
        growth_steps = random.randint(1 if holes == 0 else 0, 2 + holes)
        for _ in range(growth_steps):
            exterior = _external_background(mask)
            candidates = []
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if not exterior[r, c]:
                        continue
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                            candidates.append((r, c))
                            break
            if not candidates:
                break
            random.shuffle(candidates)
            for r, c in candidates[: random.randint(1, min(3, len(candidates)))]:
                mask[r, c] = True
        mask = np.rot90(mask, random.randint(0, 3))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = _crop_mask(mask)
        if _hole_count(mask) != holes:
            continue
        if mask.shape[0] > 7 or mask.shape[1] > 7:
            continue
        if mask.sum() < 4:
            continue
        if holes == 0 and int(mask.sum()) == mask.shape[0] * mask.shape[1]:
            continue
        return mask
    raise RuntimeError(f"failed to build object with {holes} holes")


def _sample_x_positions(width: int, object_widths: list[int]) -> list[int] | None:
    needed = sum(object_widths) + (len(object_widths) - 1)
    if needed + 2 > width:
        return None
    extra = width - needed
    gaps = [0] * (len(object_widths) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1
    x_positions = []
    cursor = gaps[0]
    for idx, w in enumerate(object_widths):
        x_positions.append(cursor)
        cursor += w
        if idx + 1 < len(object_widths):
            cursor += 1 + gaps[idx + 1]
    return x_positions


def _make_terrain(height: int, width: int) -> list[int]:
    distinct = random.sample(HEIGHT_CHOICES, k=random.randint(2, min(4, len(HEIGHT_CHOICES))))
    profile: list[int] = []
    last = None
    while len(profile) < width:
        run = random.randint(1, 4)
        choices = [h for h in distinct if h != last] or list(distinct)
        h = random.choice(choices)
        profile.extend([h] * run)
        last = h
    profile = profile[:width]
    if len(set(profile)) == 1:
        profile[random.randrange(width)] = random.choice([h for h in HEIGHT_CHOICES if h != profile[0]])
    return profile


def _render_base(height: int, width: int, bg: int, obstacle: int, terrain: list[int]) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for c, depth in enumerate(terrain):
        grid[height - depth :, c] = int(obstacle)
    return grid


def _validate_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    try:
        bg, obstacle, moving = _infer_roles(inp)
    except AssertionError:
        return False
    if len(_all_colors(inp)) != 3:
        return False
    if get_bg(inp) != bg:
        return False
    if np.any(inp[-1, :] == moving):
        return False
    if not np.array_equal(inp == obstacle, out == obstacle):
        return False

    top_rows = _top_obstacle_rows(inp, obstacle)
    in_comps = sorted(
        find_components(inp, bg=bg, connectivity=4, monochromatic=True, colors=[moving]),
        key=lambda comp: int(comp["bbox"][1]),
    )
    out_comps = sorted(
        find_components(out, bg=bg, connectivity=4, monochromatic=True, colors=[moving]),
        key=lambda comp: int(comp["bbox"][1]),
    )
    if len(in_comps) != len(out_comps) or len(in_comps) < 2:
        return False

    seen_projections: set[tuple[int, ...]] = set()
    holes = []
    drops = []
    for in_comp, out_comp in zip(in_comps, out_comps):
        in_coords = np.asarray(in_comp["coords"], dtype=int)
        out_coords = np.asarray(out_comp["coords"], dtype=int)
        sig_in = _projection_signature(in_coords)
        sig_out = _projection_signature(out_coords)
        if sig_in != sig_out or sig_in in seen_projections:
            return False
        seen_projections.add(sig_in)

        delta_r = int(out_comp["bbox"][0]) - int(in_comp["bbox"][0])
        if delta_r < 1:
            return False
        if set(map(tuple, out_coords)) != set(map(tuple, in_coords + np.array([delta_r, 0]))):
            return False

        k_in = _count_component_holes(in_coords)
        k_out = _count_component_holes(out_coords)
        if k_in != k_out:
            return False
        d_in = _contact_drop(in_coords, top_rows, inp.shape[0])
        d_out = _contact_drop(out_coords, top_rows, inp.shape[0])
        if d_in < k_in or d_out != k_in or delta_r != d_in - k_in:
            return False

        holes.append(k_in)
        drops.append(delta_r)

    if len(set(holes)) < 2:
        return False
    if max(drops) < 2:
        return False
    if not any(h == 0 for h in holes):
        return False
    if not any(h > 0 for h in holes):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each movable object so its final clearance above the terrain equals its own enclosed-hole count.

    Input structure:
    - The grid uses exactly three colors: background, a bottom-touching obstacle/terrain color, and one shared moving-object color.
    - Every 4-connected moving object is a rigid shape whose horizontal projection is disjoint from every other moving object.
    - Some moving objects contain 0-3 fully enclosed background holes inside their own boundary.

    Rule:
    - Keep the background and obstacle cells fixed.
    - For each moving object independently, compute the maximum vertical drop d that would make it first touch the obstacle or the grid bottom.
    - Let K be that object's number of enclosed background holes; move the object down by d-K rows instead of the full contact distance.
    - Objects with K=0 land in direct contact, while objects with more holes stop with more empty buffer rows.

    Key insight:
    - The visible gap under each object is not a global gravity rule; it is encoded by that object's own hole count.
    """
    g = np.asarray(grid, dtype=int)
    bg, obstacle, moving = _infer_roles(g)
    top_rows = _top_obstacle_rows(g, obstacle)

    comps = sorted(
        find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[moving]),
        key=lambda comp: int(comp["bbox"][1]),
    )
    projections = [_projection_signature(np.asarray(comp["coords"], dtype=int)) for comp in comps]
    assert len(projections) == len(set(projections)), "moving-object projections must be disjoint"

    out = g.copy()
    out[out == moving] = bg
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        k = _count_component_holes(coords)
        d = _contact_drop(coords, top_rows, g.shape[0])
        assert d >= k, "every object must have at least as much fall room as its hole count"
        shifted = coords + np.array([d - k, 0], dtype=int)
        _paint_coords(out, shifted, moving)
    return out


def _gen_attempt() -> np.ndarray | None:
    hole_pattern = random.choice(HOLE_PATTERNS)
    masks = [_make_object_mask(k) for k in hole_pattern]
    widths = [int(mask.shape[1]) for mask in masks]

    min_width = sum(widths) + len(widths) + 1
    if min_width > GRID_W_MAX:
        return None

    height = random.randint(*GRID_H_RANGE)
    width = random.randint(min_width, min(GRID_W_MAX, min_width + random.randint(*GRID_W_EXTRA)))
    x_positions = _sample_x_positions(width, widths)
    if x_positions is None:
        return None

    bg, obstacle, moving = random.sample(range(10), 3)
    terrain = _make_terrain(height, width)
    base = _render_base(height, width, bg, obstacle, terrain)
    top_rows = _top_obstacle_rows(base, obstacle)

    output_tops: list[int] = []
    max_lifts: list[int] = []
    for mask, holes, x0 in zip(masks, hole_pattern, x_positions):
        local_coords = np.argwhere(mask)
        global_cols = local_coords[:, 1] + int(x0)
        coords = np.column_stack([local_coords[:, 0], global_cols]).astype(int)
        d_contact = _contact_drop(coords, top_rows, height)
        output_top = int(d_contact - holes)
        if output_top < 2:
            return None
        max_lift = min(LIFT_MAX, output_top)
        if max_lift < 1:
            return None
        output_tops.append(output_top)
        max_lifts.append(max_lift)

    lifts = [random.randint(1, m) for m in max_lifts]
    liftable = [i for i, m in enumerate(max_lifts) if m >= 2]
    if liftable and max(lifts) < 2:
        idx = random.choice(liftable)
        lifts[idx] = random.randint(2, max_lifts[idx])

    out = base.copy()
    inp = base.copy()
    for mask, out_top, lift, x0 in zip(masks, output_tops, lifts, x_positions):
        out_coords = _shape_coords(mask, out_top, x0)
        in_coords = _shape_coords(mask, out_top - lift, x0)
        _paint_coords(out, out_coords, moving)
        _paint_coords(inp, in_coords, moving)

    if int((inp == bg).sum()) <= max(int((inp == obstacle).sum()), int((inp == moving).sum())):
        return None
    if not _validate_pair(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
