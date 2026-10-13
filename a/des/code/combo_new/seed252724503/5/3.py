from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FILL_COLOR = 9


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))

    return comps


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.tobytes()


def _matches_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    cur = a
    for _ in range(4):
        if cur.shape == b.shape and np.array_equal(cur, b):
            return True
        cur = np.rot90(cur)
    return False


def _interior_holes(coords: np.ndarray) -> list[np.ndarray]:
    mask, (r0, c0, _, _) = _tight_mask(coords)
    holes: list[np.ndarray] = []

    for hole in _components(~mask):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == mask.shape[0] - 1)
            | (hole[:, 1] == mask.shape[1] - 1)
        )
        if touches_edge:
            continue
        shifted = hole.copy()
        shifted[:, 0] += r0
        shifted[:, 1] += c0
        holes.append(shifted)

    return holes


def _pair_roles(comps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    infos = []
    for comp in comps:
        holes = _interior_holes(comp)
        infos.append({"coords": comp, "holes": holes, "area": len(comp)})

    large_idx = max(range(len(infos)), key=lambda i: (len(infos[i]["holes"]) > 0, infos[i]["area"]))
    small_idx = 1 - large_idx
    return infos[small_idx]["coords"], infos[large_idx]["coords"], infos[large_idx]["holes"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each same-color key shape to the cavity inside its larger partner and fill that cavity in maroon.

    Input structure:
    - The grid has one dominant background color and several foreground colors.
    - Each foreground color appears in exactly two 4-connected components.
    - For each color, one component is a smaller solid shape and the other is a larger shape
      whose bbox contains one enclosed background cavity congruent to the small shape up to
      90-degree rotation.

    Rule:
    - Group components by color and pair the two components of each color.
    - Identify the larger partner's enclosed cavity as the 4-connected background region inside
      its bbox that does not touch the bbox edge.
    - Rotate the smaller shape conceptually until it matches that cavity, then recolor the cavity
      maroon (9).
    - Erase the smaller shape's original location back to the background color and leave all other
      cells unchanged.

    Key insight:
    - Color establishes which two objects belong together; shape matching under rotation decides
      where the small object gets projected inside the larger object's internal hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _components(g == color)
        if len(comps) != 2:
            continue

        small, _large, holes = _pair_roles(comps)
        small_mask, _ = _tight_mask(small)

        for hole in holes:
            hole_mask, _ = _tight_mask(hole)
            if _matches_rotation(small_mask, hole_mask):
                out[hole[:, 0], hole[:, 1]] = FILL_COLOR
                out[small[:, 0], small[:, 1]] = bg
                break

    return out


# --------------- generator ---------------

ARC_COLORS = list(range(10))
BG_CHOICES = [0, 1, 2, 4, 5]
PAIR_COUNT_CHOICES = [1, 2, 2, 3]
GRID_SIDE_BASE = {1: (16, 24), 2: (20, 28), 3: (24, 30)}
PAIR_COLOR_POOL = [c for c in ARC_COLORS if c != FILL_COLOR]

POLY_AREA_RANGE = (4, 6)
POLY_ATTEMPTS = 80
LARGE_ATTEMPTS = 60
PLACEMENT_ATTEMPTS = 350
GEN_ATTEMPTS = 400

BORDER = 1
PAD = 1
MARGIN_RANGE = (1, 2)
TAB_EXT_RANGE = (1, 2)
TAB_PROB = 0.6
MAX_COMPONENT_SIDE = 12


def _random_polyomino() -> np.ndarray | None:
    for _ in range(POLY_ATTEMPTS):
        area = random.randint(*POLY_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(tuple(frontier)))

        pts = np.asarray(sorted(cells), dtype=int)
        pts -= pts.min(axis=0)
        h = int(pts[:, 0].max()) + 1
        w = int(pts[:, 1].max()) + 1
        mask = np.zeros((h, w), dtype=bool)
        mask[pts[:, 0], pts[:, 1]] = True

        if min(mask.shape) < 2 or int(mask.sum()) == mask.size:
            continue
        if len({_mask_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        return mask

    return None


def _segment(length: int) -> tuple[int, int]:
    if length <= 2:
        return 0, length
    span = random.randint(1, length - 1)
    start = random.randint(0, length - span)
    return start, span


def _build_large_shape(hole_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    hh, hw = hole_mask.shape

    for _ in range(LARGE_ATTEMPTS):
        mt = random.randint(*MARGIN_RANGE)
        mb = random.randint(*MARGIN_RANGE)
        ml = random.randint(*MARGIN_RANGE)
        mr = random.randint(*MARGIN_RANGE)
        core_h = hh + mt + mb
        core_w = hw + ml + mr

        ext_top = random.randint(*TAB_EXT_RANGE) if random.random() < TAB_PROB else 0
        ext_bottom = random.randint(*TAB_EXT_RANGE) if random.random() < TAB_PROB else 0
        ext_left = random.randint(*TAB_EXT_RANGE) if random.random() < TAB_PROB else 0
        ext_right = random.randint(*TAB_EXT_RANGE) if random.random() < TAB_PROB else 0

        total_h = core_h + ext_top + ext_bottom
        total_w = core_w + ext_left + ext_right
        if total_h > MAX_COMPONENT_SIDE or total_w > MAX_COMPONENT_SIDE:
            continue

        mask = np.zeros((total_h, total_w), dtype=bool)
        base_r = ext_top
        base_c = ext_left
        mask[base_r : base_r + core_h, base_c : base_c + core_w] = True

        hole_r = base_r + mt
        hole_c = base_c + ml
        carved = mask[hole_r : hole_r + hh, hole_c : hole_c + hw]
        carved[hole_mask] = False

        if ext_top:
            start, span = _segment(core_w)
            mask[:ext_top, base_c + start : base_c + start + span] = True
        if ext_bottom:
            start, span = _segment(core_w)
            mask[base_r + core_h :, base_c + start : base_c + start + span] = True
        if ext_left:
            start, span = _segment(core_h)
            mask[base_r + start : base_r + start + span, :ext_left] = True
        if ext_right:
            start, span = _segment(core_h)
            mask[base_r + start : base_r + start + span, base_c + core_w :] = True

        hole_coords = np.argwhere(hole_mask) + np.asarray([hole_r, hole_c])
        return mask, hole_coords

    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    for rr, cc in np.argwhere(mask):
        r = int(r0 + rr)
        c = int(c0 + cc)
        if occupied[r, c]:
            return False
        rs = max(0, r - PAD)
        re = min(h, r + PAD + 1)
        cs = max(0, c - PAD)
        ce = min(w, c + PAD + 1)
        if occupied[rs:re, cs:ce].any():
            return False
    return True


def _place_component(occupied: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    h, w = occupied.shape
    mh, mw = mask.shape
    if mh + 2 * BORDER > h or mw + 2 * BORDER > w:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(BORDER, h - mh - BORDER)
        c0 = random.randint(BORDER, w - mw - BORDER)
        if _can_place(occupied, mask, r0, c0):
            return r0, c0
    return None


def _validate_instance(grid: np.ndarray, bg: int, colors: list[int]) -> bool:
    if _bg_color(grid) != bg:
        return False
    if FILL_COLOR in colors:
        return False

    for color in colors:
        comps = _components(grid == color)
        if len(comps) != 2:
            return False
        small, _large, holes = _pair_roles(comps)
        if len(holes) != 1:
            return False
        small_mask, _ = _tight_mask(small)
        hole_mask, _ = _tight_mask(holes[0])
        if not _matches_rotation(small_mask, hole_mask):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    bg = random.choice(BG_CHOICES)
    pair_colors = random.sample([c for c in PAIR_COLOR_POOL if c != bg], pair_count)

    pair_specs = []
    max_h = 0
    max_w = 0

    for color in pair_colors:
        base = _random_polyomino()
        if base is None:
            return None

        small_rot = random.randrange(4)
        hole_rot = random.randrange(4)
        small_mask = np.rot90(base, small_rot)
        hole_mask = np.rot90(base, hole_rot)
        built = _build_large_shape(hole_mask)
        if built is None:
            return None
        large_mask, hole_coords = built

        pair_specs.append(
            {
                "color": color,
                "small_mask": small_mask,
                "large_mask": large_mask,
                "hole_coords": hole_coords,
            }
        )
        max_h = max(max_h, small_mask.shape[0], large_mask.shape[0])
        max_w = max(max_w, small_mask.shape[1], large_mask.shape[1])

    h_lo, h_hi = GRID_SIDE_BASE[pair_count]
    min_h = max(max_h + 2 * BORDER + 1, h_lo)
    min_w = max(max_w + 2 * BORDER + 1, h_lo)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, h_hi)
    grid_w = random.randint(min_w, h_hi)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    components = []
    for idx, spec in enumerate(pair_specs):
        components.append(
            {
                "pair": idx,
                "kind": "large",
                "color": spec["color"],
                "mask": spec["large_mask"],
                "hole_coords": spec["hole_coords"],
            }
        )
        components.append(
            {
                "pair": idx,
                "kind": "small",
                "color": spec["color"],
                "mask": spec["small_mask"],
            }
        )
    random.shuffle(components)

    placements = []
    for comp in sorted(components, key=lambda x: int(x["mask"].sum()), reverse=True):
        pos = _place_component(occupied, comp["mask"])
        if pos is None:
            return None
        r0, c0 = pos
        coords = np.argwhere(comp["mask"]) + np.asarray([r0, c0])
        grid[coords[:, 0], coords[:, 1]] = comp["color"]
        occupied[coords[:, 0], coords[:, 1]] = True

        placed = dict(comp)
        placed["coords"] = coords
        placed["origin"] = (r0, c0)
        placements.append(placed)

    expected = grid.copy()
    for placed in placements:
        if placed["kind"] == "small":
            coords = placed["coords"]
            expected[coords[:, 0], coords[:, 1]] = bg
        else:
            hole_coords = placed["hole_coords"] + np.asarray(placed["origin"])
            expected[hole_coords[:, 0], hole_coords[:, 1]] = FILL_COLOR

    if not _validate_instance(grid, bg, pair_colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
