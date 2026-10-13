from __future__ import annotations

import random

import numpy as np


SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_RANGE = (1, 4)
FRAME_H_RANGE = (5, 9)
FRAME_W_RANGE = (5, 9)
SPRITE_AREA_CAP = 10
PLACEMENT_RETRIES = 140
GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_same_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in SIDE_TO_DIR.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "bbox": _bbox(arr)})
    return comps


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in SIDE_TO_DIR.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _classify_frame_component(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    expected = _rect_border_cells(r0, c0, r1, c1)
    actual = {tuple(map(int, rc)) for rc in coords.tolist()}
    if not actual <= expected:
        return None
    missing = expected - actual
    if len(missing) != 1:
        return None
    gap = next(iter(missing))
    gap_r, gap_c = gap
    sides: list[str] = []
    if gap_r == r0:
        sides.append("top")
    if gap_r == r1:
        sides.append("bottom")
    if gap_c == c0:
        sides.append("left")
    if gap_c == c1:
        sides.append("right")
    if len(sides) != 1:
        return None
    side = sides[0]
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "direction": SIDE_TO_DIR[side],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each inner sprite area into an outward ray emitted through the frame gap.

    Input structure:
    - The dominant color is the background.
    - Several distinct one-pixel-thick rectangular frames appear in different colors.
    - Each frame has exactly one one-pixel gap on one side.
    - Each frame encloses exactly one 4-connected sprite of another color.

    Rule:
    - Find the gap side and gap location of each frame.
    - Count the enclosed sprite pixels.
    - Erase the sprite.
    - Draw a ray of the frame color outside the frame, starting just beyond the gap
      and extending orthogonally away from the missing side for exactly that count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()

    frame_infos: list[dict] = []
    for comp in _components_same_color(g, bg):
        frame = _classify_frame_component(comp["coords"])
        if frame is None:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        sprite_mask = interior != bg
        if not bool(sprite_mask.any()):
            continue
        inner_components = _mask_components(sprite_mask)
        assert len(inner_components) == 1, "each frame must contain exactly one inner sprite"
        sprite_rel = inner_components[0]
        sprite_colors = np.unique(interior[sprite_mask])
        assert sprite_colors.size == 1, "inner sprite must be monochrome"
        sprite_abs = sprite_rel + np.array([r0 + 1, c0 + 1], dtype=int)
        frame["color"] = int(comp["color"])
        frame["sprite_coords"] = sprite_abs
        frame["sprite_size"] = int(sprite_abs.shape[0])
        frame_infos.append(frame)

    assert frame_infos, "expected at least one valid frame"

    for info in frame_infos:
        coords = info["sprite_coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for info in frame_infos:
        gap_r, gap_c = info["gap"]
        dr, dc = info["direction"]
        for step in range(1, info["sprite_size"] + 1):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = info["color"]

    return out


def _grow_connected_shape(h: int, w: int, area: int) -> np.ndarray | None:
    if area < 1 or area > h * w:
        return None
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in SIDE_TO_DIR.values():
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    arr = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    return arr - np.array([r0, c0], dtype=int)


def _sample_sprite(interior_h: int, interior_w: int, area: int) -> np.ndarray | None:
    options: list[tuple[int, int]] = []
    slack = min(4, interior_h * interior_w - area)
    for sh in range(1, interior_h + 1):
        for sw in range(1, interior_w + 1):
            if area <= sh * sw <= area + slack:
                options.append((sh, sw))
    random.shuffle(options)
    for sh, sw in options:
        coords = _grow_connected_shape(sh, sw, area)
        if coords is None:
            continue
        if _classify_frame_component(coords) is not None:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        shape_h = r1 - r0 + 1
        shape_w = c1 - c0 + 1
        spare_r = interior_h - shape_h
        spare_c = interior_w - shape_w
        row_choices = list(range(spare_r + 1))
        col_choices = list(range(spare_c + 1))
        if spare_r >= 2:
            row_choices = list(range(1, spare_r))
        if spare_c >= 2:
            col_choices = list(range(1, spare_c))
        off_r = random.choice(row_choices) if row_choices else 0
        off_c = random.choice(col_choices) if col_choices else 0
        return coords + np.array([off_r, off_c], dtype=int)
    return None


def _ray_cells(gap: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    gap_r, gap_c = gap
    dr, dc = direction
    return [(gap_r + dr * step, gap_c + dc * step) for step in range(1, length + 1)]


def _expanded(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h - 1, r + radius) + 1):
            for cc in range(max(0, c - radius), min(w - 1, c + radius) + 1):
                out.add((rr, cc))
    return out


def _place_frame(
    inp: np.ndarray,
    out: np.ndarray,
    reserved: set[tuple[int, int]],
    frame_color: int,
    sprite_color: int,
) -> bool:
    h, w = inp.shape
    for _ in range(PLACEMENT_RETRIES):
        frame_h = random.randint(*FRAME_H_RANGE)
        frame_w = random.randint(*FRAME_W_RANGE)
        if frame_h >= h - 1 or frame_w >= w - 1:
            continue
        side = random.choice(list(SIDE_TO_DIR))
        r0 = random.randint(1, h - frame_h - 1)
        c0 = random.randint(1, w - frame_w - 1)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        if side == "top":
            ray_capacity = r0
            gap = (r0, random.randint(c0 + 1, c1 - 1))
        elif side == "bottom":
            ray_capacity = h - r1 - 1
            gap = (r1, random.randint(c0 + 1, c1 - 1))
        elif side == "left":
            ray_capacity = c0
            gap = (random.randint(r0 + 1, r1 - 1), c0)
        else:
            ray_capacity = w - c1 - 1
            gap = (random.randint(r0 + 1, r1 - 1), c1)

        interior_h = frame_h - 2
        interior_w = frame_w - 2
        interior_area = interior_h * interior_w
        max_area = min(SPRITE_AREA_CAP, ray_capacity, interior_area - (1 if interior_area > 1 else 0))
        if max_area < 1:
            continue
        area = random.randint(1, max_area)
        sprite_rel = _sample_sprite(interior_h, interior_w, area)
        if sprite_rel is None:
            continue

        border_cells = _rect_border_cells(r0, c0, r1, c1)
        border_cells.remove(gap)
        sprite_cells = {
            (r0 + 1 + int(sr), c0 + 1 + int(sc))
            for sr, sc in sprite_rel.tolist()
        }
        ray = _ray_cells(gap, SIDE_TO_DIR[side], area)
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in ray):
            continue
        ray_cells = set(ray)

        footprint = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
        claim = footprint | ray_cells
        moat = _expanded(claim, h, w, radius=1)
        if claim & reserved or moat & reserved:
            continue

        for rr, cc in border_cells:
            inp[rr, cc] = frame_color
            out[rr, cc] = frame_color
        for rr, cc in sprite_cells:
            inp[rr, cc] = sprite_color
        for rr, cc in ray_cells:
            out[rr, cc] = frame_color
        reserved.update(claim)
        reserved.update(moat)
        return True
    return False


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    min_side = max(GRID_MIN, 12 + 3 * n_frames)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)

    palette = random.sample(range(10), 1 + 2 * n_frames)
    bg = palette[0]
    frame_colors = palette[1 : 1 + n_frames]
    sprite_colors = palette[1 + n_frames :]

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    reserved: set[tuple[int, int]] = set()

    order = list(range(n_frames))
    random.shuffle(order)
    for idx in order:
        ok = _place_frame(inp, out, reserved, frame_colors[idx], sprite_colors[idx])
        if not ok:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
