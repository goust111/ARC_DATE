from __future__ import annotations

import random

import numpy as np


BG = 0
RING_RED = 2
RING_GREEN = 3


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _color_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    out = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _components_mask(grid == color):
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _is_rectangular_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if comp["area"] != 2 * (h + w) - 4:
        return False

    rr = comp["coords"][:, 0]
    cc = comp["coords"][:, 1]
    on_border = (rr == r0) | (rr == r1) | (cc == c0) | (cc == c1)
    return bool(np.all(on_border))


def _frame_components(grid: np.ndarray) -> list[dict]:
    frames = [comp for comp in _color_components(grid, bg=BG) if _is_rectangular_frame(comp)]
    assert frames, "expected at least one hollow rectangular frame"
    return sorted(frames, key=lambda comp: comp["bbox"])


def _single_motif(inner: np.ndarray, frame_color: int) -> tuple[int, np.ndarray]:
    mask = (inner != BG) & (inner != frame_color)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one motif inside each frame"
    colors = {int(inner[r, c]) for r, c in coords}
    assert len(colors) == 1, "motif must be monochromatic"
    comps = _components_mask(mask)
    assert len(comps) == 1, "motif must be 4-connected"
    return colors.pop(), coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each rectangular frame to mirror, concentric-ring, or solid-fill completion by its interior aspect ratio.

    Input structure:
    - A black background contains several one-pixel-thick monochromatic rectangular frames.
    - Each frame encloses exactly one small 4-connected monochromatic motif of a different color.
    - Motifs lie strictly inside their frames, with empty black cells elsewhere in the frame interior.

    Rule:
    - For each frame, measure the interior width W and height H, excluding the frame border.
    - If W > H, reflect the motif across the interior's vertical center axis and add the reflected pixels in the motif color.
    - If H > W, ignore the motif and recolor the entire interior as concentric rectangular rings: depth 0 uses the frame color, odd positive depths use red (2), and even positive depths use green (3).
    - If W = H, fill every black interior cell with the motif color so the whole interior becomes solid.
    - Keep the frame borders and the black background unchanged.

    Key insight:
    - The same frame-and-motif input vocabulary splits into three distinct transformations, and the frame's interior aspect ratio is the only routing signal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for frame in _frame_components(g):
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        h, w = inner.shape
        assert h > 0 and w > 0, "frame must have a non-empty interior"

        motif_color, motif_coords = _single_motif(inner, frame_color)
        if w > h:
            for rr, cc in motif_coords:
                out[r0 + 1 + rr, c0 + w - cc] = motif_color
        elif h > w:
            rows = np.arange(h)[:, None]
            cols = np.arange(w)[None, :]
            depth = np.minimum(np.minimum(rows, h - 1 - rows), np.minimum(cols, w - 1 - cols))
            region = out[r0 + 1 : r1, c0 + 1 : c1]
            region[depth == 0] = frame_color
            region[(depth > 0) & (depth % 2 == 1)] = RING_RED
            region[(depth > 0) & (depth % 2 == 0)] = RING_GREEN
        else:
            region = out[r0 + 1 : r1, c0 + 1 : c1]
            region[region == BG] = motif_color

    return out


GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 28
FRAME_COLORS = (1, 4, 5, 6, 7, 8, 9)
BRANCHES = ("wide", "tall", "square")


def _sample_frame_dims(branch: str) -> tuple[int, int]:
    if branch == "wide":
        inner_h = random.randint(4, 6)
        inner_w = random.randint(max(7, inner_h + 2), 11)
        return inner_h, inner_w
    if branch == "tall":
        inner_w = random.randint(4, 6)
        inner_h = random.randint(max(7, inner_w + 2), 11)
        return inner_h, inner_w
    side = random.randint(5, 8)
    return side, side


def _shape_box_options(max_h: int, max_w: int, area: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(1, max_h + 1):
        for w in range(1, max_w + 1):
            if h * w >= area:
                dims.append((h, w))
    return dims


def _random_connected_shape(max_h: int, max_w: int, area: int) -> np.ndarray | None:
    options = _shape_box_options(max_h, max_w, area)
    if not options:
        return None

    for _ in range(80):
        box_h, box_w = random.choice(options)
        mask = np.zeros((box_h, box_w), dtype=bool)
        start = (random.randrange(box_h), random.randrange(box_w))
        mask[start] = True
        filled = {start}

        while len(filled) < area:
            frontier = set()
            for r, c in filled:
                for nr, nc in _neighbors4(r, c, box_h, box_w):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = random.choice(tuple(frontier))
            mask[choice] = True
            filled.add(choice)

        if len(filled) != area:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] <= max_h and cropped.shape[1] <= max_w:
            return cropped
    return None


def _place_wide_motif(inner_h: int, inner_w: int, area: int) -> tuple[np.ndarray, str] | None:
    half_w = (inner_w - 2) // 2
    if inner_h < 3 or half_w < 1:
        return None
    shape = _random_connected_shape(inner_h - 2, half_w, area)
    if shape is None:
        return None

    shape_h, shape_w = shape.shape
    top = random.randint(1, inner_h - shape_h - 1)
    side = random.choice(("left", "right"))
    if side == "left":
        left = 1
    else:
        left = inner_w - shape_w - 1

    motif = np.zeros((inner_h, inner_w), dtype=bool)
    motif[top : top + shape_h, left : left + shape_w] = shape
    return motif, side


def _place_side_motif(inner_h: int, inner_w: int, area: int) -> tuple[np.ndarray, str] | None:
    if inner_h < 3 or inner_w < 3:
        return None
    shape = _random_connected_shape(inner_h - 2, inner_w - 2, area)
    if shape is None:
        return None

    shape_h, shape_w = shape.shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        top = 1
        left = random.randint(1, inner_w - shape_w - 1)
    elif side == "bottom":
        top = inner_h - shape_h - 1
        left = random.randint(1, inner_w - shape_w - 1)
    elif side == "left":
        top = random.randint(1, inner_h - shape_h - 1)
        left = 1
    else:
        top = random.randint(1, inner_h - shape_h - 1)
        left = inner_w - shape_w - 1

    motif = np.zeros((inner_h, inner_w), dtype=bool)
    motif[top : top + shape_h, left : left + shape_w] = shape
    return motif, side


def _render_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _bbox_fits(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _valid_positions(occupied: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            if _bbox_fits(occupied, top, left, height, width):
                positions.append((top, left))
    return positions


def _expected_from_plan(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in placements:
        top = spec["top"]
        left = spec["left"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        frame_color = spec["frame_color"]
        motif_color = spec["motif_color"]
        motif_mask = spec["motif_mask"]

        if spec["branch"] == "wide":
            coords = np.argwhere(motif_mask)
            for rr, cc in coords:
                out[top + 1 + rr, left + 1 + (inner_w - 1 - cc)] = motif_color
        elif spec["branch"] == "tall":
            rows = np.arange(inner_h)[:, None]
            cols = np.arange(inner_w)[None, :]
            depth = np.minimum(
                np.minimum(rows, inner_h - 1 - rows),
                np.minimum(cols, inner_w - 1 - cols),
            )
            region = out[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w]
            region[depth == 0] = frame_color
            region[(depth > 0) & (depth % 2 == 1)] = RING_RED
            region[(depth > 0) & (depth % 2 == 0)] = RING_GREEN
        else:
            region = out[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w]
            region[region == BG] = motif_color
    return out


def _validate_plan(inp: np.ndarray, out: np.ndarray, placements: list[dict]) -> bool:
    if {spec["branch"] for spec in placements} != set(BRANCHES):
        return False

    try:
        detected = _frame_components(inp)
    except AssertionError:
        return False
    if len(detected) != len(placements):
        return False

    for spec in placements:
        top = spec["top"]
        left = spec["left"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        frame_color = spec["frame_color"]
        motif_color = spec["motif_color"]
        motif_mask = spec["motif_mask"]

        region_in = inp[top : top + inner_h + 2, left : left + inner_w + 2]
        region_out = out[top : top + inner_h + 2, left : left + inner_w + 2]
        inner_in = region_in[1:-1, 1:-1]
        inner_out = region_out[1:-1, 1:-1]

        border_mask = np.zeros(region_in.shape, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.all(region_in[border_mask] == frame_color):
            return False
        if not np.array_equal(region_in[border_mask], region_out[border_mask]):
            return False

        motif_coords = np.argwhere(motif_mask)
        if motif_coords.shape[0] < 3 or motif_coords.shape[0] > 6:
            return False
        if motif_coords[:, 0].min() < 1 or motif_coords[:, 0].max() > inner_h - 2:
            return False
        if motif_coords[:, 1].min() < 1 or motif_coords[:, 1].max() > inner_w - 2:
            return False
        if np.any(inner_in[motif_mask] != motif_color):
            return False
        bg_count = int(np.count_nonzero(inner_in == BG))

        if spec["branch"] == "wide":
            mirrored = np.zeros_like(motif_mask, dtype=bool)
            for rr, cc in motif_coords:
                mirrored[rr, inner_w - 1 - cc] = True
            added = mirrored & ~motif_mask
            if not added.any():
                return False
            if np.any(inner_out[mirrored] != motif_color):
                return False
        elif spec["branch"] == "tall":
            if motif_color in set(map(int, inner_out.flat)):
                return False
            if frame_color not in set(map(int, inner_out.flat)):
                return False
            if RING_RED not in set(map(int, inner_out.flat)):
                return False
        else:
            if bg_count == 0:
                return False
            if not np.all(inner_out == motif_color):
                return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out) and np.array_equal(main(inp.copy()), out))


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)

    branches = list(BRANCHES)
    if random.random() < 0.55:
        branches.append(random.choice(BRANCHES))
    random.shuffle(branches)

    frame_colors = random.sample(FRAME_COLORS, len(branches))
    frame_specs = []
    for branch, frame_color in zip(branches, frame_colors):
        inner_h, inner_w = _sample_frame_dims(branch)
        frame_specs.append(
            {
                "branch": branch,
                "frame_color": frame_color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_area": (inner_h + 2) * (inner_w + 2),
            }
        )

    placements: list[dict] = []
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in sorted(frame_specs, key=lambda item: item["outer_area"], reverse=True):
        branch = spec["branch"]
        frame_color = spec["frame_color"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        positions = _valid_positions(occupied, outer_h, outer_w)
        if not positions:
            return None
        top, left = random.choice(positions)
        occupied[top : top + outer_h, left : left + outer_w] = True

        motif_color = random.choice([c for c in FRAME_COLORS if c != frame_color])
        area = random.randint(3, 6)
        if branch == "wide":
            motif_plan = _place_wide_motif(inner_h, inner_w, area)
        else:
            motif_plan = _place_side_motif(inner_h, inner_w, area)
        if motif_plan is None:
            return None
        motif_mask, side = motif_plan

        placements.append(
            {
                "branch": branch,
                "top": top,
                "left": left,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_color": frame_color,
                "motif_color": motif_color,
                "motif_mask": motif_mask,
                "side": side,
            }
        )

    inp = np.zeros((grid_h, grid_w), dtype=int)
    for spec in placements:
        _render_frame(inp, spec["top"], spec["left"], spec["inner_h"], spec["inner_w"], spec["frame_color"])
        inner = inp[spec["top"] + 1 : spec["top"] + 1 + spec["inner_h"], spec["left"] + 1 : spec["left"] + 1 + spec["inner_w"]]
        inner[spec["motif_mask"]] = spec["motif_color"]

    out = _expected_from_plan(inp, placements)
    if not _validate_plan(inp, out, placements):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the frame-routing puzzle")
