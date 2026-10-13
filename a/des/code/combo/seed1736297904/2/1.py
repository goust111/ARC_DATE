import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_open_frame(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    comp = {(int(r), int(c)) for r, c in coords}
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    if len(comp) != len(border) - 1 or not comp.issubset(border):
        return None

    missing = list(border - comp)
    if len(missing) != 1:
        return None
    gr, gc = missing[0]
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None

    if gr == r0:
        direction = "up"
    elif gr == r1:
        direction = "down"
    elif gc == c0:
        direction = "left"
    elif gc == c1:
        direction = "right"
    else:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (gr, gc),
        "direction": direction,
        "area": len(comp),
    }


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    infos_by_color: dict[int, list[dict]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        infos: list[dict] = []
        for comp in _components(grid == color):
            info = _parse_open_frame(comp)
            if info is not None:
                info["color"] = int(color)
                infos.append(info)
        if infos:
            infos_by_color[int(color)] = infos

    assert infos_by_color, "expected at least one open rectangular frame"

    frame_color = max(
        infos_by_color,
        key=lambda color: (len(infos_by_color[color]), sum(info["area"] for info in infos_by_color[color])),
    )
    frames = sorted(infos_by_color[frame_color], key=lambda info: info["bbox"])
    return int(frame_color), frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each inner sprite outward through its frame gap and stop it where the same-color guide dot indicates.

    Input structure:
    - Background is black.
    - Several same-color 1-pixel rectangular frames each have exactly one 1-pixel gap on one side.
    - Each frame contains a monochrome sprite of some other color.
    - Outside the frames, each sprite color has one same-color guide dot on the gap's outward ray.

    Rule:
    - Detect the gap side to get the projection direction.
    - Match the frame's inner sprite to the outside guide dot of the same color on that ray.
    - Translate the whole sprite along that axis so the sprite row/column nearest the gap aligns with the guide dot's row/column.
    - Erase the original inner sprite and keep frames and guide dots unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frame_color, frames = _find_frames(g, bg)

    interior_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        direction = frame["direction"]

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        sprite_colors = [int(v) for v in np.unique(interior) if int(v) not in (bg, frame_color)]
        assert len(sprite_colors) == 1, "each frame must contain exactly one sprite color"
        sprite_color = sprite_colors[0]

        sprite_coords = np.argwhere(g == sprite_color)
        inside = (
            (sprite_coords[:, 0] > r0)
            & (sprite_coords[:, 0] < r1)
            & (sprite_coords[:, 1] > c0)
            & (sprite_coords[:, 1] < c1)
        )
        sprite_coords = sprite_coords[inside]
        assert len(sprite_coords) > 0, "sprite must have at least one pixel inside its frame"

        outside_color = np.argwhere((g == sprite_color) & ~interior_mask)
        dot_candidates: list[tuple[int, int]] = []
        for rr, cc in outside_color:
            rr = int(rr)
            cc = int(cc)
            if direction == "up" and cc == gap_c and rr < r0:
                dot_candidates.append((rr, cc))
            elif direction == "down" and cc == gap_c and rr > r1:
                dot_candidates.append((rr, cc))
            elif direction == "left" and rr == gap_r and cc < c0:
                dot_candidates.append((rr, cc))
            elif direction == "right" and rr == gap_r and cc > c1:
                dot_candidates.append((rr, cc))
        assert len(dot_candidates) == 1, "expected one same-color guide dot on the outward ray"
        dot_r, dot_c = dot_candidates[0]

        out[sprite_coords[:, 0], sprite_coords[:, 1]] = bg

        if direction == "up":
            delta_r = dot_r - int(np.min(sprite_coords[:, 0]))
            delta_c = 0
        elif direction == "down":
            delta_r = dot_r - int(np.max(sprite_coords[:, 0]))
            delta_c = 0
        elif direction == "left":
            delta_r = 0
            delta_c = dot_c - int(np.min(sprite_coords[:, 1]))
        else:
            delta_r = 0
            delta_c = dot_c - int(np.max(sprite_coords[:, 1]))

        moved = sprite_coords + np.array([delta_r, delta_c], dtype=int)
        assert moved[:, 0].min() >= 0 and moved[:, 0].max() < g.shape[0]
        assert moved[:, 1].min() >= 0 and moved[:, 1].max() < g.shape[1]
        out[moved[:, 0], moved[:, 1]] = sprite_color

    return out


BG = 0
GRID_MIN = 22
GRID_MAX = 30
FRAME_COUNT_MIN = 2
FRAME_COUNT_MAX = 4
FRAME_SIZE_MIN = 6
FRAME_SIZE_MAX = 9
SPRITE_BOX_MAX = 4
SPRITE_AREA_MAX = 6
CLEARANCE_MAX = 4
PLACEMENT_PAD = 1
OUTER_MARGIN = 1
MAX_GENERATE_ATTEMPTS = 400
ALL_DIRECTIONS = ("up", "down", "left", "right")


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _random_sprite_mask(interior_h: int, interior_w: int) -> np.ndarray:
    box_h = random.randint(2, min(SPRITE_BOX_MAX, interior_h))
    box_w = random.randint(2, min(SPRITE_BOX_MAX, interior_w))
    area = random.randint(2, min(SPRITE_AREA_MAX, box_h * box_w))
    return _grow_connected_mask(box_h, box_w, area)


def _choose_gap(direction: str, sprite_coords: list[tuple[int, int]], frame_h: int, frame_w: int) -> tuple[int, int]:
    rows = [r for r, _ in sprite_coords]
    cols = [c for _, c in sprite_coords]
    if direction in ("up", "down"):
        edge_row = min(rows) if direction == "up" else max(rows)
        blocked = {c for r, c in sprite_coords if r == edge_row}
        candidates = [c for c in range(min(cols), max(cols) + 1) if 1 <= c <= frame_w - 2]
        preferred = [c for c in candidates if c not in blocked]
        gap_c = random.choice(preferred or candidates)
        return (0 if direction == "up" else frame_h - 1, gap_c)

    edge_col = min(cols) if direction == "left" else max(cols)
    blocked = {r for r, c in sprite_coords if c == edge_col}
    candidates = [r for r in range(min(rows), max(rows) + 1) if 1 <= r <= frame_h - 2]
    preferred = [r for r in candidates if r not in blocked]
    gap_r = random.choice(preferred or candidates)
    return (gap_r, 0 if direction == "left" else frame_w - 1)


def _make_local_spec(frame_h: int, frame_w: int, sprite_color: int, direction: str) -> dict:
    sprite_mask = _random_sprite_mask(frame_h - 2, frame_w - 2)
    sh, sw = sprite_mask.shape
    top = random.randint(1, frame_h - 1 - sh)
    left = random.randint(1, frame_w - 1 - sw)
    sprite_coords = [(top + int(r), left + int(c)) for r, c in np.argwhere(sprite_mask)]

    gap = _choose_gap(direction, sprite_coords, frame_h, frame_w)
    rows = [r for r, _ in sprite_coords]
    cols = [c for _, c in sprite_coords]
    clearance = random.randint(0, CLEARANCE_MAX)

    if direction == "up":
        shift = (-(max(rows) + 1 + clearance), 0)
        dot = (min(rows) + shift[0], gap[1])
    elif direction == "down":
        shift = (frame_h + clearance - min(rows), 0)
        dot = (max(rows) + shift[0], gap[1])
    elif direction == "left":
        shift = (0, -(max(cols) + 1 + clearance))
        dot = (gap[0], min(cols) + shift[1])
    else:
        shift = (0, frame_w + clearance - min(cols))
        dot = (gap[0], max(cols) + shift[1])

    projected_coords = [(r + shift[0], c + shift[1]) for r, c in sprite_coords]
    reserve_rows = [0, frame_h - 1, dot[0], *[r for r, _ in projected_coords]]
    reserve_cols = [0, frame_w - 1, dot[1], *[c for _, c in projected_coords]]
    reserve_bbox = (
        min(reserve_rows),
        min(reserve_cols),
        max(reserve_rows),
        max(reserve_cols),
    )

    return {
        "direction": direction,
        "sprite_color": int(sprite_color),
        "gap": gap,
        "sprite_coords": sprite_coords,
        "projected_coords": projected_coords,
        "dot": dot,
        "reserve_bbox": reserve_bbox,
    }


def _place_specs(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda idx: (
            -(specs[idx]["reserve_bbox"][2] - specs[idx]["reserve_bbox"][0] + 1)
            * (specs[idx]["reserve_bbox"][3] - specs[idx]["reserve_bbox"][1] + 1)
        ),
    )

    for idx in order:
        spec = specs[idx]
        min_r, min_c, max_r, max_c = spec["reserve_bbox"]
        r_lo = OUTER_MARGIN - min_r
        r_hi = height - OUTER_MARGIN - 1 - max_r
        c_lo = OUTER_MARGIN - min_c
        c_hi = width - OUTER_MARGIN - 1 - max_c
        if r_lo > r_hi or c_lo > c_hi:
            return None

        candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        random.shuffle(candidates)
        placed = False
        for base_r, base_c in candidates:
            rr0 = max(0, base_r + min_r - PLACEMENT_PAD)
            rr1 = min(height - 1, base_r + max_r + PLACEMENT_PAD)
            cc0 = max(0, base_c + min_c - PLACEMENT_PAD)
            cc1 = min(width - 1, base_c + max_c + PLACEMENT_PAD)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            spec["origin"] = (base_r, base_c)
            placed = True
            break
        if not placed:
            return None

    return specs


def _render_input(height: int, width: int, frame_h: int, frame_w: int, frame_color: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        base_r, base_c = spec["origin"]
        grid[base_r, base_c : base_c + frame_w] = frame_color
        grid[base_r + frame_h - 1, base_c : base_c + frame_w] = frame_color
        grid[base_r : base_r + frame_h, base_c] = frame_color
        grid[base_r : base_r + frame_h, base_c + frame_w - 1] = frame_color

        gap_r, gap_c = spec["gap"]
        grid[base_r + gap_r, base_c + gap_c] = BG

        for r, c in spec["sprite_coords"]:
            grid[base_r + r, base_c + c] = spec["sprite_color"]

        dot_r, dot_c = spec["dot"]
        grid[base_r + dot_r, base_c + dot_c] = spec["sprite_color"]

    return grid


def _render_expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        base_r, base_c = spec["origin"]
        for r, c in spec["sprite_coords"]:
            out[base_r + r, base_c + c] = BG
        for r, c in spec["projected_coords"]:
            out[base_r + r, base_c + c] = spec["sprite_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    frame_h = random.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX)
    frame_w = random.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX)
    n_frames = random.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)

    palette = [c for c in range(1, 10)]
    random.shuffle(palette)
    frame_color = palette[0]
    sprite_colors = palette[1 : 1 + n_frames]
    if len(sprite_colors) < n_frames:
        return None

    directions = random.sample(ALL_DIRECTIONS, k=n_frames)
    specs = [
        _make_local_spec(frame_h, frame_w, sprite_color, direction)
        for sprite_color, direction in zip(sprite_colors, directions)
    ]
    placed_specs = _place_specs(height, width, specs)
    if placed_specs is None:
        return None

    grid = _render_input(height, width, frame_h, frame_w, frame_color, placed_specs)
    expected = _render_expected_output(grid, placed_specs)

    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
