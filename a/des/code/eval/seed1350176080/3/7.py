from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Each arrow is defined by offsets relative to its tip cell.
ARROW_FROM_TIP = {
    "up": ((0, 0), (1, -1), (1, 0), (1, 1)),
    "down": ((0, 0), (-1, -1), (-1, 0), (-1, 1)),
    "left": ((0, 0), (-1, 1), (0, 1), (1, 1)),
    "right": ((0, 0), (-1, -1), (0, -1), (1, -1)),
}

SPECIAL_MIXES = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

INPUT_ARROW_COLORS = (RED, BLUE, YELLOW, CYAN)
SPECIAL_PAIRS = ((RED, BLUE), (RED, YELLOW), (BLUE, YELLOW))
OTHER_PAIRS = ((RED, CYAN), (BLUE, CYAN), (YELLOW, CYAN))
MULTI_TRIPLES = (
    (RED, BLUE, YELLOW),
    (RED, BLUE, CYAN),
    (RED, YELLOW, CYAN),
    (BLUE, YELLOW, CYAN),
)

TARGET_MODES = ("single", "pair_special", "pair_other", "multi", "unhit")
HIT_MODES = TARGET_MODES[:-1]

ZONE_SIZE = 11
TARGET_LOCAL_CHOICES = (4, 5)
LAYOUTS = ((1, 2), (1, 3), (2, 2), (2, 3))
SIDE_TO_DIRECTION = {"left": "right", "right": "left", "top": "down", "bottom": "up"}
SIDE_NAMES = tuple(SIDE_TO_DIRECTION)

OUTER_PAD_RANGE = (1, 3)
ZONE_GAP_RANGE = (2, 4)
TARGET_COUNT_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _arrow_signatures() -> dict[frozenset[tuple[int, int]], tuple[str, tuple[int, int]]]:
    out: dict[frozenset[tuple[int, int]], tuple[str, tuple[int, int]]] = {}
    for direction, offsets in ARROW_FROM_TIP.items():
        min_r = min(dr for dr, _dc in offsets)
        min_c = min(dc for _dr, dc in offsets)
        norm = frozenset((dr - min_r, dc - min_c) for dr, dc in offsets)
        tip_offset = (-min_r, -min_c)
        out[norm] = (direction, tip_offset)
    return out


ARROW_SIGNATURES = _arrow_signatures()


def _is_target_block(comp: dict[str, object]) -> bool:
    if int(comp["area"]) != 4:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    return (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2


def _identify_arrow(comp: dict[str, object]) -> dict[str, object] | None:
    if int(comp["area"]) != 4:
        return None
    r0, c0, _r1, _c1 = comp["bbox"]
    coords = comp["coords"]
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    info = ARROW_SIGNATURES.get(norm)
    if info is None:
        return None

    direction, tip_offset = info
    tip = (int(r0 + tip_offset[0]), int(c0 + tip_offset[1]))
    dr, dc = DIR_VECTORS[direction]
    return {
        "color": int(comp["color"]),
        "coords": coords,
        "tip": tip,
        "direction": direction,
        "vector": (dr, dc),
    }


def _detect_scene(grid: np.ndarray) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    bg = _dominant_color(grid)
    targets: list[dict[str, object]] = []
    arrows: list[dict[str, object]] = []

    for comp in _components_by_color(grid, bg):
        if _is_target_block(comp):
            targets.append(comp)
            continue
        arrow = _identify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)

    return bg, targets, arrows


def _resolve_hit_colors(colors: list[int]) -> int | None:
    if not colors:
        return None
    if len(colors) == 1:
        return int(colors[0])
    if len(colors) > 2:
        return MAROON

    unique = set(int(x) for x in colors)
    if len(unique) == 1:
        return int(colors[0])

    mixed = SPECIAL_MIXES.get(frozenset(unique))
    if mixed is not None:
        return mixed
    return MAROON


def _target_hits(
    shape: tuple[int, int], targets: list[dict[str, object]], arrows: list[dict[str, object]]
) -> dict[int, list[int]]:
    h, w = shape
    target_lookup: dict[tuple[int, int], int] = {}
    for idx, target in enumerate(targets):
        for r, c in target["coords"]:
            target_lookup[(int(r), int(c))] = idx

    hits: dict[int, list[int]] = defaultdict(list)
    for arrow in arrows:
        r, c = arrow["tip"]
        dr, dc = arrow["vector"]
        r += dr
        c += dc
        while 0 <= r < h and 0 <= c < w:
            target_id = target_lookup.get((r, c))
            if target_id is not None:
                hits[target_id].append(int(arrow["color"]))
                break
            r += dr
            c += dc
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the first 2x2 target square hit by each T-shaped arrow, mixing colors on collisions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    _bg, targets, arrows = _detect_scene(g)
    if not targets or not arrows:
        return out

    hits = _target_hits((h, w), targets, arrows)
    for idx, target in enumerate(targets):
        color = _resolve_hit_colors(hits.get(idx, []))
        if color is None:
            continue
        coords = target["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def _object_cells_from_tip(direction: str, tip: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = tip
    return [(tr + dr, tc + dc) for dr, dc in ARROW_FROM_TIP[direction]]


def _can_place(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False

    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(occupied[r, c]):
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if bool(occupied[nr, nc]):
                return False
    return True


def _paint(grid: np.ndarray, occupied: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
        occupied[r, c] = True


def _sample_target_modes(count: int) -> list[str]:
    modes = list(TARGET_MODES)
    random.shuffle(modes)
    picked = modes[: min(count, len(modes))]
    while len(picked) < count:
        picked.append(random.choice(TARGET_MODES))
    random.shuffle(picked)
    if all(mode == "unhit" for mode in picked):
        picked[random.randrange(count)] = random.choice(HIT_MODES)
    return picked


def _colors_for_mode(mode: str) -> list[int]:
    if mode == "single":
        return [random.choice(INPUT_ARROW_COLORS)]
    if mode == "pair_special":
        return list(random.choice(SPECIAL_PAIRS))
    if mode == "pair_other":
        return list(random.choice(OTHER_PAIRS))
    if mode == "multi":
        return list(random.choice(MULTI_TRIPLES))
    return []


def _arrow_options_for_side(target_r: int, target_c: int, side: str) -> list[tuple[str, tuple[int, int]]]:
    options: list[tuple[str, tuple[int, int]]] = []
    if side == "left":
        for row in (target_r, target_r + 1):
            for dist in (2, 3):
                options.append(("right", (row, target_c - dist)))
    elif side == "right":
        for row in (target_r, target_r + 1):
            for dist in (2, 3):
                options.append(("left", (row, target_c + 1 + dist)))
    elif side == "top":
        for col in (target_c, target_c + 1):
            for dist in (2, 3):
                options.append(("down", (target_r - dist, col)))
    else:
        assert side == "bottom"
        for col in (target_c, target_c + 1):
            for dist in (2, 3):
                options.append(("up", (target_r + 1 + dist, col)))
    random.shuffle(options)
    return options


def _slot_origins(count: int) -> tuple[int, int, list[tuple[int, int]]]:
    layouts = [layout for layout in LAYOUTS if layout[0] * layout[1] >= count]
    rows, cols = random.choice(layouts)
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    return rows, cols, random.sample(slots, count)


def _gen_attempt() -> np.ndarray | None:
    target_count = random.randint(*TARGET_COUNT_RANGE)
    zone_rows, zone_cols, chosen_slots = _slot_origins(target_count)
    pad_top = random.randint(*OUTER_PAD_RANGE)
    pad_bottom = random.randint(*OUTER_PAD_RANGE)
    pad_left = random.randint(*OUTER_PAD_RANGE)
    pad_right = random.randint(*OUTER_PAD_RANGE)
    row_gap = random.randint(*ZONE_GAP_RANGE)
    col_gap = random.randint(*ZONE_GAP_RANGE)

    height = pad_top + pad_bottom + zone_rows * ZONE_SIZE + (zone_rows - 1) * row_gap
    width = pad_left + pad_right + zone_cols * ZONE_SIZE + (zone_cols - 1) * col_gap
    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    modes = _sample_target_modes(target_count)
    random.shuffle(chosen_slots)
    target_infos: list[tuple[int, int, str]] = []

    for mode, (slot_r, slot_c) in zip(modes, chosen_slots):
        base_r = pad_top + slot_r * (ZONE_SIZE + row_gap)
        base_c = pad_left + slot_c * (ZONE_SIZE + col_gap)
        target_r = base_r + random.choice(TARGET_LOCAL_CHOICES)
        target_c = base_c + random.choice(TARGET_LOCAL_CHOICES)
        target_cells = [
            (target_r, target_c),
            (target_r, target_c + 1),
            (target_r + 1, target_c),
            (target_r + 1, target_c + 1),
        ]
        if not _can_place(target_cells, occupied):
            return None
        _paint(grid, occupied, target_cells, GRAY)
        target_infos.append((target_r, target_c, mode))

    for target_r, target_c, mode in target_infos:
        colors = _colors_for_mode(mode)
        if not colors:
            continue
        sides = random.sample(SIDE_NAMES, len(colors))
        for color, side in zip(colors, sides):
            placed = False
            for direction, tip in _arrow_options_for_side(target_r, target_c, side):
                arrow_cells = _object_cells_from_tip(direction, tip)
                if not _can_place(arrow_cells, occupied):
                    continue
                _paint(grid, occupied, arrow_cells, color)
                placed = True
                break
            if not placed:
                return None

    bg, targets, arrows = _detect_scene(grid)
    if bg != BLACK:
        return None
    if len(targets) != target_count:
        return None
    if not arrows:
        return None

    hits = _target_hits(grid.shape, targets, arrows)
    if not any(hits.values()):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
