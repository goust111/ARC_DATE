from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

ARROW_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "right": {
        0: {(0, 0), (-1, -1), (1, -1)},
        1: {(0, 0), (-1, -1), (1, -1), (0, -1)},
        2: {(0, 0), (-1, -1), (1, -1), (0, -1), (0, -2)},
    },
    "left": {
        0: {(0, 0), (-1, 1), (1, 1)},
        1: {(0, 0), (-1, 1), (1, 1), (0, 1)},
        2: {(0, 0), (-1, 1), (1, 1), (0, 1), (0, 2)},
    },
    "down": {
        0: {(0, 0), (-1, -1), (-1, 1)},
        1: {(0, 0), (-1, -1), (-1, 1), (-1, 0)},
        2: {(0, 0), (-1, -1), (-1, 1), (-1, 0), (-2, 0)},
    },
    "up": {
        0: {(0, 0), (1, -1), (1, 1)},
        1: {(0, 0), (1, -1), (1, 1), (1, 0)},
        2: {(0, 0), (1, -1), (1, 1), (1, 0), (2, 0)},
    },
}

ARROW_LOOKUP: dict[frozenset[tuple[int, int]], list[dict]] = {}
for _direction, _tail_map in ARROW_PATTERNS.items():
    for _tail_len, _pattern in _tail_map.items():
        _min_r = min(r for r, _ in _pattern)
        _min_c = min(c for _, c in _pattern)
        _norm = frozenset((r - _min_r, c - _min_c) for r, c in _pattern)
        ARROW_LOOKUP.setdefault(_norm, []).append({
            "direction": _direction,
            "tail_len": int(_tail_len),
            "tip_offset": (-_min_r, -_min_c),
        })


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_arrow(coords: list[tuple[int, int]]) -> dict | None:
    if len(coords) not in (3, 4, 5):
        return None
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = frozenset((r - min_r, c - min_c) for r, c in coords)
    infos = ARROW_LOOKUP.get(norm)
    if infos is None:
        return None
    candidates = []
    for info in infos:
        tip = (int(min_r + info["tip_offset"][0]), int(min_c + info["tip_offset"][1]))
        candidates.append(
            {
                "tip": tip,
                "direction": info["direction"],
                "drdc": ARROW_DIRS[info["direction"]],
                "tail_len": int(info["tail_len"]),
            }
        )
    return {"coords": coords, "candidates": candidates}


def _classify_rectangle(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if len(coords) != h * w:
        return None
    if set(coords) != {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}:
        return None
    return {"r0": int(r0), "c0": int(c0), "h": int(h), "w": int(w)}


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    bg = _get_bg(grid)
    arrows: list[dict] = []
    rectangles: list[dict] = []
    for comp in _find_components(grid, bg):
        coords = comp["coords"]
        arrow = _classify_arrow(coords)
        if arrow is not None:
            arrow["color"] = int(comp["color"])
            arrow["direction"] = arrow["candidates"][0]["direction"] if len(arrow["candidates"]) == 1 else None
            arrows.append(arrow)
            continue
        rect = _classify_rectangle(coords)
        if rect is not None:
            rect["color"] = int(comp["color"])
            rect["coords"] = coords
            rectangles.append(rect)
    rectangles.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["color"]))
    for idx, rect in enumerate(rectangles):
        rect["id"] = int(idx)
    return arrows, rectangles


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    return not (
        ar0 + ah <= br0
        or br0 + bh <= ar0
        or ac0 + aw <= bc0
        or bc0 + bw <= ac0
    )


def _ray_target(
    tip: tuple[int, int],
    direction: tuple[int, int],
    rect_grid: np.ndarray,
) -> tuple[int, int] | None:
    h, w = rect_grid.shape
    r, c = tip
    dr, dc = direction
    dist = 0
    while True:
        r += dr
        c += dc
        dist += 1
        if not (0 <= r < h and 0 <= c < w):
            return None
        rect_id = int(rect_grid[r, c])
        if rect_id >= 0:
            return rect_id, dist


def _max_shift(
    rect_id: int,
    direction: tuple[int, int],
    states: dict[int, dict],
    shape: tuple[int, int],
) -> int:
    h_grid, w_grid = shape
    dr, dc = direction
    rect = states[rect_id]
    r0, c0, rh, rw = rect["r0"], rect["c0"], rect["h"], rect["w"]
    steps = 0
    while True:
        nr0 = r0 + dr * (steps + 1)
        nc0 = c0 + dc * (steps + 1)
        if nr0 < 0 or nc0 < 0 or nr0 + rh > h_grid or nc0 + rw > w_grid:
            return steps
        candidate = (nr0, nc0, rh, rw)
        blocked = False
        for other_id, other in states.items():
            if other_id == rect_id:
                continue
            other_box = (other["r0"], other["c0"], other["h"], other["w"])
            if _bbox_overlap(candidate, other_box):
                blocked = True
                break
        if blocked:
            return steps
        steps += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each ray-targeted rectangle until it hits the edge or another rectangle, and color the swept trail with the arrow color.

    Input structure:
    - The background is black.
    - Foreground objects are small monochrome arrows and larger solid monochrome rectangles.
    - Each arrow points in one cardinal direction; its tip ray meets the edge of exactly one rectangle.

    Rule:
    - Match every arrow to the first rectangle reached from its tip along the arrow direction.
    - Process pushes in increasing arrow-to-target distance, which guarantees that arrows aimed at the same rectangle act from near to far.
    - Move the rectangle along that direction until the next step would leave the grid or overlap another rectangle, then fill every pre-final swept position with the arrow's color.
    - Erase the arrows and keep all rectangles at their final positions.

    Key insight:
    - The arrows supply two pieces of information at once: direction chooses which rectangle moves, while arrow color labels the rectangle's motion trail instead of the rectangle's own color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    arrows, rectangles = _parse_objects(g)

    out = np.full_like(g, bg)
    if not rectangles:
        return out

    rect_grid = np.full(g.shape, -1, dtype=int)
    states: dict[int, dict] = {}
    for rect in rectangles:
        rect_id = int(rect["id"])
        rect_grid[rect["r0"] : rect["r0"] + rect["h"], rect["c0"] : rect["c0"] + rect["w"]] = rect_id
        states[rect_id] = {
            "color": int(rect["color"]),
            "r0": int(rect["r0"]),
            "c0": int(rect["c0"]),
            "h": int(rect["h"]),
            "w": int(rect["w"]),
        }

    events: list[dict] = []
    for arrow in arrows:
        matches = []
        for candidate in arrow["candidates"]:
            match = _ray_target(candidate["tip"], candidate["drdc"], rect_grid)
            if match is None:
                continue
            rect_id, dist = match
            matches.append((int(dist), int(rect_id), candidate))
        if not matches:
            continue
        dist, rect_id, candidate = min(
            matches,
            key=lambda item: (item[0], item[1], item[2]["tip"][0], item[2]["tip"][1]),
        )
        events.append(
            {
                "rect_id": int(rect_id),
                "distance": int(dist),
                "direction": candidate["drdc"],
                "color": int(arrow["color"]),
                "tip": candidate["tip"],
            }
        )

    events.sort(key=lambda event: (event["distance"], event["rect_id"], event["tip"][0], event["tip"][1]))

    for event in events:
        rect_id = int(event["rect_id"])
        dr, dc = event["direction"]
        steps = _max_shift(rect_id, (dr, dc), states, g.shape)
        rect = states[rect_id]
        if steps > 0:
            for step in range(steps):
                rr0 = rect["r0"] + dr * step
                cc0 = rect["c0"] + dc * step
                out[rr0 : rr0 + rect["h"], cc0 : cc0 + rect["w"]] = int(event["color"])
            rect["r0"] += dr * steps
            rect["c0"] += dc * steps

    for rect in states.values():
        out[rect["r0"] : rect["r0"] + rect["h"], rect["c0"] : rect["c0"] + rect["w"]] = int(rect["color"])

    return out


COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SCENE_KINDS = ("single_edge", "single_blocker", "double_edge")
TAIL_CHOICES = (0, 1)
MAX_GENERATION_ATTEMPTS = 400


def _arrow_cells(
    tip: tuple[int, int],
    direction: str,
    tail_len: int,
) -> list[tuple[int, int]]:
    tr, tc = tip
    return [(tr + dr, tc + dc) for dr, dc in sorted(ARROW_PATTERNS[direction][tail_len])]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != 0:
            return False
    for r, c in cells:
        grid[r, c] = int(color)
    return True


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    if np.any(grid[r0 : r0 + h, c0 : c0 + w] != 0):
        return False
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)
    return True


def _scene_cost(kind: str) -> int:
    return {"single_edge": 2, "single_blocker": 3, "double_edge": 3}[kind]


def _sample_scene_spec(rng: random.Random) -> dict:
    kind = rng.choice(SCENE_KINDS)
    rect_h = rng.randint(2, 4)
    rect_w = rng.randint(2, 5)
    direction = rng.choice(("left", "right"))
    spec = {
        "kind": kind,
        "rect_h": rect_h,
        "rect_w": rect_w,
        "direction": direction,
        "band_h": rect_h + 4,
    }
    if kind == "single_edge":
        spec["arrow_dist"] = rng.randint(2, 5)
        spec["arrow_tail"] = rng.choice(TAIL_CHOICES)
    elif kind == "single_blocker":
        spec["arrow_dist"] = rng.randint(2, 5)
        spec["arrow_tail"] = rng.choice(TAIL_CHOICES)
        spec["blocker_w"] = rng.randint(2, 4)
        spec["gap"] = rng.randint(2, 5)
    else:
        near_dist = rng.randint(2, 4)
        far_dist = near_dist + rng.randint(4, 5)
        spec["near_dist"] = near_dist
        spec["far_dist"] = far_dist
        spec["near_tail"] = rng.choice(TAIL_CHOICES)
        spec["far_tail"] = rng.choice(TAIL_CHOICES)
    return spec


def _place_horizontal_scene(
    grid: np.ndarray,
    band_top: int,
    spec: dict,
    colors: list[int],
    rng: random.Random,
) -> bool:
    kind = spec["kind"]
    rh = spec["rect_h"]
    rw = spec["rect_w"]
    band_h = spec["band_h"]
    direction = spec["direction"]
    r0 = band_top + 2
    w_grid = grid.shape[1]

    if direction == "right":
        if kind == "single_edge":
            min_c0 = spec["arrow_dist"] + 2
            max_c0 = w_grid - rw - 2
        elif kind == "single_blocker":
            min_c0 = spec["arrow_dist"] + 2
            max_c0 = w_grid - rw - spec["gap"] - spec["blocker_w"]
        else:
            min_c0 = spec["far_dist"] + 2
            max_c0 = w_grid - rw - 2
        if min_c0 > max_c0:
            return False
        c0 = rng.randint(min_c0, max_c0)
    else:
        if kind == "single_edge":
            min_c0 = 1
            max_c0 = w_grid - rw - spec["arrow_dist"] - 2
        elif kind == "single_blocker":
            min_c0 = spec["gap"] + spec["blocker_w"]
            max_c0 = w_grid - rw - spec["arrow_dist"] - 2
        else:
            min_c0 = 1
            max_c0 = w_grid - rw - spec["far_dist"] - 2
        if min_c0 > max_c0:
            return False
        c0 = rng.randint(min_c0, max_c0)

    target_color = int(colors[0])
    if not _paint_rect(grid, r0, c0, rh, rw, target_color):
        return False

    if kind == "single_blocker":
        blocker_color = int(colors[-1])
        if direction == "right":
            blocker_c0 = c0 + rw + spec["gap"]
        else:
            blocker_c0 = c0 - spec["gap"] - spec["blocker_w"]
        if not _paint_rect(grid, r0, blocker_c0, rh, spec["blocker_w"], blocker_color):
            return False

    if kind == "single_edge":
        tip_row = rng.randint(r0, r0 + rh - 1)
        if direction == "right":
            tip = (tip_row, c0 - spec["arrow_dist"])
        else:
            tip = (tip_row, c0 + rw - 1 + spec["arrow_dist"])
        arrow = _arrow_cells(tip, direction, spec["arrow_tail"])
        return _paint_cells(grid, arrow, int(colors[1]))

    if kind == "single_blocker":
        tip_row = rng.randint(r0, r0 + rh - 1)
        if direction == "right":
            tip = (tip_row, c0 - spec["arrow_dist"])
        else:
            tip = (tip_row, c0 + rw - 1 + spec["arrow_dist"])
        arrow = _arrow_cells(tip, direction, spec["arrow_tail"])
        return _paint_cells(grid, arrow, int(colors[1]))

    tip_rows = [r0, r0 + rh - 1]
    if direction == "right":
        near_tip = (tip_rows[0], c0 - spec["near_dist"])
        far_tip = (tip_rows[1], c0 - spec["far_dist"])
    else:
        near_tip = (tip_rows[0], c0 + rw - 1 + spec["near_dist"])
        far_tip = (tip_rows[1], c0 + rw - 1 + spec["far_dist"])
    near_arrow = _arrow_cells(near_tip, direction, spec["near_tail"])
    far_arrow = _arrow_cells(far_tip, direction, spec["far_tail"])
    if not _paint_cells(grid, near_arrow, int(colors[1])):
        return False
    if not _paint_cells(grid, far_arrow, int(colors[2])):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_scenes = rng.randint(2, 3)
    specs = [_sample_scene_spec(rng) for _ in range(n_scenes)]
    gaps = [rng.randint(1, 2) for _ in range(n_scenes - 1)]
    top_margin = rng.randint(0, 1)
    bottom_margin = rng.randint(0, 1)
    height = top_margin + sum(spec["band_h"] for spec in specs) + sum(gaps) + bottom_margin
    width = rng.randint(20, 30)
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    palette = COLOR_POOL[:]
    rng.shuffle(palette)
    needed_colors = sum(_scene_cost(spec["kind"]) for spec in specs)
    if needed_colors > len(palette):
        return None

    cursor = top_margin
    palette_idx = 0
    for i, spec in enumerate(specs):
        cost = _scene_cost(spec["kind"])
        colors = palette[palette_idx : palette_idx + cost]
        palette_idx += cost
        if not _place_horizontal_scene(grid, cursor, spec, colors, rng):
            return None
        cursor += spec["band_h"]
        if i < len(gaps):
            cursor += gaps[i]

    if rng.random() < 0.5:
        grid = grid.T.copy()

    arrows, rectangles = _parse_objects(grid)
    if len(arrows) < 2 or len(rectangles) < n_scenes:
        return None
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
