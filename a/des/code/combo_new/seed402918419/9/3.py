from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]

RING_COUNT_CHOICES = [3, 4]
INNER_HEIGHT_CHOICES = [3, 5]
INNER_WIDTH_CHOICES = [3, 5, 7]
STEP_RANGE = (1, 6)

SEPARATION = 2
CODE_BOX_GAP = 1
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _box_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return _box_h(box) * _box_w(box)


def _perimeter_length(box: tuple[int, int, int, int]) -> int:
    return 2 * (_box_h(box) + _box_w(box)) - 4


def _perimeter_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _is_rectangular_border_component(comp: list[tuple[int, int]]) -> bool:
    box = _bbox(comp)
    if _box_h(box) < 3 or _box_w(box) < 3:
        return False
    border = set(_perimeter_coords(box))
    comp_set = set(comp)
    if not comp_set.issubset(border):
        return False
    return len(comp_set) >= _perimeter_length(box) - 1


def _grey_border_boxes(grid: np.ndarray, grey: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for comp in _components(grid == int(grey)):
        if _is_rectangular_border_component(comp):
            boxes.append(_bbox(comp))
    return boxes


def _choose_ring_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    by_center: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for box in boxes:
        by_center.setdefault((box[0] + box[2], box[1] + box[3]), []).append(box)

    best: list[tuple[int, int, int, int]] | None = None
    best_score: tuple[int, int] | None = None
    for group in by_center.values():
        if len(group) < 3:
            continue
        ordered = sorted(group, key=_box_area)
        ok = True
        for inner, outer in zip(ordered, ordered[1:]):
            if not (
                inner[0] - outer[0] == 2
                and inner[1] - outer[1] == 2
                and outer[2] - inner[2] == 2
                and outer[3] - inner[3] == 2
            ):
                ok = False
                break
        if not ok:
            continue
        score = (len(ordered), _box_area(ordered[-1]))
        if best is None or score > best_score:
            best = ordered
            best_score = score
    if best is None:
        raise AssertionError("failed to identify the concentric ring group")
    return best


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    counts = Counter(map(int, g.flat))
    grey = None
    for color, _ in counts.most_common():
        if int(color) != int(bg):
            grey = int(color)
            break
    if grey is None:
        raise AssertionError("expected at least one foreground color")

    grey_boxes = _grey_border_boxes(g, grey)
    rings = _choose_ring_boxes(grey_boxes)
    ring_set = set(rings)

    steps: dict[int, int] = {}
    for box in grey_boxes:
        if box in ring_set:
            continue
        if _box_h(box) < 5 or _box_w(box) < 5:
            continue
        r0, c0, r1, c1 = box
        key_color = int(g[r0 + 1, c0 + 1])
        if key_color in (bg, grey):
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero((interior != bg) & (interior != grey))) - 1
        if count < 1:
            continue
        if key_color in steps:
            raise AssertionError("duplicate encoding color")
        steps[key_color] = count

    seeds = []
    for box in rings:
        hits = [(r, c, int(g[r, c])) for r, c in _perimeter_coords(box) if int(g[r, c]) not in (bg, grey)]
        if len(hits) > 1:
            raise AssertionError("each ring can contain at most one seed")
        if hits:
            r, c, color = hits[0]
            seeds.append({"box": box, "pos": (r, c), "color": int(color)})

    if not seeds:
        raise AssertionError("expected at least one ring seed")
    if len(steps) != len({int(seed["color"]) for seed in seeds}):
        raise AssertionError("seed/code color mismatch")
    for seed in seeds:
        if int(seed["color"]) not in steps:
            raise AssertionError("missing encoding box for a seed color")

    return {
        "bg": int(bg),
        "grey": int(grey),
        "rings": rings,
        "seeds": seeds,
        "steps": steps,
    }


def _apply_rule(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    grey = int(scene["grey"])
    steps = scene["steps"]
    for seed in scene["seeds"]:
        box = seed["box"]
        start = tuple(seed["pos"])
        color = int(seed["color"])
        perimeter = _perimeter_coords(box)
        start_idx = perimeter.index(start)
        step_count = int(steps[color])
        for offset in range(1, step_count + 1):
            r, c = perimeter[(start_idx + offset) % len(perimeter)]
            if int(out[r, c]) == grey:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each ring seed clockwise by a length encoded in its matching side code box.

    Input structure:
    - The background is black.
    - A central target-like structure contains several concentric one-pixel grey rectangular rings with one-pixel black gaps between them.
    - Some rings contain a single colored seed pixel that replaces one grey border cell.
    - Separate small grey rectangular code boxes sit to the right or below; each box contains a colored pixel in its inner top-left corner and several maroon hole markers elsewhere inside.

    Rule:
    - Find the concentric ring group by its shared center and fixed two-cell insets between successive grey rectangles.
    - For each seed color, locate the code box whose inner top-left color matches the seed color.
    - Count the non-background marker pixels inside that code box, excluding the color key pixel itself.
    - Starting from the seed position, move clockwise along the seed's ring and recolor that many consecutive grey border cells with the seed color.
    - Keep the original seeds and all code boxes unchanged.

    Key insight:
    - The color match links two otherwise separate structures: a seed on a ring and a code box. The marker count inside the matched box is the hidden parameter that determines how far the seed spreads along its geometric orbit.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _apply_rule(np.asarray(grid, dtype=int), scene)


def _draw_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    for r, c in _perimeter_coords(box):
        grid[r, c] = int(color)


def _random_code_inner_dims(rng: random.Random) -> tuple[int, int]:
    return rng.choice([(3, 3), (3, 4), (4, 3), (4, 4)])


def _place_code_box(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    color: int,
    count: int,
    rng: random.Random,
) -> None:
    outer = (top, left, top + inner_h + 1, left + inner_w + 1)
    _draw_border(grid, outer, GREY)
    grid[top + 1, left + 1] = int(color)
    cells = [(r, c) for r in range(top + 1, top + 1 + inner_h) for c in range(left + 1, left + 1 + inner_w)]
    cells.remove((top + 1, left + 1))
    markers = rng.sample(cells, count)
    for r, c in markers:
        grid[r, c] = MAROON


def _build_case(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    ring_count = rng.choice(RING_COUNT_CHOICES)
    inner_h = rng.choice(INNER_HEIGHT_CHOICES)
    inner_w = rng.choice(INNER_WIDTH_CHOICES)
    outer_h = inner_h + 4 * (ring_count - 1)
    outer_w = inner_w + 4 * (ring_count - 1)

    seed_count = rng.randint(2, ring_count)
    seeded_rings = sorted(rng.sample(range(ring_count), seed_count))
    seed_colors = rng.sample(SEED_COLORS, seed_count)
    steps = {color: rng.randint(*STEP_RANGE) for color in seed_colors}

    specs = []
    for color in seed_colors:
        inner_box_h, inner_box_w = _random_code_inner_dims(rng)
        specs.append(
            {
                "color": int(color),
                "count": int(steps[color]),
                "inner_h": int(inner_box_h),
                "inner_w": int(inner_box_w),
                "outer_h": int(inner_box_h + 2),
                "outer_w": int(inner_box_w + 2),
            }
        )

    side = rng.choice(["right", "bottom"])

    if side == "right":
        strip_w = max(spec["outer_w"] for spec in specs)
        stack_h = sum(spec["outer_h"] for spec in specs) + CODE_BOX_GAP * (len(specs) - 1)
        h_min = max(outer_h, stack_h) + 2
        w_min = outer_w + SEPARATION + strip_w + 2
        if h_min > 30 or w_min > 30:
            return None
        h = rng.randint(h_min, min(30, h_min + 4))
        w = rng.randint(w_min, min(30, w_min + 4))
        slack_x = w - (outer_w + SEPARATION + strip_w)
        ring_left = rng.randint(1, slack_x - 1)
        frame_left = ring_left + outer_w + SEPARATION
        ring_top = rng.randint(1, h - outer_h - 1)
        frame_top = rng.randint(1, h - stack_h - 1)
    else:
        strip_h = max(spec["outer_h"] for spec in specs)
        row_w = sum(spec["outer_w"] for spec in specs) + CODE_BOX_GAP * (len(specs) - 1)
        h_min = outer_h + SEPARATION + strip_h + 2
        w_min = max(outer_w, row_w) + 2
        if h_min > 30 or w_min > 30:
            return None
        h = rng.randint(h_min, min(30, h_min + 4))
        w = rng.randint(w_min, min(30, w_min + 4))
        slack_y = h - (outer_h + SEPARATION + strip_h)
        ring_top = rng.randint(1, slack_y - 1)
        frame_top = ring_top + outer_h + SEPARATION
        ring_left = rng.randint(1, w - outer_w - 1)
        frame_left = rng.randint(1, w - row_w - 1)

    grid = np.full((h, w), BG, dtype=int)

    ring_boxes = []
    for idx in range(ring_count):
        r0 = ring_top + 2 * (ring_count - 1 - idx)
        c0 = ring_left + 2 * (ring_count - 1 - idx)
        box = (r0, c0, r0 + inner_h + 4 * idx - 1, c0 + inner_w + 4 * idx - 1)
        ring_boxes.append(box)
        _draw_border(grid, box, GREY)

    for ring_idx, color in zip(seeded_rings, seed_colors):
        perimeter = _perimeter_coords(ring_boxes[ring_idx])
        count = int(steps[color])
        if rng.random() < 0.5:
            start_idx = rng.randrange(len(perimeter))
        else:
            start_idx = rng.randrange(len(perimeter) - count)
        sr, sc = perimeter[start_idx]
        grid[sr, sc] = int(color)

    placed_specs = specs[:]
    rng.shuffle(placed_specs)
    if side == "right":
        y = frame_top
        for spec in placed_specs:
            x = frame_left + rng.randint(0, strip_w - spec["outer_w"])
            _place_code_box(grid, y, x, spec["inner_h"], spec["inner_w"], spec["color"], spec["count"], rng)
            y += spec["outer_h"] + CODE_BOX_GAP
    else:
        x = frame_left
        for spec in placed_specs:
            y = frame_top + rng.randint(0, strip_h - spec["outer_h"])
            _place_code_box(grid, y, x, spec["inner_h"], spec["inner_w"], spec["color"], spec["count"], rng)
            x += spec["outer_w"] + CODE_BOX_GAP

    meta = {
        "ring_count": int(ring_count),
        "seed_colors": set(int(color) for color in seed_colors),
        "steps": {int(color): int(count) for color, count in steps.items()},
    }
    return grid, meta


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_case(rng)
        if built is None:
            continue
        grid, meta = built
        try:
            scene = _parse_scene(grid)
            if len(scene["rings"]) != int(meta["ring_count"]):
                continue
            found_colors = {int(seed["color"]) for seed in scene["seeds"]}
            if found_colors != meta["seed_colors"]:
                continue
            if any(int(scene["steps"].get(color, -1)) != int(count) for color, count in meta["steps"].items()):
                continue
            out = _apply_rule(grid, scene)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("generate_input: exhausted attempts")
