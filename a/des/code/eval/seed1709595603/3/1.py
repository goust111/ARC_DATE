import random
from typing import Iterable

import numpy as np


GEN_RETRIES = 300

# The generated data only uses this small palette order so the flip relation can
# be inferred from examples instead of relying on ARC color ids.
COLOR_ORDER = [7, 2, 8, 3, 1, 4, 6, 9, 5, 0]
COLOR_RANK = {color: idx for idx, color in enumerate(COLOR_ORDER)}
PALETTE = COLOR_ORDER[:4]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _bg_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_points(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = gap
    r0, c0, r1, c1 = bbox
    on_top = r == r0 and c0 < c < c1
    on_bottom = r == r1 and c0 < c < c1
    on_left = c == c0 and r0 < r < r1
    on_right = c == c1 and r0 < r < r1
    sides = [on_top, on_bottom, on_left, on_right]
    if sum(sides) != 1:
        return None
    if on_top:
        return "up"
    if on_bottom:
        return "down"
    if on_left:
        return "left"
    return "right"


def _should_flip(pattern_color: int, border_color: int) -> bool:
    return COLOR_RANK[pattern_color] < COLOR_RANK[border_color]


def _flip_points(
    points: list[tuple[int, int]],
    direction: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    if direction in ("up", "down"):
        return [(r0 + r1 - r, c) for r, c in points]
    return [(r, c0 + c1 - c) for r, c in points]


def _translate_outside(
    points: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    pr0, pc0, pr1, pc1 = _bbox(points)
    if direction == "up":
        dr, dc = r0 - 1 - pr1, 0
    elif direction == "down":
        dr, dc = r1 + 1 - pr0, 0
    elif direction == "left":
        dr, dc = 0, c0 - 1 - pc1
    else:
        dr, dc = 0, c1 + 1 - pc0
    return [(r + dr, c + dc) for r, c in points]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    objects: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue

            perimeter = set(_perimeter_points(r0, c0, r1, c1))
            comp_set = set(comp)
            if not comp_set.issubset(perimeter):
                continue

            missing = list(perimeter - comp_set)
            if len(missing) != 1:
                continue
            gap = missing[0]
            if int(g[gap]) != bg:
                continue

            direction = _gap_direction(gap, (r0, c0, r1, c1))
            if direction is None:
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = sorted(
                int(v) for v in np.unique(inner) if int(v) not in {bg, color}
            )
            if len(inner_colors) != 1:
                continue

            pattern_color = inner_colors[0]
            pattern_points = [
                (r0 + 1 + rr, c0 + 1 + cc)
                for rr, cc in zip(*np.where(inner == pattern_color))
            ]
            if not pattern_points:
                continue

            objects.append(
                {
                    "frame_color": color,
                    "pattern_color": pattern_color,
                    "frame_bbox": (r0, c0, r1, c1),
                    "gap": gap,
                    "direction": direction,
                    "pattern_points": pattern_points,
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for obj in _extract_objects(g):
        pattern_points = list(obj["pattern_points"])
        if _should_flip(obj["pattern_color"], obj["frame_color"]):
            pattern_points = _flip_points(pattern_points, obj["direction"])
        shifted = _translate_outside(pattern_points, obj["frame_bbox"], obj["direction"])

        r0, c0, r1, c1 = obj["frame_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        for r, c in shifted:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = obj["pattern_color"]

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(pts[0])}
    stack = [tuple(pts[0])]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_pattern(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, max_h)
        w = rng.randint(2, max_w)
        mask = np.array([[rng.random() < 0.5 for _ in range(w)] for _ in range(h)], dtype=bool)
        if mask.sum() < 3 or mask.sum() == h * w:
            continue

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not _mask_connected(mask):
            continue
        if np.array_equal(mask, mask[::-1, :]):
            continue
        if np.array_equal(mask, mask[:, ::-1]):
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric pattern")


def _expand_rect(
    rect: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = rect
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _paint_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_object(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    spec: tuple[int, int, str],
) -> bool:
    pattern_color, frame_color, direction = spec
    gh, gw = grid.shape

    for _ in range(80):
        frame_h = rng.randint(6, 9)
        frame_w = rng.randint(6, 9)
        inner_h = frame_h - 2
        inner_w = frame_w - 2
        pattern = _random_pattern(rng, inner_h - 2, inner_w - 2)
        ph, pw = pattern.shape

        top_in = 1 + rng.randint(0, inner_h - ph - 2)
        left_in = 1 + rng.randint(0, inner_w - pw - 2)

        if direction == "up":
            out_rect_rel = (-ph, left_in, -1, left_in + pw - 1)
        elif direction == "down":
            out_rect_rel = (frame_h, left_in, frame_h + ph - 1, left_in + pw - 1)
        elif direction == "left":
            out_rect_rel = (top_in, -pw, top_in + ph - 1, -1)
        else:
            out_rect_rel = (top_in, frame_w, top_in + ph - 1, frame_w + pw - 1)

        all_r0 = min(0, out_rect_rel[0]) - 1
        all_c0 = min(0, out_rect_rel[1]) - 1
        all_r1 = max(frame_h - 1, out_rect_rel[2]) + 1
        all_c1 = max(frame_w - 1, out_rect_rel[3]) + 1

        min_r = -all_r0
        min_c = -all_c0
        max_r = gh - 1 - all_r1
        max_c = gw - 1 - all_c1
        if min_r > max_r or min_c > max_c:
            continue

        r0 = rng.randint(min_r, max_r)
        c0 = rng.randint(min_c, max_c)
        frame_rect = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        out_rect = (
            r0 + out_rect_rel[0],
            c0 + out_rect_rel[1],
            r0 + out_rect_rel[2],
            c0 + out_rect_rel[3],
        )

        temp = np.zeros_like(occupied)
        _paint_rect(temp, _expand_rect(frame_rect, 1, grid.shape))
        _paint_rect(temp, _expand_rect(out_rect, 1, grid.shape))
        if np.any(temp & occupied):
            continue

        gap_offset = {
            "up": (0, rng.randint(1, frame_w - 2)),
            "down": (frame_h - 1, rng.randint(1, frame_w - 2)),
            "left": (rng.randint(1, frame_h - 2), 0),
            "right": (rng.randint(1, frame_h - 2), frame_w - 1),
        }[direction]
        gap = (r0 + gap_offset[0], c0 + gap_offset[1])

        for cc in range(c0, c0 + frame_w):
            if (r0, cc) != gap:
                grid[r0, cc] = frame_color
            if (r0 + frame_h - 1, cc) != gap:
                grid[r0 + frame_h - 1, cc] = frame_color
        for rr in range(r0 + 1, r0 + frame_h - 1):
            if (rr, c0) != gap:
                grid[rr, c0] = frame_color
            if (rr, c0 + frame_w - 1) != gap:
                grid[rr, c0 + frame_w - 1] = frame_color

        pr0 = r0 + 1 + top_in
        pc0 = c0 + 1 + left_in
        for rr, cc in zip(*np.where(pattern)):
            grid[pr0 + rr, pc0 + cc] = pattern_color

        occupied |= temp
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    bg = rng.choice([0, 4, 5, 6, 9])
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    num_objects = rng.randint(4, 5)
    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)
    if num_objects == 5:
        directions.append(rng.choice(list(DIRS)))

    specs: list[tuple[int, int, str]] = [
        (7, 2, directions[0]),
        (2, 8, directions[1]),
        (8, 2, directions[2]),
        (3, 8, directions[3]),
    ]
    if num_objects == 5:
        extras = [(8, 3), (2, 7), (7, 8), (3, 2)]
        pattern_color, frame_color = rng.choice(extras)
        specs.append((pattern_color, frame_color, directions[4]))
    rng.shuffle(specs)

    for spec in specs:
        if not _place_object(rng, grid, occupied, spec):
            return None

    objects = _extract_objects(grid)
    if len(objects) != num_objects:
        return None

    directions_found = {obj["direction"] for obj in objects}
    if len(directions_found) < 4:
        return None

    flip_values = {
        _should_flip(obj["pattern_color"], obj["frame_color"]) for obj in objects
    }
    if flip_values != {False, True}:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
