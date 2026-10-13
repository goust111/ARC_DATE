from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
PLACE_RETRIES = 120
GRID_MIN = 18
GRID_MAX = 30
INNER_H_RANGE = (5, 8)
INNER_W_RANGE = (5, 8)
RECT_COUNT_CHOICES = [2, 3, 4]
RECT_COUNT_WEIGHTS = [2, 3, 2]
MAX_EXTRA_PROBES = 4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(pts)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _rectangle_ring(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _ray_geometry(
    inner_r0: int,
    inner_c0: int,
    inner_r1: int,
    inner_c1: int,
    anchor_pos: tuple[int, int],
) -> tuple[str, tuple[int, int], tuple[int, int], int]:
    ar, ac = anchor_pos
    outer_r0, outer_c0 = inner_r0 - 1, inner_c0 - 1
    outer_r1, outer_c1 = inner_r1 + 1, inner_c1 + 1

    if ar == outer_r0 and inner_c0 <= ac <= inner_c1:
        return "top", (inner_r0, ac), (1, 0), inner_r1 - inner_r0 + 1
    if ar == outer_r1 and inner_c0 <= ac <= inner_c1:
        return "bottom", (inner_r1, ac), (-1, 0), inner_r1 - inner_r0 + 1
    if ac == outer_c0 and inner_r0 <= ar <= inner_r1:
        return "left", (ar, inner_c0), (0, 1), inner_c1 - inner_c0 + 1
    if ac == outer_c1 and inner_r0 <= ar <= inner_r1:
        return "right", (ar, inner_c1), (0, -1), inner_c1 - inner_c0 + 1
    raise AssertionError("anchor must sit on exactly one non-corner border side")


def _is_isolated_probe(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    h, w = grid.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
            return False
    return True


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    rectangles: list[dict[str, object]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()

    for component in _components_mask(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in component):
            continue

        inner_r0, inner_c0, inner_r1, inner_c1 = _bbox(component)
        outer_r0, outer_c0, outer_r1, outer_c1 = inner_r0 - 1, inner_c0 - 1, inner_r1 + 1, inner_c1 + 1
        if outer_r0 < 0 or outer_c0 < 0 or outer_r1 >= h or outer_c1 >= w:
            continue

        box_key = (outer_r0, outer_c0, outer_r1, outer_c1)
        if box_key in seen_boxes:
            continue

        ring = _rectangle_ring(outer_r0, outer_c0, outer_r1, outer_c1)
        ring_colors = Counter(int(grid[r, c]) for r, c in ring)
        if len(ring_colors) != 2:
            continue

        border_color, border_count = ring_colors.most_common(1)[0]
        anchor_colors = [color for color, count in ring_colors.items() if count == 1 and color != bg]
        if border_color == bg or border_count != len(ring) - 1 or len(anchor_colors) != 1:
            continue

        anchor_color = int(anchor_colors[0])
        anchor_pos = next((r, c) for r, c in ring if int(grid[r, c]) == anchor_color)
        try:
            side, ray_start, ray_dir, capacity = _ray_geometry(
                inner_r0, inner_c0, inner_r1, inner_c1, anchor_pos
            )
        except AssertionError:
            continue

        window = grid[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1]
        if np.any(window == border_color):
            continue

        probes: list[tuple[int, int, int]] = []
        valid = True
        for r in range(inner_r0, inner_r1 + 1):
            for c in range(inner_c0, inner_c1 + 1):
                value = int(grid[r, c])
                if value == bg:
                    continue
                if not _is_isolated_probe(grid, bg, r, c):
                    valid = False
                    break
                probes.append((r, c, value))
            if not valid:
                break
        if not valid:
            continue

        seen_boxes.add(box_key)
        rectangles.append(
            {
                "outer_bbox": box_key,
                "inner_bbox": (inner_r0, inner_c0, inner_r1, inner_c1),
                "border_color": int(border_color),
                "anchor_color": anchor_color,
                "anchor_pos": anchor_pos,
                "side": side,
                "ray_start": ray_start,
                "ray_dir": ray_dir,
                "capacity": int(capacity),
                "probes": probes,
            }
        )

    rectangles.sort(key=lambda rect: tuple(int(v) for v in rect["outer_bbox"]))
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow inward rays from border anchors using the global count of same-colored probes.

    Input structure:
    - The dominant color is the background.
    - Several hollow rectangles are drawn with a shared border color, each enclosing a background interior.
    - Every rectangle border contains exactly one uniquely colored anchor pixel, and the interiors contain isolated single-pixel probes whose colors match anchor colors.

    Rule:
    - Detect each enclosed rectangle and collect every probe inside every rectangle.
    - Match each probe to the rectangle whose border anchor has the same color, counting matches globally across rectangles.
    - For each anchor, infer the ray direction from the side that contains the anchor, then draw an inward ray from the adjacent interior cell with length equal to the global probe count of that color.

    Key insight:
    - The probes are distributed counting tokens: their locations do not matter except for color-based ownership, while the anchor's border side alone determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    rectangles = _find_rectangles(g, bg)
    assert rectangles, "expected at least one anchored rectangle"

    anchor_by_color: dict[int, dict[str, object]] = {}
    probe_counts: Counter[int] = Counter()
    for rect in rectangles:
        anchor_color = int(rect["anchor_color"])
        assert anchor_color not in anchor_by_color, "anchor colors must be unique"
        anchor_by_color[anchor_color] = rect
        for _r, _c, probe_color in rect["probes"]:
            probe_counts[int(probe_color)] += 1

    assert set(probe_counts).issubset(set(anchor_by_color)), "every probe color must match an anchor color"

    out = g.copy()
    for anchor_color, rect in anchor_by_color.items():
        length = int(probe_counts.get(anchor_color, 0))
        start_r, start_c = rect["ray_start"]
        dr, dc = rect["ray_dir"]
        inner_r0, inner_c0, inner_r1, inner_c1 = rect["inner_bbox"]
        for step in range(length):
            rr = int(start_r + dr * step)
            cc = int(start_c + dc * step)
            if not (inner_r0 <= rr <= inner_r1 and inner_c0 <= cc <= inner_c1):
                break
            if int(out[rr, cc]) in (bg, anchor_color):
                out[rr, cc] = anchor_color

    return out


def _rectangles_far_enough(
    existing: list[dict[str, object]], r0: int, c0: int, r1: int, c1: int
) -> bool:
    for rect in existing:
        er0, ec0, er1, ec1 = rect["outer_bbox"]
        if not (r1 + 1 < er0 or er1 + 1 < r0 or c1 + 1 < ec0 or ec1 + 1 < c0):
            return False
    return True


def _choose_anchor_pos(
    rng: random.Random, side: str, outer_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = outer_bbox
    if side == "top":
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0
    if side == "right":
        return rng.randint(r0 + 1, r1 - 1), c1
    raise AssertionError(f"unexpected side {side}")


def _ray_cells(rect: dict[str, object]) -> set[tuple[int, int]]:
    inner_r0, inner_c0, inner_r1, inner_c1 = rect["inner_bbox"]
    start_r, start_c = rect["ray_start"]
    dr, dc = rect["ray_dir"]
    cells = set()
    rr, cc = start_r, start_c
    while inner_r0 <= rr <= inner_r1 and inner_c0 <= cc <= inner_c1:
        cells.add((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _candidate_probe_cells(rect: dict[str, object]) -> list[tuple[int, int]]:
    inner_r0, inner_c0, inner_r1, inner_c1 = rect["inner_bbox"]
    ray_cells = _ray_cells(rect)
    cells: list[tuple[int, int]] = []
    for r in range(inner_r0 + 1, inner_r1):
        for c in range(inner_c0 + 1, inner_c1):
            if (r, c) not in ray_cells:
                cells.append((r, c))
    return cells


def _available_probe_cells(
    candidates: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    for r, c in candidates:
        if (r, c) in occupied:
            continue
        if any((nr, nc) in occupied for nr, nc in _neighbors4(r, c)):
            continue
        valid.append((r, c))
    return valid


def _mandatory_color_indices(rng: random.Random, n_rect: int) -> list[int]:
    assignments = list(range(n_rect))
    if n_rect > 1 and rng.random() < 0.7:
        k = rng.randint(2, n_rect)
        subset = rng.sample(range(n_rect), k)
        rotated = subset[1:] + subset[:1]
        for host_idx, color_idx in zip(subset, rotated):
            assignments[host_idx] = color_idx
    return assignments


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)
    n_rect = rng.choices(RECT_COUNT_CHOICES, weights=RECT_COUNT_WEIGHTS, k=1)[0]

    palette = list(range(10))
    bg = 0
    border_color = rng.choice([color for color in palette if color != bg])
    anchor_colors = rng.sample([color for color in palette if color not in {bg, border_color}], n_rect)
    sides = list(SIDES)
    rng.shuffle(sides)
    sides = sides[:n_rect]

    rectangles: list[dict[str, object]] = []
    for idx in range(n_rect):
        placed = False
        for _ in range(PLACE_RETRIES):
            inner_h = rng.randint(*INNER_H_RANGE)
            inner_w = rng.randint(*INNER_W_RANGE)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 >= grid_h or outer_w + 2 >= grid_w:
                continue
            outer_r0 = rng.randint(1, grid_h - outer_h - 1)
            outer_c0 = rng.randint(1, grid_w - outer_w - 1)
            outer_r1 = outer_r0 + outer_h - 1
            outer_c1 = outer_c0 + outer_w - 1
            if not _rectangles_far_enough(rectangles, outer_r0, outer_c0, outer_r1, outer_c1):
                continue

            inner_bbox = (outer_r0 + 1, outer_c0 + 1, outer_r1 - 1, outer_c1 - 1)
            anchor_pos = _choose_anchor_pos(rng, sides[idx], (outer_r0, outer_c0, outer_r1, outer_c1))
            side, ray_start, ray_dir, capacity = _ray_geometry(*inner_bbox, anchor_pos)
            rect = {
                "index": idx,
                "outer_bbox": (outer_r0, outer_c0, outer_r1, outer_c1),
                "inner_bbox": inner_bbox,
                "anchor_pos": anchor_pos,
                "side": side,
                "ray_start": ray_start,
                "ray_dir": ray_dir,
                "capacity": capacity,
                "anchor_color": int(anchor_colors[idx]),
                "border_color": int(border_color),
            }
            if len(_candidate_probe_cells(rect)) < 2:
                continue
            rectangles.append(rect)
            placed = True
            break
        if not placed:
            return None

    mandatory_colors = _mandatory_color_indices(rng, n_rect)
    total_counts: dict[int, int] = {}
    for rect in rectangles:
        color_idx = int(rect["index"])
        min_count = 1
        max_count = min(int(rect["capacity"]), min_count + MAX_EXTRA_PROBES)
        if max_count < min_count:
            return None
        total_counts[color_idx] = rng.randint(min_count, max_count)

    if all(total_counts[idx] == 1 for idx in total_counts):
        expandable = [idx for idx, rect in enumerate(rectangles) if int(rect["capacity"]) >= 2]
        if expandable:
            bump_idx = rng.choice(expandable)
            total_counts[bump_idx] += 1

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for rect in rectangles:
        outer_r0, outer_c0, outer_r1, outer_c1 = rect["outer_bbox"]
        for r, c in _rectangle_ring(outer_r0, outer_c0, outer_r1, outer_c1):
            grid[r, c] = border_color
        ar, ac = rect["anchor_pos"]
        grid[ar, ac] = int(rect["anchor_color"])

    occupied_probes: set[tuple[int, int]] = set()
    probe_records: list[tuple[int, int, int]] = []
    probe_budget = total_counts.copy()
    candidates_by_rect = {
        int(rect["index"]): _candidate_probe_cells(rect) for rect in rectangles
    }

    for host_idx, color_idx in enumerate(mandatory_colors):
        choices = _available_probe_cells(candidates_by_rect[host_idx], occupied_probes)
        if not choices:
            return None
        cell = rng.choice(choices)
        occupied_probes.add(cell)
        probe_budget[color_idx] -= 1
        probe_records.append((cell[0], cell[1], int(anchor_colors[color_idx])))

    remaining_colors: list[int] = []
    for color_idx in range(n_rect):
        remaining_colors.extend([color_idx] * probe_budget[color_idx])
    rng.shuffle(remaining_colors)

    for color_idx in remaining_colors:
        rect_order = list(range(n_rect))
        if n_rect > 1 and rng.random() < 0.5:
            rect_order = [idx for idx in rect_order if idx != color_idx] + [color_idx]
        rng.shuffle(rect_order)
        placed = False
        for host_idx in rect_order:
            choices = _available_probe_cells(candidates_by_rect[host_idx], occupied_probes)
            if not choices:
                continue
            cell = rng.choice(choices)
            occupied_probes.add(cell)
            probe_records.append((cell[0], cell[1], int(anchor_colors[color_idx])))
            placed = True
            break
        if not placed:
            return None

    per_rect_probe_count = Counter()
    for r, c, color in probe_records:
        grid[r, c] = color
        for rect in rectangles:
            inner_r0, inner_c0, inner_r1, inner_c1 = rect["inner_bbox"]
            if inner_r0 <= r <= inner_r1 and inner_c0 <= c <= inner_c1:
                per_rect_probe_count[int(rect["index"])] += 1
                break

    if any(per_rect_probe_count[idx] == 0 for idx in range(n_rect)):
        return None

    parsed = _find_rectangles(grid, bg)
    if len(parsed) != n_rect:
        return None
    if {int(rect["anchor_color"]) for rect in parsed} != set(anchor_colors):
        return None

    parsed_counts = Counter()
    for rect in parsed:
        for _r, _c, probe_color in rect["probes"]:
            parsed_counts[int(probe_color)] += 1
    expected_counts = Counter({int(anchor_colors[idx]): total_counts[idx] for idx in range(n_rect)})
    if parsed_counts != expected_counts:
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
    raise RuntimeError("generate_input: exhausted retries")
