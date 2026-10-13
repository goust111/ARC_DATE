import math
import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(1, 10))

MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250
GROUP_ATTEMPTS = 200
PARTITION_ATTEMPTS = 200

GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 24
MAX_OUTPUT_WIDTH = 30
MAX_OUTPUT_HEIGHT = 5
MAX_FRAGMENT_AREA = 10

TOTAL_CHOICES = tuple(range(4, 17))
MODE_CHOICES = (
    ("square", "rect"),
    ("flat", "rect"),
    ("square", "flat"),
    ("rect",),
    (),
)


def _closest_factor_pair(area: int) -> tuple[int, int]:
    h = math.isqrt(area)
    while area % h != 0:
        h -= 1
    return area // h, h


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group fragments by color, rebuild each color as the closest-to-square solid
    rectangle with the same area, then pack the rectangles left-to-right by
    increasing area with bottom alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    specs: list[tuple[int, int, int, int]] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) != BG):
        area = int(np.count_nonzero(g == color))
        assert area > 0
        width, height = _closest_factor_pair(area)
        specs.append((area, color, width, height))

    assert specs
    areas = [area for area, _color, _width, _height in specs]
    assert len(set(areas)) == len(areas)

    specs.sort(key=lambda item: item[0])
    out_h = max(height for _area, _color, _width, height in specs)
    out_w = sum(width for _area, _color, width, _height in specs) + len(specs) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for i, (_area, color, width, height) in enumerate(specs):
        out[out_h - height : out_h, col : col + width] = color
        col += width
        if i != len(specs) - 1:
            col += 1

    assert col == out_w
    return out


def _area_kind(area: int) -> str:
    width, height = _closest_factor_pair(area)
    if width == height:
        return "square"
    if height == 1:
        return "flat"
    return "rect"


AREAS_BY_KIND = {
    kind: tuple(area for area in TOTAL_CHOICES if _area_kind(area) == kind)
    for kind in ("square", "flat", "rect")
}


def _packed_output_shape(areas: list[int]) -> tuple[int, int]:
    dims = [_closest_factor_pair(area) for area in sorted(areas)]
    height = max(h for _w, h in dims)
    width = sum(w for w, _h in dims) + len(dims) - 1
    return height, width


def _random_partition(total: int, count: int) -> list[int] | None:
    for _ in range(PARTITION_ATTEMPTS):
        remaining = total
        parts: list[int] = []
        ok = True
        for i in range(count - 1):
            slots_left = count - i - 1
            lo = max(1, remaining - slots_left * MAX_FRAGMENT_AREA)
            hi = min(MAX_FRAGMENT_AREA, remaining - slots_left)
            if lo > hi:
                ok = False
                break
            part = random.randint(lo, hi)
            parts.append(part)
            remaining -= part
        if not ok:
            continue
        if 1 <= remaining <= MAX_FRAGMENT_AREA:
            parts.append(remaining)
            random.shuffle(parts)
            return parts
    return None


def _random_fragment_dims(area: int) -> tuple[int, int]:
    pairs = [(h, area // h) for h in range(1, math.isqrt(area) + 1) if area % h == 0]
    h, w = random.choice(pairs)
    if random.random() < 0.5:
        h, w = w, h
    return int(h), int(w)


def _build_group(color: int, total: int) -> dict | None:
    feasible_counts = [k for k in range(2, 5) if k <= total and total <= k * MAX_FRAGMENT_AREA]
    if not feasible_counts:
        return None

    for _ in range(GROUP_ATTEMPTS):
        count = random.choice(feasible_counts)
        parts = _random_partition(total, count)
        if parts is None:
            continue
        dims = [_random_fragment_dims(part) for part in parts]
        if len(set(dims)) == 1:
            continue
        return {
            "color": int(color),
            "total": int(total),
            "kind": _area_kind(total),
            "out_dims": _closest_factor_pair(total),
            "fragments": [{"area": int(part), "shape": dim} for part, dim in zip(parts, dims)],
        }
    return None


def _pick_totals(group_count: int) -> list[int] | None:
    modes = [mode for mode in MODE_CHOICES if len(mode) <= group_count]
    required = random.choice(modes)
    used: set[int] = set()
    totals: list[int] = []

    for kind in required:
        choices = [area for area in AREAS_BY_KIND[kind] if area not in used]
        if not choices:
            return None
        area = random.choice(choices)
        used.add(area)
        totals.append(area)

    while len(totals) < group_count:
        weighted = (
            [area for area in AREAS_BY_KIND["rect"] if area not in used] * 3
            + [area for area in AREAS_BY_KIND["square"] if area not in used] * 2
            + [area for area in AREAS_BY_KIND["flat"] if area not in used]
        )
        if not weighted:
            return None
        area = random.choice(weighted)
        used.add(area)
        totals.append(area)

    random.shuffle(totals)
    return totals


def _sample_grid_shape(fragments: list[tuple[int, int]]) -> tuple[int, int] | None:
    max_h = max(h for h, _w in fragments)
    max_w = max(w for _h, w in fragments)
    padded_area = sum((h + 2) * (w + 2) for h, w in fragments)
    base = math.ceil(math.sqrt(padded_area * random.uniform(1.15, 1.45)))

    h_lo = max(GRID_SIDE_MIN, max_h + 2, base - 2)
    w_lo = max(GRID_SIDE_MIN, max_w + 2, base - 1)
    if h_lo > GRID_SIDE_MAX or w_lo > GRID_SIDE_MAX:
        return None

    h_hi = min(GRID_SIDE_MAX, h_lo + random.randint(3, 6))
    w_hi = min(GRID_SIDE_MAX, w_lo + random.randint(4, 8))
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    if random.random() < 0.4:
        h, w = w, h
    return int(h), int(w)


def _try_place_groups(groups: list[dict]) -> np.ndarray | None:
    items: list[tuple[int, int, int]] = []
    for group in groups:
        for frag in group["fragments"]:
            h, w = frag["shape"]
            items.append((int(h), int(w), int(group["color"])))

    dims_only = [(h, w) for h, w, _color in items]
    for _ in range(80):
        shape = _sample_grid_shape(dims_only)
        if shape is None:
            return None
        h, w = shape
        grid = np.full((h, w), BG, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        order = items[:]
        random.shuffle(order)
        order.sort(key=lambda item: item[0] * item[1], reverse=True)

        ok = True
        for fh, fw, color in order:
            placed = False
            for _ in range(PLACE_ATTEMPTS):
                r = random.randint(0, h - fh)
                c = random.randint(0, w - fw)
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(h, r + fh + 1)
                c1 = min(w, c + fw + 1)
                if blocked[r0:r1, c0:c1].any():
                    continue
                grid[r : r + fh, c : c + fw] = color
                blocked[r0:r1, c0:c1] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    return None


def _boxes_too_close(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = b1
    s0, t0, s1, t1 = b2
    return not (r1 + 1 < s0 or s1 + 1 < r0 or c1 + 1 < t0 or t1 + 1 < c0)


def _validate_generated(grid: np.ndarray, groups: list[dict]) -> bool:
    present = sorted(int(x) for x in np.unique(grid) if int(x) != BG)
    expected = sorted(int(group["color"]) for group in groups)
    if present != expected:
        return False

    totals = [int(group["total"]) for group in groups]
    if len(set(totals)) != len(totals):
        return False

    out_h, out_w = _packed_output_shape(totals)
    if out_h > MAX_OUTPUT_HEIGHT or out_w > MAX_OUTPUT_WIDTH:
        return False

    all_boxes: list[tuple[int, int, int, int]] = []
    for group in groups:
        color = int(group["color"])
        comps = _components(grid == color)
        if len(comps) != len(group["fragments"]):
            return False
        if sum(comp["area"] for comp in comps) != int(group["total"]):
            return False
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            if comp["area"] != (r1 - r0 + 1) * (c1 - c0 + 1):
                return False
            all_boxes.append(comp["bbox"])

    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            if _boxes_too_close(all_boxes[i], all_boxes[j]):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(MIN_GROUPS, MAX_GROUPS)
    totals = _pick_totals(group_count)
    if totals is None:
        return None

    out_h, out_w = _packed_output_shape(totals)
    if out_h > MAX_OUTPUT_HEIGHT or out_w > MAX_OUTPUT_WIDTH:
        return None

    colors = random.sample(PALETTE, group_count)
    groups: list[dict] = []
    for color, total in zip(colors, totals):
        group = _build_group(color, total)
        if group is None:
            return None
        groups.append(group)

    random.shuffle(groups)
    grid = _try_place_groups(groups)
    if grid is None:
        return None
    if not _validate_generated(grid, groups):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
