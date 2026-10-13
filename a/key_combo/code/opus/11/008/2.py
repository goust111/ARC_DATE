from __future__ import annotations

from collections import deque
import random

import numpy as np


DIM_OPTIONS = ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3))
COLOR_OPTIONS = (2, 3, 6, 7, 8, 9)
GRID_RANGE = (18, 25)
MAX_ATTEMPTS = 500
PAIR_TRIES = 800


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g) if int(v) != 0]
    assert len(colors) == 2, "expected exactly two non-black colors"

    rects: list[dict] = []
    for color in sorted(colors):
        comps = _components(g == color)
        assert len(comps) == 1, "each color must form exactly one rectangle"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert int(coords.shape[0]) == h * w, "foreground objects must be solid rectangles"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be monochrome"
        rects.append({"color": color, "r": r0, "c": c0, "h": h, "w": w})

    dims = {(rect["h"], rect["w"]) for rect in rects}
    assert len(dims) == 2, "the two input rectangles must have different sizes"
    return rects


def _body_cells(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> set[tuple[int, int]]:
    grid_h, grid_w = shape
    rr0 = max(0, int(r))
    cc0 = max(0, int(c))
    rr1 = min(grid_h, int(r + h))
    cc1 = min(grid_w, int(c + w))
    if rr0 >= rr1 or cc0 >= cc1:
        return set()
    return {(rr, cc) for rr in range(rr0, rr1) for cc in range(cc0, cc1)}


def _halo_cells(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> set[tuple[int, int]]:
    grid_h, grid_w = shape
    cells: set[tuple[int, int]] = set()
    for cc in range(c, c + w):
        if 0 <= r - 1 < grid_h and 0 <= cc < grid_w:
            cells.add((r - 1, cc))
        if 0 <= r + h < grid_h and 0 <= cc < grid_w:
            cells.add((r + h, cc))
    for rr in range(r, r + h):
        if 0 <= rr < grid_h and 0 <= c - 1 < grid_w:
            cells.add((rr, c - 1))
        if 0 <= rr < grid_h and 0 <= c + w < grid_w:
            cells.add((rr, c + w))
    return cells


def _enumerate_instances(rect: dict, shape: tuple[int, int], source: int) -> list[dict]:
    h = int(rect["h"])
    w = int(rect["w"])
    r0 = int(rect["r"])
    c0 = int(rect["c"])
    grid_h, grid_w = shape

    instances: list[dict] = []
    for step, direction in ((0, 0),):
        body = _body_cells(shape, r0, c0, h, w)
        instances.append(
            {
                "source": int(source),
                "color": int(rect["color"]),
                "step": int(step),
                "direction": int(direction),
                "r": int(r0),
                "c": int(c0),
                "h": h,
                "w": w,
                "body": body,
                "halo": _halo_cells(shape, r0, c0, h, w),
                "clipped": False,
            }
        )

    for direction in (-1, 1):
        step = 1
        while True:
            rr = r0 + step * h
            cc = c0 + direction * step * w
            if rr >= grid_h or cc >= grid_w or cc + w <= 0:
                break
            body = _body_cells(shape, rr, cc, h, w)
            if not body:
                break
            instances.append(
                {
                    "source": int(source),
                    "color": int(rect["color"]),
                    "step": int(step),
                    "direction": int(direction),
                    "r": int(rr),
                    "c": int(cc),
                    "h": h,
                    "w": w,
                    "body": body,
                    "halo": _halo_cells(shape, rr, cc, h, w),
                    "clipped": len(body) < h * w,
                }
            )
            step += 1
    return instances


def _solve_from_rectangles(rects: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    grid_h, grid_w = shape
    out = np.zeros((grid_h, grid_w), dtype=int)
    body_owner = np.full((grid_h, grid_w), -1, dtype=int)
    body_color = np.zeros((grid_h, grid_w), dtype=int)
    halo_sources = np.zeros((grid_h, grid_w), dtype=np.uint8)
    halo_copy_sources = np.zeros((grid_h, grid_w), dtype=np.uint8)
    source_colors = {idx: int(rect["color"]) for idx, rect in enumerate(rects)}
    instances_by_source: dict[int, list[dict]] = {}

    for source, rect in enumerate(rects):
        instances = _enumerate_instances(rect, shape, source)
        instances_by_source[source] = instances
        for inst in instances:
            for r, c in inst["body"]:
                owner = int(body_owner[r, c])
                assert owner in (-1, source), "stamp bodies from different sources must not overlap"
                body_owner[r, c] = int(source)
                body_color[r, c] = int(inst["color"])
            bit = np.uint8(1 << source)
            for r, c in inst["halo"]:
                halo_sources[r, c] |= bit
                if inst["step"] > 0:
                    halo_copy_sources[r, c] |= bit

    out[body_owner >= 0] = body_color[body_owner >= 0]
    collision_mask = (body_owner < 0) & (halo_sources == 3)
    source0_only = (body_owner < 0) & (halo_sources == 1)
    source1_only = (body_owner < 0) & (halo_sources == 2)
    out[source0_only] = source_colors[0]
    out[source1_only] = source_colors[1]
    out[collision_mask] = 0

    body_halo_contacts = ((body_owner == 0) & ((halo_sources & 2) != 0)) | (
        (body_owner == 1) & ((halo_sources & 1) != 0)
    )

    collision_any_copy = collision_mask & (halo_copy_sources != 0)
    collision_both_copy = collision_mask & (halo_copy_sources == 3)
    lower_original_bottom = max(int(rect["r"] + rect["h"] - 1) for rect in rects)
    lower_collision = np.zeros_like(collision_mask)
    lower_collision[lower_original_bottom + 1 :, :] = collision_mask[lower_original_bottom + 1 :, :]

    stats = {
        "instances_by_source": instances_by_source,
        "collision_mask": collision_mask,
        "collision_count": int(collision_mask.sum()),
        "collision_any_copy_count": int(collision_any_copy.sum()),
        "collision_both_copy_count": int(collision_both_copy.sum()),
        "collisions_below_originals": int(lower_collision.sum()),
        "body_halo_contact_count": int(body_halo_contacts.sum()),
        "visible_halo_counts": {
            0: int(source0_only.sum()),
            1: int(source1_only.sum()),
        },
        "raw_halo_pixels": int(((halo_sources != 0) & (body_owner < 0)).sum()),
        "any_clipped_copy": any(
            inst["step"] > 0 and inst["clipped"]
            for insts in instances_by_source.values()
            for inst in insts
        ),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each input rectangle down both diagonals, add a one-pixel halo around every copy, and erase cross-source halo collisions.

    Input structure:
    - The background is black.
    - There are exactly two non-overlapping solid axis-aligned rectangles of different sizes and different colors.
    - Each color appears in a single 4-connected component, and those original rectangles are the only input objects.

    Rule:
    - Keep each original rectangle and stamp the same rectangle at offsets `(k*h, -k*w)` and `(k*h, +k*w)` for `k=1,2,...`, clipping any out-of-bounds part and stopping once a whole stamp is outside the grid.
    - For every kept rectangle copy, including the original, add a one-pixel outward outline in the same color by extending its four sides; the four new corner pixels stay black.
    - After all halos are collected, any non-body pixel covered by halo cells from both source rectangles is erased back to black, while solid rectangle pixels themselves always remain.

    Key insight:
    - The two input rectangles act as both sprites and stride vectors, so their diagonal stamp lattices generate halo interference points that can only be resolved by tracking halo provenance before applying the final erasure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects = _extract_rectangles(g)
    out, _stats = _solve_from_rectangles(rects, g.shape)
    return out


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _direction_counts(instances: list[dict]) -> dict[int, int]:
    counts = {-1: 0, 1: 0}
    for inst in instances:
        if inst["step"] > 0:
            counts[int(inst["direction"])] += 1
    return counts


def _candidate_origins(grid_h: int, grid_w: int, h: int, w: int, min_steps: int) -> list[tuple[int, int]]:
    origins: list[tuple[int, int]] = []
    for r in range(1, grid_h - h):
        for c in range(0, grid_w - w + 1):
            probe = {"color": 1, "r": r, "c": c, "h": h, "w": w}
            counts = _direction_counts(_enumerate_instances(probe, (grid_h, grid_w), 0))
            if counts[-1] >= min_steps and counts[1] >= min_steps:
                if counts[-1] <= 5 and counts[1] <= 5:
                    origins.append((r, c))
    random.shuffle(origins)
    return origins


def _build_input(rect_specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rect_specs:
        r = int(rect["r"])
        c = int(rect["c"])
        h = int(rect["h"])
        w = int(rect["w"])
        grid[r : r + h, c : c + w] = int(rect["color"])
    return grid


def _valid_example(grid: np.ndarray, require_clipped: bool, require_body_contact: bool) -> bool:
    rects = _extract_rectangles(grid)
    try:
        out, stats = _solve_from_rectangles(rects, grid.shape)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if stats["collision_count"] < 2:
        return False
    if stats["collision_count"] > max(18, stats["raw_halo_pixels"] // 2):
        return False
    if stats["collision_any_copy_count"] == 0:
        return False
    if stats["collisions_below_originals"] == 0:
        return False
    if min(stats["visible_halo_counts"].values()) < 4:
        return False
    if require_clipped and not stats["any_clipped_copy"]:
        return False
    if require_body_contact and stats["body_halo_contact_count"] == 0:
        return False

    for source, instances in stats["instances_by_source"].items():
        counts = _direction_counts(instances)
        if counts[-1] < 2 or counts[1] < 2:
            return False
        if len(instances) < 5:
            return False
        if source not in (0, 1):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_RANGE)
    grid_w = random.randint(*GRID_RANGE)
    (h1, w1), (h2, w2) = random.sample(DIM_OPTIONS, 2)
    color1, color2 = random.sample(COLOR_OPTIONS, 2)
    min_steps = 2 if random.random() < 0.7 else 3
    require_clipped = random.random() < 0.6
    require_body_contact = random.random() < 0.3

    origins1 = _candidate_origins(grid_h, grid_w, h1, w1, min_steps)
    origins2 = _candidate_origins(grid_h, grid_w, h2, w2, min_steps)
    if not origins1 or not origins2:
        return None

    tries = 0
    for r1, c1 in origins1[: min(len(origins1), 80)]:
        box1 = (r1, c1, r1 + h1 - 1, c1 + w1 - 1)
        for r2, c2 in random.sample(origins2, k=min(len(origins2), 80)):
            tries += 1
            if tries > PAIR_TRIES:
                return None
            box2 = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)
            if _box_overlap(box1, box2):
                continue

            rects = [
                {"color": color1, "r": r1, "c": c1, "h": h1, "w": w1},
                {"color": color2, "r": r2, "c": c2, "h": h2, "w": w2},
            ]
            grid = _build_input(rects, (grid_h, grid_w))
            if _valid_example(grid, require_clipped=require_clipped, require_body_contact=require_body_contact):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate diagonal-halo collision example")
