from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BACKGROUND = 0
MARKER = 4
BODY_COLORS = (1, 2, 3, 5, 6, 7, 8)
SIZE_CHOICES = (3, 6)
DIRECTION_SLOTS = tuple((sr, sc) for sr in range(3) for sc in range(3))
DIAGONAL_SLOTS = ((0, 0), (0, 2), (2, 0), (2, 2))
CARDINAL_SLOTS = ((0, 1), (1, 0), (1, 2), (2, 1))
ALL_DIMS = ((3, 3), (3, 6), (6, 3), (6, 6))

GEN_HEIGHT_RANGE = (18, 30)
GEN_WIDTH_RANGE = (18, 30)
GEN_OBJECT_COUNT_RANGE = (3, 6)
GEN_MAX_ATTEMPTS = 400
GEN_MAX_COPIES = 3
GEN_MIN_COVERAGE = 0.10
GEN_MAX_COVERAGE = 0.55


@dataclass(frozen=True)
class RectSpec:
    r0: int
    c0: int
    h: int
    w: int
    color: int
    marker_r: int
    marker_c: int
    copies: int
    clipped: bool

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r0 + self.h - 1, self.c0 + self.w - 1

    @property
    def slot_r(self) -> int:
        return (self.marker_r - self.r0) // (self.h // 3)

    @property
    def slot_c(self) -> int:
        return (self.marker_c - self.c0) // (self.w // 3)

    @property
    def direction(self) -> tuple[int, int]:
        return self.slot_r - 1, self.slot_c - 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[tuple[int, int]] = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _rect_intersects_grid(r0: int, c0: int, h: int, w: int, shape: tuple[int, int]) -> bool:
    rows, cols = shape
    return r0 < rows and c0 < cols and r0 + h > 0 and c0 + w > 0


def _paint_rect(out: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(out.shape[0], int(r0) + int(h))
    cc1 = min(out.shape[1], int(c0) + int(w))
    if rr0 < rr1 and cc0 < cc1:
        out[rr0:rr1, cc0:cc1] = int(color)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = []
    for coords in _components(g != BACKGROUND):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, patch.ravel())) - {BACKGROUND})
        assert MARKER in colors, "each object must contain one yellow marker"
        assert len(colors) == 2, "each object must contain exactly one body color plus yellow"
        body = next(color for color in colors if color != MARKER)
        assert body in BODY_COLORS
        assert np.all((patch == body) | (patch == MARKER)), "object bbox must be a solid rectangle"
        assert patch.size == int(coords.shape[0]), "component must fill its bbox"

        h, w = patch.shape
        assert h in SIZE_CHOICES and w in SIZE_CHOICES
        assert h % 3 == 0 and w % 3 == 0

        marker_coords = np.argwhere(patch == MARKER)
        assert marker_coords.shape[0] == 1, "expected exactly one marker per rectangle"
        mr, mc = marker_coords[0]
        mr = int(mr) + r0
        mc = int(mc) + c0
        slot_r = (mr - r0) // (h // 3)
        slot_c = (mc - c0) // (w // 3)
        assert 0 <= slot_r < 3 and 0 <= slot_c < 3

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "h": int(h),
                "w": int(w),
                "color": int(body),
                "marker": (mr, mc),
                "direction": (slot_r - 1, slot_c - 1),
            }
        )

    assert objects, "expected at least one marked rectangle"
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each marked rectangle along the direction encoded by its yellow pixel, erasing the marker in every stamp.

    Input structure:
    - The grid has a black background with 3-5 separated solid axis-aligned rectangles.
    - Each rectangle uses one body color from the non-yellow ARC palette and contains exactly one yellow marker pixel.
    - Every rectangle's width and height are multiples of three, so its bbox splits cleanly into a 3x3 slot grid.

    Rule:
    - For each non-black connected component, treat its bbox as one filled rectangle with one yellow marker cell inside.
    - The marker's 3x3 slot inside that bbox gives a direction: top/middle/bottom map to row steps -1/0/+1 and left/center/right map to column steps -1/0/+1; the center slot means no extra stamps.
    - Replace the marker with the rectangle's body color, then stamp solid copies of that rectangle at offsets k * (height * row_dir, width * col_dir) for k = 1, 2, ... until the next copy is fully outside the grid; clip partial out-of-bounds copies.
    - Process rectangles in row-major order of their top-left corners; later stamps overwrite earlier ones.

    Key insight:
    - One object simultaneously provides the sprite, the motion direction via the marker's local 3x3 slot, and the horizontal/vertical stride via its own width and height.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)

    out = g.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        h = obj["h"]
        w = obj["w"]
        color = obj["color"]
        dr, dc = obj["direction"]

        out[r0 : r1 + 1, c0 : c1 + 1] = color
        if dr == 0 and dc == 0:
            continue

        step_r = dr * h
        step_c = dc * w
        k = 1
        while True:
            nr = r0 + k * step_r
            nc = c0 + k * step_c
            if not _rect_intersects_grid(nr, nc, h, w, out.shape):
                break
            _paint_rect(out, nr, nc, h, w, color)
            k += 1

    return out


def _slot_ranges(start: int, length: int, slot: int) -> tuple[int, int]:
    size = length // 3
    lo = start + slot * size
    hi = lo + size - 1
    return int(lo), int(hi)


def _choose_marker(r0: int, c0: int, h: int, w: int, slot_r: int, slot_c: int, rng: np.random.Generator) -> tuple[int, int]:
    rr0, rr1 = _slot_ranges(r0, h, slot_r)
    cc0, cc1 = _slot_ranges(c0, w, slot_c)
    mr = int(rng.integers(rr0, rr1 + 1))
    mc = int(rng.integers(cc0, cc1 + 1))
    return mr, mc


def _rect_cells(r0: int, c0: int, h: int, w: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    rows, cols = shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(rows, int(r0) + int(h))
    cc1 = min(cols, int(c0) + int(w))
    return {(r, c) for r in range(rr0, rr1) for c in range(cc0, cc1)}


def _trajectory_stats(
    r0: int,
    c0: int,
    h: int,
    w: int,
    slot_r: int,
    slot_c: int,
    shape: tuple[int, int],
) -> dict:
    dr = slot_r - 1
    dc = slot_c - 1
    footprint = _rect_cells(r0, c0, h, w, shape)
    positions = [(int(r0), int(c0))]
    copies = 0
    full_copies = 0
    clipped = False

    if dr == 0 and dc == 0:
        return {
            "footprint": footprint,
            "positions": positions,
            "copies": 0,
            "full_copies": 0,
            "clipped": False,
        }

    step_r = dr * h
    step_c = dc * w
    k = 1
    while True:
        nr = r0 + k * step_r
        nc = c0 + k * step_c
        if not _rect_intersects_grid(nr, nc, h, w, shape):
            break
        positions.append((int(nr), int(nc)))
        footprint |= _rect_cells(nr, nc, h, w, shape)
        copies += 1
        if 0 <= nr and 0 <= nc and nr + h <= shape[0] and nc + w <= shape[1]:
            full_copies += 1
        else:
            clipped = True
        k += 1
        if copies > GEN_MAX_COPIES:
            break

    return {
        "footprint": footprint,
        "positions": positions,
        "copies": int(copies),
        "full_copies": int(full_copies),
        "clipped": bool(clipped),
    }


def _boxes_touch(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], padding: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _render_input(specs: list[RectSpec], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        grid[spec.r0 : spec.r0 + spec.h, spec.c0 : spec.c0 + spec.w] = spec.color
        grid[spec.marker_r, spec.marker_c] = MARKER
    return grid


def _render_output(specs: list[RectSpec], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in sorted(specs, key=lambda s: (s.r0, s.c0)):
        for r0, c0 in _trajectory_stats(spec.r0, spec.c0, spec.h, spec.w, spec.slot_r, spec.slot_c, shape)["positions"]:
            _paint_rect(out, r0, c0, spec.h, spec.w, spec.color)
    return out


def _sample_template(index: int, need_center: bool, need_clipped: bool, rng: np.random.Generator) -> dict:
    if index == 0:
        h, w = random.choice(((3, 6), (6, 3)))
        slot_r, slot_c = random.choice(DIAGONAL_SLOTS)
        return {"h": h, "w": w, "slot_r": slot_r, "slot_c": slot_c, "require_clipped": need_clipped}
    if index == 1 and need_center:
        h, w = random.choice(ALL_DIMS)
        return {"h": h, "w": w, "slot_r": 1, "slot_c": 1, "require_clipped": False}
    if index == 1 and not need_center:
        h, w = random.choice(ALL_DIMS)
        slot_r, slot_c = random.choice(CARDINAL_SLOTS)
        return {"h": h, "w": w, "slot_r": slot_r, "slot_c": slot_c, "require_clipped": False}

    h, w = random.choice(ALL_DIMS)
    roll = float(rng.random())
    if need_center and roll < 0.20:
        slot_r, slot_c = 1, 1
    elif roll < 0.55:
        slot_r, slot_c = random.choice(DIAGONAL_SLOTS)
    else:
        slot_r, slot_c = random.choice(CARDINAL_SLOTS)
    require_clipped = need_clipped and (slot_r, slot_c) != (1, 1) and index == 2
    return {"h": h, "w": w, "slot_r": slot_r, "slot_c": slot_c, "require_clipped": require_clipped}


def _candidate_specs(
    shape: tuple[int, int],
    placed_specs: list[RectSpec],
    occupied_output: set[tuple[int, int]],
    template: dict,
    color: int,
    rng: np.random.Generator,
) -> list[RectSpec]:
    h = int(template["h"])
    w = int(template["w"])
    slot_r = int(template["slot_r"])
    slot_c = int(template["slot_c"])
    require_clipped = bool(template["require_clipped"])
    candidates: list[RectSpec] = []

    for r0 in range(shape[0] - h + 1):
        for c0 in range(shape[1] - w + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_touch(box, spec.bbox, padding=1) for spec in placed_specs):
                continue

            stats = _trajectory_stats(r0, c0, h, w, slot_r, slot_c, shape)
            copies = int(stats["copies"])
            full_copies = int(stats["full_copies"])
            clipped = bool(stats["clipped"])
            if (slot_r, slot_c) == (1, 1):
                if copies != 0:
                    continue
            else:
                if not (1 <= copies <= GEN_MAX_COPIES):
                    continue
                if full_copies < 1:
                    continue
            if require_clipped and not clipped:
                continue
            if require_clipped and copies < 2:
                continue
            if stats["footprint"] & occupied_output:
                continue

            marker_r, marker_c = _choose_marker(r0, c0, h, w, slot_r, slot_c, rng)
            candidates.append(
                RectSpec(
                    r0=r0,
                    c0=c0,
                    h=h,
                    w=w,
                    color=int(color),
                    marker_r=marker_r,
                    marker_c=marker_c,
                    copies=copies,
                    clipped=clipped,
                )
            )

    rng.shuffle(candidates)
    return candidates


def _validate_specs(specs: list[RectSpec], shape: tuple[int, int], need_center: bool, need_clipped: bool) -> bool:
    if not (3 <= len(specs) <= 5):
        return False
    if len({spec.color for spec in specs}) != len(specs):
        return False
    if any(max(spec.h, spec.w) > 6 or spec.h % 3 or spec.w % 3 for spec in specs):
        return False
    if any(_boxes_touch(specs[i].bbox, specs[j].bbox, padding=1) for i in range(len(specs)) for j in range(i + 1, len(specs))):
        return False

    footprints = []
    for spec in specs:
        stats = _trajectory_stats(spec.r0, spec.c0, spec.h, spec.w, spec.slot_r, spec.slot_c, shape)
        if stats["copies"] != spec.copies or stats["clipped"] != spec.clipped:
            return False
        footprints.append(stats["footprint"])
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            if footprints[i] & footprints[j]:
                return False

    has_center = any(spec.direction == (0, 0) for spec in specs)
    has_clipped = any(spec.clipped for spec in specs)
    has_orthogonal = any(spec.direction[0] == 0 or spec.direction[1] == 0 for spec in specs if spec.direction != (0, 0))
    has_non_square_diagonal = any(spec.h != spec.w and abs(spec.direction[0]) == 1 and abs(spec.direction[1]) == 1 for spec in specs)
    has_multi_copy = any(spec.copies >= 2 for spec in specs)
    if need_center and not has_center:
        return False
    if need_clipped and not has_clipped:
        return False
    if not has_non_square_diagonal:
        return False
    if not has_multi_copy:
        return False
    if not (has_center or has_orthogonal):
        return False

    input_grid = _render_input(specs, shape)
    output_grid = _render_output(specs, shape)
    if not np.array_equal(main(input_grid), output_grid):
        return False

    parsed = _extract_objects(input_grid)
    if len(parsed) != len(specs):
        return False
    parsed_sig = sorted((obj["bbox"], obj["color"], obj["marker"], obj["direction"]) for obj in parsed)
    spec_sig = sorted((spec.bbox, spec.color, (spec.marker_r, spec.marker_c), spec.direction) for spec in specs)
    if parsed_sig != spec_sig:
        return False

    coverage = sum(len(cells) for cells in footprints) / float(shape[0] * shape[1])
    return GEN_MIN_COVERAGE <= coverage <= GEN_MAX_COVERAGE


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    shape = (int(rng.integers(*GEN_HEIGHT_RANGE)), int(rng.integers(*GEN_WIDTH_RANGE)))
    object_count = int(rng.integers(*GEN_OBJECT_COUNT_RANGE))
    colors = [int(c) for c in rng.choice(BODY_COLORS, size=object_count, replace=False)]

    need_center = bool(rng.random() < 0.65)
    need_clipped = bool(rng.random() < 0.70)

    templates = [_sample_template(i, need_center, need_clipped, rng) for i in range(object_count)]
    if need_clipped and all((tpl["slot_r"], tpl["slot_c"]) == (1, 1) for tpl in templates):
        return None

    placed_specs: list[RectSpec] = []
    occupied_output: set[tuple[int, int]] = set()
    for template, color in zip(templates, colors):
        candidates = _candidate_specs(shape, placed_specs, occupied_output, template, color, rng)
        if not candidates:
            return None
        spec = candidates[0]
        placed_specs.append(spec)
        occupied_output |= _trajectory_stats(spec.r0, spec.c0, spec.h, spec.w, spec.slot_r, spec.slot_c, shape)["footprint"]

    if not _validate_specs(placed_specs, shape, need_center, need_clipped):
        return None
    return _render_input(placed_specs, shape)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marked-direction-stamping input")
