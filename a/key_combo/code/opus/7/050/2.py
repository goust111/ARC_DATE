from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class EncodedRect:
    color: int
    body_r0: int
    body_c0: int
    h: int
    w: int
    direction: str
    gap: int
    attach: int
    order_key: tuple[int, int]


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_DIMS = sorted(
    {
        (a, b)
        for a in (2, 3, 4, 5)
        for b in (2, 3, 4, 5)
        if a != b and min(a, b) in (2, 3) and max(a, b) in (3, 4, 5)
    }
)
PAIR_HORIZONTAL_DIMS = [(3, 2), (4, 2), (5, 2), (4, 3), (5, 3)]
PAIR_VERTICAL_DIMS = [(4, 2), (5, 2), (4, 3), (5, 3)]
GRID_RANGE = (18, 26)
GEN_RETRIES = 500
INPUT_MARGIN = 1
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each stemmed rectangle along its encoded direction, rotating invading copies on collision.

    Input structure:
    - The background is black and there are 2-3 non-touching foreground objects in distinct colors.
    - Each object is a solid non-square rectangle with a one-pixel-wide straight stem leaving the middle of one edge.
    - The stem direction is one of the four cardinal directions and its length is 1-4 pixels.

    Rule:
    - For each object, read the stem direction as the tiling direction and the stem length as the gap between consecutive rectangle copies.
    - Erase all stems but keep the original rectangles, then keep stamping more same-color copies from each rectangle along its encoded direction until they leave the canvas, clipping at the border.
    - Process objects in input reading order (top-to-bottom, then left-to-right); whenever a new copy would hit another color, rotate that invading copy 90 degrees clockwise and place it instead, skipping it only if the rotated footprint is entirely occupied by other colors.

    Key insight:
    - The stem is not preserved as geometry; it is a compact instruction that encodes both the replication axis and the spacing, and collisions convert only the later intruding copy into its rotated form.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background(g)
    colors = [c for c in np.unique(g) if int(c) != bg]
    assert 2 <= len(colors) <= 3

    objects: list[EncodedRect] = []
    for color in colors:
        mask = g == int(color)
        assert _component_count(mask) == 1
        objects.append(_decode_object(mask, int(color)))

    out, _stats = _simulate(objects, g.shape, bg=bg)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-2 input")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_RANGE)
    w = rng.randint(*GRID_RANGE)
    count = rng.choice([2, 3])
    colors = rng.sample(PALETTE, count)

    pair = _build_primary_pair(h, w, colors[:2], rng)
    if pair is None:
        return None
    objects = list(pair)

    if count == 3:
        extra = _place_extra_object(h, w, colors[2], objects, rng)
        if extra is None:
            return None
        objects.append(extra)

    if not _objects_well_separated(objects):
        return None

    grid = _render_input(objects, (h, w))
    out, stats = _simulate(objects, (h, w), bg=0)
    if stats["rotated"] < 1 or stats["placed"] < 4:
        return None
    if np.array_equal(grid, out):
        return None
    if np.count_nonzero(grid != out) < 8:
        return None
    return grid


def _build_primary_pair(
    h: int,
    w: int,
    colors: list[int],
    rng: random.Random,
) -> tuple[EncodedRect, EncodedRect] | None:
    for _ in range(200):
        a_h, a_w = rng.choice(PAIR_HORIZONTAL_DIMS)
        b_h, b_w = rng.choice(PAIR_VERTICAL_DIMS)
        a_dir = rng.choice(["left", "right"])
        a_gap = rng.randint(1, 4)
        b_gap = rng.randint(1, 4)
        a_attach = rng.choice(_central_positions(a_h))
        b_attach = rng.choice(_central_positions(b_w))
        a_idx = rng.randint(1, 2)
        b_idx = rng.randint(1, 2)

        a_step = a_w + a_gap
        b_step = b_h + b_gap

        row_lo = INPUT_MARGIN
        row_hi = h - INPUT_MARGIN - max(a_h, b_h)
        if row_lo > row_hi:
            continue

        if a_dir == "right":
            col_lo = INPUT_MARGIN + a_idx * a_step
            col_hi = w - INPUT_MARGIN - max(a_w, b_w)
        else:
            col_lo = INPUT_MARGIN
            col_hi = w - INPUT_MARGIN - max(a_w + a_idx * a_step, b_w)
        if col_lo > col_hi:
            continue

        coll_r = rng.randint(row_lo, row_hi)
        coll_c = rng.randint(col_lo, col_hi)
        a_body_c0 = coll_c - a_idx * a_step if a_dir == "right" else coll_c + a_idx * a_step
        b_body_r0 = coll_r + b_idx * b_step

        obj_a = EncodedRect(
            color=colors[0],
            body_r0=coll_r,
            body_c0=a_body_c0,
            h=a_h,
            w=a_w,
            direction=a_dir,
            gap=a_gap,
            attach=a_attach,
            order_key=_order_key(coll_r, a_body_c0, a_h, a_w, a_dir, a_gap),
        )
        obj_b = EncodedRect(
            color=colors[1],
            body_r0=b_body_r0,
            body_c0=coll_c,
            h=b_h,
            w=b_w,
            direction="up",
            gap=b_gap,
            attach=b_attach,
            order_key=_order_key(b_body_r0, coll_c, b_h, b_w, "up", b_gap),
        )

        if not _object_in_bounds(obj_a, h, w) or not _object_in_bounds(obj_b, h, w):
            continue
        if not _objects_well_separated([obj_a, obj_b]):
            continue

        _out, stats = _simulate([obj_a, obj_b], (h, w), bg=0)
        if stats["rotated"] < 1:
            continue
        return obj_a, obj_b
    return None


def _place_extra_object(
    h: int,
    w: int,
    color: int,
    existing: list[EncodedRect],
    rng: random.Random,
) -> EncodedRect | None:
    for _ in range(200):
        body_h, body_w = rng.choice(ALL_DIMS)
        direction = rng.choice(["up", "down", "left", "right"])
        gap = rng.randint(1, 4)
        attach = rng.choice(_central_positions(body_w if direction in ("up", "down") else body_h))
        row_lo = INPUT_MARGIN + (gap if direction == "up" else 0)
        row_hi = h - INPUT_MARGIN - body_h - (gap if direction == "down" else 0)
        col_lo = INPUT_MARGIN + (gap if direction == "left" else 0)
        col_hi = w - INPUT_MARGIN - body_w - (gap if direction == "right" else 0)
        if row_lo > row_hi or col_lo > col_hi:
            continue

        body_r0 = rng.randint(row_lo, row_hi)
        body_c0 = rng.randint(col_lo, col_hi)
        obj = EncodedRect(
            color=color,
            body_r0=body_r0,
            body_c0=body_c0,
            h=body_h,
            w=body_w,
            direction=direction,
            gap=gap,
            attach=attach,
            order_key=_order_key(body_r0, body_c0, body_h, body_w, direction, gap),
        )
        if not _objects_well_separated(existing + [obj]):
            continue
        return obj
    return None


def _render_input(objects: list[EncodedRect], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        r0, c0 = obj.body_r0, obj.body_c0
        grid[r0 : r0 + obj.h, c0 : c0 + obj.w] = obj.color
        for r, c in _stem_cells(obj):
            grid[r, c] = obj.color
    return grid


def _simulate(
    objects: list[EncodedRect],
    shape: tuple[int, int],
    *,
    bg: int,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full(shape, bg, dtype=int)
    for obj in objects:
        out[obj.body_r0 : obj.body_r0 + obj.h, obj.body_c0 : obj.body_c0 + obj.w] = obj.color

    stats = {"placed": 0, "rotated": 0, "skipped": 0}
    for obj in sorted(objects, key=lambda item: item.order_key):
        for r0, c0 in _copy_positions(obj, shape):
            base = _visible_slice(r0, c0, obj.h, obj.w, shape)
            if base is None:
                continue
            region = out[base]
            collided = bool(np.any((region != bg) & (region != obj.color)))
            stamp_h, stamp_w = (obj.w, obj.h) if collided else (obj.h, obj.w)
            target = _visible_slice(r0, c0, stamp_h, stamp_w, shape)
            if target is None:
                continue
            target_region = out[target]
            foreign = (target_region != bg) & (target_region != obj.color)
            if bool(np.all(foreign)):
                stats["skipped"] += 1
                continue
            out[target] = obj.color
            stats["placed"] += 1
            if collided:
                stats["rotated"] += 1

    return out, stats


def _decode_object(mask: np.ndarray, color: int) -> EncodedRect:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = patch.shape

    candidates: list[EncodedRect] = []
    for gap in range(1, ph):
        body = patch[gap:, :]
        if _solid(body) and _top_or_bottom_stem(patch[:gap, :], width=pw):
            stem_col = int(np.flatnonzero(patch[0])[0])
            if _is_central(stem_col, pw):
                candidates.append(
                    EncodedRect(color, r0 + gap, c0, ph - gap, pw, "up", gap, stem_col, (r0, c0))
                )

        body = patch[: ph - gap, :]
        if _solid(body) and _top_or_bottom_stem(patch[ph - gap :, :], width=pw):
            stem_col = int(np.flatnonzero(patch[-1])[0])
            if _is_central(stem_col, pw):
                candidates.append(
                    EncodedRect(color, r0, c0, ph - gap, pw, "down", gap, stem_col, (r0, c0))
                )

    for gap in range(1, pw):
        body = patch[:, gap:]
        if _solid(body) and _left_or_right_stem(patch[:, :gap], height=ph):
            stem_row = int(np.flatnonzero(patch[:, 0])[0])
            if _is_central(stem_row, ph):
                candidates.append(
                    EncodedRect(color, r0, c0 + gap, ph, pw - gap, "left", gap, stem_row, (r0, c0))
                )

        body = patch[:, : pw - gap]
        if _solid(body) and _left_or_right_stem(patch[:, pw - gap :], height=ph):
            stem_row = int(np.flatnonzero(patch[:, -1])[0])
            if _is_central(stem_row, ph):
                candidates.append(
                    EncodedRect(color, r0, c0, ph, pw - gap, "right", gap, stem_row, (r0, c0))
                )

    unique: list[EncodedRect] = []
    seen = set()
    for item in candidates:
        key = (item.body_r0, item.body_c0, item.h, item.w, item.direction, item.gap, item.attach)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    assert len(unique) == 1, f"expected one stemmed rectangle for color {color}, got {len(unique)}"
    obj = unique[0]
    assert obj.h != obj.w
    return obj


def _copy_positions(obj: EncodedRect, shape: tuple[int, int]):
    h, w = shape
    step = (obj.h if obj.direction in ("up", "down") else obj.w) + obj.gap
    dr, dc = DIRS[obj.direction]
    n = 1
    while True:
        r0 = obj.body_r0 + dr * step * n
        c0 = obj.body_c0 + dc * step * n
        if obj.direction == "up" and r0 + obj.h <= 0:
            break
        if obj.direction == "down" and r0 >= h:
            break
        if obj.direction == "left" and c0 + obj.w <= 0:
            break
        if obj.direction == "right" and c0 >= w:
            break
        yield r0, c0
        n += 1


def _stem_cells(obj: EncodedRect) -> list[tuple[int, int]]:
    if obj.direction == "up":
        c = obj.body_c0 + obj.attach
        return [(r, c) for r in range(obj.body_r0 - obj.gap, obj.body_r0)]
    if obj.direction == "down":
        c = obj.body_c0 + obj.attach
        start = obj.body_r0 + obj.h
        return [(r, c) for r in range(start, start + obj.gap)]
    if obj.direction == "left":
        r = obj.body_r0 + obj.attach
        return [(r, c) for c in range(obj.body_c0 - obj.gap, obj.body_c0)]
    r = obj.body_r0 + obj.attach
    start = obj.body_c0 + obj.w
    return [(r, c) for c in range(start, start + obj.gap)]


def _object_cells(obj: EncodedRect) -> set[tuple[int, int]]:
    cells = set(_stem_cells(obj))
    for r in range(obj.body_r0, obj.body_r0 + obj.h):
        for c in range(obj.body_c0, obj.body_c0 + obj.w):
            cells.add((r, c))
    return cells


def _objects_well_separated(objects: list[EncodedRect]) -> bool:
    cell_sets = [_object_cells(obj) for obj in objects]
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            if cell_sets[i] & cell_sets[j]:
                return False
            for r, c in cell_sets[i]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (r + dr, c + dc) in cell_sets[j]:
                            return False
    return True


def _object_in_bounds(obj: EncodedRect, h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in _object_cells(obj))


def _visible_slice(
    r0: int,
    c0: int,
    h: int,
    w: int,
    shape: tuple[int, int],
):
    grid_h, grid_w = shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(grid_h, r0 + h)
    cc1 = min(grid_w, c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return None
    return np.s_[rr0:rr1, cc0:cc1]


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _solid(mask: np.ndarray) -> bool:
    return mask.size > 0 and bool(np.all(mask))


def _top_or_bottom_stem(stem: np.ndarray, *, width: int) -> bool:
    if stem.size == 0 or int(stem.sum()) != stem.shape[0]:
        return False
    cols = []
    for row in stem:
        idx = np.flatnonzero(row)
        if len(idx) != 1:
            return False
        cols.append(int(idx[0]))
    return len(set(cols)) == 1 and _is_central(cols[0], width)


def _left_or_right_stem(stem: np.ndarray, *, height: int) -> bool:
    if stem.size == 0 or int(stem.sum()) != stem.shape[1]:
        return False
    rows = []
    for col in stem.T:
        idx = np.flatnonzero(col)
        if len(idx) != 1:
            return False
        rows.append(int(idx[0]))
    return len(set(rows)) == 1 and _is_central(rows[0], height)


def _central_positions(span: int) -> list[int]:
    if span % 2 == 1:
        return [span // 2]
    return [span // 2 - 1, span // 2]


def _is_central(index: int, span: int) -> bool:
    return int(index) in _central_positions(int(span))


def _order_key(body_r0: int, body_c0: int, h: int, w: int, direction: str, gap: int) -> tuple[int, int]:
    if direction == "up":
        return body_r0 - gap, body_c0
    if direction == "left":
        return body_r0, body_c0 - gap
    return body_r0, body_c0
