from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _is_rect(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(shape)
    return len(shape) == h * w


def _extract_fragments(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == int(color), dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    fragments: list[dict] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            shape = _normalize(cells)
            h, w = _shape_dims(shape)
            fragments.append(
                {
                    "shape": shape,
                    "area": len(shape),
                    "h": h,
                    "w": w,
                }
            )

    fragments.sort(key=lambda frag: (-frag["area"], frag["h"], frag["w"], frag["shape"]))
    return fragments


@lru_cache(maxsize=None)
def _placement_masks(
    shape: tuple[tuple[int, int], ...],
    rect_h: int,
    rect_w: int,
) -> tuple[int, ...]:
    frag_h, frag_w = _shape_dims(shape)
    if frag_h > rect_h or frag_w > rect_w:
        return ()

    masks: list[int] = []
    for r0 in range(rect_h - frag_h + 1):
        for c0 in range(rect_w - frag_w + 1):
            bits = 0
            for r, c in shape:
                bit = (r0 + r) * rect_w + (c0 + c)
                bits |= 1 << int(bit)
            masks.append(bits)
    return tuple(masks)


def _packing_keys(fragments: list[dict], rect_h: int, rect_w: int, limit: int = 2) -> set[tuple]:
    area = sum(int(frag["area"]) for frag in fragments)
    if rect_h * rect_w != area:
        return set()

    placements = []
    for frag in fragments:
        masks = _placement_masks(frag["shape"], int(rect_h), int(rect_w))
        if not masks:
            return set()
        placements.append(masks)

    full_mask = (1 << (rect_h * rect_w)) - 1
    shape_keys = [frag["shape"] for frag in fragments]
    seen_keys: set[tuple] = set()

    def canonical(chosen: tuple[tuple[int, int], ...]) -> tuple:
        grouped: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
        for idx, mask in chosen:
            grouped[shape_keys[idx]].append(int(mask))
        return (
            (int(rect_h), int(rect_w)),
            tuple(sorted((shape, tuple(sorted(masks))) for shape, masks in grouped.items())),
        )

    def search(remaining: tuple[int, ...], used: int, chosen: tuple[tuple[int, int], ...]) -> None:
        if len(seen_keys) >= limit:
            return
        if not remaining:
            if used == full_mask:
                seen_keys.add(canonical(chosen))
            return

        uncovered = full_mask & ~used
        if uncovered == 0:
            return
        first_bit = uncovered & -uncovered

        best_idx = None
        best_opts = None
        for idx in remaining:
            opts = [mask for mask in placements[idx] if (mask & first_bit) and not (mask & used)]
            if not opts:
                continue
            if best_opts is None or len(opts) < len(best_opts):
                best_idx = idx
                best_opts = opts

        if best_idx is None:
            return

        next_remaining = tuple(idx for idx in remaining if idx != best_idx)
        for mask in best_opts:
            search(next_remaining, used | mask, chosen + ((int(best_idx), int(mask)),))
            if len(seen_keys) >= limit:
                return

    search(tuple(range(len(fragments))), 0, ())
    return seen_keys


def _all_packing_keys(fragments: list[dict], limit: int = 2) -> set[tuple]:
    area = sum(int(frag["area"]) for frag in fragments)
    keys: set[tuple] = set()
    seen_dims: set[tuple[int, int]] = set()
    for h in range(1, int(math.isqrt(area)) + 1):
        if area % h != 0:
            continue
        for dims in ((h, area // h), (area // h, h)):
            if dims in seen_dims:
                continue
            seen_dims.add(dims)
            sub = _packing_keys(fragments, int(dims[0]), int(dims[1]), limit=max(1, limit - len(keys)))
            keys.update(sub)
            if len(keys) >= limit:
                return keys
    return keys


def _solve_rectangle(fragments: list[dict]) -> tuple[int, int]:
    keys = _all_packing_keys(fragments, limit=2)
    assert len(keys) == 1, "same-color fragments must admit exactly one rectangle assembly"
    dims, _layout = next(iter(keys))
    return int(dims[0]), int(dims[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored fragments, reassemble each color into the unique solid rectangle,
    then sort those rectangles by area and pack them left-to-right with bottom alignment.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert 3 <= len(colors) <= 5

    rects: list[tuple[int, int, int, int]] = []
    for color in colors:
        fragments = _extract_fragments(g, color)
        assert 2 <= len(fragments) <= 4
        h, w = _solve_rectangle(fragments)
        rects.append((h * w, h, w, color))

    rects.sort(key=lambda item: (item[0], item[1], item[2]))
    out_h = max(h for _area, h, _w, _color in rects)
    out_w = sum(w for _area, _h, w, _color in rects) + len(rects) - 1

    out = np.zeros((out_h, out_w), dtype=int)
    c0 = 0
    for _area, h, w, color in rects:
        out[out_h - h : out_h, c0 : c0 + w] = int(color)
        c0 += w + 1
    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
GROUP_LIBRARY: dict[int, tuple[tuple[int, int, str, tuple[int, ...]], ...]] = {
    6: (
        (3, 2, "h", (1, 2)),
        (2, 3, "v", (1, 2)),
    ),
    8: (
        (4, 2, "h", (1, 3)),
        (4, 2, "h", (1, 1, 2)),
        (2, 4, "v", (1, 3)),
    ),
    9: (
        (3, 3, "h", (1, 2)),
    ),
    10: (
        (5, 2, "h", (2, 3)),
        (5, 2, "h", (1, 1, 1, 2)),
        (2, 5, "v", (1, 1, 3)),
        (2, 5, "v", (1, 1, 1, 2)),
    ),
    12: (
        (4, 3, "h", (1, 1, 2)),
        (4, 3, "v", (1, 2)),
        (3, 4, "h", (1, 2)),
        (3, 4, "v", (1, 3)),
        (6, 2, "h", (1, 2, 3)),
        (6, 2, "h", (1, 1, 1, 3)),
    ),
    14: (
        (7, 2, "h", (3, 4)),
        (7, 2, "h", (1, 2, 2, 2)),
        (7, 2, "h", (1, 1, 1, 4)),
        (2, 7, "v", (1, 2, 4)),
        (2, 7, "v", (1, 2, 2, 2)),
        (2, 7, "v", (1, 1, 1, 4)),
        (2, 7, "v", (2, 2, 3)),
    ),
    15: (
        (5, 3, "h", (1, 2, 2)),
        (5, 3, "h", (1, 1, 1, 2)),
        (5, 3, "v", (1, 2)),
        (3, 5, "v", (1, 2, 2)),
        (3, 5, "v", (1, 1, 1, 2)),
        (3, 5, "h", (1, 2)),
    ),
}
AREA_CHOICES = tuple(sorted(GROUP_LIBRARY))
COLOR_COUNT_RANGE = (3, 5)
INPUT_SIDE_RANGE = (18, 30)
GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 80


def _template_fragments(h: int, w: int, axis: str, parts: tuple[int, ...]) -> list[dict]:
    assert axis in {"h", "v"}
    if axis == "h":
        assert sum(parts) == h
        shapes = [tuple((r, c) for r in range(part) for c in range(w)) for part in parts]
    else:
        assert sum(parts) == w
        shapes = [tuple((r, c) for r in range(h) for c in range(part)) for part in parts]

    fragments = [
        {
            "shape": shape,
            "area": len(shape),
            "h": _shape_dims(shape)[0],
            "w": _shape_dims(shape)[1],
        }
        for shape in shapes
    ]
    random.shuffle(fragments)
    return fragments


def _build_group(area: int, color: int) -> dict:
    candidates = list(GROUP_LIBRARY[int(area)])
    random.shuffle(candidates)
    for h, w, axis, parts in candidates:
        fragments = _template_fragments(int(h), int(w), axis, parts)
        try:
            dims = _solve_rectangle(fragments)
        except AssertionError:
            continue
        if dims == (int(h), int(w)):
            return {
                "color": int(color),
                "area": int(area),
                "h": int(h),
                "w": int(w),
                "fragments": fragments,
            }
    raise AssertionError(f"no valid template found for area {area}")


def _can_place(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> bool:
    H, W = occupied.shape
    for r, c in shape:
        rr = top + r
        cc = left + c
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if occupied[rr, cc]:
            return False
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                return False
    return True


def _place_fragments(groups: list[dict]) -> np.ndarray | None:
    all_frags: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for group in groups:
        for frag in group["fragments"]:
            all_frags.append((int(group["color"]), frag["shape"]))
    all_frags.sort(key=lambda item: (-len(item[1]), _shape_dims(item[1])[0] * _shape_dims(item[1])[1]))

    total_area = sum(len(shape) for _color, shape in all_frags)
    piece_count = len(all_frags)
    base = int(math.ceil(math.sqrt(total_area * 5 + piece_count * 8)))
    h_lo = max(INPUT_SIDE_RANGE[0], min(INPUT_SIDE_RANGE[1], base - 2))
    h_hi = max(h_lo, min(INPUT_SIDE_RANGE[1], base + 4))
    w_lo = max(INPUT_SIDE_RANGE[0], min(INPUT_SIDE_RANGE[1], base - 1))
    w_hi = max(w_lo, min(INPUT_SIDE_RANGE[1], base + 5))

    for _ in range(PLACEMENT_ATTEMPTS):
        H = random.randint(h_lo, h_hi)
        W = random.randint(w_lo, w_hi)
        grid = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        ok = True

        for color, shape in all_frags:
            frag_h, frag_w = _shape_dims(shape)
            positions = [
                (r0, c0)
                for r0 in range(H - frag_h + 1)
                for c0 in range(W - frag_w + 1)
            ]
            random.shuffle(positions)

            placed = False
            for r0, c0 in positions:
                if not _can_place(occupied, shape, int(r0), int(c0)):
                    continue
                for r, c in shape:
                    rr = int(r0 + r)
                    cc = int(c0 + c)
                    grid[rr, cc] = int(color)
                    occupied[rr, cc] = True
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok:
            return grid
    return None


def _expected_output(groups: list[dict]) -> np.ndarray:
    ordered = sorted(groups, key=lambda group: (group["area"], group["h"], group["w"]))
    out_h = max(group["h"] for group in ordered)
    out_w = sum(group["w"] for group in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    c0 = 0
    for group in ordered:
        out[out_h - group["h"] : out_h, c0 : c0 + group["w"]] = int(group["color"])
        c0 += int(group["w"]) + 1
    return out


def _generate_attempt() -> np.ndarray | None:
    color_count = random.randint(COLOR_COUNT_RANGE[0], COLOR_COUNT_RANGE[1])
    colors = random.sample(COLOR_POOL, color_count)
    areas = random.sample(AREA_CHOICES, color_count)

    groups: list[dict] = []
    for area, color in zip(areas, colors):
        groups.append(_build_group(int(area), int(color)))

    expected = _expected_output(groups)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    grid = _place_fragments(groups)
    if grid is None:
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
