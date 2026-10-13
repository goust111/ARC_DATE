from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 0
BLUE = 1
ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 500
PLACE_PADDING = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors4(r, c, h, w):
                if bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(cells, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale_from_area(area: int, prototype_area: int) -> int:
    q, rem = divmod(int(area), int(prototype_area))
    assert rem == 0 and q > 0, "block area must be prototype area times a square"
    k = int(math.isqrt(q))
    assert k * k == q, "block area must determine one integer scale"
    return k


def _prototype(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    comps = _components(grid == BLUE)
    assert len(comps) == 1, "expected one blue prototype sprite"
    coords = comps[0]
    box = _bbox_coords(coords)
    r0, c0, _r1, _c1 = box
    rel = [(int(r) - r0, int(c) - c0) for r, c in coords]
    return rel, box


def _paint_scaled(
    out: np.ndarray,
    proto_rel: list[tuple[int, int]],
    origin: tuple[int, int],
    scale: int,
    color: int,
) -> None:
    r0, c0 = origin
    h, w = out.shape
    for pr, pc in proto_rel:
        rr = r0 + pr * scale
        cc = c0 + pc * scale
        assert 0 <= rr and 0 <= cc and rr + scale <= h and cc + scale <= w
        out[rr : rr + scale, cc : cc + scale] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each irregular color block by an area-matched scaled copy of the blue prototype.

    Input structure:
    - A black background contains exactly one small blue prototype sprite.
    - Several detached non-blue monochrome blocks have pairwise distinct colors.
    - Each block's area is the prototype area times one positive integer square, while its
      own bounding box is not the scaled prototype bounding box.

    Rule:
    - Count the blue prototype cells, then count each non-blue block's cells.
    - For a block of area n and prototype area m, infer the unique integer k with n = m * k * k.
    - Erase the block and, from that block's original bounding-box upper-left corner, paint
      every prototype cell as a k by k block in the block's original color.
    - Leave the blue prototype unchanged.

    Key insight:
    - The irregular block shape is a decoy: only its area determines scale, while its color
      and bounding-box anchor determine the stamped replacement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit inside 30x30"

    proto_rel, proto_box = _prototype(g)
    proto_area = len(proto_rel)
    pr0, pc0, pr1, pc1 = proto_box
    proto_h = pr1 - pr0 + 1
    proto_w = pc1 - pc0 + 1

    records = []
    for color in np.unique(g):
        color = int(color)
        if color in (BG, BLUE):
            continue
        comps = _components(g == color)
        assert len(comps) == 1, "each non-blue color must form one block"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox_coords(coords)
        scale = _scale_from_area(int(coords.shape[0]), proto_area)
        assert r0 + proto_h * scale <= g.shape[0]
        assert c0 + proto_w * scale <= g.shape[1]
        assert (r1 - r0 + 1, c1 - c0 + 1) != (
            proto_h * scale,
            proto_w * scale,
        ), "block bbox must not reveal the scale directly"
        records.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "scale": scale,
            }
        )

    out = g.copy()
    for rec in records:
        coords = rec["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for rec in sorted(records, key=lambda item: item["bbox"]):
        r0, c0, _r1, _c1 = rec["bbox"]
        _paint_scaled(out, proto_rel, (r0, c0), int(rec["scale"]), int(rec["color"]))

    return out


# ---------------- generator ----------------


def _edge_l_path(h: int, w: int) -> set[tuple[int, int]]:
    mode = random.randrange(4)
    cells: set[tuple[int, int]] = set()
    if mode == 0:
        cells.update((r, 0) for r in range(h))
        cells.update((h - 1, c) for c in range(w))
    elif mode == 1:
        cells.update((0, c) for c in range(w))
        cells.update((r, w - 1) for r in range(h))
    elif mode == 2:
        cells.update((r, w - 1) for r in range(h))
        cells.update((0, c) for c in range(w))
    else:
        cells.update((h - 1, c) for c in range(w))
        cells.update((r, 0) for r in range(h))
    return cells


def _grow_blob_in_box(h: int, w: int, count: int) -> np.ndarray | None:
    if h < 2 or w < 2 or count < h + w - 1 or count >= h * w:
        return None
    cells = _edge_l_path(h, w)
    if len(cells) > count:
        return None
    while len(cells) < count:
        frontier = sorted(
            {
                nb
                for r, c in cells
                for nb in _neighbors4(r, c, h, w)
                if nb not in cells
            }
        )
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _dim_candidates(
    count: int, stamp_h: int, stamp_w: int, mode: str, max_dim: int = 13
) -> list[tuple[int, int]]:
    stamp_area = stamp_h * stamp_w
    candidates: list[tuple[int, int]] = []
    for h in range(2, max_dim + 1):
        for w in range(2, max_dim + 1):
            area = h * w
            if area <= count or count < h + w - 1:
                continue
            if (h, w) == (stamp_h, stamp_w):
                continue
            if mode == "compact" and area >= stamp_area:
                continue
            if mode == "wide" and area <= stamp_area:
                continue
            candidates.append((h, w))
    random.shuffle(candidates)
    return candidates


def _make_blob(count: int, stamp_h: int, stamp_w: int, mode: str) -> np.ndarray | None:
    modes = [mode, "any"] if mode in ("compact", "wide") else ["any"]
    for chosen_mode in modes:
        for h, w in _dim_candidates(count, stamp_h, stamp_w, chosen_mode):
            blob = _grow_blob_in_box(h, w, count)
            if blob is None:
                continue
            if blob.shape != (stamp_h, stamp_w) and not np.all(blob):
                return blob
    return None


def _make_prototype() -> np.ndarray | None:
    options = [(3, 3, 5), (3, 3, 6), (3, 4, 6), (4, 3, 6)]
    random.shuffle(options)
    for h, w, count in options:
        mask = _grow_blob_in_box(h, w, count)
        if mask is not None:
            proto = np.full((h, w), BG, dtype=int)
            proto[mask] = BLUE
            return proto
    return None


def _can_place(reserve: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    height, width = reserve.shape
    if r < 1 or c < 1 or r + h >= height or c + w >= width:
        return False
    r0 = max(0, r - PLACE_PADDING)
    c0 = max(0, c - PLACE_PADDING)
    r1 = min(height, r + h + PLACE_PADDING)
    c1 = min(width, c + w + PLACE_PADDING)
    return not bool(reserve[r0:r1, c0:c1].any())


def _place_rect(reserve: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    height, width = reserve.shape
    candidates = [
        (r, c)
        for r in range(1, height - h)
        for c in range(1, width - w)
        if _can_place(reserve, r, c, h, w)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _render_expected(
    grid: np.ndarray,
    proto_rel: list[tuple[int, int]],
    specs: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0 = spec["pos"]
        blob = spec["blob"]
        rr, cc = np.nonzero(blob)
        out[r0 + rr, c0 + cc] = BG
    for spec in specs:
        _paint_scaled(
            out,
            proto_rel,
            spec["pos"],
            int(spec["scale"]),
            int(spec["color"]),
        )
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        proto_rel, proto_box = _prototype(grid)
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(grid, expected):
        return False

    proto_area = len(proto_rel)
    pr0, pc0, pr1, pc1 = proto_box
    proto_h = pr1 - pr0 + 1
    proto_w = pc1 - pc0 + 1
    scales: list[int] = []
    block_bbox_areas: list[int] = []
    stamp_bbox_areas: list[int] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, BLUE):
            continue
        comps = _components(grid == color)
        if len(comps) != 1:
            return False
        coords = comps[0]
        k = _scale_from_area(int(coords.shape[0]), proto_area)
        r0, c0, r1, c1 = _bbox_coords(coords)
        block_h = r1 - r0 + 1
        block_w = c1 - c0 + 1
        stamp_h = proto_h * k
        stamp_w = proto_w * k
        if (block_h, block_w) == (stamp_h, stamp_w):
            return False
        if int(np.count_nonzero(expected == color)) != int(coords.shape[0]):
            return False
        scales.append(k)
        block_bbox_areas.append(block_h * block_w)
        stamp_bbox_areas.append(stamp_h * stamp_w)

    if len(scales) < 3 or len(set(scales)) < 3:
        return False
    if not any(a < b for a, b in zip(block_bbox_areas, stamp_bbox_areas, strict=False)):
        return False
    if not any(a > b for a, b in zip(block_bbox_areas, stamp_bbox_areas, strict=False)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(24, 30)
    width = random.randint(24, 30)
    proto = _make_prototype()
    if proto is None:
        return None
    proto_mask = proto == BLUE
    proto_rel = [(int(r), int(c)) for r, c in np.argwhere(proto_mask)]
    proto_h, proto_w = proto.shape
    proto_area = len(proto_rel)

    object_count = random.choice((3, 4))
    scales = [1, 2, 3]
    if object_count == 4:
        scales.append(random.choice((1, 2, 3)))
    random.shuffle(scales)
    colors = random.sample([c for c in ARC_COLORS if c not in (BG, BLUE)], object_count)

    specs: list[dict] = []
    forced_modes = ["compact", "wide", "any", "compact"]
    for idx, (scale, color) in enumerate(zip(scales, colors, strict=False)):
        stamp_h = proto_h * scale
        stamp_w = proto_w * scale
        area = proto_area * scale * scale
        blob = _make_blob(area, stamp_h, stamp_w, forced_modes[idx])
        if blob is None:
            return None
        specs.append(
            {
                "scale": scale,
                "color": color,
                "blob": blob,
                "zone": (max(blob.shape[0], stamp_h), max(blob.shape[1], stamp_w)),
            }
        )

    reserve = np.zeros((height, width), dtype=bool)
    for spec in sorted(specs, key=lambda s: s["zone"][0] * s["zone"][1], reverse=True):
        zh, zw = spec["zone"]
        pos = _place_rect(reserve, zh, zw)
        if pos is None:
            return None
        r, c = pos
        reserve[r : r + zh, c : c + zw] = True
        spec["pos"] = pos

    proto_pos = _place_rect(reserve, proto_h, proto_w)
    if proto_pos is None:
        return None
    pr, pc = proto_pos
    reserve[pr : pr + proto_h, pc : pc + proto_w] = True

    grid = np.full((height, width), BG, dtype=int)
    grid[pr : pr + proto_h, pc : pc + proto_w][proto_mask] = BLUE
    for spec in specs:
        r, c = spec["pos"]
        blob = spec["blob"]
        rr, cc = np.nonzero(blob)
        grid[r + rr, c + cc] = int(spec["color"])

    expected = _render_expected(grid, proto_rel, specs)
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid area-conserving stamp input after {GEN_ATTEMPTS} attempts")
