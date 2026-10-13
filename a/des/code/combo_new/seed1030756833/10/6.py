from __future__ import annotations

from collections import deque
import random

import numpy as np


def _background(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = (int(pts[0, 0]), int(pts[0, 1]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c, h, w):
            if bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_slots(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int, int]]:
    h, w = g.shape
    slots: list[tuple[int, int, int, int, int, int]] = []
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                a = int(g[r0, c0])
                if a == bg:
                    continue
                for c1 in range(c0 + 2, w):
                    b = int(g[r1, c0 + 1])
                    if b == bg or b == a:
                        continue
                    if int(g[r1, c1]) != a:
                        continue
                    if not np.all(g[r0, c0 : c1 + 1] == a):
                        continue
                    if not np.all(g[r0 : r1 + 1, c0] == a):
                        continue
                    if not np.all(g[r1, c0 + 1 : c1] == b):
                        continue
                    if not np.all(g[r0 + 1 : r1, c1] == b):
                        continue
                    if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg):
                        continue
                    slots.append((r0, c0, r1, c1, a, b))
    return slots


def _slot_border_mask(shape: tuple[int, int], slots: list[tuple[int, int, int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1, _a, _b in slots:
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _components_by_color(g: np.ndarray, bg: int) -> dict[int, list[dict]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: dict[int, list[dict]] = {}
    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color == bg or seen[sr, sc]:
                continue
            coords = []
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors8(r, c, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.setdefault(color, []).append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _select_fragment(comps: dict[int, list[dict]], color: int, shape: tuple[int, int]) -> dict:
    matches = []
    for comp in comps.get(color, []):
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) == shape:
            matches.append(comp)
    assert len(matches) == 1, "each slot color must identify one matching fragment"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Pair colored fragments with two-color rectangular slots and assemble each slot interior.

    Input structure:
    - The background is the dominant color.
    - Several small monochrome 8-connected fragments are scattered away from the slots, with a unique color for each fragment.
    - Each empty rectangular slot has a one-pixel border: its top and left sides use color A, its bottom and right sides use color B, and all four corner pixels use color A.
    - The two border colors of every slot match exactly two fragment colors; those fragment bboxes have the same height and width as the slot interior.

    Rule:
    - For each slot, use the top/left color A and bottom/right color B to find the matching two fragments.
    - Clear all original fragment pixels back to the background.
    - Align the two fragment bboxes to the slot interior and paste their filled pixels in their own colors.
    - Leave the two-color slot border unchanged.

    Key insight:
    The split-colored border is a pairing code: its two colors name the two complementary fragments whose aligned masks tile the empty rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)

    slots = _find_slots(g, bg)
    assert slots, "expected at least one two-color rectangular slot"
    border = _slot_border_mask(g.shape, slots)

    without_borders = g.copy()
    without_borders[border] = bg
    comps = _components_by_color(without_borders, bg)

    out = g.copy()
    placements = []
    used_colors: set[int] = set()
    used_components: list[dict] = []
    for r0, c0, r1, c1, a, b in slots:
        assert a not in used_colors and b not in used_colors
        used_colors.update({a, b})

        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        shape = (ih, iw)
        frag_a = _select_fragment(comps, a, shape)
        frag_b = _select_fragment(comps, b, shape)
        mask_a = _component_mask(frag_a)
        mask_b = _component_mask(frag_b)
        assert mask_a.shape == shape and mask_b.shape == shape
        assert not np.any(mask_a & mask_b)
        assert np.all(mask_a | mask_b)

        used_components.extend([frag_a, frag_b])
        placements.append((r0 + 1, c0 + 1, a, b, mask_a, mask_b))

    for comp in used_components:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for ir, ic, a, b, mask_a, mask_b in placements:
        ih, iw = mask_a.shape
        target = out[ir : ir + ih, ic : ic + iw]
        assert np.all(target == bg)
        target[mask_a] = a
        target[mask_b] = b

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 1000
COLOR_COUNT = 10


def _sample_mask_pair(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MASK_ATTEMPTS):
        p = random.uniform(0.35, 0.65)
        mask = np.array(
            [[random.random() < p for _c in range(width)] for _r in range(height)],
            dtype=bool,
        )
        other = ~mask
        if (
            _full_bbox(mask)
            and _full_bbox(other)
            and _mask_connected(mask)
            and _mask_connected(other)
        ):
            if random.choice([True, False]):
                mask = np.flipud(mask)
            if random.choice([True, False]):
                mask = np.fliplr(mask)
            if height == width and random.choice([True, False]):
                mask = mask.T
            return mask.copy(), (~mask).copy()

    phase = random.randint(0, 1)
    fallback = np.fromfunction(lambda r, c: ((r + c + phase) % 2) == 0, (height, width), dtype=int)
    return fallback.astype(bool), (~fallback).astype(bool)


def _can_place(occ: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    rows, cols = occ.shape
    if r < 1 or c < 1 or r + h >= rows or c + w >= cols:
        return False
    return not bool(occ[r - 1 : r + h + 1, c - 1 : c + w + 1].any())


def _mark_place(occ: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    occ[r - 1 : r + h + 1, c - 1 : c + w + 1] = True


def _place_record(occ: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    rows, cols = occ.shape
    candidates = [
        (r, c)
        for r in range(1, rows - h)
        for c in range(1, cols - w)
        if _can_place(occ, r, c, h, w)
    ]
    if not candidates:
        return None
    r, c = random.choice(candidates)
    _mark_place(occ, r, c, h, w)
    return r, c


def _paint_slot(grid: np.ndarray, r: int, c: int, ih: int, iw: int, a: int, b: int) -> None:
    r1, c1 = r + ih + 1, c + iw + 1
    grid[r, c : c1 + 1] = a
    grid[r : r1 + 1, c] = a
    grid[r1, c + 1 : c1] = b
    grid[r + 1 : r1, c1] = b
    grid[r1, c1] = a


def _paint_fragment(grid: np.ndarray, r: int, c: int, color: int, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    patch = grid[r : r + h, c : c + w]
    patch[mask] = color
    pts = np.argwhere(mask)
    return [(r + int(pr), c + int(pc)) for pr, pc in pts]


def _expected_from_records(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    out = grid.copy()
    for rec in records:
        if rec["kind"] == "frag":
            for r, c in rec["cells"]:
                out[r, c] = rec["bg"]
    for rec in records:
        if rec["kind"] != "slot":
            continue
        ir, ic = rec["r"] + 1, rec["c"] + 1
        mask_a, mask_b = rec["mask_a"], rec["mask_b"]
        h, w = mask_a.shape
        target = out[ir : ir + h, ic : ic + w]
        target[mask_a] = rec["a"]
        target[mask_b] = rec["b"]
    return out


def _gen_attempt() -> np.ndarray | None:
    n_slots = random.randint(1, 4)
    min_side = 14 + 3 * n_slots
    rows = random.randint(min_side, 30)
    cols = random.randint(min_side, 30)

    palette = random.sample(range(COLOR_COUNT), 1 + 2 * n_slots)
    bg = palette[0]
    slot_colors = palette[1:]

    grid = np.full((rows, cols), bg, dtype=int)
    occ = np.zeros((rows, cols), dtype=bool)

    dim_max = {1: 8, 2: 7, 3: 6, 4: 5}[n_slots]
    pairs = []
    for i in range(n_slots):
        ih = random.randint(3, dim_max)
        iw = random.randint(3, dim_max + 1)
        mask_a, mask_b = _sample_mask_pair(ih, iw)
        a, b = slot_colors[2 * i], slot_colors[2 * i + 1]
        if random.choice([True, False]):
            mask_a, mask_b = mask_b, mask_a
        pairs.append({"ih": ih, "iw": iw, "a": a, "b": b, "mask_a": mask_a, "mask_b": mask_b})

    placeables = []
    for idx, pair in enumerate(pairs):
        ih, iw = pair["ih"], pair["iw"]
        placeables.append({"kind": "slot", "idx": idx, "h": ih + 2, "w": iw + 2})
        placeables.append({"kind": "frag_a", "idx": idx, "h": ih, "w": iw})
        placeables.append({"kind": "frag_b", "idx": idx, "h": ih, "w": iw})
    random.shuffle(placeables)

    records: list[dict] = []
    for item in placeables:
        pos = _place_record(occ, item["h"], item["w"])
        if pos is None:
            return None
        r, c = pos
        pair = pairs[item["idx"]]
        if item["kind"] == "slot":
            _paint_slot(grid, r, c, pair["ih"], pair["iw"], pair["a"], pair["b"])
            records.append(
                {
                    "kind": "slot",
                    "r": r,
                    "c": c,
                    "a": pair["a"],
                    "b": pair["b"],
                    "mask_a": pair["mask_a"],
                    "mask_b": pair["mask_b"],
                }
            )
        elif item["kind"] == "frag_a":
            cells = _paint_fragment(grid, r, c, pair["a"], pair["mask_a"])
            records.append({"kind": "frag", "cells": cells, "bg": bg})
        else:
            cells = _paint_fragment(grid, r, c, pair["b"], pair["mask_b"])
            records.append({"kind": "frag", "cells": cells, "bg": bg})

    expected = _expected_from_records(grid, records)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
