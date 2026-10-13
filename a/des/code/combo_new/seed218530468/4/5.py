from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
STAMP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_STAMP_SIDE = 5
REFERENCE_H_RANGE = (9, 15)
REFERENCE_W_RANGE = (9, 15)
REFERENCE_FILL_RANGE = (0.38, 0.56)
STAMP_COUNT_RANGE = (2, 5)
GRID_H_MIN = 18
GRID_W_MIN = 18
GEN_RETRIES = 400
LAYOUT_RETRIES = 80
SELECT_RETRIES = 200

_RNG = np.random.default_rng()
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = _N8 if connectivity == 8 else _N4
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _local_mask_from_coords(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _extract_color_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components(grid == color, connectivity=8):
        bbox = _bbox_from_coords(coords)
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "mask": _local_mask_from_coords(coords, bbox),
                "size": len(coords),
            }
        )
    return comps


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask, connectivity=8)) == 1


def _fills_its_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return r0 == 0 and c0 == 0 and r1 == mask.shape[0] - 1 and c1 == mask.shape[1] - 1


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(int(x) for x in rot.shape), rot.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(rot.copy())
    return out


def _find_unique_match(ref_mask: np.ndarray, stamp_mask: np.ndarray) -> tuple[int, int, np.ndarray]:
    ref_h, ref_w = ref_mask.shape
    matches: dict[tuple[int, int, int, int, bytes], tuple[int, int, np.ndarray]] = {}
    for rot in _unique_rotations(stamp_mask):
        h, w = rot.shape
        if h > ref_h or w > ref_w:
            continue
        for r in range(ref_h - h + 1):
            for c in range(ref_w - w + 1):
                if np.array_equal(ref_mask[r : r + h, c : c + w], rot):
                    key = (int(r), int(c), int(h), int(w), rot.tobytes())
                    matches[key] = (int(r), int(c), rot.copy())
    assert len(matches) == 1, "every stamp must match exactly one rotated reference window"
    return next(iter(matches.values()))


def _paint_mask(out: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    rr, cc = np.nonzero(mask)
    out[r0 + rr, c0 + cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey reference pattern by matching each small stamp back to its rotated local window.

    Input structure:
    - The background is black.
    - One large connected grey component acts as the reference pattern.
    - Several small monochrome stamps of colors other than black and grey are scattered away from the reference; each stamp fits inside a 5x5 bounding box.

    Rule:
    - Treat the largest grey component as the reference pattern and crop it to its bounding box.
    - For each stamp, try all four 90-degree rotations and find the unique reference subwindow whose grey/black layout matches the rotated stamp mask exactly.
    - Process stamps by descending stamp bounding-box area; ties are broken by the stamp positions from top to bottom, then left to right.
    - Repaint the matched grey cells with the stamp color, clear all original stamp locations to black, and leave uncovered reference cells grey.

    Key insight:
    - The stamp is not matched by translation alone: the solver must compare the full binary local pattern under all rotations to find the only valid landing site.
    """
    g = np.asarray(grid, dtype=int)
    grey_components = _extract_color_components(g, GREY)
    assert grey_components, "grid must contain a grey reference component"
    reference = max(grey_components, key=lambda comp: (comp["size"], -comp["bbox"][0], -comp["bbox"][1]))
    ref_r0, ref_c0, _ref_r1, _ref_c1 = reference["bbox"]
    ref_mask = reference["mask"]

    stamps: list[dict] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in {BG, GREY}):
        for comp in _extract_color_components(g, color):
            r0, c0, r1, c1 = comp["bbox"]
            stamps.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": comp["mask"],
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )

    assert stamps, "grid must contain at least one stamp"
    stamps.sort(key=lambda item: (-item["area"], item["bbox"][0], item["bbox"][1]))

    out = np.full_like(g, BG)
    for r, c in reference["coords"]:
        out[r, c] = GREY

    for stamp in stamps:
        match_r, match_c, match_mask = _find_unique_match(ref_mask, stamp["mask"])
        _paint_mask(out, ref_r0 + match_r, ref_c0 + match_c, match_mask, stamp["color"])

    return out


def _grow_reference(base_h: int, base_w: int) -> np.ndarray | None:
    low = max(24, int(np.ceil(base_h * base_w * REFERENCE_FILL_RANGE[0])))
    high = min(base_h * base_w - 4, int(np.floor(base_h * base_w * REFERENCE_FILL_RANGE[1])))
    if low > high:
        return None
    target = int(_RNG.integers(low, high + 1))

    mask = np.zeros((base_h, base_w), dtype=bool)
    sr = int(_RNG.integers(base_h // 3, base_h - base_h // 3))
    sc = int(_RNG.integers(base_w // 3, base_w - base_w // 3))
    mask[sr, sc] = True
    cells = [(sr, sc)]
    stuck = 0

    while int(mask.sum()) < target and stuck < 1000:
        br, bc = cells[int(_RNG.integers(len(cells)))]
        added = False
        for idx in _RNG.permutation(len(_N4)):
            dr, dc = _N4[int(idx)]
            nr, nc = br + dr, bc + dc
            if 0 <= nr < base_h and 0 <= nc < base_w and not mask[nr, nc]:
                mask[nr, nc] = True
                cells.append((nr, nc))
                added = True
                if _RNG.random() < 0.28 and int(mask.sum()) < target:
                    for extra_r, extra_c in ((nr, nc), (br, bc)):
                        for idx2 in _RNG.permutation(len(_N4)):
                            dr2, dc2 = _N4[int(idx2)]
                            rr, cc = extra_r + dr2, extra_c + dc2
                            if 0 <= rr < base_h and 0 <= cc < base_w and not mask[rr, cc]:
                                mask[rr, cc] = True
                                cells.append((rr, cc))
                                break
                        if int(mask.sum()) >= target:
                            break
                break
        stuck = 0 if added else stuck + 1

    if int(mask.sum()) < low or not _is_connected(mask):
        return None

    cropped, _ = _crop_mask(mask)
    ch, cw = cropped.shape
    if ch < 7 or cw < 7:
        return None
    if cropped.sum() >= ch * cw:
        return None
    return cropped


def _find_stamp_candidates(ref_mask: np.ndarray) -> list[dict]:
    ref_h, ref_w = ref_mask.shape
    limit_h = min(MAX_STAMP_SIDE, ref_h)
    limit_w = min(MAX_STAMP_SIDE, ref_w)
    counts: dict[tuple[int, int, bytes], int] = {}

    for h in range(2, limit_h + 1):
        for w in range(2, limit_w + 1):
            for r in range(ref_h - h + 1):
                for c in range(ref_w - w + 1):
                    sub = ref_mask[r : r + h, c : c + w]
                    key = (int(h), int(w), sub.tobytes())
                    counts[key] = counts.get(key, 0) + 1

    out: list[dict] = []
    for h in range(2, limit_h + 1):
        for w in range(2, limit_w + 1):
            for r in range(ref_h - h + 1):
                for c in range(ref_w - w + 1):
                    sub = ref_mask[r : r + h, c : c + w]
                    area = int(sub.sum())
                    if area < 3 or area == h * w:
                        continue
                    if not _fills_its_bbox(sub) or not _is_connected(sub):
                        continue
                    orbit = {
                        (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
                        for rot in _unique_rotations(sub)
                    }
                    total = sum(counts.get(key, 0) for key in orbit)
                    if total != 1:
                        continue
                    cells = frozenset(
                        (int(r + dr), int(c + dc)) for dr, dc in np.argwhere(sub)
                    )
                    out.append(
                        {
                            "r": int(r),
                            "c": int(c),
                            "h": int(h),
                            "w": int(w),
                            "area": int(h * w),
                            "mask": sub.copy(),
                            "cells": cells,
                        }
                    )
    return out


def _has_overlap(chosen: list[dict]) -> bool:
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if chosen[i]["cells"] & chosen[j]["cells"]:
                return True
    return False


def _sample_candidates(candidates: list[dict]) -> list[dict] | None:
    max_count = min(STAMP_COUNT_RANGE[1] - 1, len(STAMP_COLORS), len(candidates))
    if max_count < STAMP_COUNT_RANGE[0]:
        return None

    count = int(_RNG.integers(STAMP_COUNT_RANGE[0], max_count + 1))
    overlap_possible = any(
        candidates[i]["cells"] & candidates[j]["cells"]
        for i in range(len(candidates))
        for j in range(i + 1, len(candidates))
    )
    require_overlap = bool(count >= 2 and overlap_possible and _RNG.random() < 0.5)

    for _ in range(SELECT_RETRIES):
        picks = _RNG.choice(len(candidates), size=count, replace=False)
        chosen = [candidates[int(i)] for i in picks]
        overlaps = _has_overlap(chosen)
        if require_overlap and not overlaps:
            continue
        if not require_overlap and overlaps:
            continue
        return chosen
    return None


def _bbox_is_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _make_zones(count: int) -> list[str]:
    zones = ["right", "bottom"]
    while len(zones) < count:
        zones.append("right" if _RNG.random() < 0.5 else "bottom")
    _RNG.shuffle(zones)
    return zones


def _place_stamps(
    grid_h: int,
    grid_w: int,
    ref_r0: int,
    ref_c0: int,
    ref_h: int,
    ref_w: int,
    stamps: list[dict],
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve_bbox(occupied, ref_r0, ref_c0, ref_h, ref_w)
    placements: list[tuple[int, int] | None] = [None] * len(stamps)
    zones = _make_zones(len(stamps))
    order = sorted(
        range(len(stamps)),
        key=lambda i: (
            -(stamps[i]["placed_mask"].shape[0] * stamps[i]["placed_mask"].shape[1]),
            -stamps[i]["placed_mask"].shape[0],
            -stamps[i]["placed_mask"].shape[1],
        ),
    )

    for idx in order:
        sh, sw = stamps[idx]["placed_mask"].shape
        zone = zones[idx]
        if zone == "right":
            r_lo, r_hi = 1, grid_h - sh - 1
            c_lo, c_hi = ref_c0 + ref_w + 2, grid_w - sw - 1
        else:
            r_lo, r_hi = ref_r0 + ref_h + 2, grid_h - sh - 1
            c_lo, c_hi = 1, grid_w - sw - 1
        if r_lo > r_hi or c_lo > c_hi:
            return None

        candidates = [
            (int(r), int(c))
            for r in range(r_lo, r_hi + 1)
            for c in range(c_lo, c_hi + 1)
            if _bbox_is_clear(occupied, int(r), int(c), sh, sw)
        ]
        if not candidates:
            return None
        pick = candidates[int(_RNG.integers(len(candidates)))]
        placements[idx] = pick
        _reserve_bbox(occupied, pick[0], pick[1], sh, sw)

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _build_expected_output(
    grid_h: int,
    grid_w: int,
    ref_r0: int,
    ref_c0: int,
    ref_mask: np.ndarray,
    stamps: list[dict],
) -> np.ndarray:
    out = np.full((grid_h, grid_w), BG, dtype=int)
    _paint_mask(out, ref_r0, ref_c0, ref_mask, GREY)

    ordered = sorted(
        stamps,
        key=lambda item: (
            -int(item["placed_mask"].shape[0] * item["placed_mask"].shape[1]),
            int(item["stamp_r0"]),
            int(item["stamp_c0"]),
        ),
    )
    for item in ordered:
        _paint_mask(
            out,
            ref_r0 + int(item["candidate"]["r"]),
            ref_c0 + int(item["candidate"]["c"]),
            item["candidate"]["mask"],
            int(item["color"]),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    base_h = int(_RNG.integers(*REFERENCE_H_RANGE))
    base_w = int(_RNG.integers(*REFERENCE_W_RANGE))
    ref_mask = _grow_reference(base_h, base_w)
    if ref_mask is None:
        return None

    candidates = _find_stamp_candidates(ref_mask)
    if len(candidates) < STAMP_COUNT_RANGE[0]:
        return None

    chosen = _sample_candidates(candidates)
    if chosen is None:
        return None

    colors = list(_RNG.choice(STAMP_COLORS, size=len(chosen), replace=False))
    rotations = [int(_RNG.integers(0, 4)) for _ in chosen]
    if len(chosen) >= 2 and not any(rot % 2 == 1 for rot in rotations) and _RNG.random() < 0.5:
        rotations[int(_RNG.integers(len(rotations)))] = int(_RNG.choice([1, 3]))

    stamps: list[dict] = []
    for candidate, color, rot in zip(chosen, colors, rotations):
        placed_mask = np.rot90(candidate["mask"], rot)
        stamps.append(
            {
                "candidate": candidate,
                "color": int(color),
                "rotation": int(rot),
                "placed_mask": placed_mask.copy(),
            }
        )

    ref_h, ref_w = ref_mask.shape
    min_h = max(GRID_H_MIN, ref_h + 8)
    min_w = max(GRID_W_MIN, ref_w + 8)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(LAYOUT_RETRIES):
        grid_h = int(_RNG.integers(min_h, 31))
        grid_w = int(_RNG.integers(min_w, 31))
        max_ref_r0 = grid_h - ref_h - 6
        max_ref_c0 = min(grid_w - ref_w - 6, 6)
        if max_ref_r0 < 1 or max_ref_c0 < 1:
            continue
        ref_r0 = int(_RNG.integers(1, max_ref_r0 + 1))
        ref_c0 = int(_RNG.integers(1, max_ref_c0 + 1))

        placements = _place_stamps(grid_h, grid_w, ref_r0, ref_c0, ref_h, ref_w, stamps)
        if placements is None or len(placements) != len(stamps):
            continue

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        _paint_mask(grid, ref_r0, ref_c0, ref_mask, GREY)

        for item, (stamp_r0, stamp_c0) in zip(stamps, placements):
            item["stamp_r0"] = int(stamp_r0)
            item["stamp_c0"] = int(stamp_c0)
            _paint_mask(grid, stamp_r0, stamp_c0, item["placed_mask"], item["color"])

        expected = _build_expected_output(grid_h, grid_w, ref_r0, ref_c0, ref_mask, stamps)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
