from __future__ import annotations

import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, mask.shape)), tuple(map(int, mask.ravel()))


def _orientation_map(mask: np.ndarray, *, strict: bool = True) -> list[tuple[np.ndarray, str]]:
    oriented: dict[tuple[tuple[int, int], tuple[int, ...]], tuple[np.ndarray, str]] = {}
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = _mask_key(rotated)
        if key in oriented:
            if strict and oriented[key][1] != "rotation":
                raise AssertionError("ambiguous source mask chirality")
            continue
        oriented[key] = (rotated.copy(), "rotation")

    mirrored = np.fliplr(mask)
    for k in range(4):
        transformed = np.rot90(mirrored, k)
        key = _mask_key(transformed)
        if key in oriented:
            if strict and oriented[key][1] != "mirror":
                raise AssertionError("ambiguous source mask chirality")
            continue
        oriented[key] = (transformed.copy(), "mirror")
    return list(oriented.values())


def _has_full_chirality(mask: np.ndarray) -> bool:
    return len(_orientation_map(mask, strict=False)) == 8


def _find_source(grid: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int, int, int], np.ndarray]:
    candidates: list[tuple[int, int, tuple[int, int, int, int], np.ndarray]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in components_mask(grid == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(coords)
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(region)))
            if bg in colors or len(colors) != 2 or color not in colors:
                continue
            if int(region[0, 0]) != color or int(region[0, -1]) != color or int(region[-1, 0]) != color or int(region[-1, -1]) != color:
                continue
            if int(np.count_nonzero(region == color)) != len(coords):
                continue
            marker_color = next(iter(colors - {color}))
            marker_mask = region == marker_color
            if not marker_mask.any() or marker_mask.all():
                continue
            if not _has_full_chirality(marker_mask):
                continue
            candidates.append((int(color), int(marker_color), (r0, c0, r1, c1), marker_mask.copy()))

    assert len(candidates) == 1, "expected exactly one asymmetric source rectangle"
    return candidates[0]


def _find_matches(
    grid: np.ndarray,
    bg: int,
    marker_color: int,
    marker_mask: np.ndarray,
) -> list[dict[str, object]]:
    H, W = map(int, grid.shape)
    matches: list[dict[str, object]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()

    for transformed, chirality in _orientation_map(marker_mask):
        h, w = map(int, transformed.shape)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                region = grid[r0 : r0 + h, c0 : c0 + w]
                if not np.all(region[transformed] == marker_color):
                    continue
                if not np.all(region[~transformed] == bg):
                    continue
                box = (r0, c0, h, w)
                assert box not in seen_boxes, "match should have unique chirality"
                seen_boxes.add(box)
                matches.append(
                    {
                        "r0": int(r0),
                        "c0": int(c0),
                        "mask": transformed.copy(),
                        "chirality": chirality,
                    }
                )
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source sprite onto matching marker constellations, with mirrored copies producing negative prints.

    Input structure:
    - A dominant background contains one solid rectangular source sprite built from a fill color with embedded marker-color pixels.
    - Elsewhere, scattered marker-color pixels include exact rigid transforms of the source marker mask, each inside an otherwise empty rectangle.

    Rule:
    - Find the unique source rectangle and extract its marker mask.
    - Enumerate all eight rigid transforms of that mask and test every in-bounds rectangle for an exact marker-only match.
    - For a pure rotation match, fill the rectangle's background cells with the source fill color; for a mirrored match, swap roles inside that rectangle so marker cells become fill color and the remaining cells become marker color.

    Key insight:
    - The same marker constellation geometry carries a hidden chirality bit: rotation means a normal stamp, while reflection means an inverted stamp.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fill_color, marker_color, _source_bbox, marker_mask = _find_source(g, bg)
    matches = _find_matches(g, bg, marker_color, marker_mask)

    out = g.copy()
    for match in matches:
        r0 = int(match["r0"])
        c0 = int(match["c0"])
        transformed = np.asarray(match["mask"], dtype=bool)
        h, w = transformed.shape
        patch = out[r0 : r0 + h, c0 : c0 + w]
        if match["chirality"] == "rotation":
            patch[~transformed] = fill_color
        else:
            patch[transformed] = fill_color
            patch[~transformed] = marker_color
    return out


GEN_RETRY_LIMIT = 400
GRID_MIN = 18
GRID_MAX = 30
SOURCE_MIN_SIDE = 4
SOURCE_MAX_SIDE = 7
MARKER_MIN = 3
MARKER_MAX = 6
MATCH_MIN = 2
MATCH_MAX = 5
DISTRACTOR_MIN = 4
DISTRACTOR_MAX = 14


def _mask_from_cells(shape: tuple[int, int], cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _fill_is_connected(marker_mask: np.ndarray) -> bool:
    fill_mask = ~marker_mask
    return len(components_mask(fill_mask, connectivity=4)) == 1


def _make_source_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(SOURCE_MIN_SIDE, SOURCE_MAX_SIDE)
        w = random.randint(SOURCE_MIN_SIDE, SOURCE_MAX_SIDE)
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        if len(interior) < MARKER_MIN:
            continue
        n_marks = random.randint(MARKER_MIN, min(MARKER_MAX, len(interior)))
        cells = random.sample(interior, n_marks)
        mask = _mask_from_cells((h, w), cells)
        if not _has_full_chirality(mask):
            continue
        if not _fill_is_connected(mask):
            continue
        if len(components_mask(mask, connectivity=4)) > n_marks:
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric source mask")


def _placement_index(
    grid_shape: tuple[int, int],
    marker_mask: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], list[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]]]]:
    H, W = map(int, grid_shape)
    placements: list[dict[str, object]] = []
    by_cell: dict[tuple[int, int], list[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]]] = {}

    for transformed, chirality in _orientation_map(marker_mask):
        h, w = map(int, transformed.shape)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                marker_cells = frozenset(
                    (r0 + rr, c0 + cc)
                    for rr in range(h)
                    for cc in range(w)
                    if transformed[rr, cc]
                )
                fill_cells = frozenset(
                    (r0 + rr, c0 + cc)
                    for rr in range(h)
                    for cc in range(w)
                    if not transformed[rr, cc]
                )
                footprint = marker_cells | fill_cells
                rec = {
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": transformed.copy(),
                    "chirality": chirality,
                    "marker_cells": marker_cells,
                    "fill_cells": fill_cells,
                    "footprint": footprint,
                    "key": (marker_cells, fill_cells),
                }
                placements.append(rec)
                for cell in marker_cells:
                    by_cell.setdefault(cell, []).append((marker_cells, fill_cells))
    return placements, by_cell


def _valid_match_keys(
    grid: np.ndarray,
    bg: int,
    marker_color: int,
    placements: list[dict[str, object]],
) -> set[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]]:
    keys: set[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]] = set()
    for rec in placements:
        marker_cells = rec["marker_cells"]
        fill_cells = rec["fill_cells"]
        if all(int(grid[r, c]) == marker_color for r, c in marker_cells) and all(int(grid[r, c]) == bg for r, c in fill_cells):
            keys.add((marker_cells, fill_cells))
    return keys


def _safe_distractor_cell(
    grid: np.ndarray,
    bg: int,
    marker_color: int,
    cell: tuple[int, int],
    allowed_keys: set[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]],
    by_cell: dict[tuple[int, int], list[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]]],
) -> bool:
    r, c = cell
    for marker_cells, fill_cells in by_cell.get((r, c), []):
        for mr, mc in marker_cells:
            if (mr, mc) != (r, c) and int(grid[mr, mc]) != marker_color:
                break
        else:
            for fr, fc in fill_cells:
                if int(grid[fr, fc]) != bg:
                    break
            else:
                if (marker_cells, fill_cells) not in allowed_keys:
                    return False
    return True


def _apply_expected(
    inp: np.ndarray,
    fill_color: int,
    marker_color: int,
    chosen: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for rec in chosen:
        r0 = int(rec["r0"])
        c0 = int(rec["c0"])
        transformed = np.asarray(rec["mask"], dtype=bool)
        patch = out[r0 : r0 + transformed.shape[0], c0 : c0 + transformed.shape[1]]
        if rec["chirality"] == "rotation":
            patch[~transformed] = fill_color
        else:
            patch[transformed] = fill_color
            patch[~transformed] = marker_color
    return out


def _choose_match_counts() -> tuple[int, int]:
    mode = random.choices(
        ["both", "rotation_only", "mirror_only"],
        weights=[0.6, 0.2, 0.2],
        k=1,
    )[0]
    total = random.randint(MATCH_MIN, MATCH_MAX)
    if mode == "both":
        rot = random.randint(1, total - 1)
        mir = total - rot
        return rot, mir
    if mode == "rotation_only":
        return total, 0
    return 0, total


def _pick_placements(
    placements: list[dict[str, object]],
    source_footprint: frozenset[tuple[int, int]],
    rot_needed: int,
    mir_needed: int,
) -> list[dict[str, object]] | None:
    blocked = set(source_footprint)
    chosen: list[dict[str, object]] = []
    for chirality, needed in [("rotation", rot_needed), ("mirror", mir_needed)]:
        candidates = [p for p in placements if p["chirality"] == chirality]
        random.shuffle(candidates)
        picked = 0
        for rec in candidates:
            if rec["footprint"] & blocked:
                continue
            chosen.append(rec)
            blocked |= set(rec["footprint"])
            picked += 1
            if picked == needed:
                break
        if picked != needed:
            return None
    random.shuffle(chosen)
    return chosen


def _make_attempt() -> np.ndarray | None:
    marker_mask = _make_source_mask()
    h, w = map(int, marker_mask.shape)
    H = random.randint(max(GRID_MIN, h + 7), GRID_MAX)
    W = random.randint(max(GRID_MIN, w + 7), GRID_MAX)
    bg, fill_color, marker_color = random.sample(range(10), 3)

    inp = np.full((H, W), bg, dtype=int)
    sr = random.randint(0, H - h)
    sc = random.randint(0, W - w)
    source_patch = np.where(marker_mask, marker_color, fill_color)
    inp[sr : sr + h, sc : sc + w] = source_patch
    source_footprint = frozenset((sr + rr, sc + cc) for rr in range(h) for cc in range(w))

    placements, by_cell = _placement_index((H, W), marker_mask)
    rot_needed, mir_needed = _choose_match_counts()
    chosen = _pick_placements(placements, source_footprint, rot_needed, mir_needed)
    if chosen is None:
        return None

    for rec in chosen:
        for r, c in rec["marker_cells"]:
            inp[r, c] = marker_color

    if get_bg(inp) != bg:
        return None

    chosen_keys = {rec["key"] for rec in chosen}
    valid_keys = _valid_match_keys(inp, bg, marker_color, placements)
    if valid_keys != chosen_keys:
        return None

    out = _apply_expected(inp, fill_color, marker_color, chosen)
    if not np.array_equal(main(inp), out):
        return None

    distractor_target = random.randint(DISTRACTOR_MIN, DISTRACTOR_MAX)
    for _ in range(distractor_target):
        candidates = [
            (r, c)
            for r in range(H)
            for c in range(W)
            if int(inp[r, c]) == bg and _safe_distractor_cell(inp, bg, marker_color, (r, c), chosen_keys, by_cell)
        ]
        if not candidates:
            break
        r, c = random.choice(candidates)
        inp[r, c] = marker_color
        out[r, c] = marker_color

    if get_bg(inp) != bg:
        return None

    valid_keys = _valid_match_keys(inp, bg, marker_color, placements)
    if valid_keys != chosen_keys:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if len(set(map(int, np.unique(inp)))) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
