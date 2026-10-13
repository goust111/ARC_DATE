from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
FRAME = 5
BRIDGE = 8
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _component_cells(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components_mask(grid == color)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _component_cells(grid, FRAME):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        expected = set(_frame_border_cells(box))
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, "expected each grey object to be a 1-pixel rectangular frame"
        assert r1 - r0 >= 3 and c1 - c0 >= 3, "frame must enclose a non-empty interior"
        frames.append(
            {
                "bbox": box,
                "interior_r0": r0 + 1,
                "interior_c0": c0 + 1,
                "interior_h": r1 - r0 - 1,
                "interior_w": c1 - c0 - 1,
                "border_cells": expected,
            }
        )
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    return frames


def _markers_in_frame(grid: np.ndarray, frame: dict) -> dict[tuple[int, int], int]:
    r0 = frame["interior_r0"]
    c0 = frame["interior_c0"]
    h = frame["interior_h"]
    w = frame["interior_w"]
    markers: dict[tuple[int, int], int] = {}
    for rr in range(h):
        for cc in range(w):
            color = int(grid[r0 + rr, c0 + cc])
            if color != BG:
                markers[(rr, cc)] = color
    return markers


def _bridge_pairs(grid: np.ndarray, frames: list[dict]) -> tuple[list[tuple[int, int]], set[int]]:
    border_owner: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame["border_cells"]:
            border_owner[cell] = idx

    pairs: list[tuple[int, int]] = []
    bridged = set()
    for coords in _component_cells(grid, BRIDGE):
        touched: set[int] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                owner = border_owner.get((nr, nc))
                if owner is not None:
                    touched.add(owner)
        assert len(touched) == 2, "each bridge must touch exactly two frames"
        a, b = sorted(touched)
        assert a not in bridged and b not in bridged, "each frame may belong to at most one bridge"
        bridged.add(a)
        bridged.add(b)
        pairs.append((a, b))

    pairs.sort(key=lambda pair: (frames[pair[0]]["bbox"][0], frames[pair[0]]["bbox"][1]))
    return pairs, bridged


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route marker pixels inside rectangular frames either to majority-color fills or to mirrored copies based on bridge connectivity.

    Input structure:
    - The grid has a black background with 2 to 4 grey 1-pixel rectangular outline frames.
    - Each frame encloses 2 to 4 colored marker pixels, and some frame pairs are linked by single-pixel teal bridge paths touching their outer borders.
    - Bridge-connected pairs have matching interior dimensions; isolated frames have a unique majority marker color.

    Rule:
    - Detect all grey frames and group any two frames touched by the same teal 4-connected bridge into one connected pair.
    - For each connected pair, order the two frames by top-left position, keep the first frame unchanged, and overwrite the second frame's interior with the first frame's markers mirrored horizontally within the interior box.
    - For each isolated frame, fill its whole interior with its unique majority marker color; then erase every teal bridge to background.

    Key insight:
    - The same marker pixels have two different meanings: bridge-connected frames use marker positions as a geometric template, while isolated frames use only marker color frequencies as a voting signal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    assert 2 <= len(frames) <= 4, "expected 2 to 4 frames"
    pairs, bridged = _bridge_pairs(g, frames)

    out = g.copy()
    out[out == BRIDGE] = BG

    for left_idx, right_idx in pairs:
        left = frames[left_idx]
        right = frames[right_idx]
        assert left["interior_h"] == right["interior_h"] and left["interior_w"] == right["interior_w"]
        markers = _markers_in_frame(g, left)
        rr0 = right["interior_r0"]
        rc0 = right["interior_c0"]
        rh = right["interior_h"]
        rw = right["interior_w"]
        out[rr0 : rr0 + rh, rc0 : rc0 + rw] = BG
        for (r, c), color in markers.items():
            out[rr0 + r, rc0 + (rw - 1 - c)] = color

    for idx, frame in enumerate(frames):
        if idx in bridged:
            continue
        markers = list(_markers_in_frame(g, frame).values())
        assert markers, "isolated frame must contain markers"
        counts = Counter(markers).most_common()
        assert len(counts) == 1 or counts[0][1] > counts[1][1], "isolated frame must have a unique majority"
        fill = int(counts[0][0])
        r0 = frame["interior_r0"]
        c0 = frame["interior_c0"]
        h = frame["interior_h"]
        w = frame["interior_w"]
        out[r0 : r0 + h, c0 : c0 + w] = fill

    return out


@lru_cache(maxsize=None)
def _valid_marker_sets(h: int, w: int) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    by_size: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for k in range(2, min(4, len(cells)) + 1):
        combos: list[tuple[tuple[int, int], ...]] = []
        for combo in itertools.combinations(cells, k):
            ok = True
            for (r1, c1), (r2, c2) in itertools.combinations(combo, 2):
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    ok = False
                    break
            if ok:
                combos.append(combo)
        if combos:
            by_size[k] = combos
    return by_size


def _random_marker_layout(h: int, w: int, *, majority: bool) -> dict[tuple[int, int], int]:
    options = _valid_marker_sets(h, w)
    assert options, "frame interior too small for isolated markers"
    k = random.choice(list(options))
    positions = random.choice(options[k])

    if not majority:
        colors = [random.choice(MARKER_COLORS) for _ in range(k)]
        return {pos: color for pos, color in zip(positions, colors)}

    if k == 2:
        maj_color = random.choice(MARKER_COLORS)
        return {pos: maj_color for pos in positions}

    maj_color = random.choice(MARKER_COLORS)
    maj_count = 2 if k == 3 else random.choice((2, 3))
    colors = [maj_color] * maj_count
    remaining = k - maj_count
    others = [c for c in MARKER_COLORS if c != maj_color]
    if remaining == 2 and maj_count == 2:
        colors.extend(random.sample(others, 2))
    elif remaining == 2:
        colors.extend(random.choices(others, k=2))
    elif remaining == 1:
        colors.append(random.choice(others))
    random.shuffle(colors)
    layout = {pos: color for pos, color in zip(positions, colors)}
    counts = Counter(layout.values()).most_common()
    assert len(counts) == 1 or counts[0][1] > counts[1][1]
    return layout


def _mirror_layout(layout: dict[tuple[int, int], int], width: int) -> dict[tuple[int, int], int]:
    return {(r, width - 1 - c): color for (r, c), color in layout.items()}


def _frame_patch(h: int, w: int, layout: dict[tuple[int, int], int]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    for (r, c), color in layout.items():
        patch[1 + r, 1 + c] = color
    return patch


def _paired_cluster() -> np.ndarray:
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    ih = h - 2
    iw = w - 2
    left_layout = _random_marker_layout(ih, iw, majority=False)
    mirrored = _mirror_layout(left_layout, iw)

    for _ in range(50):
        right_layout = _random_marker_layout(ih, iw, majority=False)
        if right_layout != mirrored:
            break
    else:
        right_layout = {(r, c): random.choice(MARKER_COLORS) for (r, c) in mirrored}
        if right_layout == mirrored:
            one = next(iter(right_layout))
            right_layout[one] = MARKER_COLORS[(MARKER_COLORS.index(right_layout[one]) + 1) % len(MARKER_COLORS)]

    gap = random.randint(2, 6)
    row = random.randint(1, h - 2)
    patch = np.zeros((h, 2 * w + gap), dtype=int)
    patch[:, :w] = _frame_patch(h, w, left_layout)
    right_c0 = w + gap
    patch[:, right_c0 : right_c0 + w] = _frame_patch(h, w, right_layout)
    patch[row, w:right_c0] = BRIDGE
    return patch


def _single_cluster() -> np.ndarray:
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    layout = _random_marker_layout(h - 2, w - 2, majority=True)
    return _frame_patch(h, w, layout)


def _scenario() -> tuple[int, int]:
    choices = [
        (2, 0),
        (2, 1),
        (3, 1),
        (4, 1),
        (4, 2),
    ]
    weights = [1, 1, 4, 4, 2]
    return random.choices(choices, weights=weights, k=1)[0]


def _fits(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, padding: int) -> bool:
    H, W = grid.shape
    h, w = patch.shape
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _fits(occupied, patch, r0, c0, padding):
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        occupied[r0 : r0 + h, c0 : c0 + w][mask] = True
        return True
    return False


def _generate_once() -> np.ndarray | None:
    n_frames, n_pairs = _scenario()
    clusters = [_paired_cluster() for _ in range(n_pairs)] + [_single_cluster() for _ in range(n_frames - 2 * n_pairs)]
    random.shuffle(clusters)

    max_h = max(p.shape[0] for p in clusters)
    max_w = max(p.shape[1] for p in clusters)
    H = random.randint(max(16, max_h + 2), 24)
    W = random.randint(max(16, max_w + 2), 24)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for patch in sorted(clusters, key=lambda p: p.shape[0] * p.shape[1], reverse=True):
        if not _place_patch(grid, occupied, patch, padding=2):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid == BRIDGE) == 0 and n_pairs > 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge/fill/mirror ARC input")
