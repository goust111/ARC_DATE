from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
_QUADRANT_NAMES = ("ul", "ur", "ll", "lr")
_OPPOSITE = {0: 3, 1: 2, 2: 1, 3: 0}
_GEN_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _split_quadrants(panel: np.ndarray) -> list[np.ndarray]:
    h, w = panel.shape
    assert h % 2 == 0 and w % 2 == 0
    mh, mw = h // 2, w // 2
    return [
        panel[:mh, :mw].copy(),
        panel[:mh, mw:].copy(),
        panel[mh:, :mw].copy(),
        panel[mh:, mw:].copy(),
    ]


def _largest_red_rect_area(quad: np.ndarray) -> int:
    mask = quad == RED
    h, w = quad.shape
    best = 0
    for top in range(h):
        full_cols = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            full_cols &= mask[bottom]
            col = 0
            while col < w:
                while col < w and not full_cols[col]:
                    col += 1
                if col == w:
                    break
                left = col
                while col < w and full_cols[col]:
                    col += 1
                right = col - 1
                rect_h = bottom - top + 1
                rect_w = right - left + 1
                if rect_h >= 2 and rect_w >= 2:
                    best = max(best, rect_h * rect_w)
    return best


def _all_zero_rectangles(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = arr == BLACK
    h, w = arr.shape
    rects: list[tuple[int, int, int, int]] = []
    for top in range(h):
        full_cols = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            full_cols &= mask[bottom]
            col = 0
            while col < w:
                while col < w and not full_cols[col]:
                    col += 1
                if col == w:
                    break
                left = col
                while col < w and full_cols[col]:
                    col += 1
                right = col - 1
                rect_h = bottom - top + 1
                rect_w = right - left + 1
                if rect_h >= 2 and rect_w >= 2:
                    rects.append((top, left, bottom, right))
    return rects


def _make_panel(source: np.ndarray) -> np.ndarray:
    top = np.concatenate([source, np.fliplr(source)], axis=1)
    bottom = np.concatenate([np.flipud(source), np.flipud(np.fliplr(source))], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    out = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in _DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not mask[nr, nc]:
                out.add((nr, nc))
    return list(out)


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        steps = []
        if r < er:
            steps.append((r + 1, c))
        elif r > er:
            steps.append((r - 1, c))
        if c < ec:
            steps.append((r, c + 1))
        elif c > ec:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        mask[r, c] = True


def _random_source(qh: int, qw: int) -> np.ndarray | None:
    cells = qh * qw
    for _ in range(200):
        mask = np.zeros((qh, qw), dtype=bool)
        anchors = [
            (0, random.randrange(qw)),
            (qh - 1, random.randrange(qw)),
            (random.randrange(qh), 0),
            (random.randrange(qh), qw - 1),
        ]
        mask[anchors[0]] = True
        current = anchors[0]
        for anchor in anchors[1:]:
            _carve_path(mask, current, anchor)
            current = anchor

        lo = max(int(np.ceil(0.38 * cells)), int(mask.sum()) + 1)
        hi = min(int(np.floor(0.62 * cells)), cells - 8)
        if lo > hi:
            continue
        target = random.randint(lo, hi)
        while int(mask.sum()) < target:
            choices = _frontier(mask)
            if not choices:
                break
            mask[random.choice(choices)] = True

        source = np.where(mask, BLUE, BLACK)
        if not _connected(mask):
            continue
        if (source[0] == BLUE).sum() == 0 or (source[-1] == BLUE).sum() == 0:
            continue
        if (source[:, 0] == BLUE).sum() == 0 or (source[:, -1] == BLUE).sum() == 0:
            continue
        if int((source == BLACK).sum()) < 8:
            continue
        if not _all_zero_rectangles(source):
            continue
        if np.array_equal(source, np.fliplr(source)):
            continue
        if np.array_equal(source, np.flipud(source)):
            continue
        if np.array_equal(source, np.rot90(source, 2)):
            continue
        return source
    return None


def _pick_rect(quad: np.ndarray) -> tuple[int, int, int, int] | None:
    rects = _all_zero_rectangles(quad)
    if not rects:
        return None
    rects = sorted(rects, key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1), b))
    viable = [b for b in rects if (b[2] - b[0] + 1) * (b[3] - b[1] + 1) <= 9]
    pool = viable[-4:] if viable else rects[-4:]
    return random.choice(pool)


def _noise_count(opposite: bool) -> int:
    if opposite:
        return random.choices([0, 1, 2], weights=[1, 4, 3], k=1)[0]
    return random.choices([0, 1, 2], weights=[3, 3, 2], k=1)[0]


def _place_single_noise(quad: np.ndarray, clean_quad: np.ndarray, count: int) -> bool:
    candidates = [
        (int(r), int(c))
        for r, c in np.argwhere((clean_quad == BLACK) & (quad == BLACK))
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
            continue
        quad[r, c] = RED
        chosen.append((r, c))
        if len(chosen) == count:
            return True
    return len(chosen) == count


def _clean_red(quad: np.ndarray) -> np.ndarray:
    out = quad.copy()
    out[out == RED] = BLACK
    return out


def _analyze(grid: np.ndarray) -> dict:
    box = _bbox(grid != BLACK)
    assert box is not None, "expected one non-black panel"
    panel = _crop_bbox(grid, box)
    h, w = panel.shape
    assert h % 2 == 0 and w % 2 == 0, "cropped panel must split into four equal quadrants"
    quads = _split_quadrants(panel)

    ul_blue = quads[0] == BLUE
    assert np.any(ul_blue), "quadrant source pattern must contain blue pixels"
    assert np.array_equal(quads[1] == BLUE, np.fliplr(ul_blue)), "UR blue pattern must mirror UL"
    assert np.array_equal(quads[2] == BLUE, np.flipud(ul_blue)), "LL blue pattern must mirror UL"
    assert np.array_equal(quads[3] == BLUE, np.flipud(np.fliplr(ul_blue))), "LR blue pattern must mirror UL"

    areas = [_largest_red_rect_area(q) for q in quads]
    max_area = max(areas)
    assert max_area >= 4, "one quadrant must contain a red rectangle of at least 2x2"
    polluted = [i for i, area in enumerate(areas) if area == max_area]
    assert len(polluted) == 1, "largest red rectangle must identify a unique polluted quadrant"
    polluted_idx = polluted[0]
    assert sum(area > 0 for area in areas) == 1, "only the polluted quadrant may contain a 2x2-or-larger red rectangle"

    cleaned = [_clean_red(q) for q in quads]
    output = cleaned[_OPPOSITE[polluted_idx]]
    assert np.any(output == BLUE), "output quadrant must preserve blue structure"
    assert not np.any(output == RED), "output quadrant must have red noise erased"
    assert all(not np.array_equal(output, cleaned[i]) for i in range(4) if i != _OPPOSITE[polluted_idx]), (
        "quadrant orientations must remain distinguishable so the diagonal choice matters"
    )

    return {
        "panel": panel,
        "quads": quads,
        "cleaned_quads": cleaned,
        "polluted_idx": polluted_idx,
        "output": output,
        "rect_areas": areas,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the diagonal clean quadrant from a mirrored panel whose noisiest quadrant is marked by a large red rectangle.

    Input structure:
    - The grid has a black background and one central non-black panel.
    - The panel has even height and width and is divided into four equal quadrants.
    - Blue pixels form a four-way mirrored pattern: upper-right is a left-right reflection of upper-left,
      lower-left is an up-down reflection, and lower-right is a 180-degree reflection.
    - Red noise is overlaid on the panel; exactly one quadrant contains the unique largest solid red rectangle.

    Rule:
    - Crop to the bounding box of all non-black pixels.
    - Split the cropped panel into upper-left, upper-right, lower-left, and lower-right quadrants.
    - In each quadrant, find the largest axis-aligned solid red rectangle with height and width at least two.
    - Identify the quadrant containing the unique largest such rectangle as the polluted quadrant.
    - Output the diagonally opposite quadrant and replace any red pixels in that quadrant with black.

    Key insight:
    - The blue structure only tells you how the four quadrants correspond under reflection;
      the large red rectangle is used to exclude one quadrant, and the answer is the opposite mirrored copy after erasing residual red specks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    analysis = _analyze(g)
    return analysis["output"].copy()


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(5, 6)
    qw = random.randint(5, 6)
    source = _random_source(qh, qw)
    if source is None:
        return None

    panel = _make_panel(source)
    quads = _split_quadrants(panel)

    polluted_idx = random.randrange(4)
    rect = _pick_rect(quads[polluted_idx])
    if rect is None:
        return None
    r0, c0, r1, c1 = rect
    quads[polluted_idx][r0 : r1 + 1, c0 : c1 + 1] = RED

    polluted_clean = _split_quadrants(panel)[polluted_idx]
    polluted_extras = random.choices([0, 1], weights=[2, 1], k=1)[0]
    if not _place_single_noise(quads[polluted_idx], polluted_clean, polluted_extras):
        return None

    for idx in range(4):
        if idx == polluted_idx:
            continue
        count = _noise_count(idx == _OPPOSITE[polluted_idx])
        if not _place_single_noise(quads[idx], _split_quadrants(panel)[idx], count):
            return None

    damaged_panel = np.block(
        [
            [quads[0], quads[1]],
            [quads[2], quads[3]],
        ]
    )
    canvas_h = random.randint(max(damaged_panel.shape[0] + 2, 14), min(damaged_panel.shape[0] + 8, 20))
    canvas_w = random.randint(max(damaged_panel.shape[1] + 2, 14), min(damaged_panel.shape[1] + 8, 20))
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    top = random.randint(1, canvas_h - damaged_panel.shape[0] - 1)
    left = random.randint(1, canvas_w - damaged_panel.shape[1] - 1)
    grid[top : top + damaged_panel.shape[0], left : left + damaged_panel.shape[1]] = damaged_panel

    try:
        analysis = _analyze(grid)
    except AssertionError:
        return None

    if analysis["polluted_idx"] != polluted_idx:
        return None

    expected = _clean_red(quads[_OPPOSITE[polluted_idx]])
    if not np.array_equal(main(grid), expected):
        return None

    # Non-degeneracy checks: each semantic step must matter on generated instances.
    raw_opposite = quads[_OPPOSITE[polluted_idx]]
    if np.any(raw_opposite == RED) and np.array_equal(raw_opposite, expected):
        return None
    if np.array_equal(expected, _clean_red(quads[polluted_idx])):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
