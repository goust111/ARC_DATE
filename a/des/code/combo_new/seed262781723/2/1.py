from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GRAY = 5
GEN_ATTEMPTS = 400
GEN_PADDING_RANGE = (4, 7)  # high-exclusive
GEN_QUADRANT_RANGE = (4, 7)  # high-exclusive
GEN_PATTERN_SIDE_MAX = 5
DIR_TO_QUADRANT = {
    "up": "ul",
    "right": "ur",
    "left": "ll",
    "down": "lr",
}
DIR_TO_CORNER = {
    "up": "tl",
    "right": "tr",
    "left": "bl",
    "down": "br",
}
COLOR_POOL = tuple(c for c in range(10) if c not in {BG, GRAY})
MARKER_LIBRARY: dict[int, list[np.ndarray]] = {
    1: [np.array([[1]], dtype=bool)],
    2: [
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
    ],
    3: [
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
    ],
    4: [np.array([[1, 1], [1, 1]], dtype=bool)],
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(patch: np.ndarray, bg: int = BG) -> np.ndarray:
    mask = patch != int(bg)
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_gap(side: np.ndarray) -> tuple[int, int] | None:
    if bool(side.all()):
        return None
    if not (bool(side[0]) and bool(side[-1])):
        return None
    gap_idx = np.flatnonzero(~side)
    if gap_idx.size < 2:
        return None
    if int(gap_idx[-1] - gap_idx[0] + 1) != int(gap_idx.size):
        return None
    if int(gap_idx[0]) == 0 or int(gap_idx[-1]) == int(side.size - 1):
        return None
    return int(gap_idx[0]), int(gap_idx.size)


def _find_structure(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], str, int, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, tuple[int, int, int, int], str, int, int]] = []
    for color in sorted(set(map(int, np.unique(g))) - {BG, GRAY}):
        mask = g == int(color)
        if not np.any(mask):
            continue
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        sides = {
            "up": g[r0, c0 : c1 + 1] == int(color),
            "down": g[r1, c0 : c1 + 1] == int(color),
            "left": g[r0 : r1 + 1, c0] == int(color),
            "right": g[r0 : r1 + 1, c1] == int(color),
        }

        gap_side = None
        for name, arr in sides.items():
            gap = _find_gap(arr)
            if gap is None:
                if not bool(arr.all()):
                    gap_side = None
                    break
                continue
            if gap_side is not None:
                gap_side = None
                break
            gap_side = name
        if gap_side is None:
            continue

        sep_rows = [
            int(r)
            for r in range(r0 + 1, r1)
            if np.all(g[r, c0 + 1 : c1] == int(color))
        ]
        sep_cols = [
            int(c)
            for c in range(c0 + 1, c1)
            if np.all(g[r0 + 1 : r1, c] == int(color))
        ]
        if len(sep_rows) != 1 or len(sep_cols) != 1:
            continue
        sep_r = int(sep_rows[0])
        sep_c = int(sep_cols[0])
        if sep_r - r0 - 1 != r1 - sep_r - 1:
            continue
        if sep_c - c0 - 1 != c1 - sep_c - 1:
            continue
        if sep_r - r0 - 1 <= 0 or sep_c - c0 - 1 <= 0:
            continue
        candidates.append((int(color), (int(r0), int(c0), int(r1), int(c1)), gap_side, sep_r, sep_c))

    assert len(candidates) == 1, "expected exactly one framed 4-quadrant structure"
    return candidates[0]


def _marker_counts(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    counts: dict[str, int] = {}
    for comp in _components(g == GRAY):
        rs = comp[:, 0]
        cs = comp[:, 1]
        key = None
        if int(rs.max()) < r0 and int(cs.max()) < c0:
            key = "tl"
        elif int(rs.max()) < r0 and int(cs.min()) > c1:
            key = "tr"
        elif int(rs.min()) > r1 and int(cs.max()) < c0:
            key = "bl"
        elif int(rs.min()) > r1 and int(cs.min()) > c1:
            key = "br"
        if key is not None:
            counts[key] = int(comp.shape[0])
    assert set(counts) == {"tl", "tr", "bl", "br"}, "expected four outer corner markers"
    return counts


def _quadrant_patches(
    grid: np.ndarray,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    sep_r: int,
    sep_c: int,
) -> dict[str, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    regions = {
        "ul": g[r0 + 1 : sep_r, c0 + 1 : sep_c],
        "ur": g[r0 + 1 : sep_r, sep_c + 1 : c1],
        "ll": g[sep_r + 1 : r1, c0 + 1 : sep_c],
        "lr": g[sep_r + 1 : r1, sep_c + 1 : c1],
    }
    out: dict[str, np.ndarray] = {}
    for name, region in regions.items():
        mask = (region != BG) & (region != int(frame_color))
        assert np.any(mask), f"quadrant {name} must contain a pattern"
        out[name] = _tight_crop(np.where(mask, region, BG), bg=BG)
    return out


def _tile_patch(patch: np.ndarray, direction: str, count: int) -> np.ndarray:
    tiles = [patch.copy() for _ in range(int(count))]
    axis = 0 if direction in {"up", "down"} else 1
    return np.concatenate(tiles, axis=axis)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the direction-coded quadrant pattern and repeat its crop by the count in the matching outer corner marker.

    Input structure:
    - Black background surrounds one single-color rectangular frame with exactly one contiguous gap on one side.
    - One horizontal and one vertical divider of the same color split the frame interior into four equal quadrants.
    - Each quadrant contains one connected monochrome pattern: upper-left = up, upper-right = right, lower-left = left, lower-right = down.
    - Four gray corner markers outside the frame contain 1 to 4 pixels each.

    Rule:
    - Detect which side of the frame has the gap and use that direction to choose the corresponding quadrant pattern.
    - Read the gray marker in the matching directional corner: upper-left for up, upper-right for right, lower-left for left, lower-right for down.
    - Tight-crop the chosen pattern and concatenate it count-many times with no spacing.
    - Use vertical concatenation for up/down and horizontal concatenation for left/right; the background stays black.

    Key insight:
    The gap direction is a double code: it picks one of four interior patterns and also points to the corner marker that supplies the repeat count.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, frame_bbox, direction, sep_r, sep_c = _find_structure(g)
    patches = _quadrant_patches(g, frame_color, frame_bbox, sep_r, sep_c)
    counts = _marker_counts(g, frame_bbox)
    patch = patches[DIR_TO_QUADRANT[direction]]
    count = counts[DIR_TO_CORNER[direction]]
    return _tile_patch(patch, direction, count)


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _sample_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    area_lo = max(3, int(np.ceil(h * w * 0.35)))
    area_hi = max(area_lo, h * w - 1)
    for _ in range(120):
        target = int(rng.integers(area_lo, area_hi + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[start] = True
        cells = [start]
        tries = 0
        while int(mask.sum()) < target and tries < target * 20:
            tries += 1
            rr, cc = cells[int(rng.integers(0, len(cells)))]
            options = [
                (nr, nc)
                for nr, nc in _neighbors4(int(rr), int(cc), h, w)
                if not bool(mask[nr, nc])
            ]
            if not options:
                continue
            nr, nc = _choice(rng, options)
            mask[nr, nc] = True
            cells.append((int(nr), int(nc)))
        if int(mask.sum()) != target:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    color: int,
    bg: int = BG,
) -> None:
    h, w = map(int, patch.shape)
    region = grid[top : top + h, left : left + w]
    region[patch] = int(color)
    region[~patch] = np.where(region[~patch] == int(bg), int(bg), region[~patch])


def _draw_frame_with_gap(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
    direction: str,
    gap_start: int,
    gap_width: int,
) -> None:
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    if direction == "up":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_width] = BG
    elif direction == "down":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_width] = BG
    elif direction == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0] = BG
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_width, c1] = BG


def _marker_patch(rng: np.random.Generator, count: int) -> np.ndarray:
    return _choice(rng, MARKER_LIBRARY[int(count)]).copy()


def _expected_from_parts(
    direction: str,
    patches: dict[str, np.ndarray],
    counts: dict[str, int],
) -> np.ndarray:
    patch = patches[DIR_TO_QUADRANT[direction]]
    count = counts[DIR_TO_CORNER[direction]]
    return _tile_patch(patch, direction, count)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    direction = _choice(rng, ["up", "right", "left", "down"])
    qh = int(rng.integers(*GEN_QUADRANT_RANGE))
    qw = int(rng.integers(*GEN_QUADRANT_RANGE))
    outer_h = 2 * qh + 3
    outer_w = 2 * qw + 3

    pad_top = int(rng.integers(*GEN_PADDING_RANGE))
    pad_bottom = int(rng.integers(*GEN_PADDING_RANGE))
    pad_left = int(rng.integers(*GEN_PADDING_RANGE))
    pad_right = int(rng.integers(*GEN_PADDING_RANGE))
    H = outer_h + pad_top + pad_bottom
    W = outer_w + pad_left + pad_right
    if H > 30 or W > 30:
        return None

    colors = rng.choice(COLOR_POOL, size=5, replace=False)
    frame_color = int(colors[0])
    pattern_colors = [int(c) for c in colors[1:]]

    grid = np.full((H, W), BG, dtype=int)
    r0, c0 = pad_top, pad_left
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1

    if direction in {"up", "down"}:
        edge_len = outer_w
    else:
        edge_len = outer_h
    max_gap = min(5, edge_len - 2)
    if max_gap < 2:
        return None
    gap_width = int(rng.integers(2, max_gap + 1))
    gap_start = int(rng.integers(1, edge_len - gap_width))
    _draw_frame_with_gap(grid, r0, c0, r1, c1, frame_color, direction, gap_start, gap_width)

    sep_r = r0 + qh + 1
    sep_c = c0 + qw + 1
    grid[sep_r, c0 + 1 : c1] = int(frame_color)
    grid[r0 + 1 : r1, sep_c] = int(frame_color)

    quadrant_boxes = {
        "ul": (r0 + 1, c0 + 1, qh, qw),
        "ur": (r0 + 1, sep_c + 1, qh, qw),
        "ll": (sep_r + 1, c0 + 1, qh, qw),
        "lr": (sep_r + 1, sep_c + 1, qh, qw),
    }
    patch_dict: dict[str, np.ndarray] = {}
    seen_shapes: set[tuple[tuple[int, ...], ...]] = set()

    for name, color in zip(["ul", "ur", "ll", "lr"], pattern_colors):
        placed = False
        for _ in range(120):
            ph = int(rng.integers(2, min(GEN_PATTERN_SIDE_MAX, qh) + 1))
            pw = int(rng.integers(2, min(GEN_PATTERN_SIDE_MAX, qw) + 1))
            mask = _sample_connected_mask(rng, ph, pw)
            if mask is None:
                continue
            key = _canonical_mask(mask)
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            patch = np.full((ph, pw), BG, dtype=int)
            patch[mask] = int(color)
            patch_dict[name] = patch
            qr, qc, qh_box, qw_box = quadrant_boxes[name]
            max_dr = qh_box - ph
            max_dc = qw_box - pw
            dr = int(rng.integers(0, max_dr + 1))
            dc = int(rng.integers(0, max_dc + 1))
            region = grid[qr + dr : qr + dr + ph, qc + dc : qc + dc + pw]
            region[mask] = int(color)
            placed = True
            break
        if not placed:
            return None

    marker_counts = {
        "tl": int(rng.integers(1, 5)),
        "tr": int(rng.integers(1, 5)),
        "bl": int(rng.integers(1, 5)),
        "br": int(rng.integers(1, 5)),
    }
    marker_specs = {
        "tl": _marker_patch(rng, marker_counts["tl"]),
        "tr": _marker_patch(rng, marker_counts["tr"]),
        "bl": _marker_patch(rng, marker_counts["bl"]),
        "br": _marker_patch(rng, marker_counts["br"]),
    }
    placements = {}
    for key, patch in marker_specs.items():
        mh, mw = map(int, patch.shape)
        if key == "tl":
            row_lo, row_hi = 0, r0 - mh - 1
            col_lo, col_hi = 0, c0 - mw - 1
        elif key == "tr":
            row_lo, row_hi = 0, r0 - mh - 1
            col_lo, col_hi = c1 + 2, W - mw
        elif key == "bl":
            row_lo, row_hi = r1 + 2, H - mh
            col_lo, col_hi = 0, c0 - mw - 1
        else:
            row_lo, row_hi = r1 + 2, H - mh
            col_lo, col_hi = c1 + 2, W - mw
        if row_lo > row_hi or col_lo > col_hi:
            return None
        rr = int(rng.integers(row_lo, row_hi + 1))
        cc = int(rng.integers(col_lo, col_hi + 1))
        placements[key] = (rr, cc)

    for key, patch in marker_specs.items():
        rr, cc = placements[key]
        mh, mw = map(int, patch.shape)
        region = grid[rr : rr + mh, cc : cc + mw]
        if np.any(region != BG):
            return None
        region[patch] = GRAY

    expected = _expected_from_parts(direction, patch_dict, marker_counts)
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
