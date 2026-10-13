"""ARC task: stamp a template sprite into grey-separated marker windows."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
_RETRIES = 400


def _intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    """Return half-open non-separator intervals along one grid axis."""
    result: list[tuple[int, int]] = []
    start = 0
    for index in separators:
        if start < index:
            result.append((start, index))
        start = index + 1
    if start < length:
        result.append((start, length))
    return result


def _windows(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int]:
    """Infer the equal window rectangles and their structurally grey separator."""
    h, w = grid.shape
    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == GREY))]
    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == GREY))]
    assert sep_rows and sep_cols, "expected grey full-row and full-column separators"

    row_parts = _intervals(h, sep_rows)
    col_parts = _intervals(w, sep_cols)
    assert len(row_parts) >= 2 and len(col_parts) >= 2
    shapes = {(r1 - r0, c1 - c0) for r0, r1 in row_parts for c0, c1 in col_parts}
    assert len(shapes) == 1, "all windows must have the same size"
    windows = [(r0, r1, c0, c1) for r0, r1 in row_parts for c0, c1 in col_parts]

    cells = np.concatenate([grid[r0:r1, c0:c1].ravel() for r0, r1, c0, c1 in windows])
    bg = int(Counter(map(int, cells)).most_common(1)[0][0])
    assert bg != GREY
    return windows, bg


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the template sprite into every marker window using its same-colored internal anchor.

    Input structure:
    - Grey full rows and columns divide the grid into equal windows on a shared background.
    - Exactly one window contains a multicolored sprite; every other nonempty window
      contains one marker pixel whose color occurs once or twice in that sprite.

    Rule:
    - Preserve the template window and grey separators.  In each marker window,
      translate the sprite without rotation so a sprite cell of the marker color lands
      on the marker.  If that color occurs twice, use the sole translation that keeps
      the complete sprite bounding box inside that window.

    Key insight:
    - Marker color selects a coordinate inside the sprite, not the sprite's corner;
      window containment uniquely resolves a repeated-color anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    windows, bg = _windows(g)

    non_bg: list[np.ndarray] = []
    for r0, r1, c0, c1 in windows:
        non_bg.append(np.argwhere(g[r0:r1, c0:c1] != bg))
    template_ids = [i for i, coords in enumerate(non_bg) if len(coords) > 1]
    assert len(template_ids) == 1, "expected exactly one sprite window"
    template_id = template_ids[0]

    tr0, tr1, tc0, tc1 = windows[template_id]
    template_window = g[tr0:tr1, tc0:tc1]
    coords = non_bg[template_id]
    sr0, sc0 = coords.min(axis=0)
    sr1, sc1 = coords.max(axis=0)
    sprite = template_window[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    sh, sw = sprite.shape

    sprite_colors = set(map(int, np.unique(sprite))) - {bg}
    assert len(sprite_colors) >= 2
    for color in sprite_colors:
        assert 1 <= int(np.count_nonzero(sprite == color)) <= 2

    out = g.copy()
    for i, (r0, r1, c0, c1) in enumerate(windows):
        if i == template_id or len(non_bg[i]) == 0:
            continue
        assert len(non_bg[i]) == 1, "non-template windows must be empty or singleton markers"
        mr, mc = map(int, non_bg[i][0])
        marker_color = int(g[r0 + mr, c0 + mc])
        anchors = np.argwhere(sprite == marker_color)
        assert 1 <= len(anchors) <= 2

        valid: list[tuple[int, int]] = []
        wh, ww = r1 - r0, c1 - c0
        for ar, ac in anchors:
            top, left = mr - int(ar), mc - int(ac)
            if 0 <= top <= wh - sh and 0 <= left <= ww - sw:
                valid.append((top, left))
        assert len(valid) == 1, "marker anchor must have one containing placement"

        top, left = valid[0]
        target = out[r0:r1, c0:c1]
        target[:, :] = bg
        mask = sprite != bg
        view = target[top : top + sh, left : left + sw]
        view[mask] = sprite[mask]
    return out


def _connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Make a sparse connected mask with a genuine two-dimensional bounding box."""
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    target = int(rng.integers(h + w - 1, min(h * w, h + w + 3) + 1))
    while int(mask.sum()) < target:
        candidates = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if ((r > 0 and mask[r - 1, c]) or (r + 1 < h and mask[r + 1, c])
                        or (c > 0 and mask[r, c - 1]) or (c + 1 < w and mask[r, c + 1])):
                    candidates.append((r, c))
        mask[random.choice(candidates)] = True
    return mask


def _make_sprite(rng: np.random.Generator, bg: int, slack_r: int, slack_c: int):
    h = int(rng.integers(4, 6))
    w = int(rng.integers(4, 6))
    mask = _connected_mask(rng, h, w)
    # Random orientation changes which global direction exhibits the hard boundary case.
    variant = int(rng.integers(8))
    if variant >= 4:
        mask = np.fliplr(mask)
        variant -= 4
    mask = np.rot90(mask, variant)
    h, w = mask.shape
    points = [tuple(map(int, p)) for p in np.argwhere(mask)]

    # A farthest pair is farther apart than the window slack in at least one axis.
    pairs = [(a, b) for ix, a in enumerate(points) for b in points[ix + 1:]
             if abs(a[0] - b[0]) > slack_r or abs(a[1] - b[1]) > slack_c]
    assert pairs
    dup = max(pairs, key=lambda ab: abs(ab[0][0] - ab[1][0]) + abs(ab[0][1] - ab[1][1]))

    # A nearby pair has overlapping feasible marker regions and demonstrates color offsets.
    nearby = [(a, b) for ix, a in enumerate(points) for b in points[ix + 1:]
              if a not in dup and b not in dup
              and abs(a[0] - b[0]) <= slack_r and abs(a[1] - b[1]) <= slack_c]
    assert nearby
    unique = random.choice(nearby)

    palette = [c for c in range(10) if c not in (bg, GREY)]
    rng.shuffle(palette)
    needed = 3 + (len(points) - 4 + 1) // 2
    assert needed <= len(palette)
    colors = palette[:needed]
    sprite = np.full((h, w), bg, dtype=int)
    sprite[dup[0]] = sprite[dup[1]] = colors[0]
    sprite[unique[0]] = colors[1]
    sprite[unique[1]] = colors[2]
    rest = [p for p in points if p not in dup and p not in unique]
    rng.shuffle(rest)
    for i, p in enumerate(rest):
        sprite[p] = colors[3 + i // 2]
    return sprite, dup, unique


def _feasible_markers(anchor: tuple[int, int], wh: int, ww: int, sh: int, sw: int):
    ar, ac = anchor
    return {(r, c) for r in range(ar, wh - sh + ar + 1)
            for c in range(ac, ww - sw + ac + 1)}


def _stamp(window: np.ndarray, sprite: np.ndarray, bg: int,
           marker: tuple[int, int], anchor: tuple[int, int]) -> None:
    top, left = marker[0] - anchor[0], marker[1] - anchor[1]
    mask = sprite != bg
    view = window[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    view[mask] = sprite[mask]


def _attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != GREY])
    slack_r, slack_c = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    try:
        sprite, dup, unique = _make_sprite(rng, bg, slack_r, slack_c)
    except AssertionError:
        return None
    sh, sw = sprite.shape
    wh, ww = sh + slack_r, sw + slack_c
    rows, cols = random.choice([(2, 3), (3, 2)])
    H, W = rows * wh + rows - 1, cols * ww + cols - 1
    if max(H, W) > 30:
        return None

    grid = np.full((H, W), bg, dtype=int)
    for r in range(wh, H, wh + 1):
        grid[r, :] = GREY
    for c in range(ww, W, ww + 1):
        grid[:, c] = GREY
    origins = [(rr * (wh + 1), cc * (ww + 1)) for rr in range(rows) for cc in range(cols)]
    rng.shuffle(origins)
    template_origin = origins.pop()
    tr = int(rng.integers(0, slack_r + 1))
    tc = int(rng.integers(0, slack_c + 1))
    tw = grid[template_origin[0] : template_origin[0] + wh,
              template_origin[1] : template_origin[1] + ww]
    _stamp(tw, sprite, bg, (tr + dup[0][0], tc + dup[0][1]), dup[0])

    dcolor = int(sprite[dup[0]])
    ucolors = [int(sprite[p]) for p in unique]
    marker_specs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    # Placing a marker exactly at either far anchor makes the other anchor invalid.
    marker_specs.extend([(dcolor, dup[0], dup[0]), (dcolor, dup[1], dup[1])])
    common = list(_feasible_markers(unique[0], wh, ww, sh, sw)
                  & _feasible_markers(unique[1], wh, ww, sh, sw))
    if not common:
        return None
    common_marker = random.choice(common)
    marker_specs.extend([(ucolors[0], common_marker, unique[0]),
                         (ucolors[1], common_marker, unique[1])])

    # Sometimes use the sixth window too, varying object count and anchor placement.
    if origins and bool(rng.integers(2)):
        other_points = [tuple(map(int, p)) for p in np.argwhere(sprite != bg)
                        if int(sprite[tuple(p)]) not in (dcolor, *ucolors)]
        if other_points:
            anchor = random.choice(other_points)
            marker = random.choice(list(_feasible_markers(anchor, wh, ww, sh, sw)))
            marker_specs.append((int(sprite[anchor]), marker, anchor))
    if len(origins) < len(marker_specs):
        return None

    expected = grid.copy()
    used_origins = []
    for (color, marker, anchor), (r0, c0) in zip(marker_specs, origins):
        grid[r0 + marker[0], c0 + marker[1]] = color
        target = expected[r0 : r0 + wh, c0 : c0 + ww]
        _stamp(target, sprite, bg, marker, anchor)
        used_origins.append((r0, c0, color, marker, anchor))

    # Independent semantic checks, including the two deliberately generated counterexamples.
    anchors_by_color = {c: [tuple(map(int, p)) for p in np.argwhere(sprite == c)]
                        for c in set(map(int, np.unique(sprite))) - {bg}}
    dup_choices = []
    for _r0, _c0, color, marker, intended in used_origins:
        valid = [a for a in anchors_by_color[color]
                 if marker in _feasible_markers(a, wh, ww, sh, sw)]
        if valid != [intended]:
            return None
        if color == dcolor:
            dup_choices.append(intended)
    if set(dup_choices) != set(dup):
        return None
    if common_marker[0] - unique[0][0] == common_marker[0] - unique[1][0] and \
            common_marker[1] - unique[0][1] == common_marker[1] - unique[1][1]:
        return None
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, uniquely solvable window-and-anchor stamping instance."""
    rng = np.random.default_rng()
    for _ in range(_RETRIES):
        result = _attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid anchor-stamping instance")
