import random
from collections import deque

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
BRANCHES = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_separator_columns(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        col = grid[:, c]
        if bool(np.all(col == col[0])) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))

    matches: list[tuple[int, int, int]] = []
    for i, (c0, color0) in enumerate(candidates):
        for c1, color1 in candidates[i + 1 :]:
            if color0 != color1:
                continue
            panel_w = c0
            if panel_w <= 0:
                continue
            if c1 == 2 * panel_w + 1 and w == 3 * panel_w + 2:
                matches.append((c0, c1, color0))
    assert len(matches) == 1, "expected exactly two equal-spacing vertical separator bars"
    return matches[0]


def _paint_sprite(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = sprite != bg
    coords = np.argwhere(mask)
    canvas[top + coords[:, 0], left + coords[:, 1]] = sprite[mask]


def _extract_panel_sprite(panel: np.ndarray, bg: int) -> dict:
    colors = [int(x) for x in np.unique(panel) if int(x) != bg]
    assert len(colors) == 1, "each panel must contain exactly one monochrome sprite color"
    color = colors[0]
    comps = _connected_components(panel == color)
    assert len(comps) == 1, "each panel must contain exactly one 4-connected sprite"
    coords = comps[0]
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return {
        "color": color,
        "anchor": (r0, c0),
        "sprite": sprite,
        "bbox": (r0, c0, r1, c1),
    }


def _reflect_once(anchor: int, max_anchor: int) -> tuple[int, bool]:
    if max_anchor <= 0:
        return 0, False
    if 0 <= anchor <= max_anchor:
        return anchor, False
    if anchor < 0:
        reflected = -anchor
    else:
        reflected = 2 * max_anchor - anchor
    assert 0 <= reflected <= max_anchor, "generator only permits at most one bounce per axis"
    return reflected, True


def _advance_sprite(
    sprite: np.ndarray,
    anchor: tuple[int, int],
    delta: tuple[int, int],
    panel_shape: tuple[int, int],
    bg: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[bool, bool], np.ndarray]:
    sh, sw = sprite.shape
    max_r = panel_shape[0] - sh
    max_c = panel_shape[1] - sw
    nr, flip_v = _reflect_once(anchor[0] + delta[0], max_r)
    nc, flip_h = _reflect_once(anchor[1] + delta[1], max_c)
    moved = sprite.copy()
    if flip_v:
        moved = np.flipud(moved)
    if flip_h:
        moved = np.fliplr(moved)
    out = np.full(panel_shape, bg, dtype=int)
    _paint_sprite(out, moved, nr, nc, bg)
    return out, (nr, nc), (flip_v, flip_h), moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the three equal panels as consecutive animation frames and draw the fourth frame.

    Input structure:
    - The grid is split by two solid vertical separator bars into three equal-width panels.
    - Every panel contains the same monochrome 4-connected sprite on the same background, differing only by panel-relative position.
    - Across the three given frames the sprite moves by a constant per-frame displacement and does not hit a panel boundary.

    Rule:
    - Read sprite positions in panel-local coordinates and infer the constant displacement vector from frame 1->2 and 2->3.
    - Start from the third frame and move the sprite by that same vector onto a single-panel canvas.
    - If this fourth step would push the sprite bbox past a wall, reflect the overflow back into the panel and mirror the sprite along every bounced axis.

    Key insight:
    - The three panels are time frames rather than layers, so the answer is an unseen extrapolated frame whose orientation may change only at the bounce.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    sep0, sep1, _sep_color = _find_separator_columns(g, bg)
    panel_w = sep0
    panels = [
        g[:, :panel_w],
        g[:, sep0 + 1 : sep1],
        g[:, sep1 + 1 :],
    ]
    assert all(panel.shape == panels[0].shape for panel in panels), "panels must have equal size"

    frames = [_extract_panel_sprite(panel, bg) for panel in panels]
    sprites = [frame["sprite"] for frame in frames]
    assert all(np.array_equal(sprites[0], spr) for spr in sprites[1:]), "sprite shape must match across frames"
    colors = {frame["color"] for frame in frames}
    assert len(colors) == 1, "sprite color must remain constant across frames"

    anchors = [frame["anchor"] for frame in frames]
    d0 = (anchors[1][0] - anchors[0][0], anchors[1][1] - anchors[0][1])
    d1 = (anchors[2][0] - anchors[1][0], anchors[2][1] - anchors[1][1])
    assert d0 == d1 and d0 != (0, 0), "frames must encode a non-zero constant displacement"

    out, _anchor4, _flips, _sprite4 = _advance_sprite(
        sprites[0], anchors[2], d0, panels[0].shape, bg
    )
    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    min_area = max(3, (h * w) // 2)
    max_area = h * w - 1
    for _ in range(200):
        target = random.randint(min_area, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while len(cells) < target:
            if not frontier:
                break
            options = list(frontier)
            if random.random() < 0.55:
                options.sort(
                    key=lambda rc: sum(
                        ((rc[0] + dr, rc[1] + dc) in cells)
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    ),
                    reverse=True,
                )
                cell = random.choice(options[: max(1, min(3, len(options)))])
            else:
                cell = random.choice(options)
            frontier.remove(cell)
            cells.add(cell)
            add_frontier(*cell)

        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if mask.all():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask
    raise RuntimeError("failed to build a sufficiently asymmetric sprite")


def _axis_options(max_anchor: int, bounce: bool) -> list[tuple[int, int]]:
    opts: list[tuple[int, int]] = []
    speed_limit = min(3, max_anchor)
    if bounce:
        for d in range(-speed_limit, speed_limit + 1):
            if d == 0 or 2 * abs(d) > max_anchor:
                continue
            if d > 0:
                lo = max(0, max_anchor - 3 * d + 1)
                hi = max_anchor - 2 * d
            else:
                lo = -2 * d
                hi = -3 * d - 1
            lo = max(lo, 0)
            hi = min(hi, max_anchor)
            if lo > hi:
                continue
            for start in range(lo, hi + 1):
                opts.append((start, d))
    else:
        for d in range(-speed_limit, speed_limit + 1):
            if d > 0:
                lo = 0
                hi = max_anchor - 3 * d
            elif d < 0:
                lo = -3 * d
                hi = max_anchor
            else:
                lo = 0
                hi = max_anchor
            if lo > hi:
                continue
            for start in range(lo, hi + 1):
                opts.append((start, d))
    return opts


def _choose_axis_motion(max_anchor: int, bounce: bool, prefer_nonzero: bool) -> tuple[int, int]:
    options = _axis_options(max_anchor, bounce)
    assert options, "no valid axis motions for requested branch"
    if prefer_nonzero:
        nonzero = [opt for opt in options if opt[1] != 0]
        if nonzero:
            options = nonzero
    return random.choice(options)


def _build_case(
    panel_h: int,
    panel_w: int,
    bg: int,
    sep: int,
    color: int,
    sprite: np.ndarray,
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((panel_h, 3 * panel_w + 2), bg, dtype=int)
    grid[:, panel_w] = sep
    grid[:, 2 * panel_w + 1] = sep
    starts = (0, panel_w + 1, 2 * panel_w + 2)
    for i, (r, c) in enumerate(anchors):
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        _paint_sprite(panel, sprite, r, c, bg)
        start = starts[i]
        grid[:, start : start + panel_w] = panel
    return grid


def _validate_generated_case(
    grid: np.ndarray,
    sprite: np.ndarray,
    anchors: list[tuple[int, int]],
    delta: tuple[int, int],
    branch: tuple[bool, bool],
    bg: int,
) -> bool:
    panel_h = grid.shape[0]
    panel_w = (grid.shape[1] - 2) // 3
    expected, anchor4, flips, sprite4 = _advance_sprite(sprite, anchors[2], delta, (panel_h, panel_w), bg)
    if flips != branch:
        return False
    panels = [
        grid[:, :panel_w],
        grid[:, panel_w + 1 : 2 * panel_w + 1],
        grid[:, 2 * panel_w + 2 :],
    ]
    if any(np.array_equal(expected, panel) for panel in panels):
        return False
    parsed = [_extract_panel_sprite(panel, bg) for panel in panels]
    parsed_anchors = [item["anchor"] for item in parsed]
    if parsed_anchors != anchors:
        return False
    if not all(np.array_equal(item["sprite"], sprite) for item in parsed):
        return False
    if main(grid).shape != expected.shape or not np.array_equal(main(grid), expected):
        return False
    if anchor4 == anchors[2] or np.array_equal(sprite4, sprite) != (branch == (False, False)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(6, 10)
    panel_w = random.randint(7, 9)
    sprite_h = random.randint(2, min(4, panel_h - 3))
    sprite_w = random.randint(2, min(4, panel_w - 3))
    max_r = panel_h - sprite_h
    max_c = panel_w - sprite_w
    if max_r < 3 or max_c < 3:
        return None

    branch = random.choice(BRANCHES)
    r0, dr = _choose_axis_motion(max_r, branch[0], prefer_nonzero=branch[0])
    c0, dc = _choose_axis_motion(max_c, branch[1], prefer_nonzero=branch[1])
    if dr == 0 and dc == 0:
        nonzero_rows = [opt for opt in _axis_options(max_r, False) if opt[1] != 0]
        nonzero_cols = [opt for opt in _axis_options(max_c, False) if opt[1] != 0]
        if random.random() < 0.5 and nonzero_rows:
            r0, dr = random.choice(nonzero_rows)
        elif nonzero_cols:
            c0, dc = random.choice(nonzero_cols)
        else:
            return None

    mask = _random_connected_mask(sprite_h, sprite_w)
    bg, sep, color = random.sample(range(10), 3)
    sprite = np.full((sprite_h, sprite_w), bg, dtype=int)
    sprite[mask] = color
    anchors = [(r0 + i * dr, c0 + i * dc) for i in range(3)]
    grid = _build_case(panel_h, panel_w, bg, sep, color, sprite, anchors)
    if _validate_generated_case(grid, sprite, anchors, (dr, dc), branch, bg):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid three-panel extrapolation input")
