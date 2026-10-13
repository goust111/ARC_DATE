from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 300


Frame = tuple[int, tuple[int, int, int, int]]


def _outer_ring_is_bg(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int) -> bool:
    h, w = g.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)

    ring = np.zeros(g.shape, dtype=bool)
    ring[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    ring[r0 : r1 + 1, c0 : c1 + 1] = False
    return bool(np.all(g[ring] == bg))


def _has_solid_perimeter(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> bool:
    return bool(
        np.all(g[r0, c0 : c1 + 1] == color)
        and np.all(g[r1, c0 : c1 + 1] == color)
        and np.all(g[r0 : r1 + 1, c0] == color)
        and np.all(g[r0 : r1 + 1, c1] == color)
    )


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1 and outer != inner


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _interior_shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 - 1, c1 - c0 - 1


def _find_frames(g: np.ndarray, bg: int) -> list[Frame]:
    h, w = g.shape
    candidates: list[Frame] = []

    for r0 in range(h - 3):
        for c0 in range(w - 3):
            color = int(g[r0, c0])
            if color == bg:
                continue
            for r1 in range(r0 + 3, h):
                for c1 in range(c0 + 3, w):
                    if int(g[r0, c1]) != color or int(g[r1, c0]) != color:
                        continue
                    if int(g[r1, c1]) != color:
                        continue
                    if not _has_solid_perimeter(g, r0, c0, r1, c1, color):
                        continue
                    if not _outer_ring_is_bg(g, r0, c0, r1, c1, bg):
                        continue
                    candidates.append((color, (r0, c0, r1, c1)))

    top_level: list[Frame] = []
    for color, box in candidates:
        if any(_contains(other_box, box) for _other_color, other_box in candidates):
            continue
        top_level.append((color, box))
    return top_level


def _indicator_color(g: np.ndarray, bg: int) -> int:
    h, w = g.shape
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    hits = [int(g[r, c]) for r, c in corners if int(g[r, c]) != bg]
    assert len(hits) == 1, "expected exactly one colored corner indicator"
    return hits[0]


def _select_frame_pair(frames: list[Frame], indicator: int) -> tuple[Frame, Frame]:
    options: list[tuple[int, Frame, Frame]] = []
    for a, b in combinations(frames, 2):
        acol, abox = a
        bcol, bbox = b
        if acol == bcol:
            continue
        if indicator not in (acol, bcol):
            continue
        if _interior_shape(abox) != _interior_shape(bbox):
            continue
        if _overlap(abox, bbox):
            continue
        area = (abox[2] - abox[0] + 1) * (abox[3] - abox[1] + 1)
        options.append((area, a, b))

    assert options, "expected two same-size rectangular frames linked by the indicator"
    _area, first, second = max(options, key=lambda item: item[0])
    return first, second


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine a template frame's colors with a matching data frame's binary mask.

    Input structure:
    - Background is black.
    - Two non-overlapping, same-size rectangular regions are enclosed by one-pixel borders of different colors.
    - One region is the template: its interior is fully non-black and may contain many colors.
    - The other region is the data mask: its interior contains black plus one foreground color equal to its border color.
    - Exactly one corner cell outside the frames is colored like the template border.

    Rule:
    - Use the corner indicator color to decide which frame is the template.
    - Use the other frame as the data mask.
    - Return an array with the shared interior size: template colors are copied only where the data mask has foreground; all other cells are black.

    Key insight:
    The colored corner pixel assigns the two framed patterns different roles, and the data pattern is only a spatial mask over the template colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG

    indicator = _indicator_color(g, bg)
    frames = _find_frames(g, bg)
    first, second = _select_frame_pair(frames, indicator)

    template_frame = first if first[0] == indicator else second
    data_frame = second if template_frame is first else first

    template_color, (tr0, tc0, tr1, tc1) = template_frame
    data_color, (dr0, dc0, dr1, dc1) = data_frame
    assert template_color == indicator and data_color != indicator

    template = g[tr0 + 1 : tr1, tc0 + 1 : tc1]
    data = g[dr0 + 1 : dr1, dc0 + 1 : dc1]
    assert template.shape == data.shape
    assert np.all(template != bg), "template interior must be fully colored"

    data_non_bg = set(map(int, np.unique(data[data != bg])))
    assert data_non_bg == {data_color}, "data interior foreground must match its border color"

    out = np.full(template.shape, bg, dtype=int)
    mask = data == data_color
    out[mask] = template[mask]
    return out


# --------------- generator ---------------


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> None:
    H, W = occupied.shape
    occupied[
        max(0, r0 - padding) : min(H, r0 + h + padding),
        max(0, c0 - padding) : min(W, c0 + w + padding),
    ] = True


def _free_positions(occupied: np.ndarray, h: int, w: int, padding: int) -> list[tuple[int, int]]:
    H, W = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - padding)
            c0 = max(0, c - padding)
            r1 = min(H, r + h + padding)
            c1 = min(W, c + w + padding)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _make_template(rng: random.Random, h: int, w: int, colors: list[int]) -> np.ndarray:
    mode = rng.choice(("rows", "cols", "checker", "blocks", "diagonal", "mixed"))
    arr = np.zeros((h, w), dtype=int)

    if mode == "rows":
        offset = rng.randrange(len(colors))
        for r in range(h):
            arr[r, :] = colors[(r + offset) % len(colors)]
    elif mode == "cols":
        offset = rng.randrange(len(colors))
        for c in range(w):
            arr[:, c] = colors[(c + offset) % len(colors)]
    elif mode == "checker":
        offset = rng.randrange(len(colors))
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[(r + c + offset) % len(colors)]
    elif mode == "blocks":
        bh = rng.choice((1, 2, 3))
        bw = rng.choice((1, 2, 3))
        offset = rng.randrange(len(colors))
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[(r // bh + c // bw + offset) % len(colors)]
    elif mode == "diagonal":
        offset = rng.randrange(len(colors))
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[(r - c + offset) % len(colors)]
    else:
        for r in range(h):
            for c in range(w):
                arr[r, c] = rng.choice(colors)

    if len(set(map(int, np.unique(arr)))) < 2:
        arr[0, 0] = colors[0]
        arr[-1, -1] = colors[1]
    return arr


def _grow_connected_mask(rng: random.Random, h: int, w: int, target: int) -> np.ndarray:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_data_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mode = rng.choice(("blob", "scattered", "bars", "checker", "ring"))
    mask = np.zeros((h, w), dtype=bool)
    total = h * w

    if mode == "blob":
        target = rng.randint(max(2, total // 5), max(2, (total * 3) // 5))
        mask = _grow_connected_mask(rng, h, w, target)
    elif mode == "scattered":
        target = rng.randint(max(2, total // 5), max(2, total // 2))
        cells = rng.sample([(r, c) for r in range(h) for c in range(w)], target)
        for r, c in cells:
            mask[r, c] = True
    elif mode == "bars":
        if rng.random() < 0.5:
            rows = rng.sample(range(h), rng.randint(1, max(1, h // 2)))
            for r in rows:
                c0 = rng.randrange(w)
                c1 = rng.randrange(c0, w)
                mask[r, c0 : c1 + 1] = True
        else:
            cols = rng.sample(range(w), rng.randint(1, max(1, w // 2)))
            for c in cols:
                r0 = rng.randrange(h)
                r1 = rng.randrange(r0, h)
                mask[r0 : r1 + 1, c] = True
    elif mode == "checker":
        phase = rng.randrange(2)
        for r in range(h):
            for c in range(w):
                if (r + c + phase) % 2 == 0 and rng.random() < 0.85:
                    mask[r, c] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        if h > 4 and w > 4 and rng.random() < 0.5:
            rr = rng.randrange(2, h - 2)
            mask[rr, 1:-1] = True
        if h > 4 and w > 4 and rng.random() < 0.5:
            cc = rng.randrange(2, w - 2)
            mask[1:-1, cc] = True

    if not mask.any():
        mask[rng.randrange(h), rng.randrange(w)] = True
    if mask.all():
        mask[rng.randrange(h), rng.randrange(w)] = False
    if int(mask.sum()) == 1:
        r, c = np.argwhere(mask)[0]
        choices = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w:
                choices.append((nr, nc))
        if choices:
            nr, nc = rng.choice(choices)
            mask[nr, nc] = True
    return mask


def _draw_framed_region(
    grid: np.ndarray,
    r0: int,
    c0: int,
    border_color: int,
    interior: np.ndarray,
) -> None:
    ih, iw = interior.shape
    fh, fw = ih + 2, iw + 2
    grid[r0, c0 : c0 + fw] = border_color
    grid[r0 + fh - 1, c0 : c0 + fw] = border_color
    grid[r0 : r0 + fh, c0] = border_color
    grid[r0 : r0 + fh, c0 + fw - 1] = border_color
    grid[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = interior


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    inner_h = rng.randint(4, 8)
    inner_w = rng.randint(4, 8)
    frame_h, frame_w = inner_h + 2, inner_w + 2

    if rng.random() < 0.5:
        min_h, min_w = frame_h + 5, 2 * frame_w + 5
    else:
        min_h, min_w = 2 * frame_h + 5, frame_w + 5
    if min_h > 30 or min_w > 30:
        return None
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)

    border_colors = rng.sample(COLORS, 2)
    template_border = border_colors[0]
    data_border = border_colors[1]
    template_pool = [c for c in COLORS if c not in border_colors]
    rng.shuffle(template_pool)
    template_colors = template_pool[: rng.randint(2, min(5, len(template_pool)))]

    template = _make_template(rng, inner_h, inner_w, template_colors)
    mask = _make_data_mask(rng, inner_h, inner_w)
    data = np.full((inner_h, inner_w), BG, dtype=int)
    data[mask] = data_border

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corners = ((0, 0), (0, grid_w - 1), (grid_h - 1, 0), (grid_h - 1, grid_w - 1))
    ir, ic = rng.choice(corners)
    grid[ir, ic] = template_border
    _reserve(occupied, ir, ic, 1, 1, padding=2)

    regions = [
        ("template", template_border, template),
        ("data", data_border, data),
    ]
    rng.shuffle(regions)

    expected = np.full((inner_h, inner_w), BG, dtype=int)
    expected[mask] = template[mask]

    for _role, border_color, interior in regions:
        positions = _free_positions(occupied, frame_h, frame_w, padding=1)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _draw_framed_region(grid, r0, c0, border_color, interior)
        _reserve(occupied, r0, c0, frame_h, frame_w, padding=1)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
