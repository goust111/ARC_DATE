from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _separator_col(g: np.ndarray) -> int:
    h, w = g.shape
    candidates: list[int] = []
    for c in range(1, w - 1):
        color = int(g[0, c])
        if color == BG:
            continue
        if not bool(np.all(g[:, c] == color)):
            continue
        if not np.any(g[:, :c] != BG) or not np.any(g[:, c + 1 :] != BG):
            continue
        candidates.append(c)
    assert candidates, "expected one full-height separator column"
    grey = [c for c in candidates if int(g[0, c]) == SEP]
    return int((grey or candidates)[0])


def _components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color == BG or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in NEIGHBORS8:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    return comps


def _exterior_without_wall(shape: tuple[int, int], wall_coords: np.ndarray) -> np.ndarray:
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    wall[wall_coords[:, 0], wall_coords[:, 1]] = True
    exterior = np.zeros(shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                q.append((rr, cc))
    return exterior


def _contains(shape: tuple[int, int], outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    if not (or0 <= ir0 <= ir1 <= or1 and oc0 <= ic0 <= ic1 <= oc1):
        return False
    exterior = _exterior_without_wall(shape, outer["coords"])
    coords = inner["coords"]
    return not bool(np.any(exterior[coords[:, 0], coords[:, 1]]))


def _color_rules(left: np.ndarray) -> dict[int, int]:
    comps = _components(left)
    assert len(comps) >= 2, "expected nested ring components"

    containers_by_inner: list[list[int]] = [[] for _ in comps]
    for oi, outer in enumerate(comps):
        for ii, inner in enumerate(comps):
            if oi != ii and _contains(left.shape, outer, inner):
                containers_by_inner[ii].append(oi)

    rules: dict[int, int] = {}
    for inner_idx, containers in enumerate(containers_by_inner):
        if not containers:
            continue
        outer_idx = min(containers, key=lambda idx: comps[idx]["bbox_area"])
        outer_color = int(comps[outer_idx]["color"])
        inner_color = int(comps[inner_idx]["color"])
        assert outer_color != inner_color
        assert outer_color not in rules or rules[outer_color] == inner_color
        rules[outer_color] = inner_color

    assert rules, "expected at least one outer-to-inner color rule"
    return rules


def _crop_foreground(g: np.ndarray) -> np.ndarray:
    coords = np.argwhere(g != BG)
    assert coords.size, "target region must contain a foreground pattern"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use nested marker rings as a color dictionary and recolor the separate target pattern.

    Input structure:
    - Black background with a full-height grey column separating a left reference
      area from a right target area.
    - The left area contains pairs of closed one-pixel rings; each pair has one
      larger outer ring and one smaller ring completely enclosed by it.
    - The right area contains an independent multi-color foreground pattern using
      some of the colors that appear in the left reference area.

    Rule:
    - In the left area, infer one replacement rule from every nested pair:
      outer ring color maps to the color of the ring inside it.
    - Crop away the left reference area and grey separator.
    - In the right target area, replace every pixel whose color is an outer-ring
      color by the corresponding inner-ring color; colors that only appear as
      inner-ring colors remain unchanged.

    Key insight:
    The nested rings are a spatial lookup table, so containment gives the
    replacement direction before the unrelated right-side pattern is recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_c = _separator_col(g)
    left = g[:, :sep_c]
    target = g[:, sep_c + 1 :].copy()
    rules = _color_rules(left)

    for outer_color, inner_color in rules.items():
        target[target == outer_color] = inner_color

    return _crop_foreground(target)


def _draw_rect_ring(g: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    g[r0, c0 : c0 + w] = color
    g[r0 + h - 1, c0 : c0 + w] = color
    g[r0 : r0 + h, c0] = color
    g[r0 : r0 + h, c0 + w - 1] = color


def _draw_diamond_ring(g: np.ndarray, cr: int, cc: int, radius: int, color: int) -> None:
    for dr in range(-radius, radius + 1):
        rem = radius - abs(dr)
        g[cr + dr, cc - rem] = color
        g[cr + dr, cc + rem] = color


def _make_pair_patch(outer_color: int, inner_color: int) -> np.ndarray:
    outer_kind = random.choice(("rect", "diamond"))
    if outer_kind == "rect":
        h = random.randint(7, 11)
        w = random.randint(7, 11)
        patch = np.full((h, w), BG, dtype=int)
        _draw_rect_ring(patch, 0, 0, h, w, outer_color)

        inner_options = ["rect"]
        if min(h, w) >= 7:
            inner_options.append("diamond")
        if random.choice(inner_options) == "rect":
            ih = random.randint(3, h - 4)
            iw = random.randint(3, w - 4)
            ir = random.randint(2, h - ih - 2)
            ic = random.randint(2, w - iw - 2)
            _draw_rect_ring(patch, ir, ic, ih, iw, inner_color)
        else:
            max_radius = (min(h, w) - 5) // 2
            radius = random.randint(1, max_radius)
            cr = random.randint(2 + radius, h - 3 - radius)
            cc = random.randint(2 + radius, w - 3 - radius)
            _draw_diamond_ring(patch, cr, cc, radius, inner_color)
        return patch

    radius = random.randint(4, 5)
    size = radius * 2 + 1
    patch = np.full((size, size), BG, dtype=int)
    _draw_diamond_ring(patch, radius, radius, radius, outer_color)

    if random.random() < 0.5:
        inner_radius = random.randint(1, radius - 2)
        _draw_diamond_ring(patch, radius, radius, inner_radius, inner_color)
    else:
        choices = [(a, b) for a in range(1, radius - 1) for b in range(1, radius - 1) if a + b <= radius - 2]
        a, b = random.choice(choices)
        _draw_rect_ring(patch, radius - a, radius - b, 2 * a + 1, 2 * b + 1, inner_color)
    return patch


def _place_patches(left: np.ndarray, patches: list[np.ndarray]) -> bool:
    occupied = np.zeros(left.shape, dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, left.shape[0] - ph):
            for c in range(1, left.shape[1] - pw):
                if not np.any(occupied[r - 1 : r + ph + 1, c - 1 : c + pw + 1]):
                    candidates.append((r, c))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        mask = patch != BG
        left[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
    return True


def _connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _target_patch(colors: list[int], max_h: int, max_w: int) -> np.ndarray:
    h = random.randint(4, max_h)
    w = random.randint(4, max_w)
    min_area = max(len(colors), 5)
    max_area = max(min_area, min(h * w, int(h * w * 0.65)))
    area = random.randint(min_area, max_area)
    mask = _connected_mask(h, w, area)
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    random.shuffle(cells)

    patch = np.full((h, w), BG, dtype=int)
    for (r, c), color in zip(cells, colors, strict=False):
        patch[r, c] = color
    for r, c in cells[len(colors) :]:
        patch[r, c] = random.choice(colors)
    return _crop_foreground(patch)


def _make_target(
    height: int,
    width: int,
    pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inner_only_pair = random.randrange(len(pairs))
    outer_pair_choices = [idx for idx in range(len(pairs)) if idx != inner_only_pair]
    outer_pair = random.choice(outer_pair_choices)

    colors = [pairs[inner_only_pair][1], pairs[outer_pair][0]]
    extras = [color for pair in pairs for color in pair if color not in colors]
    random.shuffle(extras)
    colors.extend(extras[: random.randint(0, min(4, len(extras)))])
    random.shuffle(colors)

    patch = _target_patch(colors, min(10, height - 2), min(9, width - 2))
    target = np.full((height, width), BG, dtype=int)
    r0 = random.randint(1, height - patch.shape[0] - 1)
    c0 = random.randint(1, width - patch.shape[1] - 1)
    target[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    rules = {outer: inner for outer, inner in pairs}
    expected_target = target.copy()
    for outer, inner in rules.items():
        expected_target[expected_target == outer] = inner
    return target, _crop_foreground(expected_target)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    colors = random.sample(PALETTE, pair_count * 2)
    pairs = [(colors[2 * i], colors[2 * i + 1]) for i in range(pair_count)]
    patches = [_make_pair_patch(outer, inner) for outer, inner in pairs]

    max_ph = max(patch.shape[0] for patch in patches)
    max_pw = max(patch.shape[1] for patch in patches)
    height = random.randint(max(14, max_ph + 2), 28)
    left_width = random.randint(max(12, max_pw + 2), 18)
    right_width = random.randint(7, 30 - left_width - 1)

    left = np.full((height, left_width), BG, dtype=int)
    if not _place_patches(left, patches):
        return None

    target, expected = _make_target(height, right_width, pairs)
    grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, :left_width] = left
    grid[:, left_width] = SEP
    grid[:, left_width + 1 :] = target

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring color dictionary input")
