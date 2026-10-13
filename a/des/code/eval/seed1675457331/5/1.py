from __future__ import annotations

import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_frame_from_component(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected = set()
    for rr in range(r0, r1 + 1):
        expected.add((rr, c0))
        expected.add((rr, c1))
    for cc in range(c0, c1 + 1):
        expected.add((r0, cc))
        expected.add((r1, cc))
    if set(coords) != expected:
        return None
    return (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _connected_components(grid == color):
            bbox = _rect_frame_from_component(comp)
            if bbox is None:
                continue
            frames.append({"color": color, "bbox": bbox})
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use seed colors inside enclosed regions to select edge templates, then stamp each template sprite centered on its seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = _mode_color(g)

    templates: dict[int, np.ndarray] = {}
    regions: list[dict[str, int | tuple[int, int, int, int]]] = []
    for frame in _find_frames(g, bg):
        color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frame["bbox"])
        if r0 == 0 or r1 == h - 1:
            assert color not in templates, "template border colors must be unique"
            templates[color] = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        else:
            regions.append(frame)

    assert templates, "expected at least one edge template"
    assert regions, "expected at least one enclosed region"

    out = g.copy()
    for frame in regions:
        border_color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frame["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((interior != bg) & (interior != border_color))
        assert len(seed_locs) == 1, "each region must contain one seed pixel"

        ir, ic = map(int, seed_locs[0])
        seed_r = r0 + 1 + ir
        seed_c = c0 + 1 + ic
        seed_color = int(interior[ir, ic])
        assert seed_color in templates, "every seed color must match a template border color"
        sprite = templates[seed_color]
        sh, sw = sprite.shape
        anchor_r = seed_r - sh // 2
        anchor_c = seed_c - sw // 2

        out[seed_r, seed_c] = bg
        for sr in range(sh):
            for sc in range(sw):
                value = int(sprite[sr, sc])
                if value == bg:
                    continue
                rr = anchor_r + sr
                cc = anchor_c + sc
                if r0 < rr < r1 and c0 < cc < c1:
                    out[rr, cc] = value
    return out


BG = 0
PALETTE = tuple(range(10))
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (20, 30)
REGION_COUNT_RANGE = (2, 4)
TEMPLATE_INNER_CHOICES = (3, 5)
TEMPLATE_GAP_RANGE = (1, 3)
REGION_MARGIN_RANGE = (2, 5)
EXTRA_TEMPLATE_PROB = 0.35
MAX_GENERATION_ATTEMPTS = 400
REGION_PLACEMENT_ATTEMPTS = 250


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _make_sprite(
    rng: random.Random,
    height: int,
    width: int,
    blocked_colors: set[int],
    center_filled: bool,
) -> np.ndarray:
    sprite = np.full((height, width), BG, dtype=int)
    area = height * width
    target = rng.randint(max(2, area // 3), max(3, (2 * area) // 3))
    center = (height // 2, width // 2)
    start = center if center_filled else (rng.randrange(height), rng.randrange(width))
    if not center_filled and start == center:
        start = ((center[0] + 1) % height, center[1])

    filled = {start}
    while len(filled) < target:
        if rng.random() < 0.82:
            base_r, base_c = rng.choice(tuple(filled))
            candidates = [
                (base_r + dr, base_c + dc)
                for dr, dc in NEIGHBORS4
                if 0 <= base_r + dr < height and 0 <= base_c + dc < width
            ]
            if candidates:
                filled.add(rng.choice(candidates))
                continue
        filled.add((rng.randrange(height), rng.randrange(width)))

    if center_filled:
        filled.add(center)
    else:
        filled.discard(center)

    color_choices = [c for c in PALETTE if c not in blocked_colors]
    palette = rng.sample(color_choices, rng.randint(1, min(3, len(color_choices))))
    for r, c in filled:
        sprite[r, c] = rng.choice(palette)

    if np.all(sprite == BG):
        sprite[center] = palette[0]
    if center_filled and sprite[center] == BG:
        sprite[center] = palette[0]
    if not center_filled:
        sprite[center] = BG
    return sprite


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(*REGION_COUNT_RANGE)
    extra_templates = 1 if region_count < REGION_COUNT_RANGE[1] and rng.random() < EXTRA_TEMPLATE_PROB else 0
    template_count = region_count + extra_templates

    template_colors = rng.sample([c for c in PALETTE if c != BG], template_count)
    region_border_colors = rng.sample([c for c in PALETTE if c not in {BG, *template_colors}], region_count)

    template_specs = []
    blocked = {BG, *template_colors}
    for idx in range(template_count):
        inner_h = rng.choice(TEMPLATE_INNER_CHOICES)
        inner_w = rng.choice(TEMPLATE_INNER_CHOICES)
        sprite = _make_sprite(
            rng,
            inner_h,
            inner_w,
            blocked_colors=blocked,
            center_filled=(rng.random() < 0.5),
        )
        template_specs.append(
            {
                "border_color": template_colors[idx],
                "sprite": sprite,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )

    used_templates = rng.sample(template_specs, region_count)
    region_specs = []
    for idx in range(region_count):
        sprite = used_templates[idx]["sprite"]
        sh, sw = sprite.shape
        inner_h = sh + rng.randint(*REGION_MARGIN_RANGE)
        inner_w = sw + rng.randint(*REGION_MARGIN_RANGE)
        region_specs.append(
            {
                "border_color": region_border_colors[idx],
                "seed_color": used_templates[idx]["border_color"],
                "sprite_shape": sprite.shape,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )

    side = "top" if rng.random() < 0.5 else "bottom"
    gap_sizes = [rng.randint(*TEMPLATE_GAP_RANGE) for _ in range(max(0, template_count - 1))]
    template_width = sum(spec["outer_w"] for spec in template_specs) + sum(gap_sizes)
    band_h = max(int(spec["outer_h"]) for spec in template_specs)
    max_region_h = max(int(spec["outer_h"]) for spec in region_specs)
    max_region_w = max(int(spec["outer_w"]) for spec in region_specs)

    min_h = max(GRID_H_RANGE[0], band_h + max_region_h + 4)
    min_w = max(GRID_W_RANGE[0], template_width + 2, max_region_w + 4)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = rng.randint(min_h, GRID_H_RANGE[1])
    grid_w = rng.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    start_c = rng.randint(1, grid_w - template_width - 1)
    col = start_c
    for idx, spec in enumerate(template_specs):
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        row = 0 if side == "top" else grid_h - outer_h
        _draw_frame(grid, row, col, outer_h, outer_w, int(spec["border_color"]))
        sprite = np.asarray(spec["sprite"], dtype=int)
        grid[row + 1 : row + 1 + sprite.shape[0], col + 1 : col + 1 + sprite.shape[1]] = sprite
        if idx < len(gap_sizes):
            col += outer_w + gap_sizes[idx]

    occupied = np.zeros_like(grid, dtype=bool)
    if side == "top":
        occupied[: band_h + 1, :] = True
        min_row = band_h + 1
        max_row = grid_h - 2
    else:
        occupied[grid_h - band_h - 1 :, :] = True
        min_row = 1
        max_row = grid_h - band_h - 2

    rng.shuffle(region_specs)
    for spec in region_specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if max_row - min_row + 1 < outer_h:
            return None

        placed = False
        for _ in range(REGION_PLACEMENT_ATTEMPTS):
            r0 = rng.randint(min_row, max_row - outer_h + 1)
            c0 = rng.randint(1, grid_w - outer_w - 1)
            if not _can_place(occupied, r0, c0, outer_h, outer_w, pad=1):
                continue

            _draw_frame(grid, r0, c0, outer_h, outer_w, int(spec["border_color"]))
            sh, sw = spec["sprite_shape"]
            inner_r0 = r0 + 1
            inner_c0 = c0 + 1
            inner_r1 = r0 + outer_h - 2
            inner_c1 = c0 + outer_w - 2
            seed_r = rng.randint(inner_r0 + sh // 2, inner_r1 - sh // 2)
            seed_c = rng.randint(inner_c0 + sw // 2, inner_c1 - sw // 2)
            grid[seed_r, seed_c] = int(spec["seed_color"])
            _mark_bbox(occupied, r0, c0, outer_h, outer_w, pad=1)
            placed = True
            break
        if not placed:
            return None

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
