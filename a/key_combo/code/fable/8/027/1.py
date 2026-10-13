from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    pieces: list[np.ndarray] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        pieces.append(np.asarray(coords, dtype=int))
    return pieces


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _round_half_up(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _extract_sprite(panel: np.ndarray, fill: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    interior_mask = panel[1:-1, 1:-1] != fill
    box = _bbox(interior_mask)
    assert box is not None, "panel must contain one sprite"
    pieces = _components(interior_mask)
    assert len(pieces) == 1, "sprite must be a single 4-connected object"
    r0, c0, r1, c1 = box
    r0 += 1
    r1 += 1
    c0 += 1
    c1 += 1
    sprite = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = sprite != fill
    return sprite, mask, int(r0), int(c0)


def _infer_panel_layout(grid: np.ndarray) -> tuple[int, int, int, list[np.ndarray]]:
    frame = int(grid[0, 0])
    h, w = grid.shape
    assert h >= 5 and w >= 5
    assert np.all(grid[0, :] == frame) and np.all(grid[-1, :] == frame)

    separators = [c for c in range(w) if np.all(grid[:, c] == frame)]
    assert len(separators) >= 4, "need at least three shared-border panels"
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1, "panel stride must be constant"
    stride = int(strides[0])
    panel_w = stride + 1

    panels: list[np.ndarray] = []
    interiors: list[np.ndarray] = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = grid[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert np.all(panel[:, 0] == frame) and np.all(panel[:, -1] == frame)
        panels.append(panel)
        interiors.append(panel[1:-1, 1:-1].ravel())

    fill = _most_common(np.concatenate(interiors))
    return frame, fill, stride, panels


def _infer_mapping_and_motion(
    panels: list[np.ndarray], fill: int, stride: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    sprites: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    top_rows: list[int] = []
    left_cols: list[int] = []

    for panel in panels:
        sprite, mask, top, left = _extract_sprite(panel, fill)
        sprites.append(sprite)
        masks.append(mask)
        top_rows.append(top)
        left_cols.append(left)

    template_mask = masks[0]
    template_shape = sprites[0].shape
    for mask, sprite, left in zip(masks[1:], sprites[1:], left_cols[1:]):
        assert sprite.shape == template_shape
        assert np.array_equal(mask, template_mask), "sprite shape must stay fixed"
        assert left == left_cols[0], "generator uses fixed local columns"

    deltas = [b - a for a, b in zip(top_rows, top_rows[1:])]
    assert deltas and len(set(deltas)) == 1, "top-row motion must be constant"
    delta = int(deltas[0])
    assert delta != 0, "movement must be nonzero"

    palette = sorted({int(v) for v in sprites[0][template_mask]})
    assert len(palette) >= 2, "need a visible recolor rule"

    mapping: dict[int, int] = {}
    for sprite_a, sprite_b in zip(sprites, sprites[1:]):
        for color_a, color_b in zip(sprite_a[template_mask], sprite_b[template_mask]):
            ca = int(color_a)
            cb = int(color_b)
            if ca in mapping:
                assert mapping[ca] == cb, "recolor rule must be fixed"
            else:
                mapping[ca] = cb

    assert set(mapping) == set(palette), "every sprite color must appear as a source"
    assert set(mapping.values()) == set(palette), "generator uses a palette permutation"
    assert any(mapping[c] != c for c in palette), "recolor rule must be non-identity"

    last_sprite = sprites[-1].copy()
    recolored = np.full_like(last_sprite, fill)
    recolored[template_mask] = [mapping[int(v)] for v in last_sprite[template_mask]]
    next_top = _round_half_up(
        Fraction(top_rows[-1], 1) + Fraction(delta * panels[0].shape[1], stride)
    )
    return recolored, template_mask, next_top, left_cols[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the next standalone framed panel by continuing both the sprite motion and the hidden recolor chain.

    Input structure:
    - The grid is a horizontal row of equal-width panels whose adjacent panels share one vertical border column.
    - Every panel has the same frame color and fill color, and contains one multicolor interior sprite with fixed shape.
    - Across panels, the sprite moves with constant vertical velocity while its visible colors change by the same fixed color-to-color mapping every frame.

    Rule:
    - Split the input into its framed panels using the shared full-height border columns.
    - Compare aligned sprite cells in consecutive panels to recover the one-step recolor mapping, and measure the constant motion from panel to panel.
    - Output one standalone panel with the same frame and fill, place the sprite one more step along the same motion using full panel width for the final extrapolation, and apply the recolor mapping once more.

    Key insight:
    - The observed panels are simultaneously a motion sequence and a recolor sequence, so the next frame is correct only if both hidden regularities are continued together.
    """
    g = np.asarray(grid, dtype=int)
    frame, fill, stride, panels = _infer_panel_layout(g)
    recolored, mask, next_top, next_left = _infer_mapping_and_motion(panels, fill, stride)

    panel_h, panel_w = panels[0].shape
    sprite_h, sprite_w = recolored.shape
    assert 1 <= next_top and next_top + sprite_h <= panel_h - 1
    assert 1 <= next_left and next_left + sprite_w <= panel_w - 1

    out = np.full((panel_h, panel_w), fill, dtype=int)
    out[0, :] = frame
    out[-1, :] = frame
    out[:, 0] = frame
    out[:, -1] = frame
    target = out[next_top : next_top + sprite_h, next_left : next_left + sprite_w]
    target[mask] = recolored[mask]
    return out


_GEN_ATTEMPTS = 500
_GEN_HEIGHTS = tuple(range(8, 15))
_GEN_INTERIOR_WIDTHS = (4, 5, 6)
_GEN_NUM_PANELS = (3, 4)
_GEN_SPRITE_HEIGHTS = (2, 3, 4)
_GEN_SPRITE_WIDTHS = (2, 3, 4)
_GEN_DELTAS = (-2, -1, 1, 2)


def _gen_connected_mask(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    for _ in range(80):
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier: list[Coord] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) == area and len(rows) == h and len(cols) == w:
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            return mask
    raise RuntimeError("failed to build connected sprite mask")


def _gen_color_cycle(rng: random.Random, colors: list[int]) -> dict[int, int]:
    order = colors[:]
    rng.shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def _apply_color_map(sprite: np.ndarray, mask: np.ndarray, mapping: dict[int, int], fill: int) -> np.ndarray:
    out = np.full_like(sprite, fill)
    out[mask] = [mapping[int(v)] for v in sprite[mask]]
    return out


def _build_panel(height: int, panel_w: int, frame: int, fill: int) -> np.ndarray:
    panel = np.full((height, panel_w), fill, dtype=int)
    panel[0, :] = frame
    panel[-1, :] = frame
    panel[:, 0] = frame
    panel[:, -1] = frame
    return panel


def _paste_sprite(panel: np.ndarray, sprite: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    target = panel[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    target[mask] = sprite[mask]


def _gen_motion_choices(height: int, sprite_h: int, panel_w: int, num_panels: int) -> list[tuple[int, int, int]]:
    max_top = height - 1 - sprite_h
    stride = panel_w - 1
    choices: list[tuple[int, int, int]] = []
    for delta in _GEN_DELTAS:
        for start in range(1, max_top + 1):
            tops = [start + i * delta for i in range(num_panels)]
            if not all(1 <= top <= max_top for top in tops):
                continue
            next_top = _round_half_up(
                Fraction(tops[-1], 1) + Fraction(delta * panel_w, stride)
            )
            if 1 <= next_top <= max_top:
                choices.append((delta, start, next_top))
    return choices


def _compose_input(panels: list[np.ndarray]) -> np.ndarray:
    height, panel_w = panels[0].shape
    stride = panel_w - 1
    width = len(panels) * stride + 1
    out = np.zeros((height, width), dtype=int)
    for i, panel in enumerate(panels):
        c0 = i * stride
        out[:, c0 : c0 + panel_w] = panel
    return out


def _make_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    for _ in range(120):
        height = rng.choice(_GEN_HEIGHTS)
        interior_w = rng.choice(_GEN_INTERIOR_WIDTHS)
        panel_w = interior_w + 2
        num_panels = rng.choice(_GEN_NUM_PANELS)

        all_colors = list(range(10))
        frame, fill = rng.sample(all_colors, 2)
        sprite_pool = [c for c in all_colors if c not in (frame, fill)]

        sprite_h = rng.choice(_GEN_SPRITE_HEIGHTS)
        sprite_w = rng.choice(_GEN_SPRITE_WIDTHS)
        if sprite_h > height - 2 or sprite_w > interior_w:
            continue
        left = rng.randint(1, panel_w - sprite_w - 1)

        motion_choices = _gen_motion_choices(height, sprite_h, panel_w, num_panels)
        if not motion_choices:
            continue

        max_area = sprite_h * sprite_w
        num_colors = rng.randint(2, min(4, len(sprite_pool), max_area))
        min_area = max(num_colors, sprite_h + sprite_w - 1)
        area = rng.randint(min_area, max_area)
        try:
            mask = _gen_connected_mask(rng, sprite_h, sprite_w, area)
        except RuntimeError:
            continue

        colors = rng.sample(sprite_pool, num_colors)
        mapping = _gen_color_cycle(rng, colors)

        assignments = colors[:]
        while len(assignments) < area:
            assignments.append(rng.choice(colors))
        rng.shuffle(assignments)
        sprite0 = np.full((sprite_h, sprite_w), fill, dtype=int)
        for (r, c), color in zip(map(tuple, np.argwhere(mask)), assignments):
            sprite0[int(r), int(c)] = int(color)

        delta, start_top, next_top = rng.choice(motion_choices)

        sprites = [sprite0]
        for _ in range(1, num_panels):
            sprites.append(_apply_color_map(sprites[-1], mask, mapping, fill))

        panels: list[np.ndarray] = []
        for i in range(num_panels):
            panel = _build_panel(height, panel_w, frame, fill)
            _paste_sprite(panel, sprites[i], mask, start_top + i * delta, left)
            panels.append(panel)

        next_sprite = _apply_color_map(sprites[-1], mask, mapping, fill)
        expected = _build_panel(height, panel_w, frame, fill)
        _paste_sprite(expected, next_sprite, mask, next_top, left)

        meta = {
            "frame": frame,
            "fill": fill,
            "delta": delta,
            "mapping": dict(mapping),
            "mask": mask.copy(),
            "left": left,
            "start_top": start_top,
            "next_top": next_top,
            "num_panels": num_panels,
        }
        return _compose_input(panels), expected, meta
    raise RuntimeError("failed to sample a valid case")


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        try:
            grid, expected, meta = _make_case(rng)
        except RuntimeError:
            continue
        if np.array_equal(main(grid), expected):
            assert meta["delta"] != 0
            assert any(k != v for k, v in dict(meta["mapping"]).items())
            return grid
    raise RuntimeError("failed to generate valid input")
