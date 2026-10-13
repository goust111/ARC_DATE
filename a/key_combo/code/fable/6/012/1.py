from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, find_components, get_bg, paste, scale_sprite


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in coords:
        mask[int(rr) - r0, int(cc) - c0] = True
    return mask


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool), (0, 0)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _binary_components(mask: np.ndarray) -> list[dict]:
    grid = mask.astype(int)
    comps = find_components(grid, bg=0, colors=[1], connectivity=4, monochromatic=True)
    comps.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    return comps


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _sprite_mask(sprite: dict) -> np.ndarray:
    return _mask_from_coords(sprite["coords"])


def _hole_mask(hole: dict) -> np.ndarray:
    return _mask_from_coords(hole["coords"])


def _hole_complement(hole: dict) -> tuple[np.ndarray, tuple[int, int]]:
    comp_mask = ~_hole_mask(hole)
    return _crop_mask(comp_mask)


def _wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(value) for value in np.unique(grid) if int(value) != int(bg)]
    assert colors, "expected at least one non-background color"
    return max(colors, key=lambda color: int(np.count_nonzero(grid == color)))


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall = _wall_color(g, bg)

    wall_rows, _wall_cols = np.where(g == wall)
    assert wall_rows.size > 0, "expected a wall"
    wall_top = int(wall_rows.min())
    wall_bottom = int(wall_rows.max())

    hole_band = np.zeros_like(g, dtype=bool)
    hole_band[wall_top : wall_bottom + 1, :] = (g[wall_top : wall_bottom + 1, :] == bg)
    holes = _binary_components(hole_band)
    assert holes, "expected carved holes inside the wall"

    sprite_colors = [int(value) for value in np.unique(g) if int(value) not in {bg, wall}]
    sprites = find_components(
        g,
        bg=bg,
        colors=sprite_colors,
        connectivity=4,
        monochromatic=True,
    )
    sprites.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    assert sprites, "expected sprites above the wall"
    return bg, wall, wall_top, holes, sprites


def _classify_relation(sprite: dict, hole: dict) -> tuple[str, int]:
    _sr0, sc0, _sr1, _sc1 = map(int, sprite["bbox"])
    hr0, hc0, _hr1, _hc1 = map(int, hole["bbox"])
    sprite_mask = _sprite_mask(sprite)
    hole_mask = _hole_mask(hole)

    if sc0 == hc0 and np.array_equal(sprite_mask, hole_mask):
        return "pass", hr0

    comp_mask, (off_r, off_c) = _hole_complement(hole)
    if comp_mask.size > 0 and sc0 == hc0 + off_c and np.array_equal(sprite_mask, comp_mask):
        return "plug", hr0 + off_r

    return "stop", -1


def _match_holes_to_sprites(holes: list[dict], sprites: list[dict]) -> list[dict]:
    matched: list[dict] = []
    for sprite in sprites:
        _sr0, sc0, _sr1, sc1 = map(int, sprite["bbox"])
        candidates = [
            hole
            for hole in holes
            if _intervals_overlap(sc0, sc1, int(hole["bbox"][1]), int(hole["bbox"][3]))
        ]
        assert len(candidates) == 1, "each sprite must align with exactly one hole lane"
        matched.append(candidates[0])
    return matched


def _paint_component(out: np.ndarray, sprite: dict, top: int) -> None:
    r0, c0, _r1, _c1 = map(int, sprite["bbox"])
    dr = int(top) - r0
    color = int(sprite["color"])
    for rr, cc in sprite["coords"]:
        out[int(rr) + dr, int(cc)] = color


def main(input_grid: np.ndarray) -> np.ndarray:
    """Drop each sprite straight down; congruent holes pass, complementary holes plug, otherwise the sprite rests on the wall."""
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    height, width = map(int, grid.shape)
    assert height <= 30 and width <= 30, "ARC grids must stay within 30x30"

    bg, wall, wall_top, holes, sprites = _extract_scene(grid)
    matched_holes = _match_holes_to_sprites(holes, sprites)

    out = grid.copy()
    for sprite in sprites:
        for rr, cc in sprite["coords"]:
            out[int(rr), int(cc)] = bg

    for sprite, hole in zip(sprites, matched_holes):
        relation, target_top = _classify_relation(sprite, hole)
        hr0, hc0, hr1, hc1 = map(int, hole["bbox"])
        hole_patch = out[hr0 : hr1 + 1, hc0 : hc1 + 1]

        if relation == "pass":
            hole_patch[_hole_mask(hole)] = wall
            sprite_height = int(sprite["bbox"][2]) - int(sprite["bbox"][0]) + 1
            landing_top = height - sprite_height
            _paint_component(out, sprite, landing_top)
        elif relation == "plug":
            hole_patch[_hole_mask(hole)] = wall
            _paint_component(out, sprite, target_top)
        else:
            stop_top = wall_top - (int(sprite["bbox"][2]) - int(sprite["bbox"][0]) + 1)
            _paint_component(out, sprite, stop_top)

    return out


_CONNECTED_SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
]

_COMPLEMENT_HOLES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
]

_STOP_PAIRS = [
    (
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ),
    (
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ),
    (
        np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
    ),
    (
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
    ),
]

_MAX_ATTEMPTS = 300


def _mask_patch(mask: np.ndarray, color: int, bg: int = Color.BLACK) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _choose_scale(base_h: int, base_w: int, max_h: int, max_w: int) -> int:
    allowed = [
        scale
        for scale in (1, 2)
        if base_h * scale <= max_h and base_w * scale <= max_w
    ]
    assert allowed, "shape does not fit the requested bounds"
    return random.choice(allowed)


def _make_pass_pair(max_h: int, max_w: int) -> tuple[np.ndarray, np.ndarray]:
    base = random.choice(_CONNECTED_SHAPES)
    scale = _choose_scale(base.shape[0], base.shape[1], max_h, max_w)
    hole_mask = scale_sprite(base, scale).astype(bool)
    return hole_mask, hole_mask.copy()


def _make_plug_pair(max_h: int, max_w: int) -> tuple[np.ndarray, np.ndarray]:
    base = random.choice(_COMPLEMENT_HOLES)
    scale = _choose_scale(base.shape[0], base.shape[1], max_h, max_w)
    hole_mask = scale_sprite(base, scale).astype(bool)
    comp_mask, _offset = _crop_mask(~hole_mask)
    return hole_mask, comp_mask


def _make_stop_pair(max_h: int, max_w: int) -> tuple[np.ndarray, np.ndarray]:
    choices = list(_STOP_PAIRS)
    random.shuffle(choices)
    for hole_base, sprite_base in choices:
        hole_allowed = [
            scale
            for scale in (1, 2)
            if hole_base.shape[0] * scale <= max_h and hole_base.shape[1] * scale <= max_w
        ]
        sprite_allowed = [
            scale
            for scale in (1, 2)
            if sprite_base.shape[0] * scale <= max_h and sprite_base.shape[1] * scale <= max_w
        ]
        if not hole_allowed or not sprite_allowed:
            continue
        hole_scale = random.choice(hole_allowed)
        sprite_scale = random.choice(sprite_allowed)
        hole_mask = scale_sprite(hole_base, hole_scale).astype(bool)
        sprite_mask = scale_sprite(sprite_base, sprite_scale).astype(bool)

        fake_hole = {"coords": np.argwhere(hole_mask), "bbox": (0, 0, hole_mask.shape[0] - 1, hole_mask.shape[1] - 1)}
        fake_sprite = {
            "coords": np.argwhere(sprite_mask),
            "bbox": (0, 0, sprite_mask.shape[0] - 1, sprite_mask.shape[1] - 1),
        }
        relation, _target = _classify_relation(fake_sprite, fake_hole)
        if relation == "stop":
            return hole_mask, sprite_mask
    raise AssertionError("failed to build a stop pair")


def _sample_pair(kind: str, max_h: int, max_w: int) -> tuple[np.ndarray, np.ndarray]:
    if kind == "pass":
        return _make_pass_pair(max_h, max_w)
    if kind == "plug":
        return _make_plug_pair(max_h, max_w)
    if kind == "stop":
        return _make_stop_pair(max_h, max_w)
    raise ValueError(f"unknown kind: {kind}")


def _layout_starts(widths: list[int], total_width: int, gap: int) -> list[int]:
    base = sum(widths) + gap * (len(widths) - 1)
    slack = total_width - base
    assert slack >= 0
    gaps = [0] * (len(widths) + 1)
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    starts: list[int] = []
    cursor = gaps[0]
    for index, width in enumerate(widths):
        starts.append(cursor)
        cursor += width
        if index < len(widths) - 1:
            cursor += gap + gaps[index + 1]
    return starts


def generate_input() -> np.ndarray:
    for _attempt in range(_MAX_ATTEMPTS):
        lane_count = random.randint(3, 5)
        kinds = ["pass", "plug", "stop"] + random.choices(["pass", "plug", "stop"], k=lane_count - 3)
        random.shuffle(kinds)

        max_h = random.randint(3, 6)
        max_w = random.randint(3, 6)

        lane_specs: list[dict] = []
        for kind in kinds:
            hole_mask, sprite_mask = _sample_pair(kind, max_h, max_w)
            comp_mask, comp_offset = _crop_mask(~hole_mask)
            lane_specs.append(
                {
                    "kind": kind,
                    "hole_mask": hole_mask,
                    "sprite_mask": sprite_mask,
                    "comp_mask": comp_mask,
                    "comp_offset": comp_offset,
                    "hole_h": int(hole_mask.shape[0]),
                    "hole_w": int(hole_mask.shape[1]),
                    "sprite_h": int(sprite_mask.shape[0]),
                    "sprite_w": int(sprite_mask.shape[1]),
                }
            )

        gap = random.randint(1, 2)
        widths = [max(int(spec["hole_w"]), int(spec["sprite_w"])) for spec in lane_specs]
        min_width = sum(widths) + gap * (lane_count - 1)
        if min_width > 28:
            continue

        width = random.randint(min_width, min(30, min_width + 4))
        starts = _layout_starts(widths, width, gap)

        max_hole_h = max(int(spec["hole_h"]) for spec in lane_specs)
        max_sprite_h = max(int(spec["sprite_h"]) for spec in lane_specs)
        wall_thickness = random.randint(max_hole_h + 2, min(9, max_hole_h + 5))
        bottom_margin = random.randint(2, 5)
        top_margin = random.randint(max_sprite_h + 2, max_sprite_h + 7)
        height = top_margin + wall_thickness + bottom_margin
        if height > 30:
            continue

        wall_top = top_margin
        wall_bottom = wall_top + wall_thickness - 1
        grid = np.full((height, width), Color.BLACK, dtype=int)
        grid[wall_top : wall_bottom + 1, :] = Color.GREY

        sprite_colors = random.sample(
            [color for color in Color.NOT_BLACK if color != Color.GREY],
            lane_count,
        )

        for index, spec in enumerate(lane_specs):
            lane_left = starts[index]
            lane_width = widths[index]
            hole_mask = spec["hole_mask"]
            sprite_mask = spec["sprite_mask"]
            hole_h = int(spec["hole_h"])
            hole_w = int(spec["hole_w"])
            sprite_h = int(spec["sprite_h"])
            sprite_w = int(spec["sprite_w"])

            hole_left = lane_left + random.randint(0, lane_width - hole_w)
            hole_top = random.randint(wall_top + 1, wall_bottom - hole_h)
            hole_patch = _mask_patch(hole_mask, Color.BLACK, bg=Color.GREY)
            paste(grid, hole_patch, hole_top, hole_left, transparent=Color.GREY)

            if spec["kind"] == "pass":
                sprite_left = hole_left
            elif spec["kind"] == "plug":
                _comp_r, comp_c = spec["comp_offset"]
                sprite_left = hole_left + int(comp_c)
            else:
                candidates = [
                    left
                    for left in range(lane_left, lane_left + lane_width - sprite_w + 1)
                    if _intervals_overlap(left, left + sprite_w - 1, hole_left, hole_left + hole_w - 1)
                ]
                if not candidates:
                    break
                random.shuffle(candidates)
                sprite_left = candidates[0]

            max_sprite_top = wall_top - sprite_h - 1
            if max_sprite_top < 0:
                break
            sprite_top = random.randint(0, max_sprite_top)
            sprite_patch = _mask_patch(sprite_mask, sprite_colors[index], bg=Color.BLACK)
            paste(grid, sprite_patch, sprite_top, sprite_left, transparent=Color.BLACK)
        else:
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
