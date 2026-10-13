from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
CONTOUR = 7
PALETTE = (1, 2, 3)


MOTIFS = {
    "L3_A": ((0, 0), (1, 0), (1, 1)),
    "L3_B": ((0, 0), (0, 1), (1, 1)),
    "L4_A": ((0, 0), (0, 1), (0, 2), (1, 0)),
    "L4_B": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L4_C": ((0, 2), (1, 0), (1, 1), (1, 2)),
}


TEMPLATES = [
    {
        "weight": 2,
        "size": (4, 8),
        "pieces": [
            {"motif": "L4_A", "scale": 2, "offset": (0, 0)},
            {"motif": "L4_C", "scale": 2, "offset": (0, 2)},
        ],
    },
    {
        "weight": 2,
        "size": (6, 12),
        "pieces": [
            {"motif": "L4_A", "scale": 3, "offset": (0, 0)},
            {"motif": "L4_C", "scale": 3, "offset": (0, 3)},
        ],
    },
    {
        "weight": 2,
        "size": (8, 16),
        "pieces": [
            {"motif": "L4_A", "scale": 4, "offset": (0, 0)},
            {"motif": "L4_C", "scale": 4, "offset": (0, 4)},
        ],
    },
    {
        "weight": 2,
        "size": (8, 10),
        "pieces": [
            {"motif": "L3_A", "scale": 4, "offset": (0, 0)},
            {"motif": "L4_A", "scale": 2, "offset": (0, 4)},
            {"motif": "L4_B", "scale": 2, "offset": (2, 6)},
        ],
    },
    {
        "weight": 2,
        "size": (8, 12),
        "pieces": [
            {"motif": "L4_A", "scale": 4, "offset": (0, 0)},
            {"motif": "L4_A", "scale": 2, "offset": (4, 4)},
            {"motif": "L4_C", "scale": 2, "offset": (4, 6)},
        ],
    },
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _sprite_from_coords(coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    sprite = np.zeros((h, w), dtype=int)
    for r, c in coords:
        sprite[r, c] = color
    return sprite


def _crop_to_bbox(grid: np.ndarray, color: int) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
    assert coords, f"missing color {color}"
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = color
    return out


def _downscale_sprite(sprite: np.ndarray, color: int, factor: int) -> np.ndarray:
    h, w = sprite.shape
    assert h % factor == 0 and w % factor == 0
    out = np.zeros((h // factor, w // factor), dtype=int)
    for rr in range(out.shape[0]):
        for cc in range(out.shape[1]):
            block = sprite[rr * factor : (rr + 1) * factor, cc * factor : (cc + 1) * factor]
            if np.any(block == color):
                assert np.all(block == color), "scaled piece must use solid color blocks"
                out[rr, cc] = color
            else:
                assert np.all(block == BG), "scaled piece blocks must be empty or solid"
    return out


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    black_components = _components(g == BG)
    region_id = -np.ones(g.shape, dtype=int)
    enclosed_regions: list[list[tuple[int, int]]] = []
    for coords in black_components:
        touches_edge = any(
            r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in coords
        )
        if touches_edge:
            continue
        idx = len(enclosed_regions)
        enclosed_regions.append(coords)
        for r, c in coords:
            region_id[r, c] = idx

    assert 2 <= len(enclosed_regions) <= 3, "expected 2 or 3 enclosed black regions"

    motif_components = _components((g != BG) & (g != CONTOUR))
    assert len(motif_components) == len(enclosed_regions), "each region must contain exactly one motif"

    pieces = []
    used_regions = set()
    for coords in motif_components:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "motif must be monochromatic"
        color = colors.pop()
        assert color in PALETTE

        adjacent_regions = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and region_id[nr, nc] >= 0:
                    adjacent_regions.add(int(region_id[nr, nc]))
        assert len(adjacent_regions) == 1, "motif must belong to exactly one enclosed region"

        region_idx = adjacent_regions.pop()
        assert region_idx not in used_regions, "each region should host one motif"
        used_regions.add(region_idx)

        region_area = len(enclosed_regions[region_idx])
        motif_area = len(coords)
        assert region_area % motif_area == 0, "area ratio must be integral"
        scale = region_area // motif_area
        assert 2 <= scale <= 4, "scale must be in 2..4"

        r0, c0, r1, c1 = _bbox(coords)
        sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in coords:
            sprite[r - r0, c - c0] = color

        pieces.append(
            {
                "color": color,
                "scale": scale,
                "motif_area": motif_area,
                "sprite": sprite,
                "scaled": _scale_sprite(sprite, scale),
            }
        )

    assert len(used_regions) == len(enclosed_regions)
    return pieces


def _rectangle_dims(total_area: int, pieces: list[dict]) -> list[tuple[int, int]]:
    max_h = max(piece["scaled"].shape[0] for piece in pieces)
    max_w = max(piece["scaled"].shape[1] for piece in pieces)
    dims = []
    for h in range(1, int(math.isqrt(total_area)) + 1):
        if total_area % h:
            continue
        w = total_area // h
        for hh, ww in ((h, w), (w, h)):
            if hh > 30 or ww > 30:
                continue
            if hh < max_h or ww < max_w:
                continue
            dims.append((hh, ww))
    dims = sorted(set(dims), key=lambda rc: (rc[0] * rc[1], rc[0], rc[1]))
    return dims


def _piece_placements(sprite: np.ndarray, h: int, w: int) -> list[dict]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(sprite != BG)]
    ph, pw = sprite.shape
    placements = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            mask = 0
            shifted = []
            for rr, cc in cells:
                r = r0 + rr
                c = c0 + cc
                shifted.append((r, c))
                mask |= 1 << (r * w + c)
            placements.append({"mask": mask, "cells": shifted})
    return placements


def _unique_packing(pieces: list[dict]) -> np.ndarray:
    ordered = sorted(pieces, key=lambda piece: (-piece["scale"], piece["color"]))
    total_area = sum(int(np.count_nonzero(piece["scaled"])) for piece in ordered)
    solutions: dict[tuple[int, int, bytes], np.ndarray] = {}

    for h, w in _rectangle_dims(total_area, ordered):
        full_mask = (1 << (h * w)) - 1
        placements = [_piece_placements(piece["scaled"], h, w) for piece in ordered]
        chosen = [None] * len(ordered)

        def rec(idx: int, used_mask: int) -> None:
            if len(solutions) > 1:
                return
            if idx == len(ordered):
                if used_mask != full_mask:
                    return
                out = np.zeros((h, w), dtype=int)
                for piece, placement in zip(ordered, chosen):
                    assert placement is not None
                    for rr, cc in placement["cells"]:
                        out[rr, cc] = piece["color"]
                key = (h, w, out.tobytes())
                solutions[key] = out
                return

            for placement in placements[idx]:
                if placement["mask"] & used_mask:
                    continue
                chosen[idx] = placement
                rec(idx + 1, used_mask | placement["mask"])
                chosen[idx] = None
                if len(solutions) > 1:
                    return

        rec(0, 0)
        if len(solutions) > 1:
            break

    assert len(solutions) == 1, "scaled motifs must pack into one unique full rectangle"
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Derive each motif's pixel scale from the enclosed black area around it, then pack the scaled motifs into one solid rectangle.

    Input structure:
    - The grid is black except for 2-3 orange closed contour loops and one small monochrome motif inside each loop.
    - Orange pixels are barriers; enclosed black background pockets do not touch the grid edge.
    - Each motif uses one distinct color from blue, red, and green, is 4-connected, and is not rotated or mirrored later.

    Rule:
    - Split black pixels into 4-connected regions and keep only the enclosed ones.
    - For each enclosed region, count its black area A and the area P of the colored motif inside it.
    - Compute the motif's scale as S = A / P, enlarge the motif by replacing each pixel with an SxS block, and keep its color.
    - Translate the enlarged motifs, without rotation or mirroring, until they form the unique gap-free rectangle; output that rectangle on black.

    Key insight:
    - The empty space inside each contour encodes the enlargement factor, and only those exact enlarged shapes can complement one another to tile a solid rectangle.
    """
    pieces = _extract_pieces(np.asarray(grid, dtype=int))
    return _unique_packing(pieces)


def _weighted_template() -> dict:
    bag = []
    for template in TEMPLATES:
        bag.extend([template] * int(template["weight"]))
    return random.choice(bag)


def _build_output_from_template(template: dict) -> tuple[np.ndarray, dict[int, int]]:
    h, w = template["size"]
    colors = random.sample(PALETTE, len(template["pieces"]))
    out = np.zeros((h, w), dtype=int)
    scales: dict[int, int] = {}
    for color, spec in zip(colors, template["pieces"]):
        sprite = _sprite_from_coords(MOTIFS[spec["motif"]], color)
        scaled = _scale_sprite(sprite, spec["scale"])
        r0, c0 = spec["offset"]
        target = out[r0 : r0 + scaled.shape[0], c0 : c0 + scaled.shape[1]]
        assert target.shape == scaled.shape
        assert not np.any((target != BG) & (scaled != BG)), "template pieces must not overlap"
        target[scaled != BG] = color
        scales[color] = int(spec["scale"])
    transformed = _transform_grid(out)
    return transformed, scales


def _factor_pairs(area: int, min_h: int, min_w: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, area + 1):
        if area % h:
            continue
        w = area // h
        if h >= min_h and w >= min_w:
            pairs.append((h, w))
        if w >= min_h and h >= min_w and h != w:
            pairs.append((w, h))
    raw_pairs = sorted(set(pairs))
    compact_pairs = [pair for pair in raw_pairs if max(pair) <= 6 and min(pair) >= 3]
    chosen = compact_pairs if compact_pairs else raw_pairs
    return sorted(chosen, key=lambda pair: (abs(pair[0] - pair[1]), max(pair)))


def _connected_black_after_placement(inner_h: int, inner_w: int, sprite: np.ndarray, top: int, left: int) -> bool:
    canvas = np.zeros((inner_h, inner_w), dtype=int)
    h, w = sprite.shape
    patch = canvas[top : top + h, left : left + w]
    patch[sprite != BG] = 1
    black = canvas == 0
    comps = _components(black)
    return len(comps) == 1


def _region_patch(piece: dict) -> np.ndarray:
    color = int(piece["color"])
    scale = int(piece["scale"])
    sprite = piece["motif"]
    motif_area = int(np.count_nonzero(sprite))
    inner_area = motif_area * (scale + 1)
    choices = _factor_pairs(inner_area, sprite.shape[0], sprite.shape[1])
    assert choices, "need at least one interior rectangle size"

    inner_h, inner_w = random.choice(choices[: min(3, len(choices))])
    positions = []
    for top in range(inner_h - sprite.shape[0] + 1):
        for left in range(inner_w - sprite.shape[1] + 1):
            if _connected_black_after_placement(inner_h, inner_w, sprite, top, left):
                positions.append((top, left))
    assert positions, "motif placement must leave one enclosed black component"

    top, left = random.choice(positions)
    patch = np.full((inner_h + 2, inner_w + 2), CONTOUR, dtype=int)
    patch[1:-1, 1:-1] = BG
    target = patch[1 + top : 1 + top + sprite.shape[0], 1 + left : 1 + left + sprite.shape[1]]
    target[sprite != BG] = color

    black_inside = _components(patch == BG)
    enclosed = [coords for coords in black_inside if all(0 < r < patch.shape[0] - 1 and 0 < c < patch.shape[1] - 1 for r, c in coords)]
    assert len(enclosed) == 1
    assert len(enclosed[0]) == motif_area * scale
    return patch


def _can_place_rect(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 + gap < rr0 or rr1 + gap < r0 or c1 + gap < cc0 or cc1 + gap < c0):
            return False
    return True


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    for _ in range(80):
        h = random.randint(max(18, max_h + 4), 26)
        w = random.randint(max(18, max_w + 4), 26)
        order = list(range(len(patches)))
        random.shuffle(order)
        rects: list[tuple[int, int, int, int]] = []
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            options = []
            for r0 in range(1, h - ph):
                for c0 in range(1, w - pw):
                    if _can_place_rect(rects, r0, c0, ph, pw, gap=2):
                        options.append((r0, c0))
            if not options:
                ok = False
                break
            placements[idx] = random.choice(options)
            r0, c0 = placements[idx]
            rects.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
        if not ok:
            continue

        canvas = np.zeros((h, w), dtype=int)
        for idx, patch in enumerate(patches):
            r0, c0 = placements[idx]
            canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        return canvas
    return None


def _output_piece_info(output_grid: np.ndarray, scales: dict[int, int]) -> list[dict]:
    pieces = []
    for color, scale in scales.items():
        sprite = _crop_to_bbox(output_grid, color)
        motif = _downscale_sprite(sprite, color, scale)
        motif_cells = int(np.count_nonzero(motif))
        assert motif_cells in (3, 4)
        pieces.append({"color": color, "scale": scale, "motif": motif})
    return pieces


def _validate_case(inp: np.ndarray, expected_out: np.ndarray, scales: dict[int, int]) -> bool:
    try:
        pieces = _extract_pieces(inp)
        if sorted((piece["color"], piece["scale"]) for piece in pieces) != sorted(scales.items()):
            return False
        actual_out = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual_out, expected_out):
        return False
    if np.any(actual_out == BG):
        return False
    if actual_out.shape != expected_out.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        expected_out, scales = _build_output_from_template(_weighted_template())
        pieces = _output_piece_info(expected_out, scales)
        random.shuffle(pieces)
        patches = [_region_patch(piece) for piece in pieces]
        inp = _place_patches(patches)
        if inp is None:
            continue
        if _validate_case(inp, expected_out, scales):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
